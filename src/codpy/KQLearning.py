import sys
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import random

import numpy as np
import pandas as pd
from scipy import optimize
from scipy.special import softmax

import codpy
import codpy.core as core
from codpy.core import get_matrix
import codpy.selection as selection
from codpy.data_processing import hot_encoder
from codpy.kernel import Kernel, KernelClassifier, get_tensor_probas
from codpy.lalg import LAlg
from codpy.algs import Alg
from codpy.sampling import get_normals
from codpy.utils import gather, cartesian_sum, cartesian_outer_product, fit_to_cov
import codpy.conditioning
from codpy.permutation import map_invertion
from codpy.clustering import MiniBatchkmeans, BalancedClustering


class ReplayBuffer(object):
    def __init__(self, capacity=None, memory=None, **kwargs):
        if memory is None:
            self.memory = None
        else:
            self.memory = list(memory)
        if capacity is None:
            self.capacity = sys.maxsize
        else:
            self.capacity = capacity
        self.games_list = []

    def update(self, **kwargs):
        def helper(i):
            self.memory[i] = self.memory[i][
                max(self.memory[i].shape[0] - self.capacity, 0) :
            ]

        [helper(i) for i in range(len(self.memory))]
        return

    def empty(self, N=None, *args):
        if N is None:
            self.memory = None
        else:

            def helper(i):
                self.memory[i] = self.memory[i][
                    max(self.memory[i].shape[0] - self.capacity, N) :
                ]

            [helper(i) for i in range(len(self.memory))]

    def push(self, *sarsd, capacity=None, **kwargs):
        self.games_list.append(sarsd)
        if capacity is not None:
            self.capacity = capacity
        if self.memory is None:
            self.memory = list(sarsd)
        else:

            def helper(i):
                return np.concatenate([self.memory[i], sarsd[i]], axis=0)

            self.memory = [helper(i) for i in range(len(sarsd))]
        self.update()
        return self

    def games(self):
        return self.games_list

    def last_game(self):
        return self.games()[-1]

    def last(self):
        return self.last_game

    def sample(self, batch_size):
        if len(self) == 0:
            return None
        indices = list(range(len(self)))
        if batch_size > len(self):
            self.last_indices = indices
        else:
            self.last_indices = random.sample(indices, batch_size)
        return self[self.last_indices]

    def __len__(self):
        if self.memory is None:
            return 0
        return len(self.memory[0])

    def __getitem__(self, indices):
        if self.memory is None:
            return None

        def helper(i):
            out = self.memory[i]
            return out[indices]

        return [helper(i) for i in range(len(self.memory))]


class GamesClustering:
    def __init__(self, kernel):
        self.k = kernel

    def get_labels(self):
        return self(self.k.get_x())

    def __call__(self, z, **kwargs):
        if not hasattr(self.k, "kernels") or self.k.kernels is None:
            return None
        labels = None
        for i in self.k.kernels.keys():
            k = self.k.kernels[i]
            labelsk = core.get_matrix(k.dnm(z, k.get_x()).min(1))
            if labels is None:
                labels = labelsk
            else:
                labels = np.concatenate([labels, labelsk], axis=1)
        return core.get_matrix(labels.argmin(1), dtype=int)


class GamesKernel(Kernel):
    def __init__(self, latent_distribution=None, max_size=None, **kwargs):

        if kwargs.get("latent_shape", None) is None:
            self.latent_distribution = None
        else:
            shape = kwargs["latent_shape"]
            self.latent_distribution = get_normals(shape[0], shape[1])

        self.max_size = max_size
        self.map_ = None
        super().__init__(**kwargs)
        self.clustering = self.set_clustering()

    def default_clustering_functor(self) -> callable:
        return lambda: GamesClustering(self)

    def __call__(self, z, **kwargs):
        z = core.get_matrix(z)

        if not hasattr(self, "kernels") or len(self.kernels) == 0:
            fy = self.get_theta()

            if fy is None:
                fy = self.get_knm_inv()
            knm = self.knm(x=z, y=self.get_y(), fy=fy, **kwargs)
            return knm

        knm = np.zeros([z.shape[0], self.get_fx().shape[1]])

        mapped_indices = self.clustering(z)
        mapped_indices = map_invertion(mapped_indices)
        for key in mapped_indices.keys():
            indices = list(mapped_indices[key])
            knm[indices] += self.kernels[key](z[indices])
        return knm

    def add_kernel(self, k, **kwargs) -> None:
        if not hasattr(self, "kernels"):
            self.kernels = {}
        self.kernels[len(self.kernels)] = k
        if self.get_x() is None:
            self.set_x(k.get_x(), **kwargs)
            self.set_fx(k.get_fx(), **kwargs)
        else:
            self.set_x(np.concatenate([self.get_x(), k.get_x()], axis=0))
            self.set_fx(np.concatenate([self.get_fx(), k.get_fx()], axis=0))

    def add(self, x, fx, **kwargs) -> None:
        if self.get_x() is None:
            self.set(x=x, fx=fx, **kwargs)
        else:
            x = np.concatenate([self.get_x(), x], axis=0)
            fx = np.concatenate([self.get_fx(), fx], axis=0)
            self.set(x=x, fx=fx, **kwargs)

    def knm(
        self, x: np.ndarray, y: np.ndarray, fy: np.ndarray = [], on_y=False, **kwargs
    ) -> np.ndarray:
        if not hasattr(self, "kernels") or len(self.kernels) == 0:
            return super().knm(x, y, fy, **kwargs)

        self.set_kernel_ptr()
        out = np.zeros([x.shape[0], y.shape[0]])
        if on_y == True:
            map_y_kernel = self.clustering(y)
            map_kernel_y = map_invertion(map_y_kernel)
            for i in map_kernel_y.keys():
                k = self.kernels[i]
                indices = list(map_kernel_y[i])
                knm_loc = k.knm(x=x, y=y[indices])
                out[:, indices] = knm_loc
        else:
            map_x_kernel = self.clustering(x)
            map_kernel_x = map_invertion(map_x_kernel)
            for i in map_kernel_x.keys():
                k = self.kernels[i]
                indices = list(map_kernel_x[i])
                knm_loc = k.knm(x=x[indices], y=y)
                out[indices, :] = knm_loc

        if fy is not None and len(fy) > 0:
            out = LAlg.prod(out, fy)
        return out

    def set(
        self,
        x: np.ndarray = None,
        fx: np.ndarray = None,
        y: np.ndarray = None,
        **kwargs,
    ) -> None:
        if self.max_size is not None and x.shape[0] > self.max_size:
            selected = Kernel(x=x).greedy_select(N=self.max_size, fx=fx)
            self.indices = selected.indices
            x, fx = selected.get_x(), selected.get_fx()
            # x, fx = x[-self.max_size:],fx[-self.max_size:]
            # clusters = MiniBatchkmeans(x=x,N=self.max_size)
            # x,indices = clusters.cluster_centers_,clusters.indices
            # temp = np.zeros([x.shape[0],fx.shape[1]])
            # indices = map_invertion(indices)
            # def helper(k):
            #     temp[k] = fx[list(indices[k])].mean()
            # [helper(i) for i in indices.keys()]
            # fx = temp

        if self.latent_distribution is not None:
            if self.latent_distribution.shape[0] > x.shape[0]:
                self.set_map(None)
            else:
                N, D = (
                    min(self.latent_distribution.shape[0], x.shape[0]),
                    self.latent_distribution.shape[1],
                )
                # if self.get_map() is None:
                selected = Kernel(x=x).greedy_select(N=N, fx=fx)
                # selected = Kernel(x=x).greedy_select(N= N)
                map_ = Kernel(x=selected.get_x(), order=1).map(
                    y=self.latent_distribution
                )
                # map_ = Kernel(x=self.latent_distribution).map(y=selected.get_x())
                # map_ = Kernel(x=map_.get_fx(),fx=map_.get_x(),order=2)
                self.set_map(map_)
                # test = self.map_(self.map_.get_x()) - self.map_.get_fx()
        super().set(x, fx, y, **kwargs)


class GamesKernelClassifier(GamesKernel):
    def __init__(
        self,
        latent_distribution=None,
        max_size=None,
        clip=Alg.proportional_fitting,
        **kwargs,
    ):
        super().__init__(
            latent_distribution=latent_distribution, max_size=max_size, **kwargs
        )

    def __call__(self, z, **kwargs):
        knm = super().__call__(z, **kwargs)
        return softmax(knm, axis=1)

    def set_fx(
        self,
        fx: np.ndarray,
        set_polynomial_regressor: bool = True,
        clip=Alg.probas_projection,
        **kwargs,
    ) -> None:
        fx_ = fx
        if fx is not None:
            if clip is not None:
                fx_ = clip(fx, axis=1)
            debug = np.where(fx_ < 1e-9, 1e-9, fx)
            fx_ = np.log(debug)
        super().set_fx(fx_, set_polynomial_regressor=set_polynomial_regressor, **kwargs)


def rl_hot_encoder(actions, actions_dim):
    out = hot_encoder(pd.DataFrame(np.float64(actions)), cat_cols_include=[0])
    if out.shape[1] != actions_dim:
        for i in range(actions_dim):
            col_name = "0_" + str(i) + ".0"
            if len(selection.get_starting_cols(out, [col_name])) == 0:
                out[col_name] = 0.0
                pass
    out = out.reindex(sorted(out.columns), axis=1)
    return out.values


def Verhulst(probs, advantages):
    # d/dt pi(t) = pi(t)(1-pi(t)) A(t) (Verhulst)
    # pi(t^{n+1}) = 1/ ( 1.+ (1 / pi(t^{n} - 1) \exp(-A(t^n)(t^{n+1}-t^{n})) )
    # advantages = advantages-core.get_matrix(advantages.mean(1))
    out = np.log(probs) + advantages
    out = softmax(out, axis=1)
    return out


class KAgent:
    def __init__(
        self, actions_dim, state_dim, gamma=0.99, kernel_type=GamesKernel, **kwargs
    ):
        self.kernel_type = kernel_type
        self.actions_dim = actions_dim
        self.state_dim = state_dim
        self.gamma = gamma
        self.all_actions_ = None
        params = kwargs.get("KActor", {})
        self.actor = self.kernel_type(**params)
        self.target = self.kernel_type(**params)
        params = kwargs.get("KCritic", {})
        self.critic = self.kernel_type(**params)
        params = kwargs.get("KNextStates", {})
        self.next_states = self.kernel_type(**params)
        params = kwargs.get("Rewards", {})
        self.rewards = self.kernel_type(**params)
        self.replay_buffer = ReplayBuffer(**kwargs)
        self.eps_threshold = kwargs.get("eps_threshold", 1.0)

    def compute_returns(self, states, actions, next_states, rewards, dones, **kwargs):
        returns, next_return = [], 0.0
        # for t in reversed(range(len(rewards))): # we already reversed time
        for t in range(len(rewards)):
            next_return = rewards[t] + self.gamma * next_return
            returns.append(next_return)
        return np.array(returns)

    def bellman_error(
        self,
        states,
        actions,
        next_states,
        rewards,
        returns,
        policy=None,
        value_function=None,
    ):
        if value_function is None:
            value_function = self.get_state_action_value_function(
                states, actions, next_states, rewards, returns, policy
            )

        states_actions = np.concatenate([states, actions], axis=1)
        next_states_actions = self.all_states_actions(next_states)

        error = self.gamma * value_function(next_states_actions).reshape(
            [states_actions.shape[0], self.actions_dim]
        )
        if policy is None:
            error = core.get_matrix(error.max(1))
        else:
            error = core.get_matrix((error * policy).sum(1))
        error = value_function(states_actions) - rewards - error

        return error

    def optimal_bellman_error(
        self, states, actions, next_states, rewards, policy, value_function=None
    ):
        if value_function is None:
            value_function = self.get_state_action_value_function(
                states, actions, next_states, rewards, policy
            )

        states_actions = np.concatenate([states, actions], axis=1)
        next_states_actions = self.all_states_actions(next_states)

        error = self.gamma * value_function(next_states_actions).reshape(
            [states_actions.shape[0], self.actions_dim]
        )
        error = core.get_matrix(error.max(1))
        error = rewards + error - value_function(states_actions)

        return error

    def update_probabilities(
        self,
        advantages,
        states,
        actions,
        next_states,
        rewards,
        dones,
        last_policy,
        dt=None,
        **kwargs,
    ):  ##this function assumes that advantages[i,j]=KCritic([states[i],j])
        if dt is None:
            dt = 1.0 / (advantages * advantages).mean()
        interpolated_policy = Verhulst(last_policy, advantages * dt)
        params = kwargs.get("KActor", {})

        return GamesKernelClassifier(x=states, fx=interpolated_policy, **params)

    def get_state_action_value_function(
        self, states, actions, next_states, rewards, returns, dones,policy=None, **kwargs
    ):
        states_actions = np.concatenate([states, actions], axis=1)
        next_states_actions = self.all_states_actions(next_states)
        value_function = self.kernel_type(**kwargs.get("KActor", {}))
        value_function.set(x=states_actions, fx=returns)
        knm = value_function.knm(x=states_actions, y=value_function.get_x())
        projection_op = value_function.knm(
            x=next_states_actions, y=value_function.get_x()
        )
        if policy is None:
            thetas, bellman_error = self.optimal_bellman_solver(
                thetas=value_function.get_theta(),
                next_states_projection=projection_op,
                knm=knm,
                rewards=rewards,
                maxiter=5,
                reg=value_function.reg,
            )
        else:
            projection_op = projection_op.reshape(
                [
                    states_actions.shape[0],
                    projection_op.shape[0] // states_actions.shape[0],
                    value_function.get_x().shape[0],
                ]
            )
            sum_policy = np.einsum("...ji,...j", projection_op, policy)
            thetas = LAlg.lstsq(knm - sum_policy * self.gamma, rewards)
        # thetas = LAlg.prod(projection_op, rewards)
        value_function.set_theta(thetas)

        # check
        # error = self.bellman_error(states,actions,next_states, rewards,policy, value_function)
        return value_function

    def get_derivatives_policy_state_action_value_function(
        self, states, actions, next_states, rewards, policy, output_value_function=False
    ):
        # return an estimator of \nabla_\pi V(\pi) where
        # - \pi is a policy, that is a probability distribution over actions
        # - V(\pi) is the value function of the policy \pi
        # - V(\pi) = E_{s,a} [ Q(s,a) | \pi ]
        states_actions = np.concatenate([states, actions], axis=1)
        next_states_actions = self.all_states_actions(next_states)
        value_function = Kernel()
        value_function.set(x=states_actions)
        knm = value_function.knm(x=states_actions, y=states_actions)
        projection_op = value_function.knm(
            x=next_states_actions, y=states_actions
        ).reshape([states_actions.shape[0], self.actions_dim, states_actions.shape[0]])
        sum_policy = np.einsum("...ji,...j", projection_op, policy)
        projection_op = LAlg.lstsq(knm - sum_policy * self.gamma)
        thetas = LAlg.prod(projection_op, rewards)
        value_function.set_theta(thetas)
        ##end
        next_states_actions_values = value_function(next_states_actions).reshape(
            [states_actions.shape[0], self.actions_dim]
        )
        coeffs = get_tensor_probas(policy)
        second_member = np.einsum("...i,...ij", next_states_actions_values, coeffs)

        derivative_estimator = Kernel()
        derivative_estimator.set_x(states_actions)
        derivative_estimator.set_theta(LAlg.prod(projection_op, second_member))
        if output_value_function:
            return derivative_estimator, value_function
        return derivative_estimator

    def get_state_value_function(
        self, states, actions, next_states, rewards_matrix, policy
    ):
        value_function = Kernel(x=states)
        operator_inv = value_function.knm(
            x=states, y=states
        ) - self.gamma * value_function.knm(x=next_states, y=states)
        operator = LAlg.lstsq(operator_inv)
        second_member = core.get_matrix((policy * rewards_matrix).sum(1))
        value_function.set_theta(LAlg.prod(operator, second_member))
        # def check():
        #     test = value_function(states)-self.gamma*value_function(next_states)-second_member
        #     assert(np.abs(test).max() < 1e-4)
        # check()
        return value_function

    def format(self, sarsd, **kwargs):
        state, action, next_state, reward, done = [core.get_matrix(e) for e in sarsd]

        action = rl_hot_encoder(action, self.actions_dim)
        done = core.get_matrix(done, dtype=bool)
        return state, action, next_state, reward, done

    def get_derivatives_policy_state_value_function(
        self,
        states,
        actions,
        next_states,
        rewards_matrix,
        policy,
        output_value_function=False,
    ):
        ##begin get_state_value_function code
        derivative_estimator = Kernel(x=states)
        operator = derivative_estimator.knm(
            x=states, y=states
        ) - self.gamma * derivative_estimator.knm(x=next_states, y=states)
        operator = LAlg.lstsq(operator)

        ##end
        @np.vectorize
        def fun(i, j, k):
            return rewards_matrix[i, j] * policy[i, j] * (float(j == k) - policy[i, k])

        coeffs = np.fromfunction(
            fun,
            shape=[rewards_matrix.shape[0], self.actions_dim, self.actions_dim],
            dtype=int,
        )

        coeffs = coeffs.sum(axis=1)
        # this is \nabla_ln(pi) sum_a R(s,a) \pi^a(s) = \nabla_ln(pi) E(R,\pi)
        # def check_derivative(policy):
        #     #Let E(R,\pi) = sum_a R(s,a) \pi^a(s) the rewards expectation
        #     #test if ( E(R,\pi) - E(R,\pi^\epsilon) )/epsilon = \nabla_ln(pi) E(R,\pi) . (log \pi^\epsilon - log \pi)
        #     old_policy = policy.copy()
        #     inc = np.random.normal(size=policy.shape) * 1e-4
        #     policy = softmax(np.log(policy) + inc,axis=1)
        #     inc = np.log(policy)-np.log(old_policy)
        #     test = ( (rewards_matrix*policy).sum(1)-(rewards_matrix*old_policy).sum(1) ) / 1e-4
        #     check_ = test - (coeffs*inc).sum(1)/ 1e-4
        #     assert(np.abs(check_).max() < 1e-4)
        #     pass
        # check_derivative(policy)

        derivative_estimator.set_theta(LAlg.prod(operator, coeffs))

        # def check():
        #     #check the derivated Bellman relation \nabla_ln \pi ( V(S_T) - gamma V(S_{T+1} - \E(R) ) =0)
        #     test = derivative_estimator(states)-self.gamma*derivative_estimator(next_states)-coeffs
        #     assert(np.abs(test).max() < 1e-4)
        # check()

        if output_value_function:
            value_function = Kernel(x=states)
            second_member = (policy * rewards_matrix).reshape(policy.shape).sum(1)
            value_function.set_theta(LAlg.prod(operator, second_member))
            return derivative_estimator, value_function
        return derivative_estimator

    def get_greedy_policy(self, states, q_values):
        shape_ = [states.shape[0], self.actions_dim]
        states_actions = self.all_states_actions(states)
        greedy_policy = q_values(states_actions).reshape(shape_).argmax(1)
        out = np.zeros(shape_)
        fill(out, np.ones([shape_[0]]), greedy_policy)
        return out

    def get_all_actions(self):
        if self.all_actions_ is None:
            self.all_actions_ = core.get_matrix(range(0, self.actions_dim))
            self.all_actions_ = hot_encoder(
                pd.DataFrame(self.all_actions_), cat_cols_include=[0]
            ).values
        return self.all_actions_

    def all_states_actions(self, states, all_actions=None):
        if all_actions == None:
            all_actions = self.get_all_actions()

        def helper(i, j):
            out = np.concatenate([states[[i]], all_actions[[j]]], axis=1)
            return out

        test = np.concatenate(
            [
                helper(i, j)
                for i in range(states.shape[0])
                for j in range(all_actions.shape[0])
            ],
            axis=0,
        )
        return test

    def optimal_bellman_solver(
        self, thetas, next_states_projection, knm, rewards, maxiter, reg=1e-9
    ):
        theta = thetas.copy()
        shape = [next_states_projection.shape[0]//self.actions_dim, self.actions_dim]
        error, count = sys.maxsize, 0
        while error > 0.01 and count < maxiter:
            max_indices = (
                LAlg.prod(next_states_projection, theta).reshape(shape).argmax(1)
            )
            indices = [
                self.actions_dim * i + max_indices[i] for i in range(len(max_indices))
            ]
            temp = next_states_projection[indices]
            temp = temp.reshape(knm.shape[0], -1, knm.shape[1]).sum(axis=1)
            max_projection = knm - temp * self.gamma
            next_theta = LAlg.lstsq(max_projection, rewards, reg)
            # var = np.var(rewards)

            def f(x):
                interpolated_thetas = theta * x + next_theta * (1.0 - x)
                bellmann_error = LAlg.prod(
                    next_states_projection, interpolated_thetas
                ).reshape(shape)
                bellmann_error = core.get_matrix(bellmann_error.max(1)) * self.gamma
                bellmann_error = bellmann_error.reshape(rewards.shape[0], -1, 1).sum(
                    axis=1
                )
                bellmann_error += rewards
                bellmann_error -= LAlg.prod(knm, interpolated_thetas)
                out = np.fabs(bellmann_error).mean()
                return out

            # return next_theta,f(0.)
            xmin, fval, iter, funcalls = optimize.brent(
                f, brack=(0.0, 1.0), maxiter=maxiter, full_output=True
            )
            if fval >= error:
                break
            error, count = fval, count + 1
            theta = theta * xmin + next_theta * (1.0 - xmin)
        return theta, fval

    def optimal_states_values_function(
        self,
        states,
        actions,
        next_states,
        rewards,
        returns,
        dones,
        kernel=None,
        verbose=False,
        **kwargs,
    ):
        states_actions = np.concatenate([states, actions], axis=1)
        next_states_actions = self.all_states_actions(next_states)
        if kernel is None:
            kernel = self.kernel_type(x=states_actions, fx=returns, **kwargs)
        if not hasattr(kernel, "clustering"):
            labels_state_action = None
        else:
            labels_state_action = kernel.clustering(states_actions)
        if labels_state_action is None:
            _projection_ = kernel.knm(x=next_states_actions, y=kernel.get_x())
            _knm_ = kernel.knm(x=states_actions, y=kernel.get_x())
            thetas, bellman_error = self.optimal_bellman_solver(
                thetas=kernel.get_theta(),
                next_states_projection=_projection_,
                knm=_knm_,
                rewards=rewards,
                maxiter=5,
                reg=kernel.reg,
            )
            kernel.set_theta(thetas)
            if verbose:
                print("Computed global error Bellman mean: ", bellman_error)

        else:
            map_labels = map_invertion(labels_state_action)
            for label in map_labels.keys():
                k = kernel.kernels[label]
                local_indices = list(map_labels[label])
                local_actions = actions[local_indices]
                local_states = states[local_indices]
                local_states_actions = np.concatenate(
                    [local_states, local_actions], axis=1
                )
                local_rewards = rewards[local_indices]
                local_next_states_actions = self.all_states_actions(
                    next_states[local_indices]
                )
                local_projection = k.knm(x=local_next_states_actions, y=k.get_x())
                local_knm = k.knm(x=local_states_actions, y=local_states_actions)
                thetas, bellman_error = self.optimal_bellman_solver(
                    thetas=k.get_theta(),
                    next_states_projection=local_projection,
                    knm=local_knm,
                    rewards=local_rewards,
                    maxiter=5,
                )
                k.set_theta(thetas)
                if verbose:
                    print(
                        "Kernel num: ",
                        label,
                        ", local error Bellman mean: ",
                        bellman_error,
                    )

        if verbose:
            bellmann_error = kernel(states_actions)
            bellmann_error -= (
                core.get_matrix(
                    kernel(next_states_actions).reshape(actions.shape).max(1)
                )
                * self.gamma
            )
            bellmann_error -= rewards
            # bellmann_error = bellmann_error[[not d for d in dones.flatten()]]

            print(
                "Global error Bellman max: ",
                np.fabs(bellmann_error).max(),
                ", global Bellman mean: ",
                np.fabs(bellmann_error).mean(),
            )
            pass

        return kernel


class KActorCritic(KAgent):

    def __call__(self, state, **kwargs):
        # return 1
        if self.actor.get_x() is not None and self.actor.get_x().shape[0] > 2:
            action_probs = self.actor(core.get_matrix(state).T)
            action_probs = action_probs.squeeze()
            # action = action_probs.argmax()
            action = np.random.choice(len(action_probs), p=action_probs)
            return action
        else:
            return np.random.randint(0, self.actions_dim)

    def get_advantages(
        self,
        states,
        actions,
        next_states,
        rewards,
        returns,
        dones,
        policy,
        **kwargs,
    ):
        value_function = self.get_state_action_value_function(
            states, actions, next_states, rewards, returns, dones,policy=policy, **kwargs
        )
        advantages = (
            value_function(self.all_states_actions(next_states)).reshape(actions.shape)
            * self.gamma
            + rewards
        )
        advantages -= value_function(np.concatenate([states, actions], axis=1))
        advantages -= core.get_matrix((advantages).mean(1))
        return advantages

    def train(self, game, max_training_game_size=None, **kwargs):
        states, actions, next_states, rewards, dones = self.format(game, **kwargs)
        returns = self.compute_returns(
            states, actions, next_states, rewards, dones, **kwargs
        )
        if max_training_game_size is not None:
            states, actions, next_states, rewards, returns, dones = (
                states[0:max_training_game_size],
                actions[0:max_training_game_size],
                next_states[0:max_training_game_size],
                rewards[0:max_training_game_size],
                returns[0:max_training_game_size],
                dones[0:max_training_game_size],
            )
            game = (states, actions, next_states, rewards, dones)
        self.replay_buffer.push(states, actions, next_states, rewards, returns, dones)
        states, actions, next_states, rewards, returns, dones = (
            self.replay_buffer.memory.copy()
        )
        states_actions = np.concatenate([states, actions], axis=1)
        self.rewards = Kernel(x=states_actions, fx=rewards, order=1)

        if self.actor.get_x() is not None and self.actor.get_x().shape[0] > 1:
            last_policy = self.actor(states)
        else:
            last_policy = np.full(
                [states.shape[0], self.actions_dim], 1.0 / self.actions_dim
            )
        count, error = 0, sys.float_info.max
        # compute advantages
        advantages = self.get_advantages(
            states,
            actions,
            next_states,
            rewards,
            returns,
            dones,
            policy=last_policy,
            **kwargs,
        )
        # update probabilities
        kernel = self.update_probabilities(
            advantages,
            states,
            actions,
            next_states,
            rewards,
            dones,
            last_policy,
            **kwargs,
        )
        self.actor = kernel


class KQLearning(KActorCritic):

    def __call__(self, state, **kwargs):
        self.eps_threshold *= 0.999
        if np.random.random() > self.eps_threshold and self.critic.get_x() is not None:
            z = self.all_states_actions(core.get_matrix(state).T)
            q_values = self.critic(z)
            q_values += np.random.random(q_values.shape) * 0.001
            return np.argmax(q_values)
        return np.random.randint(0, self.actions_dim)

    def train(self, game, max_training_game_size=None, **kwargs):
        param = kwargs.get("KActor",None)
        states, actions, next_states, rewards, dones = self.format(game, **kwargs)
        if max_training_game_size is not None:
            states, actions, next_states, rewards, dones = (
                states[0:max_training_game_size],
                actions[0:max_training_game_size],
                next_states[0:max_training_game_size],
                rewards[0:max_training_game_size],
                dones[0:max_training_game_size],
            )
            game = (states, actions, next_states, rewards, dones)

        returns = self.compute_returns(
            states, actions, next_states, rewards, dones, **kwargs
        )
        self.replay_buffer.push(states, actions, next_states, rewards, returns, dones)
        states, actions, next_states, rewards, returns, dones = (
            self.replay_buffer.memory.copy()
        )
        kernel = self.optimal_states_values_function(
            states,
            actions,
            next_states,
            rewards,
            returns,
            dones,
            verbose=True,
            **kwargs,
        )
        self.critic = kernel

class PolicyGradient(KActorCritic):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        params = kwargs.get("KCritic", {})
        self.actor = GamesKernelClassifier(**params)

    def __call__(self, state, **kwargs):
        if self.actor.get_x() is not None and self.actor.get_x().shape[0] > 1:
            action_probs = self.actor(core.get_matrix(state).T)
            action_probs = action_probs.squeeze()
            action = np.random.choice(len(action_probs), p=action_probs)
            return action
        else:
            return np.random.randint(0, self.actions_dim)

    def get_advantages(
        self, states, actions, next_states, rewards, returns, dones, policy, **kwargs
    ):
        # advantage taken as A = \nabla_\pi \pi Q^{pi}(S_T,A_T), so that the overall gradient policy can be written as
        #  d/di \pi(t) = d/d\pi Q^{pi}(S_T,A_T)
        #  Thus formally d/dt  Q^{pi}(S_T,A_T) = < \nabla_\pi \pi Q^{pi}(S_T,A_T), d/dt \pi> = | \nabla_\pi \pi Q^{pi}(S_T,A_T)|^2
        derivative_estimator = self.get_derivatives_policy_state_action_value_function(
            states, actions, next_states, rewards, policy
        )
        states_actions = np.concatenate([states, actions], axis=1)
        derivative_estimations = derivative_estimator(states_actions)
        return derivative_estimations


class KController(KAgent):
    def __init__(self, state_dim, actions_dim, controller, **kwargs):
        self.controller = controller
        self.x, self.y = None, None
        self.expectation_estimator = None
        self.label = kwargs.get("label", None)
        super().__init__(state_dim=state_dim, actions_dim=actions_dim, **kwargs)

    def __call__(self, z, **kwargs):
        return self.controller(z, **kwargs)

    def get_reward(self, game, **kwargs):
        states, actions, next_states, rewards, dones = game
        return get_matrix(rewards).mean()

    def get_expectation_estimator(self, x, y, **kwargs):
        class explore_kernel(Kernel):
            def distance(self, z, **kwargs):
                out = get_matrix(self.dnm(x=self.get_x(), y=z).min(axis=0))
                return out

        params = kwargs.get("KController",{})
        self.expectation_kernel = explore_kernel(x=x, fx=y, **params)
        return self.expectation_kernel
    
    def get_function(self, **kwargs):
        self.expectation_estimator = self.get_expectation_estimator(self.x, self.y, **kwargs)
        self.min_expectation_estimator = self.expectation_estimator(self.x).min()
        def function(x):
            expectation = self.expectation_estimator(x) - self.min_expectation_estimator
            distance = self.expectation_estimator.distance(x)
            return expectation*distance
        return function  # to cope with exploration    

    def train(self, game, env, **kwargs):
        states, actions, next_states, rewards, dones = self.format(game, **kwargs)
        self.replay_buffer.push(states, actions, next_states, rewards, dones)
        reward = get_matrix(rewards)
        last_theta = get_matrix(self.controller.get_thetas()).T
        if self.x is None:
            self.x = get_matrix(last_theta)
            self.y = get_matrix(reward)
        else:
            if (
                self.expectation_estimator is None
                or self.expectation_estimator.distance(last_theta) > 1e-9
            ):
                self.x = np.concatenate([self.x, last_theta])
                self.y = np.concatenate([self.y, reward])

        if self.x.shape[0] > 2:
            function = self.get_function(**kwargs)
            last_vals = function(self.x)
            last_val = last_vals.max()
            last_val_max_min = last_val-last_vals.min()
            max_val, new_theta = codpy.optimization.continuous_optimizer(
                function,
                self.controller.get_distribution(),
                include=self.x,
                **kwargs,
            )
            new_theta = get_matrix(new_theta).T
            if last_val < max_val: #debug to see if parameters are updated
                tot=1
                pass
            self.controller.set_thetas(new_theta)
        else:
            self.controller.set_thetas(self.controller.get_distribution()(1))

def get_conditioned_kernel(states,actions,next_states,rewards,returns,dones,expectation_kernel,**kwargs):
    class ConditionerKernel(codpy.conditioning.NadarayaWatsonKernel):
        def __init__(self, expectation_kernel,**kwargs):
            super().__init__(**kwargs)
            self.expectation_kernel = expectation_kernel
        def joint_density(self, x,y,**kwargs):
            out = super().joint_density(x,y)
            return out
        
    states_actions = np.concatenate([states, actions], axis=1)
    noise = next_states-expectation_kernel(np.concatenate([states, actions], axis=1))
    params = kwargs.get("HJBModel",kwargs)
    out = ConditionerKernel(x=states_actions,y=noise,expectation_kernel=expectation_kernel,**params)
    return out

def get_expectation_kernel(states,actions,next_states,rewards,returns,dones,**kwargs):
    class expectation_kernel(Kernel):
        def __init__(self, state_dim,**kwargs):
            super().__init__(**kwargs)
            self.state_dim = state_dim
        def __call__(self,z,**kwargs):
            out = super().__call__(z)+z[:,:self.state_dim]
            return out
    
    # if self.expectation_kernel is None:
    states_actions = np.concatenate([states, actions], axis=1)
    params = kwargs.get("HJBModel",kwargs)
    out = expectation_kernel(x=states_actions,fx=next_states-states,**params)
    noise = next_states-out(states_actions)
    out.set_fx(out.get_fx()+noise.mean(axis=0))
    # noise = next_states-out(states_actions)
    return out


class KQLearningHJB(KQLearning):
    
    def optimal_states_values_function(
        self,
        states,
        actions,
        next_states,
        rewards,
        returns,
        dones,
        kernel=None,
        verbose=False,
        **kwargs,
    ):
        states_actions = np.concatenate([states, actions], axis=1)
        if kernel is None:
            kernel = self.kernel_type(x=states_actions, fx=returns, **kwargs)

        expectation_kernel_ = get_expectation_kernel(states,actions,next_states,rewards,returns,dones,**kwargs)
        conditioned_kernel_ = get_conditioned_kernel(states,actions,next_states,rewards,returns,dones,expectation_kernel=expectation_kernel_,**kwargs)
        next_states_actions = self.all_states_actions(next_states)
        noise = next_states- expectation_kernel_(states_actions)
        _probas = conditioned_kernel_.conditional_density(x=states_actions, y=noise)
        
        _projection_ = kernel.knm(x=next_states_actions, y=kernel.get_x())
        _projection_ = LAlg.prod(_projection_,_probas)
        _knm_ = kernel.knm(x=states_actions, y=kernel.get_x())
        thetas, bellman_error = self.optimal_bellman_solver(
            thetas=kernel.get_theta(),
            next_states_projection=_projection_,
            knm=_knm_,
            rewards=rewards,
            maxiter=5,
            reg=kernel.reg,
        )
        kernel.set_theta(thetas)
        if verbose:
            print("Computed global error Bellman mean: ", bellman_error)

        return kernel

class KActorCriticHJB(KActorCritic):
    def train(self, game, max_training_game_size=None, **kwargs):
        super().train(game, max_training_game_size, **kwargs)
        pass

    def get_state_action_value_function(
        self,
        states,
        actions,
        next_states,
        rewards,
        returns,
        dones,
        policy,
        kernel=None,
        verbose=False,
        **kwargs,
    ):
        states_actions = np.concatenate([states, actions], axis=1)
        if kernel is None:
            kernel = self.kernel_type(x=states_actions, fx=returns, **kwargs)

        value_function = self.kernel_type(**kwargs.get("KActor", {}))
        value_function.set(x=states_actions, fx=returns)

        expectation_kernel_ = get_expectation_kernel(states,actions,next_states,rewards,returns,dones,**kwargs)
        conditioned_kernel_ = get_conditioned_kernel(states,actions,next_states,rewards,returns,dones,expectation_kernel=expectation_kernel_,**kwargs)
        next_states_actions = self.all_states_actions(next_states)
        noise = next_states- expectation_kernel_(states_actions)
        _probas = conditioned_kernel_.conditional_density(x=states_actions, y=noise)
        projection_op = kernel.knm(x=next_states_actions, y=kernel.get_x())
        projection_op = LAlg.prod(projection_op,_probas)
        _knm_ = kernel.knm(x=states_actions, y=kernel.get_x())

        projection_op = projection_op.reshape(
            [
                states_actions.shape[0],
                projection_op.shape[0] // states_actions.shape[0],
                states_actions.shape[0],
            ]
        )
        sum_policy = np.einsum("...ji,...j", projection_op, policy)
        thetas = LAlg.lstsq(_knm_ - sum_policy * self.gamma, rewards)
        value_function.set_theta(thetas)

        # check
        # error = self.bellman_error(states,actions,next_states, rewards,policy, value_function)
        return value_function
