import sys
import os

import codpy.core
import codpy.permutation

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
import codpy.optimization as optimization

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
    def is_full(self):
        return len(self) >= self.capacity

    def update(self, capacity=None, worst_game = False,**kwargs):
        if capacity is None:
            capacity = self.capacity
        if len(self) < capacity:
            return True
        returned = True
        if worst_game == True:
            while len(self) > capacity:
                scores = np.array([self.games_list[i][3].sum() - self.games_list[i][3].var() for i in range(len(self.games_list))])
                min_indices = scores.argmin()
                if min_indices == len(self.games_list)-1 :
                    returned = False
                self.remove_games(min_indices)

            return returned

        else:
            def helper(i):
                self.memory[i] = self.memory[i][
                    max(self.memory[i].shape[0] - capacity, 0) :
                ]

            [helper(i) for i in range(len(self.memory))]
        return True

    def empty(self, N=None, *args):
        if N is None:
            self.memory = None
            self.games_list = []
        else:

            def helper(i):
                self.memory[i] = self.memory[i][
                    max(self.memory[i].shape[0] - self.capacity, N) :
                ]

            [helper(i) for i in range(len(self.memory))]

    def push(self, *sarsd, capacity=None, **kwargs):
        self.games_list.append(sarsd)

        if self.memory is None:
            self.memory = sarsd
        else:

            def helper(i):
                if len(sarsd[i]) == 0:
                    return self.memory[i]
                return np.concatenate([self.memory[i], sarsd[i]], axis=0)

            self.memory = [helper(i) for i in range(len(sarsd)) ]
        return self.update(capacity,**kwargs)

    def games(self):
        return self.games_list
    def remove_games(self,indices):
        if isinstance(indices,list):
            [self.remove_games(i) for i in indices]
        else:
            del self.games_list[indices]
            def helper(i):
                self.memory[i] = np.concatenate([self.games_list[j][i] for j in range(len(self.games_list))], axis=0)
            [helper(i) for i in range(len(self.memory)) ]                    
        return self

    def last_game(self):
        return self.games()[-1]

    def last(self):
        return self.last_game()

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
    """A specific type of kernel for deterministic policies, handling clustering
    
    """
    def __init__(
        self, latent_distribution=None, max_size=None, next_states=None, **kwargs
    ):

        if kwargs.get("latent_shape", None) is None:
            self.latent_distribution = None
        else:
            shape = kwargs["latent_shape"]
            self.latent_distribution = get_normals(shape[0], shape[1])

        self.max_size = max_size
        self.map_ = None
        params = kwargs.get("HJBModel", kwargs)
        super().__init__(**kwargs)
        self.clustering = self.set_clustering()
        self.update_kernels = False
        self.next_states = next_states
        self.kernels = {}
        # self.gamma = gamma

    def default_clustering_functor(self) -> callable:
        return lambda: GamesClustering(self)

    def __call__(self, z, **kwargs):
        if len(self.kernels) == 0:
            return super().__call__(z, **kwargs)
        if len(self.kernels) == 1:
            return self.kernels[0](z, **kwargs)
        z = core.get_matrix(z)
        dim = self.kernels[0].get_fx().shape[1]
        knm = np.zeros([z.shape[0], dim])
        mapped_indices = self.clustering(z)
        mapped_indices = map_invertion(mapped_indices)
        for key in mapped_indices.keys():
            indices = list(mapped_indices[key])
            knm[indices] += self.kernels[key](z[indices])
        return knm        

        class helper:
            xs, fxs, ds, vals, rs, ns = None, None, None, None, None, None

            def __init__(self, k):
                self.ref_kernel = k

            def __call__(self, k):
                d = k.knm(z, k.get_x())
                # vals = k(z)
                arg_x = d.argmax(1)
                x, fx, d, r, n = (
                    k.get_x()[arg_x],
                    k.get_fx()[arg_x],
                    gather(d, arg_x),
                    k.games[3][arg_x],
                    k.games[2][arg_x],
                )
                if self.xs is None:
                    self.xs, self.fxs, self.ds, self.rs, self.ns = x, fx, d, r, n
                    # self.vals=vals
                else:
                    self.xs, self.fxs, self.ds, self.rs, self.ns = (
                        np.concatenate([self.xs, x], axis=0),
                        np.concatenate([self.fxs, fx], axis=0),
                        np.concatenate([self.ds, d], axis=0),
                        np.concatenate([self.rs, r], axis=0),
                        np.concatenate([self.ns, n], axis=0),
                    )
                    # self.vals=np.concatenate([self.vals,vals], axis = 1)

        helper_ = helper(self)
        [helper_(k) for k in self.kernels.values()]
        # knm = helper_.vals.max(1)[:,None]
        # return knm
        # knm = codpy.conditioning.ConditionerKernel(x=helper_.xs,y=helper_.fxs).get_transition_kernel()(z, reg=1e-9)
        # knm = Kernel(x=helper_.xs)(z)
        kernel_ptr = Kernel(x=helper_.xs).get_kernel()
        knm = codpy.core.KerOp.projection(
            x=helper_.xs,
            y=helper_.xs,
            z=z,
            # fx=np.concatenate([helper_.fxs, helper_.rs], axis=1),
            fx=helper_.fxs,
            reg=1e-4,
            order=2,
            kernel_ptr=kernel_ptr,
        )
        # knm = self.gamma*knm[:,[0]] +knm[:,[1]]
        return knm

    def add_kernel(self, k, max_kernel=1e30, **kwargs) -> None:
        if self.get_x() is None:
            self.set(x=k.get_x(), fx=k.get_fx())
        else:
            if self.get_x().shape[0] < 1000000:
                x, fx = self.get_x(), self.get_fx()
                x = np.concatenate([x, k.get_x()], axis=0)
                fx = np.concatenate([fx, k.get_fx()], axis=0)
                self.set(x=x, fx=fx)
            else:
                self.add(y=k.get_x(), fy=k.get_fx(), **kwargs)
        self.kernels[len(self.kernels) % max_kernel] = k
        return self


class GamesKernelClassifier(GamesKernel):
    """A specific type of kernel for stochastic policies. Outputs probabilities 
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __call__(self, z, **kwargs):
        knm = super().__call__(z, **kwargs)
        if len(self.kernels) == 1:
            return knm
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

class SparseGamesKernel(GamesKernel, codpy.kernel.SparseKernel):
    pass

def rl_hot_encoder(actions, actions_dim):
    """Hot encodes actions over actions_dim.

    :param actions: :class:`numpy.ndarray`.
    :param actions_dim: :class:`int` The dimension of the actions.

    :return: :class:`pandas.DataFrame`
    """
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
    """
    Basic KAgent. Has most of the usefull methods for other Reinforcement Learning algorithms.
    
    """
    def __init__(
        self, actions_dim, state_dim, gamma=0.99, kernel_type=GamesKernel, **kwargs
    ):
        """Initializes the KAgent with the given parameters. Every agent has an actor and critic kernel. Some classes might not use both.

        :param actions_dim: :class:`int` The action dimension of the environment.
        :param state_dim: :class:`int` The state dimension of the environment.
        :param gamma: :class:`float` Discount factor.
        :param kernel_type: :class:`codpy.kernel.Kernel` Type of kernel to be used as actor and critic.
        :type kwargs: :class:`dict` 
        """
        self.kernel_type = kernel_type
        self.actions_dim = actions_dim
        self.state_dim = state_dim
        self.gamma = gamma
        self.all_actions_ = None
        params = kwargs.get("KActor", {})
        self.actor = self.kernel_type(gamma=gamma, **params)
        self.target = self.kernel_type(gamma=gamma, **params)
        params = kwargs.get("KCritic", {})
        self.critic = self.kernel_type(gamma=gamma, **params)
        params = kwargs.get("KNextStates", {})
        self.next_states = self.kernel_type(gamma=gamma, **params)
        params = kwargs.get("Rewards", {})
        self.rewards = self.kernel_type(gamma=gamma, **params)
        self.replay_buffer = ReplayBuffer(gamma=gamma, **kwargs)
        self.eps_threshold = kwargs.get("eps_threshold", 0.0)

    def get_expectation_kernel(self, games, **kwargs):
        return get_expectation_kernel(games, **kwargs)

    def get_conditioned_kernel(self, games, expectation_kernel, **kwargs):
        return get_conditioned_kernel(
            games, expectation_kernel=expectation_kernel, **kwargs
        )
        states, actions, next_states, rewards, returns, dones = games
        states_actions = self.all_states_actions(states)
        expected_states = expectation_kernel(states_actions)
        # noise = next_states-expected_states
        params = kwargs.get("HJBModel", kwargs)
        out = codpy.conditioning.ConditionerKernel(
            x=np.concatenate([expected_states, actions], axis=1),
            y=next_states,
            expectation_kernel=expectation_kernel,
            **params,
        )
        return out

    def compute_returns(self, states, actions, next_states, rewards, dones, **kwargs):
        """
        Computes $G_t = R_t + \gamma G_{t+1}$ for the given history.

        :param states: :class:`np.ndarray` The states of the game, in reverse order.
        :param actions:  
        :param next_states:
        :param rewards: 
        :param dones: 

        :return: :class:`np.ndarray` 
        """
        returns, next_return = [], 0.0
        # for t in reversed(range(len(rewards))): # we already reversed time
        for t in range(len(rewards)):
            next_return = rewards[t] + self.gamma * next_return
            returns.append(next_return)
        return np.array(returns)

    def bellman_error(
        self,
        games,
        value_function,
        policy=None
    ):
        states, actions, next_states, rewards, returns, dones = games
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

    def optimal_bellman_error(self,games,value_function):
        return self.bellman_error(games,value_function)

    def bellman_optimal_action(
        self,
        games,
        q_value_function
    ):
        """Computes the optimal actions for the given $Q(s,a)$ function, with $$Q^*(s,a) = R(s,a) + \gamma \max_{a'} Q^{\pi}(s',a')$$.

        :param games: :class:`tuple` SARSD in reverse order.
        :param q_value_function: :class:`codpy.kernel.Kernel` The Q-value function.

        :return: :class:`np.ndarray` The optimal actions one hot encoded.
        """
        states, actions, next_states, rewards, returns, dones = games
        states_actions = np.concatenate([states, actions], axis=1)
        next_states_actions = self.all_states_actions(next_states)
        error = self.gamma * q_value_function(next_states_actions).reshape(actions.shape) + rewards
        error -= q_value_function(states_actions)
        actions = error.argmax(1)[:,None]
        return rl_hot_encoder(actions,self.actions_dim)
    
    def update_probabilities(
        self, advantages, games, last_policy, dt=None, kernel=None, clip=None,**kwargs
    ):  
        """Updates the policies for advantage-based algorithms. The advantage either is $\\nabla_{y} Q^\pi_k(\cdot)$ for Policy Gradient methods, or $R(s,a) + \gamma V^{\pi}(s') - V^{\pi}(s)$ for ActorCritic.

        It does normalize the advantages and then computes the new policy as an interpolation between the last policy and the new one.

        :param advantages: :class:`np.ndarray` 
        :param games: :class:`tuple` SARSD in reverse order.
        :param last_policy: :class:`np.ndarray` The last policy.
        
        :return: :class:`codpy.kernel.Kernel` The new policy.
        """
        ##this function assumes that advantages[i,j]=KCritic([states[i],j])
        states, actions, next_states, rewards, returns, dones = games
        advantages -= advantages.mean(axis=1)[:, None]
        if dt is None:
            dt = advantages
            # dt = np.where(np.fabs(dt) < 1e-8,0.,.1/dt)
            # dt /= (dt * dt).mean(axis=1)[:,None] + 1e-9
            dt /= ((dt*dt).mean(axis=1)[:, None] + 1e-9)
            if clip is not None:
                dt = clip*dt/ (np.fabs(dt).max(axis=1)[:, None] + 1e-9)
            # if dt < 1e-9:
            #     dt = 0.0
            # else:
            #     dt = advantages / dt
        # dt -= dt.mean(axis=1)[:,None]
        interpolated_policy = Verhulst(last_policy, dt)
        params = kwargs.get("KActor", {})
        # if kernel is None:
        return GamesKernelClassifier(
            x=states, y=states, fx=interpolated_policy, clip=None, **params
        )
        # else:
        #     kernel.update(z=states,fz=interpolated_policy)
        #     return kernel

    def get_state_action_value_function(self, games, policy=None, max_y=None,kernel=None,**kwargs):

        states, actions, next_states, rewards, returns, dones = games
        if policy is None:
            policy = actions
        if max_y is None: max_y = sys.maxsize
        # for i in range(states.shape[0] // max_y + 1):
        # step= states.shape[0]/max_y
        # indices = [int(i*step) for i in range(max_y)]
        # indices = np.random.choice(states.shape[0], size=max_y, replace=False)
        # indices = list(range(i*max_y,min((i+1)*max_y,states.shape[0])))
        # states_, actions_, next_states_, rewards_, returns_, dones_ = states[indices], actions[indices], next_states[indices], rewards[indices], returns[indices], dones[indices]
        states_, actions_, next_states_, rewards_, returns_, dones_ = states, actions, next_states, rewards, returns, dones
        policy_ = policy
        states_actions = np.concatenate([states_, actions_], axis=1)
        next_states_actions = self.all_states_actions(next_states_)
        value_function = self.kernel_type(**kwargs.get("KActor", {}))
        if kernel is None:
            value_function.set(x=states_actions, y=states_actions,fx=returns_)
        else:
            value_function.copy(kernel)
        knm = value_function.knm(x=states_actions, y=value_function.get_y())
        projection_op = value_function.knm(x=next_states_actions, y=value_function.get_y())

        def helper(i):
            if dones_[i] == True:
                return [i * self.actions_dim + j for j in range(self.actions_dim)]

        modif = [
            item
            for i in range(dones_.shape[0])
            if dones_[i] == True
            for item in helper(i)
        ]
        projection_op[modif] = 0.0

        projection_op = projection_op.reshape(
            [
                states_actions.shape[0],
                projection_op.shape[0] // states_actions.shape[0],
                value_function.get_y().shape[0]
            ]
        )
        sum_policy = np.einsum("...ji,...j", projection_op, policy_)
        mat = knm - sum_policy * self.gamma
        thetas = LAlg.lstsq(mat, rewards)
        value_function.set_theta(thetas)
        value_function.games=(states_, actions_, next_states_, rewards_, returns_, dones_)
        # thetas = LAlg.prod(projection_op, rewards)
        # kernel.add_kernel(value_function)

        # check
        # error = self.bellman_error(states,actions,next_states, rewards,policy, value_function)
        return value_function

    def get_derivatives_policy_state_action_value_function(
        self, games, policy, output_value_function=False, **kwargs
    ):
        """
        Solve for $$\\nabla_{y} \\theta^\pi = \gamma \Big(K(Z,Z) - \gamma \sum_a \pi^a(S) K(W,Z) \Big)^{-1} \sum_a\Big(Q^{\pi}_k(W) \pi^a(\delta_b(a)-\pi^b)\Big)$$

        Where: 

            - $Z$ are the state actions
            - $W$ are the next state actions
            - $K(Z,Z)$ is the Gram matrix of the training points
            - $\gamma \sum_a \pi^a(S) K(W,Z)$ is the weighted projection operator onto the next state-actions 
            - $Q^{\pi}_k(W)$ is the critic evaluated at the next state-actions
            - $\pi^a(\delta_b(a)-\pi^b)$ is an adjustment based on probability differences

        :param games: :class:`tuple` SARSD in reverse order.
        :param policy: :class:`np.ndarray` The policy to be used for weigthing the next state-actions.

        :return: :class:`codpy.kernel.Kernel` The derivative estimator of $Q(s,a)$
        """
        # return an estimator of \nabla_\pi V(\pi) where
        # - \pi is a policy, that is a probability distribution over actions
        # - V(\pi) is the value function of the policy \pi
        # - V(\pi) = E_{s,a} [ Q(s,a) | \pi ]
        states, actions, next_states, rewards, returns, dones = games
        states_actions = np.concatenate([states, actions], axis=1)
        next_states_actions = self.all_states_actions(next_states)
        value_function = Kernel()
        value_function.set(x=states_actions)
        knm = value_function.knm(x=states_actions, y=states_actions)
        projection_op = value_function.knm(
            x=next_states_actions, y=states_actions
        ).reshape([states_actions.shape[0], self.actions_dim, states_actions.shape[0]])
        projection_op[[bool(d) for d in dones]] = 0.0
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

    def get_state_value_function(self, games, policy):
        states, actions, next_states, rewards, returns, dones = games
        value_function = Kernel(x=states)
        operator_inv = value_function.knm(
            x=states, y=states
        ) - self.gamma * value_function.knm(x=next_states, y=states)
        operator = LAlg.lstsq(operator_inv)
        second_member = core.get_matrix((policy * rewards).sum(1))
        value_function.set_theta(LAlg.prod(operator, second_member))
        # def check():
        #     test = value_function(states)-self.gamma*value_function(next_states)-second_member
        #     assert(np.abs(test).max() < 1e-4)
        # check()
        return value_function

    def format(self, sarsd, max_training_game_size=None, **kwargs):
        states, actions, next_states, rewards, dones = [
            core.get_matrix(e) for e in sarsd
        ]

        actions = rl_hot_encoder(actions, self.actions_dim)
        returns = self.compute_returns(
            states, actions, next_states, rewards, dones, **kwargs
        )
        dones = core.get_matrix(dones, dtype=bool)
        if max_training_game_size is not None:
            states, actions, next_states, rewards, returns, dones = (
                states[-max_training_game_size:],
                actions[-max_training_game_size:],
                next_states[-max_training_game_size:],
                rewards[-max_training_game_size:],
                returns[-max_training_game_size:],
                dones[-max_training_game_size:],
            )

        return states, actions, next_states, rewards, returns, dones

    def get_derivatives_policy_state_value_function(
        self, games, policy, output_value_function=False
    ):
        ##begin get_state_value_function code
        states, actions, next_states, rewards, returns, dones = games
        derivative_estimator = Kernel(x=states)
        operator = derivative_estimator.knm(
            x=states, y=states
        ) - self.gamma * derivative_estimator.knm(x=next_states, y=states)
        operator = LAlg.lstsq(operator)

        ##end
        @np.vectorize
        def fun(i, j, k):
            return rewards[i, j] * policy[i, j] * (float(j == k) - policy[i, k])

        coeffs = np.fromfunction(
            fun,
            shape=[rewards.shape[0], self.actions_dim, self.actions_dim],
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
            second_member = (policy * rewards).reshape(policy.shape).sum(1)
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
        """
            Expands the states matrix by duplicating each state for each possible action. 
            Adds the action one hot encoded to each duplicated state.
            This gives back a matrix of shape (num_states * num_actions, state_dim + action_dim)        
        """
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
        self,
        thetas,
        next_states_projection,
        knm,
        rewards,
        maxiter=5,
        reg=1e-9,
        tol=1e-6,
        verbose=False,
        **kwargs,
    ):
        theta = thetas.copy()
        shape = [next_states_projection.shape[0] // self.actions_dim, self.actions_dim]

        def bellman_error(theta, full_output=False):
            error = (
                LAlg.prod(next_states_projection, theta).reshape(shape) * self.gamma
                + rewards
            )
            max_indices = error.argmax(1)
            error = gather(error, max_indices)
            error -= LAlg.prod(knm, theta)
            if full_output == True:
                return np.fabs(error).mean(), max_indices
            return np.fabs(error).mean()

        error, max_indices = bellman_error(theta, full_output=True)
        count = 0
        if error < tol:
            return thetas, error, max_indices
        while count < maxiter and error > tol:
            indices = [
                self.actions_dim * i + max_indices[i] for i in range(len(max_indices))
            ]
            max_projection = next_states_projection[indices]
            next_theta = LAlg.lstsq(knm - max_projection * self.gamma, rewards, reg)

            def f(x):
                interpolated_thetas = theta * x + next_theta * (1.0 - x)
                out = bellman_error(interpolated_thetas)
                return out

            # return next_theta,f(0.)
            xmin, fval, iter, funcalls = optimize.brent(
                f, brack=(0.0, 1.0), maxiter=maxiter, full_output=True
            )
            if fval >= error:
                break
            theta = theta * xmin + next_theta * (1.0 - xmin)
            error, max_indices = bellman_error(theta, full_output=True)
            count = count + 1

        max_indices = LAlg.prod(next_states_projection, theta).reshape(shape).argmax(1)
        indices = [
            self.actions_dim * i + max_indices[i] for i in range(len(max_indices))
        ]
        if verbose:
            print("Computed global error Bellman mean: ", fval, " iter: ", count)
        return theta, fval, indices

    def optimal_states_values_function(
        self, games, kernel=None, full_output=False, **kwargs
    ):
        """
        Find a kernel regressor solving the Bellman equation  $$Q(s,a) = r + \gamma \max_{a'} Q(s',a')$$
        The algorithm computes $Q^n(s,a)$ iteratively : 
            1. Solve $\\theta^{\pi}_{n+1/2} = \Big( K(Z, Z) - \gamma \sum_a \pi_{n+1/2}^a(S) K(W^a,Z)\Big)^{-1} R$
            2. Refines the parameters $\\theta_{n+1}^{\pi} = \lambda \\theta^{\pi}_{n+1/2} + (1 - \lambda) \\theta_{n}^{\pi}.$
        Where: 
            - Z is the concatenation of the states and actions
            - $K(Z,Z)$ is the gram matrix of current state actions pairs
            - $K(W^a,Z)$ is the gram matrix of the next states and actions
            - $\pi_{n+1/2}^a(S) = \delta_{\\arg \max q^n(S,a) }(S)$ is the max of the next Q-values, with $q^n$ the current Q-values.
            - $R$ is the rewards function
        
        The function then assures a limit condition on the Q-values by setting the last Q-values equal to the rewards.

        :param games: :class:`tuple` SARSD in reverse order.
        :param kernel: :class:`codpy.kernel.Kernel` Kernel to be used. If None, a kernel fit on the returns is used. 

        :return: :class:`codpy.kernel.Kernel` The kernel with the optimal Q-values.
        
        """

        states, actions, next_states, rewards, returns, dones = games

        states_actions = np.concatenate([states, actions], axis=1)
        if kernel is None or not kernel.is_valid():
            kernel = self.kernel_type(x=states_actions, fx=returns, **kwargs)
        else:
            states_actions = np.concatenate([kernel.get_x(), states_actions], axis=0)
            rewards = np.concatenate([kernel.games[3], rewards], axis=0)
            next_states = np.concatenate([kernel.games[2], next_states], axis=0)

        next_states_actions = self.all_states_actions(next_states)
        _projection_ = kernel.knm(x=next_states_actions, y=kernel.get_x())

        def helper(i):
            if dones[i] == True:
                return [i * self.actions_dim + j for j in range(self.actions_dim)]

        modif = [
            item
            for i in range(dones.shape[0])
            if dones[i] == True
            for item in helper(i)
        ]
        _projection_[modif] = 0.0

        _knm_ = kernel.knm(x=states_actions, y=kernel.get_x())
        thetas, bellman_error, indices = self.optimal_bellman_solver(
            thetas=kernel.get_theta(),
            next_states_projection=_projection_,
            knm=_knm_,
            rewards=rewards,
            games=games,
            **kwargs,
        )
        kernel.set_theta(thetas)
        kernel.bellman_error = bellman_error
        if full_output:
            return kernel, bellman_error, indices
        return kernel


class KActorCritic(KAgent):
    """KActorCritic Kernel algorithm. It is policy-based and uses a :class:`GamesKernelClassifier` as the actor.
    """

    def __call__(self, state, **kwargs):
        # return 1
        if self.actor.is_valid():
            action_probs = self.actor(core.get_matrix(state).T)
            # if len(self.actor.kernels) > 1:
            #     action_probs = softmax(action_probs, axis=1)
            # action = action_probs.argmax()
            action_probs = action_probs.squeeze()
            action = np.random.choice(len(action_probs), p=action_probs)
            return action
        else:
            return np.random.randint(0, self.actions_dim)

    def get_advantages(self, games, policy, **kwargs):
        """Compute the advantage function $$A^{\pi^a}(s) = R(s,a) + \gamma V^{\pi}(s') - V^{\pi}(s), \quad s'=S(s,a).$$

        Where :
            - $R(s,a)$ is the rewards function
            - $V^{\pi}(s)$ is the value function
            - $S(s,a)$ is the next state function.
        
        """
        states, actions, next_states, rewards, returns, dones = games
        value_function = self.get_state_action_value_function(
            games, policy, max_y=None,**kwargs
        )
        advantages = (
           value_function(self.all_states_actions(next_states)).reshape(actions.shape)
            * self.gamma
            + rewards
        )
        advantages -= value_function(np.concatenate([states, actions], axis=1))
        advantages -= core.get_matrix((advantages).mean(1))

        advantages[dones.flatten()] = 0.0
        return advantages,value_function

    def train(self, game, max_training_game_size=None, **kwargs):
        params = kwargs.get("KCritic", {})
        states, actions, next_states, rewards, returns, dones = self.format(
            game, max_training_game_size=max_training_game_size, **kwargs
        )
        if not hasattr(self,"scores"):
            self.scores = [rewards.sum()]
        else:
            self.scores.append(rewards.sum())
        self.replay_buffer.push(
            states, actions, next_states, rewards, returns, dones, **kwargs
        )
        states, actions, next_states, rewards, returns, dones = (
            self.replay_buffer.memory
        )
        dones[0] = True
        games = [states, actions, next_states, rewards, returns, dones]

        if self.actor.is_valid():
            last_policy = self.actor(states)
        else:
            last_policy = np.full(
                [states.shape[0], self.actions_dim], 1.0 / self.actions_dim
            )
        last_policy = np.where(last_policy < 1e-9, 1e-9,last_policy)
        last_policy = np.where(last_policy > 1.-1e-9,1.- 1e-9,last_policy)
        advantages, value_function = self.get_advantages(games, policy=last_policy, **kwargs)
        # update probabilities
        kernel = self.update_probabilities(
            advantages, games, last_policy=last_policy, kernel= value_function, **kwargs
        )
        self.actor = kernel


class KQLearning(KActorCritic):
    """Implements KQLearning algorithm. Uses clustering by default in the :func:`train` method.
    
    """

    def __call__(self, state, **kwargs):
        self.eps_threshold *= 0.999
        if np.random.random() > self.eps_threshold and self.critic.is_valid() == True:
            z = self.all_states_actions(core.get_matrix(state).T)
            q_values = self.critic(z)
            q_values += np.random.random(q_values.shape) * 1e-9
            return np.argmax(q_values)
        return np.random.randint(0, self.actions_dim)

    def train(
        self,
        game,
        max_training_game_size=None,
        format=True,
        tol=1e-2,
        **kwargs,
    ):

        if format:
            states, actions, next_states, rewards, returns, dones = self.format(
                game, max_training_game_size=max_training_game_size, **kwargs
            )
        else:
            states, actions, next_states, rewards, returns, dones = game

        if self.critic.is_valid():
            returns = self.critic(np.concatenate([states, actions], axis=1))
            games = states, actions, next_states, rewards, returns, dones

        self.replay_buffer.push(
            states, actions, next_states, rewards, returns, dones, capacity=sys.maxsize
        )

        if (
            len(self.replay_buffer) <= self.replay_buffer.capacity
        ):  # and self.critic.update_kernels == False:

            games = self.replay_buffer.memory
            kernel = self.optimal_states_values_function(games, verbose=True, **kwargs)
            kernel.games = games
            if len(self.critic.kernels) == 0:
                self.critic.add_kernel(kernel)
            else:
                self.critic.kernels[len(self.critic.kernels) - 1] = kernel
            return
        else:
            kernel = self.critic.kernels[len(self.critic.kernels) - 1]
            kernel.set(
                x=kernel.get_x()[: self.replay_buffer.capacity // 2],
                fx=kernel.get_fx()[: self.replay_buffer.capacity // 2],
            )
            kernel.games = [
                elt[: self.replay_buffer.capacity // 2]
                for elt in self.replay_buffer.memory
            ]
            games = self.replay_buffer.memory
            games = [
                elt[self.replay_buffer.capacity // 2 :]
                for elt in self.replay_buffer.memory
            ]
            kernel = self.optimal_states_values_function(games, verbose=True, **kwargs)
            kernel.games = games
            self.critic.add_kernel(kernel)
            self.replay_buffer.memory = games
        delete_kernels = []
        for i, k in self.critic.kernels.items():
            error = self.critic.kernels[i].bellman_error
            if error > tol and not hasattr(self.critic.kernels[i], "flag_kill_me"):
                kernel = self.optimal_states_values_function(
                    k.games, kernel=k, verbose=True, **kwargs
                )
                kernel.games = self.critic.kernels[i].games
                if error <= self.critic.kernels[i].bellman_error:
                    # delete_kernels.append(i)
                    kernel.flag_kill_me = "please"
                else:
                    self.critic.kernels[i] = kernel
        if (
            len(delete_kernels) > 0
            and len(self.critic.kernels) - len(delete_kernels) > 1
        ):
            new_kernels = {}
            # count = 0
            # for i in range(len(self.critic.kernels)):
            #     if i not in delete_kernels:
            #         new_kernels[count] = self.critic.kernels[i]
            #         count = count+1
            # self.critic.kernels = new_kernels


class PolicyGradient(KActorCritic):
    """
    PolicyGradient Kernel algorithm. It is policy-based and uses a :class:`GamesKernelClassifier` as the actor.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        params = kwargs.get("KCritic", {})
        self.actor = GamesKernelClassifier(**params)

    def __call__(self, state, **kwargs):
        if self.actor.get_x() is not None and self.actor.get_x().shape[0] > 1:
            action_probs = self.actor(core.get_matrix(state).T)
            action_probs = action_probs.squeeze()
            action = np.random.choice(len(action_probs), p=action_probs)
            # action = action_probs.argmax()
            return action
        else:
            return np.random.randint(0, self.actions_dim)

    def get_advantages(self, games, policy, **kwargs):
        """Compute
        $$A^{\pi}(s) = \\nabla_{y} Q^\pi_k(\cdot) = K(\cdot, Z) \\nabla_{y} \\theta^\pi.$$

        :param games: :class:`tuple` SARSD in reverse order.
        :param policy: :class:`np.ndarray` The policy to be used for weigthing the next state-actions.
        :param kwargs: :class:`dict` 

        :return: :class:`tuple` The  advantages of the policy along with a kernel estimator for new advantages on state-action pairs.
        """
        # advantage taken as A = \nabla_\pi \pi Q^{pi}(S_T,A_T), so that the overall gradient policy can be written as
        #  d/di \pi(t) = d/d\pi Q^{pi}(S_T,A_T)
        #  Thus formally d/dt  Q^{pi}(S_T,A_T) = < \nabla_\pi \pi Q^{pi}(S_T,A_T), d/dt \pi> = | \nabla_\pi \pi Q^{pi}(S_T,A_T)|^2
        states, actions, next_states, rewards, returns, dones = games
        derivative_estimator = self.get_derivatives_policy_state_action_value_function(
            games, policy, **kwargs
        )
        states_actions = np.concatenate([states, actions], axis=1)
        derivative_estimations = derivative_estimator(states_actions)
        return derivative_estimations, derivative_estimator


class KController(KAgent):
    """
    Implements the KController algorithm. The specificities of this algorithm is that it uses a heuristic controller to be tuned.
    """
    def __init__(self, state_dim, actions_dim, controller, **kwargs):
        self.controller = controller
        self.x, self.y = None, None
        self.expectation_estimator = None
        self.label = kwargs.get("label", None)
        super().__init__(state_dim=state_dim, actions_dim=actions_dim, **kwargs)

    def __call__(self, z, **kwargs):
        """
        The internal tuned heuristic controller directly outputs the action. 

        :param z: :class:`np.ndarray the state 

        :return: :class:`int`
        
        """
        return self.controller(z, **kwargs)

    def get_reward(self, game, **kwargs):
        states, actions, next_states, rewards, dones = game
        return get_matrix(rewards).mean()

    def get_expectation_estimator(self, x, y, **kwargs):
        class explore_kernel(Kernel):
            def distance(self, z, **kwargs):
                out = get_matrix(self.dnm(x=self.get_x(), y=z).min(axis=0))
                return out

        params = kwargs.get("KController", {})
        self.expectation_kernel = explore_kernel(x=x, fx=y, **params)
        return self.expectation_kernel

    def get_function(self, **kwargs):
        """Defines the function to be optimized ${L}(R_{k,\lambda_e},\\theta)$.
        """
        self.expectation_estimator = self.get_expectation_estimator(
            self.x, self.y, **kwargs
        )

        # self.min_expectation_estimator = self.expectation_estimator(self.x).flatten()
        # self.min_expectation_estimator.sort()
        # self.min_expectation_estimator = self.min_expectation_estimator[int(self.x.shape[0] * 0.9)]
        def function(x):
            expectation = self.expectation_estimator(
                x
            )  # - self.min_expectation_estimator
            distance = self.expectation_estimator.distance(x)
            return expectation  # + distance

        return function  # to cope with exploration

    def train(self, game, **kwargs):
        """
        Solves for $$\\theta_{n+1} = \\arg \max_{\\theta \in \Theta_n} \mathcal{L}(R_{k,\lambda_e},\\theta), \quad \Theta_{n} = \\bar{\\theta_e} \cup \Theta_{N,n}$$
        
        Where $\Theta_{N,n}$ is a screening around the last $\\theta_n$ and is defined as follow:
        $$\Theta_{N,n} = (\\theta_n+\\alpha^n \Theta_N) \cap \Theta$$ with $\\alpha^n$ is a contracting factor and ${L}(R_{k,\lambda_e},\\theta)$ is an optimization function which can be defined and tuned based on your needs at :func:`get_function`.
        
        :param game: :class:`tuple` SARSD in reverse order.
        """
        states, actions, next_states, rewards, dones = self.format(game, **kwargs)
        self.replay_buffer.push(states, actions, next_states, rewards, dones, **kwargs)
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
            last_val_max_min = last_val - last_vals.min()
            max_val, new_theta = optimization.continuous_optimizer(
                function,
                self.controller.get_distribution(),
                include=self.x,
                **kwargs,
            )
            new_theta = get_matrix(new_theta).T
            if last_val < max_val:  # debug to see if parameters are updated
                tot = 1
                pass
            self.controller.set_thetas(new_theta)
        else:
            self.controller.set_thetas(self.controller.get_distribution()(1))


def get_conditioned_kernel(
    games,
    expectation_kernel,
    base_class=codpy.conditioning.PiKernel,
    # base_class=codpy.conditioning.ConditionerKernel,
    # base_class=codpy.conditioning.NadarayaWatsonKernel,
    **kwargs,
):
    class ConditionerKernel(base_class):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)

    states, actions, next_states, rewards, returns, dones = games
    expected_states = expectation_kernel(np.concatenate([states, actions], axis=1))
    # permutation = Kernel(x=states).map(next_states, distance="norm22").permutation
    # states = states[permutation]
    # expected_states = expected_states[permutation]

    params = kwargs.get("HJBModel", kwargs)
    out = ConditionerKernel(
        x=np.concatenate([expected_states, actions], axis=1),
        y=next_states,
        expectation_kernel=expectation_kernel,
        **params,
    )
    # out = ConditionerKernel(x=states_actions,y=next_states,expectation_kernel=expectation_kernel,**params)
    return out


def get_expectation_kernel(games, **kwargs):
    class expectation_kernel(Kernel):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.state_dim = kwargs.get("state_dim", None)

        def __call__(self, z, **kwargs):
            out = super().__call__(z) + z[:, : self.state_dim]
            return out

    states, actions, next_states, rewards, returns, dones = games
    states_actions = np.concatenate([states, actions], axis=1)
    params = kwargs.get("HJBModel", kwargs)
    out = expectation_kernel(x=states_actions, fx=next_states - states, **params)
    noise = next_states - out(states_actions)
    out.set_fx(out.get_fx() + noise.mean(axis=0))
    return out


class KQLearningHJB(KQLearning):
    """Implements the Hamilton-Jacobi-Bellman Q-learning algorithm.
    """

    def optimal_states_values_function(
        self, games, kernel=None, full_output=False, maxiter=5, reorder=False, **kwargs
    ):
        """
        Solve the Bellman equation $$Q^{\pi}(s_t,a_t) = R(s_t,a_t) + \gamma \int \left[ \sum_{a \in \mathcal{A}} \pi^a(s_t) Q^{\pi}(s',a)\\right] d \mathbb{P}_S(s',s_t,a_t).$$
        Numerically, we effectively solve for the set of parameters $\\theta$ of the kernel $K$ such that:
        $$\\theta = \Big( K(Z, Z) - \gamma \sum_{a} \pi^a(S)\Gamma(P^a) K(P, Z)\Big)^{-1} R, \quad P = \{ S+F_k(S,a), a \}$$
        
        Where:
            - $K(Z,Z)$ is the kernel matrix of the states and actions.
            - $P$ is the set of the predicted next state actions possibilities.
            - $\Gamma(P^a)$ is the transition probability matrix. 
            - $K(P,Z)$ is the kernel matrix of the predicted next state actions and the states and actions.
        """

        states, actions, next_states, rewards, returns, dones = games
        states_actions = np.concatenate([states, actions], axis=1)
        if kernel is None or not kernel.is_valid():
            kernel = self.kernel_type(x=states_actions, fx=returns, **kwargs)

        expectation_kernel_ = self.get_expectation_kernel(games, **kwargs)
        conditioned_kernel_ = self.get_conditioned_kernel(
            games=games, expectation_kernel=expectation_kernel_, **kwargs
        )
        states_actions = np.concatenate([states, actions], axis=1)
        next_expected_states_actions = expectation_kernel_(states_actions)
        next_expected_all_states_actions = self.all_states_actions(
            next_expected_states_actions
        )
        _projection_ = kernel.knm(x=next_expected_all_states_actions, y=kernel.get_x())
        _knm_ = kernel.knm(x=states_actions, y=kernel.get_x())

        def helper(i):
            if dones[i] == True:
                return [i * self.actions_dim + j for j in range(self.actions_dim)]

        modif = [
            item
            for i in range(dones.shape[0])
            if dones[i] == True
            for item in helper(i)
        ]
        _projection_[modif] = 0.0

        thetas, bellman_error, indices = self.optimal_bellman_solver(
            thetas=kernel.get_theta(),
            next_states_projection=_projection_,
            knm=_knm_,
            rewards=rewards,
            maxiter=10,
            games=games,
            **kwargs,
        )

        thetas = conditioned_kernel_.get_transition(
            y=next_expected_all_states_actions[indices, : self.state_dim],
            x=np.concatenate([next_expected_states_actions, actions], axis=1),
            fx=thetas,
        )
        kernel.set_theta(thetas)
        # fx = conditioned_kernel_.get_transition(
        #     y = next_expected_all_states_actions[indices,:self.state_dim],
        #     x = np.concatenate([next_expected_states_actions, actions], axis=1),
        #     fx = kernel.get_fx()
        # )
        # kernel.set_fx(fx)
        kernel.bellman_error = bellman_error

        if full_output:
            return kernel, bellman_error, indices
        return kernel


class KActorCriticHJB(KActorCritic):

    def train(self, game, max_training_game_size=None, **kwargs):
        params = kwargs.get("KCritic", {})
        states, actions, next_states, rewards, returns, dones = self.format(
            game, max_training_game_size=max_training_game_size, **kwargs
        )
        self.replay_buffer.push(
            states, actions, next_states, rewards, returns, dones, **kwargs
        )
        states, actions, next_states, rewards, returns, dones = (
            self.replay_buffer.memory.copy()
        )
        games = [states, actions, next_states, rewards, returns, dones]

        if self.actor.get_x() is not None and self.actor.get_x().shape[0] > 1:
            last_policy = self.actor(states)
        else:
            last_policy = np.full(
                [states.shape[0], self.actions_dim], 1.0 / self.actions_dim
            )
        advantages = self.get_advantages(games, policy=last_policy, **kwargs)
        # update probabilities
        kernel = self.update_probabilities(
            advantages, games, last_policy=last_policy, **kwargs
        )
        self.actor = kernel

    def get_state_action_value_function(
        self, games, policy, verbose=False, kernel=None, **kwargs
    ):
        states, actions, next_states, rewards, returns, dones = games
        states_actions = np.concatenate([states, actions], axis=1)
        # next_states_actions = self.all_states_actions(next_states)
        if kernel is None:
            kernel = self.kernel_type(x=states_actions, fx=returns, **kwargs)

        expectation_kernel_ = self.get_expectation_kernel(games, **kwargs)
        conditioned_kernel_ = self.get_conditioned_kernel(
            games, expectation_kernel=expectation_kernel_, **kwargs
        )

        next_expected_states = expectation_kernel_(states_actions)
        next_expected_states_actions = self.all_states_actions(next_expected_states)
        next_expected_states = np.concatenate([next_expected_states, actions], axis=1)

        _projection_ = kernel.knm(x=next_expected_states_actions, y=kernel.get_x())
        _knm_ = kernel.knm(x=states_actions, y=kernel.get_x())

        _projection_ = conditioned_kernel_.get_transition_kernel()(
            next_expected_states_actions[:, : self.state_dim],
            next_expected_states_actions,
            _projection_,
        )
        _projection_ = _projection_.reshape(
            [
                states_actions.shape[0],
                _projection_.shape[0] // states_actions.shape[0],
                states_actions.shape[0],
            ]
        )
        sum_policy = np.einsum("...ji,...j", _projection_, policy)
        thetas = LAlg.lstsq(_knm_ - sum_policy * self.gamma, rewards)
        kernel.set_theta(thetas)

        # check
        # error = self.bellman_error(states,actions,next_states, rewards,policy, value_function)
        return kernel
