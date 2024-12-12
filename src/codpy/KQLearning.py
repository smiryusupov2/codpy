import os
import sys

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import random

import numpy as np
import pandas as pd
from scipy import optimize
from scipy.special import softmax

import codpy.core as core
import codpy.selection as selection
from codpy.data_processing import hot_encoder
from codpy.kernel import Kernel, KernelClassifier
from codpy.lalg import lalg as lalg
from codpy.utils import gather


class ReplayBuffer(object):
    def __init__(self, capacity, memory=None, **kwargs):
        if memory is None:
            self.memory = None
        else:
            self.memory = list(memory)
        self.capacity = capacity

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

    def push(self, *args, capacity=None, **kwargs):
        if capacity is not None:
            self.capacity = capacity
        if self.memory is None:
            self.memory = list(args)
        else:

            def helper(i):
                return np.concatenate([self.memory[i], args[i]], axis=0)

            self.memory = [helper(i) for i in range(len(args))]
        self.update()
        pass

    def sample(self, batch_size):
        if len(self) == 0:
            return None
        indices = list(range(len(self)))
        if batch_size > len(self):
            self.batch_indices = indices
            return [self.memory[i] for i in range(len(self.memory))]
        self.batch_indices = random.sample(indices, batch_size)
        # self.batch_indices = [indices[int(i/batch_size*len(self))] for i in range(batch_size)]
        out = [self.memory[i][self.batch_indices] for i in range(len(self.memory))]
        return out

    def __len__(self):
        if self.memory is None:
            return 0
        return len(self.memory[0])


class KActor(KernelClassifier):
    pass


def rl_hot_encoder(actions, actions_dim):
    out = hot_encoder(pd.DataFrame(actions), cat_cols_include=[0])
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
    probs = np.log(probs) + advantages
    out = softmax(probs, axis=1)
    return out


def get_tensor_probas(policy):
    @np.vectorize
    def fun(i, j, k):
        return policy[i, j] * (float(j == k) - policy[i, k])

    return np.fromfunction(
        fun, shape=[policy.shape[0], policy.shape[1], policy.shape[1]], dtype=int
    )


class KACAgent:
    def __init__(self, actions_dim, state_dim, gamma=0.99, **kwargs):
        self.actions_dim = actions_dim
        self.state_dim = state_dim
        self.gamma = gamma
        self.all_actions_ = None
        params = kwargs.get("KActor", {})
        self.actor = KActor(**params)
        params = kwargs.get("KCritic", {})
        self.critic = Kernel(**params)
        self.replay_buffer = ReplayBuffer(**kwargs)
        self.eps_threshold = kwargs.get("eps_threshold", 1.0)
        self.keep = kwargs.get("keep", int(1.0 / (1.0 - self.gamma)))

    def compute_returns(self, states, actions, next_states, rewards, dones, **kwargs):
        returns, next_return = [], 0.0
        # for t in reversed(range(len(rewards))): # we already reversed time
        for t in range(len(rewards)):
            next_return = rewards[t] + self.gamma * next_return
            returns.append(next_return)
        return np.array(returns)

    def bellman_error(
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
    ):  ##this function assumes that advantages[i,j]=KCritic([states[i],j])
        if dt is None:
            dt = 1.0 / (advantages * advantages).mean()
        interpolated_policy = Verhulst(last_policy, advantages * dt)
        return KActor(x=states, fx=np.log(interpolated_policy))

    def get_state_action_value_function(
        self, states, actions, next_states, rewards, policy
    ):
        states_actions = np.concatenate([states, actions], axis=1)
        next_states_actions = self.all_states_actions(next_states)
        value_function = Kernel()
        value_function.set(x=states_actions)
        Knm = value_function.Knm(x=states_actions, y=states_actions)
        projection_op = value_function.Knm(
            x=next_states_actions, y=states_actions
        ).reshape([states_actions.shape[0], self.actions_dim, states_actions.shape[0]])
        sum_policy = np.einsum("...ji,...j", projection_op, policy)
        projection_op = lalg.lstsq(Knm - sum_policy * self.gamma)
        thetas = lalg.prod(projection_op, rewards)
        value_function.set_theta(thetas)

        # check
        # error = self.bellman_error(states,actions,next_states, rewards,policy, value_function)
        return value_function

    def get_derivatives_policy_state_action_value_function(
        self, states, actions, next_states, rewards, policy, output_value_function=False
    ):
        ##begin get_state_action_value_function code
        states_actions = np.concatenate([states, actions], axis=1)
        next_states_actions = self.all_states_actions(next_states)
        value_function = Kernel()
        value_function.set(x=states_actions)
        Knm = value_function.Knm(x=states_actions, y=states_actions)
        projection_op = value_function.Knm(
            x=next_states_actions, y=states_actions
        ).reshape([states_actions.shape[0], self.actions_dim, states_actions.shape[0]])
        sum_policy = np.einsum("...ji,...j", projection_op, policy)
        projection_op = lalg.lstsq(Knm - sum_policy * self.gamma)
        thetas = lalg.prod(projection_op, rewards)
        value_function.set_theta(thetas)
        ##end
        next_states_values = value_function(next_states_actions).reshape(
            [states_actions.shape[0], self.actions_dim]
        )
        coeffs = get_tensor_probas(policy)
        second_member = np.einsum("...i,...ij", next_states_values, coeffs)

        derivative_estimator = Kernel()
        derivative_estimator.set_x(states_actions)
        derivative_estimator.set_theta(lalg.prod(projection_op, second_member))
        if output_value_function:
            return derivative_estimator, value_function
        return derivative_estimator

    def get_state_value_function(
        self, states, actions, next_states, rewards_matrix, policy
    ):
        value_function = Kernel(x=states)
        operator_inv = value_function.Knm(
            x=states, y=states
        ) - self.gamma * value_function.Knm(x=next_states, y=states)
        operator = lalg.lstsq(operator_inv)
        second_member = core.get_matrix((policy * rewards_matrix).sum(1))
        value_function.set_theta(lalg.prod(operator, second_member))
        # def check():
        #     test = value_function(states)-self.gamma*value_function(next_states)-second_member
        #     assert(np.abs(test).max() < 1e-4)
        # check()
        return value_function

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
        operator = derivative_estimator.Knm(
            x=states, y=states
        ) - self.gamma * derivative_estimator.Knm(x=next_states, y=states)
        operator = lalg.lstsq(operator)

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

        derivative_estimator.set_theta(lalg.prod(operator, coeffs))

        # def check():
        #     #check the derivated Bellman relation \nabla_ln \pi ( V(S_T) - gamma V(S_{T+1} - \E(R) ) =0)
        #     test = derivative_estimator(states)-self.gamma*derivative_estimator(next_states)-coeffs
        #     assert(np.abs(test).max() < 1e-4)
        # check()

        if output_value_function:
            value_function = Kernel(x=states)
            second_member = (policy * rewards_matrix).reshape(policy.shape).sum(1)
            value_function.set_theta(lalg.prod(operator, second_member))
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


class KActorCritic(KACAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __call__(self, state, **kwargs):
        # return 1
        action_probs = self.actor(core.get_matrix(state).T)
        if action_probs is not None:
            action_probs = action_probs.squeeze()
            # action = action_probs.argmax()
            action = np.random.choice(len(action_probs), p=action_probs)
            return action
        else:
            return np.random.randint(0, self.actions_dim)

    def format(self, sarsd, **kwargs):
        if (
            not hasattr(self, "keep")
            or self.keep == None
            or len(self.replay_buffer) < self.replay_buffer.capacity
        ):
            state, action, next_state, reward, done = [
                core.get_matrix(e) for e in sarsd
            ]
        else:
            state, action, next_state, reward, done = [
                core.get_matrix(e)[: self.keep] for e in sarsd
            ]

        action = rl_hot_encoder(action, self.actions_dim)
        done = core.get_matrix(done, dtype=bool)
        return state, action, next_state, reward, done

    def get_advantages(self, states, actions, next_states, rewards, dones, policy):
        value_function = self.get_state_action_value_function(
            states, actions, next_states, rewards, policy
        )
        advantages = value_function(self.all_states_actions(next_states)).reshape(
            actions.shape
        )
        advantages -= core.get_matrix((advantages).mean(1))
        return advantages

    def train(self, game, **kwargs):
        states, actions, next_states, rewards, dones = self.format(game, **kwargs)
        returns = self.compute_returns(
            states, actions, next_states, rewards, dones, **kwargs
        )
        self.replay_buffer.push(states, actions, next_states, rewards, returns, dones)
        states, actions, next_states, rewards, returns, dones = (
            self.replay_buffer.memory.copy()
        )
        states_actions = np.concatenate([states, actions], axis=1)
        self.rewards = Kernel(x=states_actions, fx=rewards, order=1)

        if self.actor.get_x() is not None:
            last_policy = self.actor(states)
        else:
            last_policy = np.full(
                [states.shape[0], self.actions_dim], 1.0 / self.actions_dim
            )
        # compute advantages
        advantages = self.get_advantages(
            states, actions, next_states, rewards, dones, last_policy
        )
        ### update probabilities
        self.actor = self.update_probabilities(
            advantages, states, actions, next_states, rewards, dones, last_policy
        )
        pass


class KQLearning(KActorCritic):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # self.gamma,self.theta = gamma,theta
        self.count = 0
        params = kwargs.get("Rewards", {})
        self.rewards = Kernel(**params)
        params = kwargs.get("NextStates", {})
        self.next_states = Kernel(**params)

    def __call__(self, state, **kwargs):
        # return 1
        sample = np.random.random()
        self.eps_threshold *= 0.995
        if sample > self.eps_threshold and self.critic.get_x() is not None:
            z = core.get_matrix(state)
            if z.shape[1] != self.actions_dim:
                z = self.all_states_actions(core.get_matrix(state).T)
            q_values = self.get_q_values(z, deep=0) + sample * 0.001
            # q_values = self.rewards(z,deep=1) + sample*.001
            return np.argmax(q_values)
        return np.random.randint(0, self.actions_dim)

    def get_q_values(
        self, states_actions, agent=None, dones=None, rewards=None, deep=1
    ):  # estimate q(s,a) = gamma max_b q(S(s,a),b)+r(s,a)
        if agent is None:
            agent = self.critic

        def get_next_q_values(
            states_actions, deep, agent
        ):  # estimate q(s,a) = gamma max_b q(S(s,a),b)+r(s,a)
            next_states = self.next_states(states_actions)
            next_states_actions = self.all_states_actions(next_states)
            rewards = self.rewards(states_actions)
            next_q_values = (
                self.get_q_values(
                    next_states_actions,
                    agent=agent,
                    dones=dones,
                    rewards=rewards,
                    deep=deep - 1,
                ).reshape([states_actions.shape[0], self.actions_dim])
                * self.gamma
                + rewards
            )
            max_indices = core.get_matrix(next_q_values.argmax(axis=1), dtype="int")
            q_values = gather(next_q_values, max_indices)
            return q_values

        if deep == 0:
            q_values = agent(z=states_actions)
        else:
            q_values = get_next_q_values(states_actions, deep=deep, agent=agent)
        return q_values

    def train(self, game, **kwargs):
        tau = 0.99
        states, actions, next_states, rewards, dones = self.format(game, **kwargs)
        tol, dt, theta, gamma = (
            kwargs.get("tol", 1.0e-6),
            kwargs.get("dt", 0.1),
            kwargs.get("theta", 0.0),
            kwargs.get("gamma", self.gamma),
        )
        itermax = kwargs.get("itermax", int(2.0 / (1.0 - self.gamma))) + 1
        step = kwargs.get("steps", None)
        self.count += states.shape[0]
        returns = self.compute_returns(
            states, actions, next_states, rewards, dones, **kwargs
        )
        self.replay_buffer.push(states, actions, next_states, rewards, returns, dones)
        states_actions = np.concatenate([states, actions], axis=1)
        next_states_actions = np.concatenate([next_states, actions], axis=1)
        err_rewards = self.rewards(states_actions)
        if err_rewards is not None:
            err_rewards = np.fabs(rewards - err_rewards)
            print("error rewards:", err_rewards.max())
            err_next_states = np.fabs(next_states - self.next_states(states_actions))
            print("error next states:", err_next_states.max())
            # q_values = self.get_q_values(states_actions,next_states_actions=next_states_actions,rewards=rewards)
            q_values = self.get_q_values(states_actions, dones=dones, deep=0)
            err_returns = np.fabs(returns - q_values)
            print("error returns:", err_returns.max())

        states, actions, next_states, rewards, returns, dones = (
            self.replay_buffer.memory.copy()
        )
        states_actions = np.concatenate([states, actions], axis=1)

        self.next_states.set(
            x=states_actions[-self.next_states.max_pool :],
            fx=next_states[-self.next_states.max_pool :],
        )
        self.rewards.set(
            x=states_actions[-self.rewards.max_pool :],
            fx=rewards[-self.rewards.max_pool :],
        )
        err_rewards = np.fabs(rewards - self.rewards(states_actions))
        print("error rewards:", err_rewards.max())
        err_next_states = np.fabs(next_states - self.next_states(states_actions))
        print("error next states:", err_next_states.max())

        indices = self.critic.select(x=states_actions, N=128, fx=returns, n_batch=1)
        states, actions, next_states, rewards, returns, dones = (
            states[indices],
            actions[indices],
            next_states[indices],
            rewards[indices],
            returns[indices],
            dones[indices],
        )
        states_actions = np.concatenate([states, actions], axis=1)

        err, tol, count = sys.float_info.max, 1e-6, 0
        prev_q_values = returns.copy()
        # while err > tol and count < 2:
        while err > tol and count < int(1.0 / (1.0 - self.gamma)):
            q_values = self.get_q_values(
                states_actions, dones=dones, rewards=rewards, deep=1
            )
            self.critic.update(z=states_actions, fz=q_values)
            err = prev_q_values - q_values
            err = np.fabs(err).max()
            prev_q_values = q_values.copy()
            count += 1
        print("error bellmann ", err, " iterations:", count)
        states, actions, next_states, rewards, returns, dones = (
            self.replay_buffer.memory.copy()
        )
        states_actions = np.concatenate([states, actions], axis=1)
        returns = self.critic(z=states_actions) * (1.0 - tau) + tau * returns
        returns[dones] = rewards[dones]
        returns = np.where(returns < 0.0, 0.0, returns)
        returns = np.where(
            returns > 1.0 / (1.0 - self.gamma), 1.0 / (1.0 - self.gamma), returns
        )
        self.critic.update(z=states_actions, fz=returns)
        self.replay_buffer.memory = (
            states,
            actions,
            next_states,
            rewards,
            returns,
            dones,
        )
        pass


class KQLearning2(KQLearning):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        params = kwargs.get("KCritic", {})
        self.actor = Kernel(**params)

    def __call__(self, state, **kwargs):
        self.eps_threshold *= 0.995
        if np.random.random() > self.eps_threshold and self.critic.get_x() is not None:
            z = self.all_states_actions(core.get_matrix(state).T)
            # q_values = self.critic(z)
            q_values = self.get_q_values(z, deep=1)
            q_values += np.random.random(q_values.shape) * 0.001
            return np.argmax(q_values)
        return np.random.randint(0, self.actions_dim)

    def bellman_solver(
        self, states, actions, next_states, rewards, returns, dones, max_count=1
    ):
        states_actions = np.concatenate([states, actions], axis=1)
        next_states_actions = self.all_states_actions(next_states)
        projection_op = self.actor.Knm(x=next_states_actions, y=self.actor.get_x())
        Knm = self.actor.Knm(x=states_actions, y=self.actor.get_x())
        error, count = 1e10, 0
        shape = actions.shape

        def loop(max_indices):
            indices = [
                self.actions_dim * i + max_indices[i] for i in range(len(max_indices))
            ]
            max_projection = Knm - projection_op[indices] * self.gamma
            next_theta = lalg.lstsq(max_projection, rewards)

            def f(x):
                interpolated_thetas = self.actor.get_theta() * x + next_theta * (
                    1.0 - x
                )
                bellman_error = (
                    lalg.prod(Knm, interpolated_thetas)
                    - core.get_matrix(
                        lalg.prod(projection_op, interpolated_thetas)
                        .reshape(shape)
                        .max(1)
                    )
                    * self.gamma
                    - rewards
                )
                return np.fabs(bellman_error).max()

            xmin, fval, iter, funcalls = optimize.brent(
                f, brack=(0.0, 1.0), maxiter=5, full_output=True
            )
            return xmin, fval, next_theta

        while error > 0.01 and count < max_count:
            next_states_values = (
                lalg.prod(projection_op, self.actor.get_theta()).reshape(shape)
                * self.gamma
                + rewards
            )
            max_indices = next_states_values.argmax(1)
            xmin, error_new, next_theta = loop(max_indices)
            if error_new >= error:
                break
            self.actor.set_theta(
                self.actor.get_theta() * xmin + next_theta * (1.0 - xmin)
            )
            count, error = count + 1, error_new

        next_states_values = (
            self.actor(next_states_actions).reshape(shape) * self.gamma + rewards
        )
        local_error = self.actor(states_actions) - core.get_matrix(
            next_states_values.max(1)
        )
        print(
            "local error Bellman max: ",
            np.fabs(local_error).max(),
            "local error Bellman mean: ",
            np.fabs(local_error).mean(),
            " count: ",
            count,
        )
        return self.actor

    def train(self, game, **kwargs):
        states, actions, next_states, rewards, dones = self.format(game, **kwargs)
        states_actions = np.concatenate([states, actions], axis=1)
        if self.critic.get_x() is not None:
            states_values = self.critic(states_actions)
        else:
            states_values = self.compute_returns(
                states, actions, next_states, rewards, dones, **kwargs
            )
        self.replay_buffer.push(
            states, actions, next_states, rewards, states_values, dones
        )
        states, actions, next_states, rewards, returns, dones = (
            self.replay_buffer.memory
        )
        states_actions = np.concatenate([states, actions], axis=1)
        self.actor.set(x=states_actions, fx=returns)
        self.next_states.set(x=states_actions, fx=next_states)
        self.rewards.set(x=states_actions, fx=rewards)

        kernel = self.bellman_solver(
            states, actions, next_states, rewards, returns, dones, max_count=10
        )
        self.critic.set(x=kernel.get_x())
        self.critic.set_theta(kernel.get_theta())
        self.replay_buffer.memory = (
            states,
            actions,
            next_states,
            rewards,
            self.critic(states_actions),
            dones,
        )


class PolicyGradient(KActorCritic):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        params = kwargs.get("KCritic", {})
        self.critic = KActor(**params)

    def __call__(self, state, **kwargs):
        # return 1
        action_probs = self.actor(core.get_matrix(state).T)
        if action_probs is not None:
            action_probs = action_probs.squeeze()
            # action = action_probs.argmax()
            action = np.random.choice(len(action_probs), p=action_probs)
            return action
        else:
            return np.random.randint(0, self.actions_dim)

    def get_advantages(self, states, actions, next_states, rewards, dones, policy):
        # advantage taken as A = \nabla_\pi \pi Q^{pi}(S_T,A_T), so that the overall gradient policy can be written as
        #  d/di \pi(t) = d/d\pi Q^{pi}(S_T,A_T)
        #  Thus formally d/dt  Q^{pi}(S_T,A_T) = < \nabla_\pi \pi Q^{pi}(S_T,A_T), d/dt \pi> = | \nabla_\pi \pi Q^{pi}(S_T,A_T)|^2
        derivative_estimator = self.get_derivatives_policy_state_action_value_function(
            states, actions, next_states, rewards, policy
        )
        states_actions = np.concatenate([states, actions], axis=1)
        derivative_estimations = derivative_estimator(states_actions)
        return derivative_estimations
