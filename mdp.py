import matplotlib.pyplot as plt
import numpy as np


class MDP(object):

    def __init__(self, states, actions, transition_probabilities, rewards, discount_factor, threshold=1e-6):

        """
        % Constructor method

        :param states: List of integers, last state corresponds to absorbing state. In our case, it is 'death'
        :param actions: List of integers
        :param transition_probabilities: 3-D Numpy array, |S|x|A|x|S|
        :param rewards: 2-D Numpy array, |S|x|A|
        :param discount_factor: float, a scalar between 0 and 1
        :param threshold: float, optional and 1e-6 by default
        """

        self.S = states
        self.A = actions
        self.P = transition_probabilities
        self.R = rewards
        self.gamma = discount_factor
        self.epsilon = threshold

        self.V = np.zeros(shape=len(self.S))  # Initially set values to zero
        self.pi = {}  # policy map, it will be filled after value iteration

    def one_step_update(self):

        """
        It updates value function for one iteration
        and returns the delta associated with it

        :return: Float
        """

        delta = 0
        for s in self.S:
            if s != self.S[-1]:  # no action to be taken for the absorbing state
                previous_val = self.V[s]
                self.V[s] = max([self.R[s, a] + self.gamma * np.sum(self.P[s, a, :] * self.V[:]) for a in self.A])
                delta = max(delta, abs(self.V[s] - previous_val))

        return delta

    def value_iteration(self):

        """
        Applies value iterations and fills the
        class attribute 'self.V'

        :return: %INPLACE%
        """

        while True:
            delta = self.one_step_update()
            if delta < self.epsilon:
                break

    def extract_policy(self):

        """
        Extracts the policy with the current value function
        :return: %INPLACE%
        """

        for s in self.S:
            if s != self.S[-1]:  # no action to be taken for the absorbing state
                self.pi[s] = np.argmax(np.array([self.R[s, a] + self.gamma *
                                                 np.sum(self.P[s, a, :] * self.V[:]) for a in self.A]))

    def run(self):

        """
        Runs value iteration algorithm and then extracts the
        optimal policy according to the optimal value functions found
        :return: %INPLACE%
        """

        self.value_iteration()
        self.extract_policy()

    def plot_policy(self):

        """
        Plot the policy

        :return: %INPLACE%
        """

        plt.figure(figsize=(12, 7))
        s_plus = [s for s in self.S if s != self.S[-1]]
        plt.scatter(s_plus, [self.pi[s] for s in s_plus], c='r', s=100)
        plt.plot(s_plus, [self.pi[s] for s in s_plus], c='r')
        plt.xlabel('states')
        plt.ylabel('actions')
        plt.xticks(np.arange(min(self.S), max(self.S), 1.0))
        plt.yticks(np.arange(min(self.A), max(self.A) + 1, 1.0))
        plt.title('Optimal Policy After Value Iteration')
        plt.grid()
        plt.show()
