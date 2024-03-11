from agents.epsilon_greedy_agent import EpsilonGreedyAgent
import numpy as np

class CssEpsilonGreedyAgent(EpsilonGreedyAgent):
    def __init__(self, reward_estimates, epsilon, ss_func= lambda n:0.1):
        EpsilonGreedyAgent.__init__(self, reward_estimates, epsilon)
        self.ss_func = ss_func

    def update_estimates(self, reward, action):
        n = self.action_count[np.arange(self.num_experiments), action]

        error = reward - self.reward_estimates[np.arange(self.num_experiments),action]

        ss = self.ss_func(n)

        self.reward_estimates[np.arange(self.num_experiments),action] += ss * error