from agents.greedy_agent import GreedyAgent
import numpy as np
from utils.functions import argmax
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from environment.multi_arm_env import ArmedBanditEnv

class EpsilonGreedyAgent(GreedyAgent):
    def __init__(self, reward_estimates, epsilon):
        GreedyAgent.__init__(self, reward_estimates)
        assert epsilon >= 0  and epsilon<= 1

        self.epsilon = epsilon

    def get_action(self):
        action_type = (np.random.random_sample(self.num_experiments) > self.epsilon).astype(int)

        exploratory_action = np.random.randint(self.num_bandits, size = self.num_experiments)
        self.greedy_action = argmax(self.reward_estimates)

        action = self.greedy_action * action_type + exploratory_action* (1-action_type)

        self.action_count[np.arange(self.num_experiments),action] += 1

        return action