import numpy as np
from non_stationary.utils.functions import argmax
from non_stationary.environment.multi_arm_env import ArmedBanditEnv
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os

class GreedyAgent():
    def __init__(self,reward_estimates):
        assert len(reward_estimates.shape) == 2

        self.num_bandits = reward_estimates.shape[1]
        self.num_experiments = reward_estimates.shape[0]

        self.reward_estimates = reward_estimates.astype(np.float64)
        self.action_count = np.zeros(reward_estimates.shape)

    def get_action(self):
        
        action = argmax(self.reward_estimates)

        self.action_count[np.arange(self.num_experiments),action] += 1

        return action
    
    def update_estimates(self,reward,action):
        n = self.action_count[np.arange(self.num_experiments),action]
        error = reward - self.reward_estimates[np.arange(self.num_experiments),action]
        self.reward_estimates[np.arange(self.num_experiments),action] += (1/n)*error
    
if __name__ == '__main__':
    # Let's expriment with our agent
    print(os.getcwd())
    num_experiments = 2
    num_bandits = 8
    num_steps = 1e3

    means = np.random.normal(size= (num_experiments,num_bandits))
    stdev = np.ones((num_experiments,num_bandits))
    env = ArmedBanditEnv(means,stdev)

    agent = GreedyAgent(np.zeros((num_experiments,num_bandits)))

    fig, axs = plt.subplots(num_experiments)
    x_pos = np.arange(num_bandits)

    # Implement a step, which involves the agent acting upon the
    # environment and learning from the received reward.
    def step(g):
        artist = []
        for i in range(num_experiments):
            ax = axs[i]
            ax.clear()
            ax.set_ylim(-4, 4)
            ax.set_xlim(-0.5, num_bandits-.5)
            if i == num_experiments-1:
                ax.set_xlabel('Actions', fontsize=14)
            ax.set_ylabel('Value', fontsize=14)
            if i == 0:
                ax.set_title(label='Estimated Values vs. Real values', fontsize=15)
                ax.annotate(f'Step {g+1}:', xy = (-.1,1.1), xycoords = 'axes fraction')
            ax.plot(x_pos, env.mean[i], marker='D', linestyle='', alpha=0.8, color='r', label='Real Values')
            ax.axhline(0, color='black', lw=1)
        
            action = agent.get_action()
            _, reward, _, _ = env.step(action)
            agent.update_estimates(reward, action)
        
            # Plot the estimated values from the agent compared to the real values
            estimates = agent.reward_estimates[i]
            values = ax.bar(x_pos, estimates, align='center', color='blue', alpha=0.4, label='Estimated Values')
            ax.legend()
            artist.append(values)
        if g%100 == 0:
            print(f'remaining steps: {num_steps-g}')
        return artist

    anim = FuncAnimation(fig, func=step, frames=np.arange(num_steps), interval=10, repeat=False)
    # plt.show()
    anim.save('./greedy_agent.gif', writer = 'imagemagick', fps = 60)
