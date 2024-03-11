from gym import spaces
from gym.utils import seeding
import numpy as np
import gym
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class NonstationaryArmedBanditsEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    def __init__(self, mean, stddev, nonstatic_amount = 0.1):
        assert len(mean.shape) == 2
        assert len(stddev.shape) == 2

        super(NonstationaryArmedBanditsEnv,self).__init__()

        self.num_bandits = mean.shape[1]
        self.num_experiments = mean.shape[0]

        self.action_space = spaces.Discrete(self.num_bandits)
        self.nonstatic_amount = nonstatic_amount

        self.observation_space = spaces.Discrete(1)
        self.mean = mean.astype(np.float64)
        self.stddev = stddev.astype(np.float64)

    def step(self,action):
        assert (action < self.num_bandits).all()

        sampled_means = self.mean[np.arange(self.num_experiments),action]
        sampled_stddevs = self.stddev[np.arange(self.num_experiments),action]

        reward = np.random.normal(loc = sampled_means, scale= sampled_stddevs, size = (self.num_experiments,))
        
        # Non-stationary modification
        self.mean += self.nonstatic_amount * np.random.normal(size=self.mean.shape)

        observation , done , info = 0, False, dict()

        return observation, reward, done, info

    def reset(self):
        return 0
    
    def render(self, mode = 'human', close=False):
        pass

    def _seed(self, seed = None):
        self.np.random, seed = seeding.np.random(seed)
        return [seed]
    
    def close(self):
        pass

if __name__ == '__main__':
    num_experiments = 2
    num_bandits = 8
    num_steps = 1e2

    means = np.random.normal(size=(num_experiments,num_bandits))
    stddevs = np.ones((num_experiments,num_bandits))
    action = np.array([0,0]) # Don't care about action!

    env = NonstationaryArmedBanditsEnv(means,stddevs)

    fig, axs = plt.subplots(num_experiments, figsize = (10,4))
    fig.suptitle('Estimated Values vs. Real values', fontsize=15)
    action_set = np.arange(num_bandits)

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
            values = ax.plot(action_set, env.mean[i], marker='D', linestyle='', alpha=0.8, color='r', label='Real Values')
            ax.axhline(0, color='black', lw=1)

            _, reward, _, _ = env.step(action)
        
            # Plot the estimated values from the agent compared to the real values
            artist.append(values)
        return artist
    
    anim = FuncAnimation(fig, func=step, frames=np.arange(num_steps), interval=10, repeat=False)
    plt.show()