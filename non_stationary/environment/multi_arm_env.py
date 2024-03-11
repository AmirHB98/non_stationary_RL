from gym import spaces
from gym.utils import seeding
import numpy as np
import gym

class ArmedBanditEnv(gym.Env):
    metadata = {'render.modes' : ['human']}

    def __init__(self,mean,stddev):
        assert len(mean.shape) == 2
        assert len(stddev.shape) == 2

        super(ArmedBanditEnv,self).__init__()
        
        self.num_bandits = mean.shape[1]
        self.num_expriments = mean.shape[0]

        self.action_space = spaces.Discrete(self.num_bandits)
        self.observation_space = spaces.Discrete(1)
        
        self.mean = mean
        self.stddev = stddev
    
    def step(self,action):
        assert (action< self.num_bandits).all()

        sampled_means = self.mean[np.arange(self.num_expriments),action]
        sampled_stddevs = self.stddev[np.arange(self.num_expriments),action]
        
        reward = np.random.normal(loc = sampled_means, scale = sampled_stddevs, \
                                  size = (self.num_expriments,))
        
        observation, done, info = 0, False, dict()
        return observation, reward, done, info
    
    def reset(self):
        return 0
    
    def render(self, mode, close = False):
        pass

    def _seed(self,seed = None):
        self.np_random, seed = seeding.np.random(seed)
        return [seed]
    
    def close(self):
        pass

if __name__=='__main__':
    # Example use  case
    # Create a casino with 4 bandits and 2 machines (experiments)
    means = np.array([[5,1,0,-10],[5,1,1,5]])
    stddevs = np.array([[1, .1, 5, 1],[.1,.1,1,2]])

    env = ArmedBanditEnv(means,stddevs)

    for i in range(means.shape[1]):
        action = np.ones(means.shape[0],dtype= np.int32) * i
        _,reward,_,_ = env.step(action)
        print(f'Bandit {i} gave reward of {reward[0]:.2f} in machine 1')
        print(f'Bandit {i} gave reward of {reward[1]:.2f} in machine 2')