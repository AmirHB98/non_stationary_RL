import numpy as np
from environment.multi_arm_ns import NonstationaryArmedBanditsEnv  
from agents.css_eps_greedy import CssEpsilonGreedyAgent
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

if __name__ =='__main__':
    num_experiments = 2
    num_bandits = 8
    num_steps = int(1e3)
    epsilon = .15

    means = np.random.normal(size= (num_experiments,num_bandits))
    stdev = np.ones((num_experiments,num_bandits))

    env = NonstationaryArmedBanditsEnv(means,stdev)
    agent = CssEpsilonGreedyAgent(np.zeros((num_experiments,num_bandits)), epsilon)

    fig, axs = plt.subplots(num_experiments, figsize = (10,4))
    fig.suptitle('Estimated Values vs. Real values', fontsize=15)
    action_set = np.arange(num_bandits)

    def step(g):
        artist = []
        for i in range(num_experiments):
            ax = axs[i]
            ax.clear()
            ax.set_ylim(-5, 5)
            ax.set_xlim(-0.5, num_bandits-.5)
            if i == num_experiments-1:
                ax.set_xlabel('Actions', fontsize=14)
            if i == 0:
                ax.annotate(f'Step : {g}', xy = (-0.1,1.1), xycoords = 'axes fraction')
            ax.set_ylabel('Value', fontsize=14)
           
            ax.set_title(label=f'Machine {i}', fontsize=10)
            ax.plot(action_set, env.mean[i], marker='D', linestyle='', alpha=0.8, color='r', label='Real Values')
            ax.axhline(0, color='black', lw=1)
        
            action = agent.get_action()
            ax.plot(agent.greedy_action[i],env.mean[i,agent.greedy_action[i]], marker = 'D', linestyle ='', alpha = 0.8, color = 'orange', label = 'Exploiting Action')
            _, reward, _, _ = env.step(action)
            agent.update_estimates(reward, action)

           

            # Plot the estimated values from the agent compared to the real values
            estimates = agent.reward_estimates[i]
            values = ax.bar(action_set, estimates, align='center', color='blue', alpha=0.4, label='Estimated Values')
            ax.legend(ncols = 3)
            artist.append(values.patches)
        if g%100 == 0:
            print(f'remaining steps {num_steps-g}')

        return artist
    
    anim = FuncAnimation(fig, func=step, frames=np.arange(num_steps), interval=10, repeat=False)
    # plt.show()
    anim.save('./ns_eps_greedy_agent.gif', writer = 'imagemagick', fps = 60)

    
