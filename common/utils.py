import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib import rc
from tqdm import tqdm
import imageio
import io
import json

"""Utils functions"""


class RhoElem:
    MEAN = 0
    STD = 1


class LearnRates:
    RHO = 0
    LAMBDA = 1
    ETA = 2


class TrajectoryResults:
    PERF = 0
    RewList = 1
    ScoreList = 2
    StateList = 3
    Logprob_target = 4
    Logprob_behavioural = 5
    ActionList = 6


class ParamSamplerResults:
    THETA = 0
    PERF = 1


class SplitResults:
    Gradient = 0
    RewardTrajectories = 1
    SplitThetas = 2
    ValidTrajectories = 3


def evaluate_planning(env, policy, num_episodes=1, horizon=500, persistence=False, planning_horizon=1, dir=None):
    """
    Evaluate a RL agent
    :param env: (Env object) the Gym environment
    :param policy: (BasePolicy object) the policy in stable_baselines3
    :param gamma: (float) the discount factor
    :param num_episodes: (int) number of episodes to evaluate it
    :return: (float) Mean reward for the last num_episodes
    """
    all_episode_rewards = []
    for i in tqdm(range(num_episodes)): # iterate over the episodes
        episode_rewards = []
        done = False
        env.reset()
        frames = []
        for t in range(horizon): # iterate over the steps until termination
            obs = env.state
            action = policy.draw_action(obs)

            if persistence:
                # repeat the action for the planning horizon
                action = np.tile(action, planning_horizon).ravel()

            # reshape the action according to the planning horizon
            action = np.array(np.split(action, planning_horizon))

            seq_reward = .0
            for i, a in enumerate(action):
                # play the action
                obs, rew, done, _ = env.step(a)
                seq_reward += (env.gamma ** i) * rew
                if done:
                    break

            # update the performance index
            episode_rewards.append((env.gamma ** (t * planning_horizon)) * seq_reward)
            
            try:
                frames.append(env.render())
                # env.render()
            except:
                pass
            
            if done:
                break

        all_episode_rewards.append(sum(episode_rewards))

    mean_episode_reward = np.mean(all_episode_rewards)
    std_episode_reward = 0
    print("Mean reward:", mean_episode_reward,
          "Std reward:", std_episode_reward,
          "Num episodes:", num_episodes)
    
    save_results(all_episode_rewards, dir)
    return frames

def save_results(returns, dir) -> None:        
        pass
        # results = {
        #     "performance": np.array(returns, dtype=float).tolist(),
        # }
        # # Save the json
        # name = self.directory + "/results.json"
        # with io.open(name, 'w', encoding='utf-8') as f:
        #     f.write(json.dumps(results, ensure_ascii=False, indent=4))
        #     f.close()
        # return

def animate(data, interval=200):
  fig = plt.figure(1)
  img = plt.imshow(data[0][0])
  plt.axis('off')

  def update_frame(i):
    img.set_data(data[i][0])

  anim = animation.FuncAnimation(fig, update_frame, frames=len(data), interval=interval)
  plt.show()
  plt.close(1)
  return anim