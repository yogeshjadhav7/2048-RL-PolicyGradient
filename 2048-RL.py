
# coding: utf-8

# In[1]:


from __future__ import print_function

import argparse
import math
import pickle
import random
import datetime
import numpy as np

import gym
import gym_2048

from policy_gradient import PolicyGradient
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")


# In[2]:


env = gym.make("2048-v0")
env.reset()

# Policy gradient has high variance, seed for reproducability
env.seed(1)

RENDER_ENV = False
EPISODES = 50000
EPISODE_WINDOW = EPISODES / 100
EPISODE_WINDOW_SHIFT = EPISODE_WINDOW

#Progress tracking metrics
rewards = []
max_tile_values = []
episode_score_card = [0.0 for x in range(15)]

QUIET = True

# Load checkpoint
load_path = "outputs/weights/2048-v0.ckpt"
save_path = "outputs/weights/2048-v0.ckpt"


# In[ ]:


if __name__ == "__main__":

    PG = PolicyGradient(
        n_x = env.observation_space.shape[0],
        n_y = env.action_space.n,
        learning_rate=0.025,
        reward_decay=0.95,
        epochs=3,
        load_path=load_path,
        save_path=save_path
    )
    
    PG.quiet = QUIET

    for episode in range(EPISODES):

        observation = env.reset()
        episode_reward = 0
        max_tile_value_so_far = 0

        while True:
            if RENDER_ENV: env.render()

            # Choose an action based on observation
            action = PG.choose_action(observation)
            
            valid_move = False

            while not valid_move:
                
                # Take action in the environment
                observation_, reward, done, info = env.step(action)
                
                if done: break
                
                # check for validity of move
                valid_move = info["valid"]
                
                if not valid_move:
                    # Get out of the invalid move loop by choosing remaining moves randomly
                    action = (action + 1 + np.random.randint(env.action_space.n - 1)) % env.action_space.n
                else:
                    reward = np.max(observation_) - np.max(observation)
        
                    # Store transition for training
                    PG.store_transition(observation, action, reward)

            if done:
                episode_rewards_sum = sum(PG.episode_rewards)
                max_tile_value = np.max(observation.flatten())
                
                rewards.append(episode_rewards_sum)
                max_tile_values.append(max_tile_value)
                
                max_reward_so_far = np.amax(rewards)
                max_tile_value_so_far = np.amax(max_tile_values)
                
                episode_score_card[np.int(np.log2(max_tile_value)) - 1] += 1
                
                if episode > 0 and episode % 10 == 0:
                    print("\n\nEpisode: ", episode)
                    for i in range(len(episode_score_card)):
                        if episode_score_card[i] > 0:
                            print(str(2 ** (i + 1)) + " : " + str(episode_score_card[i] / np.sum(episode_score_card)))
                    
                if not QUIET:
                    print("==========================================")
                    print("Max tile value: ", max_tile_value)
                    print("Reward: ", episode_rewards_sum)
                    print("Max tile value so far: ", max_tile_value_so_far)
                    print("Max reward so far: ", max_reward_so_far)

                # Train neural network
                discounted_episode_rewards_norm = PG.learn()
                
                break

            # Save new observation
            observation = observation_
        
        if episode > 0 and episode % EPISODE_WINDOW == 0:
            #PG.plot(y_data=rewards, y_label="Rewards", n_episode=episode, window=EPISODE_WINDOW, windowshift=EPISODE_WINDOW_SHIFT)
            PG.plot(y_data=max_tile_values, y_label="Max Tile", n_episode=episode, window=EPISODE_WINDOW, windowshift=EPISODE_WINDOW_SHIFT)
        
    PG.plot(y_data=rewards, y_label="Rewards", n_episode=EPISODES, window=EPISODE_WINDOW, windowshift=EPISODE_WINDOW_SHIFT)
    PG.plot(y_data=max_tile_values, y_label="Max Tile", n_episode=EPISODES, window=EPISODE_WINDOW, windowshift=EPISODE_WINDOW_SHIFT)
    

