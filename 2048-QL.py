
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

from q_learning import QLearning
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")


# In[2]:


env = gym.make("2048-v0")
env.reset()
env.seed(1)

RENDER_ENV = False
TRAINING_ON = True
EPISODES = 100000
MODEL_PATH = "outputs/keras-models/2048_model.h5"

board_size = int(math.sqrt(env.observation_space.shape[0]))
n_output = env.action_space.n


# In[3]:


QL = QLearning (
    n_x=board_size,
    n_y=n_output,
    save_path = MODEL_PATH,
    total_episodes=EPISODES,
    restore_model=True,
    is_training_on=TRAINING_ON
)


# In[ ]:


for episode in range(EPISODES):
    observation = env.reset()
    
    QL.curr_episode = episode
    
    while True:
        if RENDER_ENV: env.render()
            
        valid_move = False    
        action = None
        
        while not valid_move:
            
            # Choose an action based on observation
            if action == None: action = QL.choose_action(observation)
            
            observation_, reward, done, info = env.step(action)
            valid_move = info['valid']
            
            reward = QL.calculate_reward(valid_move, done, reward, observation_)
        
            QL.save_experience(observation=observation, action=action, 
                               reward=reward, observation_=observation_, is_game_over=done, is_move_valid=valid_move)
            
            action = (action + 1) % QL.n_y
        
        features, labels = QL.sample_from_experience()
        QL.train_model(features=features, labels=labels)
        
        if done:
            highest_tile_value = QL.get_highest_tile_value(observation_)
            QL.episodic_highest_tiles_track.append(highest_tile_value)
            print("Episode #", (episode + 1), " : Highest Tile: ", highest_tile_value)
            env.render()
            QL.plot_progress(y_data=QL.episodic_highest_tiles_track, y_label="Highest Tile Value", n_episode=episode)
            break
        

