# -*- coding: utf-8 -*-
"""
Created on Mon Jun 14 13:45:52 2021

@author: bahramis
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Jun 13 15:32:50 2021

@author: bahramis
"""

import numpy as np
from CustomENV import ShowerEnv
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam

    
env = ShowerEnv(10,30,10,24)
env.observation_space.sample()

episodes = 10
for episode in range(1, episodes+1):
    state = env.reset()
    done = False
    score = 0 
    
    while not done:
        #env.render()
        action = env.action_space.sample()
        n_state, reward, done, info = env.step(action)
        print(reward)
        score+=reward
    print('Episode:{} Score:{}'.format(episode, score))




states = env.observation_space.shape
# actions = env.action_space.n
actions =env.action_space.n

def build_model(states, actions):
    model = Sequential()  
    model.add(Dense(24, activation='relu', input_shape=states))
    model.add(Flatten())
    model.add(Dense(24, activation='relu'))
    model.add(Dense(actions, activation='linear'))
    return model

# del model
model = build_model(states, actions)

model.summary()


from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory


def build_agent(model, actions):
    policy = BoltzmannQPolicy()
    memory = SequentialMemory(limit=300000, window_length=1)
    dqn = DQNAgent(model=model, memory=memory, policy=policy, 
                  nb_actions=actions, nb_steps_warmup=10, target_model_update=30000)
    return dqn

dqn = build_agent(model, actions)
dqn.compile(Adam(lr=1e-3), metrics=['mae'])
dqn.fit(env, nb_steps=10000, visualize=False, verbose=1)

scores = dqn.test(env, nb_episodes=100, visualize=False)
print(np.mean(scores.history['episode_reward']))

dqn.save_weights('dqn_weights.h5f', overwrite=True)

dqn.load_weights('dqn_weights.h5f')

scores = dqn.test(env, nb_episodes=10, visualize=False)
print(np.mean(scores.history['episode_reward']))

































