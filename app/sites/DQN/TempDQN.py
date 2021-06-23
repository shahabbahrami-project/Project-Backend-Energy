# -*- coding: utf-8 -*-
"""
Created on Sun Jun 13 15:32:50 2021

@author: bahramis
"""
from UsefulFunc import OutdoorTemp, People, Price, Desire
from gym import Env
from gym.spaces import Discrete, Box, Tuple
import numpy as np
from numpy.random import randint

class ShowerEnv(Env):

        

    def __init__(self):
        # Actions we can take, down, stay, up
        self.action_space = (Box(low=np.array([10, 0]), high=np.array([30,30])))
        # Temperature array
        self.FromHour=10
        self.ToHour=48
        self.W=1
        self.desire=24
        self.observation_space = Box(low=0, high=30, shape=(5,1))
        # Set start temp
        self.state = np.array([12, Desire(0,self.desire,self.FromHour,self.ToHour),OutdoorTemp(0),Price(0),People(0,self.FromHour,self.ToHour)]) 
        # Set time slots in a day
        self.day_length = 96


    def step(self, action):
        # Apply action
        # 0 -1 = -1 temperature
        # 1 -1 = 0 
        # 2 -1 = 1 temperature
        time=96-self.day_length
        z=np.exp(-300/130)
        OutdoorTemp_now=OutdoorTemp(time)
        People_now=People(time,self.FromHour,self.ToHour)
        Desire_now=Desire(time,self.desire,self.FromHour,self.ToHour)
        Prev_IndTemp=self.state[0]
        if action[1]>self.state[0]:
            self.state[0]= self.state[0]+(OutdoorTemp_now-self.state[0])*z+(action[0]-self.state[0])*z
        else:
            self.state[0]= self.state[0]+(OutdoorTemp_now-self.state[0])*z
        self.state[1]= Desire_now
        self.state[2]= OutdoorTemp_now
        self.state[3]=Price(time)
        self.state[4]=People_now
        
        # Reduce shower length by 1 second
        self.day_length -= 1 
        
        # Calculate reward
        if action[1]>Prev_IndTemp: 
            reward = -(Price(time)*np.absolute(action[0])+People_now*self.W*np.absolute(Desire_now-self.state[0]))
        else: 
            reward = -(People_now*self.W*np.absolute(Desire_now-self.state[0])) 
        
        # Check if shower is done
        if self.day_length <= 0: 
            done = True
        else:
            done = False
        
        # Apply temperature noise
        #self.state += random.randint(-1,1)
        # Set placeholder for info
        info = {}
        
        # Return step information
        return self.state, reward, done, info

    def render(self):
        # Implement viz
        pass
    
    def reset(self):
        # Reset shower temperature
        self.state = np.array([12, Desire(0,self.desire,self.FromHour,self.ToHour),OutdoorTemp(0),Price(0),People(0,self.FromHour,self.ToHour)]).reshape(1,5)
        # Reset shower time
        self.day_length = 96
        return self.state
    
env = ShowerEnv()


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


import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam

states = env.observation_space.shape
# actions = env.action_space.n
actions =env.action_space.shape[0]

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
    memory = SequentialMemory(limit=50000, window_length=1)
    dqn = DQNAgent(model=model, memory=memory, policy=policy, 
                  nb_actions=actions, nb_steps_warmup=10, target_model_update=1e-2)
    return dqn

dqn = build_agent(model, actions)
dqn.compile(Adam(lr=1e-3), metrics=['mae'])
dqn.fit(env, nb_steps=2000, visualize=False, verbose=1)

scores = dqn.test(env, nb_episodes=100, visualize=False)
print(np.mean(scores.history['episode_reward']))

dqn.save_weights('dqn_weights.h5f', overwrite=True)

dqn.load_weights('dqn_weights.h5f')

scores = dqn.test(env, nb_episodes=10, visualize=False)
print(np.mean(scores.history['episode_reward']))

































