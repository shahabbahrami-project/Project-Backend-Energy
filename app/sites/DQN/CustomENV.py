# -*- coding: utf-8 -*-
"""
Created on Mon Jun 14 17:30:27 2021

@author: shaha
"""

from sites.DQN.UsefulFunc import OutdoorTemp, OutdoorTemp2,OutdoorTemp3,DailyTempstatistics,People, Price, Desire
from gym import Env
from gym.spaces import Discrete, Box, Tuple
import numpy as np
from numpy.random import randint

class ShowerEnv(Env):

    def __init__(self, FromHour,ToHour, W, desire):
        # Actions we can take, down, stay, up
        self.action_space = Discrete(20)
        # Temperature array
        self.FromHour=FromHour
        self.ToHour=ToHour
        self.W=W
        self.desire=desire
        self.observation_space = Box(low=0, high=30, shape=(1,6), dtype=np.float32)
        toutavg,toutstd=DailyTempstatistics()
        self.ToutAvg=toutavg
        self.ToutStd=toutstd
        # Set start temp
        self.state = np.array([12, Desire(0,self.desire,self.FromHour,self.ToHour),OutdoorTemp3(self.ToutAvg[0],self.ToutStd[0]),Price(0),People(0,self.FromHour,self.ToHour),0])
        # Set time slots in a day
        self.day_length = 96



    def step(self, action):
        # Apply action
        # 0 -1 = -1 temperature
        # 1 -1 = 0
        # 2 -1 = 1 temperature
        time=96-self.day_length
        z=np.exp(-300/130)
        OutdoorTemp_now=OutdoorTemp3(self.ToutAvg[time],self.ToutStd[time])
        People_now=People(time,self.FromHour,self.ToHour)
        Desire_now=Desire(time,self.desire,self.FromHour,self.ToHour)
        Prev_IndTemp=self.state[0]
        airTemp=10+action
        if airTemp>self.state[0]:
            self.state[0]= self.state[0]+(OutdoorTemp_now-self.state[0])*z+(airTemp-self.state[0])*z
        else:
            self.state[0]= self.state[0]+(OutdoorTemp_now-self.state[0])*z

        self.state[1]= Desire_now
        self.state[2]= OutdoorTemp_now
        self.state[3]=Price(time)
        self.state[4]=People_now
        self.state[5]=time

        # Reduce shower length by 1 second
        self.day_length -= 1
        # Calculate reward
        if airTemp>Prev_IndTemp:
            reward = -(Price(time)*abs(float(airTemp))+People_now*self.W*abs(float(Desire_now)-self.state[0]))
        else:
            reward = -(People_now*self.W*abs(float(Desire_now)-self.state[0]))

        # Check if shower is done
        if self.day_length <= 0:
            done = True
        else:
            done = False

        # Apply temperature noise
        #self.state += random.randint(-1,1)
        # Set placeholder for info
        info = {}
        # print(self.state[0])
        # Return step information
        return self.state, reward, done, info

    def render(self):
        # Implement viz
        pass

    def reset(self):
        # Reset shower temperature

        self.state = np.array([12, Desire(0,self.desire,self.FromHour,self.ToHour),OutdoorTemp3(self.ToutAvg[0],self.ToutStd[0]),Price(0),People(0,self.FromHour,self.ToHour),0])
        # Reset shower time
        self.day_length = 96
        return self.state
