import numpy as np
from sites.DQN.CustomENV import ShowerEnv
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import model_from_json
from tensorflow.keras.layers import Dense, Flatten
from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory
from sites.DQN.TrainDRLGYM import TrainDRLGYM
from tensorflow.keras.optimizers import Adam
from core.models import TrainingResult
import io
import h5py
from tensorflow import keras
import dill
import base64
import tempfile
import os


def build_agent(model, actions):
    policy = BoltzmannQPolicy()
    memory = SequentialMemory(limit=300000, window_length=1)
    dqn = DQNAgent(model=model, memory=memory, policy=policy,
                   nb_actions=actions)
    return dqn


def build_model(states, actions):
    model = Sequential()
    model.add(Dense(50, activation='tanh', input_shape=states))
    model.add(Flatten())
    model.add(Dense(50, activation='tanh'))
    model.add(Dense(actions, activation='linear'))
    return model


def TestDRLGYM(FromHour, ToHour, W, Desire, device_id=1):
    env = ShowerEnv(FromHour, ToHour, W, Desire)
    states = env.observation_space.shape
    actions = env.action_space.n
    model = build_model(states, actions)
    dqn = build_agent(model, actions)

    dqn.compile(Adam(lr=2e-3), metrics=['mae'])
    dqn.load_weights('sites/DQN/dqn_weights.h5f')
    scores = dqn.test(env, nb_episodes=1, visualize=False)
    print(np.mean(scores.history['episode_reward']))


def LoadTrainedModel(FromHour, ToHour, W, Desire, device_id=1):
    obj = TrainingResult.objects.get(device_id=device_id)
    if not obj:
        return None
    env = ShowerEnv(FromHour, ToHour, W, Desire)
    states = env.observation_space.shape
    actions = env.action_space.n
    loaded_model_json = obj.model
    model = model_from_json(loaded_model_json)
    dqn = build_agent(model, actions)
    dqn.compile(Adam(lr=2e-3), metrics=['mae'])
    weights_filepath = 'weights.h5'
    f = open(weights_filepath, "wb")
    f.write(obj.weights_bin)
    f.close()
    dqn.load_weights(weights_filepath)
    os.remove(weights_filepath)
    return dqn


def ForwardDRLGYM(dqn, W, Sample):
    action = dqn.forward(Sample)
    print('action is', action)
    t = Sample[5]
    z = np.exp(-300/130)
    OutdoorTemp_now = Sample[2]
    People_now = Sample[4]
    Desire_now = Sample[1]
    Prev_IndTemp = Sample[0]
    Price = Sample[3]
    airTemp = 10+action
    if airTemp > Prev_IndTemp:
        IndoorTemp_new = Prev_IndTemp + \
            (OutdoorTemp_now-Prev_IndTemp)*z+(airTemp-Prev_IndTemp)*z
        Tset = min(Prev_IndTemp+3, 30)
    else:
        IndoorTemp_new = Prev_IndTemp+(OutdoorTemp_now-Prev_IndTemp)*z
        Tset = Prev_IndTemp/2

    # Calculate reward
    if airTemp > Prev_IndTemp:
        reward = -(Price*abs(airTemp)+People_now*W *
                   abs(float(Desire_now)-IndoorTemp_new))
    else:
        reward = -(People_now*W*abs(Desire_now-IndoorTemp_new))

    Cost = -reward

    return airTemp, IndoorTemp_new, Tset, Cost
# TrainDRLGYM(28,72,1,19)
# Sample=np.array([12 , 22 , 10 , 2.5 ,  2, 70 ])
# airTemp,indTemp,Tset, Cost=ForwardDRLGYM(50,90,1,24,Sample)
