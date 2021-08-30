import numpy as np
import datetime
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


def getDefaultModel(actions):
    json_file = open('sites/DQN/dqn_model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    return loaded_model_json


def LoadTrainedModel(FromHour, ToHour, W, Desire, device_id):
    obj, created = TrainingResult.objects.get_or_create(device_id=device_id)
    env = ShowerEnv(FromHour, ToHour, W, Desire)
    states = env.observation_space.shape
    actions = env.action_space.n
    if created:
        # Load default model if not previously trained
        obj.model = getDefaultModel(actions)
    loaded_model_json = obj.model
    print(loaded_model_json)
    model = model_from_json(loaded_model_json)
    dqn = build_agent(model, actions)
    dqn.compile(Adam(lr=2e-3), metrics=['mae'])

    # Load the default weights if creatd. Otherwise use the training result and asssign the trained weights
    weights_filepath = 'weights.h5'
    if created:
        weights_filepath = 'sites/DQN/dqn_weights.h5'
    else:
        f = open(weights_filepath, "wb")
        f.write(obj.weights_bin)
        f.close()
    dqn.load_weights(weights_filepath)
    if created:
        tmp_file = 'tmp_weights.h5'
        dqn.save_weights(tmp_file, overwrite=True)
        Bytes = b''
        with open(tmp_file, "rb") as f:
            while (byte := f.read(1)):
                Bytes += byte
        obj.weights_bin = Bytes
        obj.last_updated_at = datetime.datetime.now()
        obj.save()

        os.remove(tmp_file)
        assert not os.path.exists(tmp_file)
    else:
        os.remove(weights_filepath)
    return dqn


def ForwardDRLGYM(dqn, W, Sample):
    action = dqn.forward(Sample)
    print('action is', action)
    t = Sample[5]
    z = np.exp(-300/100)
    OutdoorTemp_now = Sample[2]
    People_now = Sample[4]
    Desire_now = Sample[1]
    Prev_IndTemp = Sample[0]
    Price = Sample[3]
    airTemp = 10+action
    if airTemp < Prev_IndTemp:
        IndoorTemp_new = Prev_IndTemp + \
            (OutdoorTemp_now-Prev_IndTemp)*z+(airTemp-Prev_IndTemp)*z
        Tset = max(Prev_IndTemp-5, 10)
    else:
        IndoorTemp_new = Prev_IndTemp+(OutdoorTemp_now-Prev_IndTemp)*z
        Tset = Prev_IndTemp*1.1

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
