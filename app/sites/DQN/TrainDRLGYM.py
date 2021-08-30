import numpy as np
from sites.DQN.CustomENV import ShowerEnv
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory
import dill
from io import BytesIO
import base64


def build_model(states, actions):
    model = Sequential()
    model.add(Dense(50, activation='tanh', input_shape=states))
    model.add(Flatten())
    model.add(Dense(50, activation='tanh'))
    model.add(Dense(actions, activation='linear'))
    return model


def build_agent(model, actions):
    policy = BoltzmannQPolicy()
    memory = SequentialMemory(limit=50000, window_length=1)
    dqn = DQNAgent(model=model, memory=memory, policy=policy,
                   nb_actions=actions, nb_steps_warmup=10, target_model_update=30000)
    return dqn


def Base64Converter(ObjectFile):
    bytes_container = BytesIO()
    dill.dump(ObjectFile, bytes_container)
    bytes_container.seek(0)
    bytes_file = bytes_container.read()
    base64File = base64.b64encode(bytes_file)
    return base64File


def TrainDRLGYM(FromHour, ToHour, W, Desire):
    env = ShowerEnv(FromHour, ToHour, W, Desire)
    states = env.observation_space.shape
    actions = env.action_space.n
    model = build_model(states, actions)
    model.summary()

    dqn = build_agent(model, actions)
    dqn.compile(Adam(lr=2e-3), metrics=['mae'])
    dqn.fit(env, nb_steps=2000, visualize=False, verbose=1)

    scores = dqn.test(env, nb_episodes=1, visualize=False)
    print(np.mean(scores.history['episode_reward']))
    model_def = model.to_json()

    return dqn, model_def
