import logging
import gym
import matplotlib.pyplot as plt
import datetime
import os

import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers

from rl.agents.dqn import DQNAgent
from rl.memory import SequentialMemory

import dqn_for_slam.environment
from dqn_for_slam.custom_policy import CustomEpsGreedy

ENV_NAME = 'RobotEnv-v0'

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

file_path = __file__
dir_path = file_path[:(len(file_path) - len('rl_worker.py'))]
MODELS_PATH = dir_path + 'models/'   # model save directory
FIGURES_PATH = dir_path + 'figures/'


def kill_all_node() -> None:
    """
    kill all ros node except for roscore
    """
    nodes = os.popen('rosnode list').readlines()
    for i in range(len(nodes)):
        nodes[i] = nodes[i].replace('\n', '')

    for node in nodes:
        os.system('rosnode kill ' + node)


def main() -> None:
    env = gym.make(ENV_NAME)
    nb_actions = env.action_space.n

    model = tf.keras.Sequential()
    model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
    model.add(Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
    model.add(Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
    model.add(Dense(nb_actions, activation='linear'))
    print(model.summary())

    memory = SequentialMemory(limit=100000, window_length=1)
    policy = CustomEpsGreedy(max_eps=0.6, min_eps=0.1, eps_decay=0.9997)

    agent = DQNAgent(
        nb_actions=nb_actions,
        model=model,
        memory=memory,
        policy=policy,
        gamma=0.99,
        batch_size=64)

    agent.compile(optimizer=Adam(lr=1e-3), metrics=['mae'])

    history = agent.fit(env,
                        nb_steps=100000,
                        visualize=False,
                        nb_max_episode_steps=300,
                        log_interval=300,
                        verbose=1)

    kill_all_node()

    dt_now = datetime.datetime.now()
    agent.save_weights(
        MODELS_PATH + 'dpg_{}_weights_{}{}{}.h5f'.format(ENV_NAME, dt_now.month, dt_now.day, dt_now.hour),
        overwrite=True)
    # agent.test(env, nb_episodes=5, visualize=False)

    fig = plt.figure()
    plt.plot(history.history['episode_reward'])
    plt.xlabel("episode")
    plt.ylabel("reward")

    plt.savefig(FIGURES_PATH + 'learning_results_{}{}{}.png'
                .format(dt_now.month, dt_now.day, dt_now.hour))


if __name__ == '__main__':
    main()
