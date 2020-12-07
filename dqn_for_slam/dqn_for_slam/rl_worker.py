import logging
import gym
import matplotlib.pyplot as plt
import datetime

import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import History

from rl.agents.dqn import DQNAgent
from rl.memory import SequentialMemory

import dqn_for_slam.environment
from dqn_for_slam.custom_policy import CustomEpsGreedy


ENV_NAME = 'RobotEnv-v0'

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def main():
    logger.info({
        'action': 'main',
        'status': 'run'
    })
    env = gym.make(ENV_NAME)
    nb_actions = env.action_space.n

    model = tf.keras.Sequential()
    model.add(Flatten(input_shape= (1, ) + env.observation_space.shape))
    model.add(Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
    model.add(Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
    model.add(Dense(nb_actions, activation='linear'))
    print(model.summary())

    memory = SequentialMemory(limit=62500, window_length=1)
    policy = CustomEpsGreedy(max_eps=0.6, min_eps=0.1, eps_decay=0.9997)
    history = History()

    agent = DQNAgent(
        nb_actions=nb_actions,
        model=model,
        memory=memory,
        policy=policy,
        gamma=0.99,
        batch_size=64)

    agent.compile(optimizer=Adam(lr=1e-3), metrics=['mae'])

    agent.fit(env,
              nb_steps=62500,
              visualize=False,
              nb_max_episode_steps=250,
              log_interval=250,
              verbose=1,
              callbacks=[history])

    dt_now = datetime.datetime.now()
    agent.save_weights('../models/dpg_{}_weights_{}{}{}.h5f'.format(ENV_NAME, dt_now.month, dt_now.day, dt_now.hour), overwrite=True)
    # agent.test(env, nb_episodes=5, visualize=False)

    try:
        fig = plt.figure()
        plt.subplot(2, 1, 1)
        plt.plot(history.history['nb_episode_step'])
        plt.ylabel('step')

        plt.subplot(2, 1, 2)
        plt.plot(history.history['episode_reward'])
        plt.xlabel('episode')
        plt.ylabel('reward')
        plt.show()

        fig.savefig('../figures/learning_results_{}{}{}.png'
                    .format(dt_now.month, dt_now.day, dt_now.hour))
    except (AttributeError, TypeError) as ex:
        logger.error({'type': ex,
                      'message': 'history object has a problem'})



if __name__ == '__main__':
    main()
