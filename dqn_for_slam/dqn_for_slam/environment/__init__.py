from gym.envs.registration import register

register(
    id='RobotEnv-v0',
    #directory:class name
    entry_point='dqn_for_slam.environment.robot_rl_env:RobotEnv'
)
