from typing import Tuple
import logging
import time

import gym
from gym import spaces
import numpy as np
import rospy
from nav_msgs.msg import Odometry, OccupancyGrid
from geometry_msgs.msg import Twist, Point, Quaternion
from gazebo_msgs.srv import SetModelState
from gazebo_msgs.msg import ModelState
from sensor_msgs.msg import LaserScan

from gmapping.srv import SlamCmd

INITIAL_POS_X = 0.0
INITIAL_POS_Y = 0.0

LIDAR_SCAN_MAX_DISTANCE = 4.0
TRAINING_IMAGE_SIZE = 360

MAX_PX = 5.
MAX_PY = 5.
MAX_QW = 1.
MAX_QX = 1.
MAX_QY = 1.
MAX_QZ = 1.
MAX_ACTION_NUM = 5
MAX_STEPS = 500000
MAX_MAP_COMPLETENESS = 100.

MIN_PX = -5
MIN_PY = -5
MIN_ACTION_NUM = 0
MIN_STEPS = 0
MIN_MAP_COMPLETENESS = 0.

MAP_SIZE_RATIO = 0.27
MAX_STEPS_IN_EPS = 100
MAP_COMPLETENESS_THRESHOLD = 0.65
COLLISION_THRESHOLD = 0.2

REWARD_MAP_COMPLETED = 1.
REWARD_CRASHED = -1.

SLEEP_BETWEEN_ACTION_AND_REWARD_CALCULATION = 0.3
SLEEP_AFTER_RESET_TIME = 1

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class RobotEnv(gym.Env):
    """
    Environment for reinforce learning
    -> generally it defines
        action
        status that responds to the action
    -> also it defines
        reward for a status
    Each valuables for the state is updated by ros
    it gets information of state and run somethings with them
    -> it don't define functions like "can_move_at" or "transit" or so
    -> it defines functions like "reward"
    """

    def __init__(self) -> None:

        self.position = Point(INITIAL_POS_X, INITIAL_POS_Y, 0)
        self.orientation = Quaternion(0, 0, 0, 0)
        self.ranges = None
        self.map_completeness_pct = MIN_MAP_COMPLETENESS
        self.occupancy_grid = None
        self.done = False
        self.steps = 0
        self.steps_in_episode = 0
        self.min_distance = 100
        self.reward = None
        self.reward_in_episode = 0
        self.last_map_completeness_pct = 0
        self.next_state = None

        # define action space
        # steering(angle) is (-1, 1), throttle(speed) is (0, 1)
        # self.action_space = spaces.Box(low=np.array([-1, 0]), high=np.array([+1, +1]), dtype=np.float32)
        self.action_space = spaces.Discrete(4)

        # define observation space
        scan_high = np.array([LIDAR_SCAN_MAX_DISTANCE] * TRAINING_IMAGE_SIZE)
        scan_low = np.array([0.0] * TRAINING_IMAGE_SIZE)
        num_high = np.array([
            MAX_PX,
            MAX_PY,
            MAX_QX,
            MAX_QY,
            MAX_QZ,
            MAX_QW,
            MAX_STEPS,
            MAX_MAP_COMPLETENESS
        ])
        num_low = np.array([
            MIN_PX,
            MIN_PY,
            -1 * MAX_QX,
            -1 * MAX_QY,
            -1 * MAX_QZ,
            -1 * MAX_QW,
            MIN_STEPS,
            MIN_MAP_COMPLETENESS
        ])
        high = np.concatenate([scan_high, num_high])
        low = np.concatenate([scan_low, num_low])
        self.observation_space = spaces.Box(low, high, dtype=np.float32)

        # ROS initialization
        self.ack_publisher = rospy.Publisher('/cmd_vel', Twist, queue_size=100)
        self.gazebo_reset_service = rospy.ServiceProxy('/slam_cmd_srv', SlamCmd)
        self.gazebo_model_state_service = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        rospy.init_node('rl_dqn', anonymous=True)

        # Subscribe to ROS topics and register callbacks
        rospy.Subscriber('/odom', Odometry, self._callback_pose)
        rospy.Subscriber('/scan', LaserScan, self._callback_scan)
        rospy.Subscriber('/map', OccupancyGrid, self._callback_map)

    def reset(self) -> np.ndarray:
        """
        initiate status and  return the first observed values
        """
        rospy.loginfo('start resetting')
        self.done = False
        self.position = Point(INITIAL_POS_X, INITIAL_POS_Y, 0)
        self.orientation = Quaternion(0, 0, 0, 0)
        self.steps_in_episode = 0
        self.map_completeness_pct = 0
        self.last_map_completeness_pct = 0
        self.reward_in_episode = 0
        self.next_state = None
        self.ranges = None

        self._send_action(0, 0)  # stop robot moving
        self._turtlebot3_reset()  # initialize gazebo robot status

        # clear map
        rospy.wait_for_service('/slam_cmd_srv')
        if self.gazebo_reset_service(1):
            rospy.loginfo('map reset')
        else:
            rospy.logerr('map cannot be reset')

        time.sleep(SLEEP_AFTER_RESET_TIME)  # wait map become clear

        self._update_map_completeness()
        self._update_state()
        self._infer_reward()

        rospy.loginfo('end resetting')
        # TODO (Kuwabara): add process when self.next_stage is None
        return self.next_state

    def _turtlebot3_reset(self) -> None:
        """
        """
        rospy.wait_for_service('gazebo/set_model_state')

        model_state = ModelState()
        model_state.model_name = 'turtlebot3_burger'
        model_state.pose.position.x = INITIAL_POS_X
        model_state.pose.position.y = INITIAL_POS_Y
        model_state.pose.position.z = 0
        model_state.pose.orientation.x = 0
        model_state.pose.orientation.y = 0
        model_state.pose.orientation.z = 0
        model_state.pose.orientation.w = 0
        model_state.twist.linear.x = 0
        model_state.twist.linear.y = 0
        model_state.twist.linear.z = 0
        model_state.twist.angular.x = 0
        model_state.twist.angular.y = 0
        model_state.twist.angular.z = 0

        self.gazebo_model_state_service(model_state)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """
        run action and return results
        """
        if action == 0:  # turn left
            steering = 1.159
            throttle = 0.08
        elif action == 1:  # turn right
            steering = -1.159
            throttle = 0.08
        elif action == 2:  # straight
            steering = 0
            throttle = 0.1
        elif action == 3:  # backward
            steering = 0
            throttle = -0.1
        else:
            raise ValueError("Invalid action")

        # initialize rewards, next_state, done
        self.reward = None
        self.done = False
        self.next_state = None

        self.steps += 1
        self.steps_in_episode += 1
        self._send_action(steering, throttle)
        time.sleep(SLEEP_BETWEEN_ACTION_AND_REWARD_CALCULATION)
        self._update_map_completeness()
        self._update_state()
        self._infer_reward()

        # TODO (Kuwabara): add process when self.next_stage or self.reward is None

        info = {}
        return self.next_state, self.reward, self.done, info

    def _callback_pose(self, data: Odometry) -> None:
        self.position = data.pose.pose.position
        self.orientation = data.pose.pose.orientation

    def _callback_scan(self, data: LaserScan) -> None:
        self.ranges = data.ranges

    def _callback_map(self, data: OccupancyGrid) -> None:
        self.occupancy_grid = data.data

    def _update_state(self) -> None:
        """
        """

        # adapt number of sensor information to TRAINING_IMAGE_SIZE
        size = len(self.ranges)
        x = np.linspace(0, size - 1, TRAINING_IMAGE_SIZE)
        xp = np.arange(size)
        sensor_state = np.clip(np.interp(x, xp, self.ranges), 0, LIDAR_SCAN_MAX_DISTANCE)
        sensor_state[np.isnan(sensor_state)] = LIDAR_SCAN_MAX_DISTANCE

        # update distance to obstacles
        self.min_distance = np.amin(sensor_state)

        numeric_state = np.array([
            self.position.x,
            self.position.y,
            self.orientation.x,
            self.orientation.y,
            self.orientation.z,
            self.orientation.w,
            self.steps,
            self.map_completeness_pct
        ])

        self.next_state = np.concatenate([sensor_state, numeric_state])

    def _infer_reward(self) -> None:
        """
        """

        if self.map_completeness_pct > MAP_COMPLETENESS_THRESHOLD:
            self.reward = REWARD_MAP_COMPLETED

        elif self.min_distance < COLLISION_THRESHOLD:
            # Robot likely hit the wall
            self.reward = REWARD_CRASHED

        else:
            self.reward = self.map_completeness_pct - self.last_map_completeness_pct

        self.reward_in_episode += self.reward

    def _update_map_completeness(self) -> None:
        while not self.occupancy_grid:
            time.sleep(0.1)

        sum_grid = len(self.occupancy_grid)
        num_occupied = 0
        num_unoccupied = 0
        for n in self.occupancy_grid:
            if n == 0:
                num_unoccupied += 1
            elif n == 100:
                num_occupied += 1

        self.last_map_completeness_pct = self.map_completeness_pct
        self.map_completeness_pct = ((num_occupied + num_unoccupied) / sum_grid) / MAP_SIZE_RATIO

    def _send_action(self, steering: float, throttle: float) -> None:
        speed = Twist()
        speed.angular.z = steering
        speed.linear.x = throttle
        self.ack_publisher.publish(speed)

    def render(self, mode='human') -> None:
        """
        unused function
        """
        return
