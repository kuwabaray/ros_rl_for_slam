
from typing import Tuple
import logging
import time
import os

import gym
from gym import spaces
import numpy as np
import rospy
import rosparam
import tf2_ros
from nav_msgs.srv import GetMap
from nav_msgs.msg import Odometry, OccupancyGrid, MapMetaData
from geometry_msgs.msg import Twist, Point, Quaternion, PoseWithCovarianceStamped
from gazebo_msgs.srv import SetModelState
from gazebo_msgs.msg import ModelState
from sensor_msgs.msg import LaserScan
from std_srvs.srv import Empty
from robot_localization.srv import SetPose


file_path = __file__
dir_path = file_path[:(len(file_path) - len('environment/robot_rl_env.py'))] + 'config/'
config_file_name = 'rlslam_map_reward.yaml'
config_file_path = os.path.join(dir_path, config_file_name)
parameters_list=rosparam.load_file(config_file_path)
for params, namespace in parameters_list:
	rosparam.upload_params(namespace,params)

INITIAL_POS_X = rospy.get_param('rlslam/initial_posx')
INITIAL_POS_Y = rospy.get_param('rlslam/initial_posy')

LIDAR_SCAN_MAX_DISTANCE = rospy.get_param('rlslam/scan_max_distance')
LIDAR_SCAN_MIN_DISTANCE = rospy.get_param('rlslam/scan_min_distance')
TRAINING_IMAGE_SIZE = rospy.get_param('rlslam/training_image_size')
MAZE_SIZE = rospy.get_param('rlslam/maze_size')
MAP_COMPLETENESS_THRESHOLD = rospy.get_param('rlslam/map_completed_threshold')
COLLISION_THRESHOLD = rospy.get_param('rlslam/crash_distance')

REWARD_MAP_COMPLETED = rospy.get_param('rlslam/reward_map_completed')
REWARD_CRASHED = rospy.get_param('rlslam/reward_crashed')

MAX_PX = rospy.get_param('rlslam/obs_space_max/px')
MAX_PY = rospy.get_param('rlslam/obs_space_max/py')
MAX_QZ = rospy.get_param('rlslam/obs_space_max/qz')
MAX_ACTION_NUM = 3
MAX_MAP_COMPLETENESS = 100.
MAX_STEPS = rospy.get_param('rlslam/steps_in_episode')

MIN_PX = rospy.get_param('rlslam/obs_space_min/px')
MIN_PY = rospy.get_param('rlslam/obs_space_min/py')
MIN_QZ = rospy.get_param('rlslam/obs_space_min/qz')
MIN_ACTION_NUM = -1
MIN_STEPS = 0
MIN_MAP_COMPLETENESS = 0.

MAP_SIZE = (MAX_PX - MIN_PX) * (MAX_PY - MIN_PY)

STEERING = rospy.get_param('rlslam/steering')
THROTTLE = rospy.get_param('rlslam/throttle') 

TIMEOUT = rospy.get_param('rlslam/timeout')
SLEEP_RESET_TIME = rospy.get_param('rlslam/sleep_reset_time')


class RobotEnv(gym.Env):
    """
    Environment for reinforce learning
    """

    def __init__(self) -> None:

        rospy.init_node('rl_dqn', anonymous=True)

        self.position = Point(INITIAL_POS_X, INITIAL_POS_Y, 0)
        self.orientation = Quaternion(1, 0, 0, 0)
        self.ranges = None
        self.map_completeness_pct = MIN_MAP_COMPLETENESS
        self.occupancy_grid = None
        self.done = False
        self.steps_in_episode = 0
        self.min_distance = 100
        self.reward = None
        self.reward_in_episode = 0
        self.now_action = -1
        self.last_action = -1
        self.last_map_completeness_pct = 0
        self.map_size_ratio = MAZE_SIZE/MAP_SIZE
        self.steps_in_episode = 0
        # define action space
        # steering(angle) is (-1, 1), throttle(speed) is (0, 1)
        # self.action_space = spaces.Box(low=np.array([-1, 0]), high=np.array([+1, +1]), dtype=np.float32)
        self.action_space = spaces.Discrete(4)

        # define observation space
        scan_high = np.array([LIDAR_SCAN_MAX_DISTANCE] * TRAINING_IMAGE_SIZE)
        scan_low = np.array([LIDAR_SCAN_MIN_DISTANCE] * TRAINING_IMAGE_SIZE)
        num_high = np.array([
            MAX_PX,
            MAX_PY,
            MAX_QZ,
            MAX_ACTION_NUM,
            MAX_STEPS,
            MAX_MAP_COMPLETENESS
        ])
        num_low = np.array([
            MIN_PX,
            MIN_PY,
            MIN_QZ,
            MIN_ACTION_NUM, 
            MIN_STEPS,
            MIN_MAP_COMPLETENESS
        ])
        high = np.concatenate([scan_high, num_high])
        low = np.concatenate([scan_low, num_low])
        self.observation_space = spaces.Box(low, high, dtype=np.float32)

        # ROS initialization
        self.ack_publisher = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        self.map_reset_service = rospy.ServiceProxy('/clear_map', Empty)
        self.gazebo_model_state_service = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        self.unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        self.pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self.reset_odom_to_base = rospy.ServiceProxy('/set_pose', SetPose)
        self.tfBuffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tfBuffer)

    def reset(self) -> np.ndarray:
        """
        initiate status and  return the first observed values
        """
        rospy.loginfo('start resetting')

        self.done = False
        self.position = Point(INITIAL_POS_X, INITIAL_POS_Y, 0)
        self.orientation = Quaternion(1, 0, 0, 0)
        self.steps_in_episode = 0
        self.map_completeness_pct = 0
        self.last_map_completeness_pct = 0
        self.reward_in_episode = 0
        self.occupancy_grid = None
        self.ranges = None

        #rospy.wait_for_service('/gazebo/unpause_physics')
        #try:
        #    self.unpause()
        #except (rospy.ServiceException) as e:
        #    rospy.loginfo("/gazebo/unpause_physics service call failed")

        self._send_action(0, 0)
        self._reset_rosbot()
        self._reset_tf()

        # clear map
        rospy.wait_for_service('/clear_map')
        if self.map_reset_service():
            rospy.loginfo('reset map')
        else:
            rospy.logerr('could not reset map')
       
        #time.sleep(SLEEP_RESET_TIME)

        self._update_map_size_ratio() # sometimes map expands

        sensor_state = self._update_scan()
        self._update_map_completeness()
        numeric_state = self._update_odom()

        #rospy.wait_for_service('/gazebo/pause_physics')
        #try:
        #    self.pause()
        #except (rospy.ServiceException) as e:
        #    rospy.loginfo("/gazebo/pause_physics service call failed")

        next_state = np.concatenate([sensor_state, numeric_state])
        self._infer_reward()
        # TODO (Kuwabara): add process when self.next_stage is None
        return next_state

    def _reset_tf(self) -> None:
        rospy.wait_for_service('/set_pose')
        tf_pose = PoseWithCovarianceStamped()
        tf_pose.pose.pose.position.x = self.position.x
        tf_pose.pose.pose.position.y = self.position.y
        tf_pose.pose.pose.orientation.w = self.orientation.w
        if self.reset_odom_to_base(tf_pose):
            rospy.loginfo('initialized tf')
        else:
            rospy.logerr('/set_pose service call failed')

    def _reset_rosbot(self) -> None:
        """
        """
        rospy.wait_for_service('gazebo/set_model_state')

        model_state = ModelState()
        model_state.model_name = 'rosbot'
        model_state.pose.position.x = INITIAL_POS_X
        model_state.pose.position.y = INITIAL_POS_Y
        model_state.pose.position.z = 0
        model_state.pose.orientation.x = 0
        model_state.pose.orientation.y = 0
        model_state.pose.orientation.z = 0
        model_state.pose.orientation.w = 1
        model_state.twist.linear.x = 0
        model_state.twist.linear.y = 0
        model_state.twist.linear.z = 0
        model_state.twist.angular.x = 0
        model_state.twist.angular.y = 0
        model_state.twist.angular.z = 0

        if self.gazebo_model_state_service(model_state):
            rospy.loginfo('set robot init state')
        else:
            rospy.logerr("/gazebo/set_model_state service call failed")       
 
    def _update_map_size_ratio(self) -> None:
        data = None
        while data is None and not rospy.is_shutdown():
          try:
                data = rospy.wait_for_message('/map_metadata', MapMetaData, timeout=TIMEOUT)
          except:
                pass
        width = data.resolution * data.width
        height = data.resolution * data.height

        self.map_size_ratio = float(MAZE_SIZE/(width * height))


    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """
        run action and return results
        """        
        # rospy.loginfo('start step' + str(self.steps_in_episode + 1))

        #rospy.wait_for_service('/gazebo/unpause_physics')
        #try:
        #    self.unpause()
        #except (rospy.ServiceException) as e:
        #    rospy.loginfo("/gazebo/unpause_physics service call failed")
        
        self.last_action = self.now_action
        self.now_action = action
        if action == 0:  # turn left
            steering = STEERING
            throttle = THROTTLE
        elif action == 1:  # turn right
            steering = -1 * STEERING
            throttle = THROTTLE
        elif action == 2:  # straight
            steering = 0
            throttle = THROTTLE
        elif action == 3:  # backward
            steering = 0
            throttle = -1 * THROTTLE
        else:
            raise ValueError("Invalid action")

        # initialize rewards, next_state, done
        self.reward = None
        self.done = False

        self.steps_in_episode += 1
        self._send_action(steering, throttle)
        
        # time.sleep(SLEEP_BETWEEN_ACTION_AND_REWARD_CALCULATION)
        self._wait_until_twist_achieved(steering, throttle)

        sensor_state = self._update_scan()
        self._update_map_completeness()
        numeric_state = self._update_odom()
        
        #if self.steps_in_episode >= MAX_STEPS:
        #  rospy.wait_for_service('/gazebo/pause_physics')
        #  try:
        #      self.pause()
        #  except (rospy.ServiceException) as e:
        #      rospy.loginfo("/gazebo/pause_physics service call failed")

        next_state = np.concatenate([sensor_state, numeric_state])
        self._infer_reward()


        info = {}
        return next_state, self.reward, self.done, info
   
    def _wait_until_twist_achieved(self, angular_speed, linear_speed):
        """
        We wait for the cmd_vel twist given to be reached by the robot reading
        Bare in mind that the angular wont be controled , because its too imprecise.
        We will only consider to check if its moving or not inside the angular_speed_noise fluctiations it has.
        from the odometry.
        :param cmd_vel_value: Twist we want to wait to reach.
        :return:
        """
        rospy.loginfo("START wait_until_twist_achieved...")
        
        epsilon = 0.05
        update_rate = 10
        angular_speed_noise = 0.005
        rate = rospy.Rate(update_rate)
        
        angular_speed_is = self._check_angular_speed_dir(angular_speed, angular_speed_noise)
        
        linear_speed_plus = linear_speed + epsilon
        linear_speed_minus = linear_speed - epsilon
        
        roop_count = 0
        while not rospy.is_shutdown():
            current_odometry = None
            while current_odometry is None and not rospy.is_shutdown():
                try:
                    current_odometry = rospy.wait_for_message("/steer_drive_controller/odom", Odometry, timeout=TIMEOUT)

                except:
                    rospy.logerr("Current /odom not ready yet, retrying for getting odom")
  
            odom_linear_vel = current_odometry.twist.twist.linear.x
            odom_angular_vel = current_odometry.twist.twist.angular.z
            
            linear_vel_are_close = (odom_linear_vel <= linear_speed_plus) and (odom_linear_vel > linear_speed_minus)
            odom_angular_speed_is = self._check_angular_speed_dir(odom_angular_vel, angular_speed_noise)
                
            # We check if its turning in the same diretion or has stopped
            angular_vel_are_close = (angular_speed_is == odom_angular_speed_is)
            
            if linear_vel_are_close and angular_vel_are_close:
                rospy.loginfo("Reached Velocity!")
                break
            
            roop_count += 1
            if roop_count >= 3:
                rospy.logwarn('its regarded as crashed')
                break
            
            rospy.logwarn("Not there yet, keep waiting...")
            rate.sleep()
    
    def _check_angular_speed_dir(self, angular_speed, angular_speed_noise):
        """
        It States if the speed is zero, posititive or negative
        """
        # We check if odom angular speed is positive or negative or "zero"
        if (-angular_speed_noise < angular_speed <= angular_speed_noise):
            angular_speed_is = 0
        elif angular_speed > angular_speed_noise:
            angular_speed_is = 1
        elif angular_speed <= angular_speed_noise:
            angular_speed_is = -1
        else:
            angular_speed_is = 0
            rospy.logerr("Angular Speed has wrong value=="+str(angular_speed))
   
    def _update_scan(self) -> None:
        """
        """
        rospy.loginfo('waiting lidar scan')
        # adapt number of sensor information to TRAINING_IMAGE_SIZE
        data = None
        while data is None and not rospy.is_shutdown():
          try:
                data = rospy.wait_for_message('/scan_filtered', LaserScan, timeout=TIMEOUT)
          except:
                pass

        sensor_state = []
        self.min_distance = LIDAR_SCAN_MAX_DISTANCE
        mod = len(data.ranges)/TRAINING_IMAGE_SIZE
        for i, item in enumerate(data.ranges):
            if (i%mod==0):
                if np.isnan(data.ranges[i]):
                    sensor_state.append(LIDAR_SCAN_MAX_DISTANCE + 1.0)
                else:
                    sensor_state.append(data.ranges[i])
            if self.min_distance > data.ranges[i]:
                self.min_distance = data.ranges[i]

        return sensor_state

    def _update_odom(self):
        rospy.loginfo('waiting odom')
        trans = None
        while trans is None and not rospy.is_shutdown():
          try:
              # listen to transform
              trans = self.tfBuffer.lookup_transform('map', 'base_link', rospy.Time(0))
          except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
              pass
        rospy.loginfo('end waiting odom')

        self.position.x = trans.transform.translation.x
        self.position.y = trans.transform.translation.y
        self.orientation.z = trans.transform.rotation.z
        left_steps = MAX_STEPS - self.steps_in_episode
        numeric_state = np.array([
            self.position.x,
            self.position.y,
            self.orientation.z,
            self.last_action,
            left_steps,
            self.map_completeness_pct
        ])

        return numeric_state

    def _infer_reward(self) -> None:
        """
        """
        
        if self.map_completeness_pct > MAP_COMPLETENESS_THRESHOLD:
            self.reward = REWARD_MAP_COMPLETED
            state = 'comp'
        elif self.min_distance < COLLISION_THRESHOLD:
            # Robot likely hit the wall
            self.reward = REWARD_CRASHED
            state = 'crashed'
        else:
            self.reward = self.map_completeness_pct - self.last_map_completeness_pct
            state = ''
        
        self.reward_in_episode += self.reward
        rospy.loginfo('reward:' + str(self.reward) + ' ' + state)

    def _update_map_completeness(self) -> None:
        
        rospy.loginfo('waiting map')
        data = None
        while data is None and not rospy.is_shutdown():
          try:
                data = rospy.wait_for_message('/map', OccupancyGrid, timeout=TIMEOUT)
          except:
                pass
        rospy.loginfo('ended waiting map')

        self.occupancy_grid = data.data
        sum_grid = len(self.occupancy_grid)
        num_occupied = 0
        num_unoccupied = 0
        num_negative = 0
        for n in self.occupancy_grid:
            if n == 0:
                num_unoccupied += 1
            elif n == 100:
                num_occupied += 1 
        
        self.last_map_completeness_pct = self.map_completeness_pct
        self.map_completeness_pct = ((num_occupied + num_unoccupied) * 100 / sum_grid) / self.map_size_ratio
        if self.steps_in_episode == 1:
            self.last_map_completeness_pct = self.map_completeness_pct

        rospy.loginfo('map completenes:' + str(self.map_completeness_pct)) 
    
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
