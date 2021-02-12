[![Build Status](https://travis-ci.org/kuwabaray/ros_rl_for_slam.svg?branch=main)](https://travis-ci.org/kuwabaray/ros_rl_for_slam)
# ROS Path planning for SLAM by training a DQN model on Gazebo
It simulates Rosbot movement on Gazebo and trains a rainforcement learning model DQN.  
I mainly refer this thesis [REINFORCEMENT LEARNING HELPS SLAM: LEARNING TO BUILD MAPS](https://www.researchgate.net/publication/343874756_REINFORCEMENT_LEARNING_HELPS_SLAM_LEARNING_TO_BUILD_MAPS)  
![rviz](https://i.imgur.com/TcuPW83.png)
## Description
* **dqn\_for\_slam**: RL Enviroment (gym) and trainig a DQN model (Keras-rl2, Tensorflow) 
* **rosbot\_description**: Based on [rosbot\_description](https://github.com/husarion/rosbot_description). turned off camera and Infrared for computational load and reduced friction. 
* **simulate\_robot\_rl**: The entry point of training launches all nodes
* **simulate\_map**: Map for simulation
* **slam\_gmapping**: Based on [slam\_gmapping](https://github.com/ros-perception/slam\_gmapping). Added the service and functions that clear a map and restart gmapping.

IMU and Wheel odometry are used for Localization. RPLidar is filtered for SLAM (gmapping). 
![graph](https://i.imgur.com/MtUxYwC.png) 
## Dependency
This application don't require GPU.

environment
* Ubuntu 20.04 
* ROS Noetic

ros package
* geometry2
* openslam_gmapping
* robot_localization

install python frameworks to run 
```bash
pip3 install matplotlib tensorflow gym numpy keras-rl2 rospkg
pip3 install --upgrade tensorflow
```
## Usage
Run trainning 
 ```bash
roslaunch simulate_robot_rl simulate_robot.launch
```
Trained models are saved to ~/dqn\_for\_slam/dqn\_for\_slam/models/  
Figures show training process are saved to ~/dqn\_for\_slam/dqn\_for\_slam/figures/
## License
This library is licensed under the MIT License.

## Author
mail to: 6318036@ed.tus.ac.jp


