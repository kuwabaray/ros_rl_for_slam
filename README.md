[![Build Status](https://travis-ci.org/kuwabaray/ros_rl_for_slam.svg?branch=main)](https://travis-ci.org/kuwabaray/ros_rl_for_slam)
# ROS Path planning for SLAM by training a DQN model on Gazebo
This ia a ROS meta package. it simulates Turtlebot3 movement on Gazebo and trains a rainforcement learning model DQN.  
I mainly refer this thesis [REINFORCEMENT LEARNING HELPS SLAM: LEARNING TO BUILD MAPS](https://www.researchgate.net/publication/343874756_REINFORCEMENT_LEARNING_HELPS_SLAM_LEARNING_TO_BUILD_MAPS)  
![rviz](https://imgur.com/jkzsjpk.jpg)
## Description
* **slam\_gmapping**: Its based on [gmapping](https://github.com/ros-perception/slam\_gmapping). I added the service and functions that reset a map data of gmapping.
* **dqn\_for\_slam**: Python package. Enviroment (gym) and trainig a DQN model (Keras-rl2, Tensorflow) 
* **simulate\_robot\_rl**: Ros package. The entry point of training. it runs ros, gazebo, rviz and other packages. 
* **simulate\_map**: Ros package. Map for simulation
![graph](https://imgur.com/SeGLM1x.jpg) 
## Dependency
This application don't require GPU.
I run on 
* Ubuntu 18.04 
* ROS melodic
* python 3.7
install frameworks to run rl 
```bash
pip3 install matplotlib==3.3.3 tensorflow gym==0.17.3 numpy==1.18.5 keras-rl2==1.0.4 rospkg==1.2.9
pip3 install --upgrade tensorflow
```
## Usage
Run trainning 
 ```bash
cd ~/environment
roslaunch simulate_robot_rl simulate_robot.launch
```
Trained models are saved to ~/dqn\_for\_slam/dqn\_for\_slam/models/  
Figures show training process are saved to ~/dqn\_for\_slam/dqn\_for\_slam/figures/
## License
This library is licensed under the MIT License.

## Author
mail to: 6318036@ed.tus.ac.jp


