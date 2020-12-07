# ROS Path planning for SLAM by training a DQN model on Gazebo
This ia a ROS workspace. it simulates Turtlebot3 movement on Gazebo and trains a rainforcement learning model DQN. 
## Packages
* custom\_gmapping: gmapping (https://github.com/ros-perception/slam\_gmapping) + Map reset service, function
* dqn\_for\_slam: enviroment (gym) and trainig a DQN model (Keras-rl2, Tensorflow)
* simulate\_robot\_rl entry point of training. it runs ros, gazebo, rviz and other packages. 
* simulate\_map: Map for simulation 
## Usage
Docker URL:
Run trainning 
 ```bash
cd ~/environment
roslaunch simulate_robot_rl simulate_robot.launch
```
## Note


## Author
mail to: 6318036@ed.tus.ac.jp


