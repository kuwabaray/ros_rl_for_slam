# ROS Path planning for SLAM by training a DQN model on Gazebo
This ia a ROS workspace. it simulates Turtlebot3 movement on Gazebo and trains a rainforcement learning model DQN. 
I mainly refer the thesis [REINFORCEMENT LEARNING HELPS SLAM: LEARNING TO BUILD MAPS](https://www.researchgate.net/publication/343874756_REINFORCEMENT_LEARNING_HELPS_SLAM_LEARNING_TO_BUILD_MAPS)
![rviz](https://imgur.com/jkzsjpk.jpg)
## Description
* **custom\_gmapping**: I added the map reset service and function to [gmapping](https://github.com/ros-perception/slam\_gmapping) 
* **dqn\_for\_slam**: Enviroment (gym) and trainig a DQN model (Keras-rl2, Tensorflow) 
* **simulate\_robot\_rl**: The entry point of training. it runs ros, gazebo, rviz and other packages. 
* **simulate\_map**: Map for simulation
![graph](https://imgur.com/SeGLM1x.jpg) 
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


