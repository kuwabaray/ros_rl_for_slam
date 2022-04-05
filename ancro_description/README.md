# ancro_description
Autonomous Navigation and Communication Robot


**Install the Package**

Clone this repo into the src folder of your catkin workspace

git clone https://github.com/Avi241/ancro_description

cd .. && catkin make

**Launch the Robot in Gazebo World**

roslaunch ancro_description gazebo.launch

**Run the Robot Teleop Opeartion Script**

rosrun ancro_description robot_teleop.py

**To perform SLAM**

roslaunch ancro_description gmapping.launch

**For Navigation**

roslaunch ancro_description navigation.launch
