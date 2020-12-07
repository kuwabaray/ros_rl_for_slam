cmake_minimum_required(VERSION 3.0.2)
project(simulation_map)
 
find_package(catkin REQUIRED COMPONENTS
  gazebo_ros
)

catkin_package()

install(DIRECTORY launch models worlds maps photos routes
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

