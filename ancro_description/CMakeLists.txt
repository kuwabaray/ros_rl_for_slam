cmake_minimum_required(VERSION 3.0.2)
project(ancro_description)

find_package(catkin REQUIRED COMPONENTS
  actionlib
  actionlib_msgs
  rospy
  move_base_msgs
  geometry_msgs
)

# Generate messages in the 'msg' folders
#  add_message_files(
#    FILES
#    Person.msg
#  )

# Generate services in the 'srv' folder
# add_service_files(
#   FILES
#   SrvTutorial.srv

# )

# Generate actions in the 'action' folder
# add_action_files(
#   FILES
#   Fibonacci.action
#   WashTheDishes.action
# )

generate_messages(
  DEPENDENCIES
  actionlib_msgs
  move_base_msgs
  geometry_msgs
)

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES grc_tutorial
 CATKIN_DEPENDS rospy move_base_msgs actionlib actionlib_msgs geometry_msgs
#  DEPENDS system_lib
)

include_directories(

  ${catkin_INCLUDE_DIRS}
)

catkin_install_python(PROGRAMS
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
