cmake_minimum_required(VERSION 3.0.2)
project(image_generate)

find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  nav_msgs
  sensor_msgs
  roscpp
  rospy
  std_msgs
  pcl_ros
  tf
  eigen_conversions
  pcl_conversions
  cv_bridge
  image_transport
)

find_package(PCL 1.8 REQUIRED)
find_package(GTSAM REQUIRED QUIET)
find_package(OpenMP REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Boost REQUIRED COMPONENTS system timer)

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES image_generate
#  CATKIN_DEPENDS roscpp rospy std_msgs
#  DEPENDS system_lib
)

include_directories(
# include
  ${catkin_INCLUDE_DIRS}
)

add_executable(brute_sliding_intensity_map src/brute_sliding_intensity_map.cpp)
target_link_libraries(brute_sliding_intensity_map ${catkin_LIBRARIES} ${BOOST_LIBRARY} ${PCL_LIBRARIES} ${OpenMP_CXX_FLAGS} ${OpenCV_LIBS} gtsam)
target_compile_options(brute_sliding_intensity_map PRIVATE ${OpenMP_CXX_FLAGS})