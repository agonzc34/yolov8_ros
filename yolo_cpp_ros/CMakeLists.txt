cmake_minimum_required(VERSION 3.8)
project(yolo_cpp_ros)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(rclcpp REQUIRED)
find_package(yolo_msgs REQUIRED)
find_package(rclcpp_lifecycle REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(OpenCV REQUIRED)
find_package(yolo_onnxruntime_vendor REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(message_filters REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(geometry_msgs REQUIRED)

include_directories(
  include
  ${OpenCV_INCLUDE_DIRS}
)

# Install yolo_cpp_ros
add_executable(yolo_cpp_ros
  src/node/yolo_node.cpp
  src/node/yolo_main.cpp

  src/yolo/segment.cpp
  src/yolo/detect.cpp
  src/yolo/pose.cpp
  src/yolo/utils.cpp

  src/engine/model.cpp
)

target_include_directories(yolo_cpp_ros PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

ament_target_dependencies(yolo_cpp_ros
  sensor_msgs
  rclcpp
  yolo_msgs
  rclcpp_lifecycle
  cv_bridge
  yolo_onnxruntime_vendor
)

target_link_libraries(yolo_cpp_ros
  ${OpenCV_LIBS}
  ${ORT_LIBRARIES}
)

# Install debug node
add_executable(yolo_cpp_debug
  src/node/yolo_debug.cpp
  src/node/debug_node.cpp
)

target_include_directories(yolo_cpp_debug PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
ament_target_dependencies(yolo_cpp_debug
  sensor_msgs
  rclcpp
  yolo_msgs
  rclcpp_lifecycle
  cv_bridge
  visualization_msgs
  message_filters
)

target_link_libraries(yolo_cpp_debug
  ${OpenCV_LIBS}
)

# Install tracking node
add_executable(yolo_cpp_tracking
  src/node/tracking_node.cpp
  src/node/tracking_main.cpp

  src/tracking/byte_tracker.cpp
  src/tracking/strack.cpp

  src/tracking/utils/kalman_filter.cpp
  src/tracking/utils/lapjv.cpp
  src/tracking/utils/matching.cpp
)

target_include_directories(yolo_cpp_tracking PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

ament_target_dependencies(yolo_cpp_tracking
  sensor_msgs
  rclcpp
  yolo_msgs
  rclcpp_lifecycle
  message_filters
)

# Install 3D detection node
add_executable(yolo_cpp_3d
  src/node/detect_3d_node.cpp
  src/node/detect_3d_main.cpp
)

target_include_directories(yolo_cpp_3d PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

ament_target_dependencies(yolo_cpp_3d
  sensor_msgs
  rclcpp
  yolo_msgs
  rclcpp_lifecycle
  cv_bridge
  message_filters
  tf2_ros
  geometry_msgs
)

target_link_libraries(yolo_cpp_3d
  ${OpenCV_LIBS}
)

install(TARGETS
  yolo_cpp_ros yolo_cpp_debug yolo_cpp_tracking yolo_cpp_3d
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
