cmake_minimum_required(VERSION 3.8)
project(yolo_cpp_ros)

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

include_directories(
  include
  ${OpenCV_INCLUDE_DIRS}
)

add_executable(yolo_cpp_ros 
  src/yolo_main.cpp
  src/yolo_cpp_node/yolo_node.cpp
  src/yolo_cpp_node/yolo_onnx.cpp
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

install(TARGETS
  yolo_cpp_ros
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
