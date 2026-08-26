// Copyright (c) 2026 Alejandro González Cantón
// SPDX-License-Identifier: MIT

#ifndef YOLO_CPP_ROS__YOLO__POSE_HPP_
#define YOLO_CPP_ROS__YOLO__POSE_HPP_

#include "yolo_cpp_ros/engine/model.hpp"
#include "yolo_cpp_ros/yolo/utils.hpp"
#include "yolo_msgs/msg/detection.hpp"
#include <vector>

namespace yolo_onnx {

class YoloPose : public Model {
public:
  YoloPose(yolo_utils::YoloParams params);
  ~YoloPose();

protected:
  virtual std::vector<yolo_msgs::msg::Detection>
  postprocess(const cv::Size &original_image_size,
              const cv::Size &resized_image_size,
              const std::vector<Ort::Value> &outputTensors) override;
};
} // namespace yolo_onnx
#endif // YOLO_CPP_ROS__YOLO__POSE_HPP_
