// Copyright (c) 2026 Alejandro González Cantón
// Portions Copyright (c) 2023-2026 Miguel Ángel González Santamarta
// SPDX-License-Identifier: MIT

#ifndef YOLO_CPP_ROS__YOLO__OBB_HPP_
#define YOLO_CPP_ROS__YOLO__OBB_HPP_

#include "yolo_cpp_ros/engine/model.hpp"
#include "yolo_cpp_ros/yolo/utils.hpp"
#include "yolo_msgs/msg/detection.hpp"
#include <vector>

namespace yolo_onnx {

// An oriented bounding box in the ultralytics xywhr parameterization
// (center x, center y, width, height, rotation angle in radians). The box is
// decoded into the letterboxed model-input frame; the angle is unaffected by
// letterboxing (it is a rotation, not a scale).
struct ObbBox {
  float cx = 0.0f;
  float cy = 0.0f;
  float w = 0.0f;
  float h = 0.0f;
  float angle = 0.0f; // radians
  float score = 0.0f;
  int class_id = -1;
};

// Oriented (rotated) bounding box detection. Ultralytics OBB heads predict
// one extra channel per anchor vs. Detect: [cx, cy, w, h, class_scores...,
// angle]. The angle channel is already decoded in radians by the exported
// graph (YOLO11: (sigmoid - 0.25) * pi in [-pi/4, 3pi/4]; YOLO26: raw
// radians). OBB exports are always the raw path: rotated NMS cannot be baked
// into the graph, so NMS is performed here (the repository's pose/detect
// nodes additionally handle baked-head exports, which OBB does not produce).
class YoloOBB : public Model {
public:
  YoloOBB(yolo_utils::YoloParams params);
  ~YoloOBB();

protected:
  virtual std::vector<yolo_msgs::msg::Detection>
  postprocess(const cv::Size &original_image_size,
              const cv::Size &resized_image_size,
              const std::vector<Ort::Value> &outputTensors) override;
};
} // namespace yolo_onnx
#endif // YOLO_CPP_ROS__YOLO__OBB_HPP_
