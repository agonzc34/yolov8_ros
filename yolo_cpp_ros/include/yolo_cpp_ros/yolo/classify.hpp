// Copyright (c) 2026 Alejandro González Cantón
// SPDX-License-Identifier: MIT

#ifndef YOLO_CPP_ROS__YOLO__CLASSIFY_HPP_
#define YOLO_CPP_ROS__YOLO__CLASSIFY_HPP_

#include "yolo_cpp_ros/engine/model.hpp"
#include "yolo_cpp_ros/yolo/utils.hpp"
#include "yolo_msgs/msg/detection.hpp"
#include <vector>

namespace yolo_onnx {

// Image-level classifier (YOLO cls models). Like the other tasks it only
// overrides the generic postprocess() hook; preprocess/inference are the
// engine's. The exported graphs emit one [1, N] row of per-class scores;
// ultralytics cls exports bake a softmax into the graph, so the row is
// already a probability distribution. Each top-k class is published as a
// Detection with an EMPTY bbox (there is no spatial extent for an
// image-level label), carrying class_id / class_name / score — matching how
// the rest of the package reuses Detection.
class YoloClassify : public Model {
public:
  YoloClassify(yolo_utils::YoloParams params);
  ~YoloClassify();

protected:
  virtual std::vector<yolo_msgs::msg::Detection>
  postprocess(const cv::Size &original_image_size,
              const cv::Size &resized_image_size,
              const std::vector<Ort::Value> &preds) override;

private:
  int top_k_{5}; // classes published per image (softmax probs, desc order)
};

} // namespace yolo_onnx

#endif // YOLO_CPP_ROS__YOLO__CLASSIFY_HPP_