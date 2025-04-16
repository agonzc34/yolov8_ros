// MIT License
//
// Copyright (c) 2025 Alejandro González Cantón
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#ifndef YOLO_CPP_ROS__YOLO__UTILS_HPP_
#define YOLO_CPP_ROS__YOLO__UTILS_HPP_

#include "onnxruntime_cxx_api.h"
#include "yolo_msgs/msg/bounding_box2_d.hpp"
#include <opencv2/opencv.hpp>
#include <vector>

namespace yolo_onnx_utils {
struct Box {
  float x1, y1, x2, y2, score;
  int index, class_id;

  Box(float x1, float y1, float x2, float y2, float score, int index,
      int class_id)
      : x1(x1), y1(y1), x2(x2), y2(y2), score(score), index(index),
        class_id(class_id) {}
  Box() = default;
  virtual ~Box() = default;
};

struct BoxWithMask : public Box {
  std::vector<float> mask_coeffs;

  BoxWithMask() = default;
  BoxWithMask(Box box)
      : Box(box.x1, box.y1, box.x2, box.y2, box.score, box.index,
            box.class_id),
        mask_coeffs(std::vector<float>()) {}
  BoxWithMask(Box box,
              std::vector<float> mask_coeffs)
      : Box(box.x1, box.y1, box.x2, box.y2, box.score, box.index,
            box.class_id),
        mask_coeffs(mask_coeffs) {}
};

struct YoloParams {
  std::string model_path;
  std::string device;
  float threshold;
  float iou;
  int image_reliability;
  std::string image_topic;
  int n_threads;
};

float iou(const std::shared_ptr<Box> &box1, const std::shared_ptr<Box> &box2);

std::vector<int> nms(std::vector<std::shared_ptr<Box>>& boxes,
                     float iou_threshold, float conf_threshold);
cv::Mat letterbox(const cv::Mat &img, const cv::Size &new_shape,
                  const cv::Scalar &color);
Box scale_box(const Box &box, const cv::Size &original_image_size,
              const cv::Size &resized_image_size);
yolo_msgs::msg::BoundingBox2D
convert_to_bounding_box(const yolo_onnx_utils::Box &box);

std::vector<yolo_onnx_utils::Box>
get_boxes(const std::vector<Ort::Value> &preds, std::vector<int64_t> shape,
          const cv::Size &original_image_size,
          const cv::Size &resized_image_size, const int num_classes,
          float conf_threshold);

std::vector<yolo_onnx_utils::Box> get_detection_without_nms(
    const std::vector<Ort::Value> &preds, std::vector<int64_t> shape,
    const cv::Size &original_image_size, const cv::Size &resized_image_size);

std::vector<yolo_onnx_utils::Box> get_detection_with_nms(
    const std::vector<Ort::Value> &preds, std::vector<int64_t> shape,
    const cv::Size &original_image_size, const cv::Size &resized_image_size,
    const int num_classes, float iou_threshold, float conf_threshold);

std::vector<yolo_onnx_utils::BoxWithMask> get_segmentation_with_nms(
    const std::vector<Ort::Value> &preds, std::vector<int64_t> shape,
    const cv::Size &original_image_size, const cv::Size &resized_image_size,
    const int num_classes, float iou_threshold, float conf_threshold);

} // namespace yolo_onnx_utils

#endif // YOLO_CPP_ROS__YOLO__UTILS_HPP_