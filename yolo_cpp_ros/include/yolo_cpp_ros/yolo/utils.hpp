// Copyright (c) 2025 Alejandro González Cantón
// SPDX-License-Identifier: MIT

#ifndef YOLO_CPP_ROS__YOLO__UTILS_HPP_
#define YOLO_CPP_ROS__YOLO__UTILS_HPP_

#include "onnxruntime_cxx_api.h"
#include "yolo_msgs/msg/bounding_box2_d.hpp"
#include <opencv2/opencv.hpp>
#include <algorithm>
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
      : Box(box.x1, box.y1, box.x2, box.y2, box.score, box.index, box.class_id),
        mask_coeffs(std::vector<float>()) {}
  BoxWithMask(Box box, std::vector<float> mask_coeffs)
      : Box(box.x1, box.y1, box.x2, box.y2, box.score, box.index, box.class_id),
        mask_coeffs(mask_coeffs) {}
};

// A single 2D pose keypoint. `visible` is the sigmoid'd visibility already
// decoded by the exported ONNX graph; it doubles as the per-keypoint
// confidence and, compared against conf_threshold, as the publish filter
// (matches the Python node, which drops keypoints with conf < threshold).
struct Keypoint {
  float x = 0.0f;
  float y = 0.0f;
  float visible = 0.0f;
};

struct BoxWithKeypoints : public Box {
  std::vector<Keypoint> keypoints;  // COCO order, (x, y, visible) per kp

  BoxWithKeypoints() = default;
  BoxWithKeypoints(Box box)
      : Box(box.x1, box.y1, box.x2, box.y2, box.score, box.index, box.class_id),
        keypoints(std::vector<Keypoint>()) {}
  BoxWithKeypoints(Box box, std::vector<Keypoint> keypoints)
      : Box(box.x1, box.y1, box.x2, box.y2, box.score, box.index, box.class_id),
        keypoints(std::move(keypoints)) {}
};

struct YoloParams {
  std::string model_type;  // "YOLO"|"Detect"|"Segment"|"auto" (by file name)
  std::string model_path;
  std::string device;
  float threshold;
  float iou;
  bool enable;   // gate inference (matches the Python node's `enable`)
  int max_det;   // cap on the number of detections published per image
  int image_reliability;
  std::string image_topic;
  int n_threads;
};

template <typename BoxT>
float iou(const BoxT &box1, const BoxT &box2) {
  float x1 = std::max(box1.x1, box2.x1);
  float y1 = std::max(box1.y1, box2.y1);
  float x2 = std::min(box1.x2, box2.x2);
  float y2 = std::min(box1.y2, box2.y2);

  float intersection = std::max(0.0f, x2 - x1) * std::max(0.0f, y2 - y1);
  float area1 =
      std::max(0.0f, box1.x2 - box1.x1) * std::max(0.0f, box1.y2 - box1.y1);
  float area2 =
      std::max(0.0f, box2.x2 - box2.x1) * std::max(0.0f, box2.y2 - box2.y1);
  float union_area = area1 + area2 - intersection;

  return union_area > 0 ? intersection / union_area : 0;
}

// Per-class NMS. Sorts `boxes` in place by confidence, then returns the kept
// indices into the *same* (now-sorted) vector, so the caller reads
// boxes[indices[i]]. Works for Box or BoxWithMask (any type exposing
// .score/.class_id/.x1..y2). No per-candidate heap allocation.
template <typename BoxT>
std::vector<int> nms(std::vector<BoxT> &boxes, float iou_threshold,
                     float conf_threshold) {
  std::vector<int> indices;

  std::sort(boxes.begin(), boxes.end(),
            [](const BoxT &a, const BoxT &b) { return a.score > b.score; });

  std::vector<bool> suppressed(boxes.size(), false);

  for (size_t i = 0; i < boxes.size(); ++i) {
    if (boxes[i].score < conf_threshold || suppressed[i]) {
      continue;
    }
    indices.push_back(static_cast<int>(i));
    for (size_t j = i + 1; j < boxes.size(); ++j) {
      if (suppressed[j] || boxes[j].class_id != boxes[i].class_id) {
        continue;
      }
      if (iou(boxes[i], boxes[j]) > iou_threshold) {
        suppressed[j] = true;
      }
    }
  }

  return indices;
}

cv::Mat letterbox(const cv::Mat &img, const cv::Size &new_shape,
                  const cv::Scalar &color);
cv::Mat inverse_letterbox(const cv::Mat &letterboxed,
                          const cv::Size &original_image_size,
                          const cv::Size &resized_image_size);
Box scale_box(const Box &box, const cv::Size &original_image_size,
              const cv::Size &resized_image_size);
std::vector<Keypoint>
scale_keypoints(const std::vector<Keypoint> &keypoints,
                const cv::Size &original_image_size,
                const cv::Size &resized_image_size);
yolo_msgs::msg::BoundingBox2D
convert_to_bounding_box(const yolo_onnx_utils::Box &box);

std::vector<yolo_onnx_utils::Box>
get_boxes(const std::vector<Ort::Value> &preds,
          const cv::Size &original_image_size,
          const cv::Size &resized_image_size, const int num_classes);

} // namespace yolo_onnx_utils

#endif // YOLO_CPP_ROS__YOLO__UTILS_HPP_