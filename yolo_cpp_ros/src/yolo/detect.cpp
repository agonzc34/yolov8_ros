// Copyright (c) 2025 Alejandro González Cantón
// SPDX-License-Identifier: MIT

#include "yolo_cpp_ros/yolo/detect.hpp"
#include "yolo_cpp_ros/yolo/utils.hpp"

namespace yolo_onnx {

YoloDetect::YoloDetect(yolo_utils::YoloParams params) : Model(params) {}

YoloDetect::~YoloDetect() {}

std::vector<yolo_msgs::msg::Detection>
YoloDetect::postprocess(const cv::Size &original_image_size,
                        const cv::Size &resized_image_size,
                        const std::vector<Ort::Value> &preds) {
  std::vector<yolo_msgs::msg::Detection> detection_array;
  std::vector<yolo_utils::Box> detections;

  std::vector<int64_t> shape = preds[0].GetTensorTypeAndShapeInfo().GetShape();

  if (shape.back() == 6) {
    // Process predictions without applying NMS
    detections = get_detection_without_nms(preds, shape, original_image_size,
                                           resized_image_size);
  } else {
    // Process predictions applying NMS
    const size_t num_features = shape[1];
    const int num_classes = static_cast<int>(num_features) - 4;

    detections = get_detection_with_nms(
        preds, original_image_size, resized_image_size, num_classes,
        this->iou_threshold, this->conf_threshold);
  }

  for (size_t i = 0; i < detections.size(); ++i) {
    yolo_msgs::msg::Detection detection;
    detection.bbox = yolo_utils::convert_to_bounding_box(detections[i]);
    detection.score = detections[i].score;
    detection.class_id = detections[i].class_id;
    detection.id = "0";
    if (detections[i].class_id < static_cast<int>(this->class_names.size())) {
      detection.class_name = this->class_names[detections[i].class_id];
    } else {
      detection.class_name = "unknown";
    }
    detection_array.push_back(detection);
  }

  return detection_array;
}

std::vector<yolo_utils::Box> get_detection_without_nms(
    const std::vector<Ort::Value> &preds, std::vector<int64_t> output_shape,
    const cv::Size &original_image_size, const cv::Size &resized_image_size) {
  std::vector<yolo_utils::Box> boxes;
  for (size_t i = 0; i < preds.size(); ++i) {
    auto pred = preds[i].GetTensorData<float>();
    for (size_t j = 0; j < static_cast<size_t>(output_shape[0]); ++j) {
      yolo_utils::Box box;
      box.x1 = pred[j * 6 + 0];
      box.y1 = pred[j * 6 + 1];
      box.x2 = pred[j * 6 + 2];
      box.y2 = pred[j * 6 + 3];
      box.score = pred[j * 6 + 4];
      box.class_id = static_cast<int>(pred[j * 6 + 5]);
      box.index = j;

      yolo_utils::Box scaled_box = yolo_utils::scale_box(
          box, original_image_size, resized_image_size);
      boxes.push_back(scaled_box);
    }
  }

  return boxes;
}

std::vector<yolo_utils::Box> get_detection_with_nms(
    const std::vector<Ort::Value> &preds, const cv::Size &original_image_size,
    const cv::Size &resized_image_size, const int num_classes,
    float iou_threshold, float conf_threshold) {

  std::vector<yolo_utils::Box> boxes = yolo_utils::get_boxes(
      preds, original_image_size, resized_image_size, num_classes);

  boxes.erase(std::remove_if(boxes.begin(), boxes.end(),
                             [conf_threshold](const yolo_utils::Box &box) {
                               return box.score < conf_threshold;
                             }),
              boxes.end());

  // Boxes are sorted in place by nms(); indices index the same vector.
  auto indices = yolo_utils::nms(boxes, iou_threshold, conf_threshold);
  std::vector<yolo_utils::Box> filtered_boxes;
  for (size_t i = 0; i < indices.size(); ++i) {
    filtered_boxes.push_back(boxes[indices[i]]);
  }
  return filtered_boxes;
}

}  // namespace yolo_onnx