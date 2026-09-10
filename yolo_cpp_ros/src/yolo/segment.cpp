// Copyright (c) 2025 Alejandro González Cantón
// SPDX-License-Identifier: MIT

#include "yolo_cpp_ros/yolo/segment.hpp"
#include "yolo_cpp_ros/yolo/utils.hpp"
#include "yolo_msgs/msg/point2_d.hpp"
#include <opencv2/core/types.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

namespace yolo_onnx {

YoloSegment::YoloSegment(yolo_ros::yolo::utils::YoloParams params)
    : Model(params) {}

YoloSegment::~YoloSegment() {}

std::vector<yolo_msgs::msg::Detection>
YoloSegment::postprocess(const cv::Size &original_image_size,
                         const cv::Size &resized_image_size,
                         const std::vector<Ort::Value> &preds) {
  std::vector<yolo_msgs::msg::Detection> detection_array;
  std::vector<yolo_ros::yolo::utils::BoxWithMask> bbox_array;

  std::vector<int64_t> box_shape =
      preds[0].GetTensorTypeAndShapeInfo().GetShape();
  // Process predictions applying NMS
  const size_t num_features = box_shape[1];
  const int num_classes = static_cast<int>(num_features) - 4 - 32;

  bbox_array = get_segmentation_with_nms(
      preds, original_image_size, resized_image_size, num_classes,
      this->iou_threshold, this->conf_threshold);

  std::vector<int64_t> mask_shape =
      preds[1].GetTensorTypeAndShapeInfo().GetShape(); // [1, 32, maskH, maskW]

  const int mask_h = static_cast<int>(mask_shape[2]);
  const int mask_w = static_cast<int>(mask_shape[3]);

  std::vector<cv::Mat> mask_protos;
  const float *mask_ptr = preds[1].GetTensorData<float>();
  for (int64_t i = 0; i < mask_shape[1]; ++i) {
    cv::Mat mask(mask_h, mask_w, CV_32F,
                 const_cast<float *>(mask_ptr + i * mask_h * mask_w));
    mask_protos.push_back(mask);
  }

  std::vector<std::vector<cv::Point>> masks;
  for (size_t i = 0; i < bbox_array.size(); ++i) {
    const auto mask_coeffs = bbox_array[i].mask_coeffs;
    cv::Mat seg_mask = cv::Mat::zeros(mask_h, mask_w, CV_32F);

    // Linear combination of prototype masks
    for (int m = 0; m < 32; ++m) {
      seg_mask += mask_coeffs[m] * mask_protos[m];
    }

    // Apply sigmoid activation
    cv::exp(-seg_mask, seg_mask);
    seg_mask = 1.0 / (1.0 + seg_mask);

    // Apply threshold to get binary mask (Filter some noise)
    cv::Mat filtered_seg_mask;
    cv::threshold(seg_mask, filtered_seg_mask, 0.7, 255.0, cv::THRESH_BINARY);
    filtered_seg_mask.convertTo(filtered_seg_mask, CV_8U);

    // Rescale the mask to the original image size
    auto resized_mask = yolo_ros::yolo::utils::inverse_letterbox(
        filtered_seg_mask, original_image_size, resized_image_size);

    // Crop to bounding box
    cv::Rect roi(bbox_array[i].x1, bbox_array[i].y1,
                 bbox_array[i].x2 - bbox_array[i].x1,
                 bbox_array[i].y2 - bbox_array[i].y1);
    roi &= cv::Rect(0, 0, resized_mask.cols, resized_mask.rows);
    cv::Mat cropped_mask = cv::Mat::zeros(resized_mask.size(), CV_8U);
    if (roi.area() > 0) {
      resized_mask(roi).copyTo(cropped_mask(roi));
    }

    // Find contours in the cropped mask
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(cropped_mask, contours, cv::RETR_EXTERNAL,
                     cv::CHAIN_APPROX_SIMPLE);

    // Find the largest contour
    double max_area = 0;
    int max_contour_index = -1;
    for (size_t j = 0; j < contours.size(); ++j) {
      double area = cv::contourArea(contours[j]);
      if (area > max_area) {
        max_area = area;
        max_contour_index = j;
      }
    }
    if (max_contour_index != -1) {
      std::vector<cv::Point> largest_contour = contours[max_contour_index];
      masks.push_back(largest_contour);
    } else {
      masks.push_back(std::vector<cv::Point>());
    }
  }

  for (size_t i = 0; i < bbox_array.size(); ++i) {
    yolo_msgs::msg::Detection detection;
    detection.bbox =
        yolo_ros::yolo::utils::convert_to_bounding_box(bbox_array[i]);

    // Segmentation mask
    detection.mask.height = original_image_size.height;
    detection.mask.width = original_image_size.width;
    detection.mask.data.clear();
    for (const auto &point : masks[i]) {
      yolo_msgs::msg::Point2D point2d;
      point2d.x = point.x;
      point2d.y = point.y;
      detection.mask.data.push_back(point2d);
    }

    // Some additional information
    detection.score = bbox_array[i].score;
    detection.class_id = bbox_array[i].class_id;
    detection.id = "0";
    if (bbox_array[i].class_id < static_cast<int>(this->class_names.size())) {
      detection.class_name = this->class_names[bbox_array[i].class_id];
    } else {
      detection.class_name = "unknown";
    }
    detection_array.push_back(detection);
  }

  return detection_array;
}

std::vector<yolo_ros::yolo::utils::BoxWithMask> get_segmentation_with_nms(
    const std::vector<Ort::Value> &preds, const cv::Size &original_image_size,
    const cv::Size &resized_image_size, const int num_classes,
    float iou_threshold, float conf_threshold) {

  const float *raw_output =
      preds[0].GetTensorData<float>(); // Extract raw output data from the
  const size_t num_detections =
      preds[0].GetTensorTypeAndShapeInfo().GetShape()[2];

  std::vector<yolo_ros::yolo::utils::BoxWithMask> seg_boxes;

  // 1. Get the bounding boxes (anchors below conf_threshold are skipped)
  std::vector<yolo_ros::yolo::utils::Box> boxes =
      yolo_ros::yolo::utils::get_boxes(preds, original_image_size,
                                       resized_image_size, num_classes,
                                       conf_threshold);

  // 2. Add the mask coefficients to the boxes
  std::vector<yolo_ros::yolo::utils::BoxWithMask> boxes_with_mask;
  for (size_t i = 0; i < boxes.size(); ++i) {
    if (boxes[i].score < conf_threshold) {
      continue;
    }
    yolo_ros::yolo::utils::BoxWithMask box_with_mask(boxes[i]);
    std::vector<float> mask_coeffs(32);
    for (size_t m = 0; m < 32; ++m) {
      mask_coeffs[m] = raw_output[(num_classes + 4 + m) * num_detections + i];
    }
    box_with_mask.mask_coeffs = mask_coeffs;
    boxes_with_mask.push_back(box_with_mask);
  }

  // BoxesWithMask are sorted in place by nms(); indices index the same vector.
  auto indices = yolo_ros::yolo::utils::nms(boxes_with_mask, iou_threshold,
                                            conf_threshold);

  std::vector<yolo_ros::yolo::utils::BoxWithMask> filtered_boxes;
  for (size_t i = 0; i < indices.size(); ++i) {
    filtered_boxes.push_back(boxes_with_mask[indices[i]]);
  }

  return filtered_boxes;
}

} // namespace yolo_onnx