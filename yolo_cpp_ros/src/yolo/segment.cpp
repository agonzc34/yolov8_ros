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

#include "yolo_cpp_ros/yolo/segment.hpp"
#include "yolo_cpp_ros/yolo/utils.hpp"
#include "yolo_msgs/msg/point2_d.hpp"
#include <cstdint>
#include <cstdio>
#include <opencv2/highgui.hpp>

namespace yolo_onnx {

YoloSegment::YoloSegment(yolo_onnx_utils::YoloParams params) : Model(params) {}

YoloSegment::~YoloSegment() {}

std::vector<yolo_msgs::msg::Detection>
YoloSegment::postprocess(const cv::Size &original_image_size,
                         const cv::Size &resized_image_size,
                         const std::vector<Ort::Value> &preds) {
  std::vector<yolo_msgs::msg::Detection> detection_array;
  std::vector<yolo_onnx_utils::BoxWithMask> bbox_array;

  std::vector<int64_t> box_shape =
      preds[0].GetTensorTypeAndShapeInfo().GetShape();
  // Process predictions applying NMS
  const size_t num_features = box_shape[1];
  const int num_classes = static_cast<int>(num_features) - 4 - 32;

  bbox_array = yolo_onnx_utils::get_segmentation_with_nms(
      preds, box_shape, original_image_size, resized_image_size, num_classes,
      this->iou_threshold, this->conf_threshold);

  std::vector<int64_t> mask_shape =
      preds[1].GetTensorTypeAndShapeInfo().GetShape(); // [1, 32, maskH, maskW]
  
  const int maskH = static_cast<int>(mask_shape[2]);
  const int maskW = static_cast<int>(mask_shape[3]);

  std::vector<cv::Mat> mask_protos;
  const float* mask_ptr = preds[1].GetTensorData<float>();
  for (int64_t i = 0; i < mask_shape[1]; ++i) {
    cv::Mat mask(maskH, maskW, CV_32F, const_cast<float*>(mask_ptr + i * maskH * maskW));
    mask_protos.push_back(mask);
  }

  std::vector<cv::Mat> masks;
  for (size_t i = 0; i < bbox_array.size(); ++i) {
    const auto mask_coeffs = bbox_array[i].mask_coeffs;
    cv::Mat finalMask = cv::Mat::zeros(maskH, maskW, CV_32F);

    for (int m = 0; m < 32; ++m) {
      finalMask += mask_coeffs[m] * mask_protos[m];
    }

    cv::exp(-finalMask, finalMask); // Apply sigmoid activation
    finalMask = 1.0 / (1.0 + finalMask); // Sigmoid activation

    cv::Mat binaryMask;
    cv::threshold(finalMask, binaryMask, 0.5, 255.0, cv::THRESH_BINARY);
    binaryMask.convertTo(binaryMask, CV_8U);

    // Rescale the mask to the original image size
    cv::Mat resizedMask;
    cv::resize(binaryMask, resizedMask, original_image_size);

    cv::imshow("Binary Mask", resizedMask);
    cv::waitKey(0);
  
    // cv::Mat edges;
    // cv::Canny(binaryMask, edges, 100, 200);

    // masks.push_back(edges);
  }

  for (size_t i = 0; i < bbox_array.size(); ++i) {
    yolo_msgs::msg::Detection detection;
    detection.bbox = yolo_onnx_utils::convert_to_bounding_box(bbox_array[i]);
  
    // Segmentation mask
    detection.mask.height = original_image_size.height;
    detection.mask.width = original_image_size.width;
    detection.mask.data.clear();
    // for (int j = 0; j < original_image_size.height; ++j) {
    //   for (int k = 0; k < original_image_size.width; ++k) {
    //     if (masks[i].at<uchar>(j, k) > 0) {
    //       yolo_msgs::msg::Point2D mask_point;
    //       mask_point.x = k;
    //       mask_point.y = j;
    //       detection.mask.data.push_back(mask_point);
    //     }
    //   }
    // }

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
} // namespace yolo_onnx