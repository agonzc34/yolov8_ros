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

#include "yolo_cpp_ros/yolo/utils.hpp"
#include <algorithm>
#include <cstdio>

namespace yolo_onnx_utils {
cv::Mat letterbox(const cv::Mat &img, const cv::Size &new_shape,
                  const cv::Scalar &color) {
  float ratio = std::min(static_cast<float>(new_shape.width) / img.cols,
                         static_cast<float>(new_shape.height) / img.rows);
  cv::Mat img_out;

  int new_width = static_cast<int>(img.cols * ratio);
  int new_height = static_cast<int>(img.rows * ratio);

  int pad_w = new_shape.width - new_width;
  int pad_h = new_shape.height - new_height;

  int pad_left = pad_w / 2;
  int pad_right = pad_w - pad_left;
  int pad_top = pad_h / 2;
  int pad_bottom = pad_h - pad_top;

  cv::resize(img, img_out, cv::Size(new_width, new_height), 0, 0,
             cv::INTER_LINEAR);
  cv::copyMakeBorder(img_out, img_out, pad_top, pad_bottom, pad_left, pad_right,
                     cv::BORDER_CONSTANT, color);
  return img_out;
}

float iou(const std::shared_ptr<yolo_onnx_utils::Box> &box1, const std::shared_ptr<yolo_onnx_utils::Box> &box2) {
  float x1 = std::max(box1->x1, box2->x1);
  float y1 = std::max(box1->y1, box2->y1);
  float x2 = std::min(box1->x2, box2->x2);
  float y2 = std::min(box1->y2, box2->y2);

  float intersection = std::max(0.0f, x2 - x1) * std::max(0.0f, y2 - y1);
  float area1 =
      std::max(0.0f, box1->x2 - box1->x1) * std::max(0.0f, box1->y2 - box1->y1);
  float area2 =
      std::max(0.0f, box2->x2 - box2->x1) * std::max(0.0f, box2->y2 - box2->y1);
  float union_area = area1 + area2 - intersection;

  return union_area > 0 ? intersection / union_area : 0;
}

std::vector<int>
nms(std::vector<std::shared_ptr<yolo_onnx_utils::Box>> &boxes, float iou_threshold,
    float conf_threshold) // NMS implementation based on class_id
{
  std::vector<int> indices;

  std::sort(boxes.begin(), boxes.end(),
            [](const std::shared_ptr<yolo_onnx_utils::Box> &a, const std::shared_ptr<yolo_onnx_utils::Box> &b) {
              return a->score > b->score;
            });

  std::vector<bool> suppressed(boxes.size(), false);

  for (size_t i = 0; i < boxes.size(); ++i) {
    if (boxes[i]->score < conf_threshold) {
      suppressed[i] = true;
      continue;
    }

    if (suppressed[i] == true) {
      continue;
    }

    indices.push_back(i);

    for (size_t j = i + 1; j < boxes.size(); j++) {
      if (suppressed[j] == true || boxes[j]->class_id != boxes[i]->class_id) {
        continue;
      }

      float iou_value = yolo_onnx_utils::iou(boxes[i], boxes[j]);
      if (iou_value > iou_threshold) {
        suppressed[j] = true;
      }
    }
  }

  return indices;
}

yolo_onnx_utils::Box scale_box(const yolo_onnx_utils::Box &box,
                               const cv::Size &original_image_size,
                               const cv::Size &resized_image_size) {
  yolo_onnx_utils::Box scaled_box;
  float gain = std::min(static_cast<float>(resized_image_size.width) /
                            original_image_size.width,
                        static_cast<float>(resized_image_size.height) /
                            original_image_size.height);
  float pad_x =
      (resized_image_size.width - original_image_size.width * gain) / 2;
  float pad_y =
      (resized_image_size.height - original_image_size.height * gain) / 2;

  scaled_box.x1 = (box.x1 - pad_x) / gain;
  scaled_box.y1 = (box.y1 - pad_y) / gain;
  scaled_box.x2 = (box.x2 - pad_x) / gain;
  scaled_box.y2 = (box.y2 - pad_y) / gain;

  scaled_box.x1 = std::clamp(scaled_box.x1, 0.0f,
                             static_cast<float>(original_image_size.width));
  scaled_box.y1 = std::clamp(scaled_box.y1, 0.0f,
                             static_cast<float>(original_image_size.height));
  scaled_box.x2 = std::clamp(scaled_box.x2, 0.0f,
                             static_cast<float>(original_image_size.width));
  scaled_box.y2 = std::clamp(scaled_box.y2, 0.0f,
                             static_cast<float>(original_image_size.height));

  scaled_box.score = box.score;
  scaled_box.class_id = box.class_id;
  scaled_box.index = box.index;

  return scaled_box;
}

std::vector<yolo_onnx_utils::Box> get_detection_without_nms(
    const std::vector<Ort::Value> &preds, std::vector<int64_t> shape,
    const cv::Size &original_image_size, const cv::Size &resized_image_size) {
  std::vector<yolo_onnx_utils::Box> boxes;
  for (size_t i = 0; i < preds.size(); ++i) {
    auto pred = preds[i].GetTensorData<float>();
    for (size_t j = 0; j < static_cast<size_t>(shape[0]); ++j) {
      yolo_onnx_utils::Box box;
      box.x1 = pred[j * 6 + 0];
      box.y1 = pred[j * 6 + 1];
      box.x2 = pred[j * 6 + 2];
      box.y2 = pred[j * 6 + 3];
      box.score = pred[j * 6 + 4];
      box.class_id = static_cast<int>(pred[j * 6 + 5]);
      box.index = j;

      yolo_onnx_utils::Box scaled_box = yolo_onnx_utils::scale_box(
          box, original_image_size, resized_image_size);
      boxes.push_back(scaled_box);
    }
  }

  return boxes;
}

std::vector<yolo_onnx_utils::Box> get_detection_with_nms(
    const std::vector<Ort::Value> &preds, std::vector<int64_t> shape,
    const cv::Size &original_image_size, const cv::Size &resized_image_size,
    const int num_classes, float iou_threshold, float conf_threshold) {

  std::vector<yolo_onnx_utils::Box> boxes =
      get_boxes(preds, shape, original_image_size, resized_image_size,
                num_classes, conf_threshold);

  auto boxes_ptr = std::vector<std::shared_ptr<yolo_onnx_utils::Box>>();
  for (size_t i = 0; i < boxes.size(); ++i) {
    boxes_ptr.push_back(std::make_shared<yolo_onnx_utils::Box>(boxes[i]));
  }

  auto indices = yolo_onnx_utils::nms(boxes_ptr, iou_threshold, conf_threshold);
  std::vector<yolo_onnx_utils::Box> filtered_boxes;
  for (size_t i = 0; i < indices.size(); ++i) {
    filtered_boxes.push_back(boxes[indices[i]]);
  }
  return filtered_boxes;
}

std::vector<yolo_onnx_utils::BoxWithMask> get_segmentation_with_nms(
    const std::vector<Ort::Value> &preds, std::vector<int64_t> shape,
    const cv::Size &original_image_size, const cv::Size &resized_image_size,
    const int num_classes, float iou_threshold, float conf_threshold) {

  const float *raw_output =
      preds[0].GetTensorData<float>(); // Extract raw output data from the
  const size_t num_detections = shape[2];
  const float *ptr = raw_output;
  std::vector<yolo_onnx_utils::BoxWithMask> seg_boxes;

  // 1. Get the bounding boxes
  std::vector<yolo_onnx_utils::Box> boxes =
      get_boxes(preds, shape, original_image_size, resized_image_size,
                num_classes, conf_threshold);

  // 2. Add the mask coefficients to the boxes
  std::vector<yolo_onnx_utils::BoxWithMask> boxes_with_mask;
  for (size_t i = 0; i < boxes.size(); ++i) {
    yolo_onnx_utils::BoxWithMask box_with_mask(boxes[i]);
    std::vector<float> mask_coeffs(32);
    for (int m = 0; m < 32; ++m) {
      mask_coeffs[m] = ptr[(num_classes + 4 + m) * num_detections + i];
    }
    box_with_mask.mask_coeffs = std::move(mask_coeffs);
    boxes_with_mask.push_back(box_with_mask);
  }

  std::vector<std::shared_ptr<yolo_onnx_utils::Box>> boxes_ptr;
  for (size_t i = 0; i < boxes_with_mask.size(); ++i) {
    boxes_ptr.push_back(std::make_shared<yolo_onnx_utils::Box>(
        boxes_with_mask[i]));
  }

  // 3. Apply NMS
  auto indices = yolo_onnx_utils::nms(boxes_ptr, iou_threshold, conf_threshold);

  std::vector<yolo_onnx_utils::BoxWithMask> filtered_boxes;
  for (size_t i = 0; i < indices.size(); ++i) {
    filtered_boxes.push_back(boxes_with_mask[indices[i]]);
  }

  return filtered_boxes;
}

std::vector<yolo_onnx_utils::Box>
get_boxes(const std::vector<Ort::Value> &preds, std::vector<int64_t> shape,
          const cv::Size &original_image_size,
          const cv::Size &resized_image_size, const int num_classes,
          float conf_threshold) {
  std::vector<yolo_onnx_utils::Box> boxes;

  const float *raw_output =
      preds[0].GetTensorData<float>(); // Extract raw output data from the
                                       // first output tensor
  const size_t num_detections = shape[2];

  const float *ptr = raw_output;
  for (size_t i = 0; i < num_detections; ++i) {
    yolo_onnx_utils::Box box;
    float center_x = ptr[0 * num_detections + i];
    float center_y = ptr[1 * num_detections + i];
    float width = ptr[2 * num_detections + i];
    float height = ptr[3 * num_detections + i];

    int class_id = -1;
    float max_score = -1.0f;

    for (int j = 0; j < num_classes; ++j) {
      float score = ptr[(4 + j) * num_detections + i];
      if (score > max_score) {
        max_score = score;
        class_id = j;
      }
    }

    if (max_score > conf_threshold) {
      box.x1 = (center_x - width / 2);
      box.y1 = (center_y - height / 2);
      box.x2 = (center_x + width / 2);
      box.y2 = (center_y + height / 2);
      box.score = max_score;
      box.class_id = class_id;

      yolo_onnx_utils::Box scaled_box = yolo_onnx_utils::scale_box(
          box, original_image_size, resized_image_size);
      boxes.push_back(scaled_box);
    }
  }

  return boxes;
}

yolo_msgs::msg::BoundingBox2D
convert_to_bounding_box(const yolo_onnx_utils::Box &box) {
  yolo_msgs::msg::BoundingBox2D bounding_box;
  bounding_box.center.position.x = (box.x1 + box.x2) / 2;
  bounding_box.center.position.y = (box.y1 + box.y2) / 2;
  bounding_box.size.x = box.x2 - box.x1;
  bounding_box.size.y = box.y2 - box.y1;
  return bounding_box;
}
} // namespace yolo_onnx_utils