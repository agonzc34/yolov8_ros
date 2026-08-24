// Copyright (C) 2025 Alejandro González Cantón
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.

// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.

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

float iou(const std::shared_ptr<yolo_onnx_utils::Box> &box1,
          const std::shared_ptr<yolo_onnx_utils::Box> &box2) {
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
nms(std::vector<std::shared_ptr<yolo_onnx_utils::Box>> &boxes,
    float iou_threshold,
    float conf_threshold) // NMS implementation based on class_id
{
  std::vector<int> indices;

  std::sort(boxes.begin(), boxes.end(),
            [](const std::shared_ptr<yolo_onnx_utils::Box> &a,
               const std::shared_ptr<yolo_onnx_utils::Box> &b) {
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

std::vector<yolo_onnx_utils::Box>
get_boxes(const std::vector<Ort::Value> &preds,
          const cv::Size &original_image_size,
          const cv::Size &resized_image_size, const int num_classes) {
  std::vector<yolo_onnx_utils::Box> boxes;

  const float *raw_output =
      preds[0].GetTensorData<float>(); // Extract raw output data from the
                                       // first output tensor
  const size_t num_detections =
      preds[0].GetTensorTypeAndShapeInfo().GetShape()[2];

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

cv::Mat inverse_letterbox(
  const cv::Mat& letterboxed,
  const cv::Size &original_image_size,
  const cv::Size &resized_image_size
) {
  // Resize the (low-res) mask up to the letterboxed / model-input frame.
  cv::Mat resized_image;
  cv::resize(letterboxed, resized_image, resized_image_size, 0, 0,
             cv::INTER_LINEAR);

  // Size of the actual (unpadded) image content inside the letterboxed frame.
  float scale = std::min(
      static_cast<float>(resized_image_size.width) / original_image_size.width,
      static_cast<float>(resized_image_size.height) / original_image_size.height
  );
  int new_w = static_cast<int>(original_image_size.width * scale);
  int new_h = static_cast<int>(original_image_size.height * scale);

  int pad_x = (resized_image_size.width - new_w) / 2;
  int pad_y = (resized_image_size.height - new_h) / 2;

  // Crop only the unpadded content, then rescale back to the original size.
  cv::Rect content_roi(pad_x, pad_y, new_w, new_h);
  cv::Mat cropped = resized_image(content_roi);

  cv::Mat restored;
  cv::resize(cropped, restored, original_image_size, 0, 0,
             cv::INTER_LINEAR);

  return restored;
}
} // namespace yolo_onnx_utils