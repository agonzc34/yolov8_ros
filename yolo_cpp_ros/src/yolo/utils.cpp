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

namespace yolo_onnx_utils {
cv::Mat letterbox(const cv::Mat &img, const cv::Size &new_shape,
                  const cv::Scalar &color, bool auto_size) {
  float ratio = std::min(static_cast<float>(new_shape.width) / img.cols,
                         static_cast<float>(new_shape.height) / img.rows);
  cv::Mat img_out;

  int new_width = static_cast<int>(img.cols * ratio);
  int new_height = static_cast<int>(img.rows * ratio);

  int pad_w = new_shape.width - new_width;
  int pad_h = new_shape.height - new_height;

  if (auto_size) {
    pad_w = pad_w % 32;
    pad_h = pad_h % 32;
  }

  int pad_left = pad_w / 2;
  int pad_right = pad_w - pad_left;
  int pad_top = pad_h / 2;
  int pad_bottom = pad_h - pad_top;

	fprintf(stderr, "new_width: %d, new_height: %d, pad_left: %d, pad_right: %d, pad_top: %d, pad_bottom: %d\n", new_width, new_height, pad_left, pad_right, pad_top, pad_bottom);

  cv::resize(img, img_out, cv::Size(new_width, new_height), 0, 0,
             cv::INTER_LINEAR);
  cv::copyMakeBorder(img_out, img_out, pad_top, pad_bottom, pad_left, pad_right,
                     cv::BORDER_CONSTANT, color);
  return img_out;
}

float iou(const Box &box1, const Box &box2) {
  float x1 = std::max(box1.x1, box2.x1);
  float y1 = std::max(box1.y1, box2.y1);
  float x2 = std::min(box1.x2, box2.x2);
  float y2 = std::min(box1.y2, box2.y2);

  float intersection = std::max(0.0f, x2 - x1) * std::max(0.0f, y2 - y1);
  float area1 = (box1.x2 - box1.x1) * (box1.y2 - box1.y1);
  float area2 = (box2.x2 - box2.x1) * (box2.y2 - box2.y1);
  float unionArea = area1 + area2 - intersection;

  return intersection / unionArea;
}

std::vector<struct Box>
nms(std::vector<Box> &boxes, float iouThreshold,
    float confThreshold) // TODO: Change NMS implementation
{
  std::vector<Box> detections;

  std::sort(boxes.begin(), boxes.end(),
            [](const Box &a, const Box &b) { return a.score > b.score; });

  std::vector<bool> suppressed(boxes.size(), false);

  for (size_t i = 0; i < boxes.size(); ++i) {
    if (boxes[i].score < confThreshold) {
      suppressed[i] = true;
      continue;
    }

    detections.push_back(boxes[i]);

    for (size_t j = i + 1; j < boxes.size(); ++j) {
      if (suppressed[j]) {
        continue;
      }

      float iouValue = iou(boxes[i], boxes[j]);
      if (iouValue > iouThreshold) {
        suppressed[j] = true;
      }
    }
  }

  std::vector<Box> results;
  for (size_t i = 0; i < detections.size(); ++i) {
    if (!suppressed[i]) {
      results.push_back(detections[i]);
    }
  }

  return results;
}

Box scale_box(const Box &box, const cv::Size &originalImageSize,
              const cv::Size &resizedImageShape) {
  Box scaledBox;
  float gain = std::min(
      static_cast<float>(resizedImageShape.width) / originalImageSize.width,
      static_cast<float>(resizedImageShape.height) / originalImageSize.height);
  float pad_x = (resizedImageShape.width - originalImageSize.width * gain) / 2;
  float pad_y =
      (resizedImageShape.height - originalImageSize.height * gain) / 2;

  scaledBox.x1 = std::clamp((box.x1 - pad_x) / gain, 0.0f,
                            static_cast<float>(originalImageSize.width));
  scaledBox.y1 = std::clamp((box.y1 - pad_y) / gain, 0.0f,
                            static_cast<float>(originalImageSize.height));
  scaledBox.x2 = std::clamp((box.x2 - pad_x) / gain, 0.0f,
                            static_cast<float>(originalImageSize.width));
  scaledBox.y2 = std::clamp((box.y2 - pad_y) / gain, 0.0f,
                            static_cast<float>(originalImageSize.height));

  scaledBox.score = box.score;
  scaledBox.class_id = box.class_id;
  scaledBox.index = box.index;

  return scaledBox;
}
} // namespace yolo_onnx_utils