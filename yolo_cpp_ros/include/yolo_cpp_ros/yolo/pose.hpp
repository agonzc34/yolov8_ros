// Copyright (C) 2026 Alejandro González Cantón
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

#ifndef YOLO_CPP_ROS__YOLO__POSE_HPP_
#define YOLO_CPP_ROS__YOLO__POSE_HPP_

#include "yolo_cpp_ros/engine/model.hpp"
#include "yolo_cpp_ros/yolo/utils.hpp"
#include "yolo_msgs/msg/detection.hpp"
#include <vector>

namespace yolo_onnx {

class YoloPose : public Model {
public:
  YoloPose(yolo_onnx_utils::YoloParams params);
  ~YoloPose();

protected:
  virtual std::vector<yolo_msgs::msg::Detection>
  postprocess(const cv::Size &original_image_size,
              const cv::Size &resized_image_size,
              const std::vector<Ort::Value> &outputTensors) override;
};
} // namespace yolo_onnx
#endif // YOLO_CPP_ROS__YOLO__POSE_HPP_
