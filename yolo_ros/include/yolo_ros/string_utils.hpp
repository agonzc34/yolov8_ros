// Copyright (c) 2026 Alejandro González Cantón
// SPDX-License-Identifier: MIT

/// @file
/// @brief Lowercase helper shared by the parameter/device string parsing.

#ifndef YOLO_ROS__STRING_UTILS_HPP_
#define YOLO_ROS__STRING_UTILS_HPP_

#include <algorithm>
#include <cctype>
#include <string>

namespace yolo_ros {

/// @brief Return @p value lowercased (ASCII).
/// @param[in] value String to lowercase.
/// @return The lowercased copy.
inline std::string to_lower(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  return value;
}

} // namespace yolo_ros

#endif // YOLO_ROS__STRING_UTILS_HPP_
