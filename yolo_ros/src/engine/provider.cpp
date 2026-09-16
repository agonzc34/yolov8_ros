// Copyright (c) 2026 Alejandro González Cantón
// SPDX-License-Identifier: MIT

#include "yolo_ros/engine/provider.hpp"

#include <algorithm>
#include <cctype>
#include <charconv>
#include <string>
#include <system_error>
#include <vector>

#include "tensorrt_provider_factory.h"

namespace yolo_ros::engine {
namespace {

std::string to_lower(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  return value;
}

} // namespace

int parse_device_id(const std::string &device) {
  const std::string value = to_lower(device);
  const std::size_t colon = value.rfind(':');
  const std::string ordinal =
      colon == std::string::npos ? value : value.substr(colon + 1);
  int id = 0;
  const char *begin = ordinal.data();
  const char *end = begin + ordinal.size();
  const auto result = std::from_chars(begin, end, id);
  if (result.ec != std::errc() || result.ptr != end || id < 0) {
    return 0;
  }
  return id;
}

bool tensorrt_available() {
  for (const std::string &name : Ort::GetAvailableProviders()) {
    if (to_lower(name) == "tensorrtexecutionprovider") {
      return true;
    }
  }
  return false;
}

Ort::SessionOptions build_session_options(int device_id) {
  Ort::SessionOptions options;
  options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
  // The TensorRT EP owns all GPU execution; ORT's built-in CPU EP handles any
  // nodes TensorRT does not claim.
  options.SetIntraOpNumThreads(1);
  Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_Tensorrt(
      static_cast<OrtSessionOptions *>(options), device_id));
  return options;
}

} // namespace yolo_ros::engine
