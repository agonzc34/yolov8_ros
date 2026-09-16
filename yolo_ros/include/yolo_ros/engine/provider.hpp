// Copyright (c) 2026 Alejandro González Cantón
// SPDX-License-Identifier: MIT

/// @file
/// @brief ONNX Runtime session-option building for the YOLO engine. This branch
/// targets the aarch64/Jetson robot (ONNX Runtime 1.6) and always appends the
/// TensorRT execution provider; the CPU EP is ORT's implicit fallback.

#ifndef YOLO_ROS__ENGINE__PROVIDER_HPP_
#define YOLO_ROS__ENGINE__PROVIDER_HPP_

#include <onnxruntime_cxx_api.h>

#include <string>

/// @addtogroup yolo_engine
/// @{
namespace yolo_ros::engine {

/// @brief Parse the device ordinal from an ultralytics-style `device` string.
/// @param[in] device e.g. "cuda:0", "trt:1", "1", "cpu".
/// @return The ordinal, or 0 when absent, negative or invalid.
int parse_device_id(const std::string &device);

/// @brief Whether the linked ONNX Runtime build exposes the TensorRT EP.
/// @return True when "TensorrtExecutionProvider" is reported as available.
bool tensorrt_available();

/// @brief Build session options with the TensorRT execution provider appended.
/// @param[in] device_id TensorRT device ordinal.
/// @return The configured session options.
/// @throws Ort::Exception if the provider cannot be appended.
Ort::SessionOptions build_session_options(int device_id);

} // namespace yolo_ros::engine
/// @}

#endif // YOLO_ROS__ENGINE__PROVIDER_HPP_
