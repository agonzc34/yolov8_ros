// Copyright (c) 2025 Alejandro González Cantón
// SPDX-License-Identifier: MIT

#include "yolo_ros/engine/model.hpp"
#include "onnxruntime_cxx_api.h"
#include "yolo_ros/engine/provider.hpp"
#include "yolo_ros/yolo/utils.hpp"
#include <algorithm>
#include <ament_index_cpp/get_package_prefix.hpp>
#include <ament_index_cpp/get_package_share_directory.hpp>
#include <cctype>
#include <fstream>
#include <iostream>
#include <map>
#include <numeric>
#include <regex>
#include <stdexcept>
#include <thread>

namespace yolo_ros::engine {
Model::Model(yolo_ros::yolo::utils::YoloParams params)
    : env(ORT_LOGGING_LEVEL_WARNING, "yolo"), session_options(),
      input_image_shape(), num_input_nodes(0), num_output_nodes(0),
      memory_info(nullptr) {
  this->conf_threshold = params.threshold;
  this->iou_threshold = params.iou;
  std::string model_path = params.model_path;

  // The TensorRT EP is the only GPU provider on this branch (no provider
  // selection). CPU-only builds (smoke tests) set YOLO_ORT_USE_TENSORRT=0.
  if (kTensorrtEnabled && !tensorrt_available()) {
    throw std::runtime_error(
        "The linked ONNX Runtime build has no TensorRT execution provider. "
        "Rebuild ONNX Runtime 1.6 with --use_tensorrt.");
  }

  const int device_id = parse_device_id(params.device);
  this->session_options = build_session_options(device_id);
  this->session =
      Ort::Session(this->env, model_path.c_str(), this->session_options);
  if (kTensorrtEnabled) {
    this->active_provider_ = "tensorrt";
    std::cout << "Using execution provider: tensorrt (device " << device_id
              << ")" << std::endl;
  } else {
    this->active_provider_ = "cpu";
    std::cout << "Using execution provider: cpu (TensorRT disabled at build "
                 "time)"
              << std::endl;
  }

  Ort::AllocatorWithDefaultOptions allocator;

  // Get input and output node information
  this->num_input_nodes = this->session.GetInputCount();
  this->num_output_nodes = this->session.GetOutputCount();

  // Allocate input and output node names. ORT 1.6 returns a char* the caller
  // owns; copy into std::string storage and free it immediately.
  for (size_t i = 0; i < this->num_input_nodes; i++) {
    char *name = this->session.GetInputName(i, allocator);
    this->input_name_storage_.emplace_back(name);
    allocator.Free(name);
  }
  for (const std::string &name : this->input_name_storage_) {
    this->inputNames.push_back(name.c_str());
  }
  for (size_t i = 0; i < this->num_output_nodes; i++) {
    char *name = this->session.GetOutputName(i, allocator);
    this->output_name_storage_.emplace_back(name);
    allocator.Free(name);
  }
  for (const std::string &name : this->output_name_storage_) {
    this->outputNames.push_back(name.c_str());
  }

  Ort::TypeInfo input_type_info = this->session.GetInputTypeInfo(0);
  std::vector<int64_t> input_tensor_shape_vec =
      input_type_info.GetTensorTypeAndShapeInfo().GetShape();

  if (input_tensor_shape_vec.size() >= 4) {
    this->input_image_shape =
        cv::Size(static_cast<int>(input_tensor_shape_vec[3]),
                 static_cast<int>(input_tensor_shape_vec[2]));
    if (input_tensor_shape_vec[2] == -1 && input_tensor_shape_vec[3] == -1) {
      this->input_image_shape = cv::Size(640, 480); // Default size
    }
  } else {
    throw std::runtime_error("Invalid input tensor shape.");
  }

  // Pre-allocate the reusable input blob (1*3*H*W floats), written in place by
  // preprocess() and fed directly to ORT with no per-frame copy.
  this->input_buffer_.resize(
      static_cast<size_t>(this->input_image_shape.height) *
      this->input_image_shape.width * 3);

  // Load the class names (ONNX graph metadata first, coco.names as fallback).
  this->load_class_names();

  // Resolve the input channel order: parameter first, then the ONNX metadata
  // ("input_color") which our export tool stamps. Graphs cannot encode it.
  {
    std::string color = params.input_color;
    std::transform(color.begin(), color.end(), color.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    this->input_is_rgb_ = true; // ultralytics exports expect RGB
    if (color == "bgr") {
      this->input_is_rgb_ = false;
    } else if (!color.empty() && color != "rgb") {
      std::cerr << "Unknown input_color \"" << params.input_color
                << "\"; using rgb." << std::endl;
    }
    try {
      const Ort::ModelMetadata metadata = this->session.GetModelMetadata();
      Ort::AllocatorWithDefaultOptions allocator;
      char *value = metadata.LookupCustomMetadataMap("input_color", allocator);
      if (value != nullptr) {
        std::string meta(value);
        allocator.Free(value);
        std::transform(meta.begin(), meta.end(), meta.begin(),
                       [](unsigned char c) { return std::tolower(c); });
        if (meta == "rgb") {
          this->input_is_rgb_ = true;
        } else if (meta == "bgr") {
          this->input_is_rgb_ = false;
        }
      }
    } catch (const Ort::Exception &e) {
      std::cerr << "Warning: could not read \"input_color\" from the ONNX "
                   "metadata: "
                << e.what() << std::endl;
    }
    std::cout << "Input channel order: "
              << (this->input_is_rgb_ ? "rgb" : "bgr") << std::endl;
  }

  this->memory_info =
      Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

  std::cout << "Model " << model_path << " has been successfully loaded"
            << " (input " << this->input_image_shape.width << "x"
            << this->input_image_shape.height << ", FP32)." << std::endl;
}

Model::~Model() {}

void yolo_ros::engine::Model::load_class_names() {
  // 1) Read the vocabulary embedded in the ONNX graph by ultralytics'
  //    exporter: key "names", value a Python-dict literal such as
  //    {0: 'person', 1: 'bicycle', ...} (some exporters write JSON-style
  //    {"0": "person", ...}; both are handled here).
  try {
    const Ort::ModelMetadata metadata = this->session.GetModelMetadata();
    Ort::AllocatorWithDefaultOptions allocator;
    char *names_value = metadata.LookupCustomMetadataMap("names", allocator);
    if (names_value != nullptr) {
      const std::string names(names_value);
      allocator.Free(names_value);
      std::map<int, std::string> indexed_names;
      int max_index = -1;
      const std::regex name_re(R"((\d+):\s*['"]([^'"]*)['"])");
      auto begin = names.cbegin();
      const auto end = names.cend();
      std::smatch match;
      while (std::regex_search(begin, end, match, name_re)) {
        const int idx = std::stoi(match[1].str());
        indexed_names[idx] = match[2].str();
        max_index = std::max(max_index, idx);
        begin = match.suffix().first;
      }
      if (max_index >= 0) {
        this->class_names.assign(static_cast<size_t>(max_index + 1), "");
        for (const auto &[idx, name] : indexed_names) {
          this->class_names[static_cast<size_t>(idx)] = name;
        }
        std::cout << "Loaded " << this->class_names.size()
                  << " class names from the ONNX metadata." << std::endl;
      }
    }
  } catch (const Ort::Exception &e) {
    std::cerr << "Warning: could not read \"names\" from the ONNX metadata: "
              << e.what() << std::endl;
  }

  // 2) Fall back to the coco.names file (previous behaviour) when the model
  //    does not carry a vocabulary. The file is installed into the package
  //    share directory, so resolve it through the ament index instead of a
  //    hardcoded relative path (which only worked from the workspace root).
  if (this->class_names.empty()) {
    std::string class_names_path;
    try {
      class_names_path =
          ament_index_cpp::get_package_share_directory("yolo_ros") +
          "/conf/coco.names";
    } catch (const ament_index_cpp::PackageNotFoundError &e) {
      std::cerr << "Error: could not locate yolo_ros share directory: "
                << e.what() << std::endl;
      return;
    }
    std::ifstream class_names_file(class_names_path);
    if (!class_names_file.is_open()) {
      std::cerr << "Error: Could not open coco.names file at "
                << class_names_path << std::endl;
      return;
    }
    std::string line;
    while (std::getline(class_names_file, line)) {
      this->class_names.push_back(line);
    }
  }
}

std::vector<yolo_msgs::msg::Detection>
yolo_ros::engine::Model::detect(const cv::Mat &image) {
  std::vector<int64_t> input_tensor_shape = {
      1, 3, this->input_image_shape.height, this->input_image_shape.width};
  preprocess(image, input_tensor_shape);
  auto preds = inference(input_tensor_shape);
  return postprocess(cv::Size(image.cols, image.rows), this->input_image_shape,
                     preds);
}

void yolo_ros::engine::Model::preprocess(
    const cv::Mat &image, std::vector<int64_t> &input_tensor_shape) {
  cv::Mat resized_image = yolo_ros::yolo::utils::letterbox(
      image, cv::Size(input_tensor_shape[3], input_tensor_shape[2]),
      cv::Scalar(114, 114, 114));

  if (this->input_is_rgb_) {
    resized_image = yolo_ros::yolo::utils::bgr_to_rgb(resized_image);
  }

  // Normalize to float (OpenCV-optimized), then split channels straight into
  // the persistent buffer. This keeps the fast SIMD convertTo+split path while
  // reusing the buffer (no per-frame new[]/copy as in the original).
  resized_image.convertTo(resized_image, CV_32FC3, 1.0 / 255.0);
  const int H = static_cast<int>(input_tensor_shape[2]);
  const int W = static_cast<int>(input_tensor_shape[3]);
  std::vector<cv::Mat> chw(resized_image.channels());
  for (int i = 0; i < resized_image.channels(); ++i) {
    chw[i] = cv::Mat(H, W, CV_32FC1, input_buffer_.data() + i * H * W);
  }
  cv::split(resized_image, chw); // Split channels into the persistent blob
}

std::vector<Ort::Value>
yolo_ros::engine::Model::inference(std::vector<int64_t> &input_tensor_shape) {
  size_t input_tensor_size =
      std::accumulate(input_tensor_shape.begin(), input_tensor_shape.end(), 1,
                      std::multiplies<int64_t>());

  // Reuse the persistent buffer as the (CPU-owned) input tensor — no copy.
  Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
      this->memory_info, this->input_buffer_.data(), input_tensor_size,
      input_tensor_shape.data(), input_tensor_shape.size());

  std::vector<Ort::Value> predictions = this->session.Run(
      Ort::RunOptions{nullptr}, this->inputNames.data(), &input_tensor,
      this->num_input_nodes, this->outputNames.data(), this->num_output_nodes);

  return predictions;
}

std::vector<yolo_msgs::msg::Detection>
Model::postprocess(const cv::Size &original_image_size,
                   const cv::Size &resized_image_size,
                   const std::vector<Ort::Value> &preds) {
  (void)original_image_size;
  (void)resized_image_size;
  (void)preds;
  return std::vector<yolo_msgs::msg::Detection>();
}

} // namespace yolo_ros::engine
