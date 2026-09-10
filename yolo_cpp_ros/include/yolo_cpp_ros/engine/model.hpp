// Copyright (c) 2025 Alejandro González Cantón
// SPDX-License-Identifier: MIT

/// @file
/// @brief Base ONNX Runtime model: preprocessing, inference and the
/// postprocessing hook overridden by each YOLO task.

#ifndef YOLO_CPP_ROS__ENGINE__MODEL_HPP_
#define YOLO_CPP_ROS__ENGINE__MODEL_HPP_

#include "yolo_cpp_ros/yolo/utils.hpp"
#include "yolo_msgs/msg/detection.hpp"
#include <cv_bridge/cv_bridge.h>
#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <vector>

/// @addtogroup yolo_engine
/// @{
namespace yolo_onnx {
/// @brief Base class for a single ONNX Runtime YOLO model.
///
/// Owns the ONNX Runtime environment, session and the reusable input buffer.
/// The pipeline is fixed: preprocess() letterboxes the image into the model's
/// input tensor, inference() runs the session and postprocess() (virtual,
/// overridden by the concrete task classes) decodes the raw output tensors
/// into yolo_msgs::msg::Detection messages.
class Model {
public:
  /// @brief Load @p params.model_path (or download it from the Hugging Face
  /// Hub) and create the ONNX Runtime session, then read the class vocabulary.
  /// @param params Model, task and preprocessing configuration.
  Model(yolo_ros::yolo::utils::YoloParams params);
  /// @brief Destroy the model and release the ONNX Runtime session.
  ~Model();

  /// @brief Run the full pipeline (preprocess, inference, postprocess) on one
  /// image.
  /// @param image BGR image in original (un-letterboxed) coordinates.
  /// @return One detection per kept object, in original-image coordinates.
  std::vector<yolo_msgs::msg::Detection> detect(const cv::Mat &image);

  /// @brief Confidence threshold for detections, in [0, 1]. @see detect()
  float conf_threshold{0.5}; // Confidence threshold for detections
  /// @brief IoU threshold used by the C++ NMS, in [0, 1]. @see detect()
  float iou_threshold{0.5}; // Intersection over union threshold for detections

protected:
  /// @brief Class vocabulary indexed by class id. Populated from the ONNX
  /// graph metadata, or from the coco.names fallback. @see load_class_names()
  std::vector<std::string>
      class_names; // Vector of class names loaded from file

private:
  /// @brief Letterbox @p image into the model input size and fill the
  /// persistent CHW input buffer.
  /// @param[in] image BGR image to preprocess.
  /// @param[out] input_tensor_shape Shape of the produced input tensor.
  void preprocess(const cv::Mat &image,
                  std::vector<int64_t> &input_tensor_shape);
  /// @brief Run the ONNX Runtime session on the preprocessed input buffer.
  /// @param[in] input_tensor_shape Shape of the input tensor.
  /// @return The raw output tensors produced by the model.
  std::vector<Ort::Value> inference(std::vector<int64_t> &input_tensor_shape);
  /// @brief Decode the raw output tensors into detections. Overridden by each
  /// YOLO task; the base implementation is the identity.
  /// @param[in] original_image_size Size of the original camera image.
  /// @param[in] resized_image_size Size of the letterboxed network input.
  /// @param[in] preds Raw output tensors from inference().
  /// @return Detections in original-image coordinates.
  virtual std::vector<yolo_msgs::msg::Detection>
  postprocess(const cv::Size &original_image_size,
              const cv::Size &resized_image_size,
              const std::vector<Ort::Value> &preds);
  /// @brief Populate class_names from the ONNX graph metadata ("names"),
  /// falling back to the coco.names file when the model carries no vocabulary.
  void load_class_names();

  /// @brief ONNX Runtime environment.
  Ort::Env env{nullptr}; // ONNX Runtime environment
  /// @brief Session options for ONNX Runtime.
  Ort::SessionOptions session_options{
      nullptr}; // Session options for ONNX Runtime
  /// @brief ONNX Runtime inference session.
  Ort::Session session{nullptr}; // ONNX Runtime session for running inference
  /// @brief Expected input image shape for the model.
  cv::Size input_image_shape; // Expected input image shape for the model

  // Persistent input buffer (CHW, 1*3*H*W floats) reused every inference to
  // avoid re-allocating + copying the full blob per frame.
  /// @brief Reusable CHW input blob (1*3*H*W floats).
  std::vector<float> input_buffer_;

  // Vectors to hold allocated input and output node names
  /// @brief Allocated storage backing inputNames.
  std::vector<Ort::AllocatedStringPtr> input_node_name_alloc_strings;
  /// @brief Input node names passed to the ONNX Runtime session.
  std::vector<const char *> inputNames;
  /// @brief Allocated storage backing outputNames.
  std::vector<Ort::AllocatedStringPtr> output_node_name_alloc_strings;
  /// @brief Output node names requested from the ONNX Runtime session.
  std::vector<const char *> outputNames;

  /// @brief Number of input nodes in the model.
  size_t num_input_nodes;
  /// @brief Number of output nodes in the model.
  size_t num_output_nodes;
  /// @brief Memory information for ONNX Runtime tensor creation.
  Ort::MemoryInfo memory_info; // Memory information for ONNX Runtime
};
} // namespace yolo_onnx
/// @}

#endif // YOLO_CPP_ROS__ENGINE__MODEL_HPP_
