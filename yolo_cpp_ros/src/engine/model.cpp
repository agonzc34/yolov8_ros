// Copyright (c) 2025 Alejandro González Cantón
// SPDX-License-Identifier: MIT

// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.

// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.

#include "yolo_cpp_ros/engine/model.hpp"
#include <fstream>
#include <iostream>
#include <numeric>
#include <thread>
#include "yolo_cpp_ros/yolo/utils.hpp"

namespace yolo_onnx {
Model::Model(yolo_onnx_utils::YoloParams params)
    : env(ORT_LOGGING_LEVEL_WARNING, "yolo"), session_options(),
      input_image_shape(), num_input_nodes(0),
      num_output_nodes(0), memory_info(nullptr) {
  // Initialize session options
  int n_threads = params.n_threads;
  this->conf_threshold = params.threshold;
  this->iou_threshold = params.iou;
  std::string model_path = params.model_path;

  if (n_threads == -1) {
    n_threads = std::thread::hardware_concurrency();
  }

  this->session_options.SetGraphOptimizationLevel(
      GraphOptimizationLevel::ORT_ENABLE_ALL);

  auto providers = Ort::GetAvailableProviders();
  bool use_cuda =
      std::find(providers.begin(), providers.end(), "CUDAExecutionProvider") !=
      providers.end();

  if (use_cuda) {
    // The GPU graph does the compute: a single intra-op thread avoids the
    // overhead of spinning up a full CPU thread-pool that mostly idles on the
    // CUDA stream.
    this->session_options.SetIntraOpNumThreads(1);

    // Tuned CUDA EP: heuristic conv-algo search (skips the expensive per-shape
    // exhaustive benchmark at session load) and same-as-requested arena growth
    // (less GPU memory over-allocation).
    OrtCUDAProviderOptionsV2 *cuda_options = nullptr;
    Ort::GetApi().CreateCUDAProviderOptions(&cuda_options);
    std::vector<const char *> keys = {
        "device_id", "arena_extend_strategy", "cudnn_conv_algo_search"};
    std::vector<const char *> values = {"0", "kSameAsRequested", "HEURISTIC"};
    Ort::GetApi().UpdateCUDAProviderOptions(cuda_options, keys.data(),
                                            values.data(), keys.size());
    this->session_options.AppendExecutionProvider_CUDA_V2(*cuda_options);
    Ort::GetApi().ReleaseCUDAProviderOptions(cuda_options);
    std::cout << "CUDA Execution Provider has been added (tuned)."
              << std::endl;
  } else {
    this->session_options.SetIntraOpNumThreads(n_threads);
    std::cout << "CUDA Execution Provider is not available." << std::endl;
  }

  Ort::AllocatorWithDefaultOptions allocator;

  this->session =
      Ort::Session(this->env, model_path.c_str(), this->session_options);

  // Get input and output node information
  this->num_input_nodes = this->session.GetInputCount();
  this->num_output_nodes = this->session.GetOutputCount();

  // Allocate input and output node names
  for (size_t i = 0; i < this->num_input_nodes; i++) {
    this->input_node_name_alloc_strings.push_back(
        this->session.GetInputNameAllocated(i, allocator));
    this->inputNames.push_back(
        this->input_node_name_alloc_strings.back().get());
  }

  for (size_t i = 0; i < this->num_output_nodes; i++) {
    this->output_node_name_alloc_strings.push_back(
        this->session.GetOutputNameAllocated(i, allocator));
    this->outputNames.push_back(
        this->output_node_name_alloc_strings.back().get());
  }

  Ort::TypeInfo input_type_info = this->session.GetInputTypeInfo(0);
  std::vector<int64_t> input_tensor_shape_vec =
      input_type_info.GetTensorTypeAndShapeInfo().GetShape();

  if (input_tensor_shape_vec.size() >= 4) {
    this->input_image_shape = cv::Size(static_cast<int>(input_tensor_shape_vec[3]),
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

  // Load class names from coco.names file
  std::ifstream class_names_file("src/yolov8_ros/yolo_cpp_ros/conf/coco.names"); // TODO: change this path
  if (!class_names_file.is_open()) {
    std::cerr << "Error: Could not open coco.names file." << std::endl;
    return;
  }
  std::string line;
  while (std::getline(class_names_file, line)) {
    this->class_names.push_back(line);
  }

  this->memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

  std::cout << "Model " << model_path << " has been successfully loaded."
            << std::endl;
}

Model::~Model() {}

std::vector<yolo_msgs::msg::Detection>
yolo_onnx::Model::detect(const cv::Mat &image) {
  std::vector<int64_t> input_tensor_shape = {1, 3, this->input_image_shape.height,
                                           this->input_image_shape.width};
  preprocess(image, input_tensor_shape);
  auto preds = inference(input_tensor_shape);
  return postprocess(cv::Size(image.cols, image.rows), this->input_image_shape,
                     preds);
}

void yolo_onnx::Model::preprocess(const cv::Mat &image,
                                  std::vector<int64_t> &input_tensor_shape) {
  cv::Mat resized_image =
    yolo_onnx_utils::letterbox(image, cv::Size(input_tensor_shape[3], input_tensor_shape[2]), cv::Scalar(114, 114, 114));

  // Normalize to float (OpenCV-optimized), then split channels straight into
  // the persistent buffer. This keeps the fast SIMD convertTo+split path while
  // reusing the buffer (no per-frame new[]/copy as in the original).
  resized_image.convertTo(resized_image, CV_32FC3, 1.0 / 255.0);
  const int H = static_cast<int>(input_tensor_shape[2]);
  const int W = static_cast<int>(input_tensor_shape[3]);
  std::vector<cv::Mat> chw(resized_image.channels());
  for (int i = 0; i < resized_image.channels(); ++i) {
    chw[i] = cv::Mat(H, W, CV_32FC1,
                     input_buffer_.data() + i * H * W);
  }
  cv::split(resized_image, chw);  // Split channels into the persistent blob
}

std::vector<Ort::Value>
yolo_onnx::Model::inference(std::vector<int64_t> &input_tensor_shape) {
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
  return std::vector<yolo_msgs::msg::Detection>();
}

} // namespace yolo_onnx
