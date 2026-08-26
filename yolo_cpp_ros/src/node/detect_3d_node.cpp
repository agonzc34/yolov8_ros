// Copyright (c) 2026 Alejandro González Cantón
// Portions Copyright (c) 2023-2025 Miguel Ángel González Santamarta
// SPDX-License-Identifier: MIT

#include "yolo_cpp_ros/node/detect_3d_node.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace yolo_rclcpp {

// ---------------------------------------------------------------------------
// Robust depth-statistics helpers, ported from the Python detect_3d_node.py
// (_compute_spatial_weights, _compute_depth_bounds_weighted,
// _compute_height_bounds, _compute_width_bounds). They turn the raw depth
// samples of a detection into a 3D box: a spatially weighted, trimmed-mean
// center and MAD/percentile-based extents per axis.
// ---------------------------------------------------------------------------
namespace {

double median(std::vector<double> v) {
  if (v.empty()) return 0.0;
  const size_t n = v.size();
  const size_t mid = n / 2;
  if (n % 2 == 1) {
    std::nth_element(v.begin(), v.begin() + mid, v.end());
    return v[mid];
  }
  // Even count: nth_element places one middle element at `mid`; the other
  // middle is the largest element of the lower half.
  std::nth_element(v.begin(), v.begin() + mid, v.end());
  const double hi = v[mid];
  const double lo = *std::max_element(v.begin(), v.begin() + mid);
  return 0.5 * (lo + hi);
}

// Linear-interpolation percentile on an already-sorted vector (q in [0,1]),
// matching numpy.percentile(..., method='linear').
double percentile_sorted(const std::vector<double> &v, double q) {
  if (v.empty()) return 0.0;
  const double pos = q * (static_cast<double>(v.size()) - 1.0);
  const size_t lo = static_cast<size_t>(std::floor(pos));
  const size_t hi = static_cast<size_t>(std::ceil(pos));
  if (lo == hi) return v[lo];
  const double frac = pos - lo;
  return v[lo] * (1.0 - frac) + v[hi] * frac;
}

// Left insertion index into a normalized, monotonically non-decreasing
// cumulative-weight array (np.searchsorted(cumsum, f, side='left')).
size_t weighted_searchsorted(const std::vector<double> &cum_weights, double f) {
  auto it = std::lower_bound(cum_weights.begin(), cum_weights.end(), f);
  return static_cast<size_t>(it - cum_weights.begin());
}

double weighted_mean(const std::vector<double> &v,
                     const std::vector<double> &w) {
  const double sw = std::accumulate(w.begin(), w.end(), 0.0);
  if (sw <= 0.0) return median(v);
  double acc = 0.0;
  for (size_t i = 0; i < v.size(); ++i) acc += v[i] * w[i];
  return acc / sw;
}

// Gaussian falloff from the bbox center, floored at 0.3
// (Python _compute_spatial_weights).
std::vector<double> compute_spatial_weights(const std::vector<int> &xs,
                                            const std::vector<int> &ys,
                                            int center_x, int center_y,
                                            int size_x, int size_y) {
  const double hx = size_x / 2.0 + 1e-6;
  const double hy = size_y / 2.0 + 1e-6;
  std::vector<double> w(xs.size());
  for (size_t i = 0; i < xs.size(); ++i) {
    const double dx = (xs[i] - center_x) / hx;
    const double dy = (ys[i] - center_y) / hy;
    const double d = std::sqrt(dx * dx + dy * dy);
    w[i] = std::max(std::exp(-0.5 * std::pow(d / 0.8, 2.0)), 0.3);
  }
  return w;
}

struct DepthBounds {
  double center;
  double min;
  double max;
};

// Weighted histogram peak + MAD/IQR adaptive filtering + trimmed weighted
// center and 1st/99th weighted percentiles (Python _compute_depth_bounds_weighted).
DepthBounds compute_depth_bounds_weighted(std::vector<double> depth,
                                          std::vector<double> weight) {
  std::vector<double> d, w;
  for (size_t i = 0; i < depth.size(); ++i) {
    if (std::isfinite(depth[i]) && std::isfinite(weight[i])) {
      d.push_back(depth[i]);
      w.push_back(weight[i]);
    }
  }
  if (d.empty()) return {0.0, 0.0, 0.0};
  if (d.size() < 4) {
    const auto mm = std::minmax_element(d.begin(), d.end());
    return {median(d), *mm.first, *mm.second};
  }

  // Weighted histogram + smoothing for robust mode detection.
  const double d_min = *std::min_element(d.begin(), d.end());
  const double d_max = *std::max_element(d.begin(), d.end());
  const double range = d_max - d_min;
  const int n_bins = (!std::isfinite(range) || range <= 0.0)
                         ? 30
                         : std::clamp(
                               static_cast<int>(std::lround(range / 0.01)), 20, 60);
  const double bin_w = (d_max - d_min) / n_bins;
  if (!(bin_w > 0.0)) {
    // Degenerate: all depths identical (zero-width histogram). The median
    // equals min and max, so return them directly.
    const auto mm = std::minmax_element(d.begin(), d.end());
    return {median(d), *mm.first, *mm.second};
  }
  std::vector<double> hist(n_bins, 0.0);
  for (size_t i = 0; i < d.size(); ++i) {
    const int idx = std::clamp(
        static_cast<int>((d[i] - d_min) / bin_w), 0, n_bins - 1);
    hist[idx] += w[i];
  }
  std::vector<double> smooth = hist;
  const int ks = std::min(5, n_bins / 4);
  if (ks >= 1) {
    const int half = ks / 2;
    for (int i = 0; i < n_bins; ++i) {
      double acc = 0.0;
      int cnt = 0;
      for (int k = -half; k <= half; ++k) {
        const int j = i + k;
        if (j >= 0 && j < n_bins) {
          acc += hist[j];
          ++cnt;
        }
      }
      smooth[i] = (cnt > 0) ? acc / cnt : hist[i];
    }
  }
  int peak = 0;
  for (int i = 1; i < n_bins; ++i) {
    if (smooth[i] > smooth[peak]) peak = i;
  }
  const double mode_depth = d_min + bin_w * (peak + 0.5);

  std::vector<double> dev(d.size());
  for (size_t i = 0; i < d.size(); ++i) dev[i] = std::abs(d[i] - mode_depth);
  const double mad = median(dev);
  std::vector<double> ds(d);
  std::sort(ds.begin(), ds.end());
  const double q25 = percentile_sorted(ds, 0.25);
  const double q75 = percentile_sorted(ds, 0.75);
  const double iqr = q75 - q25;

  double thr;
  if (iqr < 0.03) {
    thr = std::clamp(3.5 * mad, 0.08, 0.30);
  } else if (iqr < 0.10) {
    thr = std::clamp(4.0 * mad, 0.12, 0.40);
  } else {
    thr = std::clamp(5.0 * mad, 0.15, 0.60);
  }

  std::vector<double> obj_d, obj_w;
  for (size_t i = 0; i < d.size(); ++i) {
    if (std::abs(d[i] - mode_depth) <= thr) {
      obj_d.push_back(d[i]);
      obj_w.push_back(w[i]);
    }
  }
  const size_t min_points =
      std::max<size_t>(6, static_cast<size_t>(d.size() * 0.15));
  if (obj_d.size() < min_points) {
    // Fallback: weighted 2nd..85th percentile range.
    std::vector<size_t> idx(d.size());
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(),
              [&](size_t a, size_t b) { return d[a] < d[b]; });
    std::vector<double> cum_weights(d.size());
    double acc = 0.0;
    for (size_t i = 0; i < d.size(); ++i) {
      acc += w[idx[i]];
      cum_weights[i] = acc;
    }
    if (cum_weights.back() > 0.0) {
      for (auto &c : cum_weights) c /= cum_weights.back();
      const double p2v = d[idx[weighted_searchsorted(cum_weights, 0.02)]];
      const double p85v = d[idx[weighted_searchsorted(cum_weights, 0.85)]];
      obj_d.clear();
      obj_w.clear();
      for (size_t i = 0; i < d.size(); ++i) {
        if (d[i] >= p2v && d[i] <= p85v) {
          obj_d.push_back(d[i]);
          obj_w.push_back(w[i]);
        }
      }
    }
  }
  if (obj_d.empty()) {
    obj_d = d;
    obj_w = w;
  }

  // Sort the depth samples once and build the weighted cumulative array once;
  // both the 2%-trimmed center and the 1st/99th weighted percentiles read from
  // the same arrays (previously each block re-sorted identically).
  std::vector<size_t> idx(obj_d.size());
  std::iota(idx.begin(), idx.end(), 0);
  std::sort(idx.begin(), idx.end(),
            [&](size_t a, size_t b) { return obj_d[a] < obj_d[b]; });
  std::vector<double> cum_weights(obj_d.size());
  {
    double acc = 0.0;
    for (size_t i = 0; i < obj_d.size(); ++i) {
      acc += obj_w[idx[i]];
      cum_weights[i] = acc;
    }
    if (cum_weights.back() > 0.0) {
      for (auto &c : cum_weights) c /= cum_weights.back();
    }
  }

  double z_center;
  if (cum_weights.back() > 0.0) {
    const size_t lo = weighted_searchsorted(cum_weights, 0.02);
    const size_t hi = weighted_searchsorted(cum_weights, 0.98);
    if (hi > lo) {
      double sw = 0.0;
      double zc = 0.0;
      for (size_t i = lo; i < hi; ++i) {
        zc += obj_d[idx[i]] * obj_w[idx[i]];
        sw += obj_w[idx[i]];
      }
      z_center = (sw > 0.0) ? zc / sw : median(obj_d);
    } else {
      z_center = weighted_mean(obj_d, obj_w);
    }
  } else {
    z_center = median(obj_d);
  }

  double z_min, z_max;
  if (cum_weights.back() > 0.0) {
    z_min = obj_d[idx[weighted_searchsorted(cum_weights, 0.01)]];
    z_max = obj_d[idx[weighted_searchsorted(cum_weights, 0.99)]];
  } else {
    z_min = *std::min_element(obj_d.begin(), obj_d.end());
    z_max = *std::max_element(obj_d.begin(), obj_d.end());
  }
  if (z_center < z_min || z_center > z_max) {
    const double ext = std::max(z_max - z_min, 0.02);
    z_min = z_center - ext / 2.0;
    z_max = z_center + ext / 2.0;
  }
  if (z_max - z_min < 0.02) {
    z_min = z_center - 0.01;
    z_max = z_center + 0.01;
  }
  return {z_center, z_min, z_max};
}

struct AxisBounds {
  double center;
  double min;
  double max;
};

// Outlier-filtered (MAD), trimmed weighted center + 3rd/97th weighted
// percentiles for one 3D axis (Python _compute_height_bounds / _compute_width_bounds).
AxisBounds compute_axis_bounds(const std::vector<double> &val3,
                               const std::vector<double> &w, double mad_mult,
                               double clip_lo, double clip_hi) {
  const size_t n = val3.size();
  if (n < 4) {
    const auto mm = std::minmax_element(val3.begin(), val3.end());
    return {median(val3), *mm.first, *mm.second};
  }
  auto sorted_idx = [&]() {
    std::vector<size_t> idx(n);
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(),
              [&](size_t a, size_t b) { return val3[a] < val3[b]; });
    return idx;
  };

  // Weighted median as reference.
  const std::vector<size_t> idx = sorted_idx();
  std::vector<double> cum_weights(n, 0.0);
  {
    double acc = 0.0;
    for (size_t i = 0; i < n; ++i) {
      acc += w[idx[i]];
      cum_weights[i] = acc;
    }
    if (cum_weights.back() > 0.0) {
      for (auto &c : cum_weights) c /= cum_weights.back();
    }
  }
  const double med = val3[idx[weighted_searchsorted(cum_weights, 0.5)]];

  std::vector<double> dev(n);
  for (size_t i = 0; i < n; ++i) dev[i] = std::abs(val3[i] - med);
  const double mad = median(dev);
  const double thr = std::clamp(mad_mult * mad, clip_lo, clip_hi);

  std::vector<double> fv, fw;
  for (size_t i = 0; i < n; ++i) {
    if (dev[i] <= thr) {
      fv.push_back(val3[i]);
      fw.push_back(w[i]);
    }
  }
  const double min_pts = std::max(4.0, static_cast<double>(n) * 0.12);
  if (fv.size() < min_pts) {
    fv = val3;
    fw = w;
  }

  // One weighted sort serves both the 5%-trimmed center and the 3rd/97th
  // weighted percentiles (previously each block re-sorted identically).
  const size_t m = fv.size();
  std::vector<size_t> i2(m);
  std::iota(i2.begin(), i2.end(), 0);
  std::sort(i2.begin(), i2.end(),
            [&](size_t a, size_t b) { return fv[a] < fv[b]; });
  std::vector<double> c2(m, 0.0);
  {
    double acc = 0.0;
    for (size_t i = 0; i < m; ++i) {
      acc += fw[i2[i]];
      c2[i] = acc;
    }
    if (c2.back() > 0.0) {
      for (auto &c : c2) c /= c2.back();
    }
  }

  double center_val;
  if (c2.back() > 0.0) {
    const size_t lo = weighted_searchsorted(c2, 0.05);
    const size_t hi = weighted_searchsorted(c2, 0.95);
    if (hi > lo) {
      double sw = 0.0;
      double zc = 0.0;
      for (size_t i = lo; i < hi; ++i) {
        zc += fv[i2[i]] * fw[i2[i]];
        sw += fw[i2[i]];
      }
      center_val = (sw > 0.0) ? zc / sw : median(fv);
    } else {
      center_val = median(fv);
    }
  } else {
    center_val = median(fv);
  }

  double vmin, vmax;
  if (c2.back() > 0.0) {
    vmin = fv[i2[weighted_searchsorted(c2, 0.03)]];
    vmax = fv[i2[weighted_searchsorted(c2, 0.97)]];
  } else {
    vmin = *std::min_element(fv.begin(), fv.end());
    vmax = *std::max_element(fv.begin(), fv.end());
  }
  const double min_size = 0.02;
  if (vmax - vmin < min_size) {
    vmin = center_val - min_size / 2.0;
    vmax = center_val + min_size / 2.0;
  }
  return {center_val, vmin, vmax};
}

}  // namespace

Detect3DNode::Detect3DNode()
    : rclcpp_lifecycle::LifecycleNode("detect_3d_node"),
      tf_buffer_(this->get_clock()) {
  this->declare_params();
}

rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
Detect3DNode::on_configure(const rclcpp_lifecycle::State &) {
  RCLCPP_INFO(get_logger(), "[%s] Configuring...", this->get_name());

  this->load_params();

  this->detections_3d_publisher_ =
      this->create_publisher<yolo_msgs::msg::DetectionArray>("detections_3d",
                                                             10);

  // The tf listener works off the executor's callbacks; it must be alive
  // from configuration onwards so the buffer stays populated.
  this->tf_listener_ = std::make_shared<tf2_ros::TransformListener>(
      this->tf_buffer_, this->shared_from_this());

  RCLCPP_INFO(get_logger(), "[%s] Configured", this->get_name());
  return rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn::
      SUCCESS;
}

rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
Detect3DNode::on_activate(const rclcpp_lifecycle::State &) {
  auto reliability_to_policy = [](int r) {
    if (r == 0) {
      return rclcpp::ReliabilityPolicy::SystemDefault;
    }
    if (r == 1) {
      return rclcpp::ReliabilityPolicy::Reliable;
    }
    return rclcpp::ReliabilityPolicy::BestEffort;
  };

  rclcpp::QoS depth_image_qos = rclcpp::QoS(1).reliability(
      reliability_to_policy(this->depth_image_reliability_));
  rclcpp::QoS depth_info_qos = rclcpp::QoS(1).reliability(
      reliability_to_policy(this->depth_info_reliability_));

  this->depth_image_subscription_.subscribe(
      this->shared_from_this(), this->depth_image_topic_,
      depth_image_qos.get_rmw_qos_profile());
  this->depth_info_subscription_.subscribe(
      this->shared_from_this(), this->depth_info_topic_,
      depth_info_qos.get_rmw_qos_profile());
  this->detection_subscription_.subscribe(
      this->shared_from_this(), this->detections_topic_,
      rclcpp::QoS(10).get_rmw_qos_profile());

  this->synchronizer_ =
      std::make_shared<message_filters::Synchronizer<SyncPolicy3D>>(10);
  this->synchronizer_->connectInput(this->depth_image_subscription_,
                                    this->depth_info_subscription_,
                                    this->detection_subscription_);
  this->synchronizer_->registerCallback(
      std::bind(&Detect3DNode::recieve_callback, this, std::placeholders::_1,
                std::placeholders::_2, std::placeholders::_3));

  RCLCPP_INFO(get_logger(), "[%s] Activated", this->get_name());
  return rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn::
      SUCCESS;
}

rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
Detect3DNode::on_deactivate(const rclcpp_lifecycle::State &) {
  this->detection_subscription_.unsubscribe();
  this->depth_info_subscription_.unsubscribe();
  this->depth_image_subscription_.unsubscribe();
  this->synchronizer_.reset();

  RCLCPP_INFO(get_logger(), "[%s] Deactivated", this->get_name());
  return rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn::
      SUCCESS;
}

rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
Detect3DNode::on_cleanup(const rclcpp_lifecycle::State &) {
  this->tf_listener_.reset();
  this->detections_3d_publisher_.reset();

  RCLCPP_INFO(get_logger(), "[%s] Cleaned up", this->get_name());
  return rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn::
      SUCCESS;
}

rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
Detect3DNode::on_shutdown(const rclcpp_lifecycle::State &) {
  RCLCPP_INFO(get_logger(), "[%s] Shutting down", this->get_name());
  return rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn::
      SUCCESS;
}

void Detect3DNode::declare_params() {
  this->declare_parameter<std::string>("target_frame", "base_link");
  this->declare_parameter<double>("maximum_detection_threshold", 0.3);
  this->declare_parameter<int>("depth_image_units_divisor", 1000);
  this->declare_parameter<int>("depth_image_reliability", 2);
  this->declare_parameter<int>("depth_info_reliability", 2);
  this->declare_parameter<std::string>("depth_image_topic", "depth_image");
  this->declare_parameter<std::string>("depth_info_topic", "depth_info");
  this->declare_parameter<std::string>("detections_topic", "detections");
}

void Detect3DNode::load_params() {
  this->get_parameter("target_frame", this->target_frame_);
  this->get_parameter("maximum_detection_threshold",
                      this->maximum_detection_threshold_);
  this->get_parameter("depth_image_units_divisor",
                      this->depth_image_units_divisor_);
  this->get_parameter("depth_image_reliability",
                      this->depth_image_reliability_);
  this->get_parameter("depth_info_reliability", this->depth_info_reliability_);
  this->get_parameter("depth_image_topic", this->depth_image_topic_);
  this->get_parameter("depth_info_topic", this->depth_info_topic_);
  this->get_parameter("detections_topic", this->detections_topic_);
}

void Detect3DNode::recieve_callback(
    const sensor_msgs::msg::Image::ConstSharedPtr &depth_msg,
    const sensor_msgs::msg::CameraInfo::ConstSharedPtr &depth_info_msg,
    const yolo_msgs::msg::DetectionArray::ConstSharedPtr &detections_msg) {
  yolo_msgs::msg::DetectionArray new_detections_msg;
  new_detections_msg.header = detections_msg->header;
  new_detections_msg.detections =
      this->process_detections(depth_msg, depth_info_msg, detections_msg);
  this->detections_3d_publisher_->publish(new_detections_msg);
}

std::vector<yolo_msgs::msg::Detection> Detect3DNode::process_detections(
    const sensor_msgs::msg::Image::ConstSharedPtr &depth_msg,
    const sensor_msgs::msg::CameraInfo::ConstSharedPtr &depth_info_msg,
    const yolo_msgs::msg::DetectionArray::ConstSharedPtr &detections_msg) {
  std::vector<yolo_msgs::msg::Detection> new_detections;

  if (detections_msg->detections.empty()) {
    return new_detections;
  }

  auto transform =
      this->get_transform(depth_info_msg->header.frame_id);
  if (!transform) {
    return new_detections;
  }

  // Build an OpenCV wrapper over the depth image without converting it, like
  // the Python node's "passthrough" encoding: keep the raw 16UC1 millimeters
  // (or 32FC1 meters) and divide by depth_image_units_divisor below.
  cv::Mat depth_image;
  try {
    auto cv_ptr =
        cv_bridge::toCvShare(depth_msg, sensor_msgs::image_encodings::TYPE_16UC1);
    depth_image = cv_ptr->image;
  } catch (cv_bridge::Exception &e) {
    RCLCPP_ERROR(get_logger(), "cv_bridge exception: %s", e.what());
    return new_detections;
  }

  for (const auto &detection : detections_msg->detections) {
    auto bbox3d = this->convert_bb_to_3d(depth_image, *depth_info_msg,
                                         detection);
    if (!bbox3d) {
      continue;
    }

    yolo_msgs::msg::Detection new_detection = detection;
    new_detection.bbox3d =
        Detect3DNode::transform_3d_box(*bbox3d, transform->first,
                                       transform->second);
    new_detection.bbox3d.frame_id = this->target_frame_;
    new_detections.push_back(new_detection);

    if (!detection.keypoints.data.empty()) {
      auto keypoints3d = this->convert_keypoints_to_3d(
          depth_image, *depth_info_msg, detection);
      keypoints3d = Detect3DNode::transform_3d_keypoints(
          keypoints3d, transform->first, transform->second);
      keypoints3d.frame_id = this->target_frame_;
      new_detections.back().keypoints3d = keypoints3d;
    }
  }

  return new_detections;
}

std::optional<yolo_msgs::msg::BoundingBox3D> Detect3DNode::convert_bb_to_3d(
    const cv::Mat &depth_image, const sensor_msgs::msg::CameraInfo &depth_info,
    const yolo_msgs::msg::Detection &detection) {
  const int center_x = static_cast<int>(detection.bbox.center.position.x);
  const int center_y = static_cast<int>(detection.bbox.center.position.y);
  const int size_x = static_cast<int>(detection.bbox.size.x);
  const int size_y = static_cast<int>(detection.bbox.size.y);

  // Raw depth at a pixel: 16UC1 is in depth-units (divide by the divisor to
  // get metres), 32FC1 is already in metres.
  auto depth_at = [&](int v, int u) -> double {
    if (depth_image.type() == CV_16UC1) {
      return static_cast<double>(depth_image.at<uint16_t>(v, u)) /
             this->depth_image_units_divisor_;
    }
    return static_cast<double>(depth_image.at<float>(v, u));
  };

  // Gather the valid depth pixels (depth > 0 and finite) of the detection
  // together with their image coordinates, cropped either by the segmentation
  // mask polygon or by the 2D bounding box.
  std::vector<double> depths;
  std::vector<int> xs, ys;
  auto collect = [&](int v, int u) {
    const double d = depth_at(v, u);
    if (std::isfinite(d) && d > 0.0) {
      depths.push_back(d);
      xs.push_back(u);
      ys.push_back(v);
    }
  };

  // Sample the detection region with an adaptive stride: large patches
    // (>=200 px per side, e.g. a person close to the camera) use step 2 so the
    // robust depth statistics see a representative 1/4 subset of the pixels.
    // The histogram/MAD/percentile pipeline is designed for dense sampling of
    // the same distribution, so this keeps the output statistics materially
    // unchanged while cutting the sample count (and every sort below) by 4x.
    const int step = (size_x * size_y >= 40000) ? 2 : 1;

    if (!detection.mask.data.empty()) {
      // Rasterize only the polygon's bounding rect, not the whole image.
      std::vector<std::vector<cv::Point>> contours(1);
      contours[0].reserve(detection.mask.data.size());
      for (const auto &p : detection.mask.data) {
        contours[0].emplace_back(cvRound(p.x), cvRound(p.y));
      }
      cv::Rect roi = cv::boundingRect(contours[0]);
      roi &= cv::Rect(0, 0, depth_image.cols, depth_image.rows);
      if (roi.width <= 0 || roi.height <= 0) {
        return std::nullopt;
      }
      std::vector<std::vector<cv::Point>> local_contours(1);
      for (const auto &p : contours[0]) {
        local_contours[0].push_back(p - roi.tl());
      }
      cv::Mat mask = cv::Mat::zeros(roi.size(), CV_8UC1);
      cv::fillPoly(mask, local_contours, cv::Scalar(255));
      const int s = (roi.width * roi.height >= 40000) ? 2 : 1;
      for (int v = 0; v < mask.rows; v += s) {
        for (int u = 0; u < mask.cols; u += s) {
          if (mask.at<uchar>(v, u)) {
            collect(v + roi.y, u + roi.x);
          }
        }
      }
    } else {
      const int u_min = std::max(center_x - size_x / 2, 0);
      const int u_max = std::min(center_x + size_x / 2, depth_image.cols - 1);
      const int v_min = std::max(center_y - size_y / 2, 0);
      const int v_max = std::min(center_y + size_y / 2, depth_image.rows - 1);
      if (u_max <= u_min || v_max <= v_min) {
        return std::nullopt;
      }
      for (int v = v_min; v < v_max; v += step) {
        for (int u = u_min; u < u_max; u += step) {
          collect(v, u);
        }
      }
    }
  if (depths.empty()) {
    return std::nullopt;
  }

  // Weight the samples by their distance from the bbox centre so that
  // background/occluding pixels at the edges weigh less.
  const std::vector<double> weights =
      compute_spatial_weights(xs, ys, center_x, center_y, size_x, size_y);

  // Robust, depth-statistics based bounding box (position + per-axis extent).
  const DepthBounds db = compute_depth_bounds_weighted(depths, weights);
  if (!std::isfinite(db.center) || db.center == 0.0) {
    return std::nullopt;
  }

  const auto &k = depth_info.k;  // [fx, 0, cx, 0, fy, cy, 0, 0, 1]
  const double fx = k[0], fy = k[4], px = k[2], py = k[5];
  if (fx == 0.0 || fy == 0.0) {
    return std::nullopt;
  }

  // Back-project each valid pixel to 3D and derive the per-axis bounds from
  // the actual 3D points (not just by projecting the 2D bbox).
  std::vector<double> x3(depths.size()), y3(depths.size());
  for (size_t i = 0; i < depths.size(); ++i) {
    x3[i] = depths[i] * (xs[i] - px) / fx;
    y3[i] = depths[i] * (ys[i] - py) / fy;
  }

  // Height uses a fixed MAD multiplier; width adapts to the depth variance to
  // distinguish occluded/3D objects from flat ones.
  const AxisBounds hb = compute_axis_bounds(y3, weights, 4.5, 0.06, 0.50);
  const double d_mean =
      std::accumulate(depths.begin(), depths.end(), 0.0) / depths.size();
  double d_var = 0.0;
  for (const double d : depths) {
    d_var += (d - d_mean) * (d - d_mean);
  }
  const double depth_std = std::sqrt(d_var / depths.size());
  const double w_mult = (depth_std > 0.15) ? 4.0 : 4.5;
  const double w_lo = (depth_std > 0.15) ? 0.06 : 0.08;
  const double w_hi = (depth_std > 0.15) ? 0.40 : 0.50;
  const AxisBounds wb = compute_axis_bounds(x3, weights, w_mult, w_lo, w_hi);

  if (!std::isfinite(hb.center) || !std::isfinite(hb.min) ||
      !std::isfinite(hb.max) || !std::isfinite(wb.center) ||
      !std::isfinite(wb.min) || !std::isfinite(wb.max)) {
    return std::nullopt;
  }

  yolo_msgs::msg::BoundingBox3D bbox3d;
  bbox3d.center.position.x = wb.center;
  bbox3d.center.position.y = hb.center;
  bbox3d.center.position.z = db.center;
  bbox3d.size.x = wb.max - wb.min;
  bbox3d.size.y = hb.max - hb.min;
  bbox3d.size.z = db.max - db.min;

  return bbox3d;
}

yolo_msgs::msg::KeyPoint3DArray Detect3DNode::convert_keypoints_to_3d(
    const cv::Mat &depth_image, const sensor_msgs::msg::CameraInfo &depth_info,
    const yolo_msgs::msg::Detection &detection) {
  const auto &k = depth_info.k;
  const double fx = k[0], fy = k[4], px = k[2], py = k[5];

  yolo_msgs::msg::KeyPoint3DArray keypoints3d;
  for (const auto &kp : detection.keypoints.data) {
    // Clamp the 2D keypoint into the image (the row index is the y pixel).
    const int u = std::clamp(static_cast<int>(kp.point.y), 0,
                             static_cast<int>(depth_info.height) - 1);
    const int v = std::clamp(static_cast<int>(kp.point.x), 0,
                             static_cast<int>(depth_info.width) - 1);

    double depth;
    if (depth_image.type() == CV_16UC1) {
      depth = static_cast<double>(depth_image.at<uint16_t>(u, v)) /
              this->depth_image_units_divisor_;  // 16UC1 raw units -> meters
    } else {
      depth = static_cast<double>(depth_image.at<float>(u, v));  // already m
    }
    if (!std::isfinite(depth)) {
      continue;
    }

    const double x = depth * (v - px) / fx;
    const double y = depth * (u - py) / fy;

    yolo_msgs::msg::KeyPoint3D kp3d;
    kp3d.id = kp.id;
    kp3d.score = kp.score;
    kp3d.point.x = x;
    kp3d.point.y = y;
    kp3d.point.z = depth;
    keypoints3d.data.push_back(kp3d);
  }

  return keypoints3d;
}

std::optional<std::pair<std::array<double, 3>, std::array<double, 4>>>
Detect3DNode::get_transform(const std::string &frame_id) {
  try {
    // Zero time = latest available transform (same as the Python node).
    const auto transform =
        this->tf_buffer_.lookupTransform(this->target_frame_, frame_id,
                                         tf2::TimePointZero);

    std::array<double, 3> translation{
        transform.transform.translation.x, transform.transform.translation.y,
        transform.transform.translation.z};
    std::array<double, 4> rotation{
        transform.transform.rotation.w, transform.transform.rotation.x,
        transform.transform.rotation.y, transform.transform.rotation.z};

    return std::make_pair(translation, rotation);
  } catch (const tf2::TransformException &ex) {
    RCLCPP_ERROR(get_logger(), "Could not transform: %s", ex.what());
    return std::nullopt;
  }
}

yolo_msgs::msg::BoundingBox3D Detect3DNode::transform_3d_box(
    const yolo_msgs::msg::BoundingBox3D &bbox,
    const std::array<double, 3> &translation,
    const std::array<double, 4> &rotation) {
  yolo_msgs::msg::BoundingBox3D out = bbox;

  // Position: rotate + translate.
  const auto position = Detect3DNode::qv_mult(
      rotation, {bbox.center.position.x, bbox.center.position.y,
                 bbox.center.position.z});
  out.center.position.x = position[0] + translation[0];
  out.center.position.y = position[1] + translation[1];
  out.center.position.z = position[2] + translation[2];

  // Size: only rotate (axis-aligned extents after the rotation).
  const auto size = Detect3DNode::qv_mult(
      rotation, {bbox.size.x, bbox.size.y, bbox.size.z});
  out.size.x = std::abs(size[0]);
  out.size.y = std::abs(size[1]);
  out.size.z = std::abs(size[2]);

  return out;
}

yolo_msgs::msg::KeyPoint3DArray Detect3DNode::transform_3d_keypoints(
    const yolo_msgs::msg::KeyPoint3DArray &keypoints,
    const std::array<double, 3> &translation,
    const std::array<double, 4> &rotation) {
  yolo_msgs::msg::KeyPoint3DArray out = keypoints;

  for (auto &point : out.data) {
    const auto position = Detect3DNode::qv_mult(
        rotation, {point.point.x, point.point.y, point.point.z});
    point.point.x = position[0] + translation[0];
    point.point.y = position[1] + translation[1];
    point.point.z = position[2] + translation[2];
  }

  return out;
}

std::array<double, 3> Detect3DNode::qv_mult(const std::array<double, 4> &q,
                                            const std::array<double, 3> &v) {
  const double qx = q[1], qy = q[2], qz = q[3], qw = q[0];

  // qvec = (qx, qy, qz); uv = qvec x v; uuv = qvec x uv
  const std::array<double, 3> uv{qy * v[2] - qz * v[1],
                                 qz * v[0] - qx * v[2],
                                 qx * v[1] - qy * v[0]};
  const std::array<double, 3> uuv{qy * uv[2] - qz * uv[1],
                                  qz * uv[0] - qx * uv[2],
                                  qx * uv[1] - qy * uv[0]};

  return {v[0] + 2.0 * (uv[0] * qw + uuv[0]),
          v[1] + 2.0 * (uv[1] * qw + uuv[1]),
          v[2] + 2.0 * (uv[2] * qw + uuv[2])};
}

}  // namespace yolo_rclcpp
