// Copyright (c) 2026 Alejandro González Cantón
// Portions Copyright (c) 2023-2025 Miguel Ángel González Santamarta
// SPDX-License-Identifier: MIT

#ifndef YOLO_CPP_ROS__3D__DEPTH_UTILS_HPP_
#define YOLO_CPP_ROS__3D__DEPTH_UTILS_HPP_

#include <array>
#include <optional>
#include <utility>
#include <vector>

#include <opencv2/core.hpp>

#include "sensor_msgs/msg/camera_info.hpp"
#include "yolo_msgs/msg/bounding_box3_d.hpp"
#include "yolo_msgs/msg/detection.hpp"
#include "yolo_msgs/msg/key_point3_d_array.hpp"

namespace yolo_3d {

// ---------------------------------------------------------------------------
// Robust depth-statistics helpers, ported from the Python detect_3d_node.py
// (_compute_spatial_weights, _compute_depth_bounds_weighted,
// _compute_height_bounds, _compute_width_bounds). They turn the raw depth
// samples of a detection into a 3D box: a spatially weighted, trimmed-mean
// center and MAD/percentile-based extents per axis.
// ---------------------------------------------------------------------------

// Median of a copy of v (nth_element-based, O(N)).
double median(std::vector<double> v);

// Linear-interpolation percentile on an already-sorted vector (q in [0,1]),
// matching numpy.percentile(..., method='linear').
double percentile_sorted(const std::vector<double> &v, double q);

double weighted_mean(const std::vector<double> &v,
                     const std::vector<double> &w);

// Gaussian falloff from the bbox center, floored at 0.3
// (Python _compute_spatial_weights).
std::vector<double> compute_spatial_weights(const std::vector<int> &xs,
                                            const std::vector<int> &ys,
                                            int center_x, int center_y,
                                            int size_x, int size_y);

struct DepthBounds {
  double center;
  double min;
  double max;
};

// Weighted histogram peak + MAD/IQR adaptive filtering + trimmed weighted
// center and 1st/99th weighted percentiles (Python _compute_depth_bounds_weighted).
DepthBounds compute_depth_bounds_weighted(std::vector<double> depth,
                                          std::vector<double> weight);

struct AxisBounds {
  double center;
  double min;
  double max;
};

// Outlier-filtered (MAD), trimmed weighted center + 3rd/97th weighted
// percentiles for one 3D axis (Python _compute_height_bounds / _compute_width_bounds).
AxisBounds compute_axis_bounds(const std::vector<double> &val3,
                               const std::vector<double> &w, double mad_mult,
                               double clip_lo, double clip_hi);

// Raw depth at a pixel (row v, col u): 16UC1 raw units divided by
// depth_units_divisor to get metres, 32FC1 already in metres.
double depth_at_pixel(const cv::Mat &depth_image, int v, int u,
                      int depth_units_divisor);

// Lift a 2D detection (bbox, optionally mask-guided depth sampling) into a
// BoundingBox3D in the depth camera frame, using the camera intrinsics and
// the robust depth statistics above.
std::optional<yolo_msgs::msg::BoundingBox3D>
convert_bb_to_3d(const cv::Mat &depth_image,
                 const sensor_msgs::msg::CameraInfo &depth_info,
                 const yolo_msgs::msg::Detection &detection,
                 int depth_units_divisor);

// Back-project the 2D pose keypoints of a detection into 3D (depth camera
// frame), keeping id/score; keypoints with no valid depth are skipped.
yolo_msgs::msg::KeyPoint3DArray
convert_keypoints_to_3d(const cv::Mat &depth_image,
                        const sensor_msgs::msg::CameraInfo &depth_info,
                        const yolo_msgs::msg::Detection &detection,
                        int depth_units_divisor);

// Quaternion-vector rotation: v' = q v q^-1 (q = [w, x, y, z]).
std::array<double, 3> qv_mult(const std::array<double, 4> &q,
                              const std::array<double, 3> &v);

// Apply a rigid transform (translation + rotation) to a 3D box: the position
// is rotated and translated, the axis-aligned size only rotated (abs of the
// rotated extents).
yolo_msgs::msg::BoundingBox3D transform_3d_box(
    const yolo_msgs::msg::BoundingBox3D &bbox,
    const std::array<double, 3> &translation,
    const std::array<double, 4> &rotation);

yolo_msgs::msg::KeyPoint3DArray transform_3d_keypoints(
    const yolo_msgs::msg::KeyPoint3DArray &keypoints,
    const std::array<double, 3> &translation,
    const std::array<double, 4> &rotation);

}  // namespace yolo_3d

#endif  // YOLO_CPP_ROS__3D__DEPTH_UTILS_HPP_