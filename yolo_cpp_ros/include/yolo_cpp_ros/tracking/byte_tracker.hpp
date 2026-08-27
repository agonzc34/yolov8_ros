// Copyright (c) 2026 Alejandro González Cantón
// Portions Copyright (c) 2021 Yifu Zhang
// SPDX-License-Identifier: MIT

#ifndef YOLO_CPP_ROS__TRACKING__BYTE_TRACKER_HPP_
#define YOLO_CPP_ROS__TRACKING__BYTE_TRACKER_HPP_

#include <memory>
#include <vector>

#include "yolo_cpp_ros/tracking/strack.hpp"
#include "yolo_cpp_ros/tracking/tracker.hpp"
#include "yolo_cpp_ros/tracking/utils/kalman_filter.hpp"

namespace yolo_tracking {

// ByteTrack configuration. `type` is set to "bytetrack", the key used by the
// tracking node's `tracker_type` parameter and by create_tracker().
struct ByteTrackParams : public TrackerParams {
  ByteTrackParams() { type = "bytetrack"; }
  double track_high_thresh = 0.25; // first-stage match threshold
  double track_low_thresh = 0.1;   // second-stage low-score threshold
  double new_track_thresh = 0.25;  // min score to start a new track
  int track_buffer = 30;           // frames a lost track is kept alive
  double match_thresh = 0.8;       // association cost threshold
  bool fuse_score = true;          // fuse IoU cost with detection score
};

// --- ROS parameter bridge (tracker-specific) ------------------------------
// Each tracker owns the declaration/loading of its own ROS parameters
// through its specific functions, so the tracking node only dispatches by the
// `tracker_type` key and never accumulates every algorithm's knobs. The
// functions are templates on the node type for two reasons: rclcpp_lifecycle
// nodes do NOT derive from rclcpp::Node on Humble (so a concrete rclcpp::Node
// reference would not accept the tracking node), and the tracker layer stays
// free of rclcpp includes — the templates only call declare_parameter /
// get_parameter, which any node type provides.

// Declare the ByteTrack parameters on `node` with their default values. Only
// called for the tracker selected by the `tracker_type` parameter, so an
// unselected tracker's knobs do not appear in `ros2 param list`.
template <typename NodeT> void declare_byte_track_params(NodeT &node) {
  node.template declare_parameter<double>("track_high_thresh", 0.25);
  node.template declare_parameter<double>("track_low_thresh", 0.1);
  node.template declare_parameter<double>("new_track_thresh", 0.25);
  node.template declare_parameter<int>("track_buffer", 30);
  node.template declare_parameter<double>("match_thresh", 0.8);
  node.template declare_parameter<bool>("fuse_score", true);
}

// Read the already-declared ByteTrack parameters from `node` and return the
// params struct ready for create_tracker().
template <typename NodeT>
ByteTrackParams load_byte_track_params(const NodeT &node) {
  ByteTrackParams params;
  node.get_parameter("track_high_thresh", params.track_high_thresh);
  node.get_parameter("track_low_thresh", params.track_low_thresh);
  node.get_parameter("new_track_thresh", params.new_track_thresh);
  node.get_parameter("track_buffer", params.track_buffer);
  node.get_parameter("match_thresh", params.match_thresh);
  node.get_parameter("fuse_score", params.fuse_score);
  return params;
}

// ByteTrack implementation based on the original paper and MIT-licensed
// reference implementation. See THIRD_PARTY_NOTICES.md.
class ByteTrack : public Tracker {
public:
  explicit ByteTrack(const ByteTrackParams &params);
  ~ByteTrack() override = default;

  // Advance the tracker one frame. `detections` should already be NMS-filtered
  // (the tracker re-splits them by confidence into high / low stages).
  // Returns the currently active (activated) tracked objects.
  std::vector<Track>
  update(const std::vector<TrackDetection> &detections) override;

  // Clear all track state and the track-id counter.
  void reset() override;

  int frame_id() const { return frame_id_; }

private:
  ByteTrackParams params_;
  KalmanFilterXYAH kalman_filter_;
  std::vector<std::shared_ptr<STrack>> tracked_stracks_;
  std::vector<std::shared_ptr<STrack>> lost_stracks_;
  std::vector<std::shared_ptr<STrack>> removed_stracks_;
  int frame_id_ = 0;
  static constexpr std::size_t kRemovedBuffer = 1000; // cap on removed_stracks_
};

} // namespace yolo_tracking

#endif // YOLO_CPP_ROS__TRACKING__BYTE_TRACKER_HPP_
