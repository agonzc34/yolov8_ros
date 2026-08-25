// Copyright (c) 2026 Alejandro González Cantón
// Portions Copyright (c) 2021 Yifu Zhang
// SPDX-License-Identifier: MIT

#ifndef YOLO_CPP_ROS__TRACKING__BYTE_TRACKER_HPP_
#define YOLO_CPP_ROS__TRACKING__BYTE_TRACKER_HPP_

#include <memory>
#include <vector>

#include "yolo_cpp_ros/tracking/utils/kalman_filter.hpp"
#include "yolo_cpp_ros/tracking/strack.hpp"

namespace yolo_tracking {

// One raw detection fed to the tracker.
struct TrackDetection {
  float cx = 0;  // bounding box center x
  float cy = 0;  // bounding box center y
  float w = 0;   // width
  float h = 0;   // height
  float score = 0;
  int class_id = 0;
  int index = 0;  // position in the original detection array (to fetch metadata)
};

// One tracked object returned by ByteTrack::update().
struct Track {
  int id = 0;  // stable track id
  float x1 = 0;
  float y1 = 0;
  float x2 = 0;
  float y2 = 0;  // Kalman-refined min/max box corners
  float score = 0;
  int class_id = 0;
  int index = 0;  // detection index (from TrackDetection::index)
};

struct ByteTrackParams {
  double track_high_thresh = 0.25;  // first-stage match threshold
  double track_low_thresh = 0.1;    // second-stage low-score threshold
  double new_track_thresh = 0.25;   // min score to start a new track
  int track_buffer = 30;            // frames a lost track is kept alive
  double match_thresh = 0.8;        // association cost threshold
  bool fuse_score = true;           // fuse IoU cost with detection score
};

// ByteTrack implementation based on the original paper and MIT-licensed
// reference implementation. See THIRD_PARTY_NOTICES.md.
class ByteTrack {
public:
  explicit ByteTrack(const ByteTrackParams &params);
  ~ByteTrack() = default;

  // Advance the tracker one frame. `detections` should already be NMS-filtered
  // (the tracker re-splits them by confidence into high / low stages).
  // Returns the currently active (activated) tracked objects.
  std::vector<Track> update(const std::vector<TrackDetection> &detections);

  // Clear all track state and the track-id counter.
  void reset();

  int frame_id() const { return frame_id_; }

private:
  ByteTrackParams params_;
  KalmanFilterXYAH kalman_filter_;
  std::vector<std::shared_ptr<STrack>> tracked_stracks_;
  std::vector<std::shared_ptr<STrack>> lost_stracks_;
  std::vector<std::shared_ptr<STrack>> removed_stracks_;
  int frame_id_ = 0;
  static constexpr std::size_t kRemovedBuffer = 1000;  // cap on removed_stracks_
};

}  // namespace yolo_tracking

#endif  // YOLO_CPP_ROS__TRACKING__BYTE_TRACKER_HPP_
