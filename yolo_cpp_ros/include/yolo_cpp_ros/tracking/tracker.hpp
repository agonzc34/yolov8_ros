// Copyright (c) 2026 Alejandro González Cantón
// SPDX-License-Identifier: MIT

#ifndef YOLO_CPP_ROS__TRACKING__TRACKER_HPP_
#define YOLO_CPP_ROS__TRACKING__TRACKER_HPP_

#include <memory>
#include <string>
#include <vector>

namespace yolo_tracking {

// One raw detection fed to a tracker. Tracker-agnostic: every tracker
// implementation consumes the same input shape, so the tracking node can
// switch algorithms without converting to a per-tracker format.
struct TrackDetection {
  float cx = 0; // bounding box center x
  float cy = 0; // bounding box center y
  float w = 0;  // width
  float h = 0;  // height
  float score = 0;
  int class_id = 0;
  int index = 0; // position in the original detection array (to fetch metadata)
};

// One tracked object returned by Tracker::update().
struct Track {
  int id = 0; // stable track id
  float x1 = 0;
  float y1 = 0;
  float x2 = 0;
  float y2 = 0; // tracker-refined min/max box corners
  float score = 0;
  int class_id = 0;
  int index = 0; // detection index (from TrackDetection::index)
};

// Base tracker configuration. `type` is the canonical tracker key — the same
// string the tracking node's `tracker_type` parameter accepts (e.g.
// "bytetrack") — and create_tracker() dispatches on it. New trackers derive
// their own params struct from this one and set `type` in their constructor;
// the derived struct holds the algorithm's tuning knobs.
struct TrackerParams {
  virtual ~TrackerParams() = default;
  std::string type;
};

// Abstract multi-object tracker interface. The tracking node (and nothing
// else in the package) interacts with trackers through this interface, so a
// new tracker implementation never touches the node, the message flow or the
// downstream topics.
class Tracker {
public:
  virtual ~Tracker() = default;

  // Advance the tracker one frame. `detections` should already be NMS-filtered
  // (implementations re-split them by confidence as their algorithm requires).
  // Returns the currently active (activated) tracked objects.
  virtual std::vector<Track>
  update(const std::vector<TrackDetection> &detections) = 0;

  // Clear all track state and the track-id counter.
  virtual void reset() = 0;
};

// Build the tracker named by `params.type` (case-insensitive). Returns nullptr
// for unknown types or when `params` is not the params struct of that tracker,
// so callers can fall back gracefully instead of crashing.
std::unique_ptr<Tracker> create_tracker(const TrackerParams &params);

} // namespace yolo_tracking

#endif // YOLO_CPP_ROS__TRACKING__TRACKER_HPP_