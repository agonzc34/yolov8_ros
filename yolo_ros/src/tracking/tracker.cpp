// Copyright (c) 2026 Alejandro González Cantón
// SPDX-License-Identifier: MIT

#include "yolo_ros/tracking/tracker.hpp"

#include <algorithm>
#include <cctype>
#include <string>

#include "yolo_ros/tracking/byte_tracker.hpp"

namespace yolo_ros::tracking {

namespace {

std::string lowercase(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  return s;
}

} // namespace

std::unique_ptr<Tracker> create_tracker(const TrackerParams &params) {
  const std::string type = lowercase(params.type);

  // --- add new trackers here ---------------------------------------------
  // Register the key accepted by the tracking node's `tracker_type` parameter
  // and construct your tracker from its own params struct (derived from
  // TrackerParams). The node additionally dispatches to the tracker's own
  // parameter functions — declare_<tracker>_params / load_<tracker>_params
  // (see byte_tracker.hpp) — so each algorithm owns its config surface;
  // nothing else in the pipeline changes.
  if (type == "bytetrack") {
    // Guard the dynamic_cast: nullptr means the caller passed the wrong params
    // struct for the requested tracker — report it via the nullptr return
    // instead of throwing std::bad_cast.
    const auto *byte_params = dynamic_cast<const ByteTrackParams *>(&params);
    if (byte_params == nullptr) {
      return nullptr;
    }
    return std::make_unique<ByteTrack>(*byte_params);
  }

  return nullptr;
}

} // namespace yolo_ros::tracking