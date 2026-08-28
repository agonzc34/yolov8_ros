// Copyright (c) 2026 Alejandro González Cantón
// Portions Copyright (c) 2021 Yifu Zhang
// SPDX-License-Identifier: MIT

#ifndef YOLO_CPP_ROS__TRACKING__UTILS__MATCHING_HPP_
#define YOLO_CPP_ROS__TRACKING__UTILS__MATCHING_HPP_

#include <cstddef>
#include <memory>
#include <utility>
#include <vector>

#include "yolo_cpp_ros/tracking/strack.hpp"

namespace yolo_tracking {

// 1 - IoU between every (a, b) pair of tracks. Cost matrix shape
// (len(a), len(b)); a and b may be lists of any objects exposing xyxy().
std::vector<std::vector<double>>
iou_distance(const std::vector<std::shared_ptr<STrack>> &atracks,
             const std::vector<std::shared_ptr<STrack>> &btracks);

// Fuse the IoU cost with detection scores as in the ByteTrack reference code:
//   fuse_cost = 1 - (1 - cost) * det_score   (per column).
// Modifies `cost_matrix` in place (no extra allocation) and returns it.
std::vector<std::vector<double>> &
fuse_score(std::vector<std::vector<double>> &cost_matrix,
           const std::vector<std::shared_ptr<STrack>> &detections);

// Hungarian (lapjv, extend_cost + cost_limit) linear assignment on the cost
// matrix. Only pairs whose assignment cost <= `thresh` are retained as matches;
// the rest are reported unmatched. The matrix dims are passed explicitly
// (`n_rows` x `n_cols`) because an empty cost matrix (no tracks and/or no
// detections) still needs to report its shape. The matrix is extended with
// dummy assignments so a finite cost threshold can represent unmatched rows.
void linear_assignment(std::size_t n_rows, std::size_t n_cols,
                       const std::vector<std::vector<double>> &cost_matrix,
                       double thresh, std::vector<std::pair<int, int>> &matches,
                       std::vector<int> &unmatched_a,
                       std::vector<int> &unmatched_b);

// Union of two track lists, de-duplicated by track_id (atracks win).
std::vector<std::shared_ptr<STrack>>
joint_stracks(const std::vector<std::shared_ptr<STrack>> &atracks,
              const std::vector<std::shared_ptr<STrack>> &btracks);

// atracks minus any track whose track_id appears in btracks.
std::vector<std::shared_ptr<STrack>>
sub_stracks(const std::vector<std::shared_ptr<STrack>> &atracks,
            const std::vector<std::shared_ptr<STrack>> &btracks);

// Remove duplicate tracks across two lists based on IoU distance
// (`dup_thresh = 0.15`); the shorter-lived track is dropped, ties drop from
// `atracks` (the tie-break used by the ByteTrack reference implementation).
std::pair<std::vector<std::shared_ptr<STrack>>,
          std::vector<std::shared_ptr<STrack>>>
remove_duplicate_stracks(const std::vector<std::shared_ptr<STrack>> &atracks,
                         const std::vector<std::shared_ptr<STrack>> &btracks,
                         double dup_thresh = 0.15);

} // namespace yolo_tracking

#endif // YOLO_CPP_ROS__TRACKING__UTILS__MATCHING_HPP_
