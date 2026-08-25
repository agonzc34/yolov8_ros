// Copyright (C) 2026 Alejandro González Cantón
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.

// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.

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
std::vector<std::vector<double>> iou_distance(
    const std::vector<std::shared_ptr<STrack>> &atracks,
    const std::vector<std::shared_ptr<STrack>> &btracks);

// Fuse the IoU cost with the detection scores (ultralytics fuse_score):
//   fuse_cost = 1 - (1 - cost) * det_score   (per column).
std::vector<std::vector<double>> fuse_score(
    const std::vector<std::vector<double>> &cost_matrix,
    const std::vector<std::shared_ptr<STrack>> &detections);

// Hungarian (lapjv, extend_cost + cost_limit) linear assignment on the cost
// matrix. Only pairs whose assignment cost <= `thresh` are retained as matches;
// the rest are reported unmatched. The matrix dims are passed explicitly
// (`n_rows` x `n_cols`) because an empty cost matrix (no tracks and/or no
// detections) still needs to report its shape, exactly like numpy. Mirrors
// ultralytics' `matching.linear_assignment(dists, thresh)`, which calls
// `lap.lapjv(dists, extend_cost=True, cost_limit=thresh)`.
void linear_assignment(std::size_t n_rows, std::size_t n_cols,
                       const std::vector<std::vector<double>> &cost_matrix,
                       double thresh,
                       std::vector<std::pair<int, int>> &matches,
                       std::vector<int> &unmatched_a,
                       std::vector<int> &unmatched_b);

// Union of two track lists, de-duplicated by track_id (atracks win).
std::vector<std::shared_ptr<STrack>> joint_stracks(
    const std::vector<std::shared_ptr<STrack>> &atracks,
    const std::vector<std::shared_ptr<STrack>> &btracks);

// atracks minus any track whose track_id appears in btracks.
std::vector<std::shared_ptr<STrack>> sub_stracks(
    const std::vector<std::shared_ptr<STrack>> &atracks,
    const std::vector<std::shared_ptr<STrack>> &btracks);

// Remove duplicate tracks across two lists based on IoU distance
// (`dup_thresh = 0.15`); the shorter-lived track is dropped, ties drop from
// `atracks` (same tie-break as ultralytics).
std::pair<std::vector<std::shared_ptr<STrack>>,
          std::vector<std::shared_ptr<STrack>>>
remove_duplicate_stracks(const std::vector<std::shared_ptr<STrack>> &atracks,
                         const std::vector<std::shared_ptr<STrack>> &btracks,
                         double dup_thresh = 0.15);

}  // namespace yolo_tracking

#endif  // YOLO_CPP_ROS__TRACKING__UTILS__MATCHING_HPP_
