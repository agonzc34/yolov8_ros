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

#include "yolo_cpp_ros/tracking/byte_tracker.hpp"

#include <algorithm>
#include <cstddef>
#include <numeric>

#include "yolo_cpp_ros/tracking/utils/matching.hpp"

namespace yolo_tracking {

ByteTrack::ByteTrack(const ByteTrackParams &params) : params_(params) {}

std::vector<Track> ByteTrack::update(const std::vector<TrackDetection> &dets) {
  ++frame_id_;
  std::vector<std::shared_ptr<STrack>> activated_stracks;
  std::vector<std::shared_ptr<STrack>> refind_stracks;
  std::vector<std::shared_ptr<STrack>> lost_stracks;
  std::vector<std::shared_ptr<STrack>> removed_stracks;

  // --- Step 1: split detections into high / low score pools. ------------------
  // Equivalent to ultralytics _split_detections + init_track: each detection
  // with a degenerate (non-positive) box is dropped and the STrack stores the
  // detection's index in the *full* array for later metadata lookup.
  std::vector<std::shared_ptr<STrack>> detections;         // high-score
  std::vector<std::shared_ptr<STrack>> detections_second;  // low-score
  for (std::size_t i = 0; i < dets.size(); ++i) {
    const TrackDetection &d = dets[i];
    if (d.w <= 0 || d.h <= 0) {
      continue;  // guard: tlwh_to_xyah divides by height
    }
    const std::array<float, 4> xywh = {d.cx, d.cy, d.w, d.h};
    if (d.score >= params_.track_high_thresh) {
      detections.push_back(
          std::make_shared<STrack>(xywh, d.score, d.class_id, static_cast<int>(i)));
    } else if (d.score > params_.track_low_thresh &&
               d.score < params_.track_high_thresh) {
      detections_second.push_back(
          std::make_shared<STrack>(xywh, d.score, d.class_id, static_cast<int>(i)));
    }
  }

  // --- Step 2: split the tracked pool into unconfirmed / confirmed. -----------
  std::vector<std::shared_ptr<STrack>> unconfirmed;
  std::vector<std::shared_ptr<STrack>> tracked;
  for (auto &t : tracked_stracks_) {
    (t->is_activated() ? tracked : unconfirmed).push_back(t);
  }

  // --- Step 3: joint pool (confirmed tracked + lost) and Kalman predict. ------
  std::vector<std::shared_ptr<STrack>> strack_pool =
      joint_stracks(tracked, lost_stracks_);
  for (auto &t : strack_pool) {
    t->predict();
  }

  // --- Step 4: first association, high-score detections. ----------------------
  std::vector<std::pair<int, int>> matches;
  std::vector<int> u_track;
  std::vector<int> u_detection;
  {
    auto dists = get_dists(strack_pool, detections);
    linear_assignment(strack_pool.size(), detections.size(), dists,
                      params_.match_thresh, matches, u_track, u_detection);
    for (const auto &m : matches) {
      apply_match(strack_pool[m.first], detections[m.second], activated_stracks,
                  refind_stracks);
    }
  }

  // --- Step 5: second association with low-score detections. -------------------
  std::vector<std::shared_ptr<STrack>> r_tracked_stracks;
  for (int i : u_track) {
    if (strack_pool[i]->state() == TrackState::Tracked) {
      r_tracked_stracks.push_back(strack_pool[i]);
    }
  }
  std::vector<int> u_track_second;
  if (!r_tracked_stracks.empty() && !detections_second.empty()) {
    matches.clear();
    std::vector<int> tmp_detection;  // discard, as in ultralytics' `_`
    auto dists = iou_distance(r_tracked_stracks, detections_second);  // no fuse
    linear_assignment(r_tracked_stracks.size(), detections_second.size(),
                      dists, 0.5, matches, u_track_second, tmp_detection);
    for (const auto &m : matches) {
      apply_match(r_tracked_stracks[m.first], detections_second[m.second],
                  activated_stracks, refind_stracks);
    }
  } else {
    u_track_second.resize(r_tracked_stracks.size());
    std::iota(u_track_second.begin(), u_track_second.end(), 0);
  }
  for (int i : u_track_second) {
    auto track = r_tracked_stracks[i];
    if (track->state() != TrackState::Lost) {
      track->mark_lost();
      lost_stracks.push_back(track);
    }
  }

  // --- Step 6: associate unconfirmed tracks with leftover high detections. ----
  std::vector<std::shared_ptr<STrack>> detections_left;
  for (int i : u_detection) {
    detections_left.push_back(detections[i]);
  }
  std::vector<int> u_detection_left;
  if (!unconfirmed.empty()) {
    matches.clear();
    std::vector<int> u_unconfirmed;
    auto dists = get_dists(unconfirmed, detections_left);
    linear_assignment(unconfirmed.size(), detections_left.size(), dists, 0.7,
                      matches, u_unconfirmed, u_detection_left);
    for (const auto &m : matches) {
      unconfirmed[m.first]->update(*detections_left[m.second], frame_id_);
      activated_stracks.push_back(unconfirmed[m.first]);
    }
    for (int i : u_unconfirmed) {
      auto track = unconfirmed[i];
      track->mark_removed();
      removed_stracks.push_back(track);
    }
  } else {
    u_detection_left.resize(detections_left.size());
    std::iota(u_detection_left.begin(), u_detection_left.end(), 0);
  }

  // --- Step 7: activate brand-new tracks. --------------------------------------
  for (int inew : u_detection_left) {
    auto track = detections_left[inew];
    if (track->score() < params_.new_track_thresh) {
      continue;
    }
    track->activate(&kalman_filter_, frame_id_);
    activated_stracks.push_back(track);
  }

  // --- Step 8: remove lost tracks aged past the buffer. ------------------------
  for (auto &track : lost_stracks_) {
    if (frame_id_ - track->end_frame() > params_.track_buffer) {
      track->mark_removed();
      removed_stracks.push_back(track);
    }
  }

  // --- Step 9: end-of-frame pool bookkeeping. -----------------------------------
  merge_track_pools(activated_stracks, refind_stracks, lost_stracks,
                    removed_stracks);

  // --- Step 10: format output (only activated tracks). --------------------------
  std::vector<Track> output;
  output.reserve(tracked_stracks_.size());
  for (auto &t : tracked_stracks_) {
    if (!t->is_activated()) {
      continue;
    }
    const auto xyxy = t->xyxy();
    Track tr;
    tr.id = t->track_id();
    tr.x1 = xyxy[0];
    tr.y1 = xyxy[1];
    tr.x2 = xyxy[2];
    tr.y2 = xyxy[3];
    tr.score = t->score();
    tr.class_id = t->class_id();
    tr.index = t->idx();
    output.push_back(tr);
  }
  return output;
}

void ByteTrack::reset() {
  tracked_stracks_.clear();
  lost_stracks_.clear();
  removed_stracks_.clear();
  frame_id_ = 0;
  kalman_filter_ = KalmanFilterXYAH();
  STrack::reset_id();
}

void ByteTrack::merge_track_pools(
    std::vector<std::shared_ptr<STrack>> &activated,
    std::vector<std::shared_ptr<STrack>> &refind,
    std::vector<std::shared_ptr<STrack>> &lost,
    std::vector<std::shared_ptr<STrack>> &removed) {
  tracked_stracks_.erase(
      std::remove_if(tracked_stracks_.begin(), tracked_stracks_.end(),
                     [](const std::shared_ptr<STrack> &t) {
                       return t->state() != TrackState::Tracked;
                     }),
      tracked_stracks_.end());

  tracked_stracks_ = joint_stracks(tracked_stracks_, activated);
  tracked_stracks_ = joint_stracks(tracked_stracks_, refind);

  lost_stracks_ = sub_stracks(lost_stracks_, tracked_stracks_);
  lost_stracks_.insert(lost_stracks_.end(), lost.begin(), lost.end());
  // Drop lost tracks that were removed in any earlier frame (ultralytics uses
  // the persistent removed pool here, not just this frame's removals).
  lost_stracks_ = sub_stracks(lost_stracks_, removed_stracks_);

  auto dedup = remove_duplicate_stracks(tracked_stracks_, lost_stracks_);
  tracked_stracks_ = dedup.first;
  lost_stracks_ = dedup.second;

  removed_stracks_.insert(removed_stracks_.end(), removed.begin(), removed.end());
  if (removed_stracks_.size() > kRemovedBuffer) {
    removed_stracks_.erase(removed_stracks_.begin(),
                           removed_stracks_.end() - kRemovedBuffer);
  }
}

void ByteTrack::apply_match(
    const std::shared_ptr<STrack> &track,
    const std::shared_ptr<STrack> &detection,
    std::vector<std::shared_ptr<STrack>> &activated,
    std::vector<std::shared_ptr<STrack>> &refind) {
  if (track->state() == TrackState::Tracked) {
    track->update(*detection, frame_id_);
    activated.push_back(track);
  } else {
    track->re_activate(*detection, frame_id_, false);
    refind.push_back(track);
  }
}

std::vector<std::vector<double>> ByteTrack::get_dists(
    const std::vector<std::shared_ptr<STrack>> &tracks,
    const std::vector<std::shared_ptr<STrack>> &detections) const {
  auto dists = iou_distance(tracks, detections);
  if (params_.fuse_score) {
    dists = fuse_score(dists, detections);
  }
  return dists;
}

}  // namespace yolo_tracking
