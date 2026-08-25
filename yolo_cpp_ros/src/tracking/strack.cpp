// Copyright (c) 2026 Alejandro González Cantón
// Portions Copyright (c) 2021 Yifu Zhang
// SPDX-License-Identifier: MIT

#include "yolo_cpp_ros/tracking/strack.hpp"

namespace yolo_tracking {

int STrack::count_ = 0;

STrack::STrack(const std::array<float, 4> &xywh, float score, int class_id,
               int idx) {
  // xywh (center) -> tlwh
  _tlwh_[0] = xywh[0] - xywh[2] / 2;
  _tlwh_[1] = xywh[1] - xywh[3] / 2;
  _tlwh_[2] = xywh[2];
  _tlwh_[3] = xywh[3];
  kf_ = nullptr;
  mean_ = {};
  covariance_ = {};
  has_state_ = false;
  is_activated_ = false;
  track_id_ = 0;
  frame_id_ = 0;
  start_frame_ = 0;
  tracklet_len_ = 0;
  state_ = TrackState::New;
  score_ = score;
  class_id_ = class_id;
  idx_ = idx;
}

void STrack::predict() {
  if (!has_state_) {
    return;
  }
  KalmanMean mean_state = mean_;
  if (state_ != TrackState::Tracked) {
    mean_state[7] = 0;  // freeze height velocity for lost tracks
  }
  auto predicted = kf_->predict(mean_state, covariance_);
  mean_ = predicted.first;
  covariance_ = predicted.second;
}

void STrack::activate(const KalmanFilterXYAH *kalman_filter, int frame_id) {
  kf_ = kalman_filter;
  track_id_ = next_id();
  auto initiated = kf_->initiate(tlwh_to_xyah(_tlwh_));
  mean_ = initiated.first;
  covariance_ = initiated.second;
  has_state_ = true;

  tracklet_len_ = 0;
  state_ = TrackState::Tracked;
  is_activated_ = (frame_id == 1);
  frame_id_ = frame_id;
  start_frame_ = frame_id;
}

void STrack::re_activate(const STrack &new_track, int frame_id, bool new_id) {
  auto updated = kf_->update(mean_, covariance_, tlwh_to_xyah(new_track._tlwh_));
  mean_ = updated.first;
  covariance_ = updated.second;

  tracklet_len_ = 0;
  state_ = TrackState::Tracked;
  is_activated_ = true;
  frame_id_ = frame_id;
  if (new_id) {
    track_id_ = next_id();
  }
  score_ = new_track.score_;
  class_id_ = new_track.class_id_;
  idx_ = new_track.idx_;
}

void STrack::update(const STrack &new_track, int frame_id) {
  frame_id_ = frame_id;
  tracklet_len_ += 1;

  auto updated = kf_->update(mean_, covariance_, tlwh_to_xyah(new_track._tlwh_));
  mean_ = updated.first;
  covariance_ = updated.second;
  state_ = TrackState::Tracked;
  is_activated_ = true;

  score_ = new_track.score_;
  class_id_ = new_track.class_id_;
  idx_ = new_track.idx_;
}

int STrack::next_id() { return ++count_; }

std::array<float, 4> STrack::tlwh() const {
  if (!has_state_) {
    return _tlwh_;
  }
  // mean: (mx, my, aspect, height) -> (x, y, w, h)
  const float w = static_cast<float>(mean_[2] * mean_[3]);
  const float h = static_cast<float>(mean_[3]);
  const float x = static_cast<float>(mean_[0] - mean_[2] * mean_[3] / 2.0);
  const float y = static_cast<float>(mean_[1] - mean_[3] / 2.0);
  return {x, y, w, h};
}

std::array<float, 4> STrack::xyxy() const {
  auto box = tlwh();
  box[2] += box[0];
  box[3] += box[1];
  return box;
}

std::array<float, 4> STrack::xywh() const {
  auto box = tlwh();
  box[0] += box[2] / 2;
  box[1] += box[3] / 2;
  return box;
}

std::array<double, 4> STrack::tlwh_to_xyah(const std::array<float, 4> &tlwh) {
  std::array<double, 4> out;
  out[0] = static_cast<double>(tlwh[0] + tlwh[2] / 2);
  out[1] = static_cast<double>(tlwh[1] + tlwh[3] / 2);
  out[2] = static_cast<double>(tlwh[2]) / tlwh[3];  // aspect (h > 0 guaranteed upstream)
  out[3] = static_cast<double>(tlwh[3]);
  return out;
}

}  // namespace yolo_tracking
