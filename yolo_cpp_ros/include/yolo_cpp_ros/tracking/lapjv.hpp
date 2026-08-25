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

#ifndef YOLO_CPP_ROS__TRACKING__LAPJV_HPP_
#define YOLO_CPP_ROS__TRACKING__LAPJV_HPP_

#include <cstddef>
#include <vector>

namespace yolo_tracking {

// Dense Jonker-Volgenant solver for the linear assignment problem.
//
// Ported from ByteTrack's C++ deployment (MIT license):
//   deploy/{ncnn,TensorRT}/cpp/src/lapjv.cpp + include/lapjv.h
// which itself is a C port of the `lap` library's lapjv.
//
//   n          : matrix order (cost must be n x n).
//   cost       : row-major cost matrix (vector of row vectors).
//   rowsol[i]  : column assigned to row i, or -1 if the solver's LARGE marker
//                (only used internally; callers manage unmatched entries).
//   colsol[j]  : row assigned to column j.
//
// Returns 0 on success, nonzero if allocation failed (very unlikely).
int lapjv_internal(std::size_t n, const std::vector<std::vector<double>> &cost,
                   std::vector<int> &rowsol, std::vector<int> &colsol);

}  // namespace yolo_tracking

#endif  // YOLO_CPP_ROS__TRACKING__LAPJV_HPP_
