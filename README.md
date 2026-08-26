# yolov8_ros

A ROS 2 (Humble) wrapper for running Ultralytics YOLO models with pure C++ /
ONNX Runtime inference. A fork of `mgonz13/yolo_ros` whose Python nodes have
been replaced by a C++ reimplementation (detection, instance segmentation,
human pose, ByteTrack tracking, and depth-based 3D detection/tracking).

Everything here is licensed under **MIT**; the Python/GPL upstream path has been
removed. See [License](#license) and each package's `LICENSE` / notices.

## Packages

| Package | Description |
| --- | --- |
| `yolo_cpp_ros` | C++ nodes + ONNX Runtime inference core (YoloDetect / YoloSegment / YoloPose), ByteTrack tracker, debug visualizer and 3D detection node. |
| `yolo_onnxruntime_vendor` | Downloads prebuilt ONNX Runtime 1.20.0 (CPU or `-gpu`) at configure time. |
| `yolo_msgs` | Detection/DetectionArray, BoundingBox2D/3D, KeyPoint2D/3D, Mask, Pose2D, `SetClasses.srv`. |
| `yolo_bringup` | Launch files + ROS-format parameter configs. |

## Prerequisites

- Ubuntu 22.04 with ROS 2 Humble (sourced overlay over `/opt/ros/humble`) and
  the CUDA toolchain for GPU builds.
- An exported ONNX model (e.g. `/path/to/yolo26m.onnx`). Models live outside
  the repo — pass `model:=<path>` on every launch (defaults are machine-specific).
- GPU build needs cuDNN9: `sudo apt install libcudnn9-cuda-12`.

## Build

GPU build (development default):

```shell
colcon build --symlink-install --cmake-args -DONNX_GPU=ON
```

C++-only fast loop (rebuild `yolo_msgs` first if messages changed):

```shell
colcon build --symlink-install --cmake-args -DONNX_GPU=ON --packages-select yolo_msgs
colcon build --symlink-install --cmake-args -DONNX_GPU=ON --packages-select yolo_cpp_ros
```

Omit `-DONNX_GPU=ON` for a CPU build. Launch from the workspace root so
relative source paths resolve.

## Run

Run from the workspace root (so the workspace is sourced as an overlay).

Detection (namespace `yolo`, defaults in `config/yolo_cpp.yaml` — model,
image topic, thresholds, QoS, tracking):

```shell
ros2 launch yolo_bringup yolo_cpp.launch.py
```

Segmentation (namespace `yolo_seg`, `model_type: Segment` forced in config):

```shell
ros2 launch yolo_bringup yolo_cpp_segment.launch.py
```

Pose (namespace `yolo_pose`, `model_type: Pose` forced in config):

```shell
ros2 launch yolo_bringup yolo_cpp_pose.launch.py
```

Add `use_3d:=True` to any of the above to also start the C++ 3D detection
node (subscribes to depth + CameraInfo and publishes `detections_3d`). The
segment/pose launches also start the ByteTrack `tracking_node`
(`use_tracking`, default True); with `use_3d:=True` the 3D node drives the
depth ROI with the mask polygon for segmentation and back-projects pose
keypoints to 3D (`debug_kp_markers`).

### Topics

All topics are published under the launch namespace (default `yolo`):

- `detections` — Detections with bounding box and class name (plus mask /
  keypoints for segment/pose models).
- `tracking` — Detections with stable ByteTrack IDs.
- `detections_3d` — 3D boxes/keypoints, when `use_3d:=True`.
- `debug_image`, `debug_bb_markers` / `debug_kp_markers` — debug visualization
  and RViz MarkerArrays.

All nodes are lifecycle nodes; executables call `configure()`/`activate()`
themselves in `main()`.

### Key parameters

Everything (model, image topic, thresholds, QoS, tracker, `enable`, `max_det`)
is configured through the params file passed to the launch; the C++ launches
make **no** topic remaps and use no inline parameter dictionaries. Model
dispatch is selected with the `model_type` param (`YOLO`/`Detect`/`Segment`/
`Pose`/`auto`, case-insensitive); `auto` falls back to a filename heuristic
(`segment` → segmentation, `pose` → pose, else detection).

## License

This repository contains independently licensed ROS 2 packages:

- `yolo_cpp_ros`, `yolo_msgs`, `yolo_bringup`, and `yolo_onnxruntime_vendor`
  are licensed under **MIT**. See each package's `LICENSE` file; third-party
  notices are installed with the applicable packages
  (`yolo_cpp_ros/THIRD_PARTY_NOTICES.md`).

The C++ pipeline adapts behavior from the original `yolo_ros` Python nodes;
those contributions were authorized by their copyright holder for release in
the MIT-licensed C++ pipeline (see `THIRD_PARTY_NOTICES.md`).

Model weights and exported ONNX files are separate artifacts and remain
subject to their respective licenses; the MIT license for the C++ pipeline
does not relicense them.
