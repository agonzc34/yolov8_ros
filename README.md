# yolo_ros

A ROS 2 (Humble) wrapper for running YOLO models with pure C++ /
ONNX Runtime inference. A fork of `mgonz13/yolo_ros` whose Python nodes have
been replaced by a C++ reimplementation (detection, instance segmentation,
human pose, ByteTrack tracking, and depth-based 3D detection/tracking).

Everything here is licensed under **MIT**; the Python/GPL upstream path has been
removed. See [License](#license) and each package's `LICENSE` / notices.

## Packages

| Package | Description |
| --- | --- |
| `yolo_ros` | C++ nodes + ONNX Runtime inference core (YoloDetect / YoloSegment / YoloPose), ByteTrack tracker, debug visualizer and 3D detection node. |
| `yolo_onnxruntime_vendor` | Downloads prebuilt ONNX Runtime 1.20.0 (CPU or `-gpu`) at configure time. |
| `yolo_msgs` | Detection/DetectionArray, BoundingBox2D/3D, KeyPoint2D/3D, Mask, Pose2D, `SetClasses.srv`. |
| `yolo_bringup` | Launch files + ROS-format parameter configs. |

## Prerequisites

- Ubuntu 22.04 with ROS 2 Humble (sourced overlay over `/opt/ros/humble`) and
  the CUDA toolchain for GPU builds.
- An exported ONNX model (e.g. `/path/to/yolo26m.onnx`). Models live outside
  the repo — pass `model:=<path>` on every launch (defaults are machine-specific).
- GPU build needs cuDNN9: `sudo apt install libcudnn9-cuda-12`.
- The Hugging Face Hub vendor needs libcurl dev headers:
  `sudo apt install libcurl4-openssl-dev libssl-dev`.

## Models (from Ultralytics)

The C++ pipeline runs any Ultralytics-exported ONNX model (`yolo11n`,
`yolo26m`, ...). Export a `.pt` checkpoint to ONNX with the `ultralytics`
package — either with `uv` (no environment needed) or a plain `pip install`:

```shell
# uv one-liner (fetches ultralytics + ONNX deps on the fly)
uv run --with ultralytics --with onnx --with onnxruntime --with onnxslim \
  yolo export model=yolo26m.pt format=onnx imgsz=640 opset=12

# classic: pip install ultralytics, then
yolo export model=yolo26m.pt format=onnx imgsz=640 opset=12
```

Notes:

- The exported `.onnx` is self-contained: Ultralytics writes the class
  vocabulary into the ONNX graph metadata (`names` key), which
  `Model::load_class_names()` reads at startup; the `coco.names` fallback is
  only used when a model has no metadata.
- Keep the exported file outside the repo and pass `model:=<path>` on every
  launch (the launch defaults are machine-specific paths).
- `model_type` selects the pipeline: `Segment` for `*-seg.onnx` exports
  (Ultralytics names them `-seg`, not `-segment`), `Pose` for pose models;
  `auto` falls back to a filename heuristic.
- The input size is fixed by the ONNX tensor (logged at startup), so the
  Python-only knobs `imgsz_height`/`imgsz_width` and
  `half`/`augment`/`agnostic_nms`/`retina_masks` were removed from the C++
  node and its configs — NMS is baked at export and the pipeline runs FP32
  with no TTA.

### Download a model from the Hugging Face Hub

Instead of a local path, the C++ node can fetch the model from the Hub at
startup via the `yolo_hfhub_vendor` package (a `huggingface-hub-cpp` vendor).
Set `model_repo` + `model_filename` in the matching `config/yolo*.yaml`
section (or on the command line); the `model` path is then ignored. The file
is cached under `~/.cache/huggingface/hub` by default (override with the
optional `cache_dir` param) and reused unless `force_download: true`:

```yaml
/yolo/yolo_node:
  ros__parameters:
    model_repo: agonzc34/yolo26m   # HF repo id
    model_filename: yolo26m.onnx   # file inside that repo
    force_download: false
```

Or on the command line: `ros2 launch yolo_bringup yolo.launch.py
model_repo:=agonzc34/yolo26m model_filename:=yolo26m.onnx`. Requires libcurl
dev headers (`sudo apt install libcurl4-openssl-dev libssl-dev`); only used
when `model_repo`/`model_filename` are set. The node logs the model source as
`[huggingface]` (with repo/filename) or `[local]` (with the path) so the two
are easy to tell apart.

> **License note**: Ultralytics models and pretrained weights are **not** MIT
> licensed. They are released under the **AGPL-3.0** license (with commercial
> / enterprise licensing available from Ultralytics), so the MIT license of
> this repository does **not** cover them. Check the terms of the specific
> model you use at <https://ultralytics.com/license> — especially if you ship
> or deploy the model.

## Build

GPU build (development default):

```shell
colcon build --symlink-install --cmake-args -DONNX_GPU=ON
```

C++-only fast loop (rebuild `yolo_msgs` first if messages changed):

```shell
colcon build --symlink-install --cmake-args -DONNX_GPU=ON --packages-select yolo_msgs
colcon build --symlink-install --cmake-args -DONNX_GPU=ON --packages-select yolo_ros
```

Omit `-DONNX_GPU=ON` for a CPU build. Launch from the workspace root so
relative source paths resolve.

## Run

Run from the workspace root (so the workspace is sourced as an overlay).

Detection (namespace `yolo`, defaults in `config/yolo.yaml` — model,
image topic, thresholds, QoS, tracking):

```shell
ros2 launch yolo_bringup yolo.launch.py
```

Segmentation (namespace `yolo_seg`, `model_type: Segment` forced in config):

```shell
ros2 launch yolo_bringup yolo_segment.launch.py
```

Pose (namespace `yolo_pose`, `model_type: Pose` forced in config):

```shell
ros2 launch yolo_bringup yolo_pose.launch.py
```

Add `use_3d:=True` to any of the above to also start the C++ 3D detection
node (subscribes to depth + CameraInfo and publishes `detections_3d`). The
segment/pose launches also start the tracking `tracking_node` (`use_tracking`,
default True; ByteTrack by default, switchable via `tracker_type`); with
`use_3d:=True` the 3D node drives the depth ROI with the mask polygon for
segmentation and back-projects pose keypoints to 3D (`debug_kp_markers`).

The 3D node estimates and publishes the 3D box orientation (an oriented
bounding box fit by PCA to a strided depth sample) when the
`enable_orientation` param is set (default `false`); `min_seg_points_for_orientation`
(default 20) is the minimum number of valid depth points required per
detection. When on, `detections_3d` carries a non-identity quaternion in each
box's `center.orientation` and the box `size` is expressed along the object's
own axes.

### Topics

All topics are published under the launch namespace (default `yolo`):

- `detections` — Detections with bounding box and class name (plus mask /
  keypoints for segment/pose models).
- `tracking` — Detections with stable tracker IDs (ByteTrack by default).
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
(`segment` → segmentation, `pose` → pose, else detection). Tracking is
selected with the `tracker_type` param on the `tracking_node` (default
`bytetrack`, case-insensitive).

## Documentation

The C++ API reference is generated with Doxygen and published to GitHub Pages
on every release:

- Latest: <https://agonzc34.github.io/yolov8_ros/>

Build it locally (Doxygen + Graphviz):

```shell
sudo apt install doxygen graphviz
doxygen .github/Doxyfile
# output: docs/index.html
```

## License

The whole repository is licensed under the **MIT License**. See the root
[`LICENSE`](./LICENSE) file; it applies across the repository where a package
does not provide a more specific license file.

Specifically, the repository contains independently licensed ROS 2 packages:

- `yolo_ros`, `yolo_msgs`, `yolo_bringup`, and `yolo_onnxruntime_vendor`
  are licensed under **MIT**. See each package's `LICENSE` file; third-party
  notices are installed with the applicable packages
  (`yolo_ros/THIRD_PARTY_NOTICES.md`).

The C++ pipeline adapts behavior from the original `yolo_ros` Python nodes;
those contributions were authorized by their copyright holder for release in
the MIT-licensed C++ pipeline (see `THIRD_PARTY_NOTICES.md`).

Model weights and exported ONNX files are separate artifacts and remain
subject to their respective licenses; the MIT license for the C++ pipeline
does not relicense them. In particular, Ultralytics models are AGPL-3.0 —
see the [Models](#models-from-ultralytics) section above.
