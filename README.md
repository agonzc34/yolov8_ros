# yolo_ros

ROS 2 wrap for YOLO models from [Ultralytics](https://github.com/ultralytics/ultralytics) to perform object detection and tracking, instance segmentation, human pose estimation, Oriented Bounding Box (OBB) and image classification. There are also 3D versions of object detection, instance segmentation and human pose estimation based on depth images.

The pipeline is a pure **C++ / ONNX Runtime** implementation: there is no Python runtime and no `ultralytics` dependency at run time. It runs any Ultralytics-exported ONNX model (see [Models](#models)), and the whole repository is licensed under **MIT**.

<div align="center">

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/license/mit) [![GitHub release](https://img.shields.io/github/release/mgonzs13/yolo_ros.svg)](https://github.com/mgonzs13/yolo_ros/releases) [![Code Size](https://img.shields.io/github/languages/code-size/mgonzs13/yolo_ros.svg?branch=main)](https://github.com/mgonzs13/yolo_ros?branch=main) [![Last Commit](https://img.shields.io/github/last-commit/mgonzs13/yolo_ros.svg?branch=main)](https://github.com/mgonzs13/yolo_ros/commits/main) [![GitHub issues](https://img.shields.io/github/issues/mgonzs13/yolo_ros)](https://github.com/mgonzs13/yolo_ros/issues) [![GitHub pull requests](https://img.shields.io/github/issues-pr/mgonzs13/yolo_ros)](https://github.com/mgonzs13/yolo_ros/pulls) [![Contributors](https://img.shields.io/github/contributors/mgonzs13/yolo_ros.svg)](https://github.com/mgonzs13/yolo_ros/graphs/contributors) [![Doxygen Deployment](https://github.com/mgonzs13/yolo_ros/actions/workflows/doxygen-deployment.yml/badge.svg?branch=main)](https://github.com/mgonzs13/yolo_ros/actions/workflows/doxygen-deployment.yml?branch=main)

| ROS 2 Distro |                          Branch                          |                                                                                                         Build status                                                                                                         |                                                               Docker Image                                                                |
| :----------: | :------------------------------------------------------: | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :---------------------------------------------------------------------------------------------------------------------------------------: |
|  **Humble**  | [`main`](https://github.com/mgonzs13/yolo_ros/tree/main) |  [![Humble Build](https://github.com/mgonzs13/yolo_ros/actions/workflows/humble-docker-build.yml/badge.svg?branch=main)](https://github.com/mgonzs13/yolo_ros/actions/workflows/humble-docker-build.yml?branch=main)   |  [![Docker Image](https://img.shields.io/badge/Docker%20Image%20-humble-blue)](https://hub.docker.com/r/mgons/yolo_ros/tags?name=humble)  |
|   **Iron**   | [`main`](https://github.com/mgonzs13/yolo_ros/tree/main) |     [![Iron Build](https://github.com/mgonzs13/yolo_ros/actions/workflows/iron-docker-build.yml/badge.svg?branch=main)](https://github.com/mgonzs13/yolo_ros/actions/workflows/iron-docker-build.yml?branch=main)      |    [![Docker Image](https://img.shields.io/badge/Docker%20Image%20-iron-blue)](https://hub.docker.com/r/mgons/yolo_ros/tags?name=iron)    |
|  **Jazzy**   | [`main`](https://github.com/mgonzs13/yolo_ros/tree/main) |    [![Jazzy Build](https://github.com/mgonzs13/yolo_ros/actions/workflows/jazzy-docker-build.yml/badge.svg?branch=main)](https://github.com/mgonzs13/yolo_ros/actions/workflows/jazzy-docker-build.yml?branch=main)    |   [![Docker Image](https://img.shields.io/badge/Docker%20Image%20-jazzy-blue)](https://hub.docker.com/r/mgons/yolo_ros/tags?name=jazzy)   |
|  **Kilted**  | [`main`](https://github.com/mgonzs13/yolo_ros/tree/main) |  [![Kilted Build](https://github.com/mgonzs13/yolo_ros/actions/workflows/kilted-docker-build.yml/badge.svg?branch=main)](https://github.com/mgonzs13/yolo_ros/actions/workflows/kilted-docker-build.yml?branch=main)   |  [![Docker Image](https://img.shields.io/badge/Docker%20Image%20-kilted-blue)](https://hub.docker.com/r/mgons/yolo_ros/tags?name=kilted)  |
| **Lyrical**  | [`main`](https://github.com/mgonzs13/yolo_ros/tree/main) | [![Lyrical Build](https://github.com/mgonzs13/yolo_ros/actions/workflows/lyrical-docker-build.yml/badge.svg?branch=main)](https://github.com/mgonzs13/yolo_ros/actions/workflows/lyrical-docker-build.yml?branch=main) | [![Docker Image](https://img.shields.io/badge/Docker%20Image%20-lyrical-blue)](https://hub.docker.com/r/mgons/yolo_ros/tags?name=lyrical) |

</div>

## Table of Contents

1. [Installation](#installation)
2. [Docker](#docker)
3. [Models](#models)
4. [Usage](#usage)
5. [Topics](#topics)
6. [Services](#services)
7. [Parameters](#parameters)
8. [Lifecycle Nodes](#lifecycle-nodes)
9. [Demos](#demos)
10. [Documentation](#documentation)
11. [License](#license)

## Installation

The nodes build as a standard ROS 2 `colcon` workspace. ONNX Runtime 1.20.0 is
downloaded automatically by `yolo_onnxruntime_vendor` at configure time — the
CPU tarball by default, the GPU tarball with `-DONNX_GPU=ON` — so the runtime is
not something you install by hand.

### Build

```shell
# libcurl/OpenSSL headers, for the Hugging Face Hub model download
sudo apt install libcurl4-openssl-dev libssl-dev

# Clone this repo
cd ~/ros2_ws/src
git clone https://github.com/mgonzs13/yolo_ros.git

# Install rosdep dependencies
cd ~/ros2_ws
rosdep install --from-paths src --ignore-src -r -y
```

Then build for the execution backend you want:

#### CPU

```shell
colcon build --symlink-install
source install/setup.bash
```

`yolo_onnxruntime_vendor` downloads the CPU ONNX Runtime and the node runs on
the CPU execution provider — no CUDA or cuDNN required.

#### CUDA / TensorRT

```shell
# cuDNN 9 for the CUDA 12 series (required by the ONNX Runtime 1.20 GPU build)
sudo apt install libcudnn9-cuda-12

colcon build --symlink-install --cmake-args -DONNX_GPU=ON
source install/setup.bash
```

`-DONNX_GPU=ON` fetches the ONNX Runtime GPU build, which links against the CUDA
12 series at run time, so the host also needs an NVIDIA driver and a CUDA 12.x
toolkit. TensorRT is optional: the default `provider: auto` tries the TensorRT
execution provider first and falls back to CUDA (then CPU), so without it
installed the model simply runs on the CUDA EP. `provider` and `device` are
chosen at runtime — see [Parameters](#parameters).

Launch from the workspace root so relative source paths resolve. For a fast
C++-only rebuild loop, rebuild just the package (select `yolo_msgs` first if the
messages changed — it must build before `yolo_ros`, and keep `-DONNX_GPU=ON` for
a GPU build):

```shell
colcon build --symlink-install --cmake-args -DONNX_GPU=ON --packages-select yolo_msgs
colcon build --symlink-install --cmake-args -DONNX_GPU=ON --packages-select yolo_ros
```

### Benchmark: CPU vs CUDA vs TensorRT

Same model (`yolo26m`), same 640x480 frames and same detection settings
(`imgsz=640`, `conf=0.7`, ~8.7 detections/frame); only the execution provider
changes. Latency is the median per-frame round trip with a single frame in
flight, throughput the saturated end-to-end rate.

| Backend | Provider | Latency (median) | Throughput | CPU | RSS | GPU memory |
| :-- | :-- | --: | --: | --: | --: | --: |
| CPU | CPU EP | 158 ms | 4.5 FPS | ~18.5 cores | 535 MiB | — |
| CUDA | CUDA EP (fp32) | 22.6 ms | 49 FPS | ~1.4 cores | 1002 MiB | 478 MiB |
| TensorRT | TensorRT EP (fp16) | 6.2 ms | 194 FPS | ~1.3 cores | 3234 MiB | 240 MiB |

### Jetson / aarch64 (offline build)

GPU inference on an aarch64 Jetson (Xavier, Orin) needs an ONNX Runtime built
with CUDA + TensorRT — the prebuilt x64 tarball cannot provide it.
`yolo_onnxruntime_vendor/scripts/` ships two tools for that, each documented in
its own file header:

- **`build_ort_aarch64.sh`** — runs on the robot and builds ONNX Runtime from
  source into a flat `lib/` + `include/` prefix. Consume it with
  `colcon build --cmake-args -DONNXRUNTIME_ROOT=<prefix>`, which overrides the
  prebuilt download. Set `ORT_CUDA_ARCH` for the target GPU (Xavier `72`,
  Orin `87`).
- **`prepare_offline_bundle.sh`** — runs on an internet-connected host and
  produces a self-contained tarball for a robot with no network: the ONNX
  Runtime source tree with its submodules, the mirrored CMake dependency
  archives, the ONNX models, a bundled CMake (ONNX Runtime 1.16+ needs
  CMake ≥ 3.26, while JetPack 6 / Ubuntu 22.04 ship 3.22) and both scripts. It
  prints the exact copy-paste sequence for the robot when it finishes.

Extract the bundle **outside** the colcon workspace — it carries
`COLCON_IGNORE` markers so colcon does not treat the ONNX Runtime tree as a
package. The robot build also passes
`-DFETCHCONTENT_SOURCE_DIR_YOLO_HFHUB=<bundle>/huggingface-hub-cpp` so
`yolo_hfhub_vendor` does not fetch `huggingface-hub-cpp` from the network.

## Docker

Build the yolo_ros docker image. Note that the bundled `Dockerfile` performs a
CPU build (plain `colcon build`, no `-DONNX_GPU=ON`); derive from it and add
`-DONNX_GPU=ON` yourself for a GPU image.

```shell
docker build -t yolo_ros .
```

Run the container. If you want to use CUDA, install the
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
and add `--gpus all` (and use a GPU-enabled image).

```shell
docker run -it --rm --gpus all yolo_ros
```

## Models

The C++ pipeline runs any Ultralytics-exported **ONNX** model whose output
matches one of the YOLO layouts below. The compatible model families are:

- [YOLOv3](https://docs.ultralytics.com/models/yolov3/) (`yolov3u`, Ultralytics'
  updated anchor-free head)
- [YOLOv5](https://docs.ultralytics.com/models/yolov5/) (`yolov5u`)
- [YOLOv8](https://docs.ultralytics.com/models/yolov8/)
- [YOLOv9](https://docs.ultralytics.com/models/yolov9/)
- [YOLOv10](https://docs.ultralytics.com/models/yolov10/)
- [YOLOv11](https://docs.ultralytics.com/models/yolo11/)
- [YOLOv12](https://docs.ultralytics.com/models/yolo12/)
- [YOLOv26](https://docs.ultralytics.com/models/yolo26/)

These families are verified end-to-end with this C++ node. Export
`yolov3u`/`yolov5u`, not `yolov3`/`yolov5` — Ultralytics only ships the updated
heads for those generations. YOLOv4, YOLOv6 and YOLOv7 appear in the Ultralytics
docs but have no downloadable weights, so they cannot be exported; YOLO-World
and YOLOE depend on open-vocabulary text prompts, which this C++ pipeline does
not implement.

Models are exported for one task — **detection** (`detect`), **instance
segmentation** (`segment`), **human pose** (`pose`), **oriented bounding box**
(`obb`) or **image classification** (`classify`) — depending on the tasks the
family's checkpoint provides (not every family offers every task). Export a
`.pt` checkpoint to ONNX with the `ultralytics` package — either with `uv` (no
environment needed) or a plain `pip install`:

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
  `Model::load_class_names()` reads at startup. The `coco.names` fallback is
  only used when a model has no metadata.
- Keep the exported file outside the repository and pass `model:=<path>` on
  every launch.
- `model_type` selects the pipeline (`YOLO`/`Detect`, `Segment`, `Pose`, `OBB`,
  `Classify`, case-insensitive). `auto` (the default) falls back to a filename
  heuristic: a path containing `segment` selects segmentation, `pose` selects
  pose, `obb` selects OBB, `cls`/`classify` selects classification, otherwise
  detection. Ultralytics names segmentation exports `*-seg.onnx` (not
  `-segment`), so the segment config sets `model_type: Segment` explicitly.
- The input size is fixed by the ONNX tensor (logged at startup), so the
  Python-only knobs `imgsz_height` / `imgsz_width` and
  `half` / `augment` / `agnostic_nms` / `retina_masks` were removed from the
  C++ node and its configs: NMS is baked at export and the pipeline runs FP32
  with no test-time augmentation.
- OBB models have no baked-NMS export (rotated NMS cannot be exported into the
  graph), so the C++ postprocessor performs its own per-class rotated NMS and
  `iou` re-tunes it. The rotation angle is published in
  `BoundingBox2D.center.theta` (radians) with `size` holding the rotated `w`/`h`.
- Classification exports bake the softmax into the graph (`output0` is `[1, N]`
  probabilities), so the postprocessor does **not** re-apply it. Top-`top_k`
  classes are published as detections with an **empty** bbox (image-level
  labels have no spatial extent).

### Download a model from the Hugging Face Hub

Instead of a local path, the node can fetch the model from the Hub at startup
via the `yolo_hfhub_vendor` package. Set `model_repo` + `model_filename` in the
matching `config/yolo*.yaml` section (or on the command line); the `model` path
is then ignored. The file is cached under `~/.cache/huggingface/hub` by default
(override with the optional `cache_dir` param) and reused unless
`force_download: true`:

```yaml
/yolo/yolo_node:
  ros__parameters:
    model_repo: unileon-robotics/YOLO26-ONNX # HF repo id
    model_filename: yolo26s.onnx # file inside that repo
    force_download: false
```

Or on the command line: `ros2 launch yolo_bringup yolo.launch.py
model_repo:=unileon-robotics/YOLO26-ONNX model_filename:=yolo26s.onnx`. Requires
the libcurl dev headers listed above; only used when
`model_repo`/`model_filename` are set. The node logs the model source as
`[huggingface]` (with repo/filename) or `[local]` (with the path) so the two are
easy to tell apart.

The shipped `config/yolo*.yaml` files already default to the
[`unileon-robotics/YOLO26-ONNX`](https://huggingface.co/unileon-robotics/YOLO26-ONNX)
mirror — ONNX exports of the
[`Ultralytics/YOLO26`](https://huggingface.co/Ultralytics/YOLO26) checkpoints
(25 files: `yolo26{n,s,m,l,x}` for detect / `-seg` / `-pose` / `-obb` / `-cls`,
exported with `imgsz=640 opset=12`). The first launch per model downloads it
(so it needs network access) and caches it; clear `model_repo` to fall back to
the local `model` path, or pass `model:=<path>` for a local file.

> **License note**: Ultralytics models and pretrained weights are **not** MIT
> licensed. They are released under the **AGPL-3.0** license (with commercial /
> enterprise licensing available from Ultralytics), so the MIT license of this
> repository does **not** cover them. Check the terms of the specific model you
> use at <https://ultralytics.com/license> — especially if you ship or deploy
> the model.

## Usage

Run from the workspace root (so the workspace is sourced as an overlay). Each
pipeline has its own launch file and namespace; the launch passes its
`config/yolo*.yaml` file to the nodes (plus any command-line overrides) and
makes no topic remaps.

### Object Detection

Standard detection, with tracking enabled by default (namespace `yolo`):

```shell
ros2 launch yolo_bringup yolo.launch.py
```

### Instance Segmentation

Namespace `yolo`; `model_type: Segment` is forced in `config/yolo_segment.yaml`:

```shell
ros2 launch yolo_bringup yolo_segment.launch.py
```

### Human Pose

Namespace `yolo`; `model_type: Pose` is forced in `config/yolo_pose.yaml`:

```shell
ros2 launch yolo_bringup yolo_pose.launch.py
```

### Oriented Bounding Box (OBB)

Namespace `yolo`; `model_type: OBB` is forced in `config/yolo_obb.yaml`. This
launch also starts the tracking and debug nodes:

```shell
ros2 launch yolo_bringup yolo_obb.launch.py
```

### Image Classification

Namespace `yolo`; `model_type: Classify` is forced in
`config/yolo_classify.yaml`. The default model is downloaded from the Hugging
Face Hub at configure time. No tracking/3D/debug nodes are started:

```shell
ros2 launch yolo_bringup yolo_classify.launch.py
```

<p align="center">
  <img src="./docs/media/rqt_graph_yolov8.png" alt="ROS 2 node graph" width="100%" />
</p>

### 3D Detection

Add `use_3d:=True` to the detection, segmentation, pose or OBB launch to also
start the C++ 3D detection node, which subscribes to the depth image +
`CameraInfo` and publishes `detections_3d`:

```shell
ros2 launch yolo_bringup yolo.launch.py use_3d:=True
```

For segmentation the depth ROI is driven by the mask polygon; for pose the 2D
keypoints are back-projected to 3D (`debug_kp_markers`). The 3D node can also
estimate the orientation of each box (an oriented bounding box fit by PCA to a
strided depth sample) when `enable_orientation` is set. When enabled,
`detections_3d` carries a non-identity quaternion in each box's
`center.orientation` and the box `size` is expressed along the object's own
axes.

## Topics

All topics are published under the launch namespace (default `yolo`):

- **detections**: Objects detected by YOLO using the RGB images. Each object
  contains a bounding box and a class name, plus a mask or a list of keypoints
  for segmentation/pose models.
- **tracking**: Objects detected and tracked by ByteTrack (or BoT-SORT with
  `tracker_type: botsort`). Each object is assigned a stable tracking ID.
- **detections_3d**: 3D objects detected (with `use_3d:=True`). YOLO results
  are used to crop the depth image and create 3D bounding boxes and keypoints.
- **debug_image**: Debug image showing the detected and tracked objects. It can
  be visualized with `rqt_image_view` or `rviz2`.
- **debug_bb_markers** / **debug_kp_markers**: RViz `MarkerArray`s driven by the
  3D box / keypoint stream (e.g. when the debug node reads `detections_3d`).

## Services

- **/yolo/enable**: Service to enable or disable the detection node at runtime.
  Accepts a boolean value (`std_srvs/SetBool`).

## Parameters

Configuration is file-driven: `yolo.launch.py` is the base launch and passes
its YAML params file plus any command-line overrides as
`parameters=[params_file, overrides]` (with no topic remaps). The
other four launch files are thin wrappers that include the base with their own
params file (`yolo_segment.yaml`, `yolo_pose.yaml`, `yolo_obb.yaml`,
`yolo_classify.yaml`). In addition, **every parameter can be overridden from
the command line**; an argument left unset keeps the YAML value:

```bash
ros2 launch yolo_bringup yolo.launch.py model:=/path/model.onnx threshold:=0.5 input_image_topic:=/camera/rgb/image_raw
ros2 launch yolo_bringup yolo_segment.launch.py threshold:=0.6
```

Override arguments use the upstream Python launch names where one existed:
`input_image_topic` → `image_topic`, `input_depth_topic` → `depth_image_topic`,
`input_depth_info_topic` → `depth_info_topic`, `tracker` → `tracker_type`. Every
other argument matches its parameter name. Because an empty value means "not
provided", a non-empty YAML string cannot be overridden to empty from the CLI.
Run `ros2 launch yolo_bringup <launch>.launch.py --show-args` for the full list.

Sections are keyed by the node's fully qualified name, so the key must include
the launch namespace (default `yolo`):

```yaml
/yolo/yolo_node:
  ros__parameters:
    model_type: auto
```

If you change `namespace:=`, update the matching config block names for any
value you do not override on the command line. The key parameters are listed
below; see `yolo_bringup/config/yolo*.yaml` for the complete set.

### Inference node (`yolo_node`)

- **model_type**: Pipeline to run: `YOLO`/`Detect`, `Segment`, `Pose`, `OBB`,
  `Classify` or `auto` (default: `auto`).
- **model**: Path to the ONNX model (default: machine-specific).
- **model_repo** / **model_filename** / **force_download** / **cache_dir**:
  Hugging Face Hub download (used instead of `model` when set). The shipped
  configs default to the `unileon-robotics/YOLO26-ONNX` mirror; clear
  `model_repo` to fall back to the local `model` path.
- **provider**: Execution provider: `auto` (TensorRT → CUDA → CPU fallback
  chain), or force `tensorrt`/`trt`, `cuda`, `cpu` (default: `auto`).
- **device**: CUDA/TensorRT device ordinal, e.g. `cuda:0`, `trt:1`, `1`
  (default: `cuda:0`). The `cuda:`/`trt:` prefix is accepted but `provider`
  selects the execution provider.
- **trt_fp16_enable**: TensorRT FP16 precision (default: `true`).
- **trt_engine_cache_enable**: Persist built TensorRT engines (default: `true`).
- **trt_engine_cache_path**: TensorRT engine cache base directory; empty →
  `~/.cache/yolo_ros/trt_engines/<model>` (default: empty).
- **threshold**: Detection confidence threshold (default: `0.7`).
- **iou**: IoU threshold for NMS. Re-tunes the C++ NMS for raw-output exports
  (segment/pose/OBB) and has no effect on baked-NMS models (default: `0.45`).
- **max_det**: Maximum number of detections per image (default: `300`).
- **enable**: Whether to start with inference enabled (default: `true`).
- **image_topic**: Input RGB image topic (default: machine-specific).
- **image_reliability**: QoS for the image topic: `0`=system default,
  `1`=Reliable, `2`=Best Effort (default: `2`).
- **n_threads**: CPU execution-provider threads; `-1` = auto (default: `-1`).
- **max_fps**: Cap the inference/publish rate in Hz; `0` = unlimited (default: `0`).
- **top_k**: Classification only — number of top classes to publish (default: `5`).

### Tracking node (`tracking_node`)

- **tracker_type**: Tracker implementation key: `bytetrack` (default) or
  `botsort`. BoT-SORT uses an XYWH Kalman filter and camera-motion
  compensation (no ReID).
- **image_topic** / **image_reliability**: Tracker image input and QoS.
- **track_high_thresh**: First-stage association threshold (default: `0.25`).
- **track_low_thresh**: Second-stage threshold for low-score matches (default: `0.1`).
- **new_track_thresh**: Score above which a new track is started (default: `0.25`).
- **track_buffer**: Frames a lost track is kept alive (default: `30`).
- **match_thresh**: Association similarity threshold (IoU/cost) (default: `0.8`).
- **fuse_score**: Fuse detection score with IoU cost for matching (default: `true`).
- **gmc_method** (BoT-SORT only): Camera-motion method: `none` (default),
  `sparseOptFlow`, `orb` or `ecc`.
- **gmc_downscale** (BoT-SORT only): Camera-motion downscale factor
  (default: `2`).

### 3D detection node (`detect_3d_node`)

- **target_frame**: Frame to transform the 3D boxes into (default: `base_link`).
- **depth_image_units_divisor**: Divisor to convert the depth image to meters
  (default: `1000`).
- **depth_image_topic** / **depth_info_topic** / **depth_image_reliability** /
  **depth_info_reliability**: Depth input, its `CameraInfo` and their QoS.
- **detections_topic**: 2D stream to lift — `tracking` or `detections`.
- **enable_orientation**: Estimate and publish the 3D box orientation (default: `false`).
- **min_seg_points_for_orientation**: Minimum valid depth points required per
  detection for orientation estimation (default: `20`).

### Debug node (`debug_node`)

- **image_topic** / **image_reliability**: Image input and QoS.
- **detections_topic**: 2D stream to draw — `tracking`, `detections` or
  `detections_3d` (default: `tracking`).
- **markers_topic**: 3D stream used to drive the RViz markers (default:
  `detections_3d`).

## Lifecycle Nodes

Every node is a lifecycle node. The executables call `configure()` and
`activate()` themselves in `main()`, so the nodes go active as soon as they are
launched. In the unconfigured and inactive states the node does no work: the
ONNX model is loaded and the image subscriber is created only on activation,
and both are torn down again on deactivation, keeping the idle CPU/VRAM
footprint low.

<p align="center">
  <img src="./docs/media/rqt_graph_yolov8_3d.png" alt="ROS 2 node graph with 3D detection" width="100%" />
</p>

## Demos

> The detection, segmentation, pose and 3D demos below were recorded with the
> earlier Python implementation. The OBB and classification demos are not yet
> recorded — drop the recordings at the listed paths to have them show up here.

### Object Detection

Standard behavior including ByteTrack (or BoT-SORT) object tracking.

```shell
ros2 launch yolo_bringup yolo.launch.py
```

[![](https://drive.google.com/thumbnail?authuser=0&sz=w1280&id=1gTQt6soSIq1g2QmK7locHDiZ-8MqVl2w)](https://drive.google.com/file/d/1gTQt6soSIq1g2QmK7locHDiZ-8MqVl2w/view?usp=sharing)

### Instance Segmentation

Instance masks are the borders of the detected objects, not all the pixels
inside the masks.

```shell
ros2 launch yolo_bringup yolo_segment.launch.py
```

[![](https://drive.google.com/thumbnail?authuser=0&sz=w1280&id=1dwArjDLSNkuOGIB0nSzZR6ABIOCJhAFq)](https://drive.google.com/file/d/1dwArjDLSNkuOGIB0nSzZR6ABIOCJhAFq/view?usp=sharing)

### Human Pose

Visible persons are detected along with their skeleton keypoints.

```shell
ros2 launch yolo_bringup yolo_pose.launch.py
```

[![](https://drive.google.com/thumbnail?authuser=0&sz=w1280&id=1pRy9lLSXiFEVFpcbesMCzmTMEoUXGWgr)](https://drive.google.com/file/d/1pRy9lLSXiFEVFpcbesMCzmTMEoUXGWgr/view?usp=sharing)

### Oriented Bounding Box

Rotated boxes are estimated for oriented objects.

```shell
ros2 launch yolo_bringup yolo_obb.launch.py
```

<!-- Demo recording pending: add ./docs/media/demo_obb.gif to display it here.
<p align="center">
  <img src="./docs/media/demo_obb.gif" alt="Oriented bounding box demo" width="100%" />
</p>
-->

### Image Classification

Image-level ImageNet-1k labels are published as detections with an empty bbox.

```shell
ros2 launch yolo_bringup yolo_classify.launch.py
```

<!-- Demo recording pending: add ./docs/media/demo_classify.gif to display it here.
<p align="center">
  <img src="./docs/media/demo_classify.gif" alt="Image classification demo" width="100%" />
</p>
-->

### 3D Object Detection

The 3D bounding boxes are calculated by filtering the depth image data from an
RGB-D camera using the 2D bounding box.

```shell
ros2 launch yolo_bringup yolo.launch.py use_3d:=True
```

[![](https://drive.google.com/thumbnail?authuser=0&sz=w1280&id=1ZcN_u9RB9_JKq37mdtpzXx3b44tlU-pr)](https://drive.google.com/file/d/1ZcN_u9RB9_JKq37mdtpzXx3b44tlU-pr/view?usp=sharing)

## Documentation

The C++ API reference is generated with Doxygen and published to GitHub Pages
on every release:

- Latest: <https://mgonzs13.github.io/yolo_ros/>

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

- `yolo_ros`, `yolo_msgs`, `yolo_bringup` and `yolo_onnxruntime_vendor` are
  licensed under **MIT**. See each package's `LICENSE` file; third-party notices
  are installed with the applicable packages (`yolo_ros/THIRD_PARTY_NOTICES.md`).

The C++ pipeline adapts behavior from the original `yolo_ros` Python nodes;
those contributions were authorized by their copyright holder for release in the
MIT-licensed C++ pipeline (see `THIRD_PARTY_NOTICES.md`).

Model weights and exported ONNX files are separate artifacts and remain subject
to their respective licenses; the MIT license for the C++ pipeline does not
relicense them. In particular, Ultralytics models are AGPL-3.0 — see the
[Models](#models) section above.
