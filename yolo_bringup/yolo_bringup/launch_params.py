# Copyright (c) 2026 Alejandro González Cantón
# SPDX-License-Identifier: MIT

"""Command-line overrides for the yolo_bringup YAML params files.

Every ROS parameter declared by the C++ nodes can be overridden from the
command line, e.g.::

    ros2 launch yolo_bringup yolo.launch.py model:=/x.onnx threshold:=0.5

An argument that is not passed keeps the value from the YAML params file, so
the YAML remains the source of defaults. The empty string is the "not provided"
sentinel; a non-empty YAML string therefore cannot be overridden *to* empty.

Argument names mirror the upstream Python launch where one existed
(``input_image_topic``, ``input_depth_topic``, ``input_depth_info_topic``,
``tracker``); every other argument has the same name as its parameter.
"""

from dataclasses import dataclass
from typing import Optional

from launch.actions import DeclareLaunchArgument


@dataclass(frozen=True)
class ParamSpec:
    """A node parameter exposed as a launch argument."""

    name: str
    type: type
    alias: Optional[str] = None

    @property
    def arg(self) -> str:
        """The launch-argument name (upstream alias when one exists)."""
        return self.alias or self.name


#: Node name (Node(name=...)) -> parameters it declares, in declaration order.
NODE_PARAMS = {
    "yolo_node": (
        ParamSpec("model_type", str),
        ParamSpec("model", str),
        ParamSpec("model_repo", str),
        ParamSpec("model_filename", str),
        ParamSpec("cache_dir", str),
        ParamSpec("force_download", bool),
        ParamSpec("device", str),
        ParamSpec("provider", str),
        ParamSpec("trt_fp16_enable", bool),
        ParamSpec("trt_engine_cache_enable", bool),
        ParamSpec("trt_engine_cache_path", str),
        ParamSpec("threshold", float),
        ParamSpec("iou", float),
        ParamSpec("max_det", int),
        ParamSpec("enable", bool),
        ParamSpec("image_reliability", int),
        ParamSpec("image_topic", str, "input_image_topic"),
        ParamSpec("n_threads", int),
        ParamSpec("max_fps", int),
        ParamSpec("top_k", int),
    ),
    "tracking_node": (
        ParamSpec("image_reliability", int),
        ParamSpec("image_topic", str, "input_image_topic"),
        ParamSpec("tracker_type", str, "tracker"),
        ParamSpec("track_high_thresh", float),
        ParamSpec("track_low_thresh", float),
        ParamSpec("new_track_thresh", float),
        ParamSpec("track_buffer", int),
        ParamSpec("match_thresh", float),
        ParamSpec("fuse_score", bool),
        ParamSpec("gmc_method", str),
        ParamSpec("gmc_downscale", int),
    ),
    "detect_3d_node": (
        ParamSpec("target_frame", str),
        ParamSpec("depth_image_units_divisor", int),
        ParamSpec("depth_image_reliability", int),
        ParamSpec("depth_info_reliability", int),
        ParamSpec("depth_image_topic", str, "input_depth_topic"),
        ParamSpec("depth_info_topic", str, "input_depth_info_topic"),
        ParamSpec("detections_topic", str),
        ParamSpec("enable_orientation", bool),
        ParamSpec("min_seg_points_for_orientation", int),
    ),
    "debug_node": (
        ParamSpec("image_reliability", int),
        ParamSpec("image_topic", str, "input_image_topic"),
        ParamSpec("detections_topic", str),
        ParamSpec("markers_topic", str),
        ParamSpec("marker_lifetime", float),
    ),
}

_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off"}


def _selected_node_params(node_names):
    """Yield (node_name, ParamSpec); raise KeyError on an unknown node."""
    for node_name in node_names:
        if node_name not in NODE_PARAMS:
            raise KeyError(f"unknown node '{node_name}'")
        for spec in NODE_PARAMS[node_name]:
            yield node_name, spec


def param_arg_names(node_names) -> list:
    """Sorted unique launch-argument names for the given nodes."""
    return sorted({spec.arg for _, spec in _selected_node_params(node_names)})


def declare_param_arguments(node_names) -> list:
    """One DeclareLaunchArgument per unique argument, defaulting to ""."""
    targets = {}
    for node_name, spec in _selected_node_params(node_names):
        targets.setdefault(spec.arg, set()).add(f"{node_name}.{spec.name}")
    return [
        DeclareLaunchArgument(
            arg_name,
            default_value="",
            description=(
                f"Override {', '.join(sorted(targets[arg_name]))}. "
                "Empty keeps the value from the YAML params file."
            ),
        )
        for arg_name in sorted(targets)
    ]


def _convert(arg_name: str, raw: str, value_type: type):
    if value_type is bool:
        lowered = raw.strip().lower()
        if lowered in _TRUE:
            return True
        if lowered in _FALSE:
            return False
        raise RuntimeError(
            f"invalid value for '{arg_name}': {raw!r} (expected a boolean)"
        )
    try:
        return value_type(raw)
    except ValueError as exc:
        raise RuntimeError(
            f"invalid value for '{arg_name}': {raw!r} "
            f"(expected {value_type.__name__})"
        ) from exc


def build_overrides(context, node_name: str) -> dict:
    """Assemble {param_name: value} for the node from provided launch args."""
    if node_name not in NODE_PARAMS:
        raise KeyError(f"unknown node '{node_name}'")
    overrides = {}
    for spec in NODE_PARAMS[node_name]:
        raw = context.launch_configurations.get(spec.arg, "")
        if raw is None or str(raw) == "":
            continue
        overrides[spec.name] = _convert(spec.arg, str(raw), spec.type)
    return overrides


def node_parameters(params_file, context, node_name: str) -> list:
    """Parameters list for a Node: the YAML file plus the CLI overrides."""
    return [params_file, build_overrides(context, node_name)]
