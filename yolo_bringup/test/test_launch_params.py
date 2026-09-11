# Copyright (c) 2026 Alejandro González Cantón
# SPDX-License-Identifier: MIT

import pytest
from launch.substitutions import LaunchConfiguration

from yolo_bringup.launch_params import (
    build_overrides,
    declare_param_arguments,
    node_parameters,
    param_arg_names,
)


class FakeContext:
    def __init__(self, configs):
        self.launch_configurations = configs


def test_types_are_converted():
    context = FakeContext(
        {
            "model": "/x.onnx",
            "threshold": "0.5",
            "max_det": "300",
            "enable": "false",
            "input_image_topic": "/cam",
        }
    )
    overrides = build_overrides(context, "yolo_node")
    assert overrides["model"] == "/x.onnx"
    assert overrides["threshold"] == pytest.approx(0.5)
    assert overrides["max_det"] == 300
    assert overrides["enable"] is False
    assert overrides["image_topic"] == "/cam"


def test_empty_values_are_omitted():
    context = FakeContext({"model": "", "threshold": "", "iou": ""})
    assert build_overrides(context, "yolo_node") == {}


def test_bool_variants():
    for raw, expected in [
        ("true", True),
        ("1", True),
        ("yes", True),
        ("on", True),
        ("FALSE", False),
        ("0", False),
        ("no", False),
        ("off", False),
    ]:
        context = FakeContext({"enable": raw})
        assert build_overrides(context, "yolo_node")["enable"] is expected


def test_invalid_value_raises_with_arg_name():
    context = FakeContext({"threshold": "high"})
    with pytest.raises(RuntimeError, match="threshold"):
        build_overrides(context, "yolo_node")


def test_invalid_bool_raises_with_arg_name():
    context = FakeContext({"enable": "maybe"})
    with pytest.raises(RuntimeError, match="enable"):
        build_overrides(context, "yolo_node")


def test_aliases_map_to_param_names():
    context = FakeContext(
        {
            "tracker": "bytetrack",
            "input_image_topic": "/cam",
            "input_depth_topic": "/depth",
            "input_depth_info_topic": "/depth_info",
        }
    )
    assert build_overrides(context, "tracking_node")["tracker_type"] == "bytetrack"
    assert build_overrides(context, "tracking_node")["image_topic"] == "/cam"
    assert build_overrides(context, "detect_3d_node")["depth_image_topic"] == "/depth"
    assert build_overrides(context, "detect_3d_node")["depth_info_topic"] == "/depth_info"


def test_shared_arg_applied_to_each_target_node():
    context = FakeContext({"image_reliability": "1", "detections_topic": "detections"})
    assert build_overrides(context, "yolo_node")["image_reliability"] == 1
    assert build_overrides(context, "tracking_node")["image_reliability"] == 1
    assert build_overrides(context, "debug_node")["image_reliability"] == 1
    assert build_overrides(context, "detect_3d_node")["detections_topic"] == "detections"
    assert build_overrides(context, "debug_node")["detections_topic"] == "detections"


def test_arg_names_are_unique():
    names = param_arg_names(["yolo_node", "tracking_node", "debug_node"])
    assert names == sorted(set(names))
    assert names.count("input_image_topic") == 1
    assert names.count("image_reliability") == 1
    assert "model" in names


def test_declare_returns_one_per_arg_name():
    nodes = ["yolo_node", "tracking_node", "detect_3d_node", "debug_node"]
    assert len(declare_param_arguments(nodes)) == len(param_arg_names(nodes))


def test_node_parameters_layers_over_file():
    params_file = LaunchConfiguration("params_file")
    context = FakeContext({"threshold": "0.5"})
    result = node_parameters(params_file, context, "yolo_node")
    assert result[0] is params_file
    assert result[1] == {"threshold": 0.5}


def test_unknown_node_raises():
    with pytest.raises(KeyError, match="nope"):
        build_overrides(FakeContext({}), "nope")
