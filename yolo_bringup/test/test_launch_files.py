# Copyright (c) 2026 Alejandro González Cantón
# SPDX-License-Identifier: MIT

import importlib.util
import os

import pytest
import yaml
from launch import LaunchContext, LaunchDescription, Substitution
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration

LAUNCH_DIR = os.path.join(os.path.dirname(__file__), "..", "launch")
LAUNCH_FILES = {
    "yolo": "yolo.launch.py",
}


def _load(name):
    path = os.path.join(LAUNCH_DIR, LAUNCH_FILES[name])
    spec = importlib.util.spec_from_file_location(f"{name}_launch", path)
    assert spec is not None and spec.loader is not None, path
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _unsub(value):
    """Evaluate the substitutions launch_ros wraps plain params values in."""
    if isinstance(value, Substitution):
        return value.perform(LaunchContext())
    if isinstance(value, dict):
        return {_unsub(key): _unsub(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        evaluated = [_unsub(item) for item in value]
        if len(evaluated) == 1:
            return evaluated[0]
        return type(value)(evaluated)
    return value


class _FakeContext:
    def __init__(self, configs):
        self.launch_configurations = configs


def _description(name):
    return _load(name).generate_launch_description()


def _declared(name):
    return {
        argument.name
        for argument, _ in _description(
            name
        ).get_launch_arguments_with_include_launch_description_actions()
    }


@pytest.mark.parametrize("name", sorted(LAUNCH_FILES))
def test_launch_description_builds(name):
    assert isinstance(_description(name), LaunchDescription)


@pytest.mark.parametrize("name", sorted(LAUNCH_FILES))
def test_catalogue_args_are_declared(name):
    assert {
        "model",
        "model_type",
        "threshold",
        "input_image_topic",
        "image_reliability",
        "tracker",
        "input_depth_topic",
        "input_depth_info_topic",
    } <= _declared(name)


@pytest.mark.parametrize("name", sorted(LAUNCH_FILES))
def test_base_flags_are_declared(name):
    assert {
        "params_file",
        "namespace",
        "use_tracking",
        "use_3d",
        "use_debug",
    } <= _declared(name)


def test_base_flag_defaults():
    defaults = {
        entity.name: _unsub(entity.default_value)
        for entity in _description("yolo").entities
        if isinstance(entity, DeclareLaunchArgument)
    }
    assert defaults["use_tracking"] == "True"
    assert defaults["use_3d"] == "False"
    assert defaults["use_debug"] == "True"
    assert "yolo.yaml" in str(defaults["params_file"])


def test_launch_setup_layers_overrides_for_yolo_node():
    module = _load("yolo")
    context = _FakeContext(
        {"model": "/x.onnx", "threshold": "0.5", "input_image_topic": "/cam"}
    )
    params_file = LaunchConfiguration("params_file")
    nodes = module._launch_setup(
        context,
        params_file,
        LaunchConfiguration("namespace"),
        LaunchConfiguration("use_tracking"),
        LaunchConfiguration("use_3d"),
        LaunchConfiguration("use_debug"),
    )
    by_name = {node._Node__node_name: node for node in nodes}
    params = by_name["yolo_node"]._Node__parameters
    assert len(params) == 2
    # normalize_parameters wraps the YAML substitution in a ParameterFile, so
    # check it still wraps the exact params_file we handed to _launch_setup.
    assert params[0].param_file[0] is params_file
    # normalize_parameter_dict yaml.dumps string values (adds a trailing
    # "...\n" document marker), so decode them back for the comparison.
    overrides = {
        key: yaml.safe_load(value) if isinstance(value, str) else value
        for key, value in _unsub(params[1]).items()
    }
    assert overrides == {
        "model": "/x.onnx",
        "threshold": 0.5,
        "image_topic": "/cam",
    }
    # The always-on inference node has no condition; the other three gate on
    # their use_* flag.
    assert by_name["yolo_node"].condition is None
    for name in ("tracking_node", "detect_3d_node", "debug_node"):
        assert isinstance(by_name[name].condition, IfCondition)


def test_launch_setup_selects_tracker_from_params_file(tmp_path):
    pipeline = tmp_path / "yolo.yaml"
    pipeline.write_text(
        "/yolo/tracking_node:\n  ros__parameters:\n    tracker: botsort_reid\n"
    )
    module = _load("yolo")
    context = _FakeContext(
        {"params_file": str(pipeline), "namespace": "yolo", "with_reid": "true"}
    )
    params_file = LaunchConfiguration("params_file")
    nodes = module._launch_setup(
        context,
        params_file,
        LaunchConfiguration("namespace"),
        LaunchConfiguration("use_tracking"),
        LaunchConfiguration("use_3d"),
        LaunchConfiguration("use_debug"),
    )
    params = {node._Node__node_name: node for node in nodes}[
        "tracking_node"
    ]._Node__parameters
    # pipeline config -> per-tracker file -> CLI overrides
    assert len(params) == 3
    assert params[0].param_file[0] is params_file
    assert str(_unsub(params[1].param_file[0])).endswith(
        os.path.join("config", "trackers", "botsort_reid.yaml")
    )
    overrides = {
        key: yaml.safe_load(value) if isinstance(value, str) else value
        for key, value in _unsub(params[2]).items()
    }
    assert overrides == {"with_reid": True}


def test_launch_setup_tracker_arg_overrides_params_file(tmp_path):
    pipeline = tmp_path / "yolo.yaml"
    pipeline.write_text(
        "/yolo/tracking_node:\n  ros__parameters:\n    tracker: bytetrack\n"
    )
    module = _load("yolo")
    context = _FakeContext(
        {"params_file": str(pipeline), "namespace": "yolo", "tracker": "botsort"}
    )
    nodes = module._launch_setup(
        context,
        LaunchConfiguration("params_file"),
        LaunchConfiguration("namespace"),
        LaunchConfiguration("use_tracking"),
        LaunchConfiguration("use_3d"),
        LaunchConfiguration("use_debug"),
    )
    params = {node._Node__node_name: node for node in nodes}[
        "tracking_node"
    ]._Node__parameters
    assert str(_unsub(params[1].param_file[0])).endswith(
        os.path.join("config", "trackers", "botsort.yaml")
    )
