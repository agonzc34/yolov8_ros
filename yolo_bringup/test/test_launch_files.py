# Copyright (c) 2026 Alejandro González Cantón
# SPDX-License-Identifier: MIT

import importlib.util
import os

import pytest
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription

LAUNCH_DIR = os.path.join(os.path.dirname(__file__), "..", "launch")
LAUNCH_FILES = {
    "yolo": "yolo.launch.py",
    "segment": "yolo_segment.launch.py",
    "pose": "yolo_pose.launch.py",
    "obb": "yolo_obb.launch.py",
    "classify": "yolo_classify.launch.py",
}
WRAPPERS = ["segment", "pose", "obb", "classify"]


def _load(name):
    path = os.path.join(LAUNCH_DIR, LAUNCH_FILES[name])
    spec = importlib.util.spec_from_file_location(f"{name}_launch", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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


@pytest.mark.parametrize("name", WRAPPERS)
def test_wrappers_include_the_base(name):
    assert any(
        isinstance(entity, IncludeLaunchDescription)
        for entity in _description(name).entities
    )
