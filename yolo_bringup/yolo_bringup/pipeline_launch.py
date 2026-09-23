# Copyright (c) 2026 Alejandro González Cantón
# SPDX-License-Identifier: MIT

"""Factory for the per-pipeline launch wrappers over yolo.launch.py.

Every ``yolo_<pipeline>.launch.py`` is the same launch (include yolo.launch.py
with its own params file and the ``use_*`` flags) apart from the pipeline name,
so they are generated from this one helper instead of being copied.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration

#: use_* flags forwarded to the base launch by the spatial pipelines, with the
#: default declared for each and the description shown by --show-args.
_SPATIAL_FLAGS = {
    "use_tracking": (
        "True",
        "Whether to enable the ByteTrack tracking node (True/False)",
    ),
    "use_3d": (
        "False",
        "Whether to enable the 3D detection node (needs a depth image + "
        "CameraInfo, see the config file)",
    ),
    "use_debug": ("True", "Whether to enable the debug/visualization node (True/False)"),
}


def pipeline_launch(pipeline: str, description: str, spatial: bool = True):
    """Return a ``generate_launch_description`` for one pipeline wrapper.

    ``spatial`` declares the use_tracking/use_3d/use_debug arguments and
    forwards them (detection/segmentation/pose/OBB). When False (classification)
    the three flags are forwarded as fixed values instead.
    """

    def generate_launch_description():
        base = os.path.join(
            get_package_share_directory("yolo_bringup"), "launch", "yolo.launch.py"
        )

        params_file = LaunchConfiguration("params_file")
        namespace = LaunchConfiguration("namespace")

        arguments = [
            DeclareLaunchArgument(
                "params_file",
                default_value=os.path.join(
                    get_package_share_directory("yolo_bringup"),
                    "config",
                    f"yolo_{pipeline}.yaml",
                ),
                description=description,
            ),
            DeclareLaunchArgument(
                "namespace",
                default_value="yolo",
                description="Namespace for the nodes",
            ),
        ]

        if spatial:
            arguments += [
                DeclareLaunchArgument(name, default_value=default, description=text)
                for name, (default, text) in _SPATIAL_FLAGS.items()
            ]
            launch_arguments = {
                "params_file": params_file,
                "namespace": namespace,
                **{name: LaunchConfiguration(name) for name in _SPATIAL_FLAGS},
            }
        else:
            launch_arguments = {
                "params_file": params_file,
                "namespace": namespace,
                "use_tracking": "False",
                "use_3d": "False",
                "use_debug": "True",
            }

        arguments.append(
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(base),
                launch_arguments=launch_arguments.items(),
            )
        )
        return LaunchDescription(arguments)

    return generate_launch_description
