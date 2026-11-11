"""Launch Orbbec publishers + dual_view CV (rack / arm overlay).

dual_view: arm/pose on cam0 (with optional center zoom), rack CV on cam1.
Optional cam1-only rack overlay window.
"""

from __future__ import annotations

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo, TimerAction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description() -> LaunchDescription:
    pkg_share = get_package_share_directory("orbbec_dual_sync")
    default_config = os.path.join(pkg_share, "config", "dual_sync.yaml")

    config = LaunchConfiguration("config")
    enable_rack_overlay = LaunchConfiguration("enable_rack_overlay")
    enable_dual_view = LaunchConfiguration("enable_dual_view")
    rack_state_mode = LaunchConfiguration("rack_state_mode")
    publish_rate_hz = ParameterValue(
        LaunchConfiguration("publish_rate_hz"), value_type=float
    )
    rate_override = {"publish_rate_hz": publish_rate_hz}

    cam0 = Node(
        package="orbbec_dual_sync",
        executable="orbbec_camera_publisher",
        name="orbbec_cam0_pub",
        output="screen",
        parameters=[config, rate_override],
        respawn=True,
        respawn_delay=3.0,
    )
    cam1 = Node(
        package="orbbec_dual_sync",
        executable="orbbec_camera_publisher",
        name="orbbec_cam1_pub",
        output="screen",
        parameters=[config, rate_override],
        respawn=True,
        respawn_delay=3.0,
    )
    cam2 = Node(
        package="orbbec_dual_sync",
        executable="orbbec_camera_publisher",
        name="orbbec_cam2_pub",
        output="screen",
        parameters=[config],
        respawn=True,
        respawn_delay=3.0,
    )
    dual_view = Node(
        package="orbbec_dual_sync",
        executable="orbbec_dual_view",
        name="orbbec_dual_view",
        output="screen",
        parameters=[
            config,
            {
                "enable_rack_cv": True,
                "state_mode": rack_state_mode,
            },
        ],
        respawn=True,
        respawn_delay=3.0,
        condition=IfCondition(enable_dual_view),
    )
    rack_overlay = Node(
        package="orbbec_dual_sync",
        executable="orbbec_rack_overlay",
        name="orbbec_rack_overlay",
        output="screen",
        parameters=[
            config,
            {"state_mode": rack_state_mode},
        ],
        respawn=True,
        respawn_delay=3.0,
        condition=IfCondition(enable_rack_overlay),
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "config",
                default_value=default_config,
                description=(
                    "YAML params installed under "
                    "share/orbbec_dual_sync/config/dual_sync.yaml"
                ),
            ),
            DeclareLaunchArgument(
                "enable_rack_overlay",
                default_value="true",
                description=(
                    "Subscribe to cam1 color and show full-rack OPEN/CLOSED "
                    "window (does not open Orbbec USB). DEFAULT true."
                ),
            ),
            DeclareLaunchArgument(
                "enable_dual_view",
                default_value="true",
                description=(
                    "Show side-by-side cam0|cam1 with arm/pose on cam0 and "
                    "rack OPEN/CLOSED CV on cam1. DEFAULT true."
                ),
            ),
            DeclareLaunchArgument(
                "rack_state_mode",
                default_value="classify",
                description=(
                    "Rack CV mode for dual_view + rack_overlay: classify "
                    "(JudgeAgent OPEN/CLOSED), force_open, or force_closed."
                ),
            ),
            DeclareLaunchArgument(
                "publish_rate_hz",
                default_value="6.0",
                description=(
                    "Timer rate for both camera publishers (default matches "
                    "dual_sync.yaml)."
                ),
            ),
            LogInfo(
                msg=(
                    "ROS publishers + DUAL VIEW "
                    "(cam0 arm/pose zoom | cam1 rack CV). "
                    "Pin rack label: rack_state_mode:=force_closed|force_open. "
                    "Tune zoom in dual_sync.yaml: cam0_zoom"
                )
            ),
            cam0,
            TimerAction(period=20.0, actions=[cam1]),
            TimerAction(period=36.0, actions=[cam2]),
            TimerAction(period=52.0, actions=[dual_view]),
            TimerAction(period=28.0, actions=[rack_overlay]),
        ]
    )
