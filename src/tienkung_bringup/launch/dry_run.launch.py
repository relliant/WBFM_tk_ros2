"""dry_run.launch.py

遥控器控制的安全验证启动文件：
  STOP   → 不动
  ZERO   → 缓慢归零到默认站姿（发真实电机指令）
  POLICY → 只推理，不发指令（dry run）

用法：
  ros2 launch tienkung_bringup dry_run.launch.py \
      policy_path:=/path/to/policy.onnx

可选参数：
  manifest_path      -- policy manifest yaml (default: '')
  device             -- cpu / cuda (default: cpu)
  policy_hz          -- 推理频率 Hz (default: 50.0)
  zero_duration      -- ZERO 模式最短保持秒数，满足后才允许切 POLICY (default: 2.0)
  report_interval    -- 打印统计的间隔秒数 (default: 5.0)
  leg_command_topic  -- 腿部电机指令 topic (default: /leg/cmd_ctrl)
  arm_command_topic  -- 手臂电机指令 topic (default: /arm/cmd_ctrl)
  waist_command_topic -- 腰部电机指令 topic (default: /waist/cmd_pos)
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription([
        DeclareLaunchArgument("policy_path",         default_value=""),
        DeclareLaunchArgument("manifest_path",       default_value=""),
        DeclareLaunchArgument("device",              default_value="cpu"),
        DeclareLaunchArgument("policy_hz",           default_value="50.0"),
        DeclareLaunchArgument("zero_duration",       default_value="2.0"),
        DeclareLaunchArgument("report_interval",     default_value="5.0"),
        DeclareLaunchArgument("motion_file",         default_value=""),
        DeclareLaunchArgument("joy_device",          default_value="/dev/input/js0"),
        DeclareLaunchArgument("leg_command_topic",   default_value="/leg/cmd_ctrl"),
        DeclareLaunchArgument("arm_command_topic",   default_value="/arm/cmd_ctrl"),
        DeclareLaunchArgument("waist_command_topic", default_value="/waist/cmd_pos"),

        # ── 遥控器驱动（发布 /sbus_data）
        Node(
            package="joy",
            executable="joy_node",
            name="joy_node",
            output="screen",
            parameters=[{"device": LaunchConfiguration("joy_device")}],
            remappings=[("joy", "/sbus_data")],
        ),

        # ── Motion source（发布 /tienkung/motion_reference）
        # Node(
        #     package="tienkung_motion_source",
        #     executable="motion_source_node",
        #     name="tienkung_motion_source",
        #     output="screen",
        #     parameters=[{"motion_file": LaunchConfiguration("motion_file")}],
        # ),

        # ── Dry run 推理节点
        Node(
            package="tienkung_policy_runner",
            executable="dry_run_node",
            name="tienkung_dry_run",
            output="screen",
            parameters=[{
                "policy_path":         LaunchConfiguration("policy_path"),
                "manifest_path":       LaunchConfiguration("manifest_path"),
                "device":              LaunchConfiguration("device"),
                "policy_hz":           LaunchConfiguration("policy_hz"),
                "zero_duration_sec":   LaunchConfiguration("zero_duration"),
                "report_interval_sec": LaunchConfiguration("report_interval"),
                "leg_command_topic":   LaunchConfiguration("leg_command_topic"),
                "arm_command_topic":   LaunchConfiguration("arm_command_topic"),
                "waist_command_topic": LaunchConfiguration("waist_command_topic"),
            }],
        ),
    ])
