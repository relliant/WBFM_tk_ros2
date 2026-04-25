from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription([
        DeclareLaunchArgument('policy_path', default_value=''),
        DeclareLaunchArgument('manifest_path', default_value=''),
        DeclareLaunchArgument('motion_file', default_value=''),
        DeclareLaunchArgument('device', default_value='cpu'),
        Node(
            package='joy',
            executable='joy_node',
            name='joy_node',
            parameters=[{'device': '/dev/input/js0'}],
            remappings=[('joy', '/sbus_data')],
            output='screen',
        ),
        # Node(
        #     package='tienkung_motion_source',
        #     executable='motion_source_node',
        #     name='tienkung_motion_source',
        #     output='screen',
        #     parameters=[
        #         {
        #             'motion_file': LaunchConfiguration('motion_file'),
        #         }
        #     ],
        # ),
        Node(
            package='tienkung_policy_runner',
            executable='policy_runner_node',
            name='tienkung_policy_runner',
            output='screen',
            parameters=[
                {
                    'policy_path': LaunchConfiguration('policy_path'),
                    'manifest_path': LaunchConfiguration('manifest_path'),
                    'device': LaunchConfiguration('device'),
                }
            ],
        ),
    ])
