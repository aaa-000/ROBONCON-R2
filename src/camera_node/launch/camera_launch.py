from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    rviz_path = os.path.join(
        get_package_share_directory("camera_node"),
        "config",
        "camera.rviz"
    )

    visual = Node(  # 配置一个节点的启动
        package='rviz2',  # 节点所在的功能包
        executable='rviz2',  # 节点的可执行文件名
        output='screen',
        # emulate_tty=True,
        arguments=["-d", rviz_path]
    )

    camera = Node(
        package="camera_node",
        executable="camera_node",
        output='screen',
        name="camera"
    )

    imgprocess = Node(
        package="imgprocess_node",
        executable="imgprocess_node",
        output='screen',
        name="imgprocess"
    )
    return LaunchDescription([camera, imgprocess, visual])