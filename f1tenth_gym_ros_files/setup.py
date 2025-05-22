from setuptools import setup, find_packages
import os
from glob import glob

package_name = "f1tenth_gym_ros"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(
        include=[
            package_name,
            "uci_f1tenth_workshop*",
            "uci_f1tenth_workshop.dreamer_node*",
            "uci_f1tenth_workshop.gyms*",
            "uci_f1tenth_workshop.gyms.racecar_gym.scenarios*",
        ]
    ),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.py")),
        (os.path.join("share", package_name, "launch"), glob("launch/*.xacro")),
        (os.path.join("share", package_name, "launch"), glob("launch/*.rviz")),
        (os.path.join("share", package_name, "config"), glob("config/*.yaml")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Billy Zheng",
    maintainer_email="billyzheng.bz@gmail.com",
    description="Bridge for using f1tenth_gym in ROS2",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "gym_bridge = f1tenth_gym_ros.gym_bridge:main",
            "ros2_agent = uci_f1tenth_workshop.dreamer_node.agents.ros2_agent:main",
        ],
    },
)
