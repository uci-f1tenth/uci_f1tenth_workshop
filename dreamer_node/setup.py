from setuptools import setup, find_packages

package_name = "dreamer_node"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="F1tenth@uci",
    maintainer_email="ucif1tenth@uci.edu",
    description="Dreamer v3 implementation for f1tenth Reinforcement Learning",
    license="MIT",
    tests_require=["pytest"],
    test_suite="tests",
    entry_points={
        "console_scripts": [
            "dreamer_node = agents.dreamer_agent:main",
        ],
    },
)
