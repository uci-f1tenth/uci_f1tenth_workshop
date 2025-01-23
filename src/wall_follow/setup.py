from setuptools import setup, find_packages

package_name = 'wall_follow'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='zzangupenn, Hongrui Zheng',
    maintainer_email='zzang@seas.upenn.edu, billyzheng.bz@gmail.com',
    description='f1tenth wall_follow lab',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'wall_follow_node = wall_follow.wall_follow_node:main',
            'dotest = wall_follow.dotest:main',
        ],
    },
)
