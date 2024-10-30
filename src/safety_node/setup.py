from setuptools import setup, find_packages

package_name = 'safety_node'

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
    maintainer='Soyeon',
    maintainer_email='soyeonkwon.0@gmail.com',
    description='AEB',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'safety_node = safety_node.safety_node:main',
        ],
    },
)
