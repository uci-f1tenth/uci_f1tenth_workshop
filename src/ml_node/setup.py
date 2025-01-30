from setuptools import setup, find_packages

package_name = 'ml_node'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/models', ['models/mega_dagger.pkl']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Soyeon',
    maintainer_email='soyeonkwon.0@gmail.com',
    description='AEB',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'ml_node = ml_node.ml_agent:main',
        ],
    },
)
