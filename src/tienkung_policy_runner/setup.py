from setuptools import setup
from glob import glob
import os

package_name = 'tienkung_policy_runner'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', [f'resource/{package_name}']),
        (f'share/{package_name}', ['package.xml']),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'test'), glob('test/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='DedSecer',
    maintainer_email='dedsecer@example.com',
    description='Tienkung ONNX policy runner for ros_lite-compatible robot IO.',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'policy_runner_node = tienkung_policy_runner.policy_runner_node:main',
        ],
    },
)
