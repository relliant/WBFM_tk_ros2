from setuptools import setup
from glob import glob
import os

package_name = 'tienkung_bringup'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', [f'resource/{package_name}']),
        (f'share/{package_name}', ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='DedSecer',
    maintainer_email='dedsecer@example.com',
    description='Launch files for Tienkung sim2real bringup.',
    license='Apache-2.0',
)
