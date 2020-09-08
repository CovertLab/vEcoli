import os
import glob
import setuptools
from distutils.core import setup

with open("README.md", 'r') as readme:
    long_description = readme.read()

# to include data in the package, use MANIFEST.in

setup(
    name='vivarium-ecoli',
    version='0.0.1',
    packages=[
        'ecoli',
        'ecoli.compartments',
        'ecoli.experiments',
        'ecoli.processes'],
    author='Ryan Spangler',
    author_email='ryan.spangler@gmail.com',
    url='https://github.com/CovertLab/vivarium-ecoli',
    license='MIT',
    entry_points={
        'console_scripts': []},
    long_description=long_description,
    long_description_content_type='text/markdown',
    install_requires=[
        'vivarium-cell>=0.0.12'])
