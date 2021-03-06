from glob import glob
from os.path import basename, splitext
from setuptools import find_packages, setup

setup(
    name='interpolation',
    version='0.1',
    packages=find_packages(include=['interpolation']),
    py_modules=[splitext(basename(path))[0] for path in glob('src/*.py')],
    entry_points={
        'console_scripts': ['interpolation=interpolation.__main__:interpolation_run'],
    }
)
