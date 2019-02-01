# TODO: https://blog.ionelmc.ro/2014/05/25/python-packaging/#the-structure
import io
import re
from glob import glob
from os.path import basename
from os.path import dirname
from os.path import join
from os.path import splitext

from setuptools import find_packages
from setuptools import setup

from setuptools import setup, find_packages

setup(name='centipyde',
    python_requires='>=3.6',
    version='0.1',
    description='Python interpreter for C code',
    url='http://github.com/rbowden91/centipyde',
    author='Rob Bowden',
    author_email='rbowden91@gmail.com',
    license='GPLv3',
    packages=find_packages('src'),
    package_dir={'':'src'},
    py_modules=[splitext(basename(path))[0] for path in glob('src/*.py')],
    install_requires=[
      'pycparser',
    ],
    entry_points={
        #'console_scripts': [
        #    'centipyde = centipyde.interpret:main',
        #]
    },
    zip_safe=False)
