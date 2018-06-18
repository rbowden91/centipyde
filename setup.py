# TODO: https://blog.ionelmc.ro/2014/05/25/python-packaging/#the-structure
from setuptools import setup

setup(name='centipyde',
      python_requires='>=3.6',
      version='0.1',
      description='The funniest joke in the world',
      url='http://github.com/rbowden91/centipyde',
      author='Rob Bowden',
      author_email='rbowden91@gmail.com',
      license='GPLv3',
      packages=['centipyde'],
      install_requires=[
        'pycparser'
      ],
      zip_safe=False)
