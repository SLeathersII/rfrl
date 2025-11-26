from setuptools import setup, find_packages

setup(name='rfrl-gym',
      version='0.1',
      author='Virginia Tech National Security Institute and Morehouse College',
      install_requires=['gymnasium',
                        'numpy',
                        'matplotlib',
                        'distinctipy',
                        'pyqtgraph',
                        'pyqt6==6.7.1',
                        'scipy'],
      extras_require={'rl_packages': ['stable_baselines3==2.5.0a1']},
      packages= find_packages())

