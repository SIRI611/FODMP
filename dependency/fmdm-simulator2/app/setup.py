from setuptools import setup

setup(name='robotic_manipulation',
      version='0.2',
      packages=['gym_env'],  #same as name
      install_requires=['gym', 'pybullet>=2.7.3', 'pyyaml']
      # Add any other dependencies required
)
