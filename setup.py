from setuptools import setup

setup(
    name="fodmp",
    version="0.0.1",
    author="Xirui Shi",
    packages=["fodmp"],
    install_requires=[
        "torch",
        "torchvision",  # For image transformations
        "pytest",
        "hydra-core",
        "diffusers[torch]",  # For EMA and discrete time noise schedulers
        "wandb",
        "robomimic",  # For ResNet with SpatialSoftmax
        "moviepy",  # Saving videos with wandb
        "imageio",  # Saving videos with wandb
        "pygame",  # For PushT environment
        "pymunk",  # For PushT environment
        "shapely",  # For PushT environment
        "scikit-image",  # For PushT environment
        "tabulate",  # For pretty printing in memory_stats.py
        "moviepy",  # For saving videos
        "GitPython",  # For getting the repositories' root directory
        "pybullet",  # For obstacle avoidance environment
        "gymnasium",  # For common RL environment interface
        "numpy<2.0.0",
        "opencv-python",
        "plotly", # For ObstacleAvoidance
        "pandas", # For ObstacleAvoidance
        "addict",
    ],
    python_requires=">=3.10",
)