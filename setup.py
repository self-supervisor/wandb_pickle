from setuptools import find_packages, setup

setup(
    name="wandb_csv",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "wandb",
    ],
)
