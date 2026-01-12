from setuptools import setup, find_packages

setup(
    name="adasr",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "torch",
        "torchvision",
        "numpy",
        "tqdm",
        "Pillow",
        "lpips",
        "piq"
    ],
)
