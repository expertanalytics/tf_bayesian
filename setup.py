from setuptools import setup, find_packages

DEPENDENCIES = [
        "tensorflow",
        "tensorflow-probability",
        "matplotlib",
        "numpy",
        ]

setup(
        name="tf_bayesian",
        version="0.1.0",
        license="MIT",
        install_requires=DEPENDENCIES,
        packages=find_packages(exclude=["tests", "*.tests", "*.tests.*", "tests.*", "scripts",]),
        python_requires="<=3.7, >=3.4",
    )
