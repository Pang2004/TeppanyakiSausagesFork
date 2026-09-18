from setuptools import setup, find_packages

setup(
    name="acv_model",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pandas>=2.0",
        "numpy>=1.24",
        "openpyxl>=3.1",
    ],
    python_requires=">=3.9",
)
