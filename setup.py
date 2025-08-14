from setuptools import setup, find_packages

setup(
    name="query_interface",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "flask>=2.0.0",
        "pytest>=6.0.0",
    ],
    python_requires=">=3.8",
)