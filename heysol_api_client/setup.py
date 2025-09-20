"""
Setup script for the HeySol API client library.

This script configures the package for distribution via PyPI.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

# Read the version from the package
about = {}
exec((this_directory / "heysol_api_client" / "heysol" / "__init__.py").read_text(encoding="utf-8"), about)

setup(
    name="heysol-api-client",
    version=about["__version__"],
    author="HeySol API Client Development Team",
    author_email="dev@heysol.ai",
    description="A comprehensive Python client for the HeySol API with MCP protocol support",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/heysol/heysol-python-client",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Healthcare Industry",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Medical Science Apps",
        "Topic :: Internet :: WWW/HTTP",
    ],
    python_requires=">=3.8",
    install_requires=[
        "requests>=2.25.0",
        "aiohttp>=3.8.0",
        "python-dotenv>=0.19.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "isort>=5.10.0",
            "mypy>=1.0.0",
            "flake8>=4.0.0",
            "pre-commit>=2.17.0",
        ],
        "docs": [
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "heysol-client=heysol_api_client.cli:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)