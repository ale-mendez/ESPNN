import os
import pathlib

from setuptools import setup

# =============================================================================
# CONSTANTS
# =============================================================================

PATH = pathlib.Path(os.path.abspath(os.path.dirname(__file__)))


REQUIREMENTS = [
    "matplotlib==3.5.2",
    "numpy>=1.21.6",
    "pandas>=1.3.5",
    "pyvalem==2.5.7",
    "torch==1.11.0",
]

with open(PATH / "ESPNN" / "__init__.py") as fp:
    for line in fp.readlines():
        if line.startswith("__version__ = "):
            VERSION = line.split("=", 1)[-1].replace('"', "").strip()
            break

with open("README.md", "r") as readme:
    LONG_DESCRIPTION = readme.read()


# =============================================================================
# FUNCTIONS
# =============================================================================

setup(
    name="ESPNN",
    version="0.1.0",
    description="Electronic Stoping Power Neural Network predictor",
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    author="""
    Felipe Bivort Haiek,
    Alejandra Mendez,
    Claudia Montanari,
    Dario Mitnik
    """,
    author_email="felipebihaiek@gmail.com, alemdz.7@gmail.com",
    url="https://github.com/ale-mendez/SPNN",
    packages=["ESPNN", "ESPNN.data"],
    include_package_data=True,
    install_requires=REQUIREMENTS,
    license="The GPLv3 License",
    keywords=[
        "Stopping Power",
        "Energy Loss",
        "Atoms",
        "Molecules",
        "Ions",
        "Neural Neutwork",
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.8",
        "Topic :: Scientific/Engineering",
    ],
)
