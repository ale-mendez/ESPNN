import os
import pathlib

from setuptools import setup

# =============================================================================
# CONSTANTS
# =============================================================================

PATH = pathlib.Path(os.path.abspath(os.path.dirname(__file__)))


REQUIREMENTS = [
    "asttokens==2.0.5",
    "attrs==21.4.0",
    "backcall==0.2.0",
    "coverage==6.3.2",
    "cycler==0.11.0",
    "debugpy==1.6.0",
    "decorator==5.1.1",
    "entrypoints==0.4",
    "executing==0.8.3",
    "fonttools==4.33.3",
    "iniconfig==1.1.1",
    "ipykernel==6.13.0",
    "ipython==8.3.0",
    "jedi==0.18.1",
    "jupyter-client==7.3.1",
    "jupyter-core==4.10.0",
    "kiwisolver==1.4.2",
    "matplotlib==3.5.2",
    "matplotlib-inline==0.1.3",
    "nest-asyncio==1.5.5",
    "numpy==1.22.3",
    "packaging==21.3",
    "pandas==1.4.2",
    "parso==0.8.3",
    "pexpect==4.8.0",
    "pickleshare==0.7.5",
    "Pillow==9.1.0",
    "pluggy==1.0.0",
    "prompt-toolkit==3.0.29",
    "psutil==5.9.0",
    "ptyprocess==0.7.0",
    "pure-eval==0.2.2",
    "py==1.11.0",
    "Pygments==2.12.0",
    "pyparsing==3.0.9",
    "pytest==7.1.2",
    "pytest-cov==3.0.0",
    "python-dateutil==2.8.2",
    "pytz==2022.1",
    "pyvalem==2.5.7",
    "pyzmq==22.3.0",
    "six==1.16.0",
    "stack-data==0.2.0",
    "tomli==2.0.1",
    "torch==1.11.0",
    "tornado==6.1",
    "traitlets==5.2.0",
    "typing_extensions==4.2.0",
    "wcwidth==0.2.5"
]

print(PATH)
with open(PATH / "SPNN" / "__init__.py") as fp:
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
    name="SPNN",
    version="0.1.1",
    author="""
    Felipe Bivort Haiek,
    Alejandra Mendez,
    Claudia Montanari,
    Darío Mitnik
    """,
    author_email="""
    felipebihaiek@gmail.com,
    alemdz.7@gmail.com
    """,
    packages=["SPNN"],
    install_requires=REQUIREMENTS,
    license="The GPLv3 License",
    description="Stoping Power Neural Network predictor",
    long_description_content_type="text/markdown",
    long_description=LONG_DESCRIPTION,
    url="https://github.com/ale-mendez/SPNN",
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
    include_package_data=True,
)
