# ESPNN - Electronic Stopping Power Neural Network

The ESPNN is a python-based deep neural network that allows the user to predict the electronic stopping power cross-section for any ion and target[^1] combination for a wide range of incident energies. The deep neural network was trained with many tens of thousands curated data points from the [IAEA database](https://www-nds.iaea.org/stopping/). See more details of the ESPNN in this [publication](arxiv).

[^1]: *ESPNN first release considers only mono-atomic targets.*

 [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) [![develstat](https://github.com/ale-mendez/ESPNN/actions/workflows/spnn_ci.yml/badge.svg)](https://github.com/ale-mendez/SPNN/actions/workflows/espnn_ci.yml/badge.svg) [![codecov](https://codecov.io/gh/ale-mendez/ESPNN/branch/master/graph/badge.svg?token=R49KN0O0I1)](https://codecov.io/gh/ale-mendez/ESPNN) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1UCDj0XT_4Ex_Mvp1vurleeeDVcjed6vP)
 <!-- [![Research software impact](http://depsy.org/api/package/pypi/)](http://depsy.org/package/python/) -->

 <!--
### Citation

```
@article{BivortHaiek2022,
author = {F. Bivort Haiek, A. M. P. Mendez, C. C. Montanari, D. M. Mitnik},
title = {ESPNN: The IAEA stopping power database neutral network. Part I: Monoatomic targets.},
year = {2022}

```
}-->

## Getting started

To use the ESPNN, we recommend using a python virtual environment. For example, [anaconda](https://docs.anaconda.com/anaconda/install/index.html) or [virtualenv](https://virtualenv.pypa.io/en/stable/installation.html). If you are not familiar with virtual environments and would like to rapidly start using python, follow the [anaconda](https://docs.anaconda.com/anaconda/install/index.html) indications according to your operating system:

- <a href="https://docs.anaconda.com/anaconda/install/linux/" target="_blank">Install anaconda in Linux</a>
- <a href="https://docs.anaconda.com/anaconda/install/windows/" target="_blank">Install anaconda in Windows</a>
- <a href="https://docs.anaconda.com/anaconda/install/mac-os/" target="_blank">Install anaconda in macOS</a>

Before installing the package you may want to give it a try on [Google Colab](https://colab.research.google.com/drive/1UCDj0XT_4Ex_Mvp1vurleeeDVcjed6vP).

### Install ESPNN
#### Using pip

The simplest way to install the ESPNN is via pip. Indistinctively, Ubuntu, Windows and macOS users can install the package by typing in the terminal or the anaconda bash terminal:
```console
$ pip install ESPNN
```

#### Using this repository

You can also install the ESPNN package by cloning or [downloading](https://github.com/ale-mendez/ESPNN/archive/refs/heads/master.zip) this repository. To clone (make sure you have git installed) this repo, use the following commands in your terminal/anaconda bash terminal:
```console
$ git clone https://github.com/ale-mendez/ESPNN.git
$ cd ESPNN
$ pip install ESPNN/
```
If you [downloaded](https://github.com/ale-mendez/ESPNN/archive/refs/heads/master.zip) the zip, change your directory to your download folder and, in your terminal/anaconda bash terminal, use
```console
$ pip install ESPNN-master.zip
```

### Run ESPNN in a notebook

A basic tutorial of the ESPNN package usage is given in [prediction.ipynb](workflow/prediction.ipynb). The package requires the following parameters as minimal input:

- ``projectile``: Chemical formula for the projectile
- ``projectile_mass``: Mass in amu for the projectile
- ``target``: Chemical formula for the target
- ``target_mass``: Mass in amu for the target

```python
import ESPNN
ESPNN.run_NN(projectile='He', projectile_mass=4.002602, target='Au', target_mass=196.966569)
```
    
![png](docs/prediction_files/prediction_2_0.png)

The package automatically produces a ``matplotlib`` figure and a sample file named ``XY_prediction.dat``, where ``X`` is the name of the projectile and ``Y`` is the name of the target system.

```console
$ ls -a
.  ..  HHe_prediction.dat  prediction.ipynb 
```

#### Optional arguments:

The energy grid used for the ESPNN calculation can be customized with arguments

- ``emin``: Minimum energy value in MeV/amu units (default: ``0.001``)
- ``emax``: Maximum energy value in MeV/amu units (default: ``10``)
- ``npoints``: Number of grid points (default: ``1000``)

Furthermore, the figure plotting and output-file directory-path can be modified via
- ``plot``: Prediction plot (default: ``True``)
- ``outdir``: Path to output folder (default: ``"./"``)


```python
ESPNN.run_NN(projectile='He', projectile_mass=4.002602, target='Au', target_mass=196.966569, emin=0.01, emax=1, npoints=50)
```


    
![png](docs/prediction_files/prediction_4_0.png)
    


### Run ESPNN from terminal

The ESPNN package can also be used from terminal with a syntaxis analogous to the above given:

```console
$ python -m ESPNN He 4.002602 Au 196.966569
```

Additional information about the optional arguments input can be obtained with the -h, --help flag:

```console
$ python -m ESPNN -h
```


##  Funding Acknowledgements

The following institutions financially support this work: the Consejo Nacional de Investigaciones Científicas y Técnicas (CONICET) by the PICT-2020-SERIEA-01931 and the Agencia Nacional de Promoción Científica y Tecnológica (ANPCyT) of Argentina PIP-11220200102421CO. CCM also acknowledges the financial support of the IAEA.