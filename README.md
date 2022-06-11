# ESPNN - Electronic Stopping Power Neural Network

 [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) [![develstat](https://github.com/ale-mendez/ESPNN/actions/workflows/espnn_ci.yml/badge.svg)](https://github.com/ale-mendez/ESPNN/actions/workflows/espnn_ci.yml/badge.svg) [![codecov](https://codecov.io/gh/ale-mendez/ESPNN/branch/master/graph/badge.svg?token=R49KN0O0I1)](https://codecov.io/gh/ale-mendez/ESPNN) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1UCDj0XT_4Ex_Mvp1vurleeeDVcjed6vP)
 <!-- [![Research software impact](http://depsy.org/api/package/pypi/)](http://depsy.org/package/python/) -->

The ESPNN is a python-based deep neural network that allows the user to predict the electronic stopping power cross-section for any ion and target[^1] combination for a wide range of incident energies. The deep neural network was trained with many tens of thousands curated data points from the [IAEA database](https://www-nds.iaea.org/stopping/). See more details of the ESPNN in this [publication](https://github.com/ale-mendez/ESPNN-doc).


 <!--
### Citation

```
@article{BivortHaiek2022,
author = {F. Bivort Haiek, A. M. P. Mendez, C. C. Montanari, D. M. Mitnik},
title = {ESPNN: The IAEA stopping power database neutral network. Part I: Monoatomic targets.},
year = {2022}

```
}-->

You can use the ESPNN package [remotely](#run-ESPNN-online) or [locally](#install-espnn). Find below all the usage options available.

## Run ESPNN online

The ESPNN package can be used remotely in the <a href="https://colab.research.google.com/drive/1UCDj0XT_4Ex_Mvp1vurleeeDVcjed6vP" target="_blank">Google Colab</a> platform[^2]. There, you'll find a jupyter notebook with a quick tutorial on how to use the ESPNN. You can also make a copy of the notebook to your own personal Drive and compute the stopping power of any projectile-target combination.

## Install ESPNN

To use the ESPNN in your computer, first you'll need to install it. We recommend using a python virtual environment to this end (for example, see <a href="https://docs.anaconda.com/anaconda/install/index.html" target="_blank">anaconda</a> or <a href="https://virtualenv.pypa.io/en/stable/installation.html" target="_blank">virtualenv</a>). If you are not familiar with virtual environments and would like to rapidly start using python, follow the <a href="https://docs.anaconda.com/anaconda/install/index.html" target="_blank">anaconda</a> indications according to your operating system:

- <a href="https://docs.anaconda.com/anaconda/install/linux/" target="_blank">Install anaconda in Linux</a>
- <a href="https://docs.anaconda.com/anaconda/install/windows/" target="_blank">Install anaconda in Windows</a>
- <a href="https://docs.anaconda.com/anaconda/install/mac-os/" target="_blank">Install anaconda in macOS</a>

### Using pip

The simplest way to install the ESPNN is via pip. Indistinctively, Ubuntu, Windows and macOS users can install the package by typing in the terminal or the anaconda bash terminal:
```console
$ pip install ESPNN
```
### Using this repository

You can also install the ESPNN package by cloning or [downloading](https://github.com/ale-mendez/ESPNN/archive/refs/heads/master.zip) this repository. To clone (make sure you have git installed) this repo, use the following commands in your terminal/anaconda bash terminal:
```console
$ git clone https://github.com/ale-mendez/ESPNN.git
$ cd ESPNN
$ pip install ESPNN/
```
If you [downloaded](https://github.com/ale-mendez/ESPNN/archive/refs/heads/master.zip) the zip, change your directory to your download folder and, in your terminal/anaconda bash terminal, type
```console
$ pip install ESPNN-master.zip
```

## Run ESPNN locally

Once you've [installed](#install-espnn) the ESPNN package in your preferred environment, you can run it by using a jupyter notebook or directly from terminal.
### Using a notebook

A basic tutorial of the ESPNN package usage is given in <a href="https://github.com/ale-mendez/ESPNN/blob/master/workflow/prediction.ipynb" target="_blank">prediction.ipynb</a>. The package requires the following parameters as minimal input:

- ``projectile``: Chemical formula for the projectile
- ``projectile_mass``: Mass in amu for the projectile
- ``target``: Chemical formula for the target
- ``target_mass``: Mass in amu for the target

```python
import ESPNN
ESPNN.run_NN(projectile='He', projectile_mass=4.002602, target='Au', target_mass=196.966569)
```

![](https://github.com/ale-mendez/ESPNN/blob/master/docs/prediction_files/prediction_2_0.png?raw=true)

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

![](https://github.com/ale-mendez/ESPNN/blob/master/docs/prediction_files/prediction_4_0.png?raw=true)

### From terminal

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


[^1]: *ESPNN first release considers only mono-atomic targets.*
[^2]: *A Google account is required.*
