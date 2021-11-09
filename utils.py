import numpy as np
import pandas as pd
import pyvalem
from pyvalem.formula import Formula

from .formula_mapper import formula_mapper


def generate_custom_table(
    target,
    projectile_name,
    projectile_mass,
    t_atomic_mass,
    init_ener,
    end_ener,
    num_points,
    file_path,
):

    """

    Conveniency function to create an input table for the model.

    """

    ener_range = np.logspace(
        init_ener,
        end_ener,
        num=num_points,
        endpoint=True,
        base=10.0,
        dtype=None,
        axis=0,
    )

    df = pd.DataFrame(
        {
            "target": num_points * [target],
            "projectile_name": num_points * [projectile_name],
            "projectile_mass": num_points * [projectile_mass],
            "t_atomic_mass": num_points * [t_atomic_mass],
            "normalized_energy": ener_range,
        }
    )

    df.to_csv(file_path)


chem_prop = pd.read_csv("/content/input/chemicalProperties.csv")

chem_prop[" Symbol"] = chem_prop[" Symbol"].str.lstrip()

chem_prop_tab = chem_prop[[" Symbol", " Atomic_Number"]]

chem_prop_tab.set_index(" Symbol", inplace=True)

atomic_number_dict = chem_prop_tab.to_dict()[" Atomic_Number"]


def get_Z_projectile(name):

    try:

        return atomic_number_dict[str(name)]

    except:

        return np.nan


def get_mass(name):

    if name in formula_mapper.keys():

        name = formula_mapper[name]
    if name.lower() == "d2o":
        return 20

    if name.lower() == "d2o":
        return 4

    try:

        f = Formula(name)

        return f.mass

    except:

        return np.nan


def get_max_Z(name):

    if name in formula_mapper.keys():

        name = formula_mapper[name]

    try:

        f = Formula(name)

        return max([atomic_number_dict[str(atom)] for atom in f.atoms])

    except:

        return np.nan


def get_mass_atoms_ratio(name):

    if name in formula_mapper.keys():

        name = formula_mapper[name]
    if name.lower() == "d2o":
        return 20 / 3

    if name.lower() == "d2o":
        return 2

    try:

        f = Formula(name)

        return f.mass / f.natoms

    except:

        return np.nan


ion_prop = pd.read_table("/content/input/ionization_energies_wiki.txt")
ion_prop["Symbol"] = ion_prop["Symbol"].str.lstrip()

ion_prop_tab = ion_prop[["Symbol", "1st"]]

ion_prop_tab.set_index("Symbol", inplace=True)

ionisation_dict = ion_prop_tab.to_dict()["1st"]


def get_ionisation_projectile(name):

    name = str(name)

    target_dict = {
        "C amorphous": "C",
        "Graphite": "C",
        "O2": "O",
        "N2": "N",
        "H2": "H",
        "Havar": "Co",
    }

    if name in ["C amorphous", "O2", "N2", "H2", "Graphite", "Havar"]:

        name = target_dict[name]

    try:

        return ionisation_dict[str(name)]

    except:

        return np.nan


import random
import torch


def seed_everything(seed=42):
    """Sets all the necessary seeds"""

    random.seed(seed)
    # os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
