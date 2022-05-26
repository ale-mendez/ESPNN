import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from .utils import generate_custom_table
from .utils import get_ionisation_projectile
from .utils import get_mass
from .utils import get_max_Z
from .utils import get_Z_projectile
from .run_Kinf import run_k_fold

# Default HyperParameters:
NFOLDS = 5
SEED = np.random.randint(0, 12347, NFOLDS)
DEVICE = "cpu"
BATCH_SIZE = 64
exp_name = "try0__00_en_ioen_0_bhe_corrected_trtestsplit_tuple"
dir_path = os.path.dirname(os.path.realpath(__file__))
out_cols = {
    "E": "Energy (MeV/amu)",
    "SP": "Stopping power (MeV cm2/mg)"
}


def run_NN(
    projectile: str = None,
    projectile_mass: int = None,
    target: str = None,
    target_mass: int = None,
    emin: int = 0.001,
    emax: int = 10,
    npoints: int = 1000,
    outdir: str = "./",
    plot: bool = True,
):
    """Compute NN prediction for electronic stopping power

    Parameters
    ----------
    projectile : str, optional
        Projectile symbol, by default None
    projectile_mass : int, optional
        Projectile mass (amu), by default None
    target : str, optional
        Target symbol, by default None
    target_mass : int, optional
        Target mass (amu), by default None
    emin : int, optional
        Minimum grid-energy value (MeV/amu), by default 0.001
    emax : int, optional
        Maximum grid-energy value (MeV/amu), by default 10
    npoints : int, optional
        Number of grid-points, by default 1000
    outdir : str, optional
        _description_, by default "./"
    plot : bool, optional
        _description_, by default True
    """

    # Generate grid
    df = generate_custom_table(
        projectile,
        projectile_mass,
        target,
        target_mass,
        emin,
        emax,
        npoints,
    )
    df["projectile_Z"] = df["projectile"].apply(get_Z_projectile)
    df["target_ionisation"] = df["target"].apply(get_ionisation_projectile)
    df["projectile_ionisation"] = df["projectile"].apply(
        get_ionisation_projectile
    )
    df["target_mass"] = df["target"].apply(get_mass)
    df["Z_max"] = df["target"].apply(get_max_Z)
    columns = [
        "target_mass",
        "projectile_mass",
        "Z_max",
        "projectile_Z",
        "normalized_energy",
        "target_ionisation",
    ]
    df[columns] = df[columns].astype(float)

    # Transform to logarithmic incident energy
    df_log = df.copy()
    df_log["normalized_energy"] = np.log(df["normalized_energy"].values)
    params = {"exp_name": exp_name, "model_dir": f"{dir_path}/data/weights"}

    # Averaging on multiple SEEDS
    for seed in SEED:
        oof_ = run_k_fold(
            df_log,
            NFOLDS,
            seed,
            device=DEVICE,
            verbose=True,
            **params
        )

    for fold in range(NFOLDS):
        df_log[f"pred_{fold}"] = oof_[:, fold]

    df["stopping_power"] = np.mean(oof_, axis=1)
    df["system"] = df["projectile"] + "_" + df["target"]
    for tup in df["system"].unique():
        df_tup = df.loc[df["system"] == tup]

    # Save dataframe with prediction to file
    filepath = os.path.join(outdir, f"{projectile + target}_prediction.dat")
    df_out = pd.DataFrame(
        {
            out_cols["E"]: df_tup["normalized_energy"],
            out_cols["SP"]: df_tup["stopping_power"]
        }
    )

    # Remove negative stopping power interpolations
    df_out = df_out[df_out[out_cols["SP"]] >= 0]
    if len(df_out) != len(df_tup):
        print(f"emin: {emin} => {df_out.iloc[0][0]}")

    df_out.to_csv(filepath, index=False, sep='\t')

    # Plot prediction
    if plot is True:
        plot_prediction(projectile, target, df_out)


def plot_prediction(projectile, target, df):
    e = out_cols['E']
    sp = out_cols['SP']
    title = ' '.join([projectile, "on", target])
    fig, ax = plt.subplots(1, 1, figsize=(8 * 1.1, 6 * 1.1))
    ax.scatter(df[e], df[sp])
    ax.set_title(title, fontsize=20)
    ax.set_xscale("log")
    ax.set_xlabel(r"Energy (MeV/amu)", fontsize=18)
    ax.set_ylabel(r"Electronic Stopping Power (MeV cm$^2$/mg)", fontsize=18)
    ax.tick_params(axis='both', labelsize=14)
    plt.show()
