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
    projectile: str,
    target: str,
    projectile_mass: int = None,
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
    projectile : str
        Projectile symbol
    target : str
        Target symbol
    projectile_mass : int, optional
        Projectile mass (amu)
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

    df["projectile_mass"] = df["projectile"].apply(get_mass)

    df["target_ionisation"] = df["target"].apply(get_ionisation_projectile)

    df["projectile_Z"] = df["projectile"].apply(get_Z_projectile)

    if target_mass is None:
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
    df_tup = df_tup.dropna(axis=0, subset=['stopping_power'])
    if len(df_tup) == 0:
        raise ValueError(f"No values found for {projectile } + {target} collisional system")

    # Save dataframe with prediction to file
    filepath = os.path.join(outdir, f"{projectile + target}_prediction.dat")
    df_out = pd.DataFrame(
        {
            out_cols["E"]: df_tup["normalized_energy"],
            out_cols["SP"]: df_tup["stopping_power"]
        }
    )

    # 1. Remove negative stopping power interpolations
    try:
        df_out = df_out[df_out[out_cols["SP"]] >= 0]
    except:
        pass

    if (target in ('Ta', 'Gd')) and (projectile in ('H', 'He')):

        # 2. Remove unphysical interpolation in the low energy region (up to 0.2 MeV/amu)
        try:
            df_lowE = df_out.loc[df_out[out_cols['E']] <= 0.2]
            x = df_lowE[out_cols['E']]
            y = df_lowE[out_cols['SP']]
            dy = np.gradient(y)
            # find zeros (of derivative)
            zeros = [x[i + 1] for i, (xi, dyi) in enumerate(zip(x[:-1], dy[:-1])) if (dyi * dy[i + 1] < 0)]
            # find value of energy at minimum SP value
            x_ymin = [xi for xi, yi in zip(x, y) if yi == min(y)][0]
            emin2 = [xi for xi in zeros if (x_ymin * 0.5 <= xi <= x_ymin + x_ymin * 1.5)][0]
            df_out = df_out[df_out[out_cols["E"]] >= emin2]
        except:
            pass

    if len(df_out) != len(df_tup):
        new_emin = df_out.iloc[0][0]
        print(f"emin: {emin} => {new_emin:.3f}")

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
