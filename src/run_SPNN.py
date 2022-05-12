import os
import sys
module_path = os.path.abspath(os.path.join('..'))
dir_path = os.path.dirname(os.path.realpath(__file__))

if module_path not in sys.path:
    sys.path.append(module_path)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from .utils import generate_custom_table, get_ionisation_projectile, get_mass, get_max_Z, get_Z_projectile 
from .run_Kinf import run_k_fold


# Default HyperParameters:
NFOLDS = 5
SEED = np.random.randint(0, 12347, NFOLDS)
DEVICE = 'cpu'
BATCH_SIZE = 64
exp_name = 'try0__00_en_ioen_0_bhe_corrected_trtestsplit_tuple'
INPUT_FILE = f'{dir_path}/data/input/grid.csv'

def run_SPNN(projectile=None, projectile_mass=None, target=None, target_mass=None, minlogE=-3, maxlogE=1, npoints=1000, fdir="./", plot=True):

    generate_custom_table(projectile, projectile_mass, target, target_mass, minlogE, maxlogE, npoints, INPUT_FILE)

    # Loading and adding features from tables
    df = pd.read_csv(INPUT_FILE)
    df['projectile_Z'] = df['projectile'].apply(get_Z_projectile)
    df['target_ionisation'] = df['target'].apply(get_ionisation_projectile)
    df['projectile_ionisation'] = df['projectile'].apply(get_ionisation_projectile)
    df['target_mass'] = df['target'].apply(get_mass)
    df['Z_max'] = df['target'].apply(get_max_Z)
    columns =  ['target_mass', 'projectile_mass', 'Z_max', 'projectile_Z', 'normalized_energy', 'target_ionisation']
    df[columns] = df[columns].astype(float)
    
    # Transform to logarithmic incident energy
    df_log= df.copy()
    df_log['normalized_energy'] = np.log(df['normalized_energy'].values)
    params = {'exp_name': exp_name, 'model_dir': f'{dir_path}/data/weights'}
    
    # Averaging on multiple SEEDS
    target_cols = ['normalized_stopping_power']
    # oof = np.zeros((len(df_log), 1))
    for seed in SEED:
        oof_ = run_k_fold(df_log, target_cols, NFOLDS, seed, device=DEVICE, verbose=True, **params)
    
    for fold in range(NFOLDS):
        df_log[f'pred_{fold}'] = oof_[:, fold]
    
    df['prediction'] = np.mean(oof_, axis=1) 
    df['system'] = df['projectile'] + '_' + df['target']
    for tup in df['system'].unique():
        df_tup = df.loc[df['system'] == tup]

    # save dataframe with prediction to file
    filepath = os.path.join(fdir, f'{projectile+target}_prediction.csv')
    clean_cols = ['projectile', 'target', 'normalized_energy', 'prediction']
    df_clean = df_tup[clean_cols]
    df_tup.to_csv(filepath, index=False)
    
    # plot prediction
    if plot:
        plt.scatter(df_tup['normalized_energy'], df_tup['prediction'])
        plt.title(tup)
        plt.xscale('log')
        plt.show()

