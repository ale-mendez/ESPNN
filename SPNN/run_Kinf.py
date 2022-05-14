import numpy as np

from .inference import run_inference


def run_k_fold(folds, NFOLDS, seed, device, verbose=False, **kwargs):
    """Wraps run_inference to peform ensemble predictions"""

    valid_df = folds

    oof = np.zeros((len(valid_df), NFOLDS))

    for fold in range(NFOLDS):

        X_valid = valid_df[
            [
                "target_mass",
                "projectile_mass",
                "projectile_Z",
                "Z_max",
                "normalized_energy",
                "target_ionisation",
            ]
        ].values

        oof_ = run_inference(X_valid, fold, seed, device, verbose, **kwargs)

        oof[:, fold] = oof_[:, 0]

    return oof
