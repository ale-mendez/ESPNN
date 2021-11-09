import numpy as np
import pandas as pd


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

    ener_range = numpy.logspace(
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
