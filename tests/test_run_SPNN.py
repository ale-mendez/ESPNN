import os
import sys

module_path = os.path.abspath(os.path.join('../SPNN/'))
if module_path not in sys.path:
    sys.path.append(module_path)

from SPNN.core import run_SPNN
import pytest


@pytest.mark.parametrize(
    "projectile_name, projectile_mass, target, target_mass, ini_ener, end_ener, num_points, fdir, plot, filename",
    [
        ("H", 1, "H", 1, 1, 1, 1, f"tests/test_files/", False, "HH_prediction.csv"),
    ],
)
def test_run_SPNN(
    projectile_name,
    projectile_mass,
    target,
    target_mass,
    ini_ener,
    end_ener,
    num_points,
    fdir,
    plot,
    filename
):
    """
    Testing run_SPNN()
    """
    run_SPNN(
        projectile_name,
        projectile_mass,
        target,
        target_mass,
        ini_ener,
        end_ener,
        num_points,
        fdir,
        plot
    )
    lines = [
        'projectile,projectile_mass,target,target_mass,normalized_energy,projectile_Z,target_ionisation,projectile_ionisation,Z_max,prediction,system\n',
        'H,1.0,H,1.008,10.0,1.0,1312.0,1312.0,1.0,0.09430693089962006,H_H\n'
    ]
    i = 0
    filepath = os.path.join(fdir, filename)
    with open(filepath, "r") as file:
        line = file.readline()
        i += 1
        assert line == lines[i-1]

