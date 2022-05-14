import os
import pytest

from SPNN.core import run_SPNN


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
        'projectile,target,normalized_energy,stopping power\n',
        'H,H,10.0,0.09430693089962006\n'
    ]
    i = 0
    filepath = os.path.join(fdir, filename)
    with open(filepath, "r") as file:
        line = file.readline()
        i += 1
        assert line == lines[i-1]

