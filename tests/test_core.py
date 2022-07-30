import os
import pytest

from ESPNN.core import run_NN


@pytest.mark.parametrize(
    "projectile_name, target, ini_ener, end_ener, num_points, fdir, plot, filename",
    [
        ("H", "H", 1, 1, 1, f"tests/test_files/", False, "HH_prediction.dat"),
    ],
)
def test_run_NN(
    projectile_name,
    target,
    ini_ener,
    end_ener,
    num_points,
    fdir,
    plot,
    filename
):
    """
    Testing run_NN()
    """
    run_NN(
        projectile=projectile_name,
        target=target,
        emin=ini_ener,
        emax=end_ener,
        npoints=num_points,
        outdir=fdir,
        plot=plot
    )
    lines = [
        'Energy (MeV/amu)	Stopping power (MeV cm2/mg)\n',
        '1.0	0.7288954973220825\n'
    ]
    i = 0
    filepath = os.path.join(fdir, filename)
    with open(filepath, "r") as file:
        line = file.readline()
        i += 1
        assert line == lines[i-1]

