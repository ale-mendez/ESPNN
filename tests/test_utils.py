import pytest
import numpy as np

from SPNN.utils import generate_custom_table, get_Z_projectile, get_mass, get_max_Z, get_mass_atoms_ratio, get_ionisation_projectile, match_symbol_to_Z


@pytest.mark.parametrize(
    "projectile, projectile_mass, target, target_mass, ini_ener, end_ener, num_points, file_path",
    [
        ("H", 1, "H", 1, 1, 1, 1, f"tests/test_files/grid.csv"),
    ],
)
def test_generate_custom_table(
    projectile,
    projectile_mass,
    target,
    target_mass,
    ini_ener,
    end_ener,
    num_points,
    file_path
):
    """
    Testing generate_custom_table()
    """
    generate_custom_table(
        projectile,
        projectile_mass,
        target,
        target_mass,
        ini_ener,
        end_ener,
        num_points,
        file_path
    )
    lines = [
        'projectile,projectile_mass,target,target_mass,normalized_energy\n',
        'H,1.0,H,1.0,10.0\n'
    ]
    i = 0
    with open(file_path, "r") as file:
        line = file.readline()
        i += 1
        assert line == lines[i-1]


@pytest.mark.parametrize(
    "name",
    [
        ("H"),
    ],
)
def test_get_Z_projectile(name):
    """
    Testing get_Z_projectile() with valid name
    """
    assert get_Z_projectile(name) == 1


@pytest.mark.parametrize(
    "name",
    [
        ("X"),
    ],
)
def test_get_Z_projectile_nan(name):
    """
    Testing get_Z_projectile() with invalid name
    """
    assert np.isnan(get_Z_projectile(name))


@pytest.mark.parametrize(
    "name",
    [
        ("H"),
    ],
)
def test_get_mass(name):
    """
    Testing get_mass() with valid name
    """
    assert get_mass(name) == 1.008


@pytest.mark.parametrize(
    "name",
    [
        ("X"),
    ],
)
def test_get_mass_nan(name):
    """
    Testing get_mass() with invalid name
    """
    assert np.isnan(get_mass(name))


@pytest.mark.parametrize(
    "name",
    [
        ("H"),
    ],
)
def test_get_max_Z(name):
    """
    Testing get_max_Z() with valid name
    """
    assert get_max_Z(name) == 1.


@pytest.mark.parametrize(
    "name",
    [
        ("X"),
    ],
)
def test_get_max_Z_nan(name):
    """
    Testing get_max_Z() with invalid name
    """
    assert np.isnan(get_max_Z(name))


@pytest.mark.parametrize(
    "name",
    [
        ("H"),
    ],
)
def test_get_mass_atoms_ratio(name):
    """
    Testing get_mass_atoms_ratio() with valid name
    """
    assert get_mass_atoms_ratio(name) == 1.008


@pytest.mark.parametrize(
    "name",
    [
        ("X"),
    ],
)
def test_get_mass_atoms_ratio_nan(name):
    """
    Testing get_max_Z() with invalid name
    """
    assert np.isnan(get_mass_atoms_ratio(name))


@pytest.mark.parametrize(
    "name",
    [
        ("H"),
    ],
)
def test_get_ionisation_projectile(name):
    """
    Testing get_ionisation_projectile() with valid name
    """
    assert get_ionisation_projectile(name) == 1312.0


@pytest.mark.parametrize(
    "name",
    [
        ("X"),
    ],
)
def test_get_ionisation_projectile_nan(name):
    """
    Testing get_ionisation_projectile() with invalid name
    """
    assert np.isnan(get_ionisation_projectile(name))


@pytest.mark.parametrize(
    "name",
    [
        (10),
    ],
)
def test_match_symbol_to_Z(name):
    """
    Testing match_symbol_to_Z() with valid name
    """
    assert match_symbol_to_Z(name) == "Ne"


@pytest.mark.parametrize(
    "name",
    [
        ("X"),
    ],
)
def test_match_symbol_to_Z_nan(name):
    """
    Testing match_symbol_to_Z() with invalid name
    """
    assert match_symbol_to_Z(name) == "Unknown"


