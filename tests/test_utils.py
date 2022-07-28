import pytest
import numpy as np
import pandas as pd

from ESPNN.utils import generate_custom_table, get_Z_projectile, get_mass, get_max_Z, get_mass_atoms_ratio, get_ionisation_projectile, match_symbol_to_Z


@pytest.mark.parametrize(
    "projectile, target, target_mass, emin, emax, npoints",
    [
        ("H", "H", 1, 1, 1, 1),
    ],
)
def test_generate_custom_table(
    projectile,
    target,
    target_mass,
    emin,
    emax,
    npoints,
):
    """
    Testing generate_custom_table()
    """
    df = generate_custom_table(
        projectile,
        target,
        target_mass,
        emin,
        emax,
        npoints,
    )
    df_ = pd.DataFrame({
        'projectile': ['H'],
        'target': ['H'],
        'target_mass': [1],
        'normalized_energy': [1.0]
    })
    for col in df.columns:
        assert df_.loc[0][col] == df.loc[0][col]


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
        ("Cyclopropane"),
    ],
)
def test_get_mass_compound(name):
    """
    Testing get_mass() with compound
    """
    assert get_mass(name) == 42.081


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


