"""Tests for physical constants, MolecularSystem validation, builders, CG mapping, and PDB I/O."""

import numpy as np
import pytest

from solvato_dyne.systems.builders import (
    build_alanine_dipeptide_system,
    build_alpha_helix_peptide,
    build_coarse_grained_elastic_protein,
    build_host_guest_unbinding_system,
    build_polarizable_water_dimer_or_cluster,
    build_system_by_name,
    export_pdb_block,
    map_all_atom_to_cg,
    parse_pdb_block,
)
from solvato_dyne.types import AtomSpec, BoxSpec, MolecularSystem


def test_box_spec_orthorhombic_and_triclinic() -> None:
    box = BoxSpec.orthorhombic(20.0, 25.0, 30.0)
    assert box.is_orthorhombic
    assert np.isclose(box.volume, 15000.0)

    tri = BoxSpec(np.array([[20.0, 0.0, 0.0], [5.0, 20.0, 0.0], [0.0, 0.0, 20.0]]))
    assert not tri.is_orthorhombic
    assert np.isclose(tri.volume, 8000.0)

    with pytest.raises(ValueError):
        BoxSpec.orthorhombic(-1.0, 10.0, 10.0)


def test_drude_atom_charge_and_mass_conservation() -> None:
    atom = AtomSpec(
        index=0,
        name="OH2",
        residue_name="WAT",
        residue_id=1,
        element="O",
        mass=15.999,
        charge=-1.11466,
        alpha_pol=0.97825,
        drude_k=1000.0,
        drude_mass=0.4,
    )
    assert atom.is_polarizable
    assert atom.drude_charge < 0.0
    assert np.isclose(atom.core_charge + atom.drude_charge, atom.charge)
    assert np.isclose(atom.core_mass + atom.drude_mass, atom.mass)


def test_water_cluster_charge_neutrality_and_drude_count() -> None:
    cluster = build_polarizable_water_dimer_or_cluster(n_molecules=6, box_size=24.0, polarizable=True)
    assert cluster.n_atoms == 18
    assert cluster.n_drude == 6
    assert abs(cluster.total_charge) < 1e-10
    assert cluster.box is not None
    assert np.isclose(cluster.box.volume, 24.0**3)


def test_alanine_dipeptide_exclusions_and_14_pairs() -> None:
    ala = build_alanine_dipeptide_system(polarizable=True)
    assert ala.n_atoms == 12
    assert abs(ala.total_charge) < 1e-10
    excl, p14 = ala.exclusion_sets()
    # Peptide bond C(1)-N(3) is 1-2 excluded; C(1)-CA(5) is 1-3 excluded; CH3(0)-CA(5) is 1-4
    assert (1, 3) in excl
    assert (1, 5) in excl
    assert (0, 5) in p14


def test_alpha_helix_and_coarse_grained_elastic_mapping() -> None:
    helix = build_alpha_helix_peptide(n_res=8, polarizable=False)
    assert helix.n_atoms == 32
    cg = map_all_atom_to_cg(helix, cutoff_a=8.5)
    assert cg.n_atoms == 8
    assert any(b.is_elastic for b in cg.bonds)
    assert all(a.element == "CG" for a in cg.atoms)

    cg_direct = build_coarse_grained_elastic_protein(n_beads=10)
    assert cg_direct.n_atoms == 10


def test_host_guest_and_system_factory() -> None:
    hg = build_host_guest_unbinding_system()
    assert hg.n_atoms == 11
    assert len(hg.metadata["host_indices"]) == 8
    assert len(hg.metadata["guest_indices"]) == 3

    for name in ["water-dimer", "water-cluster", "ala-dipeptide", "helix8", "host-guest", "cg-protein"]:
        sys_obj = build_system_by_name(name)
        assert sys_obj.n_atoms >= 6

    with pytest.raises(ValueError):
        build_system_by_name("nonexistent-system")


def test_pdb_round_trip_and_clone_independence() -> None:
    ala = build_alanine_dipeptide_system()
    pdb_txt = export_pdb_block(ala)
    parsed = parse_pdb_block(pdb_txt, polarizable_heavy=True)
    assert parsed.n_atoms == ala.n_atoms
    assert np.allclose(parsed.positions, ala.positions, atol=1e-3)

    cloned = ala.clone()
    cloned.positions[0, 0] += 5.0
    assert not np.allclose(cloned.positions, ala.positions)
    with pytest.raises(ValueError):
        MolecularSystem(name="bad", atoms=ala.atoms, positions=np.zeros((2, 3)))


def test_atom_spec_drude_mass_validation_and_empty_pdb() -> None:
    bad_atom = AtomSpec(
        index=0, name="H", residue_name="WAT", residue_id=1, element="H",
        mass=0.3, charge=0.0, alpha_pol=0.5, drude_mass=0.4,
    )
    with pytest.raises(ValueError):
        _ = bad_atom.core_mass

    with pytest.raises(ValueError):
        parse_pdb_block("REMARK EMPTY PDB")


def test_energy_breakdown_totals_and_periodic_pressure() -> None:
    from solvato_dyne.dynamics.runner import initialize_velocities
    from solvato_dyne.forcefield.engine import ForceFieldEngine

    cluster = build_polarizable_water_dimer_or_cluster(n_molecules=4, box_size=14.0, polarizable=True)
    initialize_velocities(cluster, temperature_k=300.0, seed=3)
    eng = ForceFieldEngine(cluster)
    eb, _, _ = eng.evaluate(scf_drude=True)
    assert np.isfinite(eb.total_potential)
    assert np.isfinite(eb.pressure_bar)
    assert np.isclose(eb.total_energy, eb.total_potential + eb.total_kinetic)


def test_box_and_builder_input_validation() -> None:
    with pytest.raises(ValueError):
        build_polarizable_water_dimer_or_cluster(n_molecules=0)
    with pytest.raises(ValueError):
        build_alpha_helix_peptide(n_res=2)
    with pytest.raises(ValueError):
        BoxSpec(np.eye(2))


def test_public_package_exports() -> None:
    import solvato_dyne

    assert solvato_dyne.__version__ == "0.1.0"
    for sym in ["MDSimulator", "WellTemperedMetadynamics", "TemperatureReplicaExchange", "SteeredMDSimulator"]:
        assert hasattr(solvato_dyne, sym)
