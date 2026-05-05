"""
Shared fixtures and helpers for Lingo3DMol tests.

Coordinates used across fixtures
---------------------------------
Ligand centre      : (0.0,  0.0,  0.0)

ATOM distances from ligand:
  atom 1 CA ALA 1  (3.0,  0.0,  0.0) dist = 3.0   → inside any cutoff ≥ 3
  atom 2 CA GLY 2  (0.0,  5.0,  0.0) dist = 5.0   → inside cutoff ≥ 5
  atom 3 CA VAL 3  (8.0,  0.0,  0.0) dist = 8.0   → inside cutoff ≥ 8  (edge)
  atom 4 CA LEU 4 (10.0,  0.0,  0.0) dist = 10.0  → outside cutoff 8
  atom 5 CA ILE 5  (0.0,  0.0, 15.0) dist = 15.0  → far outside
  atom 6 N  ALA 1  (2.0,  0.0,  0.0) dist = 2.0   → closest protein atom
  atom 7 O  ALA 1  (0.0,  2.5,  0.0) dist = 2.5   → second closest
  atom 8 CA SER 6  (0.0,  9.0,  0.0) dist = 9.0   → outside cutoff 8

At cutoff=8.0 → 5 atoms: 1,2,3,6,7
At cutoff=5.0 → 4 atoms: 1,2,6,7
"""

import os
import sys
import pytest
from pathlib import Path

# Make project root importable from any working directory
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

FIXTURES_DIR = Path(__file__).parent / 'fixtures'


@pytest.fixture
def protein_with_ligand_pdb() -> Path:
    """Full protein PDB with a non-solvent HETATM ligand and a water HETATM."""
    return FIXTURES_DIR / 'protein_with_ligand.pdb'


@pytest.fixture
def protein_no_ligand_pdb() -> Path:
    """Protein PDB with no ligand (only water HETATM) → triggers centroid fallback."""
    return FIXTURES_DIR / 'protein_no_ligand.pdb'


@pytest.fixture
def pocket_small_pdb() -> Path:
    """Pre-extracted pocket PDB (no HETATM records)."""
    return FIXTURES_DIR / 'pocket_small.pdb'


@pytest.fixture
def two_protein_dir(tmp_path) -> Path:
    """
    Temporary directory containing two copies of protein_with_ligand.pdb,
    useful for testing batch (directory-level) processing.
    """
    import shutil
    src = FIXTURES_DIR / 'protein_with_ligand.pdb'
    shutil.copy(src, tmp_path / 'protA.pdb')
    shutil.copy(src, tmp_path / 'protB.pdb')
    return tmp_path


@pytest.fixture
def mol_output_dir(tmp_path) -> Path:
    """
    Simulates the raw output directory structure created by inference_avoid_clash.py.
    Layout: raw_dir/0/pocket_name/*.mol
    """
    pocket_dir = tmp_path / 'raw' / '0' / 'pocket_test'
    pocket_dir.mkdir(parents=True)
    (pocket_dir / '0_pred_2_pocket_test.mol').write_text('molecule A')
    (pocket_dir / '1_pred_3_pocket_test.mol').write_text('molecule B')
    return tmp_path
