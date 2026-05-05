"""
Tests for prepare_pockets.py — pocket extraction from protein PDB files.

Each test covers one behaviour of extract_pocket() or process_directory().
The fixture coordinates are documented in conftest.py.
"""

import os
import pytest
from pathlib import Path
from prepare_pockets import extract_pocket, process_directory


# ── extract_pocket: happy path ─────────────────────────────────────────────────

class TestExtractPocketWithLigand:

    def test_returns_atom_count(self, protein_with_ligand_pdb, tmp_path):
        out = tmp_path / 'pocket.pdb'
        count = extract_pocket(str(protein_with_ligand_pdb), str(out))
        assert count == 5  # atoms 1,2,3,6,7 within default 8.0 Å

    def test_output_file_created(self, protein_with_ligand_pdb, tmp_path):
        out = tmp_path / 'pocket.pdb'
        extract_pocket(str(protein_with_ligand_pdb), str(out))
        assert out.exists()

    def test_output_contains_only_atom_records(self, protein_with_ligand_pdb, tmp_path):
        out = tmp_path / 'pocket.pdb'
        extract_pocket(str(protein_with_ligand_pdb), str(out))
        lines = out.read_text().splitlines()
        non_atom = [l for l in lines if l and not l.startswith('ATOM') and l != 'END']
        # CRYST1 is allowed, everything else must be ATOM or END
        non_atom = [l for l in non_atom if not l.startswith('CRYST1')]
        assert non_atom == [], f'Unexpected records in output: {non_atom}'

    def test_output_ends_with_end_record(self, protein_with_ligand_pdb, tmp_path):
        out = tmp_path / 'pocket.pdb'
        extract_pocket(str(protein_with_ligand_pdb), str(out))
        last_line = out.read_text().strip().splitlines()[-1]
        assert last_line == 'END'

    def test_cryst1_line_preserved(self, protein_with_ligand_pdb, tmp_path):
        out = tmp_path / 'pocket.pdb'
        extract_pocket(str(protein_with_ligand_pdb), str(out))
        first_line = out.read_text().splitlines()[0]
        assert first_line.startswith('CRYST1')

    def test_hetatm_ligand_not_written_to_output(self, protein_with_ligand_pdb, tmp_path):
        out = tmp_path / 'pocket.pdb'
        extract_pocket(str(protein_with_ligand_pdb), str(out))
        content = out.read_text()
        assert 'HETATM' not in content

    def test_water_not_treated_as_ligand(self, protein_with_ligand_pdb, tmp_path):
        """
        The fixture has a HOH HETATM at (2,2,2).  If HOH were treated as a
        ligand, the pocket centre would shift and atom counts would differ.
        The HOH should be silently ignored.
        """
        out = tmp_path / 'pocket.pdb'
        count = extract_pocket(str(protein_with_ligand_pdb), str(out))
        # With only the real LIG at (0,0,0), we expect exactly 5 atoms inside 8 Å.
        assert count == 5


# ── extract_pocket: distance cutoff ───────────────────────────────────────────

class TestDistanceCutoff:

    def test_smaller_cutoff_fewer_atoms(self, protein_with_ligand_pdb, tmp_path):
        out5 = tmp_path / 'pocket_5.pdb'
        out8 = tmp_path / 'pocket_8.pdb'
        count5 = extract_pocket(str(protein_with_ligand_pdb), str(out5), distance_cutoff=5.0)
        count8 = extract_pocket(str(protein_with_ligand_pdb), str(out8), distance_cutoff=8.0)
        assert count5 < count8

    def test_cutoff_5_excludes_atoms_beyond_5A(self, protein_with_ligand_pdb, tmp_path):
        out = tmp_path / 'pocket.pdb'
        count = extract_pocket(str(protein_with_ligand_pdb), str(out), distance_cutoff=5.0)
        # atoms 1(3.0), 2(5.0 ≤ 5.0), 6(2.0), 7(2.5) → 4
        assert count == 4

    def test_atom_exactly_at_cutoff_is_included(self, protein_with_ligand_pdb, tmp_path):
        """Atom 2 is at exactly 5.0 Å; the condition is ≤ so it must be included."""
        out = tmp_path / 'pocket.pdb'
        count5 = extract_pocket(str(protein_with_ligand_pdb), str(out), distance_cutoff=5.0)
        out2 = tmp_path / 'pocket_4.pdb'
        count4 = extract_pocket(str(protein_with_ligand_pdb), str(out2), distance_cutoff=4.9)
        # at 4.9 Å atom 2 (dist=5.0) is excluded; one fewer atom
        assert count5 == count4 + 1

    def test_very_large_cutoff_includes_all_protein_atoms(self, protein_with_ligand_pdb, tmp_path):
        out = tmp_path / 'pocket.pdb'
        count = extract_pocket(str(protein_with_ligand_pdb), str(out), distance_cutoff=100.0)
        # All 8 ATOM records should be included
        assert count == 8


# ── extract_pocket: max_atoms limit ───────────────────────────────────────────

class TestMaxAtoms:

    def test_max_atoms_caps_output(self, protein_with_ligand_pdb, tmp_path):
        out = tmp_path / 'pocket.pdb'
        count = extract_pocket(str(protein_with_ligand_pdb), str(out), max_atoms=3)
        assert count == 3

    def test_max_atoms_keeps_closest_atoms(self, protein_with_ligand_pdb, tmp_path):
        """
        When capped, the output must contain the atoms closest to the ligand.
        At cutoff=8.0 and max_atoms=3 the three closest are:
          atom 6 N  dist=2.0
          atom 7 O  dist=2.5
          atom 1 CA dist=3.0
        """
        out = tmp_path / 'pocket.pdb'
        extract_pocket(str(protein_with_ligand_pdb), str(out), max_atoms=3)
        content = out.read_text()
        # Unique check: no atom at x=5.0 or x=8.0 should appear
        lines = [l for l in content.splitlines() if l.startswith('ATOM')]
        xs = [float(l[30:38]) for l in lines]
        assert 5.0 not in xs, 'Atom at dist=5.0 should be excluded by max_atoms=3'
        assert 8.0 not in xs, 'Atom at dist=8.0 should be excluded by max_atoms=3'

    def test_max_atoms_larger_than_pocket_has_no_effect(self, protein_with_ligand_pdb, tmp_path):
        out = tmp_path / 'pocket.pdb'
        count = extract_pocket(str(protein_with_ligand_pdb), str(out), max_atoms=9999)
        assert count == 5  # same as default at cutoff=8.0


# ── extract_pocket: no-ligand fallback ────────────────────────────────────────

class TestNoLigandFallback:

    def test_no_ligand_still_produces_output(self, protein_no_ligand_pdb, tmp_path):
        out = tmp_path / 'pocket.pdb'
        count = extract_pocket(str(protein_no_ligand_pdb), str(out))
        assert count is not None and count > 0

    def test_no_ligand_output_file_created(self, protein_no_ligand_pdb, tmp_path):
        out = tmp_path / 'pocket.pdb'
        extract_pocket(str(protein_no_ligand_pdb), str(out))
        assert out.exists()

    def test_no_ligand_output_contains_atom_records(self, protein_no_ligand_pdb, tmp_path):
        out = tmp_path / 'pocket.pdb'
        extract_pocket(str(protein_no_ligand_pdb), str(out))
        lines = [l for l in out.read_text().splitlines() if l.startswith('ATOM')]
        assert len(lines) > 0


# ── extract_pocket: error handling ────────────────────────────────────────────

class TestErrorHandling:

    def test_missing_file_returns_none(self, tmp_path):
        out = tmp_path / 'pocket.pdb'
        result = extract_pocket('does_not_exist.pdb', str(out))
        assert result is None

    def test_missing_file_does_not_create_output(self, tmp_path):
        out = tmp_path / 'pocket.pdb'
        extract_pocket('does_not_exist.pdb', str(out))
        assert not out.exists()

    def test_creates_output_parent_directory(self, protein_with_ligand_pdb, tmp_path):
        out = tmp_path / 'deep' / 'nested' / 'pocket.pdb'
        extract_pocket(str(protein_with_ligand_pdb), str(out))
        assert out.exists()


# ── process_directory ─────────────────────────────────────────────────────────

class TestProcessDirectory:

    def test_batch_returns_list_of_pocket_paths(self, two_protein_dir, tmp_path):
        out_dir = tmp_path / 'pockets'
        results = process_directory(str(two_protein_dir), str(out_dir))
        assert len(results) == 2

    def test_batch_output_files_exist(self, two_protein_dir, tmp_path):
        out_dir = tmp_path / 'pockets'
        results = process_directory(str(two_protein_dir), str(out_dir))
        for path in results:
            assert Path(path).exists()

    def test_batch_output_named_with_pocket_suffix(self, two_protein_dir, tmp_path):
        out_dir = tmp_path / 'pockets'
        results = process_directory(str(two_protein_dir), str(out_dir))
        for path in results:
            assert path.endswith('_pocket.pdb')

    def test_batch_creates_output_directory(self, two_protein_dir, tmp_path):
        out_dir = tmp_path / 'new_pockets_dir'
        process_directory(str(two_protein_dir), str(out_dir))
        assert out_dir.is_dir()

    def test_empty_directory_returns_empty_list(self, tmp_path):
        empty_dir = tmp_path / 'empty'
        empty_dir.mkdir()
        out_dir = tmp_path / 'pockets'
        results = process_directory(str(empty_dir), str(out_dir))
        assert results == []

    def test_non_pdb_files_are_ignored(self, tmp_path):
        prot_dir = tmp_path / 'proteins'
        prot_dir.mkdir()
        (prot_dir / 'notes.txt').write_text('ignore me')
        (prot_dir / 'data.csv').write_text('also ignore')
        out_dir = tmp_path / 'pockets'
        results = process_directory(str(prot_dir), str(out_dir))
        assert results == []
