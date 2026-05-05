"""
Tests for the pure-Python utility functions in pipeline.py.

These tests cover config loading, file discovery, input-list writing,
and output collection — all without requiring a GPU or model weights.
"""

import os
import shutil
import pytest
from pathlib import Path
from pipeline import (
    DEFAULTS,
    _deep_update,
    load_config,
    find_pdb_files,
    write_input_list,
    collect_results,
)


# ── _deep_update ──────────────────────────────────────────────────────────────

class TestDeepUpdate:

    def test_flat_key_overridden(self):
        base = {'a': 1, 'b': 2}
        _deep_update(base, {'b': 99})
        assert base == {'a': 1, 'b': 99}

    def test_nested_key_overridden(self):
        base = {'gen': {'num': 20, 'hours': 3}}
        _deep_update(base, {'gen': {'num': 5}})
        assert base['gen']['num'] == 5
        assert base['gen']['hours'] == 3  # untouched

    def test_new_key_added(self):
        base = {'a': 1}
        _deep_update(base, {'b': 2})
        assert base['b'] == 2

    def test_nested_dict_not_replaced_whole(self):
        base = {'gen': {'num': 20, 'hours': 3, 'batch': 8}}
        _deep_update(base, {'gen': {'batch': 16}})
        assert base['gen']['num'] == 20   # preserved
        assert base['gen']['hours'] == 3  # preserved
        assert base['gen']['batch'] == 16 # updated


# ── load_config ───────────────────────────────────────────────────────────────

yaml = pytest.importorskip('yaml', reason='pyyaml not installed')


class TestLoadConfig:

    def test_missing_file_returns_defaults(self, tmp_path):
        cfg = load_config(str(tmp_path / 'no_such_file.yaml'))
        assert cfg['generation']['num_molecules'] == DEFAULTS['generation']['num_molecules']

    def test_empty_yaml_returns_defaults(self, tmp_path):
        config_file = tmp_path / 'config.yaml'
        config_file.write_text('')
        cfg = load_config(str(config_file))
        assert cfg['generation']['num_molecules'] == DEFAULTS['generation']['num_molecules']

    def test_partial_override_preserves_rest(self, tmp_path):
        config_file = tmp_path / 'config.yaml'
        config_file.write_text('generation:\n  num_molecules: 5\n')
        cfg = load_config(str(config_file))
        assert cfg['generation']['num_molecules'] == 5
        assert cfg['generation']['max_hours'] == DEFAULTS['generation']['max_hours']

    def test_override_nested_key(self, tmp_path):
        config_file = tmp_path / 'config.yaml'
        config_file.write_text('pocket:\n  distance_cutoff: 12.0\n')
        cfg = load_config(str(config_file))
        assert cfg['pocket']['distance_cutoff'] == 12.0
        assert cfg['pocket']['max_atoms'] == DEFAULTS['pocket']['max_atoms']

    def test_override_does_not_mutate_defaults(self, tmp_path):
        config_file = tmp_path / 'config.yaml'
        config_file.write_text('generation:\n  num_molecules: 999\n')
        load_config(str(config_file))
        assert DEFAULTS['generation']['num_molecules'] == 20

    def test_all_required_sections_present(self, tmp_path):
        cfg = load_config(str(tmp_path / 'no_such.yaml'))
        for section in ('generation', 'model', 'pocket', 'output', 'advanced'):
            assert section in cfg, f'Missing section: {section}'

    def test_gpu_default_is_string(self, tmp_path):
        cfg = load_config(str(tmp_path / 'no_such.yaml'))
        assert isinstance(cfg['model']['gpu'], str)


# ── find_pdb_files ────────────────────────────────────────────────────────────

class TestFindPdbFiles:

    def test_single_pdb_file(self, protein_with_ligand_pdb):
        result = find_pdb_files(str(protein_with_ligand_pdb))
        assert result == [str(protein_with_ligand_pdb)]

    def test_non_pdb_file_returns_empty(self, tmp_path):
        f = tmp_path / 'notes.txt'
        f.write_text('hello')
        assert find_pdb_files(str(f)) == []

    def test_directory_finds_all_pdb_files(self, tmp_path):
        (tmp_path / 'a.pdb').write_text('pdb')
        (tmp_path / 'b.pdb').write_text('pdb')
        (tmp_path / 'c.txt').write_text('txt')
        result = find_pdb_files(str(tmp_path))
        assert len(result) == 2
        assert all(r.endswith('.pdb') for r in result)

    def test_directory_result_is_sorted(self, tmp_path):
        (tmp_path / 'z.pdb').write_text('pdb')
        (tmp_path / 'a.pdb').write_text('pdb')
        (tmp_path / 'm.pdb').write_text('pdb')
        result = find_pdb_files(str(tmp_path))
        assert result == sorted(result)

    def test_empty_directory_returns_empty_list(self, tmp_path):
        assert find_pdb_files(str(tmp_path)) == []

    def test_non_existent_path_returns_empty(self, tmp_path):
        result = find_pdb_files(str(tmp_path / 'does_not_exist'))
        assert result == []


# ── write_input_list ──────────────────────────────────────────────────────────

class TestWriteInputList:

    def test_creates_file(self, tmp_path):
        out = tmp_path / 'input_list.txt'
        write_input_list(['/some/pocket.pdb'], str(out))
        assert out.exists()

    def test_each_line_has_double_comma_prefix(self, tmp_path):
        out = tmp_path / 'input_list.txt'
        write_input_list(['/a/pocket.pdb', '/b/pocket.pdb'], str(out))
        lines = out.read_text().splitlines()
        assert all(l.startswith(',,') for l in lines)

    def test_paths_written_correctly(self, tmp_path):
        pockets = ['/prot/A_pocket.pdb', '/prot/B_pocket.pdb']
        out = tmp_path / 'input_list.txt'
        write_input_list(pockets, str(out))
        lines = out.read_text().splitlines()
        assert lines[0] == ',,/prot/A_pocket.pdb'
        assert lines[1] == ',,/prot/B_pocket.pdb'

    def test_one_line_per_pocket(self, tmp_path):
        pockets = [f'/pocket_{i}.pdb' for i in range(7)]
        out = tmp_path / 'input_list.txt'
        write_input_list(pockets, str(out))
        lines = [l for l in out.read_text().splitlines() if l]
        assert len(lines) == 7

    def test_empty_list_creates_empty_file(self, tmp_path):
        out = tmp_path / 'input_list.txt'
        write_input_list([], str(out))
        assert out.read_text() == ''


# ── collect_results ───────────────────────────────────────────────────────────

class TestCollectResults:

    def test_mol_files_moved_to_final_dir(self, mol_output_dir):
        raw_dir = mol_output_dir / 'raw'
        final_dir = mol_output_dir / 'final'
        final_dir.mkdir()
        collect_results(str(raw_dir), str(final_dir), gpu='0')
        mol_files = list(final_dir.rglob('*.mol'))
        assert len(mol_files) == 2

    def test_gpu_subdir_removed_after_collection(self, mol_output_dir):
        raw_dir = mol_output_dir / 'raw'
        final_dir = mol_output_dir / 'final'
        final_dir.mkdir()
        collect_results(str(raw_dir), str(final_dir), gpu='0')
        gpu_subdir = raw_dir / '0'
        assert not gpu_subdir.exists()

    def test_protein_subfolders_preserved(self, mol_output_dir):
        raw_dir = mol_output_dir / 'raw'
        final_dir = mol_output_dir / 'final'
        final_dir.mkdir()
        collect_results(str(raw_dir), str(final_dir), gpu='0')
        pocket_dir = final_dir / 'pocket_test'
        assert pocket_dir.is_dir()

    def test_missing_gpu_subdir_does_not_raise(self, tmp_path):
        raw_dir = tmp_path / 'raw'
        raw_dir.mkdir()
        final_dir = tmp_path / 'final'
        final_dir.mkdir()
        # No gpu subdir inside raw — should complete silently
        collect_results(str(raw_dir), str(final_dir), gpu='0')

    def test_merge_when_destination_already_exists(self, tmp_path):
        """
        If a protein sub-folder already exists in final_dir, files from the
        GPU subdir should be merged into it rather than replacing it.
        """
        # pre-existing file in final_dir
        existing = tmp_path / 'final' / 'pocket_test'
        existing.mkdir(parents=True)
        (existing / 'pre_existing.mol').write_text('old')

        # new files in raw_dir/0/pocket_test/
        new_src = tmp_path / 'raw' / '0' / 'pocket_test'
        new_src.mkdir(parents=True)
        (new_src / 'new_mol.mol').write_text('new')

        collect_results(str(tmp_path / 'raw'), str(tmp_path / 'final'), gpu='0')
        merged = tmp_path / 'final' / 'pocket_test'
        assert (merged / 'pre_existing.mol').exists()
        assert (merged / 'new_mol.mol').exists()
