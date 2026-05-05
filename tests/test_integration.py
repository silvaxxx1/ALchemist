"""
Integration tests — require a CUDA GPU and the pre-trained checkpoints.

These tests are skipped automatically when:
  - No CUDA GPU is available, or
  - checkpoint/contact.pkl or checkpoint/gen_mol.pkl are missing.

Run them explicitly with:
    pytest tests/test_integration.py -v

They exercise the full end-to-end pipeline with a real (but short) run
to confirm model weights load correctly and molecules are generated.
"""

import os
import sys
import subprocess
import pytest
from pathlib import Path

ROOT = Path(__file__).parent.parent

# ── skip conditions ────────────────────────────────────────────────────────────

def _cuda_available():
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def _checkpoints_available():
    return (
        (ROOT / 'checkpoint' / 'contact.pkl').exists()
        and (ROOT / 'checkpoint' / 'gen_mol.pkl').exists()
    )


needs_gpu_and_checkpoints = pytest.mark.skipif(
    not (_cuda_available() and _checkpoints_available()),
    reason='Requires a CUDA GPU and checkpoint/contact.pkl + checkpoint/gen_mol.pkl',
)


# ── helpers ────────────────────────────────────────────────────────────────────

def run_pipeline(*args, cwd=None) -> subprocess.CompletedProcess:
    """Run pipeline.py with the given arguments and return the result."""
    cmd = [sys.executable, 'pipeline.py'] + list(args)
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=str(cwd or ROOT),
    )


# ── full pipeline: pocket files ────────────────────────────────────────────────

@needs_gpu_and_checkpoints
def test_pipeline_generates_mol_files_from_pocket(tmp_path):
    """
    Run the pipeline on a single Hamdi pocket file with a very short budget
    (1 molecule, 0.1 hours) and confirm at least one .mol file is produced.
    """
    pocket = ROOT / 'datasets' / 'hamdi_pockets_proper' / '2FOM_Dengue_pocket.pdb'
    if not pocket.exists():
        pytest.skip('Hamdi pocket files not found')

    # Write a minimal input list
    input_list = tmp_path / 'input.txt'
    input_list.write_text(f',,{pocket}\n')

    result = subprocess.run(
        [
            sys.executable, 'inference/inference_avoid_clash.py',
            '--cuda', '0',
            '--cuda_list', '0',
            '--input_list', str(input_list),
            '--savedir', str(tmp_path / 'raw') + '/',
            '--contact_path', 'checkpoint/contact.pkl',
            '--caption_path', 'checkpoint/gen_mol.pkl',
            '--gennums', '1',
            '--max_run_hours', '1',
            '--gen_frag_set', '4',
            '--frag_len_add', '20',
        ],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
    )
    assert result.returncode == 0, f'Inference failed:\n{result.stderr}'
    mol_files = list(tmp_path.rglob('*.mol'))
    assert len(mol_files) >= 1, 'No .mol files were generated'


@needs_gpu_and_checkpoints
def test_pipeline_entry_point_pocket_mode(tmp_path):
    """
    Run pipeline.py directly in pocket mode with a minimal budget.
    Checks that the script exits cleanly and creates at least one molecule.
    """
    pocket_dir = ROOT / 'datasets' / 'hamdi_pockets_proper'
    if not pocket_dir.exists():
        pytest.skip('Hamdi pocket files not found')

    result = run_pipeline(
        '--input', str(pocket_dir),
        '--name', 'integration_test',
        '--output', str(tmp_path / 'results'),
        '--config', str(ROOT / 'config.yaml'),
        cwd=ROOT,
    )
    assert result.returncode == 0, (
        f'pipeline.py exited with code {result.returncode}:\n{result.stderr}'
    )
    mol_files = list((tmp_path / 'results').rglob('*.mol'))
    assert len(mol_files) >= 1


@needs_gpu_and_checkpoints
def test_pipeline_entry_point_extract_pockets_mode(tmp_path):
    """
    Run pipeline.py with --extract-pockets on the full protein structures.
    Confirms pocket extraction + generation runs end-to-end.
    """
    protein_dir = ROOT / 'datasets' / 'hamdi_protiens'
    if not protein_dir.exists():
        pytest.skip('Hamdi protein files not found')

    result = run_pipeline(
        '--input', str(protein_dir),
        '--extract-pockets',
        '--name', 'integration_extract_test',
        '--output', str(tmp_path / 'results'),
        cwd=ROOT,
    )
    assert result.returncode == 0, (
        f'pipeline.py exited with code {result.returncode}:\n{result.stderr}'
    )
    # Check pockets were extracted
    pockets = list((tmp_path / 'results' / 'integration_extract_test' / 'pockets').glob('*.pdb'))
    assert len(pockets) > 0

    # Check molecules were generated
    mol_files = list((tmp_path / 'results').rglob('*.mol'))
    assert len(mol_files) >= 1


# ── pocket extraction sanity-check on real Hamdi proteins ─────────────────────

def test_pocket_extraction_on_hamdi_proteins():
    """
    No GPU needed. Verifies that prepare_pockets.py successfully extracts
    pockets from all Hamdi full-protein PDB files (requires the dataset).
    """
    protein_dir = ROOT / 'datasets' / 'hamdi_protiens'
    if not protein_dir.exists():
        pytest.skip('Hamdi protein files not found')

    from prepare_pockets import extract_pocket
    import tempfile

    pdb_files = sorted(protein_dir.glob('*.pdb'))
    assert len(pdb_files) > 0, 'No PDB files in hamdi_protiens/'

    with tempfile.TemporaryDirectory() as tmp:
        for pdb in pdb_files:
            out = Path(tmp) / f'{pdb.stem}_pocket.pdb'
            count = extract_pocket(str(pdb), str(out))
            assert count is not None and count > 0, (
                f'Failed to extract pocket from {pdb.name}: got count={count}'
            )
            assert out.exists()


def test_pocket_extraction_on_dude_samples():
    """
    No GPU needed. Spot-checks pocket extraction on 5 DUD-E pocket files
    (these are already extracted, but we verify they are valid PDB files
    that our loader can read without errors).
    """
    dude_dir = ROOT / 'datasets' / 'dude_pocket'
    if not dude_dir.exists():
        pytest.skip('DUD-E pocket files not found — unzip datasets/dude_pocket.zip first')

    pocket_files = sorted(dude_dir.glob('*.pdb'))[:5]
    assert len(pocket_files) > 0

    for pdb in pocket_files:
        content = pdb.read_text()
        atom_lines = [l for l in content.splitlines() if l.startswith('ATOM')]
        assert len(atom_lines) > 0, f'{pdb.name} contains no ATOM records'
        # Check coordinate columns parse cleanly
        for line in atom_lines[:3]:
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
            assert -9999 < x < 9999
