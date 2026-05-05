# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Lingo3DMol generates 3D drug-like molecules conditioned on a protein binding pocket. It combines a language-model-style fragment decoder with geometric deep learning to produce molecules with valid 3D coordinates that fit inside the pocket without clashing.

## Environment Setup

```bash
conda create -n lingo3dmol python=3.8
conda activate lingo3dmol
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge
pip install scipy==1.7.3 pandas==1.5.1 numpy==1.20.3 rdkit==2022.09.1 pyyaml
```

Model checkpoints go in `checkpoint/` (download `contact.pkl` and `gen_mol.pkl` per readme.md).

## Running the Pipeline

All commands must be run from the repo root.

```bash
# Pocket PDB files → generate molecules directly
python pipeline.py --input datasets/hamdi_pockets_proper/ --name my_run

# Full protein PDB files → auto-extract pockets → generate molecules
python pipeline.py --input datasets/hamdi_protiens/ --extract-pockets --name my_run

# Single protein
python pipeline.py --input my_protein.pdb --extract-pockets

# Using the pre-configured run.sh
sh run.sh
```

Results land in `results/<run_name>/`, organized one sub-folder per protein.

## Configuring a Run

Edit `config.yaml` before running. The most commonly tuned keys are:

| Key | Default | Effect |
|-----|---------|--------|
| `generation.num_molecules` | 20 | Stop per protein after this many valid molecules |
| `generation.max_hours` | 3 | Hard time limit per protein |
| `generation.fragment_batch_size` | 8 | Sampling batch size (VRAM ↔ diversity tradeoff) |
| `pocket.distance_cutoff` | 8.0 | Å radius for pocket extraction |
| `advanced.temperature` | 1.0 | Sampling temperature |

## Pocket Preparation (standalone)

```bash
# One directory of full proteins → one directory of pockets
python prepare_pockets.py --input datasets/hamdi_protiens/ --output datasets/hamdi_pockets/

# Single file
python prepare_pockets.py --input my_protein.pdb --output my_pocket.pdb --distance 10.0
```

The legacy `extract_pockets_around_ligands.py` at the root is superseded by `prepare_pockets.py`.

## Input List Format

`pipeline.py` generates the input list automatically.  If you call
`inference/inference_avoid_clash.py` directly, each line must be:
```
ligand.sdf,nci_contacts.npy,pocket.pdb
```
Leave the first two fields empty (`,,pocket.pdb`) when you have no pre-computed NCI or ligand file — the contact model predicts NCI on the fly.

## Code Architecture

### Two-model pipeline

**Contact model** (`model/transformer_v1_res_mp1.py` → `checkpoint/contact.pkl`):
Encoder-only transformer that reads pocket atom coordinates + residue types and predicts which residues are likely to form non-covalent interactions (NCI) with the ligand. Output is used to select `topk` anchor residues that guide generation.

**Molecule generation model** (`model/transformer_v1_res_fac2.py` → `checkpoint/gen_mol.pkl`):
Encoder–decoder transformer. The encoder processes the pocket; the decoder autoregressively generates molecule tokens + 3D coordinate predictions. Uses bias-guided attention (`DecoderLayerBias`) and factorized coordinate prediction (independent x/y/z via `make_cube`).

Both models share building blocks from `model/Module.py`: `MultiHeadedAttention`, `MultiHeadedAttentionBias`, `edge_vector` (relative 3D coordinate embedding), `cdist` (pairwise distance embedding), `MLP`.

### Coordinate system

The entire 3D space is discretized at **0.1 Å resolution** into a **240 × 240 × 240** voxel grid (24 Å box centered on the pocket). All model inputs and outputs use integer voxel indices, converted back to real-world Å coordinates only at save time via `changepos()` in `inference/inference_avoid_clash.py`.

### Fragment-based generation

`inference/inference_avoid_clash.py` implements a recursive tree search (`molecular_workflow`) that builds molecules fragment-by-fragment:

1. `go_factory()` — calls the decoder to produce `gen_frag_set` fragment candidates in parallel.
2. `get_partial_to_warehouse()` — filters valid fragments, deduplicates by SMILES, scores each by **value score** (model log-prob average) + **anchor score** (proximity to contact anchors), keeps top 20%.
3. `molecular_workflow()` — recursively appends fragments until a complete molecule (end token `2`) is found or the fragment budget is exhausted.
4. `CollisionCheck` (`inference/cube_collision_check.py`) — voxelizes the pocket atoms and rejects any generated atom that occupies an occupied voxel (within `clash_distance` Å).

### Molecule tokenization

`util/fragmol_frag_zyh.py` (`FragmolUtil`) maps SMILES-like tokens to integer indices. Special tokens: `0=pad`, `1=start`, `2=end`, `3=sep` (fragment boundary). Atom tokens cover C, c, N, n, S, s, O, o, F, Cl, Br, [nH], plus bond/ring/branch characters. Max sequence length is 100 tokens.

### Data loading

`dataloader/dataloader_case_nci_res_merge.py` (`mydataset`):
- Reads pocket PDB via `util/pocket_code_all.py` (`PocketCode.pocketCodeNCI`) which encodes atom elements and residue types into integer indices placed on the voxel grid.
- Loads optional pre-computed NCI `.npy` arrays and ligand `.sdf` coordinates.
- Returns `(coords, residue, atom_type, mask, center, index, contact_prob, contact_scaffold_prob)`.

### New automation layer

| File | Purpose |
|------|---------|
| `pipeline.py` | End-to-end entry point: PDB in → .mol files out |
| `prepare_pockets.py` | Standalone pocket extraction module + CLI |
| `config.yaml` | All tunable parameters in one place |
| `run.sh` | Simple wrapper with pre-configured options |

## Testing

```bash
# Fast unit tests — no GPU, no checkpoints, ~3 seconds
pytest tests/

# Verbose output
pytest tests/ -v

# Integration tests only (requires CUDA GPU + checkpoint/contact.pkl + checkpoint/gen_mol.pkl)
pytest tests/test_integration.py -v

# One specific test file
pytest tests/test_prepare_pockets.py -v
```

`pytest.ini` sets `testpaths = tests` so bare `pytest` always targets the right directory.

### Test structure

| File | Scope | Key functions under test |
|------|-------|--------------------------|
| `test_prepare_pockets.py` | Unit | `extract_pocket`, `process_directory` |
| `test_pipeline_utils.py` | Unit | `_deep_update`, `load_config`, `find_pdb_files`, `write_input_list`, `collect_results` |
| `test_fragmol.py` | Unit | `FragmolUtil` vocabulary, `decode3d`, `mergeSmiles3D` |
| `test_integration.py` | Integration | Full pipeline, real datasets (auto-skipped without GPU+checkpoints) |

### Fixture geometry

`tests/fixtures/protein_with_ligand.pdb` is built around a ligand at `(0, 0, 0)` with protein atoms at known distances. Expected atom counts are therefore exact for any cutoff:

| Cutoff | Expected atoms |
|--------|---------------|
| 5.0 Å | 4 |
| 8.0 Å | 5 (includes atom at exactly 8.0 Å — tests `≤` boundary) |
| 100.0 Å | 8 (all protein atoms) |

When writing new tests for `prepare_pockets.py`, use these values as ground truth rather than recomputing.

### Adding new tests

- Place new test files in `tests/` prefixed with `test_`.
- Tests that require GPU must use the `@pytest.mark.skipif(not torch.cuda.is_available(), ...)` pattern — see `test_integration.py` for the `needs_gpu_and_checkpoints` decorator.
- Tests that require rdkit should call `pytest.importorskip('rdkit')` at module level — see `test_fragmol.py`.
- All imports of project modules work automatically because `tests/conftest.py` inserts the project root into `sys.path`.
