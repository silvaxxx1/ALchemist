# Lingo3DMol

**Pocket-Based 3D Molecule Generation with a Language Model**

Lingo3DMol generates drug-like 3D molecules tailored to a specific protein binding pocket. Given a pocket PDB file, it produces candidate molecules whose atoms are positioned inside the pocket, avoiding steric clashes with the protein.

The method combines:
- A **contact prediction transformer** that identifies which pocket residues are most likely to interact with a ligand
- A **fragment-based generative transformer** that builds the molecule token-by-token (SMILES fragments + 3D coordinates) guided by those contact anchors

Output molecules are saved as `.mol` files and can be opened directly in PyMOL, ChimeraX, or Avogadro.

---

## Table of Contents

1. [Hardware Requirements](#hardware-requirements)
2. [Installation](#installation)
3. [Downloading Model Checkpoints](#downloading-model-checkpoints)
4. [Quick Start](#quick-start)
5. [Step-by-Step Workflow](#step-by-step-workflow)
6. [Datasets Included](#datasets-included)
7. [Configuration Reference](#configuration-reference)
8. [Output Files](#output-files)
9. [Project Structure](#project-structure)
10. [How It Works](#how-it-works)
11. [Advanced Usage](#advanced-usage)
12. [Testing](#testing)
13. [License](#license)

---

## Hardware Requirements

| Resource | Minimum | Recommended |
|----------|---------|-------------|
| GPU VRAM | 5 GB | 8 GB+ |
| RAM | 16 GB | 32 GB |
| Disk | 2 GB | — |
| OS | Linux / macOS | Linux |

A CUDA-capable GPU is required. CPU-only inference is not supported.

---

## Installation

```bash
conda create -n lingo3dmol python=3.8
conda activate lingo3dmol

conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 \
              cudatoolkit=11.3 -c pytorch -c conda-forge

pip install scipy==1.7.3 pandas==1.5.1 numpy==1.20.3 rdkit==2022.09.1 pyyaml
```

> **All commands below must be run from the `Lingo3DMol/` directory** with the
> `lingo3dmol` conda environment active.

---

## Downloading Model Checkpoints

Download both files and place them inside the `checkpoint/` folder.

| File | URL | MD5 |
|------|-----|-----|
| `contact.pkl` | https://stonewise-lingo3dmol-public.s3.cn-northwest-1.amazonaws.com.cn/contact.pkl | `6a9313726141fcf9201b9b9470dc2a7e` |
| `gen_mol.pkl` | https://stonewise-lingo3dmol-public.s3.cn-northwest-1.amazonaws.com.cn/gen_mol.pkl | `452bd401667184ae43c9818e5bdb133b` |

```bash
# Verify checksums after downloading
md5sum checkpoint/contact.pkl   # should match 6a9313726141fcf9201b9b9470dc2a7e
md5sum checkpoint/gen_mol.pkl   # should match 452bd401667184ae43c9818e5bdb133b
```

---

## Quick Start

### For Hamdi (no coding required)

**Option A — You have pocket PDB files** (file names end in `_pocket.pdb`):

```bash
sh run.sh
```

Edit the `INPUT` path inside `run.sh` to point to your pocket folder, then run. That's it.

**Option B — You have full protein PDB files** (not yet trimmed to the pocket):

Open `run.sh`, uncomment the Option B block, comment out Option A, then run:

```bash
sh run.sh
```

The script extracts the binding pocket automatically (8 Å radius around the co-crystallized ligand) before generating molecules.

---

### For developers — direct pipeline

```bash
# Pocket files → generate molecules
python pipeline.py --input datasets/hamdi_pockets_proper/ --name my_run

# Full protein PDB files → auto-extract pockets → generate molecules
python pipeline.py --input datasets/hamdi_protiens/ --extract-pockets --name my_run

# Single protein
python pipeline.py --input my_protein.pdb --extract-pockets --name test

# Use a custom config
python pipeline.py --input pockets/ --config config.yaml --name my_run
```

Results are saved to `results/<run_name>/`.

---

## Step-by-Step Workflow

```
Full protein PDB
      │
      ▼  (if needed)
prepare_pockets.py ──────────────────────────────────────────────────────────►
  Extracts protein atoms within 8 Å of the ligand (HETATM records)           │
  Saves a *_pocket.pdb file                                                   │
                                                                              │
Pocket PDB ───────────────────────────────────────────────────────────────────►
      │
      ▼
Contact Transformer (checkpoint/contact.pkl)
  Reads pocket atom types + residue types + 3D coordinates
  Predicts per-residue NCI probability
  Selects top-K anchor residues to guide generation
      │
      ▼
Molecule Generation Transformer (checkpoint/gen_mol.pkl)
  Autoregressively generates SMILES tokens + 3D coordinates
  Works fragment-by-fragment (recursive tree search)
  Each new fragment is clash-checked against the pocket voxel grid
  Fragments scored by model confidence + anchor proximity
      │
      ▼
results/<run_name>/<protein_name>/
  0_pred_2_<protein>.mol
  1_pred_2_<protein>.mol
  ...
```

### Preparing pockets manually

If you want to run pocket extraction separately from generation:

```bash
# One directory of full proteins → one directory of pockets
python prepare_pockets.py \
    --input  datasets/hamdi_protiens/ \
    --output datasets/hamdi_pockets/ \
    --distance 8.0 \
    --max-atoms 450

# Single protein
python prepare_pockets.py \
    --input  my_protein.pdb \
    --output my_pocket.pdb
```

---

## Datasets Included

### Hamdi's viral and cancer targets — `datasets/hamdi_pockets_proper/`

14 pre-extracted pocket PDB files for antiviral and oncology research:

| PDB | Target | Disease |
|-----|--------|---------|
| 1AT3 | Herpes Simplex (HS) | Herpes |
| 1CMV | Herpes Simplex (HS) | Herpes |
| 1G3X | Herpes Simplex inhibitor binding site | Herpes |
| 1T3R | HIV reverse transcriptase | HIV |
| 2FOM | NS5 polymerase | Dengue |
| 3M5O | NS5B polymerase | Hepatitis C |
| 3SJO | 3C protease | Enterovirus |
| 5GXJ | NS5 methyltransferase | Zika |
| 6NZV | NS5B polymerase | Hepatitis C |
| 6XHM | Main protease (MPro) | COVID-19 |
| 6Y2G | Main protease (MPro) | COVID-19 |
| 7D7K | Main protease (MPro) | COVID-19 |
| 7KHP | Main protease (MPro) | COVID-19 |
| 7VXY | NS5 | Zika |

Full protein structures (for pocket re-extraction) are in `datasets/hamdi_protiens/`.

### CD38 / BCMA targets — `datasets/new_data_processed/`

3 targets relevant to multiple myeloma immunotherapy:

| PDB | Target |
|-----|--------|
| 1YH3 | CD38 (1.91 Å resolution) |
| 2EF1 | CD38 (2.40 Å resolution) |
| 6J7W | B-cell Maturation Antigen (BCMA) |

### DUD-E benchmark — `datasets/dude_pocket/`

104 pre-extracted pocket PDB files from the Directory of Useful Decoys–Enhanced benchmark, covering a broad range of protein families. Unzip `datasets/dude_pocket.zip` if the folder is empty.

```bash
cd datasets && unzip dude_pocket.zip && cd ..
```

---

## Configuration Reference

All parameters live in `config.yaml`. Edit and save — no code changes needed.

### `generation` section

| Key | Default | Description |
|-----|---------|-------------|
| `num_molecules` | `20` | Minimum molecules to generate per protein before stopping |
| `max_hours` | `3` | Hard time limit per protein (hours) |
| `fragment_batch_size` | `8` | Candidates sampled per step. Higher = more diverse but needs more VRAM. Safe range for a 8 GB GPU: 4–16. |

### `model` section

| Key | Default | Description |
|-----|---------|-------------|
| `contact_checkpoint` | `checkpoint/contact.pkl` | Path to the NCI contact prediction model |
| `generation_checkpoint` | `checkpoint/gen_mol.pkl` | Path to the molecule generation model |
| `gpu` | `"0"` | GPU index (use `"0"` for the first GPU) |

### `pocket` section

| Key | Default | Description |
|-----|---------|-------------|
| `distance_cutoff` | `8.0` | Å radius around the ligand used to define the pocket during extraction |
| `max_atoms` | `450` | Maximum pocket atoms to feed to the model |
| `clash_distance` | `2.5` | Minimum allowed distance (Å) between a generated atom and any pocket atom |

### `output` section

| Key | Default | Description |
|-----|---------|-------------|
| `directory` | `results/` | Base directory for all run outputs |

### `advanced` section

| Key | Default | Description |
|-----|---------|-------------|
| `nci_threshold` | `0.7` | Probability cutoff to classify a residue as an NCI contact |
| `top_k_anchors` | `5` | Number of anchor residues sampled to guide each generation step |
| `fragment_min_length` | `20` | Minimum token length per fragment. Larger = fewer, bigger fragments (faster but less diverse) |
| `temperature` | `1.0` | Sampling temperature. Lower (e.g. `0.7`) = more focused; higher (e.g. `1.3`) = more exploratory |

---

## Output Files

After a run, results are organized as:

```
results/
└── <run_name>/
    ├── input_list.txt              ← auto-generated input manifest
    ├── pockets/                    ← only present when --extract-pockets was used
    │   ├── 1AT3_HS_pocket.pdb
    │   └── ...
    ├── 1AT3_HS_pocket.pdb/         ← one folder per protein
    │   ├── 0_pred_2_1AT3_HS_pocket.pdb.mol
    │   ├── 1_pred_3_1AT3_HS_pocket.pdb.mol
    │   └── ...
    └── 2FOM_Dengue_pocket.pdb/
        └── ...
```

**Mol file naming:** `{index}_pred_{frag_level}_{pocket_name}.mol`
- `index` — sequential molecule number
- `frag_level` — number of fragments the molecule was built from (higher = longer molecule)

### Viewing molecules

Open `.mol` files in any of these free tools:

| Tool | Platform | Download |
|------|----------|----------|
| **PyMOL** | Linux / macOS / Windows | pymol.org |
| **ChimeraX** | Linux / macOS / Windows | rbvi.ucsf.edu/chimerax |
| **Avogadro** | Linux / macOS / Windows | avogadro.cc |

To view a generated molecule overlaid on the protein pocket in PyMOL:
```
PyMOL> load datasets/hamdi_pockets_proper/2FOM_Dengue_pocket.pdb
PyMOL> load results/my_run/2FOM_Dengue_pocket.pdb/0_pred_2_2FOM_Dengue_pocket.pdb.mol
PyMOL> show sticks
```

---

## Project Structure

```
Lingo3DMol/
│
├── pipeline.py                  ← Main entry point (start here)
├── prepare_pockets.py           ← Pocket extraction utility
├── config.yaml                  ← All parameters — edit this
├── run.sh                       ← Quick-start shell script
│
├── checkpoint/
│   ├── contact.pkl              ← NCI contact prediction model weights
│   └── gen_mol.pkl              ← Molecule generation model weights
│
├── model/
│   ├── Module.py                ← Shared building blocks (attention, MLP, embeddings)
│   ├── transformer_v1_res_mp1.py← Contact prediction transformer
│   └── transformer_v1_res_fac2.py← Molecule generation transformer
│
├── inference/
│   ├── inference_avoid_clash.py ← Core inference loop (fragment tree search)
│   └── cube_collision_check.py  ← Voxel-based clash filter
│
├── dataloader/
│   └── dataloader_case_nci_res_merge.py ← Pocket PDB → tensor pipeline
│
├── util/
│   ├── fragmol_frag_zyh.py      ← SMILES tokenizer + 3D mol assembler
│   └── pocket_code_all.py       ← Pocket atom encoding (element + residue types)
│
├── datasets/
│   ├── hamdi_pockets_proper/    ← 14 pre-extracted viral/cancer pockets (Hamdi)
│   ├── hamdi_protiens/          ← 14 full protein structures (Hamdi)
│   ├── new_data_processed/      ← CD38 / BCMA pockets
│   ├── new_data/                ← CD38 / BCMA full structures
│   └── dude_pocket/             ← 104 DUD-E benchmark pockets
│
├── tests/
│   ├── conftest.py              ← Shared fixtures and PDB helpers
│   ├── fixtures/                ← Minimal synthetic PDB files for unit tests
│   ├── test_prepare_pockets.py  ← Pocket extraction tests (no GPU)
│   ├── test_pipeline_utils.py   ← Pipeline utility tests (no GPU)
│   ├── test_fragmol.py          ← Tokeniser / decoder tests (no GPU)
│   └── test_integration.py      ← End-to-end tests (GPU + checkpoints)
│
├── env/
│   └── rdkit.yml                ← Full conda environment specification
│
├── pytest.ini                   ← Test configuration
└── CLAUDE.md                    ← Developer reference for Claude Code
```

---

## How It Works

### Two-model architecture

**Step 1 — Contact prediction** (`model/transformer_v1_res_mp1.py`)

The pocket is encoded as a set of atoms with element types, residue identities, and 3D coordinates (discretized to a 240 × 240 × 240 voxel grid at 0.1 Å resolution). A transformer encoder predicts, for each residue, the probability that it forms a non-covalent interaction (NCI) with the ligand. The top-K highest-probability residues become *anchors* that bias molecule placement.

**Step 2 — Fragment-based generation** (`model/transformer_v1_res_fac2.py`)

An encoder–decoder transformer reads the pocket encoding and autoregressively generates a molecule one *fragment* at a time. Each step produces:
- A sequence of SMILES-like tokens (atom types, bonds, ring numbers) for the fragment
- 3D coordinates for each heavy atom, predicted independently per axis

Fragments are assembled using a recursive tree search:
1. Sample `fragment_batch_size` candidate fragments in parallel
2. Filter out fragments that clash with the pocket (`clash_distance` threshold)
3. Score survivors by model confidence + proximity to anchor residues
4. Keep the top 20% and recurse to grow the next fragment
5. Stop when an end-of-molecule token is generated or the time limit is reached

**Coordinate system**

All positions are integers on the 240³ voxel grid (24 Å box, 0.1 Å resolution, centered on the pocket). Real-world Ångström coordinates are recovered only at save time.

### Key design choices

- **Fragment-by-fragment generation** allows the model to produce molecules of variable length without fixing the number of atoms upfront.
- **Guided sampling** via NCI anchors ensures generated molecules grow toward residues known to be pharmacophore-relevant, improving binding pose quality.
- **Voxel clash check** eliminates any molecule whose atoms overlap with protein heavy atoms, providing a hard geometric filter without requiring docking.

---

## Advanced Usage

### Running on a specific GPU

```bash
python pipeline.py --input pockets/ --gpu 1 --name gpu1_run
```

Or set it permanently in `config.yaml`:
```yaml
model:
  gpu: "1"
```

### Generating more / fewer molecules

```bash
# Quick test — 5 molecules, 30 minutes max
python pipeline.py --input pockets/ --name quick_test
# (edit config.yaml: num_molecules: 5, max_hours: 0.5)
```

### Re-running pocket extraction with different parameters

```bash
python prepare_pockets.py \
    --input  datasets/hamdi_protiens/ \
    --output datasets/hamdi_pockets_10A/ \
    --distance 10.0 \
    --max-atoms 600
```

### Direct inference (no pipeline wrapper)

For full control over every parameter:

```bash
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

python inference/inference_avoid_clash.py \
  --cuda           0 \
  --cuda_list      0 \
  --input_list     my_input_list.txt \
  --savedir        my_output/ \
  --contact_path   checkpoint/contact.pkl \
  --caption_path   checkpoint/gen_mol.pkl \
  --gennums        20 \
  --max_run_hours  3 \
  --gen_frag_set   8 \
  --frag_len_add   20 \
  --nci_thrs       0.7 \
  --topk           5 \
  --coc_dis        2.5 \
  --tempture       1.0
```

**Input list format** — one protein per line, comma-separated:
```
ligand.sdf,nci_contacts.npy,pocket.pdb
```
Leave the first two fields empty if you have no pre-computed data:
```
,,datasets/hamdi_pockets_proper/2FOM_Dengue_pocket.pdb
,,datasets/hamdi_pockets_proper/6XHM_Corona_pocket.pdb
```

### All inference parameters

| Flag | Default | Description |
|------|---------|-------------|
| `--cuda` | — | GPU index |
| `--gennums` | — | Stop after this many valid molecules |
| `--max_run_hours` | — | Hard time limit per protein |
| `--gen_frag_set` | `1` | Batch size for fragment sampling |
| `--frag_len_add` | `0` | Minimum fragment token length |
| `--nci_thrs` | `0.7` | NCI contact probability threshold |
| `--topk` | `5` | Number of anchor residues |
| `--coc_dis` | `2.5` | Clash distance filter (Å) |
| `--tempture` | `1.0` | Softmax temperature |
| `--OnceMolGen` | off | Generate whole molecule in one pass (no tree) |
| `--prod_time` | `1` | Repeat `go_factory()` up to this many times per fragment level |

---

## Testing

The test suite verifies pocket extraction, pipeline utilities, and the molecule tokeniser without requiring a GPU or model weights. Integration tests (which do require a GPU and the downloaded checkpoints) are skipped automatically when those resources are absent.

### Install pytest

```bash
pip install pytest
```

### Run all fast tests (no GPU needed, ~3 seconds)

```bash
pytest tests/
```

### Run with verbose output

```bash
pytest tests/ -v
```

### Run integration tests (GPU + checkpoints required)

```bash
pytest tests/test_integration.py -v
```

### Test coverage

| File | Tests | What is verified |
|------|------:|-----------------|
| `test_prepare_pockets.py` | 26 | `extract_pocket` atom counts at exact/near/beyond cutoff; `max_atoms` keeps closest; HETATM not written to output; water HETATM skipped as ligand; no-ligand centroid fallback; missing file returns `None`; nested output dirs created; batch directory mode |
| `test_pipeline_utils.py` | 27 | `_deep_update` nested merge; `load_config` defaults, partial YAML override, DEFAULTS never mutated; `find_pdb_files` single file / directory / empty / missing; `write_input_list` double-comma format; `collect_results` flattens GPU sub-folder, merges if destination exists |
| `test_fragmol.py` | 19 | Vocabulary roundtrip; no duplicate indices; special-token positions (pad=0, start=1, end=2, sep=3); `ele_token` maps to heavy-atom symbols only; `decode3d` output shape and type; end-token terminates decoding; all-pad batch yields no molecules |
| `test_integration.py` | 5 | Full pipeline with real Hamdi pocket (GPU); `--extract-pockets` mode (GPU); real Hamdi proteins pocket extraction (no GPU); DUD-E PDB coordinate parsing (no GPU) |

### Test fixtures

The synthetic PDB files in `tests/fixtures/` are designed around a known geometry:

```
Ligand centre (HETATM LIG): (0.0, 0.0, 0.0)

ATOM   dist   included at cutoff=8.0?
  1    3.0 Å  yes
  2    5.0 Å  yes
  3    8.0 Å  yes  (exactly on boundary — tests ≤ behaviour)
  4   10.0 Å  no
  5   15.0 Å  no
  6    2.0 Å  yes  (closest atom)
  7    2.5 Å  yes
  8    9.0 Å  no
HETATM HOH     —   not treated as ligand
```

This makes expected atom counts deterministic for every cutoff value tested.

---

## License

Lingo3DMol is free software released under the **GNU General Public License v3.0**.
You may redistribute and/or modify it under the terms of the GPL as published by
the Free Software Foundation. See [COPYING.txt](COPYING.txt) for the full license text.
