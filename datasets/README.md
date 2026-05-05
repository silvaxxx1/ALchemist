# Datasets

PDB files are not tracked in git. Place your files in the folders below before running the pipeline.

## Folder structure

```
datasets/
├── hamdi_pockets_proper/    ← Pre-extracted pocket PDB files (*_pocket.pdb)
├── hamdi_protiens/          ← Full protein PDB files (raw, before pocket extraction)
├── new_data/                ← Additional full protein structures
├── new_data_processed/      ← Pockets extracted from new_data/
└── dude_pocket/             ← DUD-E benchmark pockets (unzip dude_pocket.zip)
```

## Quick start

If you have full protein PDB files, drop them into `hamdi_protiens/` and run:

```bash
python pipeline.py --input datasets/hamdi_protiens/ --extract-pockets --name my_run
```

If you already have extracted pocket PDB files, drop them into `hamdi_pockets_proper/` and run:

```bash
python pipeline.py --input datasets/hamdi_pockets_proper/ --name my_run
```

## DUD-E benchmark

Download and unzip the DUD-E pocket archive into `dude_pocket/`:

```bash
cd datasets && unzip dude_pocket.zip && cd ..
python pipeline.py --input datasets/dude_pocket/ --name dude_run
```
