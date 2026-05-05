#!/usr/bin/env python3
"""
prepare_pockets.py
==================
Extract binding-pocket atoms from full protein PDB files.

For each protein the script finds HETATM ligand atoms, then keeps
all ATOM (protein) records within --distance Angstroms of those
ligands.  If no ligand is present it falls back to the geometric
center of the protein.

Usage
-----
# Process a whole directory:
    python prepare_pockets.py --input datasets/hamdi_protiens/ --output datasets/hamdi_pockets/

# Process a single file:
    python prepare_pockets.py --input my_protein.pdb --output my_pocket.pdb

# Adjust pocket size:
    python prepare_pockets.py --input proteins/ --output pockets/ --distance 10.0 --max-atoms 500
"""
from __future__ import annotations

import os
import argparse
import glob
import numpy as np
from pathlib import Path

# Residue names to skip when looking for ligands
_SOLVENT = {'HOH', 'WAT', 'SO4', 'PO4', 'CL', 'NA', 'MG', 'CA', 'K', 'ZN',
            'MN', 'FE', 'CU', 'CO', 'NI', 'GOL', 'EDO', 'PEG', 'BME'}


def extract_pocket(input_pdb: str, output_pdb: str,
                   distance_cutoff: float = 8.0,
                   max_atoms: int = 450) -> int | None:
    """
    Extract pocket atoms from *input_pdb* and write them to *output_pdb*.

    Parameters
    ----------
    input_pdb       : path to the full protein PDB file
    output_pdb      : where to write the pocket PDB
    distance_cutoff : Angstrom radius around ligand atoms
    max_atoms       : keep at most this many protein atoms (closest first)

    Returns
    -------
    Number of atoms written, or None on failure.
    """
    try:
        with open(input_pdb) as f:
            lines = f.readlines()
    except OSError as e:
        print(f'  ERROR reading {input_pdb}: {e}')
        return None

    # --- collect ligand coordinates ---
    ligand_coords = []
    cryst1_line = None
    for line in lines:
        if line.startswith('CRYST1'):
            cryst1_line = line
        if line.startswith('HETATM'):
            res_name = line[17:20].strip()
            if res_name in _SOLVENT:
                continue
            try:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                ligand_coords.append([x, y, z])
            except ValueError:
                continue

    if not ligand_coords:
        # fall back to geometric center of protein backbone
        backbone = []
        for line in lines:
            if line.startswith('ATOM'):
                try:
                    backbone.append([float(line[30:38]),
                                     float(line[38:46]),
                                     float(line[46:54])])
                except ValueError:
                    continue
        if not backbone:
            print(f'  ERROR: no coordinates found in {input_pdb}')
            return None
        ligand_coords = [np.mean(backbone, axis=0).tolist()]
        print(f'  No ligand found — using protein centroid as pocket center')

    ligand_arr = np.array(ligand_coords)  # (L, 3)

    # --- collect protein atoms ---
    pocket_records = []
    for line in lines:
        if not line.startswith('ATOM'):
            continue
        try:
            xyz = np.array([float(line[30:38]),
                             float(line[38:46]),
                             float(line[46:54])])
        except ValueError:
            continue
        dists = np.linalg.norm(ligand_arr - xyz, axis=1)
        min_dist = dists.min()
        if min_dist <= distance_cutoff:
            pocket_records.append((min_dist, line))

    if not pocket_records:
        print(f'  WARNING: no protein atoms within {distance_cutoff} Å — skipping')
        return 0

    # sort by distance, keep closest max_atoms
    pocket_records.sort(key=lambda x: x[0])
    if len(pocket_records) > max_atoms:
        pocket_records = pocket_records[:max_atoms]

    os.makedirs(os.path.dirname(os.path.abspath(output_pdb)), exist_ok=True)
    with open(output_pdb, 'w') as f:
        if cryst1_line:
            f.write(cryst1_line)
        for _, record in pocket_records:
            f.write(record)
        f.write('END\n')

    return len(pocket_records)


def process_directory(input_dir: str, output_dir: str,
                      distance_cutoff: float = 8.0,
                      max_atoms: int = 450) -> list[str]:
    """
    Extract pockets for every *.pdb in *input_dir*, saving to *output_dir*.
    Returns a list of successfully created pocket PDB paths.
    """
    pdb_files = sorted(glob.glob(os.path.join(input_dir, '*.pdb')))
    if not pdb_files:
        print(f'No PDB files found in {input_dir}')
        return []

    os.makedirs(output_dir, exist_ok=True)
    pocket_files = []

    print(f'Extracting pockets from {len(pdb_files)} file(s) in {input_dir}')
    print(f'  Distance cutoff : {distance_cutoff} Å')
    print(f'  Max atoms       : {max_atoms}')
    print()

    for pdb in pdb_files:
        stem = Path(pdb).stem
        out = os.path.join(output_dir, f'{stem}_pocket.pdb')
        count = extract_pocket(pdb, out,
                               distance_cutoff=distance_cutoff,
                               max_atoms=max_atoms)
        status = f'{count} atoms' if count else 'SKIPPED'
        print(f'  {Path(pdb).name:45s} -> {status}')
        if count and count > 0:
            pocket_files.append(out)

    print(f'\nDone: {len(pocket_files)}/{len(pdb_files)} pockets extracted')
    return pocket_files


def main():
    parser = argparse.ArgumentParser(
        description='Extract binding pockets from full protein PDB files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--input', required=True,
                        help='Input PDB file or directory of PDB files')
    parser.add_argument('--output', required=True,
                        help='Output pocket PDB file or directory')
    parser.add_argument('--distance', type=float, default=8.0,
                        help='Pocket radius in Angstroms (default: 8.0)')
    parser.add_argument('--max-atoms', type=int, default=450,
                        help='Maximum pocket atoms to keep (default: 450)')
    args = parser.parse_args()

    inp = Path(args.input)
    if inp.is_dir():
        process_directory(str(inp), args.output,
                          distance_cutoff=args.distance,
                          max_atoms=args.max_atoms)
    elif inp.is_file():
        count = extract_pocket(str(inp), args.output,
                               distance_cutoff=args.distance,
                               max_atoms=args.max_atoms)
        if count:
            print(f'Wrote {count} atoms to {args.output}')
    else:
        print(f'ERROR: {args.input} does not exist')


if __name__ == '__main__':
    main()
