#!/usr/bin/env python3
"""
pipeline.py — Lingo3DMol Full Automation Pipeline
==================================================
Single entry point for the entire workflow: from raw protein PDB
files to generated 3D molecules.

Quick start (Hamdi)
-------------------
If your input files are already extracted pocket PDB files:
    python pipeline.py --input datasets/hamdi_pockets_proper/

If your input files are full protein PDB files:
    python pipeline.py --input datasets/hamdi_protiens/ --extract-pockets

Single protein:
    python pipeline.py --input my_protein.pdb --extract-pockets

Custom run name (so results don't get mixed):
    python pipeline.py --input pockets/ --name dengue_run

Custom config:
    python pipeline.py --input pockets/ --config config.yaml

Output
------
All results are saved under results/<run_name>/.
Each protein gets its own sub-folder containing .mol files.

Generated .mol files can be opened in PyMOL, ChimeraX, or Avogadro.
"""
from __future__ import annotations

import os
import sys
import glob
import shutil
import argparse
import subprocess
from pathlib import Path
from datetime import datetime

# ── default parameters (overridden by config.yaml then CLI) ───────────────────

DEFAULTS = {
    'generation': {
        'num_molecules': 20,
        'max_hours': 3,
        'fragment_batch_size': 8,
    },
    'model': {
        'contact_checkpoint': 'checkpoint/contact.pkl',
        'generation_checkpoint': 'checkpoint/gen_mol.pkl',
        'gpu': '0',
    },
    'pocket': {
        'distance_cutoff': 8.0,
        'max_atoms': 450,
        'clash_distance': 2.5,
    },
    'output': {
        'directory': 'results/',
    },
    'advanced': {
        'nci_threshold': 0.7,
        'top_k_anchors': 5,
        'fragment_min_length': 20,
        'temperature': 1.0,
    },
}


# ── config loading ─────────────────────────────────────────────────────────────

def _deep_update(base: dict, update: dict) -> dict:
    for k, v in update.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
    return base


def load_config(config_path: str) -> dict:
    import copy
    cfg = copy.deepcopy(DEFAULTS)
    if not os.path.exists(config_path):
        return cfg
    try:
        import yaml
        with open(config_path) as f:
            user = yaml.safe_load(f)
    except ImportError:
        print('NOTE: PyYAML not installed — using default config values.')
        print('      Install with:  pip install pyyaml\n')
        return cfg
    if user:
        _deep_update(cfg, user)
    return cfg


# ── helpers ────────────────────────────────────────────────────────────────────

def find_pdb_files(path: str) -> list[str]:
    p = Path(path)
    if p.is_file() and p.suffix.lower() == '.pdb':
        return [str(p)]
    files = sorted(glob.glob(str(p / '*.pdb')))
    if not files:
        # also search one level deeper
        files = sorted(glob.glob(str(p / '*' / '*.pdb')))
    return files


def write_input_list(pocket_files: list[str], list_path: str):
    with open(list_path, 'w') as f:
        for pdb in pocket_files:
            f.write(f',,{pdb}\n')


def run_inference(input_list: str, raw_savedir: str, cfg: dict):
    gpu = str(cfg['model']['gpu'])
    env = os.environ.copy()
    env['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

    cmd = [
        sys.executable,
        'inference/inference_avoid_clash.py',
        '--cuda', gpu,
        '--cuda_list', gpu,
        '--input_list', input_list,
        '--savedir', raw_savedir + '/',
        '--contact_path', cfg['model']['contact_checkpoint'],
        '--caption_path', cfg['model']['generation_checkpoint'],
        '--gennums', str(cfg['generation']['num_molecules']),
        '--max_run_hours', str(cfg['generation']['max_hours']),
        '--gen_frag_set', str(cfg['generation']['fragment_batch_size']),
        '--frag_len_add', str(cfg['advanced']['fragment_min_length']),
        '--nci_thrs', str(cfg['advanced']['nci_threshold']),
        '--topk', str(cfg['advanced']['top_k_anchors']),
        '--tempture', str(cfg['advanced']['temperature']),
        '--coc_dis', str(cfg['pocket']['clash_distance']),
    ]

    print('Running inference with command:')
    print('  ' + ' '.join(cmd) + '\n')
    subprocess.run(cmd, check=True, env=env)


def collect_results(raw_dir: str, final_dir: str, gpu: str):
    """
    inference_avoid_clash.py appends the GPU id to savedir, so results land in
    raw_dir/{gpu}/{pocket_name}/*.mol — move them up one level into final_dir.
    """
    gpu_subdir = os.path.join(raw_dir, str(gpu))
    if not os.path.isdir(gpu_subdir):
        return
    for item in os.listdir(gpu_subdir):
        src = os.path.join(gpu_subdir, item)
        dst = os.path.join(final_dir, item)
        if os.path.isdir(src):
            if os.path.isdir(dst):
                # merge if destination already exists
                for mol in os.listdir(src):
                    shutil.move(os.path.join(src, mol), os.path.join(dst, mol))
                os.rmdir(src)
            else:
                shutil.move(src, dst)
        else:
            shutil.move(src, dst)
    try:
        os.rmdir(gpu_subdir)
    except OSError:
        pass
    try:
        os.rmdir(raw_dir)
    except OSError:
        pass


def print_summary(results_dir: str, pocket_files: list[str]):
    mol_files = list(Path(results_dir).rglob('*.mol'))
    print(f'\n{"="*60}')
    print(f'  Done!')
    print(f'  Proteins processed : {len(pocket_files)}')
    print(f'  Molecules generated: {len(mol_files)}')
    print(f'  Results saved to   : {results_dir}/')
    print(f'{"="*60}')
    if mol_files:
        print('\n  Per-protein breakdown:')
        for subdir in sorted(Path(results_dir).iterdir()):
            if subdir.is_dir():
                count = len(list(subdir.glob('*.mol')))
                if count:
                    print(f'    {subdir.name}: {count} molecule(s)')
    print()


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Lingo3DMol: pocket-based 3D molecule generation pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        '--input', required=True,
        help='Input PDB file, or directory containing PDB files',
    )
    parser.add_argument(
        '--extract-pockets', action='store_true',
        help='Extract binding pockets from full protein PDB files first. '
             'Use this when --input contains full protein structures (not pocket files).',
    )
    parser.add_argument(
        '--config', default='config.yaml',
        help='Path to YAML config file (default: config.yaml)',
    )
    parser.add_argument(
        '--output',
        help='Base output directory (overrides config.yaml)',
    )
    parser.add_argument(
        '--name',
        help='Run name used for the output sub-folder '
             '(default: current timestamp)',
    )
    parser.add_argument(
        '--gpu', type=str,
        help='GPU index to use, e.g. "0" (overrides config.yaml)',
    )
    args = parser.parse_args()

    # --- load and override config ---
    cfg = load_config(args.config)
    if args.output:
        cfg['output']['directory'] = args.output
    if args.gpu:
        cfg['model']['gpu'] = args.gpu

    run_name = args.name or datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = os.path.join(cfg['output']['directory'], run_name)
    os.makedirs(results_dir, exist_ok=True)

    # --- banner ---
    print(f'\n{"="*60}')
    print(f'  Lingo3DMol — 3D Molecule Generation Pipeline')
    print(f'{"="*60}')
    print(f'  Run name  : {run_name}')
    print(f'  Input     : {args.input}')
    print(f'  Mode      : {"full protein → extract pockets → generate" if args.extract_pockets else "pocket files → generate directly"}')
    print(f'  Output    : {results_dir}/')
    print(f'{"="*60}\n')

    # --- check checkpoints ---
    for ckpt_key, label in [('contact_checkpoint', 'contact'),
                             ('generation_checkpoint', 'generation')]:
        ckpt = cfg['model'][ckpt_key]
        if not os.path.exists(ckpt):
            print(f'ERROR: {label} checkpoint not found at "{ckpt}"')
            print('       Download checkpoints per the instructions in readme.md')
            sys.exit(1)

    # --- find input PDB files ---
    input_files = find_pdb_files(args.input)
    if not input_files:
        print(f'ERROR: No PDB files found at "{args.input}"')
        sys.exit(1)
    print(f'Found {len(input_files)} PDB file(s)\n')

    # --- pocket extraction (optional) ---
    if args.extract_pockets:
        print('Step 1/2  Extract binding pockets')
        print('-' * 40)
        from prepare_pockets import process_directory, extract_pocket

        pockets_dir = os.path.join(results_dir, 'pockets')
        if Path(args.input).is_dir():
            pocket_files = process_directory(
                args.input, pockets_dir,
                distance_cutoff=cfg['pocket']['distance_cutoff'],
                max_atoms=cfg['pocket']['max_atoms'],
            )
        else:
            stem = Path(args.input).stem
            out = os.path.join(pockets_dir, f'{stem}_pocket.pdb')
            count = extract_pocket(
                args.input, out,
                distance_cutoff=cfg['pocket']['distance_cutoff'],
                max_atoms=cfg['pocket']['max_atoms'],
            )
            pocket_files = [out] if (count and count > 0) else []

        if not pocket_files:
            print('\nERROR: No valid pockets were extracted.')
            print('       Make sure your PDB files contain HETATM records for the ligand,')
            print('       or check the --distance value in config.yaml.')
            sys.exit(1)

        print(f'\nExtracted {len(pocket_files)} pocket(s)\n')
        step_prefix = 'Step 2/2'
    else:
        pocket_files = input_files
        step_prefix = 'Step 1/1'

    # --- write input list for the inference script ---
    input_list = os.path.join(results_dir, 'input_list.txt')
    write_input_list(pocket_files, input_list)

    # --- run inference ---
    print(f'{step_prefix}  Generate molecules')
    print('-' * 40)
    raw_dir = os.path.join(results_dir, '_raw')
    run_inference(input_list, raw_dir, cfg)

    # --- tidy output (remove GPU sub-folder) ---
    collect_results(raw_dir, results_dir, cfg['model']['gpu'])

    # --- summary ---
    print_summary(results_dir, pocket_files)


if __name__ == '__main__':
    main()
