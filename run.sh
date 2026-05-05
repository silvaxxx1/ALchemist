#!/bin/bash
# ============================================================
#  Lingo3DMol — Quick Start
#
#  Run this script from the Lingo3DMol directory:
#      sh run.sh
#
#  Hamdi: edit the INPUT line below to point to your folder of
#  pocket PDB files, then save and run.  Results appear in
#  results/<run_name>/ as .mol files you can open in PyMOL.
# ============================================================

export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# --- Option A: input is already-extracted pocket PDB files ---
# (use this if your PDB files are named *_pocket.pdb)
python pipeline.py \
  --input  datasets/hamdi_pockets_proper/ \
  --name   hamdi_run \
  --config config.yaml

# --- Option B: input is full protein PDB files ---
# Remove the # from the lines below and comment out Option A.
# The script will extract pockets automatically (8 Å radius).
#
# python pipeline.py \
#   --input          datasets/hamdi_protiens/ \
#   --extract-pockets \
#   --name           hamdi_run \
#   --config         config.yaml

# --- Option C: DUD-E benchmark set ---
#
# python pipeline.py \
#   --input  datasets/dude_pocket/ \
#   --name   dude_run \
#   --config config.yaml
