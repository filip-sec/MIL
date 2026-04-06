#!/bin/bash
# Submit one PBS job per chunk for Virchow2 feature extraction.
# Uses existing chunk CSVs in data/lists/chunks_500/panda_chunk_*.csv (no need to create them).
#
# Usage (from MIL project root on cluster):
#   cd /storage/brno2/home/filipsec/MIL
#   export HF_TOKEN=$(cat .hf_token)
#   bash scripts/run_all_chunks_virchow2.sh
#
# Optional:
#   CHUNKS_DIR=data/lists/chunks_500 bash scripts/run_all_chunks_virchow2.sh
#   DRY_RUN=1 bash scripts/run_all_chunks_virchow2.sh   # print qsub commands, do not submit
#
# For UNI2 (non-Virchow2) batches instead:
#   PBS_SCRIPT=$MIL/scripts/trident_chunk_extract.pbs bash scripts/run_all_chunks_virchow2.sh
#   (HF_TOKEN not required for UNI2)

set -euo pipefail

MIL="${MIL:-/storage/brno2/home/filipsec/MIL}"
CHUNKS_DIR="${CHUNKS_DIR:-$MIL/data/lists/chunks_500}"
PBS_SCRIPT="${PBS_SCRIPT:-$MIL/scripts/trident_chunk_extract_virchow2.pbs}"
DRY_RUN="${DRY_RUN:-0}"

if [[ ! -d "$CHUNKS_DIR" ]]; then
  echo "Chunks dir not found: $CHUNKS_DIR"
  echo "Create chunk CSVs first: python scripts/make_chunk_lists.py --csv data/raw/train.csv --out-dir $CHUNKS_DIR"
  exit 1
fi

# HF_TOKEN is read inside trident_chunk_extract_virchow2.pbs from $MIL/.hf_token (not passed via qsub)
shopt -s nullglob
chunks=("$CHUNKS_DIR"/panda_chunk_*.csv)
if [[ ${#chunks[@]} -eq 0 ]]; then
  echo "No panda_chunk_*.csv found in $CHUNKS_DIR"
  exit 1
fi

echo "Submitting ${#chunks[@]} jobs (Virchow2)"
for CHUNK_CSV in "${chunks[@]}"; do
  cmd=(qsub -v "CHUNK_CSV=$CHUNK_CSV" "$PBS_SCRIPT")
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "${cmd[*]}"
  else
    "${cmd[@]}"
  fi
done
echo "Done."
