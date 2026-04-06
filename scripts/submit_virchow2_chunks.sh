#!/bin/bash
# Iterate over all chunk CSVs and submit trident_chunk_extract_virchow2.pbs for each.
# HF_TOKEN is read inside the PBS script from $MIL/.hf_token (not passed via qsub).
#
# Usage (from MIL root on cluster):
#   cd /storage/brno2/home/filipsec/MIL
#   bash scripts/submit_virchow2_chunks.sh

set -euo pipefail

MIL="${MIL:-/storage/brno2/home/filipsec/MIL}"
CHUNKS_DIR="${CHUNKS_DIR:-$MIL/data/lists/chunks_500}"
PBS_SCRIPT="$MIL/scripts/trident_chunk_extract_virchow2.pbs"
DRY_RUN="${DRY_RUN:-0}"

cd "$MIL"

shopt -s nullglob
chunks=("$CHUNKS_DIR"/panda_chunk_*.csv)
if [[ ${#chunks[@]} -eq 0 ]]; then
  echo "No panda_chunk_*.csv in $CHUNKS_DIR" >&2
  exit 1
fi

echo "Submitting ${#chunks[@]} Virchow2 jobs"
for CHUNK_CSV in "${chunks[@]}"; do
  CHUNK_CSV="$MIL/data/lists/chunks_500/$(basename "$CHUNK_CSV")"
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "qsub -v CHUNK_CSV=$CHUNK_CSV $PBS_SCRIPT"
  else
    qsub -v "CHUNK_CSV=$CHUNK_CSV" "$PBS_SCRIPT"
    echo "  submitted $(basename "$CHUNK_CSV")"
  fi
done
echo "Done."
