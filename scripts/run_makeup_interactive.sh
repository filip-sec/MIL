#!/bin/bash
# Run makeup (21 slides, hest) interactively. Use from an interactive GPU session on a Brno node.
#
# 1) Start interactive job (Brno, GPU):
#    qsub -I -l select=1:ncpus=4:mem=64gb:ngpus=1:scratch_local=64gb:gpu_mem=10gb:brno=True -l walltime=2:00:00
#
# 2) In the interactive shell:
#    cd /storage/brno2/home/filipsec/MIL
#    export CHUNK_CSV=/storage/brno2/home/filipsec/MIL/data/lists/missing_21_slides.csv
#    export HF_TOKEN="$(cat /storage/brno2/home/filipsec/MIL/.hf_token)"  # gitignored; never paste tokens into scripts
#    bash scripts/run_makeup_interactive.sh
#
# 3) After it finishes, copy .h5 to production (see end of script).

set -uo pipefail

CHUNK_CSV="${CHUNK_CSV:?Set CHUNK_CSV}"
# On bee2 home may be /auto/brno2; prefer /storage/brno2 if it exists
BRNO2_HOME="/storage/brno2/home/filipsec"
[[ -d "$BRNO2_HOME" ]] || BRNO2_HOME="$HOME"
JOB_DIR="$BRNO2_HOME/MIL/data/trident_out_makeup_otsu"
WSI_DIR="$BRNO2_HOME/MIL/data/raw/train_images"
REPO_DIR="$BRNO2_HOME/repos/trident"

# Use scratch if we're in a PBS job, else a temp dir
WORK="${SCRATCHDIR:-$TMPDIR}"
if [[ -z "$WORK" || ! -d "$WORK" ]]; then
  WORK="/tmp/trident_makeup_$$"
  mkdir -p "$WORK"
fi
WSI_LOCAL="$WORK/wsis"
CACHE_DIR="$WORK/wsi_cache"
mkdir -p "$WSI_LOCAL" "$CACHE_DIR"

echo "Node: $(hostname)"
echo "Start: $(date)"
echo "CHUNK_CSV=$CHUNK_CSV"
echo "JOB_DIR=$JOB_DIR"
echo "Copying WSIs to $WSI_LOCAL ..."

awk -F',' 'NR>1 {print $1}' "$CHUNK_CSV" | while read -r name; do
  base="$name"
  [[ "$base" != *.tif && "$base" != *.tiff ]] && base="${base}.tiff"
  src="${WSI_DIR}/${base}"
  if [[ -f "$src" ]]; then
    cp "$src" "$WSI_LOCAL/"
  else
    echo "ERROR: missing WSI $src" >&2
    exit 1
  fi
done

echo "Copied $(ls -1 "$WSI_LOCAL" | wc -l) slides."

# Use trident env: prefer conda activate, else use env's python directly (bee2 has no .conda/etc/profile.d/conda.sh)
PYTRIDENT=""
for d in "$BRNO2_HOME" "$HOME" "/storage/brno2/home/filipsec" "/auto/brno2/home/filipsec"; do
  if [[ -x "$d/.conda/envs/trident/bin/python" ]]; then
    PYTRIDENT="$d/.conda/envs/trident/bin/python"
    break
  fi
done
if [[ -z "$PYTRIDENT" ]]; then
  CONDA_SH="${BRNO2_HOME}/.conda/etc/profile.d/conda.sh"
  [[ -f "$CONDA_SH" ]] || CONDA_SH="$HOME/.conda/etc/profile.d/conda.sh"
  if [[ -f "$CONDA_SH" ]]; then
    source "$CONDA_SH"
    conda activate trident
    PYTRIDENT="$(which python)"
  fi
fi
[[ -n "$PYTRIDENT" ]] || { echo "ERROR: trident env python not found" >&2; exit 1; }
export PATH="$("$PYTRIDENT" -c "import sys; print(sys.prefix)")/bin:$PATH"
echo "Using: $PYTRIDENT"

cd "$REPO_DIR"
if [[ -n "${HF_TOKEN:-}" ]]; then
  "$PYTRIDENT" -c "from huggingface_hub import login; login(token='$HF_TOKEN'); print('HF login OK')"
fi

nvidia-smi

"$PYTRIDENT" run_batch_of_slides.py \
  --task all \
  --wsi_dir "$WSI_LOCAL" \
  --job_dir "$JOB_DIR" \
  --custom_list_of_wsis "$CHUNK_CSV" \
  --wsi_ext .tiff \
  --segmenter hest \
  --patch_encoder uni_v2 \
  --mag 20 \
  --patch_size 256 \
  --overlap 0 \
  --gpu 0 \
  --max_workers 4 \
  --feat_batch_size 96 \
  --wsi_cache "$CACHE_DIR" \
  --cache_batch_size 8 \
  --remove_artifacts \
  --skip_errors

FEAT_OUT="$JOB_DIR/20x_256px_0px_overlap/features_uni_v2"
echo "Done: $(date)"
echo "Output .h5 count: $(ls -1 "$FEAT_OUT"/*.h5 2>/dev/null | wc -l)"
ls -la "$FEAT_OUT" 2>/dev/null || echo "No features dir or empty"

echo ""
echo "To copy to production:"
echo "  FEAT_MAIN=\"/storage/brno2/home/filipsec/MIL/data/trident_out/panda_uni_v2_grandqc_20x_256_ov0/20x_256px_0px_overlap/features_uni_v2\""
echo "  cp $FEAT_OUT/*.h5 \"\$FEAT_MAIN/\""
