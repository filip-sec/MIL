#!/bin/bash
# Sync modified MIL files to cluster (storage.brno2).
# Usage: ./scripts/sync_to_cluster.sh
# Set REMOTE_USER / REMOTE_HOST if needed.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MIL="$(dirname "$SCRIPT_DIR")"
REMOTE_USER="${REMOTE_USER:-filipsec}"
REMOTE_HOST="${REMOTE_HOST:-skirit.metacentrum.cz}"
REMOTE="${REMOTE_USER}@${REMOTE_HOST}:/storage/brno2/home/${REMOTE_USER}/MIL"

FILES=(
    mil/train.py
    mil/data.py
    mil/model.py
    mil/utils.py
    mil/__init__.py
    mil/loss.py
    mil/attention_map.py
    mil/eval_helpers.py
    mil/vit_attention_viz.py
    mil/wsi_metadata.py
    pyproject.toml
    scripts/run_training.py
    scripts/run_mil_training.py
    scripts/extract_attention.py
    scripts/plot_attention_map.py
    scripts/eval_cv_confusion.py
    scripts/make_submission.py
    scripts/plot_spatial_mil_figures.py
    scripts/mil_training.pbs
    scripts/inspect_checkpoint.py
    scripts/plot_attention_figures.py
    scripts/check_h5_coords.py
    scripts/print_torch_graph_nodes.py
    scripts/extract_encoder_intermediates.py
    scripts/plot_vit_attention_heads.py
    scripts/print_wsi_geometry.py
    docs/WSI_GEOMETRY.md
    scripts/sync_to_cluster.sh
)

echo "Syncing from $MIL to $REMOTE"
for f in "${FILES[@]}"; do
    if [[ -f "$MIL/$f" ]]; then
        rsync -avz --progress "$MIL/$f" "$REMOTE/$f"
    fi
done
echo "Done"
