#!/bin/bash
# Check makeup job output: logs and .h5 in trident_out_makeup_otsu.
# Usage: ./scripts/check_makeup_output.sh [JOB_ID]
# Example: ./scripts/check_makeup_output.sh 18034941

set -uo pipefail

MIL="${MIL:-/storage/brno2/home/filipsec/MIL}"
JOB_ID="${1:-}"

echo "=== Logs (stdout/stderr) ==="
if [[ -n "$JOB_ID" ]]; then
  for f in "$MIL/logs/trident_makeup_${JOB_ID}.o" "$MIL/logs/trident_makeup_${JOB_ID}.e"; do
    if [[ -f "$f" ]]; then
      echo "--- $f ---"
      cat "$f"
      echo ""
    else
      echo "Missing: $f"
      # try home logs if submitted with relative path
      if [[ "$f" == *"MIL/logs"* ]]; then
        alt="$HOME/logs/trident_makeup_${JOB_ID}.${f##*.}"
        if [[ -f "$alt" ]]; then
          echo "Found in \$HOME: $alt"
          cat "$alt"
        fi
      fi
    fi
  done
else
  echo "Latest trident_makeup log files in $MIL/logs:"
  ls -lt "$MIL/logs"/trident_makeup_*.o 2>/dev/null | head -5
  echo "Pass JOB_ID as first argument to show content."
fi

echo ""
echo "=== Makeup output dir (features_uni_v2) ==="
FEAT_DIR="$MIL/data/trident_out_makeup_otsu/20x_256px_0px_overlap/features_uni_v2"
if [[ -d "$FEAT_DIR" ]]; then
  echo "Path: $FEAT_DIR"
  echo "Count: $(ls -1 "$FEAT_DIR"/*.h5 2>/dev/null | wc -l) .h5 files"
  ls -la "$FEAT_DIR" 2>/dev/null | head -30
else
  echo "Dir does not exist: $FEAT_DIR"
fi
