# TRIDENT working config (frozen) and next steps

**Milestone:** 3-slide PANDA pipeline proven end-to-end on MetaCentrum: **seg → coords → feat** with real `.h5` feature files. Do not mutate this recipe without reason.

---

## 1. Working configuration (do not change casually)

| Item | Value |
|------|--------|
| **Repo** | `/storage/brno2/home/filipsec/repos/trident` |
| **Conda env** | `trident` (path: `/storage/brno2/home/filipsec/.conda/envs/trident`) |
| **WSI dir** | `/storage/brno2/home/filipsec/MIL/data/raw/train_images` |
| **Job dir (smoke test)** | `/storage/brno2/home/filipsec/MIL/data/trident_out/test_seg_grandqc` |
| **Slide list (test)** | `/storage/brno2/home/filipsec/MIL/data/test3_wsis.csv` |
| **Segmenter** | `grandqc` |
| **Patch encoder** | `uni_v2` |
| **Magnification** | `20` |
| **Patch size** | `256` |
| **Overlap** | `0` |
| **Max workers** | `4` |
| **GPU** | Required for `seg` and `feat` (use `--gpu 0` on 1-GPU job) |
| **Output format** | `.h5` (features under `20x_256px_0px_overlap/features_uni_v2/`) |

**Missing slide (chunks 000+001):** `163fabe883fcd17de4f899ca8ede45a8`. Cause: grandqc reported "No contour were detected" (QC false negative; slide has tissue). Fix: rerun **one slide** with a fallback segmenter (e.g. `hest`) into a **separate** `job_dir`, then copy the `.h5` into production features.

- Singleton list: `data/lists/singletons/163fabe883fcd17de4f899ca8ede45a8.csv` (column `wsi`, value `163fabe883fcd17de4f899ca8ede45a8.tiff`).
- Easiest path: copy `scripts/trident_chunk_extract.pbs` (or run interactively on a GPU node), set `JOB_DIR` to a new directory (e.g. `.../trident_out_single/163fabe883fcd17de4f899ca8ede45a8_hest`), set `CHUNK_CSV` to that singleton CSV, and in the `run_batch_of_slides.py` invocation replace `--segmenter grandqc` with `--segmenter hest` (matching the production patch encoder / mag / patch size / overlap).

**After job succeeds**, copy the `.h5` into production features (adjust paths to match your `JOB_DIR`):
```bash
FEAT_MAIN="/storage/brno2/home/filipsec/MIL/data/trident_out/panda_uni_v2_grandqc_20x_256_ov0/20x_256px_0px_overlap/features_uni_v2"
cp "$JOB_DIR/20x_256px_0px_overlap/features_uni_v2/163fabe883fcd17de4f899ca8ede45a8.h5" "$FEAT_MAIN/"
```

**HF token:** Required for UNI2 (gated repo). Run `login(token='...')` in the same session before `run_batch_of_slides.py` on any node (including GPU nodes).

---

## 1b. Missing slides and logs

For slides that never produced features, grep your existing TRIDENT job logs under `logs/` for the slide id or errors like `No contour`. For systematic gap checks vs `train.csv`, use `python scripts/find_missing_trident_features.py --features-dir ...`.

---

## 2. Smoke test vs production

- **Current path** `.../trident_out/test_seg_grandqc` is for **testing only**. Do not use it for production.
- **Production:** use a dedicated root with a clear name, e.g.  
  `panda_uni_v2_grandqc_20x_256_ov0` (dataset + encoder + segmenter + mag + patch + overlap).
- Avoid names like `final`, `final2`, `real_final`, `fixed_run_new`.

**Production root example:**
```bash
mkdir -p /storage/brno2/home/filipsec/MIL/data/trident_out/panda_uni_v2_grandqc_20x_256_ov0
```

---

## 3. Next deliverable: folds (before full extraction)

Create the split file **before** full-dataset extraction so the split is fixed and reproducible.

**Create splits dir:**
```bash
mkdir -p /storage/brno2/home/filipsec/MIL/data/splits
```

**5-fold stratified by `isup_grade`:**
```python
import pandas as pd
from sklearn.model_selection import StratifiedKFold

csv_path = '/storage/brno2/home/filipsec/MIL/data/raw/train.csv'
out_path = '/storage/brno2/home/filipsec/MIL/data/splits/panda_5fold_stratified.csv'

df = pd.read_csv(csv_path).copy()
df['fold'] = -1

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for fold, (_, val_idx) in enumerate(skf.split(df, df['isup_grade'])):
    df.loc[val_idx, 'fold'] = fold

df.to_csv(out_path, index=False)

print('saved:', out_path)
print(df['fold'].value_counts().sort_index())
print(pd.crosstab(df['fold'], df['isup_grade']))
```

**Or use the project script** (from repo root):
```bash
cd /storage/brno2/home/filipsec/MIL
python scripts/make_panda_5fold_splits.py
# Or with custom paths:
python scripts/make_panda_5fold_splits.py --csv data/raw/train.csv --out data/splits/panda_5fold_stratified.csv
```

**Inline one-off** (if you prefer):
```bash
cd /storage/brno2/home/filipsec/MIL
python -c "
import pandas as pd
from sklearn.model_selection import StratifiedKFold

csv_path = '/storage/brno2/home/filipsec/MIL/data/raw/train.csv'
out_path = '/storage/brno2/home/filipsec/MIL/data/splits/panda_5fold_stratified.csv'

df = pd.read_csv(csv_path).copy()
df['fold'] = -1

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for fold, (_, val_idx) in enumerate(skf.split(df, df['isup_grade'])):
    df.loc[val_idx, 'fold'] = fold

df.to_csv(out_path, index=False)

print('saved:', out_path)
print(df['fold'].value_counts().sort_index())
print(pd.crosstab(df['fold'], df['isup_grade']))
"
```

---

## 4. Full-dataset extraction (after folds)

- **Strategy A:** One large run (simpler, but fragile if interrupted).
- **Strategy B (recommended):** Chunked slide lists; run extraction chunk by chunk; merge under one production root. Easier to resume on a cluster.

Planned: PBS script for production extraction (chunked, resumable), using the production root and full slide list derived from `train.csv` / folds.

## 4b. Shared-coords encoder sweep (H-optimus / GigaPath)

For encoder comparisons where you want the **same patch set per slide**, do **not** re-tile the WSI for each encoder by default.

Use the feature-only workflow:

- Existing coords source: `20x_256px_0px_overlap/patches`
- Script: [scripts/trident_chunk_extract_feat_only.pbs](/Users/fs/Desktop/MIL/scripts/trident_chunk_extract_feat_only.pbs)
- Supported encoder keys: `hoptimus0`, `hoptimus1`, `gigapath`
- Standard outputs:
  - `features_hoptimus0`
  - `features_hoptimus1`
  - `features_gigapath`

Example:

```bash
cd /storage/brno2/home/filipsec/MIL
qsub -v "CHUNK_CSV=/abs/panda_chunk_000.csv,ENCODER=hoptimus1,COORDS_DIR=20x_256px_0px_overlap" \
  scripts/trident_chunk_extract_feat_only.pbs
```

The script now resolves encoder metadata from the shared registry, looks up encoder-specific token files in the repo root, falls back to `.hf_token`, then to `HF_TOKEN`, and fails before the TRIDENT run if Hugging Face access is not valid.

---

## 5. After extraction: MIL training

- Use extracted bags (e.g. `.h5` or exported `.pt`) and the fold CSV.
- Train MIL (e.g. attention-based) with stratified train/val per fold; report validation AUC; early stopping; save best model by val AUC.
- If you set `ENCODER=...` in `scripts/mil_training.pbs`, `FEATURE_DIM` is derived from the registry and checkpoints store `encoder_name`.

---

## 6. What was actually achieved

- **seg** → tissue segmentation (grandqc) → thumbnails, contours, **contours_geojson**
- **coords** → patch coordinates → `*_patches.h5`
- **feat** → uni_v2 features → `features_uni_v2/*.h5`

The front-end of the real MIL pipeline is working. Freeze this recipe; then move to production folds, production output root, full extraction, and MIL training.
