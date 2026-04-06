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

**Missing slide (chunks 000+001):** `163fabe883fcd17de4f899ca8ede45a8`. Cause: grandqc reported "No contour were detected" (QC false negative; slide has tissue). Fix: **one-slide rerun without grandqc** into a separate job_dir, then copy the resulting `.h5` into production features.

- Singleton list: `data/lists/singletons/163fabe883fcd17de4f899ca8ede45a8.csv` (column `wsi`, value `163fabe883fcd17de4f899ca8ede45a8.tiff`).
- PBS script: `scripts/trident_single_extract_noqc.pbs` (no `--segmenter grandqc`; `JOB_DIR=.../trident_out_single/163fabe883fcd17de4f899ca8ede45a8_noqc`).

**Run on cluster:**
```bash
mkdir -p /storage/brno2/home/filipsec/MIL/data/lists/singletons
# CSV already in repo: data/lists/singletons/163fabe883fcd17de4f899ca8ede45a8.csv
# HF token in gitignored .hf_token (create on cluster if needed: echo 'your_token' > .hf_token)

qsub -v CHUNK_CSV=/storage/brno2/home/filipsec/MIL/data/lists/singletons/163fabe883fcd17de4f899ca8ede45a8.csv,HF_TOKEN="$(cat /storage/brno2/home/filipsec/MIL/.hf_token)" \
  /storage/brno2/home/filipsec/MIL/scripts/trident_single_extract_noqc.pbs
```

**After job succeeds**, copy the `.h5` into production features:
```bash
FEAT_MAIN="/storage/brno2/home/filipsec/MIL/data/trident_out/panda_uni_v2_grandqc_20x_256_ov0/20x_256px_0px_overlap/features_uni_v2"
cp /storage/brno2/home/filipsec/MIL/data/trident_out_single/163fabe883fcd17de4f899ca8ede45a8_noqc/20x_256px_0px_overlap/features_uni_v2/163fabe883fcd17de4f899ca8ede45a8.h5 "$FEAT_MAIN/"
```

**HF token:** Required for UNI2 (gated repo). Run `login(token='...')` in the same session before `run_batch_of_slides.py` on any node (including GPU nodes).

---

## 1b. Analyzing 21 missing slides (grandqc/hest/otsu)

The 21 slides in `data/lists/missing_21_slides.csv` failed segmentation in both UNI2 and Virchow2 production runs (grandqc). To diagnose why no segmenter (grandqc, hest, otsu) succeeded:

**Quick scan (no GPU):** Search existing TRIDENT logs for error snippets:
```bash
cd /storage/brno2/home/filipsec/MIL
python scripts/analyze_segmentation_failures.py \
  --missing-csv data/lists/missing_21_slides.csv --scan-logs
```
Output: `data/analysis/segmentation_log_scan.csv`

**Full diagnostic:** Run TRIDENT `--task seg` for each slide × each segmenter (grandqc, hest, otsu):
```bash
qsub scripts/run_segmentation_analysis.pbs
# or interactively (2h GPU session):
python scripts/analyze_segmentation_failures.py \
  --missing-csv data/lists/missing_21_slides.csv \
  --output data/analysis/segmentation_diagnosis.csv
```
Output: `data/analysis/segmentation_diagnosis.csv` with columns `grandqc_ok`, `hest_ok`, `otsu_ok` and error notes per slide.

**Interpretation:** If `hest_ok` or `otsu_ok` = yes but grandqc failed → use hest/otsu for makeup. If all fail → WSI may be corrupted, unusual format, or need manual inspection.

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

---

## 5. After extraction: MIL training

- Use extracted bags (e.g. `.h5` or exported `.pt`) and the fold CSV.
- Train MIL (e.g. attention-based) with stratified train/val per fold; report validation AUC; early stopping; save best model by val AUC.

---

## 6. What was actually achieved

- **seg** → tissue segmentation (grandqc) → thumbnails, contours, **contours_geojson**
- **coords** → patch coordinates → `*_patches.h5`
- **feat** → uni_v2 features → `features_uni_v2/*.h5`

The front-end of the real MIL pipeline is working. Freeze this recipe; then move to production folds, production output root, full extraction, and MIL training.
