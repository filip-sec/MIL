# PANDA WSI MIL Pipeline – Planning Document

**Goal:** Train a slide-level classifier (ISUP grade 0–5) using patch-level features from a pretrained encoder. End-to-end from PANDA metadata + WSIs → slides CSV → patch/feature extraction → .pt export → MIL training with stratified split and validation AUC.

**Data:** Kaggle competition `prostate-cancer-grade-assessment` (PANDA).  
**Key files:** `train.csv` (image_id, data_provider, isup_grade, gleason_score), `train_images/<image_id>.tiff`.

---

## 1. Pipeline Overview (Conceptual)

```
PANDA (Kaggle)
    │
    ├── train.csv ──────────────────────────────────────────────┐
    │                                                             │
    └── train_images/*.tiff ──► [1] Batch download (resumable)    │
                │                     │                           │
                ▼                     ▼                           │
           BASE/train_images/    data/panda_manifests/             │
                │                (batch_aa, batch_ab, ...)         │
                │                                                     │
                └──────────────────────┬─────────────────────────────┤
                                       ▼                             │
                [2] Build slides CSV from train.csv ◄────────────────┘
                    data/slides.csv (slide_id, label, data_provider, split?, wsi_path?)
                                       │
                [3] Fill wsi_path from BASE/train_images/ ◄──────────┘
                                       │
                                       ▼
                [4] Patch + feature extraction (TRIDENT / CTransPath / UNI)
                    Batched, resumable; job_dir; skip existing
                                       │
                                       ▼
                [5] Export to .pt
                    data/encoder_out/features/<slide_id>.pt
                    data/encoder_out/coords/<slide_id>.pt
                    (+ slide list matching slides.csv)
                                       │
                                       ▼
                [6] MIL training
                    Stratified split (or from CSV split); val AUC; early stop; best model
                                       │
                [7] Split audit (optional)
                    Train vs val from features → AUC ~0.5 sanity check
```

---

## 2. Directory and Layout Plan

- **BASE** (env/config): Root for raw PANDA data (e.g. `~/datasets/panda` or `$SCRATCHDIR/panda`).
  - `BASE/train_images/` – downloaded TIFFs.
- **data/** (project-relative or env):
  - `data/panda_manifests/` – batch files (`batch_aa`, `batch_ab`, …), each line = `train_images/<slide_id>.tiff`.
  - `data/slides.csv` – slide_id, label (isup_grade), data_provider, split (optional), wsi_path (filled by step 3).
- **JOB_DIR** (env): Shared directory for patch/feature extraction runs (resumable across batches).
  - Layout depends on TRIDENT (e.g. subdirs per slide or per batch; patches + features).
- **data/encoder_out/** (or configurable):
  - `features/` – one `.pt` per slide (tensor of shape [N_patches, feat_dim]).
  - `coords/` – one `.pt` per slide (patch coordinates, e.g. [N_patches, 2] or [N_patches, 4]).
  - Optional: `slides_list.csv` or `slides.txt` aligned with slides.csv (slide_id, label, split).
- **Training outputs:** e.g. `runs/` or `checkpoints/` – best model by val AUC, logs, optional 10× seed runs.

---

## 3. Step-by-Step Plan

### Step 1: Download WSIs in batches (resumable)

- **Inputs:** List of all `train_images/<slide_id>.tiff` (from train.csv or Kaggle data listing). Target dir `BASE`.
- **Manifests:** Script to create `data/panda_manifests/` with batch files (e.g. 50–200 lines per file; batch_aa, batch_ab, …). Each line = one path to download, e.g. `train_images/00001.tiff`.
- **Download script (conceptual):**
  - For each batch file in order:
    - For each line `f` in the batch file:
      - If `$BASE/$f` already exists → skip.
      - Else: `kaggle competitions download -c prostate-cancer-grade-assessment -f "$f"` (or equivalent with kagglehub if single-file API exists); add **delay (e.g. 10 s)** between API calls to avoid 429.
    - After processing the batch: unzip any new `.zip` in current dir with `unzip -n -q` (no overwrite), then delete the zip(s).
  - Prereqs: `pip install kaggle`, `~/.kaggle/kaggle.json`, competition rules accepted. Document these in README.

### Step 2: Build slides CSV from train.csv

- **Input:** `train.csv` (after download or from Kaggle).
- **Output:** `data/slides.csv` with columns: `slide_id` (= image_id), `label` (= isup_grade 0–5), `data_provider`, optionally `split` (train/val), optionally `wsi_path` (empty initially).
- **Split:** Stratified by **data_provider** (provider as group key): no provider appears in both train and val. E.g. 80/20 provider-stratified; or fixed seed for reproducibility. Option to write split into CSV for later steps.
- **Reproducibility:** Fixed seed for split; document seed in README.

### Step 3: Fill WSI paths

- **Input:** `data/slides.csv`, WSI directory (e.g. `BASE/train_images`).
- **Logic:** For each `slide_id`, resolve file (e.g. `BASE/train_images/<slide_id>.tiff`); if exists, set `wsi_path` (absolute or relative as per convention). Optionally flag or drop slides with missing WSI.
- **Output:** Updated `data/slides.csv` with `wsi_path` filled (and optionally a filtered list if some WSIs are missing).

### Step 4: Patch + feature extraction (batched, resumable)

- **Tool:** TRIDENT (or equivalent) with pretrained encoder: e.g. CTransPath (224 px → 768-dim) or UNI (→ 1536-dim). Fixed magnification (e.g. 20x); patch size as per encoder (e.g. 224).
- **Input:** slides CSV (with wsi_path), WSI dir; batch size (e.g. 500 slides per batch).
- **Job dir:** Single shared **JOB_DIR**; each run (batch) writes into the same structure; skip slides that already have output (patches + features).
- **Execution:** Loop over slides in batches; for each slide, if output exists skip else run TRIDENT (or wrapper). Suitable for PBS array jobs or sequential batch jobs on cluster (e.g. MetaCentrum).
- **Output layout:** Defined by TRIDENT (e.g. `JOB_DIR/<slide_id>/patches/`, `JOB_DIR/<slide_id>/features.npy` or similar). Document exact layout for step 5.

### Step 5: Export to .pt for MIL

- **Input:** TRIDENT (or encoder) output in JOB_DIR; slides CSV (slide_id, label, split).
- **Output:**
  - `data/encoder_out/features/<slide_id>.pt` – tensor [N_patches, feat_dim].
  - `data/encoder_out/coords/<slide_id>.pt` – patch coordinates.
  - A slide list (CSV or txt) that matches slides.csv: slide_id, label, split (and optionally path to .pt) so MIL loader can iterate over slides with labels and split.
- **Alignment:** Only export slides present in slides.csv; skip or warn if slide in CSV has no features.

### Step 6: MIL training

- **Input:** Dirs of .pt features and coords; slide list with label and split (from slides.csv or derived).
- **Model:** Attention-based MIL (or similar) on patch features → slide-level logits → classification (ISUP 0–5). Loss: cross-entropy (or ordinal if desired); metric: validation AUC (macro or per-class as defined).
- **Split:** By default use stratified split (by provider already in step 2); optional override: if `split` column exists in CSV, use it for train/val.
- **Training:** Early stopping on val AUC; save best model by val AUC; fixed seed; optional 10× runs with different seeds → report mean ± std val AUC.
- **Config:** Paths (BASE, SLIDES_CSV, encoder_out dirs, checkpoint dir), seed, batch size, epochs, early-stop patience, feature dimension (for loader).

### Step 7: Split audit (optional)

- **Purpose:** Check that train/val split is not confounded by site/provider (i.e. model cannot predict train vs val from features alone).
- **Method:** Train a small classifier (e.g. linear) to predict binary label “train” vs “val” from slide-level features (e.g. mean-pooled or aggregated patch features). Expect AUC ≈ 0.5. If AUC > 0.6, split may be confounded.
- **Input:** Same .pt features and slide list with split; output: AUC and short report.

### Step 8: Reproducibility

- **Seeds:** Fixed seed for (a) provider-stratified split, (b) MIL training (torch, numpy, etc.). Document in README and in config.
- **Optional:** 10× runs with different seeds; report mean ± std of val AUC; same split or re-split per seed (document which).

---

## 4. Configuration and Environment Variables

Use env vars or a small config (e.g. YAML/JSON) so the pipeline works on cluster (MetaCentrum) and locally:

| Variable / key | Purpose | Example |
|----------------|---------|---------|
| BASE | Root for raw PANDA data | `/storage/brno2/home/user/datasets/panda` |
| WSI_DIR | Directory of TIFFs (often BASE/train_images) | `$BASE/train_images` |
| SLIDES_CSV | Path to slides CSV | `data/slides.csv` |
| DATA_DIR | Project data root (manifests, slides, encoder_out) | `data/` |
| PANDA_MANIFESTS_DIR | Batch manifest files | `data/panda_manifests/` |
| JOB_DIR | Patch/feature extraction output (shared, resumable) | `data/trident_job` or `$SCRATCHDIR/trident_job` |
| ENCODER_OUT_DIR | Base for features/ and coords/ | `data/encoder_out` |
| FEATURES_DIR | Per-slide .pt features | `$ENCODER_OUT_DIR/features` |
| COORDS_DIR | Per-slide .pt coords | `$ENCODER_OUT_DIR/coords` |
| SEED | Random seed for split and training | 42 |
| ENCODER_NAME | Encoder identifier (for multi-encoder support) | ctranspath / uni_v2 |

---

## 5. Optional: Multiple Encoders

- **Encoders:** e.g. CTransPath (768-dim), UNI (1536-dim), another (512-dim). Document name → feature dir and feat_dim.
- **Layout:** e.g. `features_ctranspath/`, `features_uni_v2/` (or `encoder_out/ctranspath/features/`). Same coords can be shared if patch grid is identical.
- **Downstream:** MIL script accepts `--feat-dim` and `--features-dir` (or encoder name) so one codebase can run with different encoders.

---

## 6. Deliverables Checklist

- [ ] **Manifests:** Script or instructions to create `data/panda_manifests/` batch files from the list of `train_images/*.tiff` paths.
- [ ] **Download:** Script that: BASE + batch files → download per file with skip-if-exists, delay between Kaggle calls, unzip -n -q, remove zips. Doc: kaggle.json, accept rules.
- [ ] **Slides CSV:** Script: train.csv → slides.csv (slide_id, label, data_provider, optional split, wsi_path); stratified split by data_provider; seed.
- [ ] **Fill paths:** Script: slides.csv + WSI dir → fill wsi_path (and optionally filter missing).
- [ ] **Patch/feature extraction:** TRIDENT (or equivalent) in batches; shared JOB_DIR; skip existing; doc for batch size and PBS usage.
- [ ] **Export .pt:** Script: JOB_DIR + slides.csv → features/*.pt, coords/*.pt, slide list for MIL.
- [ ] **MIL training:** Script/trainer: load .pt + slide list; stratified or CSV split; val AUC; early stop; best model; seed; optional 10× seeds.
- [ ] **Split audit:** Optional script: train vs val from features → AUC report.
- [ ] **Docs:** README with (1)–(7) in order, env vars, prereqs (kaggle, TRIDENT, Python deps), and reproducibility (seeds, encoder names/dims).

---

## 7. Order of Operations and Dependencies

1. Get train.csv (or full competition download once) to know image_ids and to build manifests.
2. Create batch manifests (list of train_images/*.tiff).
3. Run batch download script → populate BASE/train_images/.
4. Build slides.csv from train.csv (step 2); then fill wsi_path (step 3).
5. Run patch + feature extraction in batches (step 4) → JOB_DIR.
6. Export to .pt (step 5) → data/encoder_out/.
7. Run MIL training (step 6); optionally run split audit (step 7) before or after.
8. Optional: 10× seed runs and mean ± std AUC.

---

## 8. Risks and Mitigations

- **429 from Kaggle:** Delay between API calls (e.g. 10 s); batch download with skip-if-exists to avoid re-downloading; consider one-time full download then rsync to cluster.
- **Provider stratification:** Small number of providers may yield uneven train/val sizes; document choice (e.g. min 2 slides per provider in val) or fallback to random stratified with seed.
- **TRIDENT layout:** Lock and document TRIDENT output layout so export script (step 5) is stable across versions.
- **Disk:** WSIs and features are large; use scratch on cluster for JOB_DIR if needed; document cleanup (e.g. remove patches after features extracted).

---

*Document version: 1.0 – planning only; no code written yet.*
