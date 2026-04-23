# MIL — Multiple Instance Learning for PANDA histopathology

PyTorch code for **ISUP grade group** prediction on the **PANDA** challenge using **pre-extracted patch features** (bags of vectors per whole-slide image). Typical pipeline: **TRIDENT** (or similar) writes one `.h5` per slide; this repo implements attention-based MIL aggregators, training, cross-validation, and attention visualization.

The mainline remains **bag-of-patches MIL** over tile embeddings. This repo now also includes:

- shared-coords tile extraction support for **H-optimus-0**, **H-optimus-1**, and **Prov-GigaPath**
- current MIL heads: `attention`, `gated`, `ordinal`, `clam`
- a separate **experimental Prov-GigaPath slide-encoder** path outside the core `mil.model` registry

## Requirements

- Python **≥ 3.9**
- [PyTorch](https://pytorch.org/get-started/locally/) (match your CUDA/CPU setup; not pinned in `pyproject.toml`)
- Project deps: `pip install -e .`
- Optional dev: `pip install -e ".[dev]"`

Training and evaluation assume **GPU** for reasonable throughput.

## Install

```bash
cd MIL
pip install -e .
# Then install PyTorch for your platform, e.g.:
# pip install torch --index-url https://download.pytorch.org/whl/cu124
```

## Data you need locally

- **Labels:** PANDA `train.csv` (or your own table with slide IDs and ISUP labels). Default path in scripts: `data/raw/train.csv`.
- **Features:** directory of `{slide_id}.h5` files with a dataset key such as `features`, shape `[N_patches, feat_dim]`, optionally `coords` for spatial maps. Feature dimension must match the encoder you used (e.g. UNI v2 vs Virchow2 vs H-optimus / GigaPath tile encoder).
- **Splits:** repo includes `data/splits/panda_5fold_stratified.csv` for stratified 5-fold CV.

Large assets (**WSI tiles, `.h5` bags, checkpoints**) are **not** committed; see `.gitignore`.

## Encoder Support

| Encoder key | Hugging Face repo | Dim | Token requirement | Extraction mode | Downstream route |
|------|------|---:|------|------|------|
| `uni_v2` / `uni2` | `MahmoodLab/UNI2-h` | 1536 | `.hf_token -> HF_TOKEN` (access required) | native TRIDENT `--task all` | MIL bag path |
| `virchow2` | `paige-ai/Virchow2` | 2560 | `.hf_token -> HF_TOKEN` (access required) | native TRIDENT `--task all` | MIL bag path |
| `conch_v1` / `conch` | `MahmoodLab/CONCH` | 512 | `.hf_token -> HF_TOKEN` (access required) | native TRIDENT `--task all` | MIL bag path |
| `hoptimus0` | `bioptimus/H-optimus-0` | 1536 | `H0-optimus0.hf_token -> .hf_token -> HF_TOKEN` | shared-coords `scripts/trident_chunk_extract_feat_only.pbs` | MIL bag path |
| `hoptimus1` | `bioptimus/H-optimus-1` | 1536 | `H-optimus1.hf_token -> .hf_token -> HF_TOKEN` | shared-coords `scripts/trident_chunk_extract_feat_only.pbs` | MIL bag path |
| `gigapath` | `prov-gigapath/prov-gigapath` | 1536 | `prov-gigapath.hf_token -> .hf_token -> HF_TOKEN` | shared-coords `scripts/trident_chunk_extract_feat_only.pbs` | MIL bag path + experimental slide path |

Notes:

- `hoptimus1` is a gated **non-commercial academic** model on Hugging Face.
- Shared-coords extraction keeps the same TRIDENT patch grid and swaps only the encoder, so cross-encoder comparisons stay fair by default.
- The core HDF5 contract does not change: one `{slide_id}.h5` with dataset key `features`, plus optional `coords`.

## Train (main entry point)

Five-fold CV, model choices (`attention`, `gated`, `ordinal`, `clam`), patch subsampling, optional attention-biased sampling:

```bash
python scripts/run_training.py \
  --config configs/pipelines/mil_training.yaml \
  --feat-dir /path/to/features_uni_v2 \
  --model clam \
  --train-csv data/raw/train.csv \
  --splits-csv data/splits/panda_5fold_stratified.csv
```

You can also use the cluster wrapper with encoder presets:

```bash
export CONFIG=/storage/brno2/home/filipsec/MIL/configs/pipelines/mil_training.yaml
export ENCODER=hoptimus1
export FEAT_DIR=/path/to/features_hoptimus1
export MODEL=clam
qsub -V scripts/mil_training.pbs
```

If you omit `--config` or `CONFIG`, the scripts load the matching default YAML from `configs/pipelines/` automatically and then apply explicit CLI or env overrides.

Hugging Face access rules differ by encoder. The registry-backed helper script and PBS workflows resolve tokens in a deterministic order and fail fast if a model license has not been accepted. See [docs/TRIDENT_WORKING_CONFIG_AND_NEXT_STEPS.md](/Users/fs/Desktop/MIL/docs/TRIDENT_WORKING_CONFIG_AND_NEXT_STEPS.md) and [docs/VIRCHOW2_VS_UNI2.md](/Users/fs/Desktop/MIL/docs/VIRCHOW2_VS_UNI2.md).

## Feature Extraction Workflows

- **Native TRIDENT extraction:** use `scripts/trident_chunk_extract*.pbs` for encoders already wired directly into TRIDENT such as UNI2, Virchow2, and CONCH.
- **Shared-coords feature-only extraction:** use [scripts/trident_chunk_extract_feat_only.pbs](/Users/fs/Desktop/MIL/scripts/trident_chunk_extract_feat_only.pbs) for `hoptimus0`, `hoptimus1`, and `gigapath`. This reuses existing `20x_256px_0px_overlap/patches` and writes `features_hoptimus0`, `features_hoptimus1`, or `features_gigapath`.
- **Registry inspection:** `python scripts/encoder_registry.py table`
- **Config inspection:** `python scripts/config_cli.py list` or `python scripts/config_cli.py show --pipeline mil_training`

## Feature Space Analysis (Cross-encoder)

Thesis-focused comparison of patch embedding spaces across encoders (e.g. UNI2, GigaPath, Virchow2) is available via:

```bash
python scripts/analyze_feature_spaces.py \
  --config configs/pipelines/feature_space_analysis.yaml
```

MetaCentrum ensemble preset:

```bash
python scripts/analyze_feature_spaces.py \
  --config configs/pipelines/feature_space_analysis.yaml \
  --preset metacentrum_ensemble
```

Artifacts are written to `results/feature_space_analysis/` by default:

- `encoder_basic_stats.csv`
- `pairwise_space_similarity.csv` (+ `cka_matrix.csv`, `svcca_matrix.csv`)
- `alignment_results.csv`
- `probe_results.csv` (if labels are configured)
- `spatial_smoothness.csv` + `neighborhood_preservation.csv` (optional)
- `plots/*.png`
- `summary.md`

Notes:

- The analysis treats embeddings as **distributed representations** and does not assign semantics to individual dimensions.
- Pairwise metrics support encoders with different dimensions via CKA/SVCCA and optional projection.
- Patch matching uses shared coords when available; otherwise it falls back to index-based matching (reported in outputs).

## Prov-GigaPath Slide Encoder (Experimental)

Prov-GigaPath supports a separate slide encoder in addition to the tile encoder. That path stays separate from the current MIL stack:

1. extract tile features into `features_gigapath`
2. run [gigapath_slide_embed.py](/Users/fs/Desktop/MIL/scripts/gigapath_slide_embed.py) to cache one `slide_id.pt` embedding per WSI
3. train a lightweight slide-level head with [gigapath_slide_train.py](/Users/fs/Desktop/MIL/scripts/gigapath_slide_train.py)

This is an experiment path for comparison, not the default training route.

## Layout

| Path | Role |
|------|------|
| `mil/` | `model`, `data`, `train`, `loss`, `utils`; attention / WSI viz helpers |
| `configs/` | YAML-backed encoder registry and pipeline defaults |
| `scripts/` | CLIs, PBS helpers, encoder registry, TRIDENT-related utilities, GigaPath slide scripts |
| `docs/` | Pipeline notes, geometry, encoder comparisons |
| `notebooks/` | Exploratory workflows |
| `figures/` | Example plots / result tables |

## Citation / challenge

If you use PANDA data or report results, cite the **PANDA challenge** and dataset terms appropriate to your use. This repository is research tooling, not a medical device.
