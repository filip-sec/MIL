# MIL — Multiple Instance Learning for PANDA histopathology

PyTorch code for **ISUP grade group** prediction on the **PANDA** challenge using **pre-extracted patch features** (bags of vectors per whole-slide image). Typical pipeline: **TRIDENT** (or similar) writes one `.h5` per slide; this repo implements attention-based MIL aggregators, training, cross-validation, and attention visualization.

This repository now also includes the integrated **`crc-preproc`** pipeline (`crc_preproc/`) for WSI tissue masking, tiling, QC montage creation, and tile/slide index generation.

## Requirements

- Python **≥ 3.9**
- [PyTorch](https://pytorch.org/get-started/locally/) (match your CUDA/CPU setup; not pinned in `pyproject.toml`)
- Project deps: `pip install -e .`
- Optional dev: `pip install -e ".[dev]"`
- Optional WSI preprocessing extras (from integrated `crc-preproc`): `pip install -e ".[preproc]"`

Training and evaluation assume **GPU** for reasonable throughput.

## Install

```bash
cd MIL
pip install -e .
# Then install PyTorch for your platform, e.g.:
# pip install torch --index-url https://download.pytorch.org/whl/cu124

# Optional preprocessing dependencies (OpenSlide + tiling/QC stack)
pip install -r requirements-preproc.txt
```

For preprocessing on Linux, install OpenSlide system libraries (for `openslide-python`) as needed:

```bash
sudo apt-get update
sudo apt-get install -y libopenslide0 libopenslide-dev libgl1 libglib2.0-0
```

## Data you need locally

- **Labels:** PANDA `train.csv` (or your own table with slide IDs and ISUP labels). Default path in scripts: `data/raw/train.csv`.
- **Features:** directory of `{slide_id}.h5` files with a dataset key such as `features`, shape `[N_patches, feat_dim]`, optionally `coords` for spatial maps. Feature dimension must match the encoder you used (e.g. UNI v2 vs Virchow2).
- **Splits:** repo includes `data/splits/panda_5fold_stratified.csv` for stratified 5-fold CV.

Large assets (**WSI tiles, `.h5` bags, checkpoints, SICAPv2 images**) are **not** committed; see `.gitignore`.

## Train (main entry point)

Five-fold CV, model choices (`attention`, `gated`, `ordinal`), patch subsampling, optional attention-biased sampling:

```bash
python scripts/run_training.py \
  --feat-dir /path/to/features_uni_v2 \
  --model ordinal \
  --train-csv data/raw/train.csv \
  --splits-csv data/splits/panda_5fold_stratified.csv
```

Quick single-setup baseline (smaller surface area): `scripts/run_mil_training.py`.

## Preprocess WSIs (integrated from `filip-sec/crc-preproc`)

Batch WSI preprocessing into tiles + QC + CSV indexes:

```bash
python scripts/run_crc_preproc.py \
  --wsi_dir /path/to/wsi \
  --pattern "*.svs" \
  --labels_csv /path/to/labels.csv \
  --out_dir .
```

Generated structure:

- `data/images/thumbs/` (thumbnails, masks, overlays)
- `data/images/tiles/<slide_id>/` (PNG tiles)
- `data/labels/*_tiles.csv`, `tiles_index.csv`, `slides_index.csv`

## End-to-end preprocess → MIL workflow

1. Run WSI preprocessing (above) to generate quality-controlled tile outputs.
2. Extract patch embeddings (`.h5`) with your feature encoder pipeline (e.g., TRIDENT/UNI/Virchow2) into a feature directory.
3. Train MIL model from extracted features:

```bash
python scripts/run_training.py \
  --feat-dir /path/to/features \
  --train-csv data/raw/train.csv \
  --splits-csv data/splits/panda_5fold_stratified.csv
```

4. Optional integration smoke check:

```bash
python scripts/smoke_crc_preproc.py
```

Hugging Face gated models (e.g. Virchow2): use a **read token** via `huggingface-cli login`, env (`HF_TOKEN`), or a one-line file `.hf_token` (gitignored)—see `docs/VIRCHOW2_VS_UNI2.md` and cluster-oriented notes in `docs/`.

## Layout

| Path | Role |
|------|------|
| `mil/` | `model`, `data`, `train`, `loss`, `utils`; attention / WSI viz helpers |
| `scripts/` | CLIs, PBS helpers, TRIDENT-related utilities |
| `docs/` | Pipeline notes, geometry, encoder comparisons |
| `notebooks/` | Exploratory workflows |
| `figures/` | Example plots / result tables |

## Citation / challenge

If you use PANDA data or report results, cite the **PANDA challenge** and dataset terms appropriate to your use. This repository is research tooling, not a medical device.
