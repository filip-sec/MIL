# Encoder Support Matrix

This document is the canonical summary of which pathology foundation models are supported in this repo, how they are authenticated, where their features are written, and which downstream path they use.

## Supported Encoders

| Encoder key | Hugging Face repo | Dim | Feature dir | Token lookup | Extraction mode | Downstream |
|---|---|---:|---|---|---|---|
| `uni_v2` / `uni2` | `MahmoodLab/UNI2-h` | 1536 | `features_uni_v2` | `.hf_token -> HF_TOKEN` | native TRIDENT `--task all` | MIL bag |
| `virchow2` | `paige-ai/Virchow2` | 2560 | `features_virchow2` | `.hf_token -> HF_TOKEN` | native TRIDENT `--task all` | MIL bag |
| `conch_v1` / `conch` | `MahmoodLab/CONCH` | 512 | `features_conch_v1` | `.hf_token -> HF_TOKEN` | native TRIDENT `--task all` | MIL bag |
| `hoptimus0` | `bioptimus/H-optimus-0` | 1536 | `features_hoptimus0` | `H0-optimus0.hf_token -> .hf_token -> HF_TOKEN` | shared-coords `--task feat` | MIL bag |
| `hoptimus1` | `bioptimus/H-optimus-1` | 1536 | `features_hoptimus1` | `H-optimus1.hf_token -> .hf_token -> HF_TOKEN` | shared-coords `--task feat` | MIL bag |
| `gigapath` | `prov-gigapath/prov-gigapath` | 1536 | `features_gigapath` | `prov-gigapath.hf_token -> .hf_token -> HF_TOKEN` | shared-coords `--task feat` | MIL bag + experimental slide encoder |

## Defaults

- Shared patch-grid comparison is the default for `hoptimus0`, `hoptimus1`, and `gigapath`.
- Those encoders reuse existing TRIDENT coords from `20x_256px_0px_overlap/patches`.
- The tile model sees its own model-specific transform, but the slide patch set stays aligned across encoders.
- The HDF5 bag contract does not change: `features` plus optional `coords`.

## Shared-Coords Feature-Only Workflow

Use [scripts/trident_chunk_extract_feat_only.pbs](/Users/fs/Desktop/MIL/scripts/trident_chunk_extract_feat_only.pbs).

Example:

```bash
cd /storage/brno2/home/filipsec/MIL
export CONFIG=/storage/brno2/home/filipsec/MIL/configs/pipelines/trident_feat_only_shared_coords.yaml
qsub -v "CHUNK_CSV=/abs/panda_chunk_000.csv,ENCODER=hoptimus1,COORDS_DIR=20x_256px_0px_overlap" \
  scripts/trident_chunk_extract_feat_only.pbs
```

What the script now guarantees:

- encoder registry lookup from one shared source
- deterministic token resolution
- Hugging Face access smoke check before the TRIDENT run starts
- standardized output dirs:
  - `features_hoptimus0`
  - `features_hoptimus1`
  - `features_gigapath`

## Training Wrapper

Use [scripts/mil_training.pbs](/Users/fs/Desktop/MIL/scripts/mil_training.pbs) with `ENCODER` when possible.

Example:

```bash
cd /storage/brno2/home/filipsec/MIL
export CONFIG=/storage/brno2/home/filipsec/MIL/configs/pipelines/mil_training.yaml
export ENCODER=gigapath
export FEAT_DIR=/storage/brno2/home/filipsec/MIL/data/trident_out/panda_ensemble_grandqc_20x_256_ov0/20x_256px_0px_overlap/features_gigapath
export MODEL=clam
qsub -V scripts/mil_training.pbs
```

When `ENCODER` is set:

- `FEATURE_DIM` is derived from the registry if you did not set it manually
- checkpoint run naming uses the encoder key by default
- checkpoints store `encoder_name`

## Prov-GigaPath Slide Encoder Path

This path is intentionally separate from the MIL bag stack.

1. Extract tile embeddings to `features_gigapath`
2. Build slide embeddings:

```bash
python scripts/gigapath_slide_embed.py \
  --config configs/pipelines/gigapath_slide_embed.yaml \
  --feat-dir /path/to/features_gigapath \
  --coords-dir /path/to/20x_256px_0px_overlap/patches \
  --out-dir slide_embeddings_gigapath
```

3. Train the slide-level classifier:

```bash
python scripts/gigapath_slide_train.py \
  --config configs/pipelines/gigapath_slide_train.yaml \
  --emb-dir slide_embeddings_gigapath \
  --train-csv data/raw/train.csv \
  --splits-csv data/splits/panda_5fold_stratified.csv
```

This is an experiment path for comparison against bag-level MIL, not the repo default.

## Notes on Access

- `hoptimus1` is gated and restricted to non-commercial academic use on Hugging Face.
- `gigapath` requires accepting the model terms on Hugging Face before download.
- The registry helper script can inspect the matrix and verify access:

```bash
python scripts/encoder_registry.py table
python scripts/encoder_registry.py check-access --encoder hoptimus1
python scripts/config_cli.py show --pipeline mil_training
```
