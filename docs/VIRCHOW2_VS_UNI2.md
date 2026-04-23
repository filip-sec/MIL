# UNI2 (TRIDENT) vs Virchow2 feature extraction

For the full encoder matrix, including `hoptimus0`, `hoptimus1`, and `gigapath`, see [ENCODER_SUPPORT.md](/Users/fs/Desktop/MIL/docs/ENCODER_SUPPORT.md). This note stays focused on the UNI2/Virchow2 comparison and the patch-size consequences of 256px vs 224px workflows.

## First-time access to Virchow2

Virchow2 is a **gated model** on Hugging Face. You must request access before using it.

1. **Open the model page:**  
   [https://huggingface.co/paige-ai/Virchow2](https://huggingface.co/paige-ai/Virchow2)

2. **Log in** with your Hugging Face account (or create one). Use an **institutional / work email** if possible; access is often tied to that.

3. **Request access:**  
   On the model page, click **“Agree and access repository”** (or “Request access”). Accept the **terms of use** (CC-BY-NC-ND 4.0, non-commercial research). Submit the form.

4. **Wait for approval:**  
   Paige AI typically reviews requests **weekly**. You’ll get an email when access is granted.

5. **Create a token and log in locally:**  
   - Go to [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) → **New token** (read access is enough).  
   - Then in your environment (or on the cluster):
   ```bash
   huggingface-cli login
   ```
   Paste the token when prompted. Or in Python before loading the model:
   ```python
   from huggingface_hub import login
   login()  # or login(token="hf_...")
   ```

6. **Verify:**  
   After access is granted, `timm.create_model("hf-hub:paige-ai/Virchow2", pretrained=True, ...)` (or the script’s `get_virchow2_model()`) should download and run without a “gated model” error.

**License:** CC-BY-NC-ND 4.0 — non-commercial, academic research only; no redistribution of the model; commercial use requires separate approval from the authors.

---

Yes, there **will be differences** between features from UNI2 and Virchow2. They are different models and feature spaces; the MIL head must be trained separately for each encoder.

## Comparison

| Aspect | UNI2 (current TRIDENT) | Virchow2 |
|--------|-------------------------|----------|
| **Input size** | 256×256 px | 224×224 px |
| **Magnification** | 20× (fixed in your pipeline) | Pretrained at 5×, 10×, 20×, 40× (mixed) |
| **Feature dim** | From TRIDENT (e.g. 1024–1536; check your `.h5`) | **2560** (class token + mean patch; standard for this pipeline) |
| **Architecture** | Encoder in TRIDENT (UNI v2) | ViT-H/14, 632M params, SwiGLU, 4 register tokens |
| **Pretraining** | UNI pretraining (histo) | 3.1M WSIs, modified DINOv2, mixed mag |
| **Output** | One vector per patch (one tile = one instance) | One vector per 224×224 tile (patch tokens or tile embedding) |

## Practical implications

1. **Feature dimension**  
   MIL’s `AttentionMIL(feat_dim=...)` must match the encoder: use the actual dimension from your UNI2 `.h5` for UNI2, and **2560** for Virchow2 (no mixing without a projector).

2. **Patch/tile layout**  
   - UNI2: TRIDENT uses 256×256 patches at 20×, possibly with tissue masking (grandqc).  
   - Virchow2: Expects 224×224. You can either:
     - Extract a **new grid at 224×224** (and optionally align to similar locations as TRIDENT), or  
     - Take TRIDENT’s patch **coordinates**, crop 256×256 from the WSI, **center-crop or resize to 224×224**, then run Virchow2 so patch sets are comparable (same number and approximate locations).

3. **Semantics and performance**  
   Different pretraining and architectures ⇒ different feature distributions and invariances. You cannot directly compare or mix UNI2 and Virchow2 feature vectors without a learned projection. For slide-level ISUP, train a **separate MIL model** (or same code with a different `feat_dir` and `feat_dim`) and compare validation metrics (e.g. balanced accuracy, kappa).

4. **Storage**  
   Keep Virchow2 features in a **separate dir**, e.g.  
   `data/trident_out/panda_virchow2_grandqc_20x_224_ov0/.../features_virchow2/`  
   so the same MIL code can run with `feat_dir` + `feat_dim=2560` without overwriting UNI2 `.h5` files.

## Summary

- **Yes, there will be diff:** different input size, dimension, architecture, and pretraining ⇒ different features and possibly different downstream performance.  
- Use Virchow2 by extracting 224×224 tiles → Virchow2 → save `.h5` with key `"features"` and shape `[N, 2560]`, then point MIL to the new `feat_dir` and set `feat_dim=2560`. TRIDENT args: `--patch_encoder virchow2 --patch_size 224 --mag 20`.

## Coords for attention maps

**Virchow2:** TRIDENT writes `coords` inside the feature `.h5` file. No `--coords-dir` needed.

**UNI2:** Coords may be in separate `*_patches.h5` under `20x_256px_0px_overlap/`. Use `--coords-dir` if feat `.h5` lacks coords.

Verify: `python scripts/check_h5_coords.py <feat_dir> -n 5`

### Semantics (TRIDENT / OpenSlide) — critical for figures

From [TRIDENT `WSIPatcher`](https://github.com/mahmoodlab/TRIDENT): each `(x, y)` in `coords` is the **top-left** corner of the patch in **level-0** slide pixels (same reference as OpenSlide `read_region(..., location=(x,y), ...)`).

The HDF5 `coords` dataset usually has attributes, including:

- **`patch_size_level0`** — width/height of that patch in level-0 pixels (use this with `read_region` at level 0).
- **`patch_size`** — patch edge length at target magnification (e.g. 224).
- **`level0_magnification`**, **`target_magnification`** — TRIDENT also documents `patch_size_level0 ≈ patch_size * level0_mag // target_mag`.

**Do not** treat coords as patch centers when cropping the WSI; MIL `plot_attention_figure_publishable` uses top-left + `patch_size_level0` for crops, and patch **centers** only for heatmap / top-k marker placement.

## Extraction options

**A) TRIDENT with Virchow2 (if supported)**  
Use the same chunked PBS workflow with Virchow2 encoder and 224 px patches:

- Script: `scripts/trident_chunk_extract.pbs` with `ENCODER=virchow2` (or `trident_chunk_extract_feat_only.pbs` for feat-only runs).
- Same as the canonical uni_v2 run but: `--patch_encoder virchow2 --patch_size 224 --mag 20`, and `JOB_DIR=.../panda_virchow2_grandqc_20x_224_ov0`.
- Submit: `qsub -v CHUNK_CSV=/path/to/chunk.csv,ENCODER=virchow2 scripts/trident_chunk_extract.pbs` (HF token resolved via the registry — see `.cursor/rules/trident-extraction.mdc`).
- This only works if the TRIDENT repo (`run_batch_of_slides.py`) supports `--patch_encoder virchow2`. If not, add a Virchow2 encoder in TRIDENT or use (B). In the notebook set `feat_dim=2560` for Virchow2.

**B) Other extractors**  
This repo does not ship a standalone Virchow2 extractor script; use path **A** via TRIDENT, or any external pipeline that writes per-slide `.h5` with key `"features"` (Virchow2 is typically 2560-dim). Point training at that directory and set `feat_dim=2560`.
