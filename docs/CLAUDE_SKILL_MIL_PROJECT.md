---
name: skill-mil-project
description: Context and instructions for the PANDA WSI MIL project. Use when assisting with this codebase, notebook (mil_workflow.ipynb), TRIDENT features, PyTorch MIL, or designing MIL algorithms.
---

# PANDA WSI MIL project – context for Claude

**Use this document when helping with the MIL (Multiple Instance Learning) codebase.** Upload or paste it into Claude so it can assist with code, debugging, and extensions in this project.

---

## 1. Project goal

- **Task:** Slide-level classification of prostate biopsies (ISUP grade 0–5) from whole-slide images (WSI).
- **Data:** Kaggle PANDA (`prostate-cancer-grade-assessment`): `train.csv` (image_id, isup_grade, gleason_score, data_provider), `train_images/<id>.tiff`, optional `train_label_masks/`.
- **Pipeline:** WSIs → TRIDENT (patch + encoder) → `.h5` patch features → PyTorch MIL (attention over patches) → slide-level prediction.

---

## 2. Repository layout

| Path | Purpose |
|------|--------|
| `MIL_ROOT` | Project root (env `MIL` or auto-detected by walking up until `data/` exists). |
| `data/raw/train.csv` | Labels: image_id, isup_grade, gleason_score, data_provider. |
| `data/raw/train_images/` | WSI TIFFs: `<image_id>.tiff`. |
| `data/raw/train_label_masks/` | Optional mask TIFFs. |
| `data/lists/` | Slide lists: e.g. `missing_21_slides.csv` (column `wsi` = `<id>.tiff`), `singletons/<id>.csv` for one-slide jobs. |
| `data/splits/panda_5fold_stratified.csv` | 5-fold split: columns `image_id`, `fold` (0–4). Created by `scripts/make_panda_5fold_splits.py`. |
| `data/trident_out/.../features_uni_v2/` | TRIDENT output: one `<slide_id>.h5` per slide; key `features` → array `[N_patches, feat_dim]`. |
| `notebooks/mil_workflow.ipynb` | Main workflow: setup, labels, OpenSlide inspection, splits, H5 loading, PyTorch MIL (Dataset, model, training). |
| `scripts/` | CLIs and PBS: `run_training.py`, `mil_training.pbs`, `trident_chunk_extract*.pbs`, `make_panda_5fold_splits.py`, `plot_attention_map.py`, etc. |
| `docs/` | TRIDENT_WORKING_CONFIG_AND_NEXT_STEPS.md, PANDA_WSI_MIL_PIPELINE_PLAN.md. |

---

## 3. Notebook workflow (mil_workflow.ipynb)

**Execution order:** Run cells top to bottom. Key variables:

- **MIL_ROOT** – Path, set from env or by walking up to `data/`.
- **data_dir** – `MIL_ROOT / "data" / "raw" / "train_images"`.
- **train_labels** – `pd.read_csv(train.csv).set_index("image_id")`; used for `isup_grade`, `gleason_score`.
- **train** – `train_labels.reset_index()` for compatibility.
- **splits_path** – `MIL_ROOT / "data" / "splits" / "panda_5fold_stratified.csv"`; **splits** – DataFrame with `image_id`, `fold`.
- **feat_dir** – Directory of `.h5` feature files; fallback to `trident_out_makeup_otsu` if main path missing.
- **train_ids, val_ids** – Built from splits (fold 0 = val) or 80/20 stratified; filtered to slides that have both labels and an `.h5` in `feat_dir`.

**OpenSlide:** Always close slides (e.g. `try`/`finally` and `biopsy.close()`). Paths use `Path`; pass `str(path)` to `openslide.OpenSlide`.

**Example slides:** Hardcoded list `example_slides` (21 slide IDs from `missing_21_slides.csv`) used for WSI inspection loops.

---

## 4. PyTorch MIL (notebook)

- **H5BagDataset(feat_dir, slide_ids, train_labels, key="features")**  
  Returns `(feats, label, slide_id)`; feats from `f["features"][:]` as float tensor.

- **collate_bags(batch)**  
  Pads variable-length bags to `[B, N_max, D]`, returns `(feats, mask, labels, ids)`. `mask` is `[B, N_max]`, True where valid.

- **AttentionMIL(feat_dim, num_classes=6, hidden=128)**  
  Forward: `encoder(x)` → `attention(h)` with `masked_fill(~mask, -1e9)` before softmax → weighted sum → `classifier(bag)`. Input `x`: `[B, N, D]`, `mask`: `[B, N]`.

- **Training cell**  
  If `model` is not defined, the cell builds DataLoaders, model, optimizer, and criterion from `train_ids`, `val_ids`, `feat_dir`, `train_labels`. Then 3 epochs of cross-entropy; after that, validation accuracy and balanced accuracy (sklearn).

**Conventions:** Use `device` (cuda if available). Batch size 8; `num_workers=0` in DataLoader. Label is integer `isup_grade` 0–5.

---

## 5. TRIDENT and feature extraction

- Features live under `.../20x_256px_0px_overlap/features_uni_v2/*.h5`. Each `.h5` has at least key `"features"` with shape `(N_patches, feat_dim)`.
- Segmenter: grandqc (production); for slides with “No contour” (QC false negative), rerun without grandqc and copy the single `.h5` into the main features dir.
- Scripts: PBS for chunked extraction; singleton lists in `data/lists/singletons/<id>.csv` (column `wsi` = `<id>.tiff`). See `docs/TRIDENT_WORKING_CONFIG_AND_NEXT_STEPS.md`.

---

## 6. Alternative feature extractor: Virchow2

**Virchow2** (Paige + Microsoft) is a self-supervised ViT pretrained on 3.1M histopathology WSIs; can be used as a **tile-level feature extractor** (frozen or finetuned) alongside or instead of TRIDENT/UNI.

| Item | Value |
|------|--------|
| Hub | `hf-hub:paige-ai/Virchow2` (HuggingFace; gated, login required) |
| Input | 224×224 RGB tile |
| Output | `output`: [B, 261, 1280]; token 0 = class, 1–4 = register (ignore), 5: = patch tokens [B, 256, 1280] |
| Tile embedding | `embedding = cat([class_token, patch_tokens.mean(1)], dim=-1)` → [B, **2560**] (use for MIL; 1280 = class or mean patch only) |
| Stack | PyTorch 2.0+, `timm` (>=0.9.11), `huggingface_hub`; init with `mlp_layer=SwiGLUPacked`, `act_layer=nn.SiLU` |
| Inference | Prefer GPU with `torch.autocast(device_type="cuda", dtype=torch.float16)` |

**Usage for MIL:** Extract 224×224 tiles (TRIDENT: `--patch_encoder virchow2 --patch_size 224 --mag 20`), run through Virchow2, save per-tile **2560-d** embedding into `.h5` with key `features`. Set `feat_dim=2560` in the MIL notebook.

**Login:** `huggingface-cli login` or `from huggingface_hub import login; login()` (required after access approval).  
**Ref:** Virchow2: Scaling Self-Supervised Mixed Magnification Models in Pathology (arxiv.org/abs/2408.00738); license CC-BY-NC-ND-4.0.

---

## 7. MIL-Lab (optional)

**MIL-Lab** (Mahmood Lab, ICML 2025) is a standardized MIL library with pretrained **FEATHER** slide foundation models and many MIL architectures. Useful if you want to try TransMIL, CLAM, etc., or start from FEATHER pretrained on PC-108 (24K slides).

- **Repo:** https://github.com/mahmoodlab/MIL-Lab  
- **Install:** `pip install -e .` (+ `smooth-topk` for CLAM).  
- **TRIDENT:** FEATHER-24K (CONCHv1.5) is integrated into TRIDENT; patch features from TRIDENT can be fed into MIL-Lab models.  
- **Encoders:** MIL-Lab maps encoder names to feature dims; **virchow2 → 2560**, uni_v2 → 1536, conch_v15 → 768, etc. (see `builders/_global_mappings.py`).  
- **Create model:** `create_model('abmil.base.uni_v2.pc108-24k', num_classes=6)` for FEATHER-24K (UNIv2) finetuned on PANDA; or `create_model('abmil.base.virchow2.none', num_classes=6)` for ABMIL with Virchow2 dim (no pretrained head).  
- **Input:** Batch of patch features `(batch_size, num_patches, feature_dim)`; same as our `.h5` → collate.  
- **Inference:** `results_dict, log_dict = model(features, loss_fn=..., label=..., return_attention=True)`.

**Ako to použiť, keď už máš vyextrahované features (.h5):**

1. **Inštalácia MIL-Lab** (v env kde beží notebook / training):
   ```bash
   git clone https://github.com/mahmoodlab/MIL-Lab.git && cd MIL-Lab && pip install -e .
   pip install git+https://github.com/oval-group/smooth-topk   # pre CLAM
   ```

2. **Dáta ostávajú rovnaké:** `feat_dir` s `.h5`, `train_labels`, `train_ids`/`val_ids` z foldov, `H5BagDataset` a `collate_bags` – nič nemusíš meniť.

3. **Vytvorenie modelu podľa encoderu (feat_dim):**
   - Virchow2 (2560): `create_model('abmil.base.virchow2.none', num_classes=6)`  
   - UNIv2 (1536): `create_model('abmil.base.uni_v2.none', num_classes=6)` alebo pretrained  
     `create_model('abmil.base.uni_v2.pc108-24k', num_classes=6, from_pretrained=True)`  
   - Ak MIL-Lab nemá tvoj encoder, použi standalone: `create_model('abmil.base.uni.none', in_dim=feat_dim, num_classes=6)` (alebo rovno `ABMILModel(in_dim=feat_dim, num_classes=6)`).

4. **Forward:** MIL-Lab očakáva `(batch_size, num_patches, feature_dim)`. Z našej dávky máme `feats` [B, N_max, D] a `mask`. Väčšina modelov v MIL-Lab nemá parameter pre mask; môžeš poslať len `feats` (padding sú nuly) alebo poslať po jednom slide v dávke (bez paddingu). Príklad s jednou dávkou:
   ```python
   results_dict, log_dict = model(feats.to(device), loss_fn=criterion, label=labels.to(device), return_attention=True)
   logits = results_dict["logits"]
   loss = results_dict["loss"]
   ```

5. **Tréning:** Rovnaký loop – `opt.zero_grad()`, `loss.backward()`, `opt.step()`; na validáciu volaj model bez `loss_fn`/`label` a ber `results_dict["logits"]`.

Ak chceš len vymeniť architektúru (TransMIL, CLAM, …), ponechaj `feat_dir`, loadery a dáta; v bunke „DataLoaders and model“ namiesto `AttentionMIL(...)` zavolaj `create_model(...)` a v tréningovej bunke volaj `model(feats, loss_fn=criterion, label=labels)` a použij `results_dict["logits"]` a `results_dict["loss"]`.

---

## 8. Conventions when editing

- Prefer **Path** over string paths; use `str(path)` only where needed (e.g. OpenSlide, h5py).
- **train_labels** is indexed by `image_id`; use `train_labels.loc[slide_id, "isup_grade"]` and check `slide_id in train_labels.index` when unsure.
- **Splits:** `splits["fold"]` 0–4; typical use: one fold as val, rest as train; filter slide lists to `available_h5` (slides that have a `.h5` in `feat_dir`).
- Keep notebook runnable in order: MIL_ROOT → imports → data_dir / train_labels → splits → feat_dir → train_ids/val_ids → Dataset/collate/model → training loop.
- Requirements: in `requirements-notebook.txt` and in the notebook’s `%pip install` cell (openslide, pandas, h5py, torch, scikit-learn, etc.).

---

## 9. Common tasks

- **Add a new metric (e.g. Cohen’s kappa):** After gathering `all_pred` and `all_label` in the val section, use `sklearn.metrics.cohen_kappa_score(all_label.numpy(), all_pred.numpy(), weights='quadratic')`.
- **Save best model:** Track best val metric (e.g. balanced accuracy), then `torch.save(model.state_dict(), MIL_ROOT / "checkpoints" / "mil_best.pt")`.
- **5-fold CV:** Loop over `fold` in 0..4; for each fold set `val_ids = splits[splits["fold"]==fold]["image_id"]`, `train_ids = splits[splits["fold"]!=fold]["image_id"]`, then build loaders and train.
- **Use a different feature key:** Pass `key="..."` to `H5BagDataset`; ensure collate and model receive the same feature dimension.

---

## 10. How to design / propose MIL algorithms

When asked to come up with or improve the MIL algorithm, follow this structure.

**Problem framing**
- One **bag** = one slide (one `.h5` = one set of patch features).
- **Instances** = patches; each instance has a feature vector `[feat_dim]`.
- Only the **bag label** (slide-level ISUP 0–5) is observed; patch-level labels are unknown.
- Goal: map bag of variable-size instances → single slide-level prediction.

**Aggregation options (bag → single vector)**
- **Mean pooling:** `bag = feats.mean(dim=1)`. Simple, no parameters; ignores which patches matter.
- **Max pooling:** `bag = feats.max(dim=1).values`. Emphasizes “hottest” patch per dimension; can be noisy.
- **Attention pooling (current):** Learn weights `a = softmax(MLP(feats))`, then `bag = (a.unsqueeze(-1) * feats).sum(dim=1)`. Use mask for variable-length bags. Interpretable (attention = importance per patch).
- **Multiple-instance attention (e.g. Ilse et al.):** Same idea with gated or two-branch attention.
- **Transformer / self-attention:** Encode patches with positional or no positional encoding, then [CLS]-token or mean over patch tokens as bag representation. Heavier; use when more capacity is needed.
- **Chunked / hierarchical:** Split bag into chunks (e.g. by spatial region if coords available), aggregate within chunk, then aggregate chunk embeddings. Use when slide is very large or structure matters.

**Head and loss**
- **Classification:** Linear(bag_dim, 6) + CrossEntropyLoss. Standard for ISUP 0–5.
- **Ordinal:** Treat ISUP as ordered; options: cumulative link (e.g. CORAL), or regression to 0–5 with rounding and optional MSE + classification auxiliary. Use when respecting order (0 < 1 < … < 5) should help.
- **Multi-task:** e.g. ISUP + Gleason components if Gleason is useful; share encoder, separate heads.

**Evaluation**
- **Accuracy** and **balanced accuracy** (important for imbalanced ISUP).
- **Quadratic-weighted Cohen’s kappa** (ordinal agreement).
- **AUC** per one-vs-rest or macro; report mean if useful.
- For ordinal: consider **MAE** (mean absolute error) of predicted vs true grade.

**When proposing a new algorithm**
1. State the **aggregation** (e.g. “attention” or “transformer”) and the **head/loss** (e.g. “6-class CE” or “CORAL ordinal”).
2. Describe the **forward pass** in words (patch encoder → aggregation → head → logits/loss).
3. Keep **masking** for variable-length bags (padding + mask in attention/transformer).
4. Suggest **minimal code changes**: e.g. swap `AttentionMIL` for a new class, or add an optional ordinal loss; keep `H5BagDataset` and `collate_bags` unchanged unless the new method needs different inputs (e.g. coordinates).
5. If the user has no preference, **default suggestion:** keep attention pooling; add optional ordinal loss (CORAL or similar) and report kappa and balanced accuracy; optionally add a second head for Gleason if in `train_labels`.

---

*End of context. Use this when answering questions or generating code for the MIL project.*
