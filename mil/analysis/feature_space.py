"""Cross-encoder feature-space analysis for patch-level .h5 embeddings.

This module compares encoder spaces at the representation level (geometry/alignment/usefulness),
not at the level of individual embedding dimensions.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
from sklearn.cross_decomposition import CCA
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

try:
    import matplotlib.pyplot as plt
except ImportError as exc:  # pragma: no cover - runtime environment dependent
    raise RuntimeError(
        "matplotlib is required for feature-space plotting. Install it via `pip install matplotlib`."
    ) from exc


def _safe_mkdir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def _l2_normalize(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    denom = np.linalg.norm(x, axis=1, keepdims=True)
    denom = np.clip(denom, eps, None)
    return x / denom


def _row_cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aa = _l2_normalize(a)
    bb = _l2_normalize(b)
    return np.sum(aa * bb, axis=1)


def _df_to_md_table(df: pd.DataFrame, max_rows: int = 20) -> str:
    if df.empty:
        return "_No rows._"
    head = df.head(max_rows)
    cols = list(head.columns)
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join("---" for _ in cols) + " |",
    ]
    for _, row in head.iterrows():
        vals = []
        for col in cols:
            val = row[col]
            if isinstance(val, float):
                vals.append(f"{val:.6g}")
            else:
                vals.append(str(val))
        lines.append("| " + " | ".join(vals) + " |")
    if len(df) > len(head):
        lines.append(f"\n_... {len(df) - len(head)} more rows in CSV artifact._")
    return "\n".join(lines)


def _find_h5_slides(feat_dir: Path) -> set[str]:
    if not feat_dir.exists():
        return set()
    return {p.stem for p in feat_dir.glob("*.h5")}


def _coords_to_keys(coords: np.ndarray) -> np.ndarray:
    """Convert coords array to hashable xy keys used for patch matching."""
    if coords.ndim != 2 or coords.shape[1] < 2:
        raise ValueError(f"Expected coords [N, >=2], got shape {coords.shape}")
    xy = coords[:, :2]
    # TRIDENT coords are typically integer pixels (often saved as float/int64). Round for safety.
    xy = np.rint(xy).astype(np.int64, copy=False)
    return xy


def _common_coord_indices(a_coords: np.ndarray, b_coords: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    a_xy = _coords_to_keys(a_coords)
    b_xy = _coords_to_keys(b_coords)
    b_map = {(int(x), int(y)): i for i, (x, y) in enumerate(b_xy)}
    a_idx: list[int] = []
    b_idx: list[int] = []
    for i, (x, y) in enumerate(a_xy):
        j = b_map.get((int(x), int(y)))
        if j is not None:
            a_idx.append(i)
            b_idx.append(j)
    return np.asarray(a_idx, dtype=np.int64), np.asarray(b_idx, dtype=np.int64)


def _sample_idx(n: int, max_n: int | None, rng: np.random.Generator) -> np.ndarray:
    if max_n is None or max_n >= n:
        return np.arange(n, dtype=np.int64)
    return rng.choice(n, size=max_n, replace=False)


def linear_cka(x: np.ndarray, y: np.ndarray) -> float:
    """Linear CKA for paired samples.

    x: [n, d1], y: [n, d2]
    """
    if x.shape[0] != y.shape[0]:
        raise ValueError("CKA requires same sample count")
    if x.shape[0] < 2:
        return float("nan")
    xc = x - x.mean(axis=0, keepdims=True)
    yc = y - y.mean(axis=0, keepdims=True)
    hsic = np.linalg.norm(xc.T @ yc, ord="fro") ** 2
    x_norm = np.linalg.norm(xc.T @ xc, ord="fro")
    y_norm = np.linalg.norm(yc.T @ yc, ord="fro")
    denom = x_norm * y_norm
    if denom <= 0:
        return float("nan")
    return float(hsic / denom)


def svcca_score(
    x: np.ndarray,
    y: np.ndarray,
    variance_threshold: float = 0.99,
    max_components: int = 64,
    seed: int = 42,
) -> float:
    """Approximate SVCCA with PCA truncation + linear CCA."""
    if x.shape[0] != y.shape[0] or x.shape[0] < 4:
        return float("nan")
    n = x.shape[0]
    x_max = max(1, min(max_components, x.shape[1], n - 1))
    y_max = max(1, min(max_components, y.shape[1], n - 1))
    x_pca = PCA(n_components=x_max, random_state=seed).fit(x)
    y_pca = PCA(n_components=y_max, random_state=seed).fit(y)

    x_cum = np.cumsum(x_pca.explained_variance_ratio_)
    y_cum = np.cumsum(y_pca.explained_variance_ratio_)
    x_k = int(np.searchsorted(x_cum, variance_threshold) + 1)
    y_k = int(np.searchsorted(y_cum, variance_threshold) + 1)
    k = max(1, min(x_k, y_k, x_max, y_max, n - 1))
    if k < 1:
        return float("nan")

    x_r = x_pca.transform(x)[:, :k]
    y_r = y_pca.transform(y)[:, :k]
    cca = CCA(n_components=k, max_iter=1000)
    x_c, y_c = cca.fit_transform(x_r, y_r)
    corr = []
    for i in range(k):
        xi = x_c[:, i]
        yi = y_c[:, i]
        sx = xi.std()
        sy = yi.std()
        if sx <= 0 or sy <= 0:
            continue
        corr.append(np.corrcoef(xi, yi)[0, 1])
    if not corr:
        return float("nan")
    return float(np.nanmean(corr))


@dataclass(frozen=True)
class EncoderInput:
    """Input declaration for one encoder's patch feature directory."""

    name: str
    feat_dir: Path
    feature_key: str = "features"
    coords_key: str = "coords"


@dataclass
class AnalysisConfig:
    """Runtime configuration for feature-space analysis."""

    out_dir: Path
    encoders: list[EncoderInput]
    seed: int = 42
    min_present_encoders: int | None = None
    per_slide_patch_cap: int = 512
    max_total_pairs: int = 120_000
    use_coords_matching: bool = True
    pairwise_projection: str = "pca"  # one of: none, pca
    pairwise_projection_dim: int = 256
    viz_methods: tuple[str, ...] = ("pca", "umap")
    viz_points: int = 20_000
    viz_color_by: tuple[str, ...] = ("slide", "label")
    labels_csv: Path | None = None
    labels_id_col: str = "image_id"
    labels_target_col: str = "isup_grade"
    run_probe: bool = True
    probe_model: str = "logreg"  # logreg or mlp
    probe_folds: int = 5
    run_svcca: bool = True
    run_alignment: bool = True
    alignment_train_fraction: float = 0.8
    alignment_alpha: float = 1.0
    run_spatial: bool = True
    spatial_k: int = 5
    spatial_patch_cap: int = 256
    spatial_pair_sample: int = 2000

    @staticmethod
    def from_mapping(values: dict[str, Any], root_dir: Path) -> "AnalysisConfig":
        out_dir = Path(values.get("out_dir", "results/feature_space_analysis"))
        if not out_dir.is_absolute():
            out_dir = root_dir / out_dir
        raw_encoders = values.get("encoders")
        if not isinstance(raw_encoders, list) or not raw_encoders:
            raise ValueError("Config must define non-empty 'encoders' list")
        encoders: list[EncoderInput] = []
        for item in raw_encoders:
            if not isinstance(item, dict):
                raise ValueError("Each encoder entry must be a mapping")
            name = str(item["name"])
            feat_dir = Path(item["feat_dir"])
            if not feat_dir.is_absolute():
                feat_dir = root_dir / feat_dir
            encoders.append(
                EncoderInput(
                    name=name,
                    feat_dir=feat_dir,
                    feature_key=str(item.get("feature_key", "features")),
                    coords_key=str(item.get("coords_key", "coords")),
                )
            )
        labels_csv = values.get("labels_csv")
        if labels_csv:
            labels_csv = Path(labels_csv)
            if not labels_csv.is_absolute():
                labels_csv = root_dir / labels_csv
        viz_methods = tuple(values.get("viz_methods", ("pca", "umap")))
        viz_color_by = tuple(values.get("viz_color_by", ("slide", "label")))
        cfg = AnalysisConfig(
            out_dir=out_dir,
            encoders=encoders,
            seed=int(values.get("seed", 42)),
            min_present_encoders=values.get("min_present_encoders"),
            per_slide_patch_cap=int(values.get("per_slide_patch_cap", 512)),
            max_total_pairs=int(values.get("max_total_pairs", 120_000)),
            use_coords_matching=bool(values.get("use_coords_matching", True)),
            pairwise_projection=str(values.get("pairwise_projection", "pca")),
            pairwise_projection_dim=int(values.get("pairwise_projection_dim", 256)),
            viz_methods=viz_methods,
            viz_points=int(values.get("viz_points", 20_000)),
            viz_color_by=viz_color_by,
            labels_csv=labels_csv,
            labels_id_col=str(values.get("labels_id_col", "image_id")),
            labels_target_col=str(values.get("labels_target_col", "isup_grade")),
            run_probe=bool(values.get("run_probe", True)),
            probe_model=str(values.get("probe_model", "logreg")),
            probe_folds=int(values.get("probe_folds", 5)),
            run_svcca=bool(values.get("run_svcca", True)),
            run_alignment=bool(values.get("run_alignment", True)),
            alignment_train_fraction=float(values.get("alignment_train_fraction", 0.8)),
            alignment_alpha=float(values.get("alignment_alpha", 1.0)),
            run_spatial=bool(values.get("run_spatial", True)),
            spatial_k=int(values.get("spatial_k", 5)),
            spatial_patch_cap=int(values.get("spatial_patch_cap", 256)),
            spatial_pair_sample=int(values.get("spatial_pair_sample", 2000)),
        )
        return cfg


class MultiEncoderFeatureLoader:
    """Robust loader for multi-encoder patch feature spaces."""

    def __init__(self, encoders: list[EncoderInput]):
        self.encoders = {e.name: e for e in encoders}
        self.available_slides = {name: _find_h5_slides(enc.feat_dir) for name, enc in self.encoders.items()}

    def encoder_names(self) -> list[str]:
        return list(self.encoders.keys())

    def matched_slide_ids(self, min_present: int | None = None) -> list[str]:
        if not self.available_slides:
            return []
        names = list(self.available_slides.keys())
        if min_present is None:
            inter = set(self.available_slides[names[0]])
            for name in names[1:]:
                inter &= self.available_slides[name]
            return sorted(inter)
        counts: dict[str, int] = {}
        for slides in self.available_slides.values():
            for sid in slides:
                counts[sid] = counts.get(sid, 0) + 1
        return sorted([sid for sid, c in counts.items() if c >= min_present])

    def load_slide(self, encoder_name: str, slide_id: str) -> tuple[np.ndarray, np.ndarray | None]:
        enc = self.encoders[encoder_name]
        path = enc.feat_dir / f"{slide_id}.h5"
        if not path.exists():
            raise FileNotFoundError(path)
        with h5py.File(path, "r") as f:
            if enc.feature_key not in f:
                raise KeyError(f"Missing key {enc.feature_key!r} in {path}")
            feat = np.asarray(f[enc.feature_key][:], dtype=np.float32)
            coords = None
            if enc.coords_key in f:
                coords = np.asarray(f[enc.coords_key][:], dtype=np.float32)
        return feat, coords

    def load_slide_mean_feature(self, encoder_name: str, slide_id: str, patch_cap: int, seed: int) -> np.ndarray:
        feat, _ = self.load_slide(encoder_name, slide_id)
        if feat.size == 0:
            raise ValueError(f"Empty feature array for {encoder_name}/{slide_id}")
        rng = _rng(seed)
        idx = _sample_idx(feat.shape[0], patch_cap, rng)
        return feat[idx].mean(axis=0)

    def matched_pair_arrays(
        self,
        encoder_a: str,
        encoder_b: str,
        slide_ids: list[str],
        per_slide_cap: int,
        max_total: int,
        seed: int,
        use_coords: bool,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        rng = _rng(seed)
        xa_parts: list[np.ndarray] = []
        xb_parts: list[np.ndarray] = []
        per_slide_rows: list[dict[str, Any]] = []
        total = 0
        for sid in slide_ids:
            try:
                fa, ca = self.load_slide(encoder_a, sid)
                fb, cb = self.load_slide(encoder_b, sid)
            except Exception:
                continue
            if use_coords and ca is not None and cb is not None:
                ia, ib = _common_coord_indices(ca, cb)
                match_mode = "coords"
            else:
                n = min(fa.shape[0], fb.shape[0])
                ia = np.arange(n, dtype=np.int64)
                ib = np.arange(n, dtype=np.int64)
                match_mode = "index"
            n_match = min(len(ia), len(ib))
            if n_match == 0:
                continue
            if per_slide_cap > 0 and n_match > per_slide_cap:
                choice = rng.choice(n_match, size=per_slide_cap, replace=False)
                ia = ia[choice]
                ib = ib[choice]
                n_match = per_slide_cap
            xa_parts.append(fa[ia])
            xb_parts.append(fb[ib])
            total += n_match
            per_slide_rows.append({"slide_id": sid, "matched_patches": n_match, "match_mode": match_mode})
            if total >= max_total:
                break
        if not xa_parts:
            return (
                np.empty((0, 0), dtype=np.float32),
                np.empty((0, 0), dtype=np.float32),
                {"slides_used": 0, "patches_used": 0, "match_mode": "none"},
            )
        xa = np.concatenate(xa_parts, axis=0)
        xb = np.concatenate(xb_parts, axis=0)
        if xa.shape[0] > max_total:
            choice = rng.choice(xa.shape[0], size=max_total, replace=False)
            xa = xa[choice]
            xb = xb[choice]
        mode_counts = pd.DataFrame(per_slide_rows)["match_mode"].value_counts().to_dict()
        meta = {
            "slides_used": len(per_slide_rows),
            "patches_used": int(xa.shape[0]),
            "match_mode_counts": mode_counts,
        }
        return xa, xb, meta


def _pairwise_cosine_stats(
    x: np.ndarray,
    y: np.ndarray,
    projection: str,
    projection_dim: int,
    seed: int,
) -> dict[str, Any]:
    if x.shape[0] == 0:
        return {"cosine_mean": np.nan, "cosine_std": np.nan, "cosine_p05": np.nan, "cosine_p95": np.nan}
    x_use = x
    y_use = y
    if projection == "none":
        if x.shape[1] != y.shape[1]:
            return {"cosine_mean": np.nan, "cosine_std": np.nan, "cosine_p05": np.nan, "cosine_p95": np.nan}
    elif projection == "pca":
        k = min(x.shape[1], y.shape[1], projection_dim, x.shape[0] - 1)
        if k < 2:
            return {"cosine_mean": np.nan, "cosine_std": np.nan, "cosine_p05": np.nan, "cosine_p95": np.nan}
        x_use = PCA(n_components=k, random_state=seed).fit_transform(x)
        y_use = PCA(n_components=k, random_state=seed).fit_transform(y)
    else:
        raise ValueError(f"Unknown projection mode: {projection}")
    cos = _row_cosine(x_use, y_use)
    return {
        "cosine_mean": float(np.mean(cos)),
        "cosine_std": float(np.std(cos)),
        "cosine_p05": float(np.quantile(cos, 0.05)),
        "cosine_p95": float(np.quantile(cos, 0.95)),
    }


def _reduce_2d(x: np.ndarray, method: str, seed: int) -> tuple[np.ndarray, str]:
    if method == "pca":
        return PCA(n_components=2, random_state=seed).fit_transform(x), "pca"
    if method == "umap":
        try:
            import umap
        except ImportError:
            method = "tsne"
        else:
            reducer = umap.UMAP(n_components=2, random_state=seed, n_neighbors=20, min_dist=0.1)
            return reducer.fit_transform(x), "umap"
    if method == "tsne":
        perplexity = min(30, max(5, (x.shape[0] - 1) // 3))
        emb = TSNE(n_components=2, random_state=seed, init="pca", perplexity=perplexity).fit_transform(x)
        return emb, "tsne"
    raise ValueError(f"Unknown reduction method: {method}")


def _plot_scatter(
    points: np.ndarray,
    title: str,
    out_path: Path,
    color_values: np.ndarray | None = None,
    color_label: str | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 6))
    if color_values is None:
        ax.scatter(points[:, 0], points[:, 1], s=5, alpha=0.5)
    else:
        uniq = np.unique(color_values)
        if len(uniq) > 20:
            ax.scatter(points[:, 0], points[:, 1], s=5, alpha=0.5)
            if color_label:
                ax.set_xlabel(f"Color labels hidden ({len(uniq)} unique {color_label})")
        else:
            for value in uniq:
                mask = color_values == value
                ax.scatter(points[mask, 0], points[mask, 1], s=6, alpha=0.65, label=str(value))
            ax.legend(fontsize=7, markerscale=2, loc="best")
    ax.set_title(title)
    ax.set_xlabel("component_1")
    ax.set_ylabel("component_2")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _load_labels(path: Path, id_col: str, target_col: str) -> pd.Series:
    df = pd.read_csv(path)
    if id_col not in df.columns or target_col not in df.columns:
        raise KeyError(f"labels_csv must include columns {id_col!r} and {target_col!r}")
    out = df[[id_col, target_col]].copy()
    out[id_col] = out[id_col].astype(str)
    out = out.dropna()
    return out.set_index(id_col)[target_col]


def _encoder_basic_stats(
    loader: MultiEncoderFeatureLoader,
    slide_ids: list[str],
    per_slide_cap: int,
    seed: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    rng = _rng(seed)
    for name in loader.encoder_names():
        feat_dim: int | None = None
        patches_per_slide: list[int] = []
        norms: list[np.ndarray] = []
        slides_ok = 0
        for sid in slide_ids:
            if sid not in loader.available_slides[name]:
                continue
            try:
                feat, _ = loader.load_slide(name, sid)
            except Exception:
                continue
            if feat_dim is None:
                feat_dim = int(feat.shape[1])
            idx = _sample_idx(feat.shape[0], per_slide_cap, rng)
            sub = feat[idx]
            patches_per_slide.append(int(sub.shape[0]))
            norms.append(np.linalg.norm(sub, axis=1))
            slides_ok += 1
        norm_concat = np.concatenate(norms, axis=0) if norms else np.array([], dtype=np.float32)
        rows.append(
            {
                "encoder": name,
                "feat_dim": feat_dim if feat_dim is not None else np.nan,
                "slides_available": len(loader.available_slides[name]),
                "slides_analyzed": slides_ok,
                "patches_per_slide_mean": float(np.mean(patches_per_slide)) if patches_per_slide else np.nan,
                "patches_per_slide_std": float(np.std(patches_per_slide)) if patches_per_slide else np.nan,
                "feature_norm_mean": float(np.mean(norm_concat)) if norm_concat.size else np.nan,
                "feature_norm_std": float(np.std(norm_concat)) if norm_concat.size else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _pairwise_space_similarity(
    loader: MultiEncoderFeatureLoader,
    slide_ids: list[str],
    cfg: AnalysisConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    names = loader.encoder_names()
    pair_rows: list[dict[str, Any]] = []
    cka_mat = pd.DataFrame(np.eye(len(names)), index=names, columns=names, dtype=float)
    svcca_mat = pd.DataFrame(np.eye(len(names)), index=names, columns=names, dtype=float)
    for i, a in enumerate(names):
        for j in range(i + 1, len(names)):
            b = names[j]
            x, y, meta = loader.matched_pair_arrays(
                a,
                b,
                slide_ids,
                per_slide_cap=cfg.per_slide_patch_cap,
                max_total=cfg.max_total_pairs,
                seed=cfg.seed + i * 1000 + j,
                use_coords=cfg.use_coords_matching,
            )
            if x.shape[0] == 0:
                cka = np.nan
                sv = np.nan
                cos_stats = _pairwise_cosine_stats(x, y, cfg.pairwise_projection, cfg.pairwise_projection_dim, cfg.seed)
            else:
                cka = linear_cka(x, y)
                sv = (
                    svcca_score(x, y, max_components=min(cfg.pairwise_projection_dim, 128), seed=cfg.seed)
                    if cfg.run_svcca
                    else np.nan
                )
                cos_stats = _pairwise_cosine_stats(x, y, cfg.pairwise_projection, cfg.pairwise_projection_dim, cfg.seed)
            cka_mat.loc[a, b] = cka_mat.loc[b, a] = cka
            svcca_mat.loc[a, b] = svcca_mat.loc[b, a] = sv
            row = {
                "encoder_a": a,
                "encoder_b": b,
                "matched_slides_used": meta.get("slides_used", 0),
                "matched_patches_used": meta.get("patches_used", 0),
                "match_mode_counts": str(meta.get("match_mode_counts", {})),
                "linear_cka": cka,
                "svcca": sv,
                **cos_stats,
            }
            pair_rows.append(row)
    return pd.DataFrame(pair_rows), cka_mat, svcca_mat


def _build_joint_sample_for_viz(
    loader: MultiEncoderFeatureLoader,
    slide_ids: list[str],
    cfg: AnalysisConfig,
    labels: pd.Series | None,
) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    names = loader.encoder_names()
    rng = _rng(cfg.seed + 123)
    enc_parts: dict[str, list[np.ndarray]] = {name: [] for name in names}
    meta_rows: list[dict[str, Any]] = []
    total = 0
    for sid in slide_ids:
        slides_data: dict[str, tuple[np.ndarray, np.ndarray | None]] = {}
        ok = True
        for name in names:
            if sid not in loader.available_slides[name]:
                ok = False
                break
            try:
                slides_data[name] = loader.load_slide(name, sid)
            except Exception:
                ok = False
                break
        if not ok:
            continue
        if cfg.use_coords_matching and all(v[1] is not None for v in slides_data.values()):
            common: set[tuple[int, int]] | None = None
            idx_map: dict[str, dict[tuple[int, int], int]] = {}
            for name, (_, coord) in slides_data.items():
                assert coord is not None
                xy = _coords_to_keys(coord)
                mapping = {(int(x), int(y)): i for i, (x, y) in enumerate(xy)}
                idx_map[name] = mapping
                keys = set(mapping.keys())
                common = keys if common is None else (common & keys)
            common_keys = list(common or [])
            if not common_keys:
                continue
            if len(common_keys) > cfg.per_slide_patch_cap:
                chosen = rng.choice(len(common_keys), size=cfg.per_slide_patch_cap, replace=False)
                common_keys = [common_keys[i] for i in chosen]
            for name in names:
                feat = slides_data[name][0]
                idx = np.asarray([idx_map[name][k] for k in common_keys], dtype=np.int64)
                enc_parts[name].append(feat[idx])
            n = len(common_keys)
        else:
            n = min(slides_data[name][0].shape[0] for name in names)
            if n == 0:
                continue
            idx = _sample_idx(n, cfg.per_slide_patch_cap, rng)
            for name in names:
                enc_parts[name].append(slides_data[name][0][idx])
            n = len(idx)
        label = labels.get(sid) if labels is not None and sid in labels.index else np.nan
        meta_rows.extend([{"slide_id": sid, "label": label}] * n)
        total += n
        if total >= cfg.viz_points:
            break
    if not meta_rows:
        return {}, pd.DataFrame(columns=["slide_id", "label"])
    enc = {name: np.concatenate(parts, axis=0) for name, parts in enc_parts.items() if parts}
    meta = pd.DataFrame(meta_rows)
    if len(meta) > cfg.viz_points:
        choose = rng.choice(len(meta), size=cfg.viz_points, replace=False)
        meta = meta.iloc[choose].reset_index(drop=True)
        for name in enc:
            enc[name] = enc[name][choose]
    return enc, meta


def _make_projection_plots(
    features: dict[str, np.ndarray],
    meta: pd.DataFrame,
    cfg: AnalysisConfig,
    out_dir: Path,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if not features or meta.empty:
        return pd.DataFrame(rows)
    for enc_name, x in features.items():
        x_use = x
        if x_use.shape[0] > cfg.viz_points:
            x_use = x_use[: cfg.viz_points]
        for method in cfg.viz_methods:
            try:
                emb2d, method_resolved = _reduce_2d(x_use, method, cfg.seed)
            except Exception as exc:
                rows.append(
                    {
                        "encoder": enc_name,
                        "requested_method": method,
                        "resolved_method": "error",
                        "plot_path": "",
                        "status": f"failed: {exc}",
                    }
                )
                continue
            base = f"{enc_name}_{method_resolved}"
            out_plain = out_dir / f"{base}_plain.png"
            _plot_scatter(emb2d, f"{enc_name} - {method_resolved.upper()} (no color)", out_plain)
            rows.append(
                {
                    "encoder": enc_name,
                    "requested_method": method,
                    "resolved_method": method_resolved,
                    "plot_path": str(out_plain),
                    "status": "ok",
                }
            )
            if "slide" in cfg.viz_color_by:
                out_slide = out_dir / f"{base}_by_slide.png"
                _plot_scatter(
                    emb2d,
                    f"{enc_name} - {method_resolved.upper()} (color: slide_id)",
                    out_slide,
                    color_values=meta["slide_id"].astype(str).to_numpy(),
                    color_label="slide_id",
                )
                rows.append(
                    {
                        "encoder": enc_name,
                        "requested_method": method,
                        "resolved_method": method_resolved,
                        "plot_path": str(out_slide),
                        "status": "ok",
                    }
                )
            if "label" in cfg.viz_color_by and "label" in meta:
                valid = meta["label"].notna().to_numpy()
                if valid.any():
                    out_label = out_dir / f"{base}_by_label.png"
                    _plot_scatter(
                        emb2d[valid],
                        f"{enc_name} - {method_resolved.upper()} (color: label)",
                        out_label,
                        color_values=meta.loc[valid, "label"].astype(str).to_numpy(),
                        color_label="label",
                    )
                    rows.append(
                        {
                            "encoder": enc_name,
                            "requested_method": method,
                            "resolved_method": method_resolved,
                            "plot_path": str(out_label),
                            "status": "ok",
                        }
                    )
    return pd.DataFrame(rows)


def _probe_results(
    loader: MultiEncoderFeatureLoader,
    slide_ids: list[str],
    labels: pd.Series,
    cfg: AnalysisConfig,
) -> pd.DataFrame:
    names = loader.encoder_names()
    common_slides = [sid for sid in slide_ids if sid in labels.index]
    if not common_slides:
        return pd.DataFrame()
    per_encoder_embed: dict[str, np.ndarray] = {}
    kept_slides: list[str] = []
    for sid in common_slides:
        ok = True
        for name in names:
            if sid not in loader.available_slides[name]:
                ok = False
                break
        if not ok:
            continue
        kept_slides.append(sid)
    if not kept_slides:
        return pd.DataFrame()

    for i, name in enumerate(names):
        vecs = []
        for sid in kept_slides:
            vecs.append(loader.load_slide_mean_feature(name, sid, cfg.per_slide_patch_cap, cfg.seed + i))
        per_encoder_embed[name] = np.stack(vecs, axis=0)

    y = labels.loc[kept_slides].astype(int).to_numpy()
    uniq, counts = np.unique(y, return_counts=True)
    min_count = int(counts.min()) if len(counts) else 0
    n_splits = max(2, min(cfg.probe_folds, len(y), min_count))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=cfg.seed)

    def build_model():
        if cfg.probe_model == "mlp":
            model = MLPClassifier(
                hidden_layer_sizes=(128,),
                activation="relu",
                alpha=1e-4,
                max_iter=300,
                random_state=cfg.seed,
            )
            return make_pipeline(StandardScaler(), model)
        model = LogisticRegression(max_iter=1000, multi_class="auto", random_state=cfg.seed)
        return make_pipeline(StandardScaler(), model)

    rows: list[dict[str, Any]] = []
    fused = np.concatenate([per_encoder_embed[name] for name in names], axis=1)
    for enc_name, x in {**per_encoder_embed, "fused": fused}.items():
        accs = []
        f1s = []
        for train_idx, test_idx in skf.split(x, y):
            model = build_model()
            model.fit(x[train_idx], y[train_idx])
            pred = model.predict(x[test_idx])
            accs.append(accuracy_score(y[test_idx], pred))
            f1s.append(f1_score(y[test_idx], pred, average="macro"))
        rows.append(
            {
                "encoder": enc_name,
                "probe_model": cfg.probe_model,
                "slides_used": len(y),
                "cv_folds": n_splits,
                "accuracy_mean": float(np.mean(accs)),
                "accuracy_std": float(np.std(accs)),
                "macro_f1_mean": float(np.mean(f1s)),
                "macro_f1_std": float(np.std(f1s)),
            }
        )
    return pd.DataFrame(rows)


def _alignment_results(
    loader: MultiEncoderFeatureLoader,
    slide_ids: list[str],
    cfg: AnalysisConfig,
) -> pd.DataFrame:
    names = loader.encoder_names()
    rows: list[dict[str, Any]] = []
    for i, src in enumerate(names):
        for j, tgt in enumerate(names):
            if src == tgt:
                continue
            x, y, meta = loader.matched_pair_arrays(
                src,
                tgt,
                slide_ids,
                per_slide_cap=cfg.per_slide_patch_cap,
                max_total=cfg.max_total_pairs,
                seed=cfg.seed + i * 101 + j,
                use_coords=cfg.use_coords_matching,
            )
            if x.shape[0] < 16:
                rows.append(
                    {
                        "source_encoder": src,
                        "target_encoder": tgt,
                        "samples": int(x.shape[0]),
                        "slides_used": meta.get("slides_used", 0),
                        "mse": np.nan,
                        "r2": np.nan,
                        "cosine_mean": np.nan,
                    }
                )
                continue
            x_train, x_test, y_train, y_test = train_test_split(
                x,
                y,
                train_size=cfg.alignment_train_fraction,
                random_state=cfg.seed,
                shuffle=True,
            )
            model = Ridge(alpha=cfg.alignment_alpha, random_state=cfg.seed)
            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)
            cos = _row_cosine(y_pred, y_test)
            rows.append(
                {
                    "source_encoder": src,
                    "target_encoder": tgt,
                    "samples": int(x.shape[0]),
                    "slides_used": meta.get("slides_used", 0),
                    "mse": float(mean_squared_error(y_test, y_pred)),
                    "r2": float(r2_score(y_test, y_pred, multioutput="variance_weighted")),
                    "cosine_mean": float(np.mean(cos)),
                }
            )
    return pd.DataFrame(rows)


def _spatial_smoothness(
    loader: MultiEncoderFeatureLoader,
    slide_ids: list[str],
    cfg: AnalysisConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Per-encoder: are spatial neighbors feature-neighbors?
    enc_rows: list[dict[str, Any]] = []
    # Across encoders: neighborhood preservation in feature space.
    cross_rows: list[dict[str, Any]] = []
    names = loader.encoder_names()
    rng = _rng(cfg.seed + 999)

    for name in names:
        deltas = []
        slides_used = 0
        for sid in slide_ids:
            try:
                feat, coords = loader.load_slide(name, sid)
            except Exception:
                continue
            if coords is None or feat.shape[0] < max(cfg.spatial_k + 2, 8):
                continue
            n = feat.shape[0]
            idx = _sample_idx(n, cfg.spatial_patch_cap, rng)
            f = feat[idx]
            c = coords[idx][:, :2]
            fn = _l2_normalize(f)
            feat_cos = fn @ fn.T
            nbr = NearestNeighbors(n_neighbors=min(cfg.spatial_k + 1, len(c))).fit(c)
            neigh_idx = nbr.kneighbors(return_distance=False)[:, 1:]
            neigh_vals = feat_cos[np.repeat(np.arange(len(c)), neigh_idx.shape[1]), neigh_idx.reshape(-1)]
            # Random baseline with same sample count.
            m = neigh_vals.shape[0]
            ri = rng.integers(0, len(c), size=m)
            rj = rng.integers(0, len(c), size=m)
            rand_vals = feat_cos[ri, rj]
            deltas.append(float(np.mean(neigh_vals) - np.mean(rand_vals)))
            slides_used += 1
        enc_rows.append(
            {
                "encoder": name,
                "slides_used": slides_used,
                "neighbor_cosine_minus_random_mean": float(np.mean(deltas)) if deltas else np.nan,
                "neighbor_cosine_minus_random_std": float(np.std(deltas)) if deltas else np.nan,
            }
        )

    for i, a in enumerate(names):
        for j in range(i + 1, len(names)):
            b = names[j]
            x, y, meta = loader.matched_pair_arrays(
                a,
                b,
                slide_ids,
                per_slide_cap=cfg.spatial_patch_cap,
                max_total=min(cfg.max_total_pairs, 30_000),
                seed=cfg.seed + i * 700 + j,
                use_coords=cfg.use_coords_matching,
            )
            if x.shape[0] < max(cfg.spatial_k + 2, 16):
                cross_rows.append(
                    {
                        "encoder_a": a,
                        "encoder_b": b,
                        "samples": int(x.shape[0]),
                        "slides_used": meta.get("slides_used", 0),
                        "knn_overlap_mean": np.nan,
                    }
                )
                continue
            xa = _l2_normalize(x)
            yb = _l2_normalize(y)
            k = min(cfg.spatial_k, x.shape[0] - 1)
            na = NearestNeighbors(n_neighbors=k + 1, metric="cosine").fit(xa).kneighbors(return_distance=False)[:, 1:]
            nb = NearestNeighbors(n_neighbors=k + 1, metric="cosine").fit(yb).kneighbors(return_distance=False)[:, 1:]
            overlap = []
            for n1, n2 in zip(na, nb):
                overlap.append(len(set(n1.tolist()) & set(n2.tolist())) / float(k))
            cross_rows.append(
                {
                    "encoder_a": a,
                    "encoder_b": b,
                    "samples": int(x.shape[0]),
                    "slides_used": meta.get("slides_used", 0),
                    "knn_overlap_mean": float(np.mean(overlap)),
                }
            )
    return pd.DataFrame(enc_rows), pd.DataFrame(cross_rows)


def _write_markdown_report(
    cfg: AnalysisConfig,
    out_dir: Path,
    matched_slides: list[str],
    basic_df: pd.DataFrame,
    pair_df: pd.DataFrame,
    cka_mat: pd.DataFrame,
    align_df: pd.DataFrame,
    probe_df: pd.DataFrame | None,
    viz_df: pd.DataFrame,
    spatial_df: pd.DataFrame | None,
    neigh_df: pd.DataFrame | None,
) -> Path:
    report = out_dir / "summary.md"
    lines: list[str] = []
    lines.append("# Feature-Space Comparison Report\n")
    lines.append("## Scope")
    lines.append(
        "- This analysis compares encoder feature spaces as **distributed representations** "
        "(geometry, similarity, alignment, and downstream utility)."
    )
    lines.append(
        "- It deliberately **does not interpret individual embedding dimensions** semantically."
    )
    lines.append("- Different embedding dimensions are handled via CKA/SVCCA and optional projection.")
    lines.append("")
    lines.append("## Configuration")
    lines.append(f"- Output directory: `{out_dir}`")
    lines.append(f"- Encoders: {', '.join(e.name for e in cfg.encoders)}")
    lines.append(f"- Matched slides used in analysis: {len(matched_slides)}")
    lines.append(f"- Per-slide patch cap: {cfg.per_slide_patch_cap}")
    lines.append(f"- Pairwise projection mode: `{cfg.pairwise_projection}`")
    lines.append("")
    lines.append("## Encoder Basic Statistics")
    lines.append(_df_to_md_table(basic_df))
    lines.append("")
    lines.append("## Pairwise Space Similarity")
    lines.append(_df_to_md_table(pair_df))
    lines.append("")
    lines.append("### Linear CKA Matrix")
    lines.append(_df_to_md_table(cka_mat.reset_index().rename(columns={"index": "encoder"})))
    lines.append("")
    lines.append("## Cross-space Linear Alignment (A -> B)")
    lines.append(_df_to_md_table(align_df))
    lines.append("")
    lines.append("## Projection Figures")
    lines.append(_df_to_md_table(viz_df[["encoder", "resolved_method", "plot_path", "status"]] if not viz_df.empty else viz_df))
    lines.append("")
    if probe_df is not None and not probe_df.empty:
        lines.append("## Functional Complementarity (Slide-level Probe)")
        lines.append(_df_to_md_table(probe_df))
        lines.append("")
    if spatial_df is not None and not spatial_df.empty:
        lines.append("## Spatial Smoothness (Optional)")
        lines.append(_df_to_md_table(spatial_df))
        lines.append("")
    if neigh_df is not None and not neigh_df.empty:
        lines.append("## Cross-encoder Neighborhood Preservation (Optional)")
        lines.append(_df_to_md_table(neigh_df))
        lines.append("")
    lines.append("## Interpretation Notes")
    lines.append(
        "- Higher CKA/SVCCA suggests stronger global representational similarity, but does not imply identical decision use."
    )
    lines.append(
        "- Strong linear alignment (high cosine / R², low MSE) suggests substantial linear mapability between spaces."
    )
    lines.append(
        "- Probe gains from fused features over single encoders indicate useful complementarity beyond redundancy."
    )
    lines.append(
        "- If coordinate matching is unavailable, index-based matching is an approximation and should be interpreted cautiously."
    )
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def run_feature_space_analysis(cfg: AnalysisConfig) -> dict[str, Path]:
    """Run end-to-end feature-space analysis and save artifacts."""
    print("[analysis] Starting feature-space analysis...", flush=True)
    out_dir = _safe_mkdir(cfg.out_dir)
    plot_dir = _safe_mkdir(out_dir / "plots")
    print(f"[analysis] Output directory: {out_dir}", flush=True)
    loader = MultiEncoderFeatureLoader(cfg.encoders)
    print(f"[analysis] Encoders: {', '.join(loader.encoder_names())}", flush=True)
    matched_slides = loader.matched_slide_ids(min_present=cfg.min_present_encoders)
    if not matched_slides:
        raise RuntimeError("No matched slides found across selected encoders")
    print(f"[analysis] Matched slides: {len(matched_slides)}", flush=True)

    labels = None
    if cfg.labels_csv is not None and cfg.labels_csv.exists():
        print(f"[analysis] Loading labels: {cfg.labels_csv}", flush=True)
        labels = _load_labels(cfg.labels_csv, cfg.labels_id_col, cfg.labels_target_col)
        print(f"[analysis] Labels loaded for {len(labels)} slides", flush=True)
    else:
        print("[analysis] Labels unavailable; probe analysis may be skipped", flush=True)

    print("[analysis] Computing basic encoder statistics...", flush=True)
    basic_df = _encoder_basic_stats(loader, matched_slides, per_slide_cap=cfg.per_slide_patch_cap, seed=cfg.seed)
    print("[analysis] Computing pairwise space similarity (CKA/SVCCA/cosine)...", flush=True)
    pair_df, cka_mat, svcca_mat = _pairwise_space_similarity(loader, matched_slides, cfg)
    print("[analysis] Building joint sample for projection plots...", flush=True)
    viz_features, viz_meta = _build_joint_sample_for_viz(loader, matched_slides, cfg, labels)
    print("[analysis] Rendering projection plots...", flush=True)
    viz_df = _make_projection_plots(viz_features, viz_meta, cfg, plot_dir)

    probe_df = pd.DataFrame()
    if cfg.run_probe and labels is not None:
        print("[analysis] Running functional complementarity probes...", flush=True)
        probe_df = _probe_results(loader, matched_slides, labels, cfg)
    elif cfg.run_probe:
        print("[analysis] Probe requested but skipped (labels not available)", flush=True)

    align_df = pd.DataFrame()
    if cfg.run_alignment:
        print("[analysis] Running cross-space linear alignment...", flush=True)
        align_df = _alignment_results(loader, matched_slides, cfg)

    spatial_df = pd.DataFrame()
    neigh_df = pd.DataFrame()
    if cfg.run_spatial:
        print("[analysis] Running spatial smoothness/neighborhood analyses...", flush=True)
        spatial_df, neigh_df = _spatial_smoothness(loader, matched_slides, cfg)

    basic_csv = out_dir / "encoder_basic_stats.csv"
    pair_csv = out_dir / "pairwise_space_similarity.csv"
    cka_csv = out_dir / "cka_matrix.csv"
    svcca_csv = out_dir / "svcca_matrix.csv"
    align_csv = out_dir / "alignment_results.csv"
    viz_csv = out_dir / "projection_plots.csv"
    probe_csv = out_dir / "probe_results.csv"
    spatial_csv = out_dir / "spatial_smoothness.csv"
    neigh_csv = out_dir / "neighborhood_preservation.csv"

    basic_df.to_csv(basic_csv, index=False)
    pair_df.to_csv(pair_csv, index=False)
    cka_mat.to_csv(cka_csv)
    svcca_mat.to_csv(svcca_csv)
    align_df.to_csv(align_csv, index=False)
    viz_df.to_csv(viz_csv, index=False)
    if not probe_df.empty:
        probe_df.to_csv(probe_csv, index=False)
    if not spatial_df.empty:
        spatial_df.to_csv(spatial_csv, index=False)
    if not neigh_df.empty:
        neigh_df.to_csv(neigh_csv, index=False)

    print("[analysis] Writing Markdown summary report...", flush=True)
    report_path = _write_markdown_report(
        cfg=cfg,
        out_dir=out_dir,
        matched_slides=matched_slides,
        basic_df=basic_df,
        pair_df=pair_df,
        cka_mat=cka_mat,
        align_df=align_df,
        probe_df=probe_df if not probe_df.empty else None,
        viz_df=viz_df,
        spatial_df=spatial_df if not spatial_df.empty else None,
        neigh_df=neigh_df if not neigh_df.empty else None,
    )
    print("[analysis] Done.", flush=True)
    return {
        "out_dir": out_dir,
        "report": report_path,
        "basic_csv": basic_csv,
        "pair_csv": pair_csv,
        "cka_csv": cka_csv,
        "svcca_csv": svcca_csv,
        "align_csv": align_csv,
        "viz_csv": viz_csv,
        "probe_csv": probe_csv,
        "spatial_csv": spatial_csv,
        "neigh_csv": neigh_csv,
    }
