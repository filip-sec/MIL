"""Optimized tiling with optional Numba JIT compilation."""

from tqdm import tqdm

from .io_utils import alpha_to_white

try:
    import numpy as np
    from numba import jit

    HAS_NUMBA = True
except (ImportError, ModuleNotFoundError):  # pragma: no cover - optional acceleration only
    HAS_NUMBA = False


if HAS_NUMBA:

    @jit(nopython=True)
    def _check_tissue_mask(mask_roi, strict):
        if mask_roi.size == 0:
            return False
        if strict:
            return bool(np.min(mask_roi) == 255)
        return bool(np.mean(mask_roi == 255) >= 0.95)


def tile_wsi_fast(slide, mask_thumb, tw_th, w0_h0, out_dir, slide_id, tile_px=512, strict=True):
    """Tile extraction with optional numba-accelerated mask check."""
    tw, th = tw_th
    width_l0, height_l0 = w0_h0
    sx, sy = tw / width_l0, th / height_l0
    out_dir.mkdir(parents=True, exist_ok=True)

    rows, saved = [], 0

    for y in tqdm(range(0, height_l0, tile_px), desc=slide_id, leave=False):
        for x in range(0, width_l0, tile_px):
            if x + tile_px > width_l0 or y + tile_px > height_l0:
                continue

            tx0, ty0, tx1, ty1 = (
                max(0, min(tw, int(x * sx))),
                max(0, min(th, int(y * sy))),
                max(0, min(tw, int((x + tile_px) * sx))),
                max(0, min(th, int((y + tile_px) * sy))),
            )
            roi = mask_thumb[ty0:ty1, tx0:tx1]

            if not roi.size:
                continue

            passes_tissue = (
                _check_tissue_mask(roi.flatten(), strict)
                if HAS_NUMBA
                else (roi.min() == 255 if strict else (roi == 255).mean() >= 0.95)
            )
            if not passes_tissue:
                continue

            try:
                out_path = out_dir / f"{slide_id}_x{x}_y{y}.png"
                alpha_to_white(slide.read_region((x, y), 0, (tile_px, tile_px))).save(out_path, optimize=True)
                rows.append(
                    {
                        "slide_id": slide_id,
                        "x": x,
                        "y": y,
                        "tile_px": tile_px,
                        "file": str(out_path.relative_to(out_dir)),
                    }
                )
                saved += 1
            except OSError:
                pass

    return rows, saved
