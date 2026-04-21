"""Process single WSI."""

import numpy as np
import pandas as pd
from PIL import Image

from .io_utils import slide_key
from .qc import save_qc
from .tissue import extract_s_channel, make_thumbnail, otsu_mask

try:
    import openslide
except Exception as e:  # pragma: no cover - runtime dependency validation
    openslide = None
    _OPENSLIDE_IMPORT_ERROR = e
else:
    _OPENSLIDE_IMPORT_ERROR = None

try:
    from .tiler_numba import tile_wsi_fast as tile_wsi
except Exception:
    from .tiler import tile_wsi


def process_one(
    wsi_path,
    out_root,
    labels_map,
    downsample=32,
    tile_px=512,
    strict=True,
    skip=False,
):
    """Process a single WSI: thumbnail, mask, tile, save CSV."""
    if openslide is None:
        raise ImportError(
            "openslide-python is required for preprocessing. Install project extra '[preproc]' and system libopenslide."
        ) from _OPENSLIDE_IMPORT_ERROR

    slide_id = slide_key(wsi_path)

    img_root = out_root / "data" / "images"
    labels_dir = out_root / "data" / "labels"
    thumb_dir = img_root / "thumbs"
    tiles_dir = img_root / "tiles"
    csv_path = labels_dir / f"{slide_id}_tiles.csv"

    if skip and csv_path.exists():
        df = pd.read_csv(csv_path)
        if not df.empty and "label" in df.columns and not df["label"].isna().all():
            return slide_id, len(df), labels_map.get(slide_id)

    slide = openslide.OpenSlide(str(wsi_path))
    thumb_rgb, (tw, th), (w0, h0) = make_thumbnail(slide, downsample)
    s_channel = extract_s_channel(thumb_rgb)
    mask = otsu_mask(s_channel)

    thumb_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    for name, arr in [("thumb_rgb", thumb_rgb), ("thumb_S", s_channel), ("thumb_mask", mask)]:
        Image.fromarray(arr).save(thumb_dir / f"{slide_id}_{name}.png", optimize=True)

    overlay = thumb_rgb.copy()
    overlay[mask == 255] = (0.5 * overlay[mask == 255] + 0.5 * np.array([0, 255, 0])).astype(np.uint8)
    Image.fromarray(overlay).save(thumb_dir / f"{slide_id}_thumb_overlay.png", optimize=True)

    rows, saved = tile_wsi(slide, mask, (tw, th), (w0, h0), tiles_dir / slide_id, slide_id, tile_px, strict)
    slide.close()

    df = pd.DataFrame(rows)
    df["label"] = labels_map.get(slide_id, "")
    df.to_csv(csv_path, index=False)

    if saved > 0:
        save_qc(tiles_dir / slide_id, tiles_dir / f"{slide_id}_QC.png")

    return slide_id, saved, labels_map.get(slide_id)
