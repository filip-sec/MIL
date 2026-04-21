"""Tile WSI based on tissue mask."""

from tqdm import tqdm

from .io_utils import alpha_to_white


def tile_wsi(slide, mask_thumb, tw_th, w0_h0, out_dir, slide_id, tile_px=512, strict=True):
    """Extract tiles from WSI, filtering by tissue mask."""
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

            if roi.size == 0:
                continue
            tissue_mask = roi == 255
            if (tissue_mask.all() if strict else tissue_mask.mean() >= 0.95):
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
