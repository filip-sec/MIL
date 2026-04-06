"""Whole-slide image (WSI) geometry for reproducible reporting.

Always report these **four** items for any slide used in segmentation / MIL pipelines:

1. Level 0 size W0 x H0 (px) — full resolution (OpenSlide level 0).
2. Segmentation level L_s and size Ws x Hs (px).
3. Downsample d_s = level_downsamples[L_s].
4. MPP — microns per pixel mpp_x, mpp_y from slide metadata when present.

OpenSlide: level_dimensions[0] is highest resolution; level_downsamples[i] maps level 0 to level i.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

Language = Literal["en", "sk"]


@dataclass(frozen=True)
class WSIGeometry:
    """Canonical WSI geometry for one chosen segmentation / analysis pyramid level."""

    w0: int
    h0: int
    seg_level: int
    ws: int
    hs: int
    d_s: float
    mpp_x: float | None
    mpp_y: float | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "w0": self.w0,
            "h0": self.h0,
            "seg_level": self.seg_level,
            "ws": self.ws,
            "hs": self.hs,
            "d_s": self.d_s,
            "mpp_x": self.mpp_x,
            "mpp_y": self.mpp_y,
        }


def _require_openslide():
    try:
        import openslide  # noqa: F401
    except ImportError as e:
        raise ImportError("openslide-python is required for WSI geometry") from e


def normalize_pyramid_level(slide: Any, level: int) -> int:
    """Resolve negative index (e.g. -1 = coarsest level)."""
    n = int(slide.level_count)
    if level < 0:
        level = n + level
    if level < 0 or level >= n:
        raise ValueError(f"level {level} out of range for slide with {n} levels (0..{n - 1})")
    return level


def parse_mpp_from_slide(slide: Any) -> tuple[float | None, float | None]:
    """Read openslide.mpp-x / openslide.mpp-y if present."""

    import openslide

    def _f(key: str) -> float | None:
        raw = slide.properties.get(key)
        if raw is None or raw == "":
            return None
        try:
            return float(raw)
        except ValueError:
            return None

    return _f(openslide.PROPERTY_NAME_MPP_X), _f(openslide.PROPERTY_NAME_MPP_Y)


def wsi_geometry_from_slide(slide: Any, *, seg_level: int = -1) -> WSIGeometry:
    """Build WSIGeometry from an open OpenSlide instance."""
    _require_openslide()

    Ls = normalize_pyramid_level(slide, seg_level)
    w0, h0 = slide.level_dimensions[0]
    ws, hs = slide.level_dimensions[Ls]
    ds = float(slide.level_downsamples[Ls])
    mx, my = parse_mpp_from_slide(slide)
    return WSIGeometry(
        w0=int(w0),
        h0=int(h0),
        seg_level=Ls,
        ws=int(ws),
        hs=int(hs),
        d_s=ds,
        mpp_x=mx,
        mpp_y=my,
    )


def wsi_geometry_from_path(path: Path | str, *, seg_level: int = -1) -> WSIGeometry:
    """Open slide path, compute geometry, close handle."""
    _require_openslide()
    import openslide

    p = Path(path)
    slide = openslide.OpenSlide(str(p))
    try:
        return wsi_geometry_from_slide(slide, seg_level=seg_level)
    finally:
        slide.close()


def format_wsi_geometry_lines(geom: WSIGeometry, *, lang: Language = "en") -> str:
    """Human-readable four-line block (for logs, figure captions, methods text)."""

    if lang == "sk":
        lines = [
            f"(1) Rozmer levelu 0: {geom.w0} × {geom.h0} px",
            f"(2) Level segmentácie L_s = {geom.seg_level}, rozmer: {geom.ws} × {geom.hs} px",
            f"(3) Downsample d_s = level_downsamples[L_s] = {geom.d_s}",
        ]
        if geom.mpp_x is not None or geom.mpp_y is not None:
            mx = geom.mpp_x if geom.mpp_x is not None else float("nan")
            my = geom.mpp_y if geom.mpp_y is not None else float("nan")
            lines.append(f"(4) MPP (µm/px): mpp_x = {mx}, mpp_y = {my}")
        else:
            lines.append("(4) MPP (µm/px): nie je v metadátach slidov")
        return "\n".join(lines)

    lines = [
        f"(1) Level 0 dimensions: {geom.w0} × {geom.h0} px",
        f"(2) Segmentation level L_s = {geom.seg_level}, dimensions: {geom.ws} × {geom.hs} px",
        f"(3) Downsample d_s = level_downsamples[L_s] = {geom.d_s}",
    ]
    if geom.mpp_x is not None or geom.mpp_y is not None:
        mx = geom.mpp_x if geom.mpp_x is not None else float("nan")
        my = geom.mpp_y if geom.mpp_y is not None else float("nan")
        lines.append(f"(4) MPP (µm/pixel): mpp_x = {mx}, mpp_y = {my}")
    else:
        lines.append("(4) MPP (µm/pixel): not available in slide metadata")
    return "\n".join(lines)


def print_wsi_geometry(path: Path | str, *, seg_level: int = -1, lang: Language = "en") -> WSIGeometry:
    """Load slide, print four-line report, return geometry."""
    geom = wsi_geometry_from_path(path, seg_level=seg_level)
    print(format_wsi_geometry_lines(geom, lang=lang))
    return geom
