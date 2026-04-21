"""I/O utilities for slide processing."""

import re
from pathlib import Path

from PIL import Image


def slide_key(name):
    """Normalize slide name by removing path and WSI extension."""
    return re.sub(
        r"\.(svs|tif|tiff|scn|ndpi|mrxs)$",
        "",
        Path(str(name)).name,
        flags=re.I,
        count=1,
    )


def alpha_to_white(img):
    """Convert RGBA image to RGB with alpha composited to white background."""
    return (
        img.convert("RGB")
        if img.mode != "RGBA"
        else Image.alpha_composite(
            Image.new("RGBA", img.size, (255, 255, 255, 255)),
            img,
        ).convert("RGB")
    )
