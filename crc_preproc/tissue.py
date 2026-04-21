"""Tissue detection via thumbnail and Otsu masking."""

import cv2
import numpy as np


def make_thumbnail(slide, downsample=32):
    """Generate RGB thumbnail from WSI."""
    width_l0, height_l0 = slide.dimensions
    thumb_size = (max(1, width_l0 // downsample), max(1, height_l0 // downsample))
    thumb_np = np.array(slide.get_thumbnail(thumb_size).convert("RGB"))
    slide_size = (width_l0, height_l0)
    return thumb_np, thumb_size, slide_size


def extract_s_channel(thumb_rgb):
    """Extract HSV saturation channel from RGB thumbnail."""
    return cv2.cvtColor(thumb_rgb, cv2.COLOR_RGB2HSV)[:, :, 1]


def otsu_mask(s_channel, open_close=5, min_frac=1e-3):
    """Create binary tissue mask using Otsu thresholding."""
    _, mask = cv2.threshold(s_channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    if open_close > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_close, open_close))
        mask = cv2.morphologyEx(cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel), cv2.MORPH_CLOSE, kernel)

    if min_frac > 0:
        height, width = mask.shape
        _, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        keep = stats[1:, cv2.CC_STAT_AREA] >= int(min_frac * height * width)
        mask = np.concatenate([[False], keep])[labels].astype(np.uint8) * 255

    return mask
