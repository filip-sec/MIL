"""Load slide-level labels from CSV."""

import pandas as pd

from .io_utils import slide_key


def build_labels(labels_csv):
    """Build dictionary mapping slide_id -> label from CSV."""
    if not labels_csv or not labels_csv.exists():
        return {}

    labels_df = pd.read_csv(labels_csv)

    name_col = next(
        (c for c in ("filename", "slide_name", "slide", "wsi", "id") if c in labels_df.columns),
        None,
    )
    lab_col = next(
        (c for c in ("label", "slide_label", "Label", "class", "Class") if c in labels_df.columns),
        None,
    )

    if not name_col or not lab_col:
        return {}

    labels_df["_key"] = labels_df[name_col].astype(str).map(slide_key)
    return (
        labels_df.dropna(subset=["_key"])
        .drop_duplicates("_key")
        .set_index("_key")[lab_col]
        .to_dict()
    )
