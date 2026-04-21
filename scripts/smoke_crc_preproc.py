#!/usr/bin/env python3
"""Basic smoke checks for integrated crc_preproc components."""

from pathlib import Path
from tempfile import TemporaryDirectory

from PIL import Image

from crc_preproc.io_utils import alpha_to_white, slide_key
from crc_preproc.labels import build_labels


def main() -> None:
    assert slide_key("foo/bar/CRC_0001.svs") == "CRC_0001"
    assert slide_key("CRC_0002.TIFF") == "CRC_0002"

    rgba = Image.new("RGBA", (4, 4), color=(0, 255, 0, 128))
    rgb = alpha_to_white(rgba)
    assert rgb.mode == "RGB"
    assert rgb.size == (4, 4)

    with TemporaryDirectory() as td:
        csv_path = Path(td) / "labels.csv"
        csv_path.write_text("filename,label\nCRC_0001.svs,1\nCRC_0002.tiff,2\n", encoding="utf-8")
        labels = build_labels(csv_path)
        assert labels["CRC_0001"] == 1
        assert labels["CRC_0002"] == 2

    print("crc_preproc smoke checks passed")


if __name__ == "__main__":
    main()
