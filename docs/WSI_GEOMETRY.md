# WSI geometry: čo vždy uviesť (4 čísla)

Pre každý whole-slide obraz použitý v segmentácii alebo MIL pipeline by mali čitatelia (a metódy) vidieť **štyri** položky:

| # | Označenie | Význam |
|---|-----------|--------|
| 1 | \(W_0 \times H_0\) | Rozmer **levelu 0** (plné rozlíšenie) v pixeloch |
| 2 | \(L_s\), \(W_s \times H_s\) | **Pyramid level** použitý na segmentáciu (alebo na zarovnanie masky) a jeho rozmer v px |
| 3 | \(d_s = \texttt{level\_downsamples}[L_s]\) | Downsample faktor z levelu 0 na \(L_s\) (OpenSlide) |
| 4 | \(\mathrm{mpp}_x, \mathrm{mpp}_y\) | Mikróny na pixel z metadát slidov (ak sú; často `openslide.mpp-x` / `openslide.mpp-y`) |

## OpenSlide konvencie

- `level_dimensions[0]` = najvyššie rozlíšenie (level 0).
- `level_downsamples[i]` = koľkokrát je level `i` menší ako level 0 (šírka/výška približne deliteľná týmto faktorom).

## Nástroj v repozitári

```bash
python scripts/print_wsi_geometry.py /cesta/k/slide.tiff --seg-level -1
python scripts/print_wsi_geometry.py slide.tiff --lang sk
```

`--seg-level` musí zodpovedať levelu, ktorý **skutočne** používa tvoja segmentácia (napr. TRIDENT). `-1` = najhrubší level (časté pri PANDA maskách).

## Python API

```python
from mil.wsi_metadata import wsi_geometry_from_path, format_wsi_geometry_lines

g = wsi_geometry_from_path("slide.tiff", seg_level=-1)
print(format_wsi_geometry_lines(g, lang="sk"))
```

Publikovateľné figúry z `mil.attention_map.plot_attention_figure_publishable` môžu mať tento blok v pätičke (parametre ako `seg_level` v tej funkcii alebo CLI v `scripts/plot_attention_map.py`).
