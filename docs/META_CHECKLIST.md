# MetaCentrum Best Practices – MIL Training Checklist

Podľa [Vorel_MC_2021.pdf](https://wiki.metacentrum.cz) – „Jak sejmout MetaCentrum v 10 krocích“.

---

## 1. Kopírovať vstupy do $SCRATCHDIR

> „Very intensive I/O operations can cause network overload and the slowdown of central storage (/storage/city/…). Copy the input data into the scratch directory on a dedicated machine.“

| Stav | Implementácia |
|------|----------------|
| ✅ Hotovo | `COPY_TO_SCRATCH=1` kopíruje `FEAT_DIR` do `$SCRATCHDIR/features` pred tréningom |
| | Použitie: `export COPY_TO_SCRATCH=1` a `qsub -V -l "select=1:ncpus=4:mem=128gb:ngpus=1:scratch_local=100gb:gpu_mem=10gb" scripts/mil_training.pbs` (z koreňa MIL) |
| | Vyžaduje `scratch_local=100gb` pri `qsub -l` (viď komentár v `scripts/mil_training.pbs`) |

---

## 2. Čistiť scratch cez `trap` TERM EXIT

> „Do not forget to clean the scratch directory when your calculation is done or have been killed by PBS.“

| Stav | Implementácia |
|------|----------------|
| ✅ Hotovo | `trap on_exit EXIT` → `clean_scratch_if_ok` pri úspešnom ukončení |
| | `trap '...; exit 143' TERM` → exit spustí `on_exit` → `clean_scratch_if_ok` |
| | Pri faili sa scratch necháva (debug) – rozumný kompromis |

PDF odporúča `trap 'clean_scratch' TERM EXIT` – náš `on_exit` robí to isté s dodatočnou logikou (run_meta.txt, podmienené clean pri faili).

---

## 3. Dĺžka jobu ~30+ minút

> „An ideal job is running at least for 30 minutes. Startup overhead may be a significant part of the whole processing time.“

| Stav | Implementácia |
|------|----------------|
| ✅ OK | PBS `walltime=24:00:00` – job beží hodiny, nie minúty |
| ⚠️ Startup | Epoch 1 je drahá (load dat, init model, warmup) – to je očakávané overhead |
| | Ak by si delil 1 run na viac krátkych jobov – zle. Jeden job, viac foldov = dobre. |

---

## 4. Zápisy mimo scratch – /tmp a veľké stdout/stderr

> „Computing nodes have very limited quotas (only 1 Gb) to write out of the scratch directory. The most common problems: Write to /tmp, Very large stdout and stderr streams.“

| Stav | Implementácia |
|------|----------------|
| ⚠️ Čiastočne | Logy idú do `$MIL/logs/` (centrálny storage) – pri 24h jobe to môže narásť |
| ❌ TODO | `export TMPDIR=$SCRATCHDIR` – Python/h5py/torch temp súbory môžu ísť do /tmp |
| | PDF: `export TMPDIR=$SCRATCHDIR` a `1>$SCRATCHDIR/stdout 2>$SCRATCHDIR/stderr` |
| | Opcia: logovať do scratch, na konci `cp` do storage (ak chceš zachrániť) |

---

## 5. Neefektívna rezervácia zdrojov

> „Reservation of too many resources decrease your fairshare score and reduces the priority for your future jobs.“

| Stav | Implementácia |
|------|----------------|
| ⚠️ Kompromis | 128 GB RAM – stabilizácia ~75 GB je obhajiteľná, nie elegantná |
| | Skús: `mem=96gb` pri ďalšom rune ak vidíš headroom (treba monitorovať) |
| | GPU 10 GB – MIL je I/O-bound, GPU mem väčšinou nie je kritická |

---

## 6. I/O pattern – on-the-fly .h5 loading

> „Very intensive I/O… Copy the input data into the scratch directory.“

| Stav | Implementácia |
|------|----------------|
| ✅ PRELOAD=1 | Fold-local cache do RAM – odstráni on-the-fly čítanie počas epochy |
| ✅ COPY_TO_SCRATCH | Keď 1: copy do scratch, potom čítanie z lokálneho SSD |
| | Kombinácia: `COPY_TO_SCRATCH=1` + `PRELOAD=1` = scratch → RAM = najrýchlejší path |

---

## Ďalšie výkonové tuny (nie priamo z PDF)

| Položka | Akcia |
|---------|-------|
| `num_workers` | Benchmark 0 / 2 / 4 (pri PRELOAD môže byť 0 rýchlejší – menej IPC) |
| `max_patches_train` | Skús 256/384 – menej I/O a load, rýchlejšie epochy |
| `max_patches_val` | Už 512 default – rozumne |

---

## Zhrnutie – čo aplikovať hneď

1. **`COPY_TO_SCRATCH=1`** pri ďalšom rune – hlavný I/O upgrade.
2. **`export TMPDIR=$SCRATCHDIR`** – pridať do PBS skriptu pred Python (rýchly fix).
3. Logy – ponechať v storage; ak by job mal >100 MB logov, zvážiť log do scratch + copy na konci.
4. Mem – 128 GB nechaj; ak uvidíš konzistentne <80 GB, v budúcnosti skús 96 GB.

---

## Citácia

> „There is no reason to be afraid to use MetaCentrum. By your activity, you are not able to 'destroy' something.“  
> — Jiří Vorel, MetaCentrum User Support, 2021
