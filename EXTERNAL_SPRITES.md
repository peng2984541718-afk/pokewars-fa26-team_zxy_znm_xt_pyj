# PokéAPI sprite experiment

The competition's 1,025 anonymous classes are **not** National Dex numbers.
External images are stored under `data/external/pokeapi/` and ignored by Git.
The downloader records source URLs and download results in each variant's
manifest. Source: https://github.com/PokeAPI/sprites

## Fold-safe procedure

1. Download `default`, `home`, and `official-artwork` with
   `python fetch_pokeapi_sprites.py --variant <variant>`.
2. Train or use a classifier for fold *f* using only the other two folds.
3. Run `probe_external_sprite_mapping.py` for `home` and `official-artwork`
   with that classifier. The probe renders each source as a noisy 32×32 image.
4. Run `resolve_external_mapping.py` on the two probe outputs. It finds a
   one-to-one assignment of National Dex IDs to anonymous class IDs.
5. Run `build_external_sprite_bank.py` with that fold's assignment. It renders
   rotated, translated, scaled, grayscale, noisy versions of the two high
   resolution sources. Keep the bank for fold *f* separate from other folds.
6. Use `train_large_image.py --initial-weights <fold-f checkpoint>
   --external-bank <fold-f bank> --external-fraction 0.1` for a short fine-tune.
   Compare with a continuation from the same checkpoint, schedule, and seed
   using `--external-fraction 0`. Select the fraction on one fold, then apply
   the fixed choice to the other two folds.

The default 96×96 sprites are included in the probe but excluded from the
training bank: their top-1 class predictions were much less stable than the
two high resolution sources. Near-identical species or forms, such as Dex
963/964, need special care; one artwork template cannot distinguish every
competition image from these classes. Do not infer that a confident mapping
alone proves a validation gain.
