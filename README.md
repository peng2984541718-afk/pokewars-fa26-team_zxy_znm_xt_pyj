# Pokéwars

Public leaderboard **81.32036**. The file that scored it is `outputs/large/submission_rx_v9_grand_council.csv`, also copied to `outputs/large/submission_latest_best.csv`.

That submission is the v9 grand council: the v7 specialist council, the v8 exact-quota debiaser, and two extra specialists that only vote on the uncertain tail (`p_max < 0.95`, blend weight capped at 0.36).

- `CentroidRelPosViT` — soft-centroid alignment, overlapping conv stem, 8×8 tokens, 2D relative-position attention. Fold-0 6-view top-1 0.94800.
- `CentroidConvNeXtV2` — same centroid stem, 7×7 depthwise kernels, global response norm. Fold-0 6-view top-1 0.95407.

Both are wrapped so a horizontal flip is bit-identical. Fold-0 had to beat the locked v8 anchor before folds 1 and 2 were trained. It did:

| | v8 anchor | v9 |
|---|---:|---:|
| Image top-1 | 0.97057 | 0.97108 |
| Brier | 0.04008 | 0.03935 |
| HG_test | 80.205 | 80.499 |
| M7_test | 80.227 | 80.518 |

Chosen blend weights `(logpolar, logpolar×border, vit, vit×border, convnext, convnext×hf, temperature)` are `(0.04, 0.04, 0.05, 0.03, 0.05, 0.03, 0.80)`.

## Run

CUDA PyTorch, plus:

```bash
python -m pip install -r requirements_large.txt
```

`cuda_extra/` is a local CUDA library drop and is not in git. If `libcusparseLt.so.0` is missing:

```bash
export LD_LIBRARY_PATH="$(find "$PWD/cuda_extra" -name '*.so*' -printf '%h\n' | sort -u | paste -sd: -):${LD_LIBRARY_PATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

Symmetry check only:

```bash
python milestone9b_orthogonal_council_v9.py --smoke-only
```

Full Fold-0 gate, then folds 1–2 if the gate passes, then the submission CSV:

```bash
python milestone9b_orthogonal_council_v9.py
```

Finished specialist checkpoints (`centroid_vit_v9_f*_best.pt`, `centroid_convnextv2_v9_f*_best.pt` plus the matching `.done` files) are skipped. The run reads the image arrays and the cached council artifacts below; it does not retrain milestones 1–7 when those files are already present.

## What has to be on disk

Git ignores data, checkpoints, and submissions. A clone runs only after these local caches exist:

- `labeled_32x32.npy`, `labeled_ids.npy`, `test_32x32.npy`
- `unl/test_32x32.npy`, `data/train.csv.gz`, `data/unlabeled_paths.txt`, `data/known_battles.npz`
- `outputs/large/` pair tables and calibration (`milestone2_pair_table_v4.json`, `milestone3_calibration.json`), v7/v8 council packs (`rx_v1_*.npz`, `rx_v8_debias_*.npz`), log-polar specialist scores, and `transductive_bank_f{0,1,2}.npz`

Rebuild the transductive banks, if those three npz files are missing, with `build_fold_transductive_banks.py`. The milestone scripts `milestone1` through `milestone7` and `milestone9a_exact_quota_debias.py` are the programs that produced the other caches.

## Layout

| File | Role |
|---|---|
| `milestone9b_orthogonal_council_v9.py` | Winning entry. Trains the two specialists, gates on Fold-0, writes the submission. |
| `milestone9a_exact_quota_debias.py` | Fits the species-bias correction on the exact-quota pools and applies it with a zero-perturbation gate. |
| `milestone7_specialist_council.py` | Log-polar specialist and the council fuse used as the v9 base. |
| `milestone6b_twin_spatial_specialist.py` | Twin-class loss weights shared by specialist training. |
| `milestone2_pair_table_v4.py`, `milestone3_inference_calibration.py` | Battle-payoff table and heteroscedastic BP calibration that the council reads back. |
| `train_large_image.py`, `train_large_image_external.py` | Augmentation, preprocessing, backbones. |
| `export_probs_rot5.py` | 6-view TTA scoring. |
