# Pokéwars image recognition

Current scope: predict the anonymous `pokemon_id` from one noisy 32×32 grayscale image. This project uses the labeled images in the [Kaggle competition](https://www.kaggle.com/competitions/cs-3780-5780-fa-26-the-pokewar/) and evaluates recognition locally. It does not submit predictions to Kaggle. Code and compact validation summaries are versioned here; competition data, model weights, and per-image predictions stay outside Git.

## Data setup

Download the competition's **Download as zip** archive from its [Data page](https://www.kaggle.com/competitions/cs-3780-5780-fa-26-the-pokewar/data) and save it as `data/competition.zip`. The experiment reads the PNGs directly from the archive and caches the 160,925 unique labeled images in `data/labeled_32x32.npy`. Data and generated files are ignored by Git.

The CSV path `images/train/abc.png` maps to ZIP member `data/data/train/abc.png`.

## Run

Use Python 3.11 or newer and an Apple GPU if available:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python image_experiment.py --epochs 20 --translate-pixels 1 --run-name shift1_20e
```

The script runs three stratified folds. Each batch contains 32 identities and 4 images per identity. A small CNN produces a 128-dimensional embedding, trained with cross-entropy and [supervised contrastive loss](https://arxiv.org/abs/2004.11362). The selected schedule uses one epoch of linear learning-rate warmup, cosine decay from `1e-3` to `1e-5`, gradient clipping, and validation-based checkpoint selection with early stopping. Training images receive a random shift of at most one pixel. Validation images are never augmented.

Outputs:

- `outputs/image_shift1_20e_cv.json`: current three-fold top-1 and top-5 results.
- `outputs/image_shift1_20e_fold{1,2,3}.pt`: saved fold model weights when using the command above. The currently saved fold weights have the names `image_long_shift1_f{0,1,2}_fold{1,2,3}.pt`, since those folds were run separately.
- `outputs/image_shift1_20e_fold{1,2,3}_history.json`: per-epoch training and validation curves when using the command above.
- `data/labeled_32x32.npy`: cached labeled images.

## Validation

Folds are assigned by **unique image path** and stratified by identity. The validation image is never used to train its model or construct its class prototype. Scores are calculated on validation images only. Identity labels are internally shifted from Kaggle's 1–1025 to zero-based 0–1024 for training.

To test whether the contrastive loss helps beyond a CNN trained with cross-entropy, run one matched-fold ablation:

```bash
.venv/bin/python image_experiment.py --fold 0 --epochs 5 --contrastive-weight 0 --run-name ce_raw
```

## Local results (M4 GPU)

The selected 20-epoch schedule with one-pixel translation reached three-fold linear-head top-1 **68.42%** and top-5 **84.41%**. Prototype classification reached top-1 **61.85%** and top-5 **78.97%**. Best epochs were 20, 19, and 19. The folds took about 30 minutes of model time total on this M4 machine. A matched single-fold run without translation reached **65.00%** top-1 versus **68.20%** with translation.

An additional low-learning-rate fine-tune of fold 1 started from 68.20% top-1. It triggered early stopping after four epochs without a meaningful improvement, so the 20-epoch checkpoint was retained. The selected model is the best of the **tested local configurations**, not a claim of global optimality.

The earlier 5-epoch three-fold model reached linear-head top-1 **43.80%**. On one matched 5-epoch fold, cross-entropy alone reached **37.07%**, while cross-entropy plus contrastive loss reached **42.92%**. This ablation does not establish an advantage over a separately tuned ResNet20.

## Final image model

After choosing the 20-epoch schedule using cross-validation, train on all labeled images and predict identities for the 160,000 unique test images:

```bash
.venv/bin/python train_final_image.py --epochs 20 --translate-pixels 1
```

The saved model is `outputs/image_final_all_labeled.pt`. `outputs/test_image_identity_predictions.csv` contains `image_path`, `predicted_pokemon_id` (1–1025), `top1_softmax`, and `top5_pokemon_ids`. Softmax scores are **not calibrated probabilities**. This file contains image identity predictions, not battle-score predictions or a Kaggle submission.

`baseline.py` is retained as a historical first experiment. It also contains score prediction code and is outside the current image-only scope.

## GPU experiments

`train_large_image.py` compares a native 32×32 WideResNet-28-10 with ConvNeXt V2 Nano. Training samples a fresh affine view of each image on every epoch: translation up to 1.5 pixels, rotation up to 12 degrees, and scale from 0.9 to 1.1. The validation images stay untransformed. Both models use AdamW, two warmup epochs followed by cosine decay, label smoothing, gradient clipping, and best-validation checkpoint selection with early stopping.

Install the extra dependencies with `pip install -r requirements_large.txt` on a CUDA machine with a compatible PyTorch build. The optional `data/labeled_ids.npy` contains the zero-based labels in the same sorted-path order as `data/labeled_32x32.npy`; if absent, the script reads labels from `data/competition.zip`.

```bash
python train_large_image.py --model wrn28_10 --fold 0 --epochs 40
python train_large_image.py --model convnextv2_nano --fold 0 --epochs 40
```

To continue a fold after its initial cosine schedule, pass its best checkpoint to a new run with a lower learning rate. The initial checkpoint is measured on validation and saved as epoch 0, so a continuation that fails to improve cannot replace it with a worse model:

```bash
python train_large_image.py --model wrn28_10 --fold 0 --epochs 60 \
  --initial-weights outputs/large/wrn28_10_f0_best.pt \
  --learning-rate 1e-4 --min-learning-rate 1e-7 \
  --warmup-epochs 1 --min-epochs 20 --patience 10 \
  --min-improvement 0.001 --run-name wrn28_10_f0_stage2
```

The remote `run_large_cv_queue.sh` waits for the current fold-0 run, then starts its continuation and the remaining folds. Each fold keeps its own best checkpoint and training history. This script launches only local image experiments and does not submit or publish anything.

`run_wrn_mild_queue.sh` waits for that WRN queue, then runs a second three-fold WRN experiment with a different training seed and milder translation, rotation, and scaling. The fold split stays fixed so its held-out predictions can later be compared with and ensembled with the first WRN. It has a four-hour wall-time cap that starts after the first queue completes; best checkpoints are saved during training.

To export image identities for a slice of the sorted test paths, run `python predict_wrn_topk.py --offset 0 --limit 80000 --top-k 10`. It averages softmax distributions from the three best continuation checkpoints and writes a CSV with top-10 IDs and their scores. The test set has 80,000 battles but 160,000 distinct images, so both halves are needed for complete battle coverage. The combined local artifact is `outputs/large/test_wrn28_10_ensemble_160000_top10.csv`. These scores are not calibrated probabilities. The CSV is an image-level artifact, not a Kaggle submission.

The milder-augmentation WRN reached 93.19% top-1 and 98.45% top-10 in held-out three-fold validation. Averaging its fold-matched probabilities with the original WRN reached 93.65% top-1 and 98.61% top-10; see the tracked summary `results/wrn28_10_pair_ensemble_3fold_topk.json`. The updated local full-test artifact is `outputs/large/test_wrn28_10_six_model_160000_top10.csv`, generated with `python predict_wrn_topk.py --variant ensemble --limit 160000 --top-k 10`.

For error analysis, run `python analyze_wrn_errors.py` on the CUDA machine. It saves fold-matched out-of-fold candidates, probabilities, true-label ranks, and each WRN variant's prediction to `outputs/large/wrn_pair_oof_diagnostics.npz`. The aggregate summary is tracked as `results/wrn_pair_oof_error_summary.json`; the per-image error CSV and `error_gallery_*.png` contact sheets remain local under `outputs/large/`.

`evaluate_wrn_tta.py` compares deterministic test-time views on the same three held-out folds. Averaging the original view with rotations of ±5 degrees raised top-1 from 93.65% to 93.89% and top-10 from 98.61% to 98.73%. Four one-pixel shifts reduced top-1 to 93.57%. Full counts and per-fold results are tracked in `results/wrn_pair_tta_cv.json`.

The corresponding full-test image artifact is `outputs/large/test_wrn28_10_six_model_rot5_160000_top10.csv`, generated with `python predict_wrn_topk.py --variant ensemble --tta rotate_5deg --limit 160000 --top-k 10`. Its 160,000 rows were checked against the sorted test path cache. It contains image-level top-10 scores, not battle-score predictions or a Kaggle submission.
