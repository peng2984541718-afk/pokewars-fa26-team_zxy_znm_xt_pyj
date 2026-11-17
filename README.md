# CS 3780/5780 Pokéwars baseline

This repository contains a local baseline for the [Kaggle competition](https://www.kaggle.com/competitions/cs-3780-5780-fa-26-the-pokewar/). It starts with image recognition: classify each noisy 32×32 grayscale Pokémon image into one of 1,025 anonymous IDs. A simple per-ID strength model turns two predicted IDs into a battle score and a valid Kaggle submission.

## Data setup

Download the competition's **Download as zip** archive from its [Data page](https://www.kaggle.com/competitions/cs-3780-5780-fa-26-the-pokewar/data) and save it as `data/competition.zip`. The script reads the PNGs directly from the archive. The archive and generated files are ignored by Git; competition data is not redistributed through this repository.

The CSV path `images/train/abc.png` maps to ZIP member `data/data/train/abc.png`.

## Run

Use Python 3.9 or newer:

```bash
python3 -m pip install -r requirements.txt
python3 baseline.py
```

Outputs:

- `outputs/cv_metrics.json`: three-fold validation metrics.
- `outputs/submission.csv`: 80,000 `row_id,result` predictions in sample-submission order.
- `outputs/train_oof_identity.csv`: out-of-fold image identity predictions for training images.
- `outputs/test_identity.csv`: image identity predictions for test images.
- `data/features_16x16.npy`: cached image features, reused on later runs.

The image baseline averages each 2×2 pixel block, computes a mean 16×16 image for each labeled ID, and assigns an image to its nearest class mean. The score baseline fits `result ≈ 5 + strength[id1] − strength[id2]` with ridge regularization and clips predictions to `[0, 10]`. An out-of-fold scalar blend with the constant score 5 protects the initial submission if image recognition is weak.

## Validation

Image folds are assigned by **unique image path**, so the image classifier never trains on the same image it validates. Each class prediction used to assess the score pipeline is out of fold. A separate three-fold split of matchup rows fits the score model. We report both image ID accuracy and end-to-end score RMSE, plus RMSE using the supplied true IDs to show how much image errors matter.

The Kaggle public leaderboard uses 30% of test rows and the final private leaderboard uses 70%; use local cross-validation to choose model changes.

## Next improvements

The class-mean recognizer is deliberately simple. A CNN trained on labeled images is a natural next step. Its inference should keep the same image-path-to-ID interface so the battle-score model can be compared independently.
