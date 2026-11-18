"""Cache the training images whose battles have no species labels."""

from __future__ import annotations

import argparse
from pathlib import Path
from zipfile import ZipFile

import pandas as pd

from image_experiment import image_cache


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", type=Path, default=Path("data/competition.zip"))
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    args = parser.parse_args()
    with ZipFile(args.zip) as archive:
        rows = pd.read_csv(archive.open("data/data/train.csv"))
        unknown = rows.pokemon_id1.isna() & rows.pokemon_id2.isna()
        paths = sorted(set(rows.noisy_image1[unknown]) | set(rows.noisy_image2[unknown]))
        labeled = set(rows.noisy_image1[~unknown]) | set(rows.noisy_image2[~unknown])
        if labeled & set(paths):
            raise ValueError("An unlabeled-battle image also appears in a labeled battle")
        (args.output_dir / "unlabeled_paths.txt").write_text("\n".join(paths) + "\n")
        image_cache(archive, paths, args.output_dir / "unlabeled_32x32.npy")
    print(f"{unknown.sum():,} battles, {len(paths):,} images", flush=True)


if __name__ == "__main__":
    main()
