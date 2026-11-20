"""One-pass forensic scan of battle order, filenames, and PNG chunks."""
from __future__ import annotations

import json
import struct
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path("outputs/large/table_metadata_probe.json")


def rank_corr(a: np.ndarray, b: np.ndarray) -> float:
    ar = pd.Series(a).rank(method="average").to_numpy()
    br = pd.Series(b).rank(method="average").to_numpy()
    ar = ar - ar.mean()
    br = br - br.mean()
    denom = np.sqrt((ar * ar).sum() * (br * br).sum())
    if denom == 0:
        return float("nan")
    return float((ar * br).sum() / denom)


def lag1(values: np.ndarray) -> float:
    x = values.astype(np.float64)
    x = x - x.mean()
    denom = float((x * x).sum())
    if denom == 0:
        return float("nan")
    return float((x[:-1] * x[1:]).sum() / denom)


def deciles(values: np.ndarray) -> list[float]:
    edges = np.linspace(0, len(values), 11, dtype=int)
    return [float(values[edges[i]:edges[i + 1]].mean()) for i in range(10)]


def png_chunks(payload: bytes) -> list[str]:
    if payload[:8] != b"\x89PNG\r\n\x1a\n":
        return ["not-png"]
    kinds = []
    pos = 8
    while pos + 8 <= len(payload) and len(kinds) < 12:
        length = struct.unpack(">I", payload[pos:pos + 4])[0]
        kind = payload[pos + 4:pos + 8].decode("ascii", errors="replace")
        kinds.append(kind)
        pos += 12 + length
        if kind == "IDAT":
            break
    return kinds


def main() -> None:
    train = pd.read_csv("data/train.csv.gz")
    test = pd.read_csv("data/test.csv.gz")
    known = train.pokemon_id1.notna() & train.pokemon_id2.notna()
    labeled = train.loc[known].reset_index(drop=True)
    y = labeled.result.to_numpy(np.float64)
    id1 = labeled.pokemon_id1.to_numpy(np.int64)
    id2 = labeled.pokemon_id2.to_numpy(np.int64)
    gap = id2 - id1
    order = np.arange(len(labeled))

    identity = {}
    for side in (1, 2):
        for path, value in zip(train[f"noisy_image{side}"], train[f"pokemon_id{side}"]):
            if pd.notna(value):
                identity[path] = int(value)
    paths = sorted(identity)
    labels = np.load("labeled_ids.npy").astype(np.int64)
    path_match = len(paths) == len(labels) and np.array_equal(
        np.array([identity[p] - 1 for p in paths]), labels)
    stems = [Path(p).stem for p in paths]
    digit_stems = [stem for stem in stems if stem.isdigit() and len(stem) <= 18]
    numbers = (np.array([int(stem) for stem in digit_stems], dtype=np.int64)
               if len(digit_stems) == len(stems) else None)
    images = np.load("labeled_32x32.npy", mmap_mode="r")
    impulse = np.empty(len(labels), dtype=np.float32)
    for start in range(0, len(labels), 8192):
        chunk = np.asarray(images[start:start + 8192])
        impulse[start:start + len(chunk)] = np.mean((chunk == 25) | (chunk == 245), axis=(1, 2))
    facing_path = Path("outputs/large/facing_labeled.npy")
    facing = np.load(facing_path) if facing_path.exists() else None

    submission_path = Path("outputs/large/submission_distill_fusion.csv")
    submission = pd.read_csv(submission_path) if submission_path.exists() else None

    banks = {}
    for name in ("unl_gallery_bank_95.npz", "transductive_bank_f0.npz",
                 "transductive_bank_p2imp_full.npz"):
        path = Path("outputs/large") / name
        if not path.exists():
            banks[name] = None
            continue
        with np.load(path) as bank:
            info = {"keys": list(bank.files), "images": int(bank["images"].shape[0])}
            if "confidence" in bank.files:
                conf = bank["confidence"]
                info["confidence_min"] = float(conf.min())
                info["confidence_at_least_0.99"] = int((conf >= 0.99).sum())
            banks[name] = info

    png = {"zip_present": Path("data/competition.zip").exists(), "samples": []}
    if png["zip_present"]:
        with zipfile.ZipFile("data/competition.zip") as archive:
            members = [n for n in archive.namelist() if n.lower().endswith(".png")]
            png["png_members"] = len(members)
            for name in members[:12]:
                with archive.open(name) as handle:
                    payload = handle.read(512)
                png["samples"].append({"name": name, "chunks": png_chunks(payload)})
    loose = list(Path("data").glob("*.png"))[:5]
    png["loose_png_sample"] = [str(p) for p in loose]

    report = {
        "rows": {"train": int(len(train)), "labeled_battles": int(known.sum()),
                 "unlabeled_battles": int((~known).sum()), "test": int(len(test))},
        "columns": {"train": list(train.columns), "test": list(test.columns)},
        "id_order": {
            "fraction_id1_less_than_id2": float(np.mean(gap > 0)),
            "fraction_id1_equals_id2": float(np.mean(gap == 0)),
            "gap_min": int(gap.min()),
            "gap_max": int(gap.max()),
            "gap_mean": float(gap.mean()),
            "unique_unordered_pairs": int(len(set(zip(id1.tolist(), id2.tolist())))),
        },
        "battle_order": {
            "lag1_result": lag1(y),
            "lag1_id1": lag1(id1.astype(np.float64)),
            "lag1_id2": lag1(id2.astype(np.float64)),
            "lag1_gap": lag1(gap.astype(np.float64)),
            "spearman_order_result": rank_corr(order, y),
            "spearman_order_id1": rank_corr(order, id1),
            "spearman_order_id2": rank_corr(order, id2),
            "spearman_order_gap": rank_corr(order, gap),
            "result_mean_by_decile": deciles(y),
            "id1_mean_by_decile": deciles(id1.astype(np.float64)),
        },
        "filenames": {
            "sorted_labeled_paths_match_npy": bool(path_match),
            "labeled_images": int(len(labels)),
            "path_count": int(len(paths)),
            "filename_example": paths[:3],
            "stem_example": stems[:3],
            "all_stems_are_short_integers": numbers is not None,
            "spearman_sorted_path_label": rank_corr(np.arange(len(labels)), labels) if path_match else None,
            "lag1_label_in_filename_order": lag1(labels.astype(np.float64)) if path_match else None,
            "lag1_impulse_in_filename_order": lag1(impulse.astype(np.float64)) if path_match else None,
            "spearman_number_label": rank_corr(numbers, labels) if path_match and numbers is not None else None,
            "spearman_number_impulse": rank_corr(numbers, impulse) if path_match and numbers is not None else None,
            "spearman_number_facing": (
                rank_corr(numbers, np.asarray(facing).reshape(-1))
                if path_match and numbers is not None and facing is not None
                and len(np.asarray(facing).reshape(-1)) == len(numbers) else None),
            "facing_shape": None if facing is None else list(np.asarray(facing).shape),
        },
        "result_shift": {
            "train_labeled_mean": float(y.mean()),
            "train_labeled_std": float(y.std()),
            "train_unlabeled_mean": float(train.loc[~known, "result"].mean()),
            "train_unlabeled_std": float(train.loc[~known, "result"].std()),
            "submission_mean": None if submission is None else float(submission.result.mean()),
            "submission_std": None if submission is None else float(submission.result.std()),
        },
        "gallery_banks": banks,
        "png": png,
    }
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
