"""Evaluate fixed deep-feature score fusion on widened WRN candidate pools."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=Path("outputs/large"))
    parser.add_argument("--run-name", default="feature_rerank_rot5_top20")
    parser.add_argument("--output", type=Path, default=Path(
        "outputs/large/feature_top20_cv.json"))
    args = parser.parse_args()
    rows = [np.load(args.input_dir / f"{args.run_name}_f{fold}.npz")
            for fold in range(3)]
    output = {"run_name": args.run_name, "betas": {}}
    for beta in (0, 8, 16, 24, 32, 40, 48):
        parts = []
        for fold, z in enumerate(rows):
            ids = z["candidates"]
            y = z["labels"]
            score = np.log(np.maximum(z["cnn_probs"], 1e-12)) + beta*z["knn1"]
            order = np.argsort(-score, axis=1)
            top10 = np.take_along_axis(ids, order[:, :10], axis=1)
            part = {"fold": fold, "count": len(y),
                    "top1_correct": int((top10[:, 0] == y).sum()),
                    "top10_correct": int((top10 == y[:, None]).any(1).sum()),
                    "top20_oracle_correct": int((ids == y[:, None]).any(1).sum())}
            parts.append(part)
        total = {field: sum(row[field] for row in parts)
                 for field in ("count", "top1_correct", "top10_correct",
                               "top20_oracle_correct")}
        output["betas"][str(beta)] = {"folds": parts, "overall": total}
        print(beta, total["top1_correct"]/total["count"],
              total["top10_correct"]/total["count"], flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    main()
