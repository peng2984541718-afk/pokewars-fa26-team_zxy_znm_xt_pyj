"""Mix the 79.9 ranker with a distilled student, inside the same 40 candidates."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def temper_rows(p: np.ndarray, temperature: float) -> np.ndarray:
    """Soften or sharpen a row-stochastic matrix. T=1 leaves the rows unchanged."""
    if abs(temperature - 1.0) <= 1e-8:
        return p
    logits = np.log(np.clip(p.astype(np.float64), 1e-12, None)) / temperature
    logits -= logits.max(axis=1, keepdims=True)
    out = np.exp(logits)
    out /= np.maximum(out.sum(axis=1, keepdims=True), 1e-12)
    return out.astype(np.float32)


def fuse_posteriors(post: np.ndarray, student: np.ndarray, alpha: float,
                    mode: str = "linear", temperature: float = 1.0) -> np.ndarray:
    """Mix ranker and student distributions on the same candidate rows.

    `alpha` is the student weight. Linear fusion is the convex combination
    `(1 - alpha) * p_old + alpha * p_new`, with `p_new` tempered first.
    Log fusion is `softmax((1 - alpha) * log p_old + alpha * log p_new / T)`.
    """
    if mode == "linear" and abs(temperature - 1.0) <= 1e-8:
        mixed = (1.0 - alpha) * post + alpha * student
        mixed /= np.maximum(mixed.sum(axis=1, keepdims=True), 1e-8)
        return np.asarray(mixed, dtype=np.float32)
    elif mode == "linear":
        cooled = temper_rows(student, temperature)
        mixed = (1.0 - alpha) * post + alpha * cooled
    elif mode == "log":
        old = np.log(np.clip(post.astype(np.float64), 1e-12, None))
        new = np.log(np.clip(student.astype(np.float64), 1e-12, None))
        logits = (1.0 - alpha) * old + (alpha / temperature) * new
        logits -= logits.max(axis=1, keepdims=True)
        mixed = np.exp(logits)
    else:
        raise ValueError(f"unknown fusion mode {mode!r}")
    mixed = np.asarray(mixed, dtype=np.float64)
    mixed /= np.maximum(mixed.sum(axis=1, keepdims=True), 1e-8)
    return mixed.astype(np.float32)


def mix(ranker: Path, probs: list[tuple[Path, float]], out: Path,
        rows: np.ndarray | None, weight: float, mode: str = "linear",
        temperature: float = 1.0) -> None:
    with np.load(ranker) as z:
        data = {k: z[k] for k in z.files}
    cand = data["candidates"].astype(np.int64) - 1
    post = data["posteriors"].astype(np.float32)
    if rows is None:
        rows = np.arange(len(post))
    student = None
    for path, w in probs:
        mat = np.load(path, mmap_mode="r")
        sub = np.asarray(mat[rows], dtype=np.float32)
        student = w * sub if student is None else student + w * sub
    gathered = np.take_along_axis(student, cand, axis=1)
    gathered /= np.maximum(gathered.sum(1, keepdims=True), 1e-8)
    mixed = fuse_posteriors(post, gathered, weight, mode, temperature)
    data["posteriors"] = mixed
    np.savez_compressed(out, **data)
    moved = int((mixed.argmax(1) != post.argmax(1)).sum())
    print(f"saved {out.name} mode={mode} alpha={weight} T={temperature} "
          f"moved={moved}/{len(post)}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ranker", type=Path, required=True)
    parser.add_argument("--probs", type=Path, nargs="+", required=True)
    parser.add_argument("--prob-weights", type=float, nargs="+", required=True)
    parser.add_argument("--rows", type=Path, default=None,
                        help="npy of row indices into the probability files")
    parser.add_argument("--weight", type=float, default=0.25,
                        help="Student weight alpha in (1 - alpha) * old + alpha * new")
    parser.add_argument("--mode", choices=("linear", "log"), default="linear")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Temperature on the student. Linear tempers p_new first; "
                             "log divides the student log term by T")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.probs) != len(args.prob_weights):
        parser.error("one weight per probability file")
    if args.temperature <= 0:
        parser.error("temperature must be positive")
    if not 0.0 <= args.weight <= 1.0:
        parser.error("weight must be in [0, 1]")
    weights = np.array(args.prob_weights, dtype=np.float64)
    weights /= weights.sum()
    rows = None if args.rows is None else np.load(args.rows).astype(np.int64)
    mix(args.ranker, list(zip(args.probs, weights)), args.output, rows,
        args.weight, args.mode, args.temperature)


if __name__ == "__main__":
    main()
