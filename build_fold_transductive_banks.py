"""Build strictly fold-safe ~195k-image transductive banks (Unlabeled + Test).

For each fold f in {0, 1, 2}, uses ONLY fold-f checkpoints (which never saw
fold-f validation images) to score the 68,675 unlabeled and 160,000 test
images, runs bidirectional constrained BP + test cavity, and exports high-
confidence pseudo-labeled sprites to transductive_bank_f{f}.npz.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from degradation import clean_impulses_gpu
from export_probs_rot5 import checkpoint_path, score_model
from image_experiment import NUM_CLASSES, SEED, stratified_folds
from joint_constrained_inference import run_unlabeled_bp, target_cavity_predict
from train_large_image import make_model


@torch.inference_mode()
def fold_blend_topk(fold: int, images_np: np.ndarray, checkpoint_dir: Path,
                    top_k: int = 30, batch_size: int = 256, clean_impulses: bool = True):
    specs = [
        ("wrn28_10", "wrn28_10_mild_pseudo", 0.18),
        ("wrn28_10", "wrn28_10_mild_pseudo_synth", 0.20),
        ("wrn28_10", "wrn28_10_mild_pseudo_ctrl", 0.20),
        ("wrn40_10", "wrn40_10", 0.12),
        ("densenet169_cifar", "densenet169_cifar", 0.17),
        ("convnextv2_nano", "convnextv2_scratch64", 0.13),
    ]
    imgs_t = torch.as_tensor(images_np, device="cuda")
    if clean_impulses:
        imgs_t = clean_impulses_gpu(imgs_t)
    idx = np.arange(len(images_np))
    blend = np.zeros((len(images_np), NUM_CLASSES), dtype=np.float32)
    for model_name, prefix, weight in specs:
        ckpt = checkpoint_path(checkpoint_dir, prefix, fold)
        model = make_model(model_name, pretrained=False).cuda().eval()
        model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
        blend += weight * score_model(model, model_name, imgs_t, idx, batch_size)
        del model
        torch.cuda.empty_cache()
    del imgs_t

    part = np.argpartition(blend, -top_k, axis=1)[:, -top_k:]
    pvals = np.take_along_axis(blend, part, axis=1)
    order = np.argsort(-pvals, axis=1)
    cand = np.take_along_axis(part, order, axis=1) + 1
    probs = np.take_along_axis(blend, cand - 1, axis=1)
    oop = np.clip(1.0 - probs.sum(1), 1e-6, 0.5).astype(np.float32)
    post = probs / np.maximum(probs.sum(1, keepdims=True), 1e-12)
    return cand.astype(np.int64), post.astype(np.float32), oop


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", type=int, nargs="+", choices=(0, 1, 2), default=[0, 1, 2])
    parser.add_argument("--data-dir", type=Path, default=Path("."))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("outputs/large"))
    parser.add_argument("--unl-min-conf", type=float, default=0.92)
    parser.add_argument("--test-min-conf", type=float, default=0.94)
    args = parser.parse_args()

    rows = pd.read_csv(args.data_dir / "data/train.csv.gz")
    identity = {}
    for side in (1, 2):
        for p, v in zip(rows[f"noisy_image{side}"], rows[f"pokemon_id{side}"]):
            if pd.notna(v):
                identity[p] = int(v)
    paths = sorted(identity)
    true = np.array([identity[p] for p in paths], dtype=np.int64)
    img_folds = stratified_folds(true - 1, 3, SEED)
    p_idx = pd.Index(paths)
    f1, f2 = p_idx.get_indexer(rows.noisy_image1), p_idx.get_indexer(rows.noisy_image2)
    known = (f1 >= 0) & (f2 >= 0)
    f1, f2 = f1[known], f2[known]
    ka_all, kb_all = true[f1], true[f2]

    unk = rows.pokemon_id1.isna() & rows.pokemon_id2.isna()
    u_idx = pd.Index((args.data_dir / "data/unlabeled_paths.txt").read_text().split())
    eu = u_idx.get_indexer(rows.noisy_image1[unk])
    ev = u_idx.get_indexer(rows.noisy_image2[unk])
    uy = rows.result[unk].to_numpy(np.float32)

    test_paths = (args.data_dir / "test_paths.txt").read_text().splitlines()
    t_idx = pd.Index(test_paths)
    test_df = pd.read_csv(args.data_dir / "data/test.csv.gz")
    t1 = t_idx.get_indexer(test_df.noisy_image1)
    t2 = t_idx.get_indexer(test_df.noisy_image2)

    unl_imgs = np.asarray(np.load(Path("unl") / "test_32x32.npy"))
    test_imgs = np.asarray(np.load(args.data_dir / "test_32x32.npy"))

    for fold in args.folds:
        out_path = args.checkpoint_dir / f"transductive_bank_f{fold}.npz"
        if out_path.exists():
            print(f"Skipping existing {out_path}", flush=True)
            continue
        pair_holdout = (img_folds[f1] == fold) & (img_folds[f2] == fold)
        known_allowed = np.ones((1026, 1026), dtype=np.float32)
        np.fill_diagonal(known_allowed, 0.0)
        ka, kb = ka_all[~pair_holdout], kb_all[~pair_holdout]
        known_allowed[ka, kb] = 0.0
        known_allowed[kb, ka] = 0.0

        table_path = (args.checkpoint_dir / "pair_semi_nofold0_new_table_round1.npy"
                      if fold == 0 else args.checkpoint_dir / "pair_semi_v2_full_table_round1.npy")
        table = np.load(table_path)

        u_cand, u_post, _ = fold_blend_topk(fold, unl_imgs, args.checkpoint_dir)
        t_cand, t_post, t_oop = fold_blend_topk(fold, test_imgs, args.checkpoint_dir)

        u_node, _, unl_lnot = run_unlabeled_bp(
            u_cand, u_post, eu, ev, uy, table, known_allowed, device="cuda")
        _, _, test_lnot, m1, m2 = target_cavity_predict(
            t_post, t_oop, t_cand, t1, t2, known_allowed, unl_lnot, table, device="cuda")
        u_node, _, unl_lnot = run_unlabeled_bp(
            u_cand, u_post, eu, ev, uy, table, known_allowed,
            external_log_avail=test_lnot, device="cuda")
        _, _, _, m1, m2 = target_cavity_predict(
            t_post, t_oop, t_cand, t1, t2, known_allowed, unl_lnot, table, device="cuda")

        u_best = u_node.argmax(1)
        u_conf = u_node[np.arange(len(u_node)), u_best]
        u_lab0 = u_cand[np.arange(len(u_node)), u_best] - 1
        u_keep = u_conf >= args.unl_min_conf

        t_marg = np.zeros_like(t_post)
        t_marg[t1] = m1
        t_marg[t2] = m2
        t_best = t_marg.argmax(1)
        t_conf = t_marg[np.arange(len(t_marg)), t_best]
        t_lab0 = t_cand[np.arange(len(t_marg)), t_best] - 1
        t_keep = t_conf >= args.test_min_conf

        bank_imgs = np.concatenate([unl_imgs[u_keep], test_imgs[t_keep]], axis=0)
        bank_labs = np.concatenate([u_lab0[u_keep], t_lab0[t_keep]], axis=0).astype(np.int16)
        bank_conf = np.concatenate([u_conf[u_keep], t_conf[t_keep]], axis=0).astype(np.float32)
        np.savez_compressed(out_path, images=bank_imgs, labels=bank_labs, confidence=bank_conf)
        summary = {
            "fold": fold,
            "unlabeled_kept": int(u_keep.sum()),
            "test_kept": int(t_keep.sum()),
            "total_images": int(len(bank_imgs)),
            "min_per_class": int(np.bincount(bank_labs, minlength=NUM_CLASSES).min()),
            "max_per_class": int(np.bincount(bank_labs, minlength=NUM_CLASSES).max()),
        }
        out_path.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
        print("Built fold-safe transductive bank:", json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
