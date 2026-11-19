"""Four-backbone top-K candidate selection, rot5-TTA kNN gallery, and noise features.

Upgrades feature_rerank.py and predict_test_top20_features.py by:
1. Selecting top-K (default 40) candidates from the 4-backbone probability blend
   (WRN-28 + WRN-40 + DenseNet-169 + ConvNeXt V2), cutting missing-pool errors ~3x.
2. Averaging penultimate embeddings over {-5°, 0°, +5°} views before L2 norm.
3. Supporting variable-count galleries (adding fold-safe pseudo-labeled banks)
   and SNR-weighted clean prototypes alongside knn1, knn5, knn15, prototype.
4. Attaching per-image physical degradation stats (noise, contrast, impulse).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from degradation import clean_impulses_gpu, estimate_params, laplacian_noise
from evaluate_wrn_tta import transform
from export_probs_rot5 import checkpoint_path
from image_experiment import NUM_CLASSES, SEED, stratified_folds
from train_large_image import make_model, preprocess


@torch.inference_mode()
def extract_rot5(model: torch.nn.Module, images: torch.Tensor, batch_size: int) -> torch.Tensor:
    features = torch.empty((len(images), 640), device="cuda", dtype=torch.float16)
    for start in range(0, len(images), batch_size):
        end = min(start + batch_size, len(images))
        x0 = preprocess(images[start:end])
        acc = 0
        for view in (None, "rotate_minus5", "rotate_plus5"):
            xv = x0 if view is None else transform(x0, view)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                h = model.stages(model.stem(xv))
                h = F.relu(model.bn(h)).mean(dim=(-2, -1))
            acc = acc + F.normalize(h.float(), dim=1)
        features[start:end] = F.normalize(acc, dim=1).half()
    return features


@torch.inference_mode()
def variable_gallery_scores(gallery_feat: torch.Tensor, gallery_labels: np.ndarray,
                            gallery_clean_w: np.ndarray, query_feat: torch.Tensor,
                            candidates_zero: np.ndarray, batch_size: int = 128) -> dict[str, np.ndarray]:
    counts = np.bincount(gallery_labels, minlength=NUM_CLASSES)
    max_c = int(counts.max())
    order = np.argsort(gallery_labels, kind="stable")
    sorted_labels = gallery_labels[order]
    # Vectorized slot index within each class: 0 .. count[c]-1
    starts = np.r_[0, np.cumsum(counts[:-1])]
    slot = np.arange(len(order)) - starts[sorted_labels]

    order_t = torch.as_tensor(order, device="cuda", dtype=torch.long)
    cls_t = torch.as_tensor(sorted_labels, device="cuda", dtype=torch.long)
    slot_t = torch.as_tensor(slot, device="cuda", dtype=torch.long)
    cw_t = torch.as_tensor(gallery_clean_w[order], device="cuda", dtype=torch.float32)

    padded = torch.zeros((NUM_CLASSES, max_c, 640), device="cuda", dtype=torch.float16)
    valid_mask = torch.zeros((NUM_CLASSES, max_c), device="cuda", dtype=torch.bool)
    padded[cls_t, slot_t] = gallery_feat[order_t]
    valid_mask[cls_t, slot_t] = True

    raw_sum = torch.zeros((NUM_CLASSES, 640), device="cuda", dtype=torch.float32)
    clean_sum = torch.zeros((NUM_CLASSES, 640), device="cuda", dtype=torch.float32)
    sf = gallery_feat[order_t].float()
    raw_sum.index_add_(0, cls_t, sf)
    clean_sum.index_add_(0, cls_t, sf * cw_t[:, None])
    proto = F.normalize(raw_sum, dim=1)
    clean_proto = F.normalize(clean_sum, dim=1)

    n, top_k = candidates_zero.shape
    out = {k: np.empty((n, top_k), dtype=np.float32)
           for k in ("prototype", "clean_prototype", "knn1", "knn5", "knn15")}
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        c_idx = torch.as_tensor(candidates_zero[start:end], device="cuda", dtype=torch.long)
        q = query_feat[start:end].float()
        out["prototype"][start:end] = (proto[c_idx] * q[:, None, :]).sum(-1).cpu().numpy()
        out["clean_prototype"][start:end] = (clean_proto[c_idx] * q[:, None, :]).sum(-1).cpu().numpy()
        neigh = padded[c_idx].float()
        vmask = valid_mask[c_idx]
        sim = torch.einsum("bkcd,bd->bkc", neigh, q)
        sim = sim.masked_fill(~vmask, -1e4)
        out["knn1"][start:end] = sim.max(dim=-1).values.cpu().numpy()
        out["knn5"][start:end] = sim.topk(min(5, max_c), dim=-1).values.mean(-1).cpu().numpy()
        out["knn15"][start:end] = sim.topk(min(15, max_c), dim=-1).values.mean(-1).cpu().numpy()
    return out


def image_quality_features(images_np: np.ndarray) -> np.ndarray:
    """Return (N, 4) float32: [laplacian_noise, contrast, sigma_est, impulse_rate]."""
    lap = laplacian_noise(images_np).astype(np.float32)
    p = estimate_params(images_np)
    contrast = p[:, 0] - p[:, 1]
    impulse = p[:, 3] + p[:, 4]
    return np.stack([lap, contrast, p[:, 2], impulse], axis=1).astype(np.float32)


def select_blend_candidates(probs_dict: dict[str, np.ndarray], blend_weights: dict[str, float],
                            indices: np.ndarray | None, top_k: int):
    rows = slice(None) if indices is None else indices
    blend = None
    per_arch = {}
    for name, mat in probs_dict.items():
        sub = np.asarray(mat[rows], dtype=np.float32)
        per_arch[name] = sub
        blend = blend_weights[name] * sub if blend is None else blend + blend_weights[name] * sub
    part = np.argpartition(blend, -top_k, axis=1)[:, -top_k:]
    part_vals = np.take_along_axis(blend, part, axis=1)
    order = np.argsort(-part_vals, axis=1)
    cand_zero = np.take_along_axis(part, order, axis=1)
    blend_topk = np.take_along_axis(blend, cand_zero, axis=1)
    oop_mass = np.clip(1.0 - blend_topk.sum(1), 1e-6, 1.0).astype(np.float32)

    arch_stats = {}
    for name, sub in per_arch.items():
        p_clip = np.clip(sub, 1e-8, 1.0)
        ent = -np.sum(p_clip * np.log(p_clip), axis=1).astype(np.float32)
        top2 = np.partition(sub, -2, axis=1)[:, -2:]
        top1_p = top2[:, 1].astype(np.float32)
        margin = (top2[:, 1] - top2[:, 0]).astype(np.float32)
        arch_stats[f"{name}_probs"] = np.take_along_axis(sub, cand_zero, axis=1).astype(np.float16)
        arch_stats[f"{name}_img"] = np.stack([top1_p, margin, ent], axis=1).astype(np.float16)
    return cand_zero, blend_topk.astype(np.float16), oop_mass, arch_stats


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("oof-fold", "query"), required=True)
    parser.add_argument("--fold", type=int, choices=(0, 1, 2), default=0)
    parser.add_argument("--folds", type=int, nargs="+", choices=(0, 1, 2), default=[0, 1, 2])
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--data-dir", type=Path, default=Path("."))
    parser.add_argument("--query-dir", type=Path, default=Path("."))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("outputs/large"))
    parser.add_argument("--wrn28-prefixes", nargs="+", required=True)
    parser.add_argument("--wrn40-prefixes", nargs="+", default=["wrn40_10"])
    parser.add_argument("--wrn28-probs", type=Path, required=True)
    parser.add_argument("--wrn40-probs", type=Path, required=True)
    parser.add_argument("--dense-probs", type=Path, required=True)
    parser.add_argument("--conv-probs", type=Path, required=True)
    parser.add_argument("--extra-bank", type=Path, default=None,
                        help="Optional fold-safe bank .npz (images, labels 0-based) added to kNN gallery")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--clean-impulses", action="store_true",
                        help="Repair exact 25/245 impulse pixels before feature extraction")
    args = parser.parse_args()

    labels = np.load(args.data_dir / "labeled_ids.npy").astype(np.int64)
    labeled_np = np.asarray(np.load(args.data_dir / "labeled_32x32.npy"))
    folds = stratified_folds(labels, 3, SEED)

    probs_dict = {
        "wrn28": np.load(args.wrn28_probs, mmap_mode="r"),
        "wrn40": np.load(args.wrn40_probs, mmap_mode="r"),
        "dense": np.load(args.dense_probs, mmap_mode="r"),
        "conv": np.load(args.conv_probs, mmap_mode="r"),
    }
    blend_w = {"wrn28": 0.60, "wrn40": 0.12, "dense": 0.16, "conv": 0.12}
    model_specs = [("wrn28_10", p) for p in args.wrn28_prefixes] + [("wrn40_10", p) for p in args.wrn40_prefixes]

    if args.mode == "oof-fold":
        train_idx = np.flatnonzero(folds != args.fold)
        valid_idx = np.flatnonzero(folds == args.fold)
        cand_zero, blend_topk, oop_mass, arch_stats = select_blend_candidates(
            probs_dict, blend_w, valid_idx, args.top_k)

        gal_imgs = labeled_np[train_idx]
        gal_labels = labels[train_idx]
        if args.extra_bank is not None and args.extra_bank.exists():
            with np.load(args.extra_bank) as bk:
                gal_imgs = np.concatenate([gal_imgs, bk["images"]], axis=0)
                gal_labels = np.concatenate([gal_labels, bk["labels"].astype(np.int64)], axis=0)
        gal_lap = laplacian_noise(gal_imgs)
        gal_clean_w = np.exp(-np.clip((gal_lap - 18.0) / 10.0, 0.0, 4.0)).astype(np.float32)

        qry_imgs = labeled_np[valid_idx]
        img_qual = image_quality_features(qry_imgs)
        gal_t = torch.as_tensor(gal_imgs, device="cuda")
        qry_t = torch.as_tensor(qry_imgs, device="cuda")
        if args.clean_impulses:
            gal_t = clean_impulses_gpu(gal_t)
            qry_t = clean_impulses_gpu(qry_t)

        knn_acc = {k: np.zeros((len(valid_idx), args.top_k), dtype=np.float32)
                   for k in ("prototype", "clean_prototype", "knn1", "knn5", "knn15")}
        for m_name, prefix in model_specs:
            ckpt = checkpoint_path(args.checkpoint_dir, prefix, args.fold)
            model = make_model(m_name, pretrained=False).cuda().eval()
            model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
            gf = extract_rot5(model, gal_t, args.batch_size)
            qf = extract_rot5(model, qry_t, args.batch_size)
            one = variable_gallery_scores(gf, gal_labels, gal_clean_w, qf, cand_zero)
            for k in knn_acc:
                knn_acc[k] += one[k] / len(model_specs)
            del model, gf, qf
            torch.cuda.empty_cache()
            print(f"fold {args.fold}: gallery scored {ckpt.name}", flush=True)

        true_f = labels[valid_idx]
        recall = float((cand_zero == true_f[:, None]).any(1).mean())
        top1 = float((cand_zero[:, 0] == true_f).mean())
        print(f"fold {args.fold}: blend top1={top1:.5f} top{args.top_k}_recall={recall:.5f}", flush=True)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.output,
            valid_indices=valid_idx,
            labels=(true_f + 1).astype(np.uint16),
            candidates=(cand_zero + 1).astype(np.uint16),
            blend_probs=blend_topk,
            oop_mass=oop_mass,
            image_quality=img_qual.astype(np.float16),
            **arch_stats,
            **{k: v.astype(np.float16) for k, v in knn_acc.items()},
        )
    else:
        qry_imgs = np.asarray(np.load(args.query_dir / "test_32x32.npy"))
        cand_zero, blend_topk, oop_mass, arch_stats = select_blend_candidates(
            probs_dict, blend_w, None, args.top_k)
        img_qual = image_quality_features(qry_imgs)
        qry_t = torch.as_tensor(qry_imgs, device="cuda")
        if args.clean_impulses:
            qry_t = clean_impulses_gpu(qry_t)

        extra_imgs = extra_labels = None
        if args.extra_bank is not None and args.extra_bank.exists():
            with np.load(args.extra_bank) as bk:
                extra_imgs = bk["images"]
                extra_labels = bk["labels"].astype(np.int64)

        knn_acc = {k: np.zeros((len(qry_imgs), args.top_k), dtype=np.float32)
                   for k in ("prototype", "clean_prototype", "knn1", "knn5", "knn15")}
        total_models = len(args.folds) * len(model_specs)
        for fold in args.folds:
            tr_idx = np.flatnonzero(folds != fold)
            gal_imgs = labeled_np[tr_idx]
            gal_labels = labels[tr_idx]
            if extra_imgs is not None:
                gal_imgs = np.concatenate([gal_imgs, extra_imgs], axis=0)
                gal_labels = np.concatenate([gal_labels, extra_labels], axis=0)
            gal_lap = laplacian_noise(gal_imgs)
            gal_clean_w = np.exp(-np.clip((gal_lap - 18.0) / 10.0, 0.0, 4.0)).astype(np.float32)
            gal_t = torch.as_tensor(gal_imgs, device="cuda")
            if args.clean_impulses:
                gal_t = clean_impulses_gpu(gal_t)
            for m_name, prefix in model_specs:
                ckpt = checkpoint_path(args.checkpoint_dir, prefix, fold)
                model = make_model(m_name, pretrained=False).cuda().eval()
                model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
                gf = extract_rot5(model, gal_t, args.batch_size)
                qf = extract_rot5(model, qry_t, args.batch_size)
                one = variable_gallery_scores(gf, gal_labels, gal_clean_w, qf, cand_zero)
                for k in knn_acc:
                    knn_acc[k] += one[k] / total_models
                del model, gf, qf
                torch.cuda.empty_cache()
                print(f"query ({len(qry_imgs)}): scored {ckpt.name}", flush=True)
            del gal_t

        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.output,
            candidates=(cand_zero + 1).astype(np.uint16),
            blend_probs=blend_topk,
            oop_mass=oop_mass,
            image_quality=img_qual.astype(np.float16),
            **arch_stats,
            **{k: v.astype(np.float16) for k, v in knn_acc.items()},
        )
        print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
