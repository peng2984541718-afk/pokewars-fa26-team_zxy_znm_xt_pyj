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


def _penultimate(model: torch.nn.Module, model_name: str, xv: torch.Tensor) -> torch.Tensor:
    if model_name == "resnext29_8x64d":
        h = model.stages(F.relu(model.bn(model.stem(xv))))
        return h.mean(dim=(-2, -1))
    h = model.stages(model.stem(xv))
    return F.relu(model.bn(h)).mean(dim=(-2, -1))


@torch.inference_mode()
def extract_rot5(model: torch.nn.Module, model_name: str, images: torch.Tensor,
                 batch_size: int) -> torch.Tensor:
    """Return L2-normalized 6-view embeddings on CPU as float16."""
    width = 4096 if model_name == "resnext29_8x64d" else 640
    features = torch.empty((len(images), width), device="cpu", dtype=torch.float16)
    for start in range(0, len(images), batch_size):
        end = min(start + batch_size, len(images))
        x0 = preprocess(images[start:end].cuda(non_blocking=True))
        acc = 0
        for base in (x0, torch.flip(x0, dims=[-1])):
            for view in (None, "rotate_minus5", "rotate_plus5"):
                xv = base if view is None else transform(base, view)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    h = _penultimate(model, model_name, xv)
                acc = acc + F.normalize(h.float(), dim=1)
        features[start:end].copy_(F.normalize(acc, dim=1).half().cpu())
    return features


@torch.inference_mode()
def variable_gallery_scores(gallery_feat: torch.Tensor, gallery_labels: np.ndarray,
                            gallery_clean_w: np.ndarray, query_feat: torch.Tensor,
                            candidates_zero: np.ndarray, batch_size: int = 128,
                            exclude_index: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """Score candidates against the gallery.

    `exclude_index[i]` is the gallery row that is the query itself, or -1.
    That row is removed from kNN and from the class prototype.
    """
    counts = np.bincount(gallery_labels, minlength=NUM_CLASSES)
    max_c = int(counts.max())
    order = np.argsort(gallery_labels, kind="stable")
    sorted_labels = gallery_labels[order]
    starts = np.r_[0, np.cumsum(counts[:-1])]
    slot = np.arange(len(order)) - starts[sorted_labels]
    slot_of = np.empty(len(gallery_labels), dtype=np.int64)
    slot_of[order] = slot

    width = gallery_feat.shape[-1]
    padded = torch.zeros((NUM_CLASSES, max_c, width), device="cuda", dtype=torch.float16)
    valid_mask = torch.zeros((NUM_CLASSES, max_c), device="cuda", dtype=torch.bool)
    raw_sum = torch.zeros((NUM_CLASSES, width), device="cuda", dtype=torch.float32)
    clean_sum = torch.zeros((NUM_CLASSES, width), device="cuda", dtype=torch.float32)
    for start in range(0, len(order), 16384):
        end = min(start + 16384, len(order))
        rows = order[start:end]
        cls_b = torch.as_tensor(sorted_labels[start:end], device="cuda", dtype=torch.long)
        slot_b = torch.as_tensor(slot[start:end], device="cuda", dtype=torch.long)
        cw_b = torch.as_tensor(gallery_clean_w[rows], device="cuda", dtype=torch.float32)
        gf_b = gallery_feat[rows].cuda(non_blocking=True)
        padded[cls_b, slot_b] = gf_b
        valid_mask[cls_b, slot_b] = True
        sf_b = gf_b.float()
        raw_sum.index_add_(0, cls_b, sf_b)
        clean_sum.index_add_(0, cls_b, sf_b * cw_b[:, None])
    proto = F.normalize(raw_sum, dim=1)
    clean_proto = F.normalize(clean_sum, dim=1)

    # Keep the inner gather well under 2 GiB even for 4096-d ResNeXt embeddings.
    step = batch_size if width <= 640 else min(batch_size, 16)
    n, top_k = candidates_zero.shape
    out = {k: np.empty((n, top_k), dtype=np.float32)
           for k in ("prototype", "clean_prototype", "knn1", "knn5", "knn15")}
    for start in range(0, n, step):
        end = min(start + step, n)
        c_idx = torch.as_tensor(candidates_zero[start:end], device="cuda", dtype=torch.long)
        q = query_feat[start:end].cuda(non_blocking=True).float()
        proto_score = (proto[c_idx] * q[:, None, :]).sum(-1)
        clean_score = (clean_proto[c_idx] * q[:, None, :]).sum(-1)
        sim = torch.einsum("bkcd,bd->bkc", padded[c_idx], q.half()).float()
        sim = sim.masked_fill(~valid_mask[c_idx], -1e4)
        if exclude_index is not None:
            for row, gidx in enumerate(exclude_index[start:end].tolist()):
                if gidx < 0:
                    continue
                cls = int(gallery_labels[gidx])
                cols = torch.nonzero(c_idx[row] == cls, as_tuple=False).flatten()
                if len(cols) == 0:
                    continue
                sim[row, cols, int(slot_of[gidx])] = -1e4
                if int(counts[cls]) <= 1:
                    proto_score[row, cols] = -1.0
                    clean_score[row, cols] = -1.0
                    continue
                removed = gallery_feat[gidx].cuda().float()
                weight = float(gallery_clean_w[gidx])
                fixed = F.normalize(raw_sum[cls] - removed, dim=0)
                fixed_clean = F.normalize(clean_sum[cls] - removed * weight, dim=0)
                proto_score[row, cols] = torch.dot(fixed, q[row])
                clean_score[row, cols] = torch.dot(fixed_clean, q[row])
        out["prototype"][start:end] = proto_score.cpu().numpy()
        out["clean_prototype"][start:end] = clean_score.cpu().numpy()
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
    parser.add_argument("--resnext-prefixes", nargs="*", default=[])
    parser.add_argument("--blend-weights", type=float, nargs=4, default=(0.35, 0.22, 0.33, 0.10),
                        metavar=("WRN28", "WRN40", "DENSE", "FOURTH"),
                        help="Candidate-blend weights. The fourth file is ConvNeXt or ResNeXt")
    parser.add_argument("--extra-bank", type=Path, default=None,
                        help="Optional fold-safe bank .npz (images, labels 0-based) added to kNN gallery")
    parser.add_argument("--loo", action="store_true",
                        help="Drop a query that is also stored in the extra bank from its own kNN and prototype")
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
    blend_weights = np.asarray(args.blend_weights, dtype=np.float64)
    blend_weights /= blend_weights.sum()
    blend_w = {"wrn28": float(blend_weights[0]), "wrn40": float(blend_weights[1]),
               "dense": float(blend_weights[2]), "conv": float(blend_weights[3])}
    model_specs = [("wrn28_10", p) for p in args.wrn28_prefixes] + [("wrn40_10", p) for p in args.wrn40_prefixes]
    model_specs += [("resnext29_8x64d", p) for p in args.resnext_prefixes]

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
            gf = extract_rot5(model, m_name, gal_t, args.batch_size)
            qf = extract_rot5(model, m_name, qry_t, args.batch_size)
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
        extra_slots = None
        if args.extra_bank is not None and args.extra_bank.exists():
            with np.load(args.extra_bank) as bk:
                extra_imgs = bk["images"]
                extra_labels = bk["labels"].astype(np.int64)
        if args.loo and extra_imgs is not None:
            query_key = {row.tobytes(): i for i, row in enumerate(
                np.ascontiguousarray(qry_imgs).reshape(len(qry_imgs), -1))}
            extra_slots = np.full(len(qry_imgs), -1, dtype=np.int64)
            extra_flat = np.ascontiguousarray(extra_imgs).reshape(len(extra_imgs), -1)
            matched = 0
            for bank_i, row in enumerate(extra_flat):
                query_i = query_key.get(row.tobytes())
                if query_i is not None:
                    extra_slots[query_i] = bank_i
                    matched += 1
            print(f"leave-one-out matched {matched} query images inside the extra bank", flush=True)

        knn_acc = {k: np.zeros((len(qry_imgs), args.top_k), dtype=np.float32)
                   for k in ("prototype", "clean_prototype", "knn1", "knn5", "knn15")}
        partial_path = args.output.with_name(args.output.stem + ".partial.npz")
        done_ckpts: set[str] = set()
        if partial_path.exists():
            try:
                with np.load(partial_path, allow_pickle=False) as pz:
                    done_ckpts = set(str(x) for x in pz["done_ckpts"].tolist())
                    for k in knn_acc:
                        knn_acc[k] = pz[k].astype(np.float32)
                print(f"Resumed {len(done_ckpts)}/{len(args.folds) * len(model_specs)} models from {partial_path}", flush=True)
            except Exception as exc:
                print(f"Ignoring unreadable partial checkpoint {partial_path}: {exc}", flush=True)
                done_ckpts = set()
        total_models = len(args.folds) * len(model_specs)
        for fold in args.folds:
            fold_specs = [(m_name, prefix, checkpoint_path(args.checkpoint_dir, prefix, fold))
                          for m_name, prefix in model_specs]
            if all(c.name in done_ckpts for _, _, c in fold_specs):
                print(f"query ({len(qry_imgs)}): fold {fold} already complete in partial checkpoint", flush=True)
                continue
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
            for m_name, prefix, ckpt in fold_specs:
                if ckpt.name in done_ckpts:
                    print(f"query ({len(qry_imgs)}): skipping already scored {ckpt.name}", flush=True)
                    continue
                model = make_model(m_name, pretrained=False).cuda().eval()
                model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
                gf = extract_rot5(model, m_name, gal_t, args.batch_size)
                qf = extract_rot5(model, m_name, qry_t, args.batch_size)
                exclude = None
                if args.loo and extra_slots is not None:
                    exclude = np.full(len(qry_imgs), -1, dtype=np.int64)
                    present = extra_slots >= 0
                    exclude[present] = len(tr_idx) + extra_slots[present]
                one = variable_gallery_scores(
                    gf, gal_labels, gal_clean_w, qf, cand_zero, exclude_index=exclude)
                for k in knn_acc:
                    knn_acc[k] += one[k] / total_models
                del model, gf, qf
                torch.cuda.empty_cache()
                done_ckpts.add(ckpt.name)
                tmp_partial = partial_path.with_name(partial_path.stem + ".tmp.npz")
                np.savez(tmp_partial, done_ckpts=np.array(sorted(done_ckpts)), **knn_acc)
                tmp_partial.replace(partial_path)
                print(f"query ({len(qry_imgs)}): scored {ckpt.name} ({len(done_ckpts)}/{total_models})", flush=True)
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
        if partial_path.exists():
            partial_path.unlink()
        print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
