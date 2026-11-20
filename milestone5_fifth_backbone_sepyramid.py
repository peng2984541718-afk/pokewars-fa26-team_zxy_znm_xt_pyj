"""Milestone 5: 5th Orthogonal 32x32 Backbone — Additive SE-PyramidNet with
zero-padded avgpool shortcuts and stochastic depth.

Strictly gated on Fold 0 first (both 6-view TTA top-1 and end-to-end Fold-0
battle test_score). Trains Folds 1 & 2 and exports `submission_rx_v5_5arch.csv`
only if Fold 0 clears both gates.
"""
from __future__ import annotations

import csv
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from degradation import clean_impulses_gpu
from evaluate_soft_score import DENOMINATOR, rmse
from export_probs_rot5 import score_model
from image_experiment import NUM_CLASSES, SEED, stratified_folds
from joint_constrained_inference import target_cavity_predict
from milestone1_rx_fusion_and_pack import gather_cand
from milestone3_inference_calibration import (
    adaptive_power_scale,
    run_unlabeled_bp_hetero,
    z_lap_from_gallery,
)
from mix_distill_posteriors import fuse_posteriors
from train_large_image_external import augment, lr_at_step, preprocess, soft_cross_entropy

L = Path("outputs/large")


def score_fn(val: float) -> float:
    return 100.0 * (1.0 - val / DENOMINATOR)


class SEModule(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        mid = max(channels // reduction, 16)
        self.fc1 = nn.Linear(channels, mid)
        self.fc2 = nn.Linear(mid, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = x.mean(dim=(-2, -1))
        w = torch.sigmoid(self.fc2(F.silu(self.fc1(w))))
        return x * w[:, :, None, None]


class PyramidSEBasicUnit(nn.Module):
    """Additive PyramidNet PreAct unit with SE gating and zero-pad shortcut."""

    def __init__(self, in_ch: int, out_ch: int, stride: int, drop_path: float):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.stride = stride
        self.drop_path = drop_path
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_ch)
        self.se = SEModule(out_ch, reduction=16)
        self.pool = nn.AvgPool2d(2, stride=2) if stride > 1 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(self.bn1(x))
        out = self.conv2(F.silu(self.bn2(out)))
        out = self.se(self.bn3(out))
        if self.training and self.drop_path > 0.0:
            keep = 1.0 - self.drop_path
            mask = torch.empty((x.shape[0], 1, 1, 1), dtype=out.dtype,
                               device=out.device).bernoulli_(keep)
            out = out.div(keep) * mask

        sc = self.pool(x)
        if self.out_ch > self.in_ch:
            pad = torch.zeros(
                (sc.shape[0], self.out_ch - self.in_ch, sc.shape[2], sc.shape[3]),
                dtype=sc.dtype, device=sc.device,
            )
            sc = torch.cat((sc, pad), dim=1)
        return sc + out


class SEPyramidNet(nn.Module):
    """18-unit (depth-38) additive SE-PyramidNet widening 64 -> 448 on 32x32."""

    def __init__(self, start_ch: int = 64, alpha: int = 384,
                 units_per_stage: int = 6, max_drop_path: float = 0.10):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, start_ch, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(start_ch),
        )
        total_units = 3 * units_per_stage
        blocks: list[nn.Module] = []
        in_ch = start_ch
        float_ch = float(start_ch)
        step_add = float(alpha) / float(total_units)
        for idx in range(total_units):
            float_ch += step_add
            out_ch = int(round(float_ch / 4.0)) * 4
            stride = 2 if (idx in (units_per_stage, 2 * units_per_stage)) else 1
            dp = max_drop_path * idx / max(total_units - 1, 1)
            blocks.append(PyramidSEBasicUnit(in_ch, out_ch, stride=stride, drop_path=dp))
            in_ch = out_ch
        self.units = nn.Sequential(*blocks)
        self.bn_final = nn.BatchNorm2d(in_ch)
        self.head = nn.Linear(in_ch, NUM_CLASSES)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.units(self.stem(x))
        h = F.silu(self.bn_final(h)).mean(dim=(-2, -1))
        return self.head(h)


@torch.inference_mode()
def evaluate_single(model: nn.Module, images: torch.Tensor, labels: torch.Tensor,
                    valid_idx: np.ndarray, batch_size: int = 256) -> dict:
    model.eval()
    c1 = c5 = 0
    loss_sum = 0.0
    for start in range(0, len(valid_idx), batch_size):
        idx = torch.as_tensor(valid_idx[start:start + batch_size], device=images.device)
        x = preprocess(images[idx])
        target = labels[idx]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x).float()
        c1 += int((logits.argmax(1) == target).sum().item())
        c5 += int((logits.topk(5, dim=1).indices == target[:, None]).any(1).sum().item())
        loss_sum += float(F.cross_entropy(logits, target, reduction="sum").item())
    n = len(valid_idx)
    return {"top1": c1 / n, "top5": c5 / n, "ce": loss_sum / n}


def train_fold_two_stage(fold: int, images: torch.Tensor, labels_np: np.ndarray,
                         seed_base: int = 515151) -> Path:
    best_dist_pt = L / f"sepyramid_gen2_f{fold}_best.pt"
    if best_dist_pt.exists():
        print(f"skip existing {best_dist_pt.name}", flush=True)
        return best_dist_pt

    folds = stratified_folds(labels_np, 3, SEED)
    valid_idx = np.flatnonzero(folds == fold)
    train_idx = np.flatnonzero(folds != fold)
    labels = torch.as_tensor(labels_np, device="cuda", dtype=torch.long)

    # Stage 1: From scratch + fold-safe transductive bank
    s1_pt = L / f"sepyramid_s1_f{fold}_best.pt"
    if not s1_pt.exists():
        with np.load(L / f"transductive_bank_f{fold}.npz") as bk:
            ext_imgs = clean_impulses_gpu(torch.as_tensor(bk["images"], device="cuda"))
            ext_labs = torch.as_tensor(bk["labels"].astype(np.int64), device="cuda")

        torch.manual_seed(seed_base + fold)
        rng = np.random.default_rng(seed_base + fold)
        model = SEPyramidNet().cuda()
        batch_size = 96
        epochs = 30
        peak_lr = 8.5e-4
        opt = torch.optim.AdamW(model.parameters(), lr=peak_lr, weight_decay=0.03, fused=True)
        steps_ep = math.ceil(len(train_idx) / batch_size)
        total_steps = steps_ep * epochs
        warmup_steps = steps_ep * 2
        best_top1 = -1.0
        best_ep = 0
        t0 = time.monotonic()

        for epoch in range(1, epochs + 1):
            model.train()
            order = rng.permutation(train_idx)
            for b_id, off in enumerate(range(0, len(order), batch_size)):
                idx = torch.as_tensor(order[off:off + batch_size], device="cuda")
                raw = images[idx]
                tgt = labels[idx]
                n_ext = round(len(raw) * 0.45)
                if n_ext > 0:
                    e_pick = torch.as_tensor(rng.integers(0, len(ext_imgs), size=n_ext),
                                             device="cuda")
                    raw = torch.cat((raw[:-n_ext], ext_imgs[e_pick]))
                    tgt = torch.cat((tgt[:-n_ext], ext_labs[e_pick]))
                x = augment(preprocess(raw), 1.5, 12.0, 0.9, 1.1)
                step = (epoch - 1) * steps_ep + b_id
                lr = lr_at_step(step, total_steps, warmup_steps, peak_lr, 1e-6)
                for g in opt.param_groups:
                    g["lr"] = lr
                opt.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits = model(x).float()
                loss = F.cross_entropy(logits, tgt, label_smoothing=0.03)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

            m = evaluate_single(model, images, labels, valid_idx)
            print(f"[sepyramid s1 f{fold}] ep={epoch:02d} top1={m['top1']:.4f} "
                  f"top5={m['top5']:.4f} ({time.monotonic()-t0:.0f}s)", flush=True)
            if m["top1"] > best_top1 + 0.0002:
                best_top1 = m["top1"]
                best_ep = epoch
                torch.save(model.state_dict(), s1_pt)
                (L / f"sepyramid_s1_f{fold}_best.json").write_text(
                    json.dumps({"epoch": epoch, **m}, indent=2) + "\n")
            if epoch >= 18 and epoch - best_ep >= 8:
                break
        del model, ext_imgs, ext_labs
        torch.cuda.empty_cache()

    # Stage 2: 12-epoch Gen-2 soft distillation on labeled OOF teacher
    with np.load(L / "gen2_distill_labeled_oof.npz") as tz:
        lab_cand = torch.as_tensor(tz["candidates"].astype(np.int64), device="cuda")
        lab_q = torch.as_tensor(tz["posteriors"].astype(np.float32), device="cuda")

    torch.manual_seed(seed_base + 100 + fold)
    rng = np.random.default_rng(seed_base + 100 + fold)
    model = SEPyramidNet().cuda()
    model.load_state_dict(torch.load(s1_pt, map_location="cpu", weights_only=True))
    batch_size = 96
    epochs = 12
    peak_lr = 1.2e-4
    opt = torch.optim.AdamW(model.parameters(), lr=peak_lr, weight_decay=0.02, fused=True)
    steps_ep = math.ceil(len(train_idx) / batch_size)
    total_steps = steps_ep * epochs
    warmup_steps = steps_ep * 1
    init_m = evaluate_single(model, images, labels, valid_idx)
    best_top1 = init_m["top1"]
    best_ep = 0
    torch.save(model.state_dict(), best_dist_pt)
    t0 = time.monotonic()

    for epoch in range(1, epochs + 1):
        model.train()
        order = rng.permutation(train_idx)
        for b_id, off in enumerate(range(0, len(order), batch_size)):
            idx = torch.as_tensor(order[off:off + batch_size], device="cuda")
            x = augment(preprocess(images[idx]), 1.0, 8.0, 0.95, 1.05)
            tgt = labels[idx]
            step = (epoch - 1) * steps_ep + b_id
            lr = lr_at_step(step, total_steps, warmup_steps, peak_lr, 1e-6)
            for g in opt.param_groups:
                g["lr"] = lr
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(x).float()
            ce = F.cross_entropy(logits, tgt, label_smoothing=0.03)
            kd = soft_cross_entropy(logits, lab_cand[idx], lab_q[idx])
            loss = 0.45 * ce + 0.55 * kd
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        m = evaluate_single(model, images, labels, valid_idx)
        print(f"[sepyramid s2 f{fold}] ep={epoch:02d} top1={m['top1']:.4f} "
              f"top5={m['top5']:.4f} ({time.monotonic()-t0:.0f}s)", flush=True)
        if m["top1"] > best_top1 + 0.0002:
            best_top1 = m["top1"]
            best_ep = epoch
            torch.save(model.state_dict(), best_dist_pt)
            (L / f"sepyramid_gen2_f{fold}_best.json").write_text(
                json.dumps({"epoch": epoch, **m}, indent=2) + "\n")
        if epoch >= 8 and epoch - best_ep >= 6:
            break

    del model, lab_cand, lab_q
    torch.cuda.empty_cache()
    return best_dist_pt


@torch.inference_mode()
def export_sepyramid_probs(folds_to_use: tuple[int, ...], query_imgs: torch.Tensor,
                           indices: np.ndarray, batch_size: int = 192) -> np.ndarray:
    acc = np.zeros((len(indices), NUM_CLASSES), dtype=np.float32)
    for fold in folds_to_use:
        ckpt = L / f"sepyramid_gen2_f{fold}_best.pt"
        model = SEPyramidNet().cuda().eval()
        model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
        acc += score_model(model, "sepyramid", query_imgs, indices, batch_size=batch_size)
        del model
        torch.cuda.empty_cache()
    acc /= len(folds_to_use)
    return acc.astype(np.float16)


def main() -> None:
    device = "cuda"
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

    labels_np = np.load("labeled_ids.npy").astype(np.int64)
    folds = stratified_folds(labels_np, 3, SEED)
    lab_imgs = clean_impulses_gpu(
        torch.as_tensor(np.asarray(np.load("labeled_32x32.npy", mmap_mode="r")).copy(),
                        device="cuda"))

    # 1. Pilot Fold 0 first
    train_fold_two_stage(0, lab_imgs, labels_np)
    v0_idx = np.flatnonzero(folds == 0)
    f0_probs_path = L / "sepyramid_gen2_f0_rot5.npy"
    if not f0_probs_path.exists():
        p0 = export_sepyramid_probs((0,), lab_imgs, v0_idx)
        np.save(f0_probs_path, p0)
    else:
        p0 = np.load(f0_probs_path)
    f0_top1 = float((p0.argmax(1) == labels_np[v0_idx]).mean())
    print(f"SE-PyramidNet Fold-0 6-view TTA top1 = {f0_top1:.5f}", flush=True)

    if f0_top1 < 0.9570:
        report = {"fold0_rot5_top1": f0_top1, "passed_stage_gate": False, "passed": False}
        (L / "milestone5_sepyramid.json").write_text(json.dumps(report, indent=2) + "\n")
        print("Stop-loss gate triggered: Fold-0 6-view TTA top1 < 0.9570; skipping folds 1 & 2.",
              flush=True)
        return

    u_imgs = clean_impulses_gpu(
        torch.as_tensor(np.asarray(np.load("unl/test_32x32.npy", mmap_mode="r")).copy(),
                        device="cuda"))
    u0_probs_path = L / "unl_f0_sepyramid_gen2.npy"
    if not u0_probs_path.exists():
        u0_p = export_sepyramid_probs((0,), u_imgs, np.arange(len(u_imgs), dtype=np.int64))
        np.save(u0_probs_path, u0_p)

    # 2. Check Fold-0 battle score gate before training Folds 1 & 2
    m2 = json.loads((L / "milestone2_pair_table_v4.json").read_text())
    m3 = json.loads((L / "milestone3_calibration.json").read_text())
    m4 = json.loads((L / "milestone4_gen2_distill.json").read_text())
    nof0_table = np.load(Path(m2["active_nof0_table"]))
    full_table = np.load(Path(m2["active_full_table"]))
    cal = m3["active"]
    s0, s1 = float(cal["s0"]), float(cal["s1"])
    debias = float(cal["debias"])
    beta_lap = float(cal["beta_lap"])

    if m4["passed"]:
        base_ratio = tuple(m4["winner"]["ratio"])
        base_alpha = float(m4["winner"]["alpha"])
        base_temp = float(m4["winner"]["temperature"])
        prev_best_test = float(m4["winner"]["selected"]["test_score"])
    else:
        m1 = json.loads((L / "milestone1_rx_fusion.json").read_text())["winner"]
        dn_s = float(m1["dense_share"])
        base_ratio = (1.0 - dn_s, dn_s, 0.0)
        base_alpha = max(float(m1["alpha"]), 0.35)
        base_temp = float(m1["temperature"])
        prev_best_test = float(cal["test_score"])

    rows = pd.read_csv("data/train.csv.gz")
    identity = {}
    for side in (1, 2):
        for p, v in zip(rows[f"noisy_image{side}"], rows[f"pokemon_id{side}"]):
            if pd.notna(v):
                identity[p] = int(v)
    paths = sorted(identity)
    true = np.array([identity[p] for p in paths], dtype=np.int64)
    index = pd.Index(paths)
    f1, f2 = index.get_indexer(rows.noisy_image1), index.get_indexer(rows.noisy_image2)
    known = (f1 >= 0) & (f2 >= 0)
    f1, f2 = f1[known], f2[known]
    a, b = true[f1], true[f2]
    y = rows.result.to_numpy(np.float32)[known]
    pair_f0 = (folds[f1] == 0) & (folds[f2] == 0)
    rng = np.random.default_rng(3780)
    pair_rows = np.flatnonzero(pair_f0)
    rng.shuffle(pair_rows)
    split = len(pair_rows) // 2

    known_allowed = np.triu(np.ones((1026, 1026), dtype=np.float32), k=1)
    known_allowed[0, :] = 0.0
    known_allowed[:, 0] = 0.0
    known_allowed[a[~pair_f0], b[~pair_f0]] = 0.0

    unk = rows.pokemon_id1.isna() & rows.pokemon_id2.isna()
    u_idx = pd.Index(Path("data/unlabeled_paths.txt").read_text().split())
    eu = u_idx.get_indexer(rows.noisy_image1[unk])
    ev = u_idx.get_indexer(rows.noisy_image2[unk])
    uy = rows.result[unk].to_numpy(np.float32)

    with np.load(L / "rx_ranker_unlabeled_f0only.npz") as uz:
        u_cand = uz["candidates"].astype(np.int64)
        u_base_post = uz["posteriors"].astype(np.float32)
    with np.load(L / "rx_ranker_fold0_scores.npz") as fz:
        v_idx = fz["valid_indices"].astype(np.int64)
        cand = fz["candidates"].astype(np.int64)
        q_base_post = fz["posteriors"].astype(np.float32)
        oop = fz["oop_mass"].astype(np.float32)

    z_lap_f0 = z_lap_from_gallery(L / "gallery_rx_top40_f0.npz", v_idx)
    pos = np.full(len(paths), -1, dtype=np.int32)
    pos[v_idx] = np.arange(len(v_idx))
    one, two = pos[f1[pair_rows]], pos[f2[pair_rows]]
    actual = y[pair_rows]
    ta, tb = a[pair_rows], b[pair_rows]

    q_rx = gather_cand(np.load(L / "oof_resnext29_dist.npy", mmap_mode="r"), cand, rows=v_idx)
    q_dn = gather_cand(np.load(L / "oof_dense_gen2.npy", mmap_mode="r"), cand, rows=v_idx)
    q_w4 = gather_cand(np.load(L / "oof_wrn40_gen2.npy", mmap_mode="r"), cand, rows=v_idx)
    q_sp = gather_cand(np.load(f0_probs_path, mmap_mode="r"), cand)

    u_rx = gather_cand(np.load(L / "unl_f0_resnext29_dist.npy", mmap_mode="r"), u_cand)
    u_dn = gather_cand(np.load(L / "unl_f0_dense_gen2.npy", mmap_mode="r"), u_cand)
    u_w4 = gather_cand(np.load(L / "unl_f0_wrn40_gen2.npy", mmap_mode="r"), u_cand)
    u_sp = gather_cand(np.load(u0_probs_path, mmap_mode="r"), u_cand)

    q_prev = base_ratio[0] * q_rx + base_ratio[1] * q_dn + base_ratio[2] * q_w4
    q_prev /= np.maximum(q_prev.sum(1, keepdims=True), 1e-8)
    u_prev = base_ratio[0] * u_rx + base_ratio[1] * u_dn + base_ratio[2] * u_w4
    u_prev /= np.maximum(u_prev.sum(1, keepdims=True), 1e-8)

    trials = []
    for sp_share in (0.00, 0.15, 0.25, 0.35):
        stu_q = (1.0 - sp_share) * q_prev + sp_share * q_sp
        stu_q /= np.maximum(stu_q.sum(1, keepdims=True), 1e-8)
        stu_u = (1.0 - sp_share) * u_prev + sp_share * u_sp
        stu_u /= np.maximum(stu_u.sum(1, keepdims=True), 1e-8)
        for alpha in sorted({base_alpha, min(base_alpha + 0.05, 0.60)}):
            q_mix = fuse_posteriors(q_base_post, stu_q, alpha, "log", base_temp)
            u_mix = fuse_posteriors(u_base_post, stu_u, alpha, "log", base_temp)
            _, _, unl_lnot0 = run_unlabeled_bp_hetero(
                u_cand, u_mix, eu, ev, uy, nof0_table, known_allowed,
                s0=s0, s1=s1, device=device)
            _, _, test_lnot, _, _ = target_cavity_predict(
                q_mix, oop, cand, one, two, known_allowed, unl_lnot0, nof0_table,
                debias=debias, device=device)
            _, _, unl_lnot = run_unlabeled_bp_hetero(
                u_cand, u_mix, eu, ev, uy, nof0_table, known_allowed,
                s0=s0, s1=s1, external_log_avail=test_lnot, device=device)
            best_s = None
            for sc in (0.85, 0.90, 0.95, 1.00, 1.05):
                s_vec = np.clip(sc - beta_lap * z_lap_f0, 0.55, 1.45)
                qs = adaptive_power_scale(q_mix, s_vec)
                pred, top_p, _, _, _ = target_cavity_predict(
                    qs, oop, cand, one, two, known_allowed, unl_lnot, nof0_table,
                    debias=debias, cavity_iters=12, use_escape=False, device=device)
                tr_r = rmse(pred[:split], actual[:split])
                te_r = rmse(pred[split:], actual[split:])
                both_ok = float(((top_p[:, 0] == ta) & (top_p[:, 1] == tb))[split:].mean())
                it = {
                    "scale_base": sc, "tune_rmse": tr_r, "test_rmse": te_r,
                    "tune_score": score_fn(tr_r), "test_score": score_fn(te_r),
                    "test_both_correct": both_ok,
                }
                if best_s is None or it["tune_rmse"] < best_s["tune_rmse"]:
                    best_s = it
            rec = {"sp_share": sp_share, "alpha": alpha, "selected": best_s}
            trials.append(rec)
            print(f"[sepyramid_f0_mix] sp_share={sp_share:.2f} alpha={alpha:.2f} "
                  f"tune={best_s['tune_score']:.4f} test={best_s['test_score']:.4f}",
                  flush=True)

    anchor = min([r for r in trials if r["sp_share"] == 0.0],
                 key=lambda r: r["selected"]["tune_rmse"])
    chosen = min(trials, key=lambda r: (r["selected"]["tune_rmse"], r["sp_share"]))
    passed = (chosen["sp_share"] > 0.0
              and chosen["selected"]["test_score"] > max(prev_best_test,
                                                         anchor["selected"]["test_score"]))
    print(f"Milestone 5 Fold-0 gate: sp_share={chosen['sp_share']} passed={passed} "
          f"(test {anchor['selected']['test_score']:.4f} -> {chosen['selected']['test_score']:.4f})",
          flush=True)

    sub_csv = None
    if passed:
        # 3. Train Folds 1 & 2, export 3-fold 6-view TTA on unlabeled & test, submit!
        for fold in (1, 2):
            train_fold_two_stage(fold, lab_imgs, labels_np)
        del lab_imgs
        torch.cuda.empty_cache()

        unl_sp_path = L / "unl_sepyramid_gen2.npy"
        if not unl_sp_path.exists():
            np.save(unl_sp_path,
                    export_sepyramid_probs((0, 1, 2), u_imgs,
                                           np.arange(len(u_imgs), dtype=np.int64)))
        del u_imgs
        torch.cuda.empty_cache()

        test_sp_path = L / "test_sepyramid_gen2.npy"
        if not test_sp_path.exists():
            t_imgs = clean_impulses_gpu(
                torch.as_tensor(np.asarray(np.load("test_32x32.npy", mmap_mode="r")).copy(),
                                device="cuda"))
            np.save(test_sp_path,
                    export_sepyramid_probs((0, 1, 2), t_imgs,
                                           np.arange(len(t_imgs), dtype=np.int64)))
            del t_imgs
            torch.cuda.empty_cache()

        sp_w = chosen["sp_share"]
        alpha = chosen["alpha"]
        sc_win = chosen["selected"]["scale_base"]

        with np.load(L / "rx_ranker_unlabeled.npz") as uz:
            tu_cand = uz["candidates"].astype(np.int64)
            tu_base = uz["posteriors"].astype(np.float32)
        with np.load(L / "rx_ranker_test.npz") as tz:
            t_cand = tz["candidates"].astype(np.int64)
            t_base = tz["posteriors"].astype(np.float32)
            t_oop = tz["oop_mass"].astype(np.float32)

        tu_prev = (
            base_ratio[0] * gather_cand(np.load(L / "unl_resnext29_dist.npy", mmap_mode="r"), tu_cand)
            + base_ratio[1] * gather_cand(np.load(L / "unl_dense_gen2.npy", mmap_mode="r"), tu_cand)
            + base_ratio[2] * gather_cand(np.load(L / "unl_wrn40_gen2.npy", mmap_mode="r"), tu_cand)
        )
        tu_prev /= np.maximum(tu_prev.sum(1, keepdims=True), 1e-8)
        tu_stu = (1.0 - sp_w) * tu_prev + sp_w * gather_cand(np.load(unl_sp_path, mmap_mode="r"), tu_cand)
        tu_stu /= np.maximum(tu_stu.sum(1, keepdims=True), 1e-8)

        t_prev = (
            base_ratio[0] * gather_cand(np.load(L / "test_resnext29_dist.npy", mmap_mode="r"), t_cand)
            + base_ratio[1] * gather_cand(np.load(L / "test_dense_gen2.npy", mmap_mode="r"), t_cand)
            + base_ratio[2] * gather_cand(np.load(L / "test_wrn40_gen2.npy", mmap_mode="r"), t_cand)
        )
        t_prev /= np.maximum(t_prev.sum(1, keepdims=True), 1e-8)
        t_stu = (1.0 - sp_w) * t_prev + sp_w * gather_cand(np.load(test_sp_path, mmap_mode="r"), t_cand)
        t_stu /= np.maximum(t_stu.sum(1, keepdims=True), 1e-8)

        tu_pri = fuse_posteriors(tu_base, tu_stu, alpha, "log", base_temp)
        t_post_raw = fuse_posteriors(t_base, t_stu, alpha, "log", base_temp)
        z_lap_test = z_lap_from_gallery(L / "gallery_rx_top40_test.npz", None)
        t_post = adaptive_power_scale(t_post_raw, np.clip(sc_win - beta_lap * z_lap_test, 0.55, 1.45))

        with np.load("data/known_battles.npz") as kz:
            ka_all, kb_all = kz["a"].astype(np.int64), kz["b"].astype(np.int64)
        known_full = np.triu(np.ones((1026, 1026), dtype=np.float32), k=1)
        known_full[0, :] = 0.0
        known_full[:, 0] = 0.0
        known_full[ka_all, kb_all] = 0.0

        test_paths = Path("test_paths.txt").read_text().splitlines()
        p_idx = pd.Index(test_paths)
        test_df = pd.read_csv("data/test.csv.gz")
        idx1 = p_idx.get_indexer(test_df.noisy_image1)
        idx2 = p_idx.get_indexer(test_df.noisy_image2)

        _, _, u_lnot = run_unlabeled_bp_hetero(
            tu_cand, tu_pri, eu, ev, uy, full_table, known_full,
            s0=s0, s1=s1, device=device)
        for outer in range(2):
            pred, _, test_lnot, _, _ = target_cavity_predict(
                t_post, t_oop, t_cand, idx1, idx2, known_full, u_lnot, full_table,
                debias=debias, cavity_iters=12, use_escape=False, device=device)
            if outer == 0:
                _, _, u_lnot = run_unlabeled_bp_hetero(
                    tu_cand, tu_pri, eu, ev, uy, full_table, known_full,
                    s0=s0, s1=s1, external_log_avail=test_lnot, device=device)

        pred = np.clip(pred, 0.0, 10.0)
        sub_csv = L / "submission_rx_v5_5arch.csv"
        with sub_csv.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["row_id", "result"])
            for row_idx, val in enumerate(pred):
                writer.writerow([f"test-{row_idx:06d}", f"{float(val):.6f}"])
        print(f"Milestone 5 complete -> {sub_csv}", flush=True)

    report = {
        "fold0_rot5_top1": f0_top1,
        "passed_stage_gate": True,
        "anchor": anchor,
        "chosen": chosen,
        "passed": passed,
        "submission_csv": str(sub_csv) if sub_csv else None,
        "trials": trials,
    }
    (L / "milestone5_sepyramid.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
