"""Noise-aware 4-backbone + expanded-kNN listwise softmax & LightGBM candidate ranker.

Combines:
1. A permutation-equivariant DeepSets neural ranker trained with exact K-way
   multiclass cross-entropy across all K=40 candidates of each image.
2. A LightGBM candidate classifier on the same 45 noise/backbone/kNN features
   plus categorical species ID.
3. Exact OOF multiclass NLL calibration blending both rankers, exporting:
   - 3-fold OOF predictions and CV metrics
   - Strict folds-(1+2)-trained Fold-0 predictions (for leak-free e2e benchmark)
   - Full-data ensemble predictions on unlabeled (68,675) and test (160,000) sets.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import lightgbm as lgb
import numpy as np
import torch
from scipy.optimize import minimize
from torch import nn
from torch.nn import functional as F


def build_features(z: np.lib.npyio.NpzFile) -> tuple[np.ndarray, np.ndarray]:
    """Return (N, K, F) float32 features and (N, K) int64 1-based candidate IDs."""
    ids = z["candidates"].astype(np.int64)
    n, k = ids.shape
    wrn28 = np.clip(z["wrn28_probs"].astype(np.float32), 1e-8, 1.0)
    wrn40 = np.clip(z["wrn40_probs"].astype(np.float32), 1e-8, 1.0)
    dense = np.clip(z["dense_probs"].astype(np.float32), 1e-8, 1.0)
    conv = np.clip(z["conv_probs"].astype(np.float32), 1e-8, 1.0)
    blend = np.clip(z["blend_probs"].astype(np.float32), 1e-8, 1.0)

    lw28 = np.log(wrn28)
    lw40 = np.log(wrn40)
    lden = np.log(dense)
    lcon = np.log(conv)
    lbln = np.log(blend)

    knn1 = z["knn1"].astype(np.float32)
    knn5 = z["knn5"].astype(np.float32)
    knn15 = z["knn15"].astype(np.float32)
    proto = z["prototype"].astype(np.float32)
    cproto = z["clean_prototype"].astype(np.float32)

    qual = z["image_quality"].astype(np.float32)  # (N, 4): lap, contrast, sigma, impulse
    lap_norm = ((qual[:, 0:1] - 26.0) / 10.0)
    bcast = lambda arr: np.broadcast_to(arr, (n, k)).astype(np.float32)

    ranks = np.arange(1, k + 1, dtype=np.float32)[None, :] / float(k)
    cols = [
        lw28, lw40, lden, lcon, lbln,
        lw28 - lw28.max(1, keepdims=True),
        lw40 - lw40.max(1, keepdims=True),
        lden - lden.max(1, keepdims=True),
        lcon - lcon.max(1, keepdims=True),
        lbln - lbln[:, :1],
        wrn28, wrn40, dense, conv, blend,
        bcast(ranks),
        (np.arange(k)[None, :] == 0).astype(np.float32) * np.ones((n, 1), np.float32),
        (np.arange(k)[None, :] < 3).astype(np.float32) * np.ones((n, 1), np.float32),
        knn1, knn5, knn15, proto, cproto,
        knn1 - knn1.max(1, keepdims=True),
        knn5 - knn5.max(1, keepdims=True),
        knn15 - knn15.max(1, keepdims=True),
        proto - proto.max(1, keepdims=True),
        cproto - cproto.max(1, keepdims=True),
        lw28 + 32.0 * knn1,
        lbln + 28.0 * knn5,
    ]
    for arch in ("wrn28", "wrn40", "dense", "conv"):
        img_s = z[f"{arch}_img"].astype(np.float32)  # (N, 3): top1_p, margin, ent
        for j in range(3):
            cols.append(bcast(img_s[:, j:j + 1]))
    for j in range(4):
        cols.append(bcast(qual[:, j:j + 1]))
    cols.extend([
        lap_norm * lw28,
        lap_norm * knn5,
        lap_norm * lden,
        lap_norm * lcon,
    ])
    feat = np.stack(cols, axis=2).astype(np.float32)
    return feat, ids


class ListwiseSetRanker(nn.Module):
    """Permutation-equivariant DeepSets ranker over K candidates per image."""

    def __init__(self, num_features: int, hidden: int = 192, drop: float = 0.15):
        super().__init__()
        self.species_emb = nn.Embedding(1026, 8)
        nn.init.normal_(self.species_emb.weight, std=0.02)
        self.enc = nn.Sequential(
            nn.Linear(num_features + 8, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Dropout(drop),
        )
        self.ctx = nn.Sequential(
            nn.Linear(hidden * 3, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Dropout(drop),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
        )
        self.head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor, cand_ids: torch.Tensor) -> torch.Tensor:
        h = self.enc(torch.cat([x, self.species_emb(cand_ids)], dim=-1))
        h_max = h.max(dim=1, keepdim=True).values.expand_as(h)
        h_mean = h.mean(dim=1, keepdim=True).expand_as(h)
        out = h + self.ctx(torch.cat([h, h_max, h_mean], dim=-1))
        return self.head(out).squeeze(-1)


def fit_listwise_ensemble(x_train: np.ndarray, ids_train: np.ndarray, y_train: np.ndarray,
                          seeds: tuple[int, ...] = (0, 1, 2), epochs: int = 22,
                          batch_size: int = 512) -> tuple[list[ListwiseSetRanker], np.ndarray, np.ndarray]:
    mean = x_train.mean(axis=(0, 1), keepdims=True).astype(np.float32)
    std = np.maximum(x_train.std(axis=(0, 1), keepdims=True), 1e-5).astype(np.float32)

    in_pool = ids_train == y_train[:, None]
    keep = in_pool.any(1)
    xk = torch.as_tensor(((x_train[keep] - mean) / std), device="cuda", dtype=torch.float32)
    ik = torch.as_tensor(ids_train[keep], device="cuda", dtype=torch.long)
    tk = torch.as_tensor(in_pool[keep].argmax(1), device="cuda", dtype=torch.long)

    models = []
    n = len(xk)
    steps_per_epoch = math.ceil(n / batch_size)
    total_steps = steps_per_epoch * epochs
    for seed in seeds:
        torch.manual_seed(seed + 100)
        model = ListwiseSetRanker(x_train.shape[2]).cuda()
        opt = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=1e-2)
        step = 0
        for _ in range(epochs):
            model.train()
            perm = torch.randperm(n, device="cuda")
            for start in range(0, n, batch_size):
                b = perm[start:start + batch_size]
                progress = step / max(total_steps, 1)
                lr = 1e-5 + (1.5e-3 - 1e-5) * 0.5 * (1.0 + math.cos(math.pi * progress))
                for g in opt.param_groups:
                    g["lr"] = lr
                logits = model(xk[b], ik[b])
                loss = F.cross_entropy(logits, tk[b])
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                step += 1
        model.eval()
        models.append(model)
    return models, mean, std


@torch.inference_mode()
def predict_listwise(models: list[ListwiseSetRanker], mean: np.ndarray, std: np.ndarray,
                     x: np.ndarray, ids: np.ndarray, batch_size: int = 2048) -> np.ndarray:
    xn = ((x - mean) / std).astype(np.float32)
    out = np.zeros((len(x), x.shape[1]), dtype=np.float32)
    for start in range(0, len(x), batch_size):
        end = min(start + batch_size, len(x))
        xt = torch.as_tensor(xn[start:end], device="cuda")
        it = torch.as_tensor(ids[start:end], device="cuda", dtype=torch.long)
        log_probs = 0
        for m in models:
            log_probs = log_probs + F.log_softmax(m(xt, it), dim=1)
        out[start:end] = (log_probs / len(models)).cpu().numpy()
    return out


def fit_lgb(x_train: np.ndarray, ids_train: np.ndarray, y_train: np.ndarray) -> lgb.LGBMClassifier:
    n, k, f = x_train.shape
    xf = np.concatenate([x_train.reshape(n * k, f),
                         ids_train.reshape(n * k, 1).astype(np.float32)], axis=1)
    yf = (ids_train == y_train[:, None]).reshape(-1).astype(np.uint8)
    names = [f"f{i}" for i in range(f)] + ["candidate_id"]
    model = lgb.LGBMClassifier(
        n_estimators=300, learning_rate=0.04, num_leaves=23,
        min_child_samples=1000, colsample_bytree=0.8, subsample=0.8,
        reg_lambda=3.0, n_jobs=16, random_state=3780, verbosity=-1,
    )
    model.fit(xf, yf, feature_name=names, categorical_feature=[f])
    return model


def predict_lgb(model: lgb.LGBMClassifier, x: np.ndarray, ids: np.ndarray) -> np.ndarray:
    n, k, f = x.shape
    xf = np.concatenate([x.reshape(n * k, f),
                         ids.reshape(n * k, 1).astype(np.float32)], axis=1)
    return model.predict_proba(xf)[:, 1].reshape(n, k).astype(np.float32)


def fit_blend_weights(nn_logp: np.ndarray, lgb_p: np.ndarray, blend_p: np.ndarray,
                      ids: np.ndarray, labels: np.ndarray) -> np.ndarray:
    in_pool = ids == labels[:, None]
    keep = in_pool.any(1)
    target = in_pool[keep].argmax(1)
    f1 = nn_logp[keep].astype(np.float64)
    rk = np.clip(lgb_p[keep].astype(np.float64), 1e-7, 1.0 - 1e-7)
    f2 = np.log(rk / (1.0 - rk))
    f3 = np.log(rk)
    f4 = np.log(np.clip(blend_p[keep].astype(np.float64), 1e-9, 1.0))

    def nll(w: np.ndarray) -> float:
        s = w[0] * f1 + w[1] * f2 + w[2] * f3 + w[3] * f4
        s -= s.max(1, keepdims=True)
        lq = s - np.log(np.exp(s).sum(1, keepdims=True))
        return -float(np.mean(lq[np.arange(len(target)), target]))

    res = minimize(nll, np.array([0.55, 0.15, 0.25, 0.05]), method="Nelder-Mead")
    return res.x.astype(np.float64)


def combine_posteriors(nn_logp: np.ndarray, lgb_p: np.ndarray, blend_p: np.ndarray,
                       w: np.ndarray) -> np.ndarray:
    rk = np.clip(lgb_p.astype(np.float64), 1e-7, 1.0 - 1e-7)
    s = (w[0] * nn_logp.astype(np.float64)
         + w[1] * np.log(rk / (1.0 - rk))
         + w[2] * np.log(rk)
         + w[3] * np.log(np.clip(blend_p.astype(np.float64), 1e-9, 1.0)))
    s -= s.max(1, keepdims=True)
    p = np.exp(s)
    return (p / p.sum(1, keepdims=True)).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=Path("outputs/large"))
    parser.add_argument("--oof-prefix", default="gallery_v3_top40")
    parser.add_argument("--test-features", type=Path, default=None)
    parser.add_argument("--unlabeled-features", type=Path, default=None)
    parser.add_argument("--unlabeled-f0-features", type=Path, default=None)
    parser.add_argument("--output-prefix", type=Path,
                        default=Path("outputs/large/phase1_ranker"))
    args = parser.parse_args()

    xs, id_list, y_list, idx_list, fold_list, bln_list, oop_list = [], [], [], [], [], [], []
    for fold in range(3):
        with np.load(args.input_dir / f"{args.oof_prefix}_f{fold}.npz") as z:
            feat, ids = build_features(z)
            xs.append(feat)
            id_list.append(ids)
            y_list.append(z["labels"].astype(np.int64))
            idx_list.append(z["valid_indices"])
            fold_list.append(np.full(len(ids), fold, dtype=np.int8))
            bln_list.append(z["blend_probs"].astype(np.float32))
            oop_list.append(z["oop_mass"].astype(np.float32))

    x_all = np.concatenate(xs)
    ids_all = np.concatenate(id_list)
    y_all = np.concatenate(y_list)
    idx_all = np.concatenate(idx_list)
    folds_all = np.concatenate(fold_list)
    bln_all = np.concatenate(bln_list)
    oop_all = np.concatenate(oop_list)
    del xs, id_list, y_list, idx_list, fold_list, bln_list, oop_list

    oof_nn = np.empty(ids_all.shape, dtype=np.float32)
    oof_lgb = np.empty(ids_all.shape, dtype=np.float32)
    f0_models = None
    f0_lgb = None

    for fold in range(3):
        tr = folds_all != fold
        va = ~tr
        nn_models, mean, std = fit_listwise_ensemble(x_all[tr], ids_all[tr], y_all[tr])
        oof_nn[va] = predict_listwise(nn_models, mean, std, x_all[va], ids_all[va])
        lgb_m = fit_lgb(x_all[tr], ids_all[tr], y_all[tr])
        oof_lgb[va] = predict_lgb(lgb_m, x_all[va], ids_all[va])
        if fold == 0:
            f0_models = (nn_models, mean, std)
            f0_lgb = lgb_m
        nn_acc = float((ids_all[va, oof_nn[va].argmax(1)] == y_all[va]).mean())
        lgb_acc = float((ids_all[va, oof_lgb[va].argmax(1)] == y_all[va]).mean())
        print(f"fold {fold}: listwise_nn_top1={nn_acc:.5f} lgb_top1={lgb_acc:.5f}", flush=True)

    # Fold-1+2 blend weights (leak-free for Fold-0 benchmark)
    tr0 = folds_all != 0
    va0 = ~tr0
    w_f12 = fit_blend_weights(oof_nn[tr0], oof_lgb[tr0], bln_all[tr0], ids_all[tr0], y_all[tr0])
    w_full = fit_blend_weights(oof_nn, oof_lgb, bln_all, ids_all, y_all)
    oof_post = combine_posteriors(oof_nn, oof_lgb, bln_all, w_full)

    overall = {
        "count": int(len(y_all)),
        "blend_cnn_top1": float((ids_all[:, 0] == y_all).mean()),
        "listwise_nn_top1": float((ids_all[np.arange(len(y_all)), oof_nn.argmax(1)] == y_all).mean()),
        "lgb_top1": float((ids_all[np.arange(len(y_all)), oof_lgb.argmax(1)] == y_all).mean()),
        "combined_top1": float((ids_all[np.arange(len(y_all)), oof_post.argmax(1)] == y_all).mean()),
        "candidate_pool_recall": float((ids_all == y_all[:, None]).any(1).mean()),
        "blend_weights_full": w_full.tolist(),
        "blend_weights_f12": w_f12.tolist(),
    }
    print("OOF SUMMARY:", json.dumps(overall, indent=2), flush=True)

    out_base = args.output_prefix
    out_base.parent.mkdir(parents=True, exist_ok=True)
    Path(f"{out_base}_cv.json").write_text(json.dumps(overall, indent=2) + "\n")
    np.savez_compressed(
        f"{out_base}_oof.npz",
        indices=idx_all, labels=y_all.astype(np.uint16), folds=folds_all,
        candidates=ids_all.astype(np.uint16), posteriors=oof_post.astype(np.float32),
        nn_logp=oof_nn.astype(np.float16), lgb_scores=oof_lgb.astype(np.float16),
        blend_probs=bln_all.astype(np.float16), oop_mass=oop_all.astype(np.float32),
    )

    f0_post = combine_posteriors(oof_nn[va0], oof_lgb[va0], bln_all[va0], w_f12)
    np.savez_compressed(
        f"{out_base}_fold0_scores.npz",
        valid_indices=idx_all[va0], labels=y_all[va0].astype(np.uint16),
        candidates=ids_all[va0].astype(np.uint16), posteriors=f0_post.astype(np.float32),
        oop_mass=oop_all[va0].astype(np.float32),
    )

    # If fold-0-only unlabeled features are provided, score them with ONLY the fold-0 (folds 1+2) models!
    if args.unlabeled_f0_features is not None and args.unlabeled_f0_features.exists():
        with np.load(args.unlabeled_f0_features) as uz0:
            ux0, uids0 = build_features(uz0)
            ubln0 = uz0["blend_probs"].astype(np.float32)
            uoop0 = uz0["oop_mass"].astype(np.float32)
        nn_m0, m0, s0 = f0_models
        u0_nn = predict_listwise(nn_m0, m0, s0, ux0, uids0)
        u0_lgb = predict_lgb(f0_lgb, ux0, uids0)
        u0_post = combine_posteriors(u0_nn, u0_lgb, ubln0, w_f12)
        np.savez_compressed(
            f"{out_base}_unlabeled_f0only.npz",
            candidates=uids0.astype(np.uint16), posteriors=u0_post.astype(np.float32),
            oop_mass=uoop0,
        )
        print("Saved fold-0-safe unlabeled posteriors", flush=True)

    # Fit final full-data rankers on all 160,925 labeled rows for unlabeled & test sets
    if (args.test_features is not None and args.test_features.exists()) or (
            args.unlabeled_features is not None and args.unlabeled_features.exists()):
        full_nn, full_mean, full_std = fit_listwise_ensemble(
            x_all, ids_all, y_all, seeds=(0, 1, 2, 3, 4))
        full_lgb = fit_lgb(x_all, ids_all, y_all)
        for tag, fpath in (("unlabeled", args.unlabeled_features), ("test", args.test_features)):
            if fpath is None or not fpath.exists():
                continue
            with np.load(fpath) as qz:
                qx, qids = build_features(qz)
                qbln = qz["blend_probs"].astype(np.float32)
                qoop = qz["oop_mass"].astype(np.float32)
            q_nn = predict_listwise(full_nn, full_mean, full_std, qx, qids)
            q_lgb = predict_lgb(full_lgb, qx, qids)
            q_post = combine_posteriors(q_nn, q_lgb, qbln, w_full)
            np.savez_compressed(
                f"{out_base}_{tag}.npz",
                candidates=qids.astype(np.uint16), posteriors=q_post.astype(np.float32),
                oop_mass=qoop,
            )
            print(f"Saved {out_base}_{tag}.npz ({qids.shape})", flush=True)


if __name__ == "__main__":
    main()
