"""Pick one horizontal facing per image by gallery-prototype similarity.

WRN-28 impulse embeddings, 0 degrees. Each labeled exemplar is aligned to its
own class, then a query keeps the facing with the higher cosine to the nearest
class prototype and drops the other. Fold-0 unlabeled uses the fold-0 model
only; test and full unlabeled average the three fold models.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from degradation import clean_impulses_gpu
from export_probs_rot5 import checkpoint_path
from image_experiment import NUM_CLASSES, SEED, stratified_folds
from train_large_image import make_model, preprocess

L = Path("outputs/large")


@torch.inference_mode()
def embed_both(model: torch.nn.Module, images: torch.Tensor, batch_size: int) -> np.ndarray:
    out = np.empty((len(images), 2, 640), dtype=np.float16)
    for start in range(0, len(images), batch_size):
        end = min(start + batch_size, len(images))
        x0 = preprocess(images[start:end])
        faces = []
        for base in (x0, torch.flip(x0, dims=[-1])):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                h = model.stages(model.stem(base))
                h = F.relu(model.bn(h)).mean(dim=(-2, -1))
            faces.append(F.normalize(h.float(), dim=1))
        out[start:end] = torch.stack(faces, dim=1).half().cpu().numpy()
    return out


def aligned_prototypes(emb: np.ndarray, labels: np.ndarray, rounds: int = 3) -> torch.Tensor:
    e = torch.as_tensor(emb, device="cuda").float()
    y = torch.as_tensor(labels, device="cuda", dtype=torch.long)
    chosen = e[:, 0]
    proto = torch.empty((NUM_CLASSES, e.shape[-1]), device="cuda")
    for _ in range(rounds):
        proto.zero_()
        proto.index_add_(0, y, chosen)
        counts = torch.bincount(y, minlength=NUM_CLASSES).clamp_min(1).unsqueeze(1).float()
        proto = F.normalize(proto / counts, dim=1)
        take_flip = (e[:, 1] * proto[y]).sum(-1) > (e[:, 0] * proto[y]).sum(-1)
        chosen = torch.where(take_flip[:, None], e[:, 1], e[:, 0])
    return proto


def nearest_scores(emb: np.ndarray, proto: torch.Tensor, batch: int = 4096) -> np.ndarray:
    out = np.empty((len(emb), 2), dtype=np.float32)
    for start in range(0, len(emb), batch):
        end = min(len(emb), start + batch)
        e = torch.as_tensor(emb[start:end], device="cuda").float()
        out[start:end] = torch.matmul(e, proto.T).amax(-1).cpu().numpy()
    return out


def main() -> None:
    labels = np.load("labeled_ids.npy").astype(np.int64)
    folds = stratified_folds(labels, 3, SEED)
    labeled = torch.as_tensor(np.asarray(np.load("labeled_32x32.npy")).copy(), device="cuda")
    labeled = clean_impulses_gpu(labeled)
    unl = torch.as_tensor(np.asarray(np.load("unl/test_32x32.npy")).copy(), device="cuda")
    unl = clean_impulses_gpu(unl)
    test = torch.as_tensor(np.asarray(np.load("test_32x32.npy")).copy(), device="cuda")
    test = clean_impulses_gpu(test)

    facing_labeled = np.zeros(len(labels), dtype=np.uint8)
    score_unl = np.zeros((len(unl), 2), dtype=np.float32)
    score_test = np.zeros((len(test), 2), dtype=np.float32)
    score_unl_f0 = None

    for fold in range(3):
        ckpt = checkpoint_path(L, "wrn28_10_imp", fold)
        model = make_model("wrn28_10", pretrained=False).cuda().eval()
        model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
        lab = embed_both(model, labeled, 512)
        query_u = embed_both(model, unl, 512)
        query_t = embed_both(model, test, 512)
        del model
        torch.cuda.empty_cache()

        train = folds != fold
        proto_train = aligned_prototypes(lab[train], labels[train])
        proto_all = aligned_prototypes(lab, labels)
        valid = np.flatnonzero(folds == fold)
        s_valid = nearest_scores(lab[valid], proto_train)
        facing_labeled[valid] = (s_valid[:, 1] > s_valid[:, 0]).astype(np.uint8)
        score_unl += nearest_scores(query_u, proto_all)
        score_test += nearest_scores(query_t, proto_all)
        if fold == 0:
            score_unl_f0 = nearest_scores(query_u, proto_train)
        agree = float((s_valid[:, 1] > s_valid[:, 0]).mean())
        margin = float(np.abs(s_valid[:, 1] - s_valid[:, 0]).mean())
        print(f"fold {fold}: flip_rate={agree:.3f} mean|margin|={margin:.4f} ckpt={ckpt.name}",
              flush=True)
        del lab, query_u, query_t, proto_train, proto_all

    facing_unl = (score_unl[:, 1] > score_unl[:, 0]).astype(np.uint8)
    facing_test = (score_test[:, 1] > score_test[:, 0]).astype(np.uint8)
    facing_unl_f0 = (score_unl_f0[:, 1] > score_unl_f0[:, 0]).astype(np.uint8)
    np.save(L / "facing_labeled.npy", facing_labeled)
    np.save(L / "facing_unl.npy", facing_unl)
    np.save(L / "facing_test.npy", facing_test)
    np.save(L / "facing_unl_f0.npy", facing_unl_f0)
    print(
        f"flip rates labeled={facing_labeled.mean():.3f} "
        f"unl={facing_unl.mean():.3f} unl_f0={facing_unl_f0.mean():.3f} "
        f"test={facing_test.mean():.3f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
