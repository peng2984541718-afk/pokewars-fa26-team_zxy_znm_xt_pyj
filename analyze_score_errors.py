"""Break down battle-score errors and render representative noisy-image cases."""

from __future__ import annotations

import io
import json
from collections import defaultdict
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


ROOT = Path("outputs/large")


def stats(actual: np.ndarray, predicted: np.ndarray, mask: np.ndarray,
          total_squared: float) -> dict:
    error = predicted[mask]-actual[mask]
    return {"count":int(mask.sum()), "fraction":float(mask.mean()),
            "rmse":float(np.sqrt(np.mean(error**2))),
            "mae":float(np.mean(np.abs(error))),
            "squared_error_share":float(np.sum(error**2)/total_squared),
            "absolute_error_at_least_4":float(np.mean(np.abs(error)>=4)),
            "absolute_error_at_least_8":float(np.mean(np.abs(error)>=8))}


def image(archive: ZipFile, path: str) -> Image.Image:
    member = "data/data/"+path.removeprefix("images/")
    with Image.open(io.BytesIO(archive.read(member))) as source:
        return source.convert("L").resize((112, 112), Image.Resampling.NEAREST).convert("RGB")


def gallery(archive: ZipFile, rows: pd.DataFrame, z, indices: list[int],
            output: Path, title: str) -> None:
    font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 17)
    font_big = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 22)
    height = 148
    canvas = Image.new("RGB", (850, 48+height*len(indices)), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((16, 10), title, font=font_big, fill=(25, 30, 40))
    for rank, i in enumerate(indices):
        rowid = int(z["train_row_indices"][i])
        row = rows.iloc[rowid]
        top = 48+rank*height
        first = image(archive, row.noisy_image1)
        second = image(archive, row.noisy_image2)
        canvas.paste(first, (18, top+10))
        canvas.paste(second, (295, top+10))
        a, b = int(z["true_id1"][i]), int(z["true_id2"][i])
        pa, pb = int(z["predicted_id1"][i]), int(z["predicted_id2"][i])
        actual = float(z["result"][i])
        predicted = float(z["score_predicted_ids"][i])
        draw.text((140, top+17), f"Image 1", font=font, fill=(25, 30, 40))
        draw.text((140, top+45), f"true {a}", font=font, fill=(25, 30, 40))
        draw.text((140, top+73), f"pred {pa}", font=font,
                  fill=(170, 50, 35) if a != pa else (40, 90, 65))
        draw.text((417, top+17), f"Image 2", font=font, fill=(25, 30, 40))
        draw.text((417, top+45), f"true {b}", font=font, fill=(25, 30, 40))
        draw.text((417, top+73), f"pred {pb}", font=font,
                  fill=(170, 50, 35) if b != pb else (40, 90, 65))
        draw.text((595, top+17), f"row {rowid}", font=font, fill=(70, 75, 85))
        draw.text((595, top+47), f"actual {actual:.2f}", font=font, fill=(25, 30, 40))
        draw.text((595, top+77), f"model {predicted:.2f}", font=font, fill=(170, 50, 35))
        draw.text((595, top+107), f"|error| {abs(predicted-actual):.2f}",
                  font=font, fill=(25, 30, 40))
        draw.line((16, top+height-2, 834, top+height-2), fill=(215, 219, 223))
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def main() -> None:
    with np.load(ROOT / "score_model_error_rows.npz") as source:
        z = {key:source[key] for key in source.files}
    with np.load(ROOT / "meta_rerank_three_arch_top20_oof.npz") as source:
        covered = np.zeros(len(source["indices"]), dtype=bool)
        covered[source["indices"]] = (
            source["candidates"] == source["labels"][:, None]).any(1)
    y = z["result"]
    pred = z["score_predicted_ids"]
    true_pred = z["score_true_ids"]
    a, b = z["true_id1"], z["true_id2"]
    pa, pb = z["predicted_id1"], z["predicted_id2"]
    wrong1, wrong2 = a != pa, b != pb
    wrong_count = wrong1.astype(np.int8)+wrong2.astype(np.int8)
    wrong = wrong_count > 0
    squared = (pred-y)**2
    total_squared = float(squared.sum())
    within_candidates = wrong & (~wrong1 | covered[z["image1_indices"]]) & (
        ~wrong2 | covered[z["image2_indices"]])
    outside_candidates = wrong & ~within_candidates
    oracle_top20_prediction = pred.copy()
    oracle_top20_prediction[within_candidates] = true_pred[within_candidates]
    oracle_top20_rmse = float(np.sqrt(np.mean((oracle_top20_prediction-y)**2)))
    report = {
        "validation_battles":len(y),
        "image_oof_top1":.9491315830355755,
        "score_model":"rank-64 antisymmetric pair embedding; score-row 80/20 holdout",
        "overall_rmse":float(np.sqrt(np.mean(squared))),
        "true_identity_rmse":float(np.sqrt(np.mean((true_pred-y)**2))),
        "oracle_fix_all_recoverable_top20_rmse":oracle_top20_rmse,
        "oracle_fix_all_recoverable_top20_score":100*(1-oracle_top20_rmse/4.382900214041612),
        "groups":{
            "both_images_correct":stats(y,pred,wrong_count==0,total_squared),
            "one_image_wrong":stats(y,pred,wrong_count==1,total_squared),
            "two_images_wrong":stats(y,pred,wrong_count==2,total_squared),
            "wrong_but_true_ids_in_top20":stats(y,pred,within_candidates,total_squared),
            "wrong_with_true_id_outside_top20":stats(y,pred,outside_candidates,total_squared),
        },
        "score_only_by_true_result":[],
    }
    buckets = (("0", -.01, .01), ("0–1", .01, 1), ("1–4", 1, 4),
               ("4–6", 4, 6), ("6–9", 6, 9), ("9–10", 9, 9.99),
               ("10", 9.99, 10.01))
    for label, low, high in buckets:
        mask = (wrong_count==0) & (y>=low) & (y<high)
        report["score_only_by_true_result"].append({
            "result_bucket":label, "count":int(mask.sum()),
            "rmse":float(np.sqrt(np.mean(squared[mask])))})
    confusion = defaultdict(lambda:[0, 0., 0.])
    for true, estimated in ((a, pa), (b, pb)):
        for i in np.flatnonzero(true != estimated):
            values = confusion[(int(true[i]), int(estimated[i]))]
            values[0] += 1
            values[1] += float(squared[i])
            values[2] += float(abs(pred[i]-y[i]))
    report["top_confusions_by_score_loss"] = [
        {"true_id":pair[0], "predicted_id":pair[1], "battle_appearances":v[0],
         "squared_error":v[1], "mean_absolute_score_error":v[2]/v[0]}
        for pair,v in sorted(confusion.items(), key=lambda item:-item[1][1])[:12]
    ]
    ordered = np.argsort(-np.abs(pred-y))
    identity_cases = []
    seen = set()
    for i in ordered:
        if not wrong[i]:
            continue
        key = ((int(a[i]),int(pa[i])) if wrong1[i] else
               (int(b[i]),int(pb[i])))
        if key in seen:
            continue
        seen.add(key)
        identity_cases.append(int(i))
        if len(identity_cases) == 10:
            break
    score_cases = [int(i) for i in ordered if not wrong[i]][:8]
    report["gallery_identity_row_indices"] = [int(z["train_row_indices"][i])
                                               for i in identity_cases]
    report["gallery_score_only_row_indices"] = [int(z["train_row_indices"][i])
                                               for i in score_cases]
    with ZipFile("data/competition.zip") as archive:
        rows = pd.read_csv(archive.open("data/data/train.csv"))
        gallery(archive, rows, z, identity_cases,
                ROOT / "score_error_gallery_identity.png",
                "Identity mistakes causing large score errors")
        gallery(archive, rows, z, score_cases,
                ROOT / "score_error_gallery_score_only.png",
                "Score-model mistakes with both identities correct")
    output = ROOT / "score_error_analysis.json"
    output.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
