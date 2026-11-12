"""규칙 기반 파이프라인(대조군). 학습하는 값은 임계값 2개뿐.
3점 앵커 정규화 → train 공통 임계값 3분류 → 후광 제거 → 두께 공식 → 4클래스."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from komap import io
from komap.config import RUNS_DIR
from komap.metrics import competition_score
from komap.postprocess import halo_fix, thickness_rule
from komap.preprocess import anchor_normalize


def _hist3(gray, y3):
    return np.stack([np.bincount(gray[y3 == k].ravel(), minlength=256) for k in range(3)])


def _best_pair(H):
    """모든 (t1, t2) 조합의 3클래스 mIoU를 누적합으로 한 번에 계산해 최댓값 쌍을 반환."""
    C = np.cumsum(H, axis=1); tot = H.sum(1); allc = C.sum(0)
    ts = np.arange(1, 256); t1 = ts[:, None]; t2 = ts[None, :]
    below = lambda k, t: C[k, t - 1]
    tp_si = below(2, t1); pr_si = allc[t1 - 1]
    iou_si = tp_si / (tot[2] + pr_si - tp_si + 1e-9)
    tp_ni = below(1, t2) - below(1, t1); pr_ni = allc[t2 - 1] - allc[t1 - 1]
    iou_ni = tp_ni / (tot[1] + pr_ni - tp_ni + 1e-9)
    tp_al = tot[0] - below(0, t2); pr_al = allc[-1] - allc[t2 - 1]
    iou_al = tp_al / (tot[0] + pr_al - tp_al + 1e-9)
    m = (iou_si + iou_ni + iou_al) / 3
    m[t1 >= t2] = -1
    i, j = np.unravel_index(m.argmax(), m.shape)
    return int(ts[i]), int(ts[j])


def fit_thresholds(train_pairs):
    H = sum(_hist3(g, y3) for g, y3 in train_pairs)
    return _best_pair(H)


def segment3(gray_norm, t1, t2):
    p = np.zeros(gray_norm.shape, np.int64)
    p[gray_norm < t2] = 1
    p[gray_norm < t1] = 2
    return p


def _load(split, with_mask=True):
    out = []
    for ip in io.list_split(split):
        g = anchor_normalize(io.load_gray(ip))
        y = io.load_mask(io.mask_path_for(ip)) if with_mask else None
        out.append((ip.name, g, y))
    return out


def run(split="valid", out_dir=None):
    train = _load("train")
    t1, t2 = fit_thresholds([(g, io.to3(y)) for _, g, y in train])
    data = _load(split, with_mask=(split != "test"))
    preds, targets, names = [], [], []
    out_dir = Path(out_dir or RUNS_DIR / "rule")
    (out_dir / split).mkdir(parents=True, exist_ok=True)
    for name, g, y in data:
        p3 = halo_fix(g, segment3(g, t1, t2))
        p4 = thickness_rule(p3, width=g.shape[1])
        Image.fromarray(io.to_rgb(p4)).save(out_dir / split / name.replace("_image", "_mask"))
        preds.append(p4); names.append(name)
        if y is not None:
            targets.append(y)
    result = {"t1": t1, "t2": t2, "split": split, "n": len(preds)}
    if targets:
        s = competition_score(preds, targets, 4)
        result.update(miou=s["miou"], per_class=s["per_class_mean"], per_image=dict(zip(names, s["per_image"])))
    (out_dir / f"{split}_result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="valid", choices=["valid", "test"])
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    r = run(a.split, a.out)
    print(json.dumps({k: v for k, v in r.items() if k != "per_image"}, ensure_ascii=False, indent=2))
