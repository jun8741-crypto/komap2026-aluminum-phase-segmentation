"""fold별로 저장된 OOF(out-of-fold) 확률을 모아 원본 해상도로 되돌린다.

OOF 확률은 train.py가 리샘플 해상도로 저장한다(fold 앙상블은 같은 격자여야 하므로).
그러나 채점과 두께 공식은 원본 해상도에서만 옳다 — 두께 기준값이 원본 폭에서 나오기
때문이다(predict.py와 같은 이유). 그래서 여기서 predict.predict_image와 똑같은 순서로
확률을 먼저 원본 크기로 되돌린 뒤 argmax·후처리에 넘긴다.

원본 크기와 리샘플 배수는 이미지 파일에서 유도한다(effective_factor는 폭의 함수).
별도 메타 파일을 두지 않는 이유: W1 실험(runs/w1_*)에는 메타가 없고, 폭만 알면 결정된다.
"""
from pathlib import Path

import numpy as np
from skimage.transform import resize

from komap import io
from komap.combine import combine
from komap.cv import overlap_tags
from komap.metrics import competition_score
from komap.preprocess import anchor_normalize


def list_oof(exp_dir: Path) -> dict[str, tuple[int, Path]]:
    """이미지명 → (fold, npy 경로). 한 이미지는 정확히 한 fold의 검증셋에만 속해야 한다."""
    out: dict[str, tuple[int, Path]] = {}
    for npy in sorted(Path(exp_dir).glob("fold*/oof_probs/*.npy")):
        fold = int(npy.parent.parent.name[4:])
        name = npy.name.replace(".npy", ".png")
        if name in out:
            raise ValueError(f"{name} 이(가) 두 fold({out[name][0]}, {fold})에 모두 있다 — fold 분할이 겹쳤다")
        out[name] = (fold, npy)
    return out


def load_full_prob(npy_path: Path, height: int, width: int) -> np.ndarray:
    """float16 [C,H',W'] → float32 [C,height,width]. 채널별 선형 보간(predict.py와 동일)."""
    prob = np.load(npy_path).astype(np.float32)
    return restore_full_prob(prob, height, width).astype(np.float32, copy=False)


def restore_full_prob(prob: np.ndarray, height: int, width: int) -> np.ndarray:
    """확률맵을 원본 좌표계로 복원한다.

    OOF와 제출 추론이 같은 보간 규칙을 공유해야 서로 다른 배율의 확률을 같은 지점에서
    평균할 수 있다. 이미 원본 크기라면 불필요한 보간을 피하고 입력을 그대로 돌려준다.
    """
    if prob.shape[1:] == (height, width):
        return prob
    return np.stack([
        resize(c, (height, width), order=1, preserve_range=True, anti_aliasing=False) for c in prob
    ])


def iter_oof(exp_dir: Path, image_paths: list[Path]):
    """(name, prob_full[C,H,W], gray_norm[H,W] uint8, target4[H,W], meta) 를 이미지마다 낸다.
    gray_norm은 후광 제거가, target4는 채점이, meta의 width는 두께 공식이 쓴다."""
    table = list_oof(exp_dir)
    for ip in image_paths:
        ip = Path(ip)
        if ip.name not in table:
            raise KeyError(f"{ip.name}: {exp_dir} 에 OOF 확률이 없다(fold 학습이 끝나지 않았거나 이름 불일치)")
        fold, npy = table[ip.name]
        gray = io.load_gray(ip)
        h, w = gray.shape
        prob = load_full_prob(npy, h, w)
        target = io.load_mask(io.mask_path_for(ip)).astype(np.int64)
        yield ip.name, prob, anchor_normalize(gray), target, {"fold": fold, "width": w, "height": h}


def score_oof(exp_dir, image_paths, combiner="thickness", halo=True, k=None, absent="skip",
              test_names=None, **kw):
    """train 전체의 OOF 예측을 4클래스로 결합해 대회 산식으로 채점한다.

    이것이 W2의 모든 선택 기준이다. valid 20장은 여기에 끼지 않는다(스펙 §5-1).
    halo=True면 결합 전에 후광 제거를 건다(후광 제거 자체의 효과도 이 플래그로 잰다)."""
    preds, targets, names, folds = [], [], [], []
    for name, prob, gray_norm, target, meta in iter_oof(exp_dir, image_paths):
        p4 = combine(combiner, prob, width=meta["width"], gray_norm=gray_norm if halo else None, k=k, **kw)
        preds.append(p4); targets.append(target); names.append(name); folds.append(meta["fold"])
    s = competition_score(preds, targets, 4, absent)
    per_image = dict(zip(names, s["per_image"], strict=True))
    per_fold = {}
    for f in sorted(set(folds)):
        per_fold[f] = float(np.mean([v for v, ff in zip(s["per_image"], folds, strict=True) if ff == f]))
    out = {"miou": s["miou"], "per_class": s["per_class_mean"], "per_image": per_image,
           "per_fold": per_fold, "combiner": combiner, "halo": halo, "k": k,
           "miou_alloy_in_test": None, "miou_alloy_new": None}
    if test_names:
        tags = overlap_tags(names, test_names)
        for flag, key in [(True, "miou_alloy_in_test"), (False, "miou_alloy_new")]:
            sel = [v for n, v in per_image.items() if tags[n] == flag]
            out[key] = float(np.mean(sel)) if sel else None
    return out
