"""대회 채점 산식과 진단 지표. 산식: 이미지별 4클래스 IoU 평균 → 이미지 평균.

부재 클래스(정답·예측 모두에 없어 합집합이 0인 클래스) 처리는 세 가지다.
- "zero": IoU 0으로 평균에 넣는다. **주최 측 공식 산식**(2026-09-29 멘토링 답변) — OFFICIAL_ABSENT
- "skip": 평균에서 뺀다. 답변 전 실험 기록은 모두 이 값이라 재현을 위해 기본값으로 남긴다
- "one": IoU 1로 넣는다(비교용)
정답에 없는데 예측한 클래스는 합집합이 0이 아니므로 세 방식 모두 IoU 0이다."""
import numpy as np
from scipy import ndimage as ndi

from komap.config import BOUNDARY_BAND_PX, CLS_EUTSI, CLS_PRISI

OFFICIAL_ABSENT = "zero"


def si_pair_iou(prob, target):
    """정답 Si 픽셀에서 채널 2·3 우열로 예측한 두 클래스 IoU의 평균(동률은 공정 Si)."""
    si = (target == CLS_PRISI) | (target == CLS_EUTSI)
    if not si.any():
        return None
    primary = si & (prob[CLS_PRISI] > prob[CLS_EUTSI])
    values = []
    for pred, truth in ((primary, target == CLS_PRISI), (si & ~primary, target == CLS_EUTSI)):
        union = (pred | truth).sum()
        if union:
            values.append(float((pred & truth).sum() / union))
    return float(np.mean(values)) * 100.0


def si_pair_score(probs, targets):
    """이미지별 Si 쌍 IoU를 평균한다. Si 없는 이미지는 제외한다."""
    values = [si_pair_iou(prob, target) for prob, target in zip(probs, targets, strict=True)]
    valid = [value for value in values if value is not None]
    return float(np.mean(valid)) if valid else None


def official_miou_from_per_class_by_image(per_class_by_image):
    """재분류 결과의 이미지별 4클래스 IoU를 공식 산식으로 환산한다(None은 0)."""
    if not per_class_by_image:
        raise ValueError("이미지별 클래스 점수가 비었다")
    by_image = {
        name: float(np.mean([scores[str(k)] or 0.0 for k in range(4)]))
        for name, scores in per_class_by_image.items()
    }
    return float(np.mean(list(by_image.values()))), by_image


def iou_per_class(pred, target, n_classes):
    out = {}
    for k in range(n_classes):
        p, t = pred == k, target == k
        union = (p | t).sum()
        out[k] = None if union == 0 else float((p & t).sum() / union)
    return out


def image_miou(pred, target, n_classes, absent="skip"):
    ious = iou_per_class(pred, target, n_classes)
    vals = []
    for k, v in ious.items():
        if v is None:
            if absent == "one":
                vals.append(1.0)
            elif absent == "zero":
                vals.append(0.0)
            elif absent != "skip":
                raise ValueError(f"알 수 없는 부재 클래스 처리: {absent!r}")
            continue
        vals.append(v)
    return float(np.mean(vals)) * 100.0


def competition_score(preds, targets, n_classes=4, absent="skip"):
    per_image = [image_miou(p, t, n_classes, absent) for p, t in zip(preds, targets, strict=True)]
    per_class = {k: [] for k in range(n_classes)}
    for p, t in zip(preds, targets, strict=True):
        for k, v in iou_per_class(p, t, n_classes).items():
            if v is not None:
                per_class[k].append(v)
    return {
        "miou": float(np.mean(per_image)),
        "per_image": per_image,
        "per_class_mean": {k: (float(np.mean(v)) * 100 if v else None) for k, v in per_class.items()},
    }


def boundary_band(target, band_px=BOUNDARY_BAND_PX):
    band = np.zeros(target.shape, bool)
    for k in np.unique(target):
        m = target == k
        band |= m & ndi.binary_dilation(~m, iterations=band_px)
    return band


def boundary_error_split(pred, target, band_px=BOUNDARY_BAND_PX):
    band = boundary_band(target, band_px)
    wrong = pred != target
    return {
        "err_in_band": float(wrong[band].mean()) if band.any() else 0.0,
        "err_outside": float(wrong[~band].mean()) if (~band).any() else 0.0,
        "band_share": float(band.mean()),
    }
