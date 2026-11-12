"""Si 덩어리의 형태 지표와 초정/공정 판정.

왜 두께만으로는 부족한가: 정답 마스크를 두께 공식으로 재현하면 10px 이상 두꺼운 덩어리
구간에서 Eutectic Si 정답률이 70.3%까지 떨어진다(2026-09-18 실측). 공정 Si는 얇은 판이
이어진 망 구조라 국소적으로 두껍게 측정되는 지점이 생기는데, 그 지점 때문에 덩어리 전체가
초정으로 넘어간다. 초정 Si는 덩어리져 있고 공정 Si는 길쭉하다는 형태 차이로 되돌린다.
"""
import numpy as np
from scipy import ndimage as ndi
from skimage.measure import label

from komap.config import THICKNESS_K
from komap.preprocess import group_of

ELONG_MIN = 8.0


def blob_features(si_mask):
    """Si 이진 마스크 → (라벨 배열, 덩어리별 지표 dict). 4방향 연결이다."""
    lab = label(si_mask, connectivity=1)
    n = int(lab.max())
    if n == 0:
        z = np.zeros(0)
        return lab, {"thickness": z, "area": z, "elongation": z}
    idx = np.arange(1, n + 1)
    dist = ndi.distance_transform_edt(si_mask)
    thickness = 2 * np.asarray(ndi.maximum(dist, lab, idx))
    area = np.bincount(lab.ravel(), minlength=n + 1)[1:].astype(float)
    # 길쭉함: 같은 두께의 원판이라면 면적은 두께²에 비례한다. 그 비를 넘는 만큼이 길쭉함이다.
    elongation = area / np.maximum(thickness, 1.0) ** 2
    return lab, {"thickness": thickness, "area": area, "elongation": elongation}


def classify(feats, width, k=None, rule="thickness", elong_min=ELONG_MIN):
    """덩어리별 Primary Si 여부(bool 배열). 순서는 라벨 1..n과 같다."""
    k = THICKNESS_K if k is None else k
    thr = width / k[group_of(width)]
    is_pri = feats["thickness"] > thr
    if rule == "thickness":
        return is_pri
    if rule == "thickness_elong":
        return is_pri & (feats["elongation"] < elong_min)
    raise ValueError(f"알 수 없는 판정 규칙: {rule}")
