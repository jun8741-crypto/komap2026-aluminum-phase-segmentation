"""3클래스 예측을 4클래스 대회 마스크로 바꾸는 규칙. 초정/공정 Si는 덩어리 단위 두께로 가른다."""
import json
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi
from skimage.measure import label

from komap.config import CLS_AL, CLS_AL3NI, CLS_EUTSI, CLS_PRISI, CLS_SI3, THICKNESS_K
from komap.preprocess import group_of

AL, NI, SI, PRI, EUT = CLS_AL, CLS_AL3NI, CLS_SI3, CLS_PRISI, CLS_EUTSI

K_FIT_FILE = "k_fit.json"


def load_k(exp_dir) -> dict | None:
    """runs/<exp>/k_fit.json 이 있으면 재추정 k, 없으면 None(→ config 기본값)."""
    p = Path(exp_dir) / K_FIT_FILE
    if not p.exists():
        return None
    return {g: int(v) for g, v in json.loads(p.read_text())["k"].items()}


def component_thickness(si_mask):
    lab = label(si_mask, connectivity=1)          # 4방향 연결
    n = int(lab.max())
    if n == 0:
        return lab, np.zeros(0)
    dist = ndi.distance_transform_edt(si_mask)
    thick = 2 * ndi.maximum(dist, lab, np.arange(1, n + 1))   # 최대 내접원 지름
    return lab, np.asarray(thick)


def thickness_rule(pred3, width, k=None):
    k = THICKNESS_K if k is None else k
    thr = width / k[group_of(width)]
    si = pred3 == SI
    out = pred3.copy()
    lab, thick = component_thickness(si)
    if thick.size == 0:
        return out
    is_primary = np.r_[False, thick > thr]        # 라벨 0(배경) 자리 채움
    out[si] = np.where(is_primary[lab[si]], PRI, EUT)
    return out


def halo_fix(gray_norm, pred3):
    """Al–Si 경계의 흐린 중간 밝기 테두리가 Al3Ni로 오판된 것을 되돌린다."""
    p = pred3.copy()
    if (p == AL).sum() == 0 or (p == SI).sum() == 0:
        return p  # 둘 중 하나가 없으면 경계 후광 자체가 있을 수 없다
    fp = np.ones((3, 3), bool)
    al_core = ndi.binary_erosion(p == AL, iterations=2)
    si_core = ndi.binary_erosion(p == SI, iterations=1)
    al_med = np.median(gray_norm[al_core]) if al_core.any() else np.median(gray_norm[p == AL])
    si_med = np.median(gray_norm[si_core]) if si_core.any() else np.median(gray_norm[p == SI])
    mid = (al_med + si_med) / 2
    near_al = ndi.binary_dilation(p == AL, structure=fp)
    near_si = ndi.binary_dilation(p == SI, structure=fp)
    halo = (p == NI) & near_al & near_si
    p[halo & (gray_norm >= mid)] = AL
    p[halo & (gray_norm < mid)] = SI
    return p
