"""밝기 정규화와 배율 정규화. 정답이 필요 없는 연산만 둔다(test에도 동일 적용)."""
import logging

import numpy as np
from skimage.filters import threshold_multiotsu
from skimage.transform import resize

from komap.config import (
    ANCHOR_MIN_REL_SEP,
    ANCHOR_TARGETS,
    GROUP_BY_WIDTH,
    PIXEL_SCALE,
    RESAMPLE_MODE,
)

logger = logging.getLogger(__name__)

NORM_MODES = ("anchor3", "anchor2", "none")


def otsu_groups(gray):
    """Multi-Otsu 3분할의 군 마스크 셋. 앵커 통계를 군마다 따로 내기 위한 공통 부분이다."""
    t = threshold_multiotsu(gray, classes=3)
    g = gray.astype(np.float64)
    return [g < t[0], (g >= t[0]) & (g < t[1]), g >= t[1]]


def cluster_gap(gray):
    """군 평균과 군 최빈값의 차이(최댓값). 라벨 없이 쓸 수 있는 실패 신호 후보다.

    한 군이 단일 봉우리면 평균과 최빈값이 가깝다. 군이 두 상을 삼키면 평균이 봉우리에서
    멀어진다. A71의 어두운 군이 그 예다(실제 상 4, 군 평균 45.9)."""
    g = gray.astype(np.float64)
    out = []
    for mask in otsu_groups(gray):
        peak = float(np.argmax(np.bincount(gray[mask].ravel(), minlength=256)))
        out.append(abs(float(g[mask].mean()) - peak))
    return float(max(out))


def anchor_levels(gray: np.ndarray) -> tuple[float, float, float | None]:
    """Multi-Otsu 3분할로 어두운·중간·밝은 군의 평균 밝기를 구한다.
    진짜 2군 이미지에서는 Multi-Otsu가 빈 골짜기 대신 한쪽 덩어리를 쪼개므로,
    중간 군이 다크·브라이트 양쪽에서 얼마나 떨어져 있는지(rel_sep)로 실재 여부를 판정한다.
    rel_sep이 기준(ANCHOR_MIN_REL_SEP) 미만이면 mid를 None으로 돌려준다."""
    g = gray.astype(np.float64)
    # 완전 평탄 이미지는 밝은 군과 어두운 군이 같아 구간 선형 매핑의 기울기가 정의되지 않는다.
    # threshold_multiotsu가 먼저 영어 오류를 던지므로 그 앞에서 우리 말로 막는다.
    if float(g.max()) - float(g.min()) < 1e-6:
        raise ValueError("완전 평탄 이미지: 앵커 정규화 불가")
    t = threshold_multiotsu(gray, classes=3)
    dark = g[g < t[0]].mean()
    mid = g[(g >= t[0]) & (g < t[1])].mean()
    bright = g[g >= t[1]].mean()
    if bright - dark < 1e-6:
        raise ValueError("완전 평탄 이미지: 앵커 정규화 불가")
    rel_sep = min(mid - dark, bright - mid) / (bright - dark)
    if rel_sep < ANCHOR_MIN_REL_SEP:
        logger.warning(
            "군 분리 거리 rel_sep=%.4f가 기준 %.4f 미만이라 2군 이미지로 판단, 2점 매핑으로 폴백",
            rel_sep, ANCHOR_MIN_REL_SEP,
        )
        return dark, None, bright
    return dark, mid, bright


def anchor_normalize(gray: np.ndarray) -> np.ndarray:
    """어두운·중간·밝은 군 평균을 (50, 140, 226)에 구간 선형으로 맞춘다.
    중간 군이 없는 2군 이미지는 어두운/밝은 군만으로 2점 매핑을 쓴다."""
    g = gray.astype(np.float64)
    dark, mid, bright = anchor_levels(gray)
    if mid is None:
        xs = [0.0, dark, bright, 255.0]
        ys = [0.0, ANCHOR_TARGETS[0], ANCHOR_TARGETS[2], 255.0]
    else:
        xs = [0.0, dark, mid, bright, 255.0]
        ys = [0.0, *ANCHOR_TARGETS, 255.0]
    return np.clip(np.round(np.interp(g, xs, ys)), 0, 255).astype(np.uint8)


def normalize(gray: np.ndarray, mode: str = "anchor3") -> np.ndarray:
    """선택한 방식으로 회색조를 정규화한다.

    ``anchor3``는 기존 3점 앵커 경로를 그대로 쓰고, ``anchor2``는 중간 군의
    존재 여부와 무관하게 어두운·밝은 군만 맞춘다. ``none``은 독립된 uint8 복사본이다.
    """
    if mode == "anchor3":
        return anchor_normalize(gray)
    if mode == "anchor2":
        g = gray.astype(np.float64)
        dark, _, bright = anchor_levels(gray)
        xs = [0.0, dark, bright, 255.0]
        ys = [0.0, ANCHOR_TARGETS[0], ANCHOR_TARGETS[2], 255.0]
        return np.clip(np.round(np.interp(g, xs, ys)), 0, 255).astype(np.uint8)
    if mode == "none":
        return gray.astype(np.uint8, copy=True)
    raise ValueError(f"알 수 없는 정규화 모드: {mode}")


def group_of(width: int) -> str:
    if width in GROUP_BY_WIDTH:
        return GROUP_BY_WIDTH[width]
    nearest = min(GROUP_BY_WIDTH, key=lambda w: abs(w - width))
    return GROUP_BY_WIDTH[nearest]


def scale_factor(width: int) -> float:
    """1440px 이미지의 픽셀 크기를 1.0으로 두고, 이 폭의 이미지를 같은 μm/px로 맞추는 확대 배수."""
    if width in PIXEL_SCALE:
        return PIXEL_SCALE[width]
    nearest = min(PIXEL_SCALE, key=lambda w: abs(w - width))
    return PIXEL_SCALE[nearest]


def effective_factor(width: int, mode: str = RESAMPLE_MODE, scale: float = 1.0,
                     size_correction: bool = True) -> float:
    """실제로 적용할 리샘플 배수. scale_factor는 1440px을 기준(1.0)으로 한 확대 배수인데,
    전부 확대하면 675px 이미지가 4.6배 넓어져 메모리·시간이 커진다. 축소 모드는 기준을
    가장 거친 675px 쪽으로 옮겨 상대 배율만 유지한다(1440과 B군이 작아진다).

    scale은 모든 폭에 공통으로 곱하는 전역 배율이다. 상대 배율(A군과 B군의 비)은 곱셈이라
    그대로 보존되고 전체 해상도만 오르내린다. 얇은 Eutectic Si 판(원본 2~6px)이 네트워크에
    몇 픽셀로 들어가는지를 이 값이 정한다. size_correction=False는 폭별 base를 1.0으로
    두는 것과 같아 mode와 무관하다."""
    if not size_correction:
        return scale
    if mode == "upsample":
        base = scale_factor(width)
    elif mode == "downsample_to_1440":
        base = scale_factor(width) / scale_factor(675)
    else:
        raise ValueError(f"알 수 없는 리샘플 모드: {mode}")
    return base * scale


def resample(arr: np.ndarray, factor: float, is_mask: bool) -> np.ndarray:
    if abs(factor - 1.0) < 1e-6:
        return arr.copy()
    h, w = arr.shape[:2]
    shape = (int(round(h * factor)), int(round(w * factor)))
    if is_mask:
        return resize(arr, shape, order=0, preserve_range=True, anti_aliasing=False).astype(arr.dtype)
    out = resize(arr, shape, order=1, preserve_range=True, anti_aliasing=factor < 1)
    return np.clip(np.round(out), 0, 255).astype(np.uint8)
