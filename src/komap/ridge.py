"""얇은 판을 겨냥한 능선(ridge) 응답. Eutectic Si 전용 검출기다.

왜 필요한가: Eutectic Si는 면적이 7.0%인데 그중 77.4%가 2px 경계 띠에 들어간다(2026-09-18
실측). 경계에서 떨어진 깨끗한 픽셀이 이미지 전체의 1.6%뿐이라, 모델이 이 클래스의 생김새를
데이터에서 배우기에는 신호가 너무 적다. 그래서 생김새를 직접 계산해 채널로 넣어 준다.

무엇을 재는가: 밝기를 높이 지도로 보면 어두운 얇은 판은 골짜기 능선이다. 헤세 행렬의 고윳값
두 개 중 절댓값이 큰 쪽이 판을 가로지르는 방향의 곡률, 작은 쪽이 판을 따라가는 방향이다.
판 위에서는 이 둘이 크게 비대칭이고, 두툼한 덩어리에서는 둘 다 크다. 그 차이가 판 같음이다.
척도(sigma)를 여러 개 두고 최댓값을 취해 두께 2~6px 범위를 함께 덮는다.
"""
import numpy as np
from skimage.feature import hessian_matrix, hessian_matrix_eigvals

RIDGE_SIGMAS = (1.0, 2.0, 3.0)   # 실측 특징 길이 2~6px의 반두께에 대응
RIDGE_CH = 1

# 고정 정규화 상수. train 24장에서 정규화 이전 응답의 이미지별 최댓값이 0.251~0.347
# (중앙 0.289)로 좁게 모여 있어 이 값을 기준으로 삼는다(2026-09-18 실측).
# 이미지마다 최댓값으로 나누면 앵커 정규화가 세워 둔 이미지 간 비교 가능성이 무너지고,
# 판이 드문 이미지에서는 잡음이 1.0까지 증폭된다. test 10장은 라벨이 없어 그 상황을
# 확인할 수도 없으므로 감수하지 않는다.
# 대가는 포화다. 이 값에서 1.0으로 잘리는 픽셀은 전체의 0.0002%, Eutectic Si 픽셀의
# 0.0025%뿐이고(train 24장, 2026-09-18 실측) 잘리는 구간은 이미 확실한 판이라 판별에
# 손해가 없다. 이 숫자가 0.30을 고른 근거이므로 값을 조정하려면 여기부터 다시 재라.
RIDGE_SCALE = 0.30


def scale_response(gray_norm: np.ndarray, sigma: float) -> np.ndarray:
    """척도 하나에서 잰 원(정규화 이전) 판 같음 응답. float32 [H,W], 0 이상.

    척도 정규화(sigma^2)를 곱해 두므로 이 값끼리는 척도를 넘나들며 비교할 수 있다 —
    판 두께에 맞는 척도에서 가장 커진다. ridge_response는 이 값들의 최댓값을 쓴다."""
    if gray_norm.ndim != 2:
        raise ValueError(f"2차원 회색조가 필요하다(받은 차원 {gray_norm.ndim})")
    g = gray_norm.astype(np.float32) / 255.0
    # use_gaussian_derivatives=False 로 고전적 정의를 쓴다(skimage 기본값 경고 회피).
    h = hessian_matrix(g, sigma=sigma, order="rc", use_gaussian_derivatives=False)
    l1, l2 = hessian_matrix_eigvals(h)        # |l1| >= |l2|
    # 어두운 판: 가로지르는 곡률 l1 > 0. 따라가는 곡률 l2는 0 근처.
    r = (sigma ** 2) * np.clip(l1, 0, None) * (1.0 - np.abs(l2) / (np.abs(l1) + 1e-6))
    return r.astype(np.float32)


def ridge_response(gray_norm: np.ndarray, sigmas=RIDGE_SIGMAS) -> np.ndarray:
    """앵커 정규화된 uint8 회색조 → float32 [H,W] 0~1 판 같음 점수.

    척도별 응답의 최댓값을 취한 뒤 고정 상수 RIDGE_SCALE로 나눈다. 이미지별 최댓값이 아니라
    고정 상수를 쓰는 것이 핵심이다 — 값의 뜻이 이미지마다 달라지지 않으므로 앵커 정규화가
    세워 둔 이미지 간 비교 가능성이 그대로 유지되고, 판이 드문 이미지에서 잡음이 1.0까지
    증폭되지도 않는다."""
    if gray_norm.ndim != 2:
        raise ValueError(f"2차원 회색조가 필요하다(받은 차원 {gray_norm.ndim})")
    best = np.zeros(gray_norm.shape, np.float32)
    for s in sigmas:
        best = np.maximum(best, scale_response(gray_norm, s))
    return np.clip(best / RIDGE_SCALE, 0.0, 1.0).astype(np.float32)
