"""ridge.py: 얇은 판에만 크게 반응하는 능선 응답. 합성 도형으로 성질을 확인한다."""
import numpy as np
import pytest

from komap import ridge as R


def _canvas(h=128, w=128, bg=226):
    return np.full((h, w), bg, np.uint8)


def test_thin_plate_scores_higher_than_background():
    g = _canvas(); g[60:64, 20:110] = 50          # 두께 4px 어두운 판
    r = R.ridge_response(g)
    assert r.dtype == np.float32 and r.shape == g.shape
    assert 0.0 <= r.min() and r.max() <= 1.0
    assert r[62, 60] > 5 * r[10, 10]              # 판 위 > 평탄한 배경


def test_thin_plate_scores_higher_than_thick_blob():
    """같은 밝기라도 얇은 판이 두툼한 덩어리보다 능선 응답이 커야 한다.
    이것이 Primary Si와 Eutectic Si를 가르는 근거다."""
    plate = _canvas(); plate[60:64, 20:110] = 50          # 4px 판
    blob = _canvas(); blob[40:90, 40:90] = 50             # 50x50 덩어리
    rp = R.ridge_response(plate)[62, 60]
    rb = R.ridge_response(blob)[65, 65]                   # 덩어리 한가운데
    assert rp > 3 * rb


def test_response_is_scale_selective():
    """척도를 판 두께에 맞췄을 때 응답이 가장 크다."""
    g = _canvas(); g[60:64, 20:110] = 50
    r_small = R.ridge_response(g, sigmas=(0.5,))[62, 60]
    r_match = R.ridge_response(g, sigmas=(2.0,))[62, 60]
    r_large = R.ridge_response(g, sigmas=(8.0,))[62, 60]
    assert r_match > r_small and r_match > r_large


def test_multi_scale_max_picks_the_matching_scale():
    """여러 척도를 함께 주면 판 두께에 맞는 척도가 최댓값을 쥐어야 한다.
    scale_response는 진단용 공개 함수다 — 정규화 이전 값을 직접 들여다볼 때 쓴다."""
    g = _canvas(); g[60:64, 20:110] = 50
    best = max(R.RIDGE_SIGMAS, key=lambda s: R.scale_response(g, s)[62, 60])
    assert best == 2.0
    assert R.ridge_response(g)[62, 60] == pytest.approx(
        R.scale_response(g, 2.0)[62, 60] / R.RIDGE_SCALE)


def test_normalization_is_fixed_not_per_image():
    """정규화 상수가 이미지에 따라 달라지면 안 된다.

    같은 판을 두 이미지에 두되 한쪽에는 대비가 더 큰 판을 하나 더 얹어 이미지 최댓값을
    1.28배로 올린다. 고정 상수로 나누면 공통 판의 값이 두 이미지에서 같고, 이미지별
    최댓값으로 나누면 최댓값이 오른 쪽에서 값이 그만큼 눌려 이 단언이 깨진다."""
    a = _canvas(); a[20:24, 20:110] = 50
    b = a.copy(); b[90:94, 20:110] = 0                 # 대비가 더 큰 판 추가
    ra, rb = R.ridge_response(a), R.ridge_response(b)
    assert rb[92, 60] > ra[22, 60]                     # 강한 판이 실제로 더 크게 반응한다
    assert rb[22, 60] == pytest.approx(ra[22, 60])     # 공통 판의 값은 이미지와 무관하다


def test_noise_only_image_stays_near_zero():
    """판이 전혀 없는 이미지는 응답이 0 근처에 머물러야 한다(잡음 증폭 금지)."""
    rng = np.random.default_rng(0)
    g = np.clip(np.round(226 + rng.normal(0, 3, (128, 128))), 0, 255).astype(np.uint8)
    assert R.ridge_response(g).max() < 0.2


def test_flat_image_is_all_zero():
    r = R.ridge_response(_canvas())
    assert np.allclose(r, 0.0)


def test_rejects_non_2d():
    with pytest.raises(ValueError, match="2차원"):
        R.ridge_response(np.zeros((3, 8, 8), np.uint8))
