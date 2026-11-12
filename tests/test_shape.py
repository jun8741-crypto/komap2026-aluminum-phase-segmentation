"""shape.py: 덩어리 형태 지표와 초정/공정 판정. 합성 도형으로 성질을 확인한다."""
import numpy as np
import pytest

from komap import shape as S


def _mask(h=200, w=200):
    return np.zeros((h, w), bool)


def test_blob_features_separates_plate_and_block():
    m = _mask()
    m[20:24, 10:190] = True            # 얇고 긴 판: 두께 4, 길쭉함 큼
    m[100:150, 100:150] = True         # 두툼한 블록: 두께 50, 길쭉함 작음
    lab, f = S.blob_features(m)
    assert lab.max() == 2
    order = np.argsort(f["thickness"])
    plate, block = order[0], order[1]
    assert f["thickness"][plate] < 10 and f["thickness"][block] > 40
    assert f["elongation"][plate] > 10 * f["elongation"][block]


def test_classify_thickness_matches_current_rule():
    """rule="thickness"는 기존 두께 공식과 같은 판정을 내야 한다(회귀 방지)."""
    from komap.postprocess import thickness_rule
    from komap.config import CLS_PRISI, CLS_SI3
    p3 = np.zeros((240, 320), np.int64)
    p3[40:120, 20:100] = CLS_SI3       # 두께 80 → 초정 (320/60 = 5.3 기준)
    p3[150:154, 200:260] = CLS_SI3     # 두께 4 → 공정
    ref = thickness_rule(p3, width=320)
    lab, f = S.blob_features(p3 == CLS_SI3)
    is_pri = S.classify(f, width=320, rule="thickness")
    got = np.where(is_pri[lab - 1][p3 == CLS_SI3], CLS_PRISI, 3)
    assert (got == ref[p3 == CLS_SI3]).all()


def test_thickness_elong_returns_long_thick_blob_to_eutectic():
    """두께 기준은 넘지만 길쭉한 덩어리는 공정으로 되돌린다."""
    m = _mask(400, 400)
    m[100:120, 20:380] = True          # 두께 20(기준 400/60=6.7 초과), 아주 길쭉
    lab, f = S.blob_features(m)
    assert S.classify(f, width=400, rule="thickness")[0]            # 두께만 보면 초정
    assert not S.classify(f, width=400, rule="thickness_elong")[0]  # 형태를 보면 공정


def test_thickness_elong_keeps_compact_blob_as_primary():
    m = _mask(400, 400)
    m[100:200, 100:200] = True         # 두툼한 정사각 덩어리
    lab, f = S.blob_features(m)
    assert S.classify(f, width=400, rule="thickness_elong")[0]


def test_unknown_rule_raises():
    m = _mask(); m[10:20, 10:20] = True
    _, f = S.blob_features(m)
    with pytest.raises(ValueError, match="알 수 없는"):
        S.classify(f, width=320, rule="nope")
