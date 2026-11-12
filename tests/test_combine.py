"""combine.py: 확률 맵 → 4클래스 마스크 결합기 3종. 합성 확률맵으로 규칙을 하나씩 확인한다."""
import numpy as np
import pytest

from komap import combine as C
from komap.config import CLS_AL, CLS_AL3NI, CLS_EUTSI, CLS_PRISI

W, H = 320, 240   # width/60 = 5.3px 기준 (그룹 A 최근접)


def _prob4(y4, conf=0.9):
    """정답 마스크에서 4채널 확률맵을 만든다. 정답 클래스 conf, 나머지 균등."""
    p = np.full((4, *y4.shape), (1 - conf) / 3, np.float32)
    for k in range(4):
        p[k][y4 == k] = conf
    return p


def test_combine_thickness_from_3ch_prob(synth_pair):
    _, y4, _ = synth_pair
    p3 = np.zeros((3, H, W), np.float32)
    p3[0][y4 == 0] = 1; p3[1][y4 == 1] = 1; p3[2][(y4 == 2) | (y4 == 3)] = 1
    out = C.combine_thickness(p3, width=W)
    assert (out[40:120, 20:100] == CLS_PRISI).all()      # 두께 80 블록 → Primary
    assert (out[150:154, 200:260] == CLS_EUTSI).all()    # 두께 4 판 → Eutectic
    assert (out[y4 == 0] == CLS_AL).all() and (out[y4 == 1] == CLS_AL3NI).all()


def test_combine_thickness_merges_4ch_then_resplits(synth_pair):
    """4채널 확률이 들어와도 Si를 합친 뒤 두께로 다시 가른다 — 모델의 Pri/Eut 채널은 무시된다."""
    _, y4, _ = synth_pair
    swapped = y4.copy(); swapped[y4 == 2] = 3; swapped[y4 == 3] = 2   # 모델이 Pri/Eut를 뒤집어 예측
    out = C.combine_thickness(_prob4(swapped), width=W)
    assert (out[40:120, 20:100] == CLS_PRISI).all() and (out[150:154, 200:260] == CLS_EUTSI).all()


def test_combine_thickness_accepts_custom_k(synth_pair):
    _, y4, _ = synth_pair
    p = _prob4(y4)
    # k를 아주 작게(폭/2 = 160px 기준) 주면 두께 80 블록도 Eutectic이 된다
    out = C.combine_thickness(p, width=W, k={"A": 2, "B": 2})
    assert (out[40:120, 20:100] == CLS_EUTSI).all()


def test_combine_direct4_is_plain_argmax(synth_pair):
    _, y4, _ = synth_pair
    swapped = y4.copy(); swapped[y4 == 2] = 3; swapped[y4 == 3] = 2
    out = C.combine_direct4(_prob4(swapped))
    assert np.array_equal(out, swapped)                  # 두께 공식이 끼어들지 않는다


def test_combine_direct4_rejects_3ch():
    with pytest.raises(ValueError, match="4채널"):
        C.combine_direct4(np.zeros((3, 8, 8), np.float32))


def test_combine_blob_alpha1_follows_probability(synth_pair):
    """alpha=1: 두께를 무시하고 덩어리 평균 확률만 본다 → 뒤집힌 예측을 그대로 따른다."""
    _, y4, _ = synth_pair
    swapped = y4.copy(); swapped[y4 == 2] = 3; swapped[y4 == 3] = 2
    out = C.combine_blob(_prob4(swapped), width=W, alpha=1.0)
    assert (out[40:120, 20:100] == CLS_EUTSI).all() and (out[150:154, 200:260] == CLS_PRISI).all()


def test_combine_blob_alpha0_follows_thickness(synth_pair):
    """alpha=0: 확률을 무시하고 두께만 본다 → 뒤집힌 예측이 교정된다."""
    _, y4, _ = synth_pair
    swapped = y4.copy(); swapped[y4 == 2] = 3; swapped[y4 == 3] = 2
    out = C.combine_blob(_prob4(swapped), width=W, alpha=0.0)
    assert (out[40:120, 20:100] == CLS_PRISI).all() and (out[150:154, 200:260] == CLS_EUTSI).all()


def test_combine_blob_component_is_whole(synth_pair):
    """덩어리 안에서 픽셀별로 Pri/Eut가 섞이면 안 된다(덩어리 단위 판정)."""
    _, y4, _ = synth_pair
    p = _prob4(y4)
    # 큰 블록의 절반만 Eutectic 확률을 높여 픽셀 단위로 갈리게 유도
    p[2, 40:80, 20:100], p[3, 40:80, 20:100] = 0.2, 0.7
    out = C.combine_blob(p, width=W, alpha=0.5)
    assert len(np.unique(out[40:120, 20:100])) == 1


def _long_plate_prob():
    """두께 20(기준 5.3 초과)에 길쭉함 15인 판 하나. 두께 결합기와 형태 결합기가 갈리는 도형이다."""
    y4 = np.zeros((H, W), int)
    y4[100:120, 10:310] = CLS_PRISI
    return y4, _prob4(y4)


def test_combine_shape_keeps_compact_block_as_primary(synth_pair):
    """두툼한 덩어리는 길쭉하지 않으므로 초정 판정이 그대로 남는다."""
    _, y4, _ = synth_pair
    out = C.combine_shape(_prob4(y4), width=W)
    assert (out[40:120, 20:100] == CLS_PRISI).all()
    assert (out[150:154, 200:260] == CLS_EUTSI).all()


def test_combine_shape_returns_thick_but_long_plate_to_eutectic():
    _, p = _long_plate_prob()
    assert (C.combine_thickness(p, width=W)[100:120, 10:310] == CLS_PRISI).all()
    assert (C.combine_shape(p, width=W)[100:120, 10:310] == CLS_EUTSI).all()


def test_combine_shape_elong_min_is_tunable():
    """되돌림 조건은 길쭉함 < 임계값이 '아닐' 때다. 임계값을 실제 길쭉함(15)보다 올리면 초정으로 남는다."""
    _, p = _long_plate_prob()
    assert (C.combine_shape(p, width=W, elong_min=20.0)[100:120, 10:310] == CLS_PRISI).all()
    assert (C.combine_shape(p, width=W, elong_min=12.0)[100:120, 10:310] == CLS_EUTSI).all()


def test_combine_shape_rule_thickness_matches_thickness_combiner():
    """rule="thickness"로 부르면 기존 두께 결합기와 픽셀 단위로 같아야 한다(회귀 방지)."""
    _, p = _long_plate_prob()
    assert np.array_equal(C.combine_shape(p, width=W, rule="thickness"), C.combine_thickness(p, width=W))


def test_combine_by_name_dispatch(synth_pair):
    _, y4, _ = synth_pair
    p = _prob4(y4)
    assert np.array_equal(C.combine("direct4", p, width=W), C.combine_direct4(p))
    assert set(C.COMBINERS) == {"thickness", "direct4", "blob", "shape", "learned"}
    with pytest.raises(KeyError):
        C.combine("nope", p, width=W)


def test_learned_combiner_has_common_signature():
    import inspect

    parameters = inspect.signature(C.combine_learned).parameters
    assert tuple(parameters) == (
        "prob",
        "width",
        "gray_norm",
        "k",
        "model",
        "prob4",
        "threshold",
    )
    assert parameters["threshold"].default is None
