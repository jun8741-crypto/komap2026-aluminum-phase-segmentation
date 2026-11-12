import numpy as np
from komap import preprocess as pp


def test_anchor_maps_three_levels(synth_pair):
    gray, y4, _ = synth_pair
    out = pp.anchor_normalize(gray)
    assert out.dtype == np.uint8
    assert abs(np.median(out[y4 == 0]) - 226) <= 3
    assert abs(np.median(out[y4 == 1]) - 140) <= 3
    assert abs(np.median(out[(y4 == 2) | (y4 == 3)]) - 50) <= 3


def test_anchor_is_invariant_to_gain_and_offset(synth_pair):
    gray, _, _ = synth_pair
    shifted = np.clip(gray.astype(float) * 0.7 + 30, 0, 255).astype(np.uint8)
    a, b = pp.anchor_normalize(gray), pp.anchor_normalize(shifted)
    assert np.abs(a.astype(int) - b.astype(int)).mean() < 4


def test_group_and_scale():
    assert pp.group_of(675) == "A" and pp.group_of(1230) == "B"
    assert pp.scale_factor(1440) == 1.0
    assert abs(pp.scale_factor(675) - 2.14) < 1e-6


def test_resample_mask_keeps_labels(synth_pair):
    _, y4, _ = synth_pair
    up = pp.resample(y4, 2.0, is_mask=True)
    assert up.shape == (480, 640)
    assert set(np.unique(up)) <= {0, 1, 2, 3}


def test_anchor_two_group_fallback(caplog):
    """중간 밝기 군이 없는 2군 이미지(Al만 밝고 나머지는 어두운 경우)에서
    Multi-Otsu 3분할이 실패하지 않고 2점 매핑으로 안전하게 폴백하는지 확인한다.

    Multi-Otsu는 두 덩어리만 있는 히스토그램에서도 분산 최소화 관점에서 두 임계값을
    '빈 골짜기'가 아니라 한쪽 덩어리 내부에 두는 쪽을 선택한다(브루트포스로 실측 확인:
    이 이미지에서 최적해는 t=[50, 64]로 어두운 군 내부를 가른다). 그래서 중간 군의
    '픽셀 비율'만으로는 2군 여부를 판정할 수 없다(오히려 0.27로 크게 나온다). 대신 그
    가짜 중간 군이 다크·브라이트 양쪽에서 실제로 얼마나 떨어져 있는지(rel_sep)를 보면
    이 이미지는 0.027로 실데이터 최소치(0.336, 2026-09-17 실측)에 한참 못 미친다.
    ANCHOR_MIN_REL_SEP(0.15) 기준으로 자연스럽게 폴백이 걸린다."""
    rng = np.random.default_rng(0)
    img = np.full((200, 200), 226.0)
    img[:, 100:] = 50
    img += rng.normal(0, 3, img.shape)
    img = np.clip(np.round(img), 0, 255).astype(np.uint8)

    dark, mid, bright = pp.anchor_levels(img)
    assert mid is None

    with caplog.at_level("WARNING"):
        out = pp.anchor_normalize(img)
    assert any(r.levelname == "WARNING" for r in caplog.records)
    assert abs(np.median(out[:, :100].astype(int)) - 226) <= 3
    assert abs(np.median(out[:, 100:].astype(int)) - 50) <= 3


def test_anchor_three_group_not_fallback(synth_pair):
    """정상적인 3군 이미지(Al·Al3Ni·Si)에서는 폴백이 걸리지 않고 세 군 평균이 그대로 나와야 한다."""
    gray, _, _ = synth_pair
    assert pp.anchor_levels(gray)[1] is not None


def test_anchor_levels_rejects_flat_image():
    """완전 평탄 이미지는 밝은 군과 어두운 군이 같아 앵커 정규화가 정의되지 않는다.
    skimage의 영어 오류 대신 무엇이 잘못됐는지 말해 주는 우리 오류로 막는다."""
    import pytest

    flat = np.full((32, 32), 128, np.uint8)
    with pytest.raises(ValueError, match="완전 평탄 이미지"):
        pp.anchor_levels(flat)


def test_effective_factor_applies_global_scale():
    from komap.preprocess import effective_factor
    base = effective_factor(1440)
    assert base == 1.0
    assert effective_factor(1440, scale=2.0) == 2.0
    # 폭마다 다른 기본 배수에 곱으로 얹힌다(상대 배율은 보존된다)
    assert effective_factor(675, scale=2.0) == effective_factor(675) * 2.0
    r = effective_factor(675, scale=1.5) / effective_factor(1440, scale=1.5)
    assert abs(r - effective_factor(675) / effective_factor(1440)) < 1e-9


def test_cluster_gap_separates_clear_modes_from_swallowed_phase():
    rng = np.random.default_rng(1)
    clear = np.clip(
        np.concatenate(
            [
                rng.normal(40, 2, 4096),
                rng.normal(140, 2, 4096),
                rng.normal(220, 2, 4096),
            ]
        ),
        0,
        255,
    ).astype(np.uint8).reshape(96, 128)
    swallowed = np.concatenate(
        [
            np.full(3000, 10),
            np.full(1096, 70),
            np.full(4096, 140),
            np.full(4096, 220),
        ]
    ).astype(np.uint8).reshape(96, 128)

    assert pp.cluster_gap(clear) < 5
    assert pp.cluster_gap(swallowed) > 30


def test_cluster_gap_matches_pre_move_characterization():
    rng = np.random.default_rng(7)
    gray = np.clip(
        np.concatenate(
            [
                rng.normal(35, 4, 1024),
                rng.normal(130, 7, 1536),
                rng.normal(225, 3, 1536),
            ]
        ),
        0,
        255,
    ).astype(np.uint8).reshape(64, 64)

    assert pp.cluster_gap(gray) == 0.7100130039011674
