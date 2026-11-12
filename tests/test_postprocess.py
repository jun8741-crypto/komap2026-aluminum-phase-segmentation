import numpy as np
from komap import postprocess as P


def test_thickness_rule_splits_block_and_plate(synth_pair):
    _, y4, _ = synth_pair
    pred3 = y4.copy(); pred3[pred3 == 3] = 2
    out = P.thickness_rule(pred3, width=320)          # width/60 = 5.3px 기준
    assert (out[40:120, 20:100] == 2).all()           # 두께 80 블록 → 초정
    assert (out[150:154, 200:260] == 3).all()         # 두께 4 판 → 공정
    assert (out[y4 == 0] == 0).all() and (out[y4 == 1] == 1).all()


def test_component_is_whole():
    """한 덩어리는 통째로 한 클래스여야 한다."""
    p = np.zeros((50, 50), int); p[10:40, 10:40] = 2
    out = P.thickness_rule(p, width=675)
    assert len(np.unique(out[10:40, 10:40])) == 1


def test_halo_fix_no_al_or_no_si_is_noop(caplog):
    """Al이나 Si가 아예 없는 예측(전부 Al3Ni)에서는 후광 보정 대상이 존재할 수 없으므로
    원본을 그대로 돌려주고 경고도 남기지 않아야 한다."""
    g = np.random.default_rng(0).integers(0, 256, (20, 20)).astype(np.uint8)
    p3 = np.full((20, 20), 1, int)  # 전부 Al3Ni
    with caplog.at_level("WARNING"):
        out = P.halo_fix(g, p3)
    assert np.array_equal(out, p3)
    assert not caplog.records


def test_halo_fix_reassigns_sandwiched_pixels():
    g = np.full((20, 20), 226, np.uint8); g[:, 10:] = 50
    g[:, 9] = 138                                      # Al|Si 경계의 중간 밝기 한 줄
    p3 = np.zeros((20, 20), int); p3[:, 10:] = 2; p3[:, 9] = 1   # 임계값이 그 줄을 Al3Ni로 오판
    out = P.halo_fix(g, p3)
    assert (out[:, :10] == 0).all() and (out[:, 10:] == 2).all()
