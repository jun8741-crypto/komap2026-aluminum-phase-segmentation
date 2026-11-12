import numpy as np
import pytest
from komap import metrics as M


def test_perfect_is_100():
    y = np.array([[0, 1], [2, 3]])
    assert M.image_miou(y, y, 4) == 100.0


def test_all_al_prediction():
    """정답 4클래스가 25%씩일 때 전부 Al로 예측하면 Al IoU 0.25, 나머지 0 → mIoU 6.25"""
    y = np.repeat(np.arange(4), 4).reshape(4, 4)
    p = np.zeros_like(y)
    assert abs(M.image_miou(p, y, 4) - 6.25) < 1e-6


def test_absent_class_modes():
    y = np.array([[0, 1], [1, 0]]); p = y.copy()
    assert M.image_miou(p, y, 4, absent="skip") == 100.0
    assert M.image_miou(p, y, 4, absent="one") == 100.0
    p2 = np.array([[0, 1], [1, 2]])  # 부재 클래스 2를 예측 → IoU 0
    assert M.image_miou(p2, y, 4, absent="skip") < 100.0


def test_absent_class_zero_is_official():
    """주최 측 답변(2026-09-29): 정답·예측 모두에 없는 클래스도, 정답에 없는데 예측한 클래스도 IoU 0으로 평균에 넣는다."""
    y = np.array([[0, 1], [1, 0]])
    assert M.image_miou(y.copy(), y, 4, absent="zero") == 50.0  # 1, 1, 0, 0
    p2 = np.array([[0, 1], [1, 2]])  # Al 1/2, Al₃Ni 2/2, 클래스 2 예측(정답 없음) 0, 클래스 3 둘 다 없음 0
    assert M.image_miou(p2, y, 4, absent="zero") == 37.5
    assert M.OFFICIAL_ABSENT == "zero"


def test_competition_score_averages_images():
    y = np.array([[0, 1], [1, 0]])
    s = M.competition_score([y, y], [y, np.array([[0, 0], [1, 0]])], n_classes=4)
    assert s["miou"] == np.mean(s["per_image"])
    assert len(s["per_image"]) == 2


def test_competition_score_rejects_length_mismatch():
    y = np.array([[0, 1], [1, 0]])
    with pytest.raises(ValueError):
        M.competition_score([y, y], [y], n_classes=4)


def test_boundary_band_and_split(synth_pair):
    _, y4, _ = synth_pair
    band = M.boundary_band(y4, 2)
    assert 0 < band.mean() < 0.5
    p = y4.copy(); p[0, 0] = 1  # 경계 아닌 곳 1픽셀 오류
    r = M.boundary_error_split(p, y4, 2)
    assert r["err_in_band"] == 0.0 and r["err_outside"] > 0


def _si_hand_example():
    target = np.zeros((4, 4), dtype=int)
    target[:2] = 2
    target[2] = 3
    prob = np.zeros((4, 4, 4), dtype=np.float32)
    prob[3] = 0.9
    prob[2] = 0.1
    prob[2, :2, :3] = 0.9
    prob[3, :2, :3] = 0.1
    prob[2, 2, 0] = 0.9
    prob[3, 2, 0] = 0.1
    return prob, target


def test_si_pair_iou_hand_example():
    prob, target = _si_hand_example()
    assert M.si_pair_iou(prob, target) == pytest.approx(58.3333333, abs=1e-6)


def test_si_pair_iou_ignores_non_si_pixels():
    prob, target = _si_hand_example()
    changed = prob.copy()
    changed[:, 3] = np.random.default_rng(0).random((4, 4))
    assert M.si_pair_iou(changed, target) == M.si_pair_iou(prob, target)
    target[3] = 255
    assert M.si_pair_iou(changed, target) == M.si_pair_iou(prob, target)


def test_si_pair_iou_absent_class_skipped():
    prob = np.zeros((4, 2, 2))
    prob[2] = 1
    assert M.si_pair_iou(prob, np.full((2, 2), 2)) == 100.0


def test_si_pair_score_skips_none_and_returns_none_when_all_none():
    prob, target = _si_hand_example()
    empty = np.zeros_like(target)
    assert M.si_pair_score([prob, prob], [target, empty]) == M.si_pair_iou(prob, target)
    assert M.si_pair_score([prob], [empty]) is None
    with pytest.raises(ValueError):
        M.si_pair_score([prob], [target, target])


def test_official_miou_from_per_class_by_image():
    full = {str(k): 100.0 for k in range(4)}
    absent = {str(k): None for k in range(4)}
    partial = {"0": 100.0, "1": 100.0, "2": None, "3": None}
    assert M.official_miou_from_per_class_by_image({"a": partial}) == (50.0, {"a": 50.0})
    assert M.official_miou_from_per_class_by_image({"a": full, "b": absent}) == (
        50.0, {"a": 100.0, "b": 0.0}
    )
    with pytest.raises(ValueError):
        M.official_miou_from_per_class_by_image({})
    high = {str(k): 80.2 for k in range(4)}
    edge = {str(k): 80.0 for k in range(4)}
    low = {str(k): 80.1 for k in range(4)}
    assert M.official_miou_from_per_class_by_image({"a": high, "b": edge})[0] >= 80.095
    assert M.official_miou_from_per_class_by_image({"a": low, "b": edge})[0] < 80.095


