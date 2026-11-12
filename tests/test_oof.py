"""oof.py: fold별 OOF 확률을 모아 원본 해상도로 되돌린다. 학습 없이 합성 npy로 검증."""
import numpy as np
import pytest
from PIL import Image

from komap import oof
from komap.io import to_rgb


def _write_oof(exp_dir, fold, name, prob):
    d = exp_dir / f"fold{fold}" / "oof_probs"
    d.mkdir(parents=True, exist_ok=True)
    np.save(d / name.replace(".png", ".npy"), prob.astype(np.float16))


def test_list_oof_maps_name_to_fold(tmp_path):
    _write_oof(tmp_path, 0, "A1_step1_sample1_image.png", np.zeros((3, 4, 4)))
    _write_oof(tmp_path, 2, "A2_step1_sample1_image.png", np.zeros((3, 4, 4)))
    m = oof.list_oof(tmp_path)
    assert m["A1_step1_sample1_image.png"][0] == 0
    assert m["A2_step1_sample1_image.png"][0] == 2
    assert m["A2_step1_sample1_image.png"][1].name == "A2_step1_sample1_image.npy"


def test_list_oof_rejects_duplicate_across_folds(tmp_path):
    _write_oof(tmp_path, 0, "A1_step1_sample1_image.png", np.zeros((3, 4, 4)))
    _write_oof(tmp_path, 1, "A1_step1_sample1_image.png", np.zeros((3, 4, 4)))
    with pytest.raises(ValueError, match="두 fold"):
        oof.list_oof(tmp_path)


def test_load_full_prob_restores_size_and_keeps_argmax(tmp_path):
    # 리샘플 해상도 8x8에서 왼쪽 절반 Si(2), 오른쪽 절반 Al(0)인 확률맵
    p = np.zeros((3, 8, 8), np.float32)
    p[2, :, :4] = 0.9; p[0, :, :4] = 0.1
    p[0, :, 4:] = 0.9; p[2, :, 4:] = 0.1
    path = tmp_path / "x.npy"; np.save(path, p.astype(np.float16))
    full = oof.load_full_prob(path, height=6, width=12)
    assert full.shape == (3, 6, 12) and full.dtype == np.float32
    assert (full.argmax(0)[:, :5] == 2).all() and (full.argmax(0)[:, 7:] == 0).all()


def test_iter_oof_yields_gray_target_and_meta(tmp_path, synth_pair):
    gray, y4, _ = synth_pair                                  # 320x240
    (tmp_path / "data" / "images").mkdir(parents=True); (tmp_path / "data" / "masks").mkdir()
    ip = tmp_path / "data" / "images" / "A1_step1_sample1_image.png"
    Image.fromarray(gray).save(ip)
    Image.fromarray(to_rgb(y4)).save(tmp_path / "data" / "masks" / "A1_step1_sample1_mask.png")
    # 폭 320은 PIXEL_SCALE에 없어 최근접(675→2.14)으로 리샘플된다. 그 크기의 확률맵을 둔다.
    from komap.preprocess import effective_factor
    f = effective_factor(320)
    hp, wp = int(round(240 * f)), int(round(320 * f))
    prob = np.zeros((3, hp, wp), np.float32); prob[0] = 1.0
    _write_oof(tmp_path / "exp", 3, ip.name, prob)

    rows = list(oof.iter_oof(tmp_path / "exp", [ip]))
    assert len(rows) == 1
    name, pf, gn, t4, meta = rows[0]
    assert name == ip.name and pf.shape == (3, 240, 320)
    assert gn.shape == (240, 320) and gn.dtype == np.uint8
    assert t4.shape == (240, 320) and set(np.unique(t4)) <= {0, 1, 2, 3}
    assert meta == {"fold": 3, "width": 320, "height": 240}


def test_iter_oof_raises_when_image_has_no_oof(tmp_path, synth_pair):
    gray, _, _ = synth_pair
    (tmp_path / "data" / "images").mkdir(parents=True)
    ip = tmp_path / "data" / "images" / "A9_step1_sample1_image.png"
    Image.fromarray(gray).save(ip)
    (tmp_path / "exp").mkdir()
    with pytest.raises(KeyError, match="A9_step1_sample1_image.png"):
        list(oof.iter_oof(tmp_path / "exp", [ip]))


def _oof_fixture(tmp_path, synth_pair, prob_fn):
    """합성 이미지 2장(fold 0·1)과 각 이미지의 OOF 확률(prob_fn(y4)→[C,h',w'])을 놓는다."""
    from komap.preprocess import effective_factor, resample
    gray, y4, _ = synth_pair
    (tmp_path / "data" / "images").mkdir(parents=True); (tmp_path / "data" / "masks").mkdir()
    paths = []
    for fold, name in [(0, "A1_step1_sample1_image.png"), (1, "A2_step1_sample1_image.png")]:
        ip = tmp_path / "data" / "images" / name
        Image.fromarray(gray).save(ip)
        Image.fromarray(to_rgb(y4)).save(tmp_path / "data" / "masks" / name.replace("_image", "_mask"))
        f = effective_factor(gray.shape[1])
        y_rs = resample(y4, f, is_mask=True)
        _write_oof(tmp_path / "exp", fold, name, prob_fn(y_rs))
        paths.append(ip)
    return tmp_path / "exp", paths


def _perfect3(y4):
    p = np.zeros((3, *y4.shape), np.float32)
    p[0][y4 == 0] = 1; p[1][y4 == 1] = 1; p[2][(y4 == 2) | (y4 == 3)] = 1
    return p


def test_score_oof_perfect_3ch_prob_scores_100(tmp_path, synth_pair):
    exp, paths = _oof_fixture(tmp_path, synth_pair, _perfect3)
    r = oof.score_oof(exp, paths, combiner="thickness", halo=False)
    assert r["miou"] == pytest.approx(100.0, abs=1e-6)
    assert set(r["per_image"]) == {p.name for p in paths}
    assert r["per_fold"] == {0: pytest.approx(100.0), 1: pytest.approx(100.0)}
    assert r["combiner"] == "thickness" and r["halo"] is False
    assert r["miou_alloy_in_test"] is None and r["miou_alloy_new"] is None


def test_score_oof_splits_alloy_overlap(tmp_path, synth_pair):
    exp, paths = _oof_fixture(tmp_path, synth_pair, _perfect3)
    r = oof.score_oof(exp, paths, halo=False, test_names=["A1_step9_sample1_image.png"])
    assert r["miou_alloy_in_test"] == pytest.approx(100.0)   # A1만 test에 있다
    assert r["miou_alloy_new"] == pytest.approx(100.0)       # A2


def test_score_oof_direct4_needs_4ch(tmp_path, synth_pair):
    exp, paths = _oof_fixture(tmp_path, synth_pair, _perfect3)
    with pytest.raises(ValueError, match="4채널"):
        oof.score_oof(exp, paths, combiner="direct4", halo=False)


def test_score_oof_custom_k_changes_result(tmp_path, synth_pair):
    exp, paths = _oof_fixture(tmp_path, synth_pair, _perfect3)
    r = oof.score_oof(exp, paths, halo=False, k={"A": 2, "B": 2})   # 큰 블록도 Eutectic → 점수 하락
    assert r["miou"] < 100.0 and r["k"] == {"A": 2, "B": 2}
