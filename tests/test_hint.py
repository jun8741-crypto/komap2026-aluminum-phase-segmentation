"""hint.py: 규칙 지도 힌트 채널. 합성 이미지에서 지도가 정답과 일치하고 dropout이 채널만 지운다."""
import json

import numpy as np
import torch

from komap import hint as Hn


def test_rule_hint_matches_synth_truth(synth_pair):
    gray, y4, _ = synth_pair
    # 합성 이미지는 50/140/226 세 값 + 잡음이라 임계값 95·183이면 정확히 갈린다
    h = Hn.rule_hint(gray, t1=95, t2=183, width=320)
    assert h.dtype == np.uint8 and h.shape == y4.shape
    assert (h == y4).mean() > 0.99


def test_onehot_shape_and_sum():
    h = np.array([[0, 1], [2, 3]], np.uint8)
    o = Hn.onehot(h)
    assert o.shape == (4, 2, 2) and o.dtype == np.float32
    assert (o.sum(0) == 1).all() and o[3, 1, 1] == 1


def test_drop_zeroes_hint_channels_only():
    x = torch.ones(5, 4, 4)
    y = Hn.drop(x, p=1.0, rng=np.random.default_rng(0))
    assert (y[0] == 1).all() and (y[1:] == 0).all()
    z = Hn.drop(x, p=0.0, rng=np.random.default_rng(0))
    assert torch.equal(z, x)


def test_fit_thresholds_cached_reads_file(tmp_path):
    (tmp_path / "rule").mkdir()
    (tmp_path / "rule" / "thresholds.json").write_text(json.dumps({"t1": 90, "t2": 180}))
    assert Hn.fit_thresholds_cached(tmp_path) == (90, 180)
