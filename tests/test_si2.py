"""Si 2채널 헤드와 체인 완비성 점검의 계약 테스트."""
import json

import numpy as np
import pytest
import torch

from komap import si2


def test_args_differences_by_definition():
    assert si2.args_differences({"a": 1}, {"a": 1}) == {}
    assert si2.args_differences({"a": 1}, {"a": 2}) == {"a": (1, 2)}
    assert si2.args_differences({}, {"a": 2}) == {"a": (None, 2)}
    assert si2.args_differences({"a": 1}, {}) == {"a": (1, None)}
    saved = {"exp": "old", "fold": 1, "runs_dir": "a", "hint_thr": [1, 2]}
    expected = {"exp": "new", "fold": 0, "runs_dir": "b", "hint_thr": None, "head": "si2"}
    assert si2.args_differences(saved, expected) == {"head": (None, "si2")}


def test_oof_problems_detects_each_defect(tmp_path):
    exp, ref = tmp_path / "exp", tmp_path / "ref"
    fold = exp / "fold0"
    probs = fold / "oof_probs"
    probs.mkdir(parents=True)
    ref_probs = ref / "fold0" / "oof_probs"
    ref_probs.mkdir(parents=True)
    name = "A1_step1_sample1_image.npy"
    p = np.linspace(0.1, 0.9, 48, dtype=np.float16).reshape(6, 8)
    good = np.stack([np.zeros_like(p), np.zeros_like(p), p, 1 - p])
    np.save(probs / name, good)
    np.save(ref_probs / name, good)
    (fold / "result.json").write_text(json.dumps({"head": "si2", "select_metric": "si_pair"}))
    (fold / "best.pt").write_bytes(b"checkpoint")
    def check():
        return si2.oof_problems(exp, 0, [name.replace(".npy", ".png")], ref)
    assert check() == []
    result = fold / "result.json"
    result.unlink()
    assert check()
    result.write_text(json.dumps({"head": "si2", "select_metric": "si_pair"}))
    (fold / "best.pt").unlink()
    assert check()
    (fold / "best.pt").write_bytes(b"checkpoint")
    result.write_text(json.dumps({"head": "full", "select_metric": "si_pair"}))
    assert check()
    result.write_text(json.dumps({"head": "si2", "select_metric": "si_pair"}))
    (probs / name).unlink()
    assert any("누락" in issue for issue in check())
    np.save(probs / name, good)
    np.save(probs / "extra.npy", good)
    assert any("여분" in issue for issue in check())
    (probs / "extra.npy").unlink()
    bad_dtype = good.astype(np.float32)
    np.save(probs / name, bad_dtype)
    assert check()
    np.save(probs / name, good[:3])
    assert check()
    bad_channel = good.copy()
    bad_channel[0, 0, 0] = 0.1
    np.save(probs / name, bad_channel)
    assert check()
    bad_nan = good.copy()
    bad_nan[2, 0, 0] = np.nan
    np.save(probs / name, bad_nan)
    assert check()
    bad = good.copy()
    bad[2] = 0.9 - bad[3]
    np.save(probs / name, bad)
    assert check()
    np.save(probs / name, good)
    np.save(ref_probs / name, np.zeros((4, 5, 8), np.float16))
    assert check()
    (ref_probs / name).unlink()
    assert check()


def test_output_channels():
    assert si2.output_channels(4, "full") == 4
    assert si2.output_channels(3, "full") == 3
    assert si2.output_channels(4, "si2") == 2
    with pytest.raises(ValueError):
        si2.output_channels(3, "si2")
    with pytest.raises(ValueError):
        si2.output_channels(4, "foo")


def test_remap_target_si2_by_definition():
    target = torch.tensor([[0, 1, 2, 3, 255]], dtype=torch.long)
    original = target.clone()
    result = si2.remap_target_si2(target)
    assert torch.equal(result, torch.tensor([[255, 255, 0, 1, 255]]))
    assert result.dtype == target.dtype and result.device == target.device
    assert torch.equal(target, original)


@pytest.mark.parametrize("dtype", [np.float16, np.float32])
def test_expand_prob_si2_places_channels_2_and_3(dtype):
    p = np.array([[[0.2, 0.7]], [[0.8, 0.3]]], dtype=dtype)
    out = si2.expand_prob_si2(p)
    assert out.shape == (4, 1, 2) and out.dtype == dtype
    assert not out[:2].any()
    assert np.array_equal(out[2:], p)
    assert np.allclose(out.sum(0), p.sum(0))
    with pytest.raises(ValueError):
        si2.expand_prob_si2(out)


def test_expand_prob_full_returns_same_object():
    p = np.zeros((4, 2, 2))
    assert si2.expand_prob(p, "full") is p
    with pytest.raises(ValueError):
        si2.expand_prob(p, "foo")
