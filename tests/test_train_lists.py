import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import torch

from conftest import make_synth
from komap import train


def _pair(root: Path, alloy: int, seed: int) -> Path:
    images = root / "images"
    masks = root / "masks"
    images.mkdir(parents=True, exist_ok=True)
    masks.mkdir(parents=True, exist_ok=True)
    name = f"A{alloy}_step1_sample1_image.png"
    gray, _, rgb = make_synth(seed=seed)
    Image.fromarray(gray).save(images / name)
    Image.fromarray(rgb).save(masks / name.replace("_image.png", "_mask.png"))
    return images / name


def _args(tmp_path, select_epoch="last", extra=()):
    trp, vap = _pair(tmp_path, 1, 0), _pair(tmp_path, 2, 1)
    tr = tmp_path / "train.txt"
    va = tmp_path / "valid.txt"
    tr.write_text(f"{trp}\n")
    va.write_text(f"{vap}\n")
    return train.parse_args([
        "--exp", "u", "--fold", "0", "--runs-dir", str(tmp_path),
        "--train-list", str(tr), "--val-list", str(va),
        "--select-epoch", select_epoch, "--oof-res", "original",
        "--encoder", "resnet18", "--weights", "none", "--loss", "ce",
        "--tile", "128", "--epochs", "2", "--bs", "2", "--spi", "2",
        *extra,
    ])


def test_parse_args_list_defaults_and_pairing():
    args = train.parse_args(["--exp", "x", "--fold", "0"])
    assert (args.train_list, args.val_list) == (None, None)
    assert args.select_epoch == "best" and args.oof_res == "resampled"
    with pytest.raises(SystemExit):
        train.parse_args(["--exp", "x", "--fold", "0", "--train-list", "x"])


def test_list_mode_last_writes_atomic_outputs(tmp_path, monkeypatch):
    args = _args(tmp_path)
    original = train.evaluate
    calls = []
    monkeypatch.setattr(train, "evaluate", lambda *a, **k: (calls.append(1), original(*a, **k))[1])
    result = train.run_fold(args)
    out = tmp_path / "u" / "fold0"
    logs = [json.loads(x) for x in (out / "log.jsonl").read_text().splitlines()]
    assert len(calls) == 1 and "val_miou3" not in logs[0] and "val_miou3" in logs[1]
    assert result["epoch"] == 1 and result["select_epoch"] == "last" and len(result["run_id"]) == 32
    prob = np.load(out / "oof_probs" / "A2_step1_sample1_image.npy")
    assert prob.shape == (3, 240, 320) and prob.dtype == np.float16
    assert np.allclose(prob.astype(np.float32).sum(0), 1, atol=0.01)
    saved = json.loads((out / "args.json").read_text())
    complete = json.loads((out / "COMPLETE.json").read_text())
    checkpoint = torch.load(out / "best.pt", map_location="cpu", weights_only=True)
    assert saved["run_id"] == checkpoint["args"]["run_id"] == complete["run_id"]
    assert complete["args_sha256"] == hashlib.sha256((out / "args.json").read_bytes()).hexdigest()


def test_list_mode_best_keeps_existing_selection(tmp_path, monkeypatch):
    args = _args(tmp_path, "best")
    original = train.evaluate
    calls = []
    monkeypatch.setattr(train, "evaluate", lambda *a, **k: (calls.append(1), original(*a, **k))[1])
    result = train.run_fold(args)
    assert len(calls) == 2 and result["select_epoch"] == "best" and "best_val_miou3" in result


def test_list_mode_rerun_cleans_stale_probability(tmp_path):
    args = _args(tmp_path)
    first = train.run_fold(args)
    stale = tmp_path / "u" / "fold0" / "oof_probs" / "Z.npy"
    np.save(stale, np.zeros(1))
    second = train.run_fold(args)
    assert not stale.exists() and first["run_id"] != second["run_id"]


def test_list_mode_rejects_overlap_and_escape(tmp_path):
    args = _args(tmp_path)
    Path(args.val_list).write_text(Path(args.train_list).read_text())
    with pytest.raises(ValueError, match="겹친다"):
        train.run_fold(args)
    args = _args(tmp_path / "again")
    args.exp = "../x"
    with pytest.raises(ValueError):
        train.run_fold(args)


def test_list_mode_si_pair_selection_records_metric(tmp_path):
    args = _args(tmp_path, "best", extra=["--n-classes", "4", "--loss-scope", "si",
                                     "--select-metric", "si_pair"])
    result = train.run_fold(args)
    out = tmp_path / "u" / "fold0"
    logs = [json.loads(row) for row in (out / "log.jsonl").read_text().splitlines()]
    assert result["select_metric"] == "si_pair" and "best_val_si_pair" in result
    assert len(logs) == 2 and all("val_si_pair" in row and "val_miou3" in row for row in logs)
    assert np.load(next((out / "oof_probs").glob("*.npy"))).shape[0] == 4


def test_list_mode_last_si_pair_keeps_metadata_keys(tmp_path):
    args = _args(tmp_path, "last", extra=["--n-classes", "4", "--loss-scope", "si",
                                     "--select-metric", "si_pair"])
    result = train.run_fold(args)
    logs = [json.loads(row) for row in (tmp_path / "u" / "fold0" / "log.jsonl").read_text().splitlines()]
    assert "val_si_pair" not in logs[0] and "val_si_pair" in logs[-1]
    assert {"select_epoch", "n_train", "n_val", "n_oof_saved", "run_id",
            "best_val_si_pair", "select_metric"} <= set(result)


def test_list_mode_default_log_has_no_si_pair_key(tmp_path):
    result = train.run_fold(_args(tmp_path))
    logs = [json.loads(row) for row in (tmp_path / "u" / "fold0" / "log.jsonl").read_text().splitlines()]
    assert all("val_si_pair" not in row for row in logs)
    assert set(result) == {"best_val_miou3", "epoch", "select_epoch", "n_train", "n_val",
                           "n_oof_saved", "run_id"}


def _fake_selection_run(tmp_path, monkeypatch, metric):
    calls = []

    def fake_evaluate(model, paths, args, device, oof_res="resampled"):
        n = len(calls)
        calls.append(n)
        return ({"miou": [90.0, 10.0][n], "si_pair": [10.0, 90.0][n],
                 "per_class_mean": {}},
                {"A2_step1_sample1_image.png": np.full((4, 8, 8), n, np.float32)})

    monkeypatch.setattr(train, "evaluate", fake_evaluate)
    args = _args(tmp_path, "best", extra=["--n-classes", "4", "--select-metric", metric])
    result = train.run_fold(args)
    out = tmp_path / "u" / "fold0"
    prob = np.load(out / "oof_probs" / "A2_step1_sample1_image.npy")
    logs = [json.loads(row) for row in (out / "log.jsonl").read_text().splitlines()]
    return result, prob, logs


def test_list_mode_best_si_pair_picks_si_pair_winner(tmp_path, monkeypatch):
    result, prob, logs = _fake_selection_run(tmp_path, monkeypatch, "si_pair")
    assert result["epoch"] == 1 and result["best_val_si_pair"] == 90.0
    assert result["best_val_miou3"] == 10.0 and np.all(prob == 1)
    assert [row["val_si_pair"] for row in logs] == [10.0, 90.0]


def test_list_mode_best_miou_picks_miou_winner(tmp_path, monkeypatch):
    result, prob, logs = _fake_selection_run(tmp_path, monkeypatch, "miou")
    assert result["epoch"] == 0 and result["best_val_miou3"] == 90.0
    assert np.all(prob == 0) and "best_val_si_pair" not in result
    assert all("val_si_pair" not in row for row in logs)


def test_list_mode_si2_head_saves_four_channel_probs(tmp_path):
    args = _args(tmp_path, "best", extra=["--n-classes", "4", "--loss", "lovasz",
                                          "--loss-scope", "si", "--select-metric", "si_pair",
                                          "--head", "si2"])
    result = train.run_fold(args)
    out = tmp_path / "u" / "fold0"
    prob = np.load(out / "oof_probs" / "A2_step1_sample1_image.npy")
    logs = [json.loads(row) for row in (out / "log.jsonl").read_text().splitlines()]
    assert prob.shape == (4, 240, 320) and prob.dtype == np.float16
    assert not prob[:2].any() and np.allclose(prob[2] + prob[3], 1, atol=0.01)
    assert result["head"] == "si2" and all("val_si_pair" in row for row in logs)


def test_list_mode_default_result_has_no_head_key(tmp_path):
    result = train.run_fold(_args(tmp_path))
    assert "head" not in result
