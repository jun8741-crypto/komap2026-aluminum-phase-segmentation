"""train.py 스모크 테스트. 실데이터가 있어야 의미가 있으므로 없으면 건너뛴다.

fold 수를 2로 낮추는 이유: StratifiedGroupKFold는 층(step)마다 최소 n_splits개의 표본을
요구한다. --limit로 이미지를 10장까지 줄이면 기본 5-fold는 만들 수 없다(ValueError).
스모크 테스트의 목적은 한 에폭이 끝까지 돌아 산출물이 떨어지는지 확인하는 것이므로
가장 작은 분할인 2-fold로 충분하다.
"""
import pytest

from komap.config import DATA_DIR

SMOKE_ARGV = [
    "--exp", "smoke", "--fold", "0", "--encoder", "resnet18", "--weights", "none",
    "--loss", "ce", "--tile", "128", "--epochs", "1", "--bs", "2",
    "--limit", "10", "--n-folds", "2",
]


def test_train_augmentation_flags_parse_and_defaults():
    from komap.train import parse_args

    defaults = parse_args(["--exp", "test", "--fold", "0"])
    enabled = parse_args([
        "--exp", "test", "--fold", "0", "--anchor-jitter", "0.75", "--no-interp-geo",
    ])

    assert defaults.anchor_jitter == 0.0
    assert defaults.no_interp_geo is False
    assert enabled.anchor_jitter == pytest.approx(0.75)
    assert enabled.no_interp_geo is True


@pytest.mark.skipif(not (DATA_DIR / "1train").exists(), reason="실데이터 없음")
def test_train_one_epoch_smoke(tmp_path):
    from komap.train import parse_args, run_fold

    args = parse_args([*SMOKE_ARGV, "--runs-dir", str(tmp_path)])
    r = run_fold(args)

    out = tmp_path / "smoke" / "fold0"
    assert (out / "best.pt").exists()
    assert 0 <= r["best_val_miou3"] <= 100
    assert r["epoch"] == 0
    # 에폭별 로그와 검증 이미지 OOF 확률이 실제로 떨어져야 한다.
    assert len((out / "log.jsonl").read_text().strip().splitlines()) == 1
    assert list((out / "oof_probs").glob("*.npy"))


@pytest.mark.skipif(not (DATA_DIR / "1train").exists(), reason="실데이터 없음")
def test_train_with_hint_smoke(tmp_path):
    """규칙 지도 힌트를 켜면 입력이 5채널이 되고 임계값이 체크포인트에 남아야 한다."""
    import torch
    from komap.train import parse_args, run_fold

    run_fold(parse_args([*SMOKE_ARGV, "--hint", "--runs-dir", str(tmp_path)]))
    st = torch.load(tmp_path / "smoke" / "fold0" / "best.pt", map_location="cpu", weights_only=True)
    assert st["args"]["hint"] is True and len(st["args"]["hint_thr"]) == 2
    assert st["model"]["encoder.conv1.weight"].shape[1] == 5


def test_anchor_jitter_rejects_non_probability():
    """확률이 아닌 값을 조용히 받으면 체크포인트에 적힌 값과 실제 동작이 어긋난다.
    음수·NaN은 증강이 꺼지고 1 초과는 항상 켜지는데 args에는 입력값이 그대로 남는다."""
    import pytest
    from komap.train import parse_args

    base = ["--exp", "x", "--fold", "0"]
    assert parse_args([*base, "--anchor-jitter", "0.3"]).anchor_jitter == 0.3
    assert parse_args([*base, "--anchor-jitter", "0"]).anchor_jitter == 0.0
    assert parse_args([*base, "--anchor-jitter", "1"]).anchor_jitter == 1.0
    for bad in ("-0.1", "1.1", "nan", "inf", "-inf"):
        with pytest.raises(SystemExit):
            parse_args([*base, "--anchor-jitter", bad])


def test_train_scope_and_metric_flags_parse_and_defaults():
    from komap.train import parse_args

    base = ["--exp", "x", "--fold", "0"]
    defaults = parse_args(base)
    assert defaults.loss_scope == "all" and defaults.select_metric == "miou"
    for flag in (["--loss-scope", "si"], ["--select-metric", "si_pair"]):
        with pytest.raises(SystemExit):
            parse_args([*base, *flag])
    selected = parse_args([*base, "--n-classes", "4", "--loss-scope", "si",
                           "--select-metric", "si_pair"])
    assert selected.loss_scope == "si" and selected.select_metric == "si_pair"


@pytest.mark.skipif(not (DATA_DIR / "1train").exists(), reason="실데이터 없음")
def test_train_si_only_smoke(tmp_path):
    import json

    import numpy as np
    import torch
    from komap.train import parse_args, run_fold

    args = parse_args([*SMOKE_ARGV, "--loss", "lovasz", "--n-classes", "4",
                       "--loss-scope", "si", "--select-metric", "si_pair",
                       "--runs-dir", str(tmp_path)])
    result = run_fold(args)
    out = tmp_path / "smoke" / "fold0"
    assert (out / "best.pt").exists()
    assert 0 <= result["best_val_si_pair"] <= 100
    assert "best_val_miou3" in result and result["select_metric"] == "si_pair"
    assert np.load(next((out / "oof_probs").glob("*.npy"))).shape[0] == 4
    assert "val_si_pair" in json.loads((out / "log.jsonl").read_text().splitlines()[0])
    checkpoint = torch.load(out / "best.pt", map_location="cpu", weights_only=True)
    assert checkpoint["args"]["loss_scope"] == "si"


def test_train_head_flag_parse_and_defaults():
    from komap.train import parse_args

    base = ["--exp", "x", "--fold", "0"]
    assert parse_args(base).head == "full"
    for extra in (["--head", "si2"],
                  ["--head", "si2", "--n-classes", "4"],
                  ["--head", "si2", "--n-classes", "4", "--loss-scope", "si"]):
        with pytest.raises(SystemExit):
            parse_args([*base, *extra])
    selected = parse_args([*base, "--n-classes", "4", "--loss-scope", "si",
                           "--select-metric", "si_pair", "--head", "si2"])
    assert selected.head == "si2"


@pytest.mark.skipif(not (DATA_DIR / "1train").exists(), reason="실데이터 없음")
def test_train_si2_smoke(tmp_path):
    import json

    import numpy as np
    import torch
    from komap.train import parse_args, run_fold

    args = parse_args([*SMOKE_ARGV, "--loss", "lovasz", "--n-classes", "4",
                       "--loss-scope", "si", "--select-metric", "si_pair", "--head", "si2",
                       "--runs-dir", str(tmp_path)])
    result = run_fold(args)
    out = tmp_path / "smoke" / "fold0"
    assert result["head"] == "si2" and result["select_metric"] == "si_pair"
    assert 0 <= result["best_val_si_pair"] <= 100 and "best_val_miou3" in result
    st = torch.load(out / "best.pt", map_location="cpu", weights_only=True)
    assert st["model"]["segmentation_head.0.weight"].shape[0] == 2
    assert st["args"]["head"] == "si2"
    probs = [np.load(p) for p in (out / "oof_probs").glob("*.npy")]
    assert probs and all(p.shape[0] == 4 and p.dtype == np.float16 for p in probs)
    assert all(not p[:2].any() and np.allclose(p[2] + p[3], 1, atol=0.01) for p in probs)
    assert "val_si_pair" in json.loads((out / "log.jsonl").read_text().splitlines()[0])


@pytest.mark.skipif(not (DATA_DIR / "1train").exists(), reason="실데이터 없음")
def test_train_edge_aux_smoke(tmp_path):
    """보조 손실 학습 뒤 체크포인트·로그·기존 추론 로더의 호환성을 검증한다."""
    import json

    import numpy as np
    import torch

    from komap.model import build_model
    from komap.predict import load_models
    from komap.train import parse_args, run_fold

    run_fold(parse_args([*SMOKE_ARGV, '--edge-aux-weight', '0.5', '--runs-dir', str(tmp_path)]))
    out = tmp_path / 'smoke' / 'fold0'
    checkpoint = torch.load(out / 'best.pt', map_location='cpu', weights_only=True)
    assert list(checkpoint['model']) == list(build_model('resnet18', 'none', 3).state_dict())
    assert not any(key.startswith('edge_head') for key in checkpoint['model'])
    assert checkpoint['args']['edge_aux_weight'] == 0.5
    record = json.loads((out / 'log.jsonl').read_text().splitlines()[0])
    assert {'train_seg_loss', 'train_edge_loss'} <= record.keys()
    assert abs(record['train_loss'] - record['train_seg_loss'] - 0.5 * record['train_edge_loss']) < 1e-5
    loaded = load_models(tmp_path / 'smoke', folds=[0])
    assert len(loaded.models) == 1 and loaded.n_classes == 3
    assert all(np.load(path).shape[0] == 3 for path in (out / 'oof_probs').glob('*.npy'))
