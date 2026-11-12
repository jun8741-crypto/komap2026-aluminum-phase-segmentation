"""predict.py end-to-end 테스트. 학습 없이 가짜 체크포인트 하나로 전 구간을 통과시킨다."""
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from komap.io import to_rgb


def _loaded(n_classes=3):
    """실제 신경망 없이 추론 흐름만 검사할 최소 모델 묶음이다."""
    from komap.predict import LoadedModels

    return LoadedModels([object()], 128, n_classes, None, 1.0, False)


def test_t1_default_combiner_keeps_thickness_path(monkeypatch, tmp_path):
    """T1: 결합기 기본값은 두께 공식이며 산출물에도 그대로 기록한다."""
    import json
    from komap import predict as Pr

    image_path = tmp_path / "A1_step1_sample1_image.png"
    Image.fromarray(np.zeros((4, 5), dtype=np.uint8)).save(image_path)
    probability = np.full((3, 4, 5), 1 / 3, dtype=np.float32)
    calls = []
    monkeypatch.setattr(Pr, "pick_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(Pr, "load_models", lambda *args, **kwargs: _loaded())
    monkeypatch.setattr(Pr, "predict_ensemble_prob", lambda *args, **kwargs: probability)
    monkeypatch.setattr(
        Pr,
        "prob_to_mask",
        lambda *args, **kwargs: calls.append("thickness") or np.zeros((4, 5), dtype=np.int64),
    )

    result = Pr.run(tmp_path / "exp", [image_path], tmp_path / "out", folds=[0])
    saved = json.loads((tmp_path / "out" / "score.json").read_text())

    assert calls == ["thickness"]
    assert result["combiner"] == saved["combiner"] == "thickness"


def test_t2_learned_combiner_requires_prob4_at_parse_time(capsys):
    """T2: 4채널 원천 누락은 긴 추론 전에 인자 파싱 단계에서 거부한다."""
    from komap.predict import parse_cli

    with pytest.raises(SystemExit):
        parse_cli(["--exp", "x", "--combiner", "learned"])

    assert "--prob4-exp" in capsys.readouterr().err


def test_t3_learned_combiner_rejects_three_class_prob4(monkeypatch, tmp_path):
    """T3: prob4 실험 체크포인트가 4클래스가 아니면 즉시 멈춘다."""
    from komap import predict as Pr

    main_exp = tmp_path / "main"
    wrong_prob4_exp = tmp_path / "wrong_prob4"
    loaded = {main_exp: _loaded(3), wrong_prob4_exp: _loaded(3)}
    monkeypatch.setattr(Pr, "pick_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(Pr, "load_models", lambda path, *args: loaded[path])

    with pytest.raises(ValueError, match="4클래스"):
        Pr.run(
            main_exp,
            [],
            tmp_path / "out",
            folds=[0],
            combiner="learned",
            halo=False,
            prob4_exp_dir=wrong_prob4_exp,
            prob4_exp_names=["wrong_prob4"],
            reclass_model_path=tmp_path / "model.pkl",
        )


def test_t4_missing_fold_is_rejected_unless_folds_are_explicit(tmp_path):
    """T4: 전체 앙상블에서 누락 fold를 밝히고 명시 선택은 허용한다."""
    from komap.predict import load_models

    exp = tmp_path / "exp"
    for fold in range(4):
        _save_fake_fold(exp, fold, tile=128)

    with pytest.raises(FileNotFoundError, match=r"없는 fold.*4"):
        load_models(exp, None, torch.device("cpu"))

    loaded = load_models(exp, [0], torch.device("cpu"))
    assert len(loaded.models) == 1


def test_t5_learned_combiner_and_prob4_experiments_are_recorded(monkeypatch, tmp_path):
    """T5: learned 산출물은 결합기와 4채널 실험 이름을 함께 기록한다."""
    import json
    from komap import predict as Pr

    main_exp = tmp_path / "main"
    prob4_exp = tmp_path / "prob4"
    loaded = {main_exp: _loaded(3), prob4_exp: _loaded(4)}
    model_path = tmp_path / "model.pkl"
    model_path.write_bytes(b"model")
    monkeypatch.setattr(Pr, "pick_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(Pr, "load_models", lambda path, *args: loaded[path])
    monkeypatch.setattr(Pr.ReclassModel, "load", lambda path: object())

    result = Pr.run(
        main_exp,
        [],
        tmp_path / "out",
        folds=[0],
        combiner="learned",
        halo=False,
        prob4_exp_dir=prob4_exp,
        prob4_exp_names=["prob4"],
        reclass_model_path=model_path,
    )
    saved = json.loads((tmp_path / "out" / "score.json").read_text())

    assert result["combiner"] == saved["combiner"] == "learned"
    assert result["prob4_exps"] == saved["prob4_exps"] == ["prob4"]


def test_t6_missing_reclass_model_reports_regeneration_command(monkeypatch, tmp_path):
    """T6: 재분류 모델이 없으면 정확한 재생성 명령을 안내한다."""
    from komap import predict as Pr

    main_exp = tmp_path / "main"
    prob4_exp = tmp_path / "prob4"
    loaded = {main_exp: _loaded(3), prob4_exp: _loaded(4)}
    monkeypatch.setattr(Pr, "pick_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(Pr, "load_models", lambda path, *args: loaded[path])

    with pytest.raises(
        FileNotFoundError,
        match=r"\.venv/bin/python -m komap\.experiments\.reclass_fit",
    ):
        Pr.run(
            main_exp,
            [],
            tmp_path / "out",
            folds=[0],
            combiner="learned",
            halo=False,
            prob4_exp_dir=prob4_exp,
            prob4_exp_names=["prob4"],
            reclass_model_path=tmp_path / "missing.pkl",
        )


def test_learned_combiner_rejects_halo_before_creating_output(tmp_path):
    """후광을 켠 learned 추론은 학습 특징과 달라지므로 산출물 생성 전에 거부한다."""
    from komap import predict as Pr

    out_dir = tmp_path / "out"
    with pytest.raises(ValueError, match="halo") as caught:
        Pr.run(
            tmp_path / "main",
            [],
            out_dir,
            combiner="learned",
            halo=True,
        )

    assert "--no-halo" in str(caught.value)
    assert not out_dir.exists()


@pytest.fixture
def fake_exp(tmp_path, synth_pair):
    """합성 이미지·마스크 한 쌍과, 초기값 그대로인 fold0 체크포인트를 만든다."""
    from komap.model import build_model

    gray, y4, _ = synth_pair
    (tmp_path / "data" / "images").mkdir(parents=True)
    (tmp_path / "data" / "masks").mkdir()
    Image.fromarray(gray).save(tmp_path / "data" / "images" / "A1_step1_sample1_image.png")
    Image.fromarray(to_rgb(y4)).save(tmp_path / "data" / "masks" / "A1_step1_sample1_mask.png")

    m = build_model("resnet18", "none", 3)
    (tmp_path / "exp" / "fold0").mkdir(parents=True)
    torch.save(
        {"model": m.state_dict(),
         "args": {"encoder": "resnet18", "arch": "unetplusplus", "n_classes": 3, "tile": 128}},
        tmp_path / "exp" / "fold0" / "best.pt",
    )
    return tmp_path


def test_predict_writes_rgba_and_score(fake_exp):
    from komap import predict as Pr

    r = Pr.run(exp_dir=fake_exp / "exp",
               image_paths=[fake_exp / "data" / "images" / "A1_step1_sample1_image.png"],
               out_dir=fake_exp / "out", folds=[0], tta=False, halo=True)

    png = fake_exp / "out" / "A1_step1_sample1_mask.png"
    assert png.exists()
    with Image.open(png) as im:
        assert im.mode == "RGBA", "제출 마스크는 대회 정답과 같은 RGBA여야 한다"
        arr = np.array(im)
    # 리샘플 해상도가 아니라 원본 해상도로 되돌아와야 한다.
    assert arr.shape == (240, 320, 4)
    assert "miou_skip" in r and "miou_one" in r
    assert (fake_exp / "out" / "score.json").exists()


def test_load_models_raises_without_checkpoint(tmp_path):
    from komap.predict import load_models

    with pytest.raises(FileNotFoundError):
        load_models(tmp_path, None, torch.device("cpu"))


def _save_fake_fold(exp_dir, fold, tile, n_classes=3, resample_scale=None):
    from komap.model import build_model

    d = exp_dir / f"fold{fold}"
    d.mkdir(parents=True)
    m = build_model("resnet18", "none", n_classes)
    # resample_scale=None이면 키 자체를 넣지 않는다 — 인자가 생기기 전의 옛 체크포인트 모사.
    a = {"encoder": "resnet18", "arch": "unetplusplus", "n_classes": n_classes, "tile": tile}
    if resample_scale is not None:
        a["resample_scale"] = resample_scale
    torch.save({"model": m.state_dict(), "args": a}, d / "best.pt")


def _stub_fold_loading(monkeypatch, exp_dir, folds):
    """fold 선택 가드만 검증하도록 체크포인트 내용과 모델 복원을 가볍게 대체한다."""
    from komap import predict as Pr

    for fold in folds:
        path = exp_dir / f"fold{fold}" / "best.pt"
        path.parent.mkdir(parents=True)
        path.touch()

    checkpoint = {
        "model": {},
        "args": {"encoder": "stub", "n_classes": 3, "tile": 128},
    }

    class StubModel:
        def load_state_dict(self, state):
            return None

        def to(self, device):
            return self

        def eval(self):
            return self

    monkeypatch.setattr(Pr.torch, "load", lambda *args, **kwargs: checkpoint)
    monkeypatch.setattr(Pr, "build_model", lambda *args, **kwargs: StubModel())


def test_load_models_rejects_missing_explicit_fold(monkeypatch, tmp_path):
    from komap.predict import load_models

    exp = tmp_path / "exp"
    _stub_fold_loading(monkeypatch, exp, range(4))

    with pytest.raises(FileNotFoundError, match=r"없는 fold.*4"):
        load_models(exp, [0, 1, 2, 3, 4], torch.device("cpu"))


def test_load_models_rejects_extra_fold_by_default(monkeypatch, tmp_path):
    from komap.predict import load_models

    exp = tmp_path / "exp"
    _stub_fold_loading(monkeypatch, exp, range(6))

    with pytest.raises(ValueError, match=r"여분 fold.*5.*--folds"):
        load_models(exp, None, torch.device("cpu"))


def test_load_models_records_full_fold_scope(monkeypatch, tmp_path):
    from komap.predict import load_models, metric_scope_name

    exp = tmp_path / "exp"
    _stub_fold_loading(monkeypatch, exp, range(5))

    loaded = load_models(exp, None, torch.device("cpu"))

    assert loaded.folds == (0, 1, 2, 3, 4)
    assert metric_scope_name("valid", loaded.folds) == "valid_full_fold_ensemble"


def test_load_models_records_explicit_partial_fold_scope(monkeypatch, tmp_path):
    from komap.predict import load_models, metric_scope_name

    exp = tmp_path / "exp"
    _stub_fold_loading(monkeypatch, exp, range(5))

    loaded = load_models(exp, [0, 1], torch.device("cpu"))

    assert loaded.folds == (0, 1)
    assert metric_scope_name("valid", loaded.folds) == "valid_folds0-1"


def test_load_models_rejects_mismatched_tile(tmp_path):
    """fold마다 tile이 다르면 마지막 값이 조용히 이기는 대신 멈춰야 한다."""
    from komap.predict import load_models

    exp = tmp_path / "exp"
    _save_fake_fold(exp, 0, tile=128)
    _save_fake_fold(exp, 1, tile=256)

    with pytest.raises(ValueError, match="fold 간 설정이 다르다"):
        load_models(exp, [0, 1], torch.device("cpu"))

    # 같은 설정의 fold만 골라내면 통과한다.
    ld = load_models(exp, [0], torch.device("cpu"))
    assert len(ld.models) == 1 and ld.tile == 128 and ld.n_classes == 3 and ld.hint_thr is None
    # 옛 체크포인트에는 resample_scale·ridge가 없다 → 당시 동작인 1.0·False로 읽혀야 한다(하위 호환).
    assert ld.resample_scale == 1.0 and ld.ridge is False


def test_cli_defaults_match_reported_configuration():
    """k 미지정은 run에 위임하고 TTA·후광의 OOF 선택 설정은 유지해야 한다."""
    from komap import predict as Pr

    a = Pr.parse_cli(["--exp", "x"])
    assert a.tta is True and a.halo is False and a.k is None
    assert a.split == "valid" and a.device is None
    assert a.save_prob is False


def test_run_save_prob_writes_float16_without_changing_default(monkeypatch, tmp_path):
    """--save-prob만 확률 파일을 만들고 기본 실행은 기존 산출물을 유지해야 한다."""
    from komap import predict as Pr

    image_path = tmp_path / "A1_step1_sample1_image.png"
    Image.fromarray(np.zeros((4, 5), dtype=np.uint8)).save(image_path)
    loaded = Pr.LoadedModels([object()], 128, 3, None, 1.0, False)
    probability = np.full((3, 4, 5), 1 / 3, dtype=np.float32)
    monkeypatch.setattr(Pr, "pick_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(Pr, "load_models", lambda *args, **kwargs: loaded)
    monkeypatch.setattr(Pr, "resolve_k", lambda *args, **kwargs: {"A": 60, "B": 80})
    monkeypatch.setattr(Pr, "predict_ensemble_prob", lambda *args, **kwargs: probability)
    monkeypatch.setattr(
        Pr, "prob_to_mask", lambda *args, **kwargs: np.zeros((4, 5), dtype=np.int64)
    )

    Pr.run(tmp_path / "exp", [image_path], tmp_path / "with", save_prob=True)
    saved = np.load(tmp_path / "with" / "probs" / "A1_step1_sample1_image.npy")
    assert saved.dtype == np.float16
    assert saved.shape == probability.shape
    assert not (tmp_path / "with" / "score.json").exists()
    assert not (tmp_path / "with" / "A1_step1_sample1_mask.png").exists()

    Pr.run(tmp_path / "exp", [image_path], tmp_path / "without")
    assert not (tmp_path / "without" / "probs").exists()


def test_cli_can_switch_off():
    from komap import predict as Pr

    a = Pr.parse_cli(["--exp", "x", "--no-tta", "--no-halo", "--device", "cpu"])
    assert a.tta is False and a.halo is False and a.device == "cpu"


def test_cli_parses_experiments_and_k():
    from komap import predict as Pr

    a = Pr.parse_cli(["--exp", "w2_res_s15,w2_res_s20", "--k", "55,85"])
    assert Pr.parse_exps(a.exp) == ["w2_res_s15", "w2_res_s20"]
    assert a.k == {"A": 55, "B": 85}

    with pytest.raises(ValueError, match="두 정수"):
        Pr.parse_k("60")


def test_out_dir_name_encodes_non_default_settings():
    """변형 설정으로 돌린 결과가 기본 설정의 score.json을 덮어쓰면 안 된다."""
    from komap.predict import default_out_name

    assert default_out_name("valid", True, True) == "pred_valid"
    assert default_out_name("valid", False, True) == "pred_valid_notta"
    assert default_out_name("test", True, False) == "pred_test_nohalo"
    assert default_out_name("test", False, False) == "pred_test_notta_nohalo"
    assert default_out_name("valid", True, False, {"A": 55, "B": 85}) == \
        "pred_valid_nohalo_k55-85"
    assert default_out_name("valid", True, False, folds=[0]) == "pred_valid_nohalo_fold0"
    assert default_out_name("valid", True, False, folds=[0, 1]) == "pred_valid_nohalo_folds01"
    assert default_out_name("valid", True, False, folds=[0, 1, 2, 3, 4]) == \
        "pred_valid_nohalo"
    assert default_out_name("valid", True, False, {"A": 60, "B": 80}) == \
        "pred_valid_nohalo"


def test_main_routes_non_default_settings_to_distinct_directories(monkeypatch, tmp_path):
    """CLI의 k·fold 변형은 기본 산출물과 서로 다른 폴더로 전달돼야 한다."""
    from komap import predict as Pr

    seen = []
    monkeypatch.setattr(Pr, "RUNS_DIR", tmp_path)
    monkeypatch.setattr(Pr.io, "list_split", lambda split: [])

    def fake_run(*args, **kwargs):
        seen.append((args[2], args[6], args[3], kwargs["metric_scope"]))
        return {}

    monkeypatch.setattr(Pr, "run", fake_run)
    Pr.main(["--exp", "x"])
    Pr.main(["--exp", "x", "--k", "55,85"])
    Pr.main(["--exp", "x", "--folds", "0"])

    assert seen == [
        (tmp_path / "x" / "pred_valid_nohalo", {"A": 60, "B": 80}, None,
         "valid_full_fold_ensemble"),
        (tmp_path / "x" / "pred_valid_nohalo_k55-85", {"A": 55, "B": 85}, None,
         "valid_full_fold_ensemble"),
        (tmp_path / "x" / "pred_valid_nohalo_fold0", {"A": 60, "B": 80}, [0],
         "valid_folds0"),
    ]


def test_main_resolves_k_fit_before_choosing_output_directory(monkeypatch, tmp_path):
    """k_fit 55/85 생략 실행과 명시적 기본 k 실행은 서로 다른 폴더를 써야 한다."""
    import json
    from komap import predict as Pr

    exp_dir = tmp_path / "x"
    exp_dir.mkdir()
    (exp_dir / "k_fit.json").write_text(json.dumps({"k": {"A": 55, "B": 85}}))
    seen = []
    monkeypatch.setattr(Pr, "RUNS_DIR", tmp_path)
    monkeypatch.setattr(Pr.io, "list_split", lambda split: [])
    monkeypatch.setattr(
        Pr, "run", lambda *args, **kwargs: seen.append((args[2], args[6])) or {})

    Pr.main(["--exp", "x"])
    Pr.main(["--exp", "x", "--k", "60,80"])

    assert seen == [
        (exp_dir / "pred_valid_nohalo_k55-85", {"A": 55, "B": 85}),
        (exp_dir / "pred_valid_nohalo", {"A": 60, "B": 80}),
    ]


def test_main_uses_default_reclass_model_path(monkeypatch, tmp_path):
    """경로를 생략하면 선택한 runs 디렉터리의 제출용 모델을 읽어야 한다."""
    from komap import predict as Pr

    seen = []
    monkeypatch.setattr(Pr.io, "list_split", lambda split: [])
    monkeypatch.setattr(
        Pr,
        "run",
        lambda *args, **kwargs: seen.append(kwargs["reclass_model_path"]) or {},
    )

    Pr.main(["--exp", "x", "--runs-dir", str(tmp_path)])

    assert seen == [tmp_path / "reclass" / "model.pkl"]


def test_main_uses_explicit_reclass_model_path(monkeypatch, tmp_path):
    """명시한 재분류 모델은 runs 디렉터리와 무관하게 그대로 전달해야 한다."""
    from komap import predict as Pr

    selected = Path("/tmp/x.pkl")
    seen = []
    monkeypatch.setattr(Pr.io, "list_split", lambda split: [])
    monkeypatch.setattr(
        Pr,
        "run",
        lambda *args, **kwargs: seen.append(kwargs["reclass_model_path"]) or {},
    )

    Pr.main(
        [
            "--exp",
            "x",
            "--runs-dir",
            str(tmp_path),
            "--reclass-model",
            str(selected),
        ]
    )

    assert seen == [selected]


def test_resolve_k_rejects_mismatched_experiments(tmp_path):
    """서로 다른 k_fit을 묵시적으로 섞는 앙상블은 이름 결정 전에 거부한다."""
    import json
    from komap.predict import resolve_k

    exp_a, exp_b = tmp_path / "a", tmp_path / "b"
    exp_a.mkdir()
    exp_b.mkdir()
    (exp_a / "k_fit.json").write_text(json.dumps({"k": {"A": 55, "B": 85}}))
    (exp_b / "k_fit.json").write_text(json.dumps({"k": {"A": 60, "B": 80}}))

    with pytest.raises(ValueError, match="실험별 k_fit이 다르다"):
        resolve_k([exp_a, exp_b])


def test_run_records_partial_fold_scope_in_score_json(monkeypatch, tmp_path):
    """일부 fold 실행은 score.json의 metric_scope만으로도 전체 fold와 구별돼야 한다."""
    import json
    from komap import predict as Pr

    loaded = Pr.LoadedModels([object()], 128, 3, None, 1.0, False)
    monkeypatch.setattr(Pr, "pick_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(Pr, "load_models", lambda *args, **kwargs: loaded)

    r = Pr.run(tmp_path / "exp", [], tmp_path / "out", folds=[0])
    saved = json.loads((tmp_path / "out" / "score.json").read_text())

    assert r["metric_scope"] == "valid_folds0"
    assert saved["metric_scope"] == "valid_folds0"
    assert saved["folds"] == [0]


def test_predict_uses_k_fit_when_present(fake_exp, monkeypatch):
    """k_fit.json이 있으면 두께 공식이 그 k를 쓰고 score.json에 k가 기록된다."""
    import json
    from komap import predict as P
    calls = []
    real = P.thickness_rule
    monkeypatch.setattr(P, "thickness_rule", lambda p3, width, k=None: calls.append(k) or real(p3, width, k))
    (fake_exp / "exp" / "k_fit.json").write_text(json.dumps({"k": {"A": 33, "B": 77}}))
    paths = sorted((fake_exp / "data" / "images").glob("*.png"))
    r = P.run(fake_exp / "exp", paths, fake_exp / "out", folds=[0], tta=False, halo=False)
    assert calls == [{"A": 33, "B": 77}]
    assert r["k"] == {"A": 33, "B": 77}


def test_predict_uses_default_k_when_k_fit_is_absent(fake_exp, monkeypatch):
    """k_fit이 없는 옛 실험은 전역 기본 60/80을 쓰고 유효값을 score.json에 기록한다."""
    from komap.config import THICKNESS_K
    from komap import predict as P

    calls = []
    real = P.thickness_rule
    monkeypatch.setattr(
        P, "thickness_rule", lambda p3, width, k=None: calls.append(k) or real(p3, width, k))
    paths = sorted((fake_exp / "data" / "images").glob("*.png"))
    r = P.run(fake_exp / "exp", paths, fake_exp / "out", folds=[0], tta=False, halo=False)

    assert calls == [THICKNESS_K]
    assert r["k"] == THICKNESS_K


def test_load_models_returns_hint_thr_and_rejects_mismatch(tmp_path):
    """힌트 임계값이 fold마다 다르면 추론이 학습과 다른 힌트를 보게 되므로 멈춰야 한다."""
    import torch
    from komap.model import build_model
    from komap.predict import load_models
    m5 = build_model("resnet18", "none", 3, in_channels=5)
    for k, thr in [(0, [90, 180]), (1, [90, 180])]:
        (tmp_path / f"fold{k}").mkdir(parents=True)
        torch.save({"model": m5.state_dict(), "args": {"encoder": "resnet18", "arch": "unetplusplus", "n_classes": 3,
                                                        "tile": 128, "hint": True, "hint_thr": thr}}, tmp_path / f"fold{k}" / "best.pt")
    ld = load_models(tmp_path, [0, 1], device=torch.device("cpu"))
    assert len(ld.models) == 2 and ld.hint_thr == (90, 180)
    torch.save({"model": m5.state_dict(), "args": {"encoder": "resnet18", "arch": "unetplusplus", "n_classes": 3,
                                                    "tile": 128, "hint": True, "hint_thr": [91, 180]}}, tmp_path / "fold1" / "best.pt")
    with pytest.raises(ValueError, match="hint"):
        load_models(tmp_path, [0, 1], device=torch.device("cpu"))


def test_load_models_returns_resample_scale_and_rejects_mismatch(tmp_path):
    """입력 해상도가 fold마다 다르면 앙상블이 서로 다른 격자에서 추론하므로 멈춰야 한다."""
    from komap.predict import load_models

    exp = tmp_path / "exp"
    _save_fake_fold(exp, 0, tile=128, resample_scale=1.5)
    ld = load_models(exp, [0], torch.device("cpu"))
    assert ld.resample_scale == 1.5

    _save_fake_fold(exp, 1, tile=128, resample_scale=2.0)
    with pytest.raises(ValueError, match="resample"):
        load_models(exp, [0, 1], torch.device("cpu"))


def test_predict_end_to_end_with_ridge_checkpoint(tmp_path, synth_pair):
    """능선 체크포인트는 추론에서도 2채널 입력을 받아야 한다.

    ridge 플래그가 체크포인트에서 load_full까지 닿지 않으면 여기서 채널 수가 어긋나 터진다."""
    from komap import predict as Pr
    from komap.model import build_model

    gray, y4, _ = synth_pair
    (tmp_path / "data" / "images").mkdir(parents=True)
    (tmp_path / "data" / "masks").mkdir()
    ip = tmp_path / "data" / "images" / "A1_step1_sample1_image.png"
    Image.fromarray(gray).save(ip)
    Image.fromarray(to_rgb(y4)).save(tmp_path / "data" / "masks" / "A1_step1_sample1_mask.png")

    m = build_model("resnet18", "none", 3, in_channels=2)
    (tmp_path / "exp" / "fold0").mkdir(parents=True)
    torch.save({"model": m.state_dict(),
                "args": {"encoder": "resnet18", "arch": "unetplusplus", "n_classes": 3,
                         "tile": 128, "ridge": True}},
               tmp_path / "exp" / "fold0" / "best.pt")

    r = Pr.run(exp_dir=tmp_path / "exp", image_paths=[ip], out_dir=tmp_path / "out",
               folds=[0], tta=False, halo=False)
    assert r["n"] == 1
    with Image.open(tmp_path / "out" / "A1_step1_sample1_mask.png") as im:
        assert im.mode == "RGBA", "제출 마스크는 대회 정답과 같은 RGBA여야 한다"
        assert np.array(im).shape == (240, 320, 4)


def test_loaded_models_fields_are_named(tmp_path):
    """설정이 늘어도 호출부가 속성으로 읽도록 NamedTuple을 유지한다.

    위치 언패킹으로 돌아가면 필드가 하나 늘 때 순서가 조용히 어긋날 수 있다."""
    from komap.predict import LoadedModels, load_models

    assert LoadedModels._fields == (
        "models", "tile", "n_classes", "hint_thr", "resample_scale", "ridge", "folds",
        "norm", "size_correction", "head")
    exp = tmp_path / "exp"
    _save_fake_fold(exp, 0, tile=128)
    ld = load_models(exp, [0], torch.device("cpu"))
    assert isinstance(ld, LoadedModels)
    assert ld.tile == 128 and ld.ridge is False and ld.folds == (0,)


def test_score_json_records_training_settings(tmp_path, synth_pair):
    """격자를 여러 개 돌리면 score.json만 남는다 — 그 파일 하나로 설정을 알 수 있어야 한다."""
    import json
    from komap import predict as Pr
    from komap.model import build_model

    gray, y4, _ = synth_pair
    (tmp_path / "data" / "images").mkdir(parents=True)
    (tmp_path / "data" / "masks").mkdir()
    ip = tmp_path / "data" / "images" / "A1_step1_sample1_image.png"
    Image.fromarray(gray).save(ip)
    Image.fromarray(to_rgb(y4)).save(tmp_path / "data" / "masks" / "A1_step1_sample1_mask.png")

    m = build_model("resnet18", "none", 3, in_channels=2)
    (tmp_path / "exp" / "fold0").mkdir(parents=True)
    torch.save({"model": m.state_dict(),
                "args": {"encoder": "resnet18", "arch": "unetplusplus", "n_classes": 3,
                         "tile": 128, "ridge": True, "resample_scale": 1.5}},
               tmp_path / "exp" / "fold0" / "best.pt")

    r = Pr.run(exp_dir=tmp_path / "exp", image_paths=[ip], out_dir=tmp_path / "out",
               folds=[0], tta=False, halo=False)
    assert r["resample_scale"] == 1.5 and r["ridge"] is True and r["hint_thr"] is None
    saved = json.loads((tmp_path / "out" / "score.json").read_text())
    assert saved["resample_scale"] == 1.5 and saved["ridge"] is True
    assert saved["metric_scope"] == "valid_folds0"
    assert {"exps", "combiner", "k", "halo", "tta"} <= saved.keys()


def test_ensemble_restores_different_probability_sizes_before_average(monkeypatch, tmp_path):
    """서로 다른 격자의 실제 확률을 restore_full_prob로 복원한 뒤 평균해야 한다."""
    from komap import predict as Pr
    from komap.oof import restore_full_prob

    image_path = tmp_path / "A1_step1_sample1_image.png"
    Image.fromarray(np.zeros((16, 20), dtype=np.uint8)).save(image_path)
    first = np.arange(3 * 8 * 10, dtype=np.float32).reshape(3, 8, 10) / 300
    second = np.arange(3 * 12 * 15, dtype=np.float32).reshape(3, 12, 15) / 600
    loaded = [
        Pr.LoadedModels([object()], 128, 3, None, 1.5, False),
        Pr.LoadedModels([object()], 256, 3, None, 2.0, True),
    ]

    def fake_load_full(image_path, n_classes, hint_thr, resample_scale, ridge,
                       norm, size_correction):
        assert norm == "anchor3" and size_correction is True
        shape = (8, 10) if resample_scale == 1.5 else (12, 15)
        return np.zeros((1, *shape), dtype=np.float32), None, {"height": 16, "width": 20}

    def fake_predict_probs(model, xf, tile, overlap, tta, device):
        return first if xf.shape[1:] == (8, 10) else second

    combined = []
    monkeypatch.setattr(Pr, "load_full", fake_load_full)
    monkeypatch.setattr(Pr, "predict_probs", fake_predict_probs)
    monkeypatch.setattr(
        Pr,
        "prob_to_mask",
        lambda prob, image_path, n_classes, halo, k: combined.append(prob.copy()) or prob.argmax(0),
    )

    pred = Pr.predict_ensemble_image(loaded, image_path, tta=False, halo=False,
                                     device=torch.device("cpu"), k={"A": 60, "B": 80})

    assert len(combined) == 1
    expected = np.mean([
        restore_full_prob(first, 16, 20),
        restore_full_prob(second, 16, 20),
    ], axis=0)
    np.testing.assert_allclose(combined[0], expected)
    np.testing.assert_array_equal(pred, combined[0].argmax(0))


def test_single_experiment_uses_restored_probability_without_extra_average(monkeypatch, tmp_path):
    """단일 실험은 실제 복원 배열을 추가 평균 없이 그대로 후처리에 넘긴다."""
    from komap import predict as Pr
    from komap.oof import restore_full_prob

    image_path = tmp_path / "A1_step1_sample1_image.png"
    Image.fromarray(np.zeros((16, 20), dtype=np.uint8)).save(image_path)
    source = np.arange(3 * 8 * 10, dtype=np.float32).reshape(3, 8, 10) / 300
    expected = restore_full_prob(source, 16, 20)
    loaded = [Pr.LoadedModels([object()], 128, 3, None, 1.5, False)]

    monkeypatch.setattr(
        Pr, "load_full",
        lambda *args, **kwargs: (
            np.zeros((1, 8, 10), dtype=np.float32), None, {"height": 16, "width": 20}),
    )
    monkeypatch.setattr(Pr, "predict_probs", lambda *args, **kwargs: source)
    monkeypatch.setattr(Pr, "restore_full_prob", lambda *args, **kwargs: expected)
    seen = []
    monkeypatch.setattr(
        Pr, "prob_to_mask",
        lambda prob, *args, **kwargs: seen.append(prob) or prob.argmax(0),
    )

    Pr.predict_ensemble_image(loaded, image_path, tta=False, halo=False,
                              device=torch.device("cpu"), k={"A": 60, "B": 80})

    assert len(seen) == 1
    assert seen[0] is expected


def test_run_records_multiple_experiments_and_combiner(monkeypatch, tmp_path):
    """산출물 하나만으로 앙상블의 실험 목록과 결합 방식을 복원할 수 있어야 한다."""
    from komap import predict as Pr

    exp_a, exp_b = tmp_path / "s15", tmp_path / "s20"
    loaded = {
        exp_a: Pr.LoadedModels([object()], 128, 3, None, 1.5, False),
        exp_b: Pr.LoadedModels([object()], 256, 3, None, 2.0, True),
    }
    monkeypatch.setattr(Pr, "pick_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(Pr, "load_models", lambda exp_dir, folds, device: loaded[exp_dir])

    r = Pr.run([exp_a, exp_b], [], tmp_path / "out", tta=True, halo=False,
               k={"A": 60, "B": 80}, exp_names=["w2_res_s15", "w2_res_s20"])

    assert r["exps"] == ["w2_res_s15", "w2_res_s20"]
    assert r["combiner"] == "thickness"
    assert r["resample_scale"] == [1.5, 2.0]
    assert r["ridge"] == [False, True]


def _save_si2_fold(exp):
    from komap.model import build_model

    fold = exp / "fold0"
    fold.mkdir(parents=True)
    model = build_model("resnet18", "none", 2)
    torch.save({"model": model.state_dict(), "args": {"encoder": "resnet18", "arch": "unetplusplus",
               "n_classes": 4, "tile": 128, "head": "si2"}}, fold / "best.pt")
    return model


def test_load_models_reads_si2_head_and_builds_two_channel_model(tmp_path):
    from komap.predict import load_models

    exp = tmp_path / "exp"
    _save_si2_fold(exp)
    ld = load_models(exp, [0], torch.device("cpu"))
    assert ld.head == "si2" and ld.n_classes == 4
    assert ld.models[0](torch.zeros(1, 1, 128, 128)).shape[1] == 2
    _save_fake_fold(exp, 1, 128, n_classes=4)
    with pytest.raises(ValueError, match="head"):
        load_models(exp, [0, 1], torch.device("cpu"))


def test_load_models_defaults_head_to_full_for_old_checkpoints(tmp_path):
    from komap.predict import load_models

    exp = tmp_path / "exp"
    _save_fake_fold(exp, 0, 128, n_classes=4)
    assert load_models(exp, [0], torch.device("cpu")).head == "full"


def test_predict_full_prob_expands_si2_to_four_channels(tmp_path, synth_pair):
    from komap.predict import predict_full_prob
    from komap.model import build_model

    ip = tmp_path / "A1_step1_sample1_image.png"
    Image.fromarray(synth_pair[0]).save(ip)
    model = build_model("resnet18", "none", 2).eval()
    full = predict_full_prob([model], ip, 128, True, torch.device("cpu"), head="full")
    si2 = predict_full_prob([model], ip, 128, True, torch.device("cpu"), head="si2")
    assert full.shape == (2, 240, 320) and si2.shape == (4, 240, 320)
    assert not si2[:2].any() and np.allclose(si2[2:], full, atol=1e-6)
    assert np.allclose(si2[2] + si2[3], 1, atol=1e-3) and np.isfinite(si2).all()
    assert not np.allclose(si2[2], si2[2][:, ::-1])


def test_run_save_prob_with_si2_checkpoint_writes_four_channels(tmp_path, synth_pair):
    from komap import predict as Pr
    from komap.config import CLASS_NAMES

    ip = tmp_path / "A1_step1_sample1_image.png"
    Image.fromarray(synth_pair[0]).save(ip)
    exp = tmp_path / "exp"
    _save_si2_fold(exp)
    Pr.run(exp, [ip], tmp_path / "out", folds=[0], tta=False, halo=False, save_prob=True)
    prob = np.load(tmp_path / "out" / "probs" / f"{ip.stem}.npy")
    assert prob.shape == (4, 240, 320) and prob.dtype == np.float16
    assert prob.shape[0] == len(CLASS_NAMES)
    assert not prob[:2].any() and np.allclose(prob[2] + prob[3], 1, atol=0.01)


def test_run_learned_combiner_with_si2_prob4_end_to_end(tmp_path, synth_pair):
    import json
    from komap import io
    from komap import predict as Pr
    from komap.reclass import FEATURE_NAMES, ReclassModel

    ip = tmp_path / "A1_step1_sample1_image.png"
    Image.fromarray(synth_pair[0]).save(ip)
    main, aux = tmp_path / "main", tmp_path / "aux"
    _save_fake_fold(main, 0, 128, n_classes=3)
    _save_si2_fold(aux)
    x = np.zeros((2, len(FEATURE_NAMES)))
    x[1] = 1
    model = ReclassModel("logreg")
    model.fit(x, np.array([True, False]))
    model_path = tmp_path / "model.pkl"
    model.save(model_path)
    out = tmp_path / "learned"
    result = Pr.run(main, [ip], out, folds=[0], tta=True, halo=False,
                    combiner="learned", prob4_exp_dir=aux, prob4_exp_names=["aux"],
                    reclass_model_path=model_path)
    mask = io.load_mask(out / "A1_step1_sample1_mask.png")
    assert mask.shape == (240, 320) and set(np.unique(mask)) <= {0, 1, 2, 3}
    Pr.run(main, [ip], tmp_path / "thickness", folds=[0], tta=True, halo=False)
    thickness = io.load_mask(tmp_path / "thickness" / "A1_step1_sample1_mask.png")
    main_model = Pr.load_models(main, [0], torch.device("cpu"))
    pred3 = Pr.predict_ensemble_prob([main_model], ip, True, torch.device("cpu")).argmax(0)
    assert np.array_equal(mask[pred3 != 2], thickness[pred3 != 2])
    score = json.loads((out / "score.json").read_text())
    assert result["combiner"] == score["combiner"] == "learned"
    assert score["prob4_exps"] == ["aux"]
