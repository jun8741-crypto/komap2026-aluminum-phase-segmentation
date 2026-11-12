"""학습형 Si 덩어리 재분류기의 특징, 타깃, 폴드 격리를 검증한다."""
from pathlib import Path

import numpy as np
import pytest

from komap import combine as C
from komap.config import CLS_EUTSI, CLS_PRISI, CLS_SI3
from komap.experiments import reclass_fit as RF
from komap.experiments.reclass_fit import fold_indices
from komap.reclass import (
    FEATURE_NAMES,
    ReclassModel,
    blob_feature_table,
    feature_matrix,
    majority_target,
)


def _three_blob_inputs():
    lab = np.array(
        [
            [1, 0, 2, 2],
            [0, 0, 0, 0],
            [3, 3, 3, 0],
        ],
        dtype=np.int32,
    )
    prob3 = np.zeros((3, *lab.shape), dtype=np.float32)
    prob3[CLS_SI3] = np.where(lab > 0, 0.9, 0.1)
    prob4 = np.zeros((4, *lab.shape), dtype=np.float32)
    prob4[CLS_PRISI] = np.where(lab == 1, 0.8, 0.2)
    prob4[CLS_EUTSI] = np.where(lab == 1, 0.2, 0.8)
    return prob3, prob4, lab, np.array([2.0, 4.0, 6.0])


def test_blob_feature_table_returns_seven_features_in_declared_order():
    prob3, prob4, lab, thick = _three_blob_inputs()
    table = blob_feature_table(prob3, prob4, lab, thick, width=320)

    assert tuple(table) == FEATURE_NAMES
    assert len(table) == 7
    assert all(values.shape == (3,) for values in table.values())
    assert np.allclose(table["log_area"], np.log1p([1.0, 2.0, 3.0]))
    assert table["p_primary"][0] > table["p_primary"][1]


def test_blob_feature_table_empty_input_has_all_seven_empty_features():
    empty_lab = np.zeros((2, 3), dtype=np.int32)
    empty = blob_feature_table(
        np.zeros((3, 2, 3)),
        np.zeros((4, 2, 3)),
        empty_lab,
        np.zeros(0),
        width=320,
    )
    assert tuple(empty) == FEATURE_NAMES
    assert len(empty) == 7
    assert all(values.shape == (0,) for values in empty.values())

    model = ReclassModel("logreg").fit(
        np.vstack([np.zeros(len(FEATURE_NAMES)), np.ones(len(FEATURE_NAMES))]),
        np.array([False, True]),
    )
    empty_matrix = feature_matrix(empty)
    assert model.predict_proba(empty_matrix).shape == (0,)
    assert model.predict(empty_matrix).shape == (0,)


def test_reclass_model_rejects_v1_pickle_with_regeneration_command(tmp_path: Path):
    model = ReclassModel("logreg")
    del model.feature_version
    path = tmp_path / "reclass-v1.pkl"
    model.save(path)

    with pytest.raises(ValueError, match="reclass_fit") as caught:
        ReclassModel.load(path)

    message = str(caught.value)
    assert "저장 버전 1" in message
    assert "현재 버전 3" in message


def test_reclass_model_rejects_v2_pickle_with_regeneration_command(tmp_path: Path):
    model = ReclassModel("logreg")
    model.feature_version = 2
    path = tmp_path / "reclass-v2.pkl"
    model.save(path)

    with pytest.raises(ValueError, match="reclass_fit") as caught:
        ReclassModel.load(path)

    message = str(caught.value)
    assert "저장 버전 2" in message
    assert "현재 버전 3" in message


def test_reclass_model_rejects_same_version_with_reordered_features(tmp_path: Path):
    model = ReclassModel("logreg")
    names = list(model.feature_names)
    names[0], names[1] = names[1], names[0]
    model.feature_names = tuple(names)
    path = tmp_path / "reclass-reordered.pkl"
    model.save(path)

    with pytest.raises(ValueError, match="특징 이름 또는 순서") as caught:
        ReclassModel.load(path)

    message = str(caught.value)
    assert "인덱스 0" in message
    assert "인덱스 1" in message
    assert "reclass_fit" in message


def test_reclass_result_metadata_and_paths_are_versioned(tmp_path: Path):
    metadata = RF.result_metadata()
    json_path, markdown_path = RF.result_output_paths(tmp_path)
    default_json_path, default_markdown_path = RF.result_output_paths()

    assert metadata == {
        "feature_version": 3,
        "feature_names": list(FEATURE_NAMES),
    }
    assert json_path.name == "2026-09-w2-reclass-fit-v3.json"
    assert markdown_path.name == "2026-09-w2-reclass-fit-v3.md"
    assert default_json_path.name == "2026-09-w2-reclass-fit-v3.json"
    assert default_markdown_path.name == "2026-09-w2-reclass-fit-v3.md"


def test_reclass_result_paths_separate_nondefault_prob4_experiment(tmp_path: Path):
    json_path, markdown_path = RF.result_output_paths(
        tmp_path, prob4_exp="w3_prob4_s15"
    )

    assert json_path.name == "2026-09-w2-reclass-fit-v3-w3_prob4_s15.json"
    assert markdown_path.name == "2026-09-w2-reclass-fit-v3-w3_prob4_s15.md"


def test_reclass_model_path_follows_runs_dir(tmp_path: Path):
    """재분류 모델은 선택한 runs 디렉터리 아래에만 저장돼야 한다."""
    from komap.config import RECLASS_MODEL_PATH, RUNS_DIR

    assert RF.model_output_path(RUNS_DIR) == RECLASS_MODEL_PATH
    assert RF.model_output_path(tmp_path) == tmp_path / "reclass" / "model.pkl"


def test_reclass_model_path_separates_nondefault_prob4_experiment(tmp_path: Path):
    assert RF.model_output_path(tmp_path, prob4_exp="w3_prob4_s15") == (
        tmp_path / "reclass" / "w3_prob4_s15" / "model.pkl"
    )


@pytest.mark.parametrize(
    "prob4_exp",
    ["", ".", "../x", "a/b", "a\\b", " x", "w2_res_s15"],
)
def test_reclass_rejects_unsafe_or_three_class_prob4_experiment(prob4_exp: str):
    with pytest.raises(ValueError):
        RF.result_output_paths(prob4_exp=prob4_exp)


def test_reclass_model_path_rejects_alias_to_default_model(tmp_path: Path):
    with pytest.raises(ValueError):
        RF.model_output_path(tmp_path, prob4_exp=".")


def test_majority_target_excludes_no_truth_si_and_ties():
    lab = np.array([[1, 1, 2, 2, 3, 3]], dtype=np.int32)
    target = np.array(
        [[CLS_PRISI, CLS_PRISI, 0, 1, CLS_PRISI, CLS_EUTSI]], dtype=np.int64
    )

    y, usable = majority_target(lab, target)

    assert np.array_equal(y, [True, False, False])
    assert np.array_equal(usable, [True, False, False])


def test_fold_indices_never_train_on_validation_fold_blobs():
    image_folds = np.array([0, 0, 1, 2])
    component_counts = np.array([2, 3, 4, 1])
    component_folds = np.repeat(image_folds, component_counts)

    for fold in np.unique(image_folds):
        train_idx, valid_idx = fold_indices(component_folds, int(fold))
        assert np.intersect1d(train_idx, valid_idx).size == 0
        assert np.all(component_folds[train_idx] != fold)
        assert np.all(component_folds[valid_idx] == fold)


def test_cross_validate_never_fits_or_selects_threshold_on_outer_validation(
    monkeypatch,
):
    """이 테스트는 외부 fold 모델을 전체 데이터로 학습하는 변이를 반드시 잡아야 한다."""
    records = []
    for fold in range(3):
        records.append(
            {
                "name": f"image_{fold}.png",
                "fold": fold,
                "x": np.full((1, len(FEATURE_NAMES)), float(fold)),
                "y": np.array([fold % 2 == 0]),
                "usable": np.array([True]),
                "fixed_confusion": np.eye(4, dtype=np.int64),
                "truth_counts": np.ones((1, 4), dtype=np.int64),
            }
        )

    class SpyModel:
        def predict_proba(self, x):
            return np.full(x.shape[0], 0.5)

    fit_calls = []
    threshold_calls = []
    summarized_calls = []
    active_outer_fold = [None]
    real_select = RF._select_threshold
    real_summarize = RF._summarize_assignments

    def spy_fit(records_arg, indices, model_name):
        fit_calls.append((active_outer_fold[0], tuple(indices)))
        return SpyModel()

    def spy_select(records_arg, outer_train, outer_fold, model_name, outer_model):
        threshold_calls.append((outer_fold, tuple(outer_train)))
        # _cross_validate의 직전 _fit 호출이 바로 이 외부 fold 모델 학습이어야 한다.
        assert fit_calls[-1][1] == tuple(outer_train)
        active_outer_fold[0] = outer_fold
        try:
            return real_select(
                records_arg, outer_train, outer_fold, model_name, outer_model
            )
        finally:
            active_outer_fold[0] = None

    def spy_summarize(records_arg, indices, assignments):
        if active_outer_fold[0] is not None:
            summarized_calls.append((active_outer_fold[0], tuple(indices)))
        return real_summarize(records_arg, indices, assignments)

    monkeypatch.setattr(RF, "_fit", spy_fit)
    monkeypatch.setattr(RF, "_select_threshold", spy_select)
    monkeypatch.setattr(RF, "_summarize_assignments", spy_summarize)
    RF._cross_validate(records, "logreg")

    assert len(threshold_calls) == 3
    for validation_fold, threshold_indices in threshold_calls:
        assert all(
            records[index]["fold"] != validation_fold for index in threshold_indices
        )
    for validation_fold, trained in fit_calls:
        if validation_fold is not None:
            assert all(records[index]["fold"] != validation_fold for index in trained)
    for validation_fold, summarized in summarized_calls:
        assert all(records[index]["fold"] != validation_fold for index in summarized)


def test_metric_summary_records_absent_one_counts_and_image_values():
    """부재 클래스를 1로 보는 점수와 클래스별 평균 대상 이미지 수를 함께 남긴다."""
    record = {
        "name": "sample_image.png",
        "fixed_confusion": np.array(
            [
                [1, 0, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ],
            dtype=np.int64,
        ),
        "truth_counts": np.zeros((0, 4), dtype=np.int64),
    }

    summary = RF._summarize_assignments([record], [0], {0: np.zeros(0, dtype=bool)})

    assert summary["miou"] == pytest.approx(100 / 3)
    assert summary["miou_one"] == pytest.approx(50.0)
    assert summary["per_class_n"] == {"0": 1, "1": 1, "2": 1, "3": 0}
    assert summary["per_image"]["sample_image.png"] == pytest.approx(100 / 3)


def test_comparison_records_image_gain_share_and_common_class_denominator():
    current = {
        "per_image": {"a_image.png": 40.0, "b_image.png": 60.0},
        "per_class_by_image": {
            "a_image.png": {"2": 0.0},
            "b_image.png": {"2": 80.0},
        },
    }
    learned = {
        "per_image": {"a_image.png": 50.0, "b_image.png": 55.0},
        "per_class_by_image": {
            "a_image.png": {"2": None},
            "b_image.png": {"2": 90.0},
        },
    }

    comparison = RF._compare_summaries(current, learned)

    assert comparison["images"][0] == {
        "image": "a",
        "file_name": "a_image.png",
        "current_miou": 40.0,
        "learned_miou": 50.0,
        "difference": 10.0,
        "gain_share_pct": 200.0,
    }
    assert comparison["counts"] == {"positive": 1, "negative": 1, "unchanged": 0}
    assert comparison["common_per_class"]["2"] == {
        "n_images": 1,
        "current": 80.0,
        "learned": 90.0,
        "difference": 10.0,
    }


class _ThresholdModel:
    threshold = 0.5

    def predict(self, features, threshold=None):
        selected = self.threshold if threshold is None else threshold
        return features[:, FEATURE_NAMES.index("p_primary")] > selected


def test_learned_combiner_dispatches_and_requires_four_channel_prob(synth_pair):
    _, y4, _ = synth_pair
    prob3 = np.zeros((3, *y4.shape), dtype=np.float32)
    prob3[0] = 0.05
    prob3[1] = 0.05
    prob3[CLS_SI3] = np.where((y4 == CLS_PRISI) | (y4 == CLS_EUTSI), 0.9, 0.0)
    prob4 = np.full((4, *y4.shape), 0.05, dtype=np.float32)
    prob4[CLS_PRISI] = np.where(y4 == CLS_PRISI, 0.9, 0.1)
    prob4[CLS_EUTSI] = np.where(y4 == CLS_EUTSI, 0.9, 0.1)

    out = C.combine(
        "learned",
        prob3,
        width=y4.shape[1],
        model=_ThresholdModel(),
        prob4=prob4,
    )

    assert out.shape == y4.shape
    assert out.dtype == np.int64
    with pytest.raises(ValueError, match="4채널"):
        C.combine(
            "learned",
            prob3,
            width=y4.shape[1],
            model=_ThresholdModel(),
        )


def test_reclass_model_threshold_is_monotonic_and_persists(tmp_path: Path):
    x7 = np.array(
        [
            [0.1, 1.0, 2.0, 0.1, 0.8, 0.2, 0.5],
            [0.3, 1.2, 2.2, 0.3, 0.8, 0.3, 0.4],
            [0.7, 2.0, 1.2, 0.7, 0.9, 0.5, 0.2],
            [0.9, 2.2, 1.0, 0.9, 0.9, 0.6, 0.1],
        ]
    )
    x = x7
    y = np.array([False, False, True, True])
    model = ReclassModel("logreg", threshold=0.65).fit(x, y)

    low = model.predict(x, threshold=0.25)
    high = model.predict(x, threshold=0.75)
    assert high.sum() <= low.sum()
    assert np.all(~high | low)

    path = tmp_path / "reclass.joblib"
    model.save(path)
    loaded = ReclassModel.load(path)
    assert np.allclose(loaded.predict_proba(x), model.predict_proba(x))
    assert loaded.threshold == 0.65
    assert np.array_equal(loaded.predict(x), loaded.predict(x, threshold=0.65))


def test_valid_probability_error_names_paths_and_generation_command(tmp_path: Path):
    image_path = tmp_path / "A1_step1_sample1_image.png"
    image_path.touch()

    with pytest.raises(FileNotFoundError) as caught:
        RF._prepare_tables(tmp_path / "runs", [image_path], split="valid")

    message = str(caught.value)
    assert "pred_valid/probs" in message
    assert "--split valid --save-prob" in message
    # --out이 빠지면 안내한 명령이 오류를 해소하지 못한다. 후광 보정 기본값이 꺼짐이라
    # 출력 폴더가 pred_valid_nohalo가 되고, 확률을 만들어도 여기서 찾는 경로에 놓이지 않는다.
    # 실제로 그 순환을 한 번 겪었으므로 안내 문구를 테스트로 고정한다.
    for name in ("w2_res_s15", "w2_res_s20", "w2_combine4_direct4"):
        assert f"--out runs/{name}/pred_valid" in message


def test_valid_probability_error_uses_selected_prob4_experiment(tmp_path: Path):
    image_path = tmp_path / "A1_step1_sample1_image.png"
    image_path.touch()

    with pytest.raises(FileNotFoundError) as caught:
        RF._prepare_tables(
            tmp_path / "runs",
            [image_path],
            split="valid",
            prob4_exp="w3_prob4_s15",
        )

    message = str(caught.value)
    assert "--out runs/w3_prob4_s15/pred_valid" in message
    assert "w2_combine4_direct4" not in message


def _render_missing_valid_result(prob4_exp: str):
    """valid 확률 안내문 테스트에 필요한 최소 평가 결과를 만든다."""
    metrics = {
        "miou": 1.0,
        "miou_one": 1.0,
        "per_class": {str(cls): 1.0 for cls in range(4)},
        "per_class_n": {str(cls): 1 for cls in range(4)},
    }
    train = {
        "sources": [*RF.SOURCE_DIRS, prob4_exp],
        "final_model": {"model": "hgb", "threshold": 0.5},
    }
    return {
        "current": metrics,
        "majority_all": metrics,
        "train": train,
        "valid": {"status": "probabilities_missing_not_evaluated"},
    }


def test_render_missing_valid_uses_selected_prob4_experiment_in_reclass_command():
    rendered = RF.render(_render_missing_valid_result("w3_prob4_s15"))

    command = next(
        line
        for line in rendered.splitlines()
        if "komap.experiments.reclass_fit" in line
    )
    assert "--split valid --prob4-exp w3_prob4_s15" in command


def test_render_missing_valid_keeps_default_reclass_command_unchanged():
    rendered = RF.render(_render_missing_valid_result(RF.PROB4_DIR))

    command = next(
        line
        for line in rendered.splitlines()
        if "komap.experiments.reclass_fit" in line
    )
    assert command == f"{RF.VENV_PYTHON} -m komap.experiments.reclass_fit --split valid"


def test_valid_probs_complete_checks_missing_corrupt_and_channel_count(
    monkeypatch, tmp_path: Path
):
    runs_dir = tmp_path / "runs"
    prob_dir = runs_dir / "w3_prob4_s15" / "pred_valid" / "probs"
    prob_dir.mkdir(parents=True)
    image_paths = [tmp_path / "a_image.png", tmp_path / "b_image.png"]
    mask_paths = {path: tmp_path / f"{path.stem}.mask" for path in image_paths}
    for mask_path in mask_paths.values():
        mask_path.touch()
    monkeypatch.setattr(RF.io, "list_split", lambda split: image_paths)
    monkeypatch.setattr(RF.io, "mask_path_for", lambda path: mask_paths[path])

    for image_path in image_paths:
        np.save(prob_dir / f"{image_path.stem}.npy", np.zeros((4, 2, 3)))
    assert RF.valid_probs_complete(runs_dir, "w3_prob4_s15") is True

    (prob_dir / "b_image.npy").unlink()
    assert RF.valid_probs_complete(runs_dir, "w3_prob4_s15") is False

    (prob_dir / "b_image.npy").write_bytes(b"\x93NUMPY")
    assert RF.valid_probs_complete(runs_dir, "w3_prob4_s15") is False

    np.save(prob_dir / "b_image.npy", np.zeros((3, 2, 3)))
    assert RF.valid_probs_complete(runs_dir, "w3_prob4_s15") is False


def _minimal_main_result():
    """CLI 전달 테스트에서 출력 직렬화에 필요한 최소 결과를 만든다."""
    metric = {"miou": 1.0}
    return {
        "n_images": 1,
        "current": metric,
        "majority_all": metric,
        "learned": {"model": "hgb", "miou": 1.0, "miou_one": 1.0},
        "valid": {"status": "probabilities_missing_not_evaluated"},
    }


def test_reclass_main_passes_selected_prob4_experiment(monkeypatch):
    seen = []
    monkeypatch.setattr(
        RF,
        "measure",
        lambda **kwargs: seen.append(kwargs["prob4_exp"]) or _minimal_main_result(),
    )

    RF.main(["--prob4-exp", "w3_prob4_s15", "--limit", "1"])

    assert seen == ["w3_prob4_s15"]


def test_reclass_main_passes_default_prob4_experiment(monkeypatch):
    seen = []
    monkeypatch.setattr(
        RF,
        "measure",
        lambda **kwargs: seen.append(kwargs["prob4_exp"]) or _minimal_main_result(),
    )

    RF.main(["--limit", "1"])

    assert seen == ["w2_combine4_direct4"]
