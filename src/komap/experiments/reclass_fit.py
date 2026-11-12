"""OOF Si 덩어리 특징으로 재분류기를 학습·평가한다.

실행: ``python -m komap.experiments.reclass_fit``
"""
import argparse
import json
from pathlib import Path

import numpy as np

from komap import io
from komap.combine import pred3_from_prob
from komap.config import (
    CLASS_NAMES,
    CLS_SI3,
    RECLASS_MODEL_PATH,
    RECLASS_THRESHOLDS,
    ROOT,
    RUNS_DIR,
    THICKNESS_K,
    VENV_PYTHON,
)
from komap.experiments.cheap_wins import _check_folds
from komap.oof import list_oof, load_full_prob
from komap.postprocess import component_thickness
from komap.preprocess import group_of
from komap.reclass import (
    FEATURE_NAMES,
    FEATURE_VERSION,
    ReclassModel,
    blob_feature_table,
    feature_matrix,
    majority_target,
)

DOCS_DIR = ROOT / "docs" / "experiments"
STEM = "2026-09-w2-reclass-fit"
OUTPUT_STEM = f"{STEM}-v{FEATURE_VERSION}"
SOURCE_DIRS = ("w2_res_s15", "w2_res_s20")
PROB4_DIR = "w2_combine4_direct4"
MODEL_NAMES = ("logreg", "hgb")
CLASS_LABELS = ("Al 기지", "Al₃Ni", "Primary Si", "Eutectic Si")


def result_metadata():
    """평가 파일만으로 특징 스키마를 식별할 수 있는 메타데이터를 만든다."""
    return {
        "feature_version": FEATURE_VERSION,
        "feature_names": list(FEATURE_NAMES),
    }


def _validate_prob4_exp(prob4_exp):
    """4클래스 실험명이 안전하고 3클래스 원천과 겹치지 않는지 확인한다."""
    if (
        not prob4_exp
        or prob4_exp.strip() != prob4_exp
        or prob4_exp == "."
        or ".." in prob4_exp
        or "/" in prob4_exp
        or "\\" in prob4_exp
    ):
        raise ValueError(
            "4클래스 실험명은 단일 경로 구성요소여야 하며 공백, '.', '..', '/', '\\'를 허용하지 않는다"
        )
    if prob4_exp in SOURCE_DIRS:
        raise ValueError("3클래스 실험은 4클래스 확률 원천으로 사용할 수 없다")
    return prob4_exp


def result_output_paths(docs_dir=DOCS_DIR, prob4_exp=PROB4_DIR):
    """특징 버전별 평가 결과 경로를 반환한다."""
    docs_dir = Path(docs_dir)
    _validate_prob4_exp(prob4_exp)
    stem = OUTPUT_STEM if prob4_exp == PROB4_DIR else f"{OUTPUT_STEM}-{prob4_exp}"
    return docs_dir / f"{stem}.json", docs_dir / f"{stem}.md"


def model_output_path(runs_dir=RUNS_DIR, prob4_exp=PROB4_DIR):
    """선택한 runs 디렉터리에 대응하는 재분류 모델 경로를 반환한다."""
    _validate_prob4_exp(prob4_exp)
    default_path = Path(runs_dir) / "reclass" / "model.pkl"
    if prob4_exp != PROB4_DIR:
        selected_path = Path(runs_dir) / "reclass" / prob4_exp / "model.pkl"
        if selected_path.resolve() == default_path.resolve():
            raise ValueError("비기본 4클래스 실험 모델 경로가 기본 모델 경로와 같다")
        return selected_path
    relative_path = RECLASS_MODEL_PATH.relative_to(RUNS_DIR)
    return Path(runs_dir) / relative_path


def valid_probs_complete(runs_dir, prob4_exp):
    """마스크가 있는 valid 이미지의 4채널 확률이 모두 정상인지 확인한다."""
    _validate_prob4_exp(prob4_exp)
    paths = [
        path
        for path in io.list_split("valid")
        if io.mask_path_for(path).exists()
    ]
    if not paths:
        return False
    prob_dir = Path(runs_dir) / prob4_exp / "pred_valid" / "probs"
    for image_path in paths:
        prob_path = prob_dir / f"{image_path.stem}.npy"
        try:
            probability = np.load(prob_path, allow_pickle=False)
        except (OSError, ValueError, EOFError):
            return False
        if probability.ndim < 1 or probability.shape[0] != len(CLASS_NAMES):
            return False
    return True


def fold_indices(component_folds, validation_fold):
    """검증 fold와 겹치지 않는 덩어리 학습·검증 인덱스를 만든다."""
    component_folds = np.asarray(component_folds)
    valid = np.flatnonzero(component_folds == validation_fold)
    train = np.flatnonzero(component_folds != validation_fold)
    return train, valid


def _prepare_tables(runs_dir, paths, split="train", prob4_exp=PROB4_DIR):
    _validate_prob4_exp(prob4_exp)
    directories = (*SOURCE_DIRS, prob4_exp)
    if split == "train":
        tables = {name: list_oof(Path(runs_dir) / name) for name in directories}
        _check_folds(tables)
    elif split == "valid":
        tables = {
            name: {
                path.name: (
                    None,
                    Path(runs_dir) / name / "pred_valid" / "probs" / f"{path.stem}.npy",
                )
                for path in paths
            }
            for name in directories
        }
        missing = [
            str(prob_path)
            for table in tables.values()
            for _, prob_path in table.values()
            if not prob_path.exists()
        ]
        if missing:
            commands = "\n".join(
                f"  {VENV_PYTHON} -m komap.predict --exp {name} --split valid --save-prob"
                f" --out runs/{name}/pred_valid"
                for name in directories
            )
            raise FileNotFoundError(
                f"valid 확률 파일 {len(missing)}개가 없다. 첫 누락 경로: {missing[0]}\n"
                f"다음 명령으로 세 실험의 확률을 생성하라:\n{commands}"
            )
    else:
        raise ValueError(f"지원하지 않는 split: {split}")
    required = {path.name for path in paths}
    for source, table in tables.items():
        missing = sorted(required - set(table))
        if missing:
            raise KeyError(f"{source}: 확률이 없는 {split} 이미지 {len(missing)}장")
    return tables


def _fixed_confusion(pred3, target):
    fixed = pred3 != CLS_SI3
    encoded = pred3[fixed] * len(CLASS_NAMES) + target[fixed]
    return np.bincount(encoded, minlength=len(CLASS_NAMES) ** 2).reshape(4, 4)


def _blob_truth_counts(lab, target, n_components):
    return np.column_stack(
        [
            np.bincount(lab[target == cls], minlength=n_components + 1)[1:]
            for cls in range(len(CLASS_NAMES))
        ]
    )


def _load_records(
    runs_dir, paths, split="train", prob4_exp=PROB4_DIR, prob3_transform=None, k=None
):
    """확률은 한 장씩 해제하고 작은 덩어리 표와 혼동행렬만 남긴다."""
    tables = _prepare_tables(runs_dir, paths, split=split, prob4_exp=prob4_exp)
    records = []
    k = THICKNESS_K if k is None else k
    for image_path in paths:
        target = io.load_mask(io.mask_path_for(image_path)).astype(np.int64)
        height, width = target.shape
        prob3 = load_full_prob(tables[SOURCE_DIRS[0]][image_path.name][1], height, width)
        prob3 += load_full_prob(tables[SOURCE_DIRS[1]][image_path.name][1], height, width)
        prob3 *= 0.5
        if prob3_transform is not None:
            transformed = prob3_transform(prob3, image_path)
            if not isinstance(transformed, np.ndarray) or transformed.shape != prob3.shape:
                raise ValueError("prob3_transform 반환값은 원래 prob3와 같은 형상의 ndarray여야 한다")
            prob3 = transformed.astype(np.float32, copy=False)
        prob4 = load_full_prob(tables[prob4_exp][image_path.name][1], height, width)

        pred3 = pred3_from_prob(prob3, None)
        lab, thick = component_thickness(pred3 == CLS_SI3)
        table = blob_feature_table(prob3, prob4, lab, thick, width, k=k)
        threshold = width / k[group_of(width)]
        record = {
            "name": image_path.name,
            "fold": tables[SOURCE_DIRS[0]][image_path.name][0],
            "x": feature_matrix(table),
            "current": thick > threshold,
            "fixed_confusion": _fixed_confusion(pred3, target),
            "truth_counts": _blob_truth_counts(lab, target, thick.size),
        }
        if split == "train":
            record["y"], record["usable"] = majority_target(lab, target)
        records.append(record)
    return records


def _confusion(record, is_primary):
    confusion = record["fixed_confusion"].copy()
    counts = record["truth_counts"]
    confusion[2] += counts[is_primary].sum(axis=0)
    confusion[3] += counts[~is_primary].sum(axis=0)
    return confusion


def _metrics_from_confusion(confusion):
    intersection = np.diag(confusion).astype(np.float64)
    union = confusion.sum(axis=0) + confusion.sum(axis=1) - intersection
    valid = union > 0
    per_class = np.full(len(CLASS_NAMES), np.nan)
    per_class[valid] = intersection[valid] / union[valid]
    miou_skip = float(np.mean(per_class[valid]) * 100.0)
    miou_one = float(np.mean(np.where(valid, per_class, 1.0)) * 100.0)
    return miou_skip, miou_one, per_class


def _summarize_assignments(records, indices, assignments):
    per_image = []
    per_image_one = []
    image_names = []
    per_class = {cls: [] for cls in range(len(CLASS_NAMES))}
    per_class_by_image = {}
    for index in indices:
        miou, miou_one, class_values = _metrics_from_confusion(
            _confusion(records[index], assignments[index])
        )
        name = records[index]["name"]
        image_names.append(name)
        per_image.append(miou)
        per_image_one.append(miou_one)
        per_class_by_image[name] = {}
        for cls, value in enumerate(class_values):
            if not np.isnan(value):
                per_class[cls].append(float(value))
                per_class_by_image[name][str(cls)] = float(value) * 100.0
            else:
                per_class_by_image[name][str(cls)] = None
    return {
        "miou": float(np.mean(per_image)),
        "miou_one": float(np.mean(per_image_one)),
        "per_class": {
            str(cls): (float(np.mean(values)) * 100.0 if values else None)
            for cls, values in per_class.items()
        },
        "per_class_n": {str(cls): len(values) for cls, values in per_class.items()},
        "per_image": dict(zip(image_names, per_image, strict=True)),
        "per_image_one": dict(zip(image_names, per_image_one, strict=True)),
        "per_class_by_image": per_class_by_image,
    }


def _compare_summaries(current, learned):
    """두 배정의 이미지별 이득과 동일 분모 클래스 평균을 계산한다."""
    names = list(current["per_image"])
    differences = {
        name: learned["per_image"][name] - current["per_image"][name]
        for name in names
    }
    # 분모는 하락을 차감한 순증가 합계다. 개선된 이미지의 양수 이득만 더한 값과 다르므로
    # (valid 기준 28.494 대 34.061) 어느 쪽인지 밝히지 않으면 같은 데이터에서 두 비율이 나온다.
    # 실제로 리뷰 과정에서 A71의 기여율이 46.45%와 38.85%로 엇갈렸다.
    total_gain = float(sum(differences.values()))
    images = []
    for name in names:
        difference = float(differences[name])
        images.append(
            {
                "image": name.replace("_image.png", ""),
                "file_name": name,
                "current_miou": float(current["per_image"][name]),
                "learned_miou": float(learned["per_image"][name]),
                "difference": difference,
                "gain_share_pct": (
                    100.0 * difference / total_gain if total_gain != 0.0 else None
                ),
            }
        )

    tolerance = 1e-12
    counts = {
        "positive": sum(value > tolerance for value in differences.values()),
        "negative": sum(value < -tolerance for value in differences.values()),
        "unchanged": sum(abs(value) <= tolerance for value in differences.values()),
    }
    ordered = sorted(images, key=lambda row: row["difference"], reverse=True)
    excluding_top = (
        float(np.mean([row["difference"] for row in ordered[1:]]))
        if len(ordered) > 1
        else None
    )

    common_per_class = {}
    for cls in range(len(CLASS_NAMES)):
        key = str(cls)
        pairs = [
                (
                    current["per_class_by_image"][name].get(key),
                    learned["per_class_by_image"][name].get(key),
                )
                for name in names
                if current["per_class_by_image"][name].get(key) is not None
                and learned["per_class_by_image"][name].get(key) is not None
        ]
        current_mean = float(np.mean([pair[0] for pair in pairs])) if pairs else None
        learned_mean = float(np.mean([pair[1] for pair in pairs])) if pairs else None
        common_per_class[key] = {
            "n_images": len(pairs),
            "current": current_mean,
            "learned": learned_mean,
            "difference": (
                learned_mean - current_mean
                if current_mean is not None and learned_mean is not None
                else None
            ),
        }
    return {
        "mean_difference": float(np.mean(list(differences.values()))),
        "counts": counts,
        "top_gain_image": ordered[0] if ordered else None,
        "mean_difference_excluding_top_gain_image": excluding_top,
        "images": images,
        "common_per_class": common_per_class,
    }


def _training_arrays(records, indices):
    xs = [records[index]["x"][records[index]["usable"]] for index in indices]
    ys = [records[index]["y"][records[index]["usable"]] for index in indices]
    nonempty = [index for index, values in enumerate(ys) if values.size]
    if not nonempty:
        raise ValueError("학습 가능한 명확한 다수결 Si 덩어리가 없다")
    return np.concatenate([xs[index] for index in nonempty]), np.concatenate(
        [ys[index] for index in nonempty]
    )


def _fit(records, indices, model_name):
    x, y = _training_arrays(records, indices)
    return ReclassModel(model_name).fit(x, y)


def _select_threshold(records, outer_train, outer_fold, model_name, outer_model):
    """외부 검증 fold를 제외한 내부 OOF 점수로 결정 임계값을 고른다."""
    scores = {}
    inner_folds = sorted({records[index]["fold"] for index in outer_train})
    for inner_fold in inner_folds:
        inner_valid = [index for index in outer_train if records[index]["fold"] == inner_fold]
        inner_train = [index for index in outer_train if records[index]["fold"] != inner_fold]
        try:
            model = _fit(records, inner_train, model_name)
        except ValueError:
            model = outer_model
        for index in inner_valid:
            scores[index] = model.predict_proba(records[index]["x"])

    curve = []
    for threshold in RECLASS_THRESHOLDS:
        assignments = {index: scores[index] > threshold for index in outer_train}
        metrics = _summarize_assignments(records, outer_train, assignments)
        curve.append({"threshold": threshold, "miou": metrics["miou"]})
    selected = max(curve, key=lambda row: (row["miou"], -abs(row["threshold"] - 0.5)))
    return selected["threshold"], {
        "validation_fold": outer_fold,
        "selected_threshold": selected["threshold"],
        "threshold_curve": curve,
    }


def _select_final_threshold(records, model_name):
    """train 전체의 이미지-fold OOF 예측으로 최종 저장 모델 임계값을 정한다."""
    indices = list(range(len(records)))
    folds = sorted({record["fold"] for record in records})
    scores = {}
    for fold in folds:
        train = [index for index in indices if records[index]["fold"] != fold]
        valid = [index for index in indices if records[index]["fold"] == fold]
        model = _fit(records, train, model_name)
        for index in valid:
            scores[index] = model.predict_proba(records[index]["x"])
    curve = []
    for threshold in RECLASS_THRESHOLDS:
        assignments = {index: scores[index] > threshold for index in indices}
        metrics = _summarize_assignments(records, indices, assignments)
        curve.append({"threshold": threshold, "miou": metrics["miou"]})
    selected = max(curve, key=lambda row: (row["miou"], -abs(row["threshold"] - 0.5)))
    return {
        "method": "train_image_fold_oof_miou_sweep",
        "metric_scope": "train_70_segmentation_oof_features",
        "selected_threshold": selected["threshold"],
        "threshold_curve": curve,
        "note": "fold별 평가 임계값의 평균이 아니라 train 전체 이미지-fold OOF 예측으로 선택",
    }


def _cross_validate(records, model_name, return_assignments=False):
    folds = sorted({record["fold"] for record in records})
    assignments = {}
    fold_results = []
    all_indices = list(range(len(records)))
    record_folds = np.asarray([record["fold"] for record in records])
    for fold in folds:
        train_array, valid_array = fold_indices(record_folds, fold)
        outer_train = train_array.tolist()
        outer_valid = valid_array.tolist()
        model = _fit(records, outer_train, model_name)
        threshold, fold_result = _select_threshold(
            records, outer_train, fold, model_name, model
        )
        for index in outer_valid:
            assignments[index] = model.predict_proba(records[index]["x"]) > threshold
        fold_results.append(fold_result)
    metrics = _summarize_assignments(records, all_indices, assignments)
    result = {**metrics, "folds": fold_results}
    if return_assignments:
        return result, assignments
    return result


def oof_image_confusions(
    runs_dir=RUNS_DIR,
    limit=0,
    model_name=None,
    prob4_exp=PROB4_DIR,
):
    """재분류기 외부 OOF 배정으로 만든 이미지별 4×4 혼동행렬을 반환한다."""
    _validate_prob4_exp(prob4_exp)
    if limit < 0:
        raise ValueError("--limit은 0 이상이어야 한다")
    paths = list(io.list_split("train"))
    if limit:
        paths = paths[:limit]
    if not paths:
        raise FileNotFoundError("train 분할에 이미지가 없다")
    records = _load_records(
        runs_dir, paths, split="train", prob4_exp=prob4_exp
    )

    if model_name is None:
        evaluated = {
            name: _cross_validate(records, name, return_assignments=True)
            for name in MODEL_NAMES
        }
        model_name = max(MODEL_NAMES, key=lambda name: evaluated[name][0]["miou"])
        _, assignments = evaluated[model_name]
    else:
        if model_name not in MODEL_NAMES:
            raise ValueError(f"알 수 없는 재분류 모델: {model_name}")
        _, assignments = _cross_validate(
            records, model_name, return_assignments=True
        )
    return [
        {
            "name": record["name"],
            "fold": record["fold"],
            "confusion": _confusion(record, assignments[index]),
        }
        for index, record in enumerate(records)
    ]


def measure(
    runs_dir=RUNS_DIR,
    limit=0,
    model_names=MODEL_NAMES,
    split="train",
    prob4_exp=PROB4_DIR,
):
    """저장된 OOF 확률로 기준선, 다수결 참조, 학습형 모델을 함께 평가한다."""
    _validate_prob4_exp(prob4_exp)
    if limit < 0:
        raise ValueError("--limit은 0 이상이어야 한다")
    train_paths = [path for path in io.list_split("train") if io.mask_path_for(path).exists()]
    if limit:
        train_paths = train_paths[:limit]
    if not train_paths:
        raise FileNotFoundError("train 분할에 정답 마스크가 없다")
    valid_records = None
    valid_paths = [path for path in io.list_split("valid") if io.mask_path_for(path).exists()]
    if limit:
        valid_paths = valid_paths[:limit]
    if split == "valid":
        if not valid_paths:
            raise FileNotFoundError("valid 분할에 정답 마스크가 없다")
        # 확률 누락은 train 학습과 모델 저장 전에 확인해 실패 실행이 산출물을 바꾸지 않게 한다.
        valid_records = _load_records(
            runs_dir, valid_paths, split="valid", prob4_exp=prob4_exp
        )
    elif split == "train" and valid_paths:
        # 기본 train 재실행이 이미 측정된 valid 결과를 pending 상태로 되돌리지 않게 한다.
        # 확률이 아직 없다면 기존 스모크 동작대로 train만 평가한다.
        try:
            valid_records = _load_records(
                runs_dir, valid_paths, split="valid", prob4_exp=prob4_exp
            )
        except FileNotFoundError:
            valid_records = None
    elif split != "train":
        raise ValueError(f"지원하지 않는 split: {split}")

    records = _load_records(
        runs_dir, train_paths, split="train", prob4_exp=prob4_exp
    )
    indices = list(range(len(records)))
    current = {index: record["current"] for index, record in enumerate(records)}
    majority = {index: record["y"] for index, record in enumerate(records)}
    train_result = {
        "split": "train",
        "metric_scope": "train_70_reclassifier_cv_on_segmentation_oof_features",
        "n_images": len(records),
        "sources": [*SOURCE_DIRS, prob4_exp],
        "prob4_exp": prob4_exp,
        "current": _summarize_assignments(records, indices, current),
        "majority_all": _summarize_assignments(records, indices, majority),
    }
    for model_name in model_names:
        train_result[model_name] = _cross_validate(records, model_name)

    best_name = max(model_names, key=lambda name: train_result[name]["miou"])
    train_result["learned"] = {
        "model": best_name,
        "miou": train_result[best_name]["miou"],
        "miou_one": train_result[best_name]["miou_one"],
        "per_class": train_result[best_name]["per_class"],
        "per_class_n": train_result[best_name]["per_class_n"],
        "per_image": train_result[best_name]["per_image"],
        "per_image_one": train_result[best_name]["per_image_one"],
        "per_class_by_image": train_result[best_name]["per_class_by_image"],
    }
    train_result["comparison"] = _compare_summaries(
        train_result["current"], train_result["learned"]
    )
    threshold_selection = _select_final_threshold(records, best_name)
    final_model = _fit(records, indices, best_name)
    final_model.threshold = threshold_selection["selected_threshold"]
    train_result["final_model"] = {
        "model": best_name,
        "threshold": final_model.threshold,
        "threshold_selection": threshold_selection,
    }

    result = {**result_metadata(), **train_result, "train": train_result}
    if valid_records is None:
        result["valid"] = {
            "metric_scope": "valid_20_not_used_for_reclassifier_fit_or_threshold",
            "status": "probabilities_missing_not_evaluated",
        }
    else:
        valid_indices = list(range(len(valid_records)))
        current_valid = {
            index: record["current"] for index, record in enumerate(valid_records)
        }
        learned_valid = {
            index: final_model.predict(record["x"])
            for index, record in enumerate(valid_records)
        }
        current_metrics = _summarize_assignments(
            valid_records, valid_indices, current_valid
        )
        learned_metrics = _summarize_assignments(
            valid_records, valid_indices, learned_valid
        )
        result["valid"] = {
            "split": "valid",
            "metric_scope": "valid_20_not_used_for_reclassifier_fit_or_threshold",
            "n_images": len(valid_records),
            "current": current_metrics,
            "learned": learned_metrics,
            "comparison": _compare_summaries(current_metrics, learned_metrics),
            "model": best_name,
            "threshold": final_model.threshold,
        }
    if not limit:
        model_path = model_output_path(runs_dir, prob4_exp=prob4_exp)
        final_model.save(model_path)
        result["model_path"] = str(model_path)
    return result


def render(result):
    """교차검증 결과와 fold별 내부 임계값을 한국어 문서로 만든다."""
    prob4_exp = result["train"]["sources"][-1]
    reclass_command = f"{VENV_PYTHON} -m komap.experiments.reclass_fit --split valid"
    if prob4_exp != PROB4_DIR:
        reclass_command += f" --prob4-exp {prob4_exp}"
    lines = [
        "# 학습형 Si 덩어리 재분류",
        "",
        "s15+s20 OOF 분할을 고정하고, 4클래스 OOF 확률과 덩어리 형태 특징으로",
        "Primary/Eutectic 이름만 다시 예측했다.",
        "",
        "## train 교차검증 — 낙관 편향 가능성이 있는 추정",
        "",
        "재분류기 자체는 각 외부 검증 fold를 학습과 임계값 선택에서 제외했다. 그러나 특징을 만든",
        "세그멘테이션 fold 모델은 다른 fold 이미지를 학습에 사용한다. 따라서 다른 fold에서 만든",
        "특징으로 재분류기를 학습하면 외부 검증 이미지의 정보가 간접적으로 들어오는 스태킹 누출",
        "경로가 있다. 아래 train CV 숫자는 완전한 무누출 성능이 아니라 낙관 편향 가능성이 있는 추정이다.",
        "",
        "| 방법 | mIoU4(skip) | mIoU4(one) | Al 기지 | Al₃Ni | Primary Si | Eutectic Si |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    names = ["current", "majority_all", *[name for name in MODEL_NAMES if name in result]]
    for name in names:
        metrics = result[name]
        class_values = " | ".join(
            "—" if metrics["per_class"][str(cls)] is None else f"{metrics['per_class'][str(cls)]:.3f}"
            for cls in range(len(CLASS_NAMES))
        )
        lines.append(
            f"| `{name}` | **{metrics['miou']:.3f}** | {metrics['miou_one']:.3f} | "
            f"{class_values} |"
        )
    lines += [
        "",
        "아래 표는 해당 클래스 IoU 평균에 포함된 이미지 수다.",
        "",
        "| 방법 | Al 기지 n | Al₃Ni n | Primary Si n | Eutectic Si n |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in names:
        counts = result[name]["per_class_n"]
        lines.append(
            f"| `{name}` | {counts['0']} | {counts['1']} | {counts['2']} | {counts['3']} |"
        )
    lines += ["", "## fold 내부 임계값", ""]
    for name in MODEL_NAMES:
        if name not in result:
            continue
        selected = ", ".join(
            f"fold {row['validation_fold']}: {row['selected_threshold']:.2f}"
            for row in result[name]["folds"]
        )
        lines.append(f"- `{name}`: {selected}")
    final = result["train"]["final_model"]
    lines += [
        "",
        "## 최종 저장 설정",
        "",
        f"최종 `{final['model']}` 모델의 결정 임계값은 **{final['threshold']:.2f}**다. fold별 값의",
        "평균이 아니라 train 전체 이미지-fold OOF 예측에서 mIoU4 임계값 격자를 다시 훑어 정했다.",
        "이 값과 전체 곡선은 JSON의 `train.final_model.threshold_selection`에 기록된다.",
        "",
        "## valid 20장 분류기 단계 홀드아웃 평가",
        "",
    ]
    if result["valid"].get("status"):
        lines += [
            "valid 확률 배열이 아직 없어 평가는 수행하지 않았다. 다음 명령으로 기존 PNG·score.json과",
            "분리된 `pred_valid/probs/*.npy`를 만든 뒤 `reclass_fit --split valid`를 실행한다.",
            "`--save-prob` 모드는 기존 PNG와 `score.json`을 다시 쓰지 않고 확률 파일만 추가한다.",
            "`--out`을 반드시 붙인다 — 후광 보정 기본값이 꺼짐이라 출력 폴더 이름이 "
            "`pred_valid_nohalo`가 되고, 그러면 확률을 만들어도 이 실험이 찾지 못한다.",
            "",
            "```bash",
            f"{VENV_PYTHON} -m komap.predict --exp w2_res_s15 --split valid --save-prob"
            f" --out runs/w2_res_s15/pred_valid",
            f"{VENV_PYTHON} -m komap.predict --exp w2_res_s20 --split valid --save-prob"
            f" --out runs/w2_res_s20/pred_valid",
            f"{VENV_PYTHON} -m komap.predict --exp {result['train']['sources'][-1]} "
            f"--split valid --save-prob --out runs/{result['train']['sources'][-1]}/pred_valid",
            reclass_command,
            "```",
        ]
    else:
        valid = result["valid"]
        comparison = valid["comparison"]
        primary_common = comparison["common_per_class"]["2"]
        primary_reported_gain = (
            valid["learned"]["per_class"]["2"] - valid["current"]["per_class"]["2"]
        )
        a71 = next(row for row in comparison["images"] if row["image"] == "A71_step2_sample1")
        top = comparison["top_gain_image"]
        lines += [
            "valid 20장의 정답은 재분류기 학습과 결정 임계값 선택에는 사용하지 않았다. 따라서",
            "이 숫자는 **재분류기 단계에 한정한 홀드아웃 평가**다. 아래 `skip`은 부재 클래스를",
            "평균에서 제외하고, `one`은 부재 클래스 IoU를 1로 계산한다. 대회 산식의 부재 클래스",
            "처리는 아직 미확정이므로 두 값을 병기한다.",
            "",
            "| 방법 | valid mIoU4(skip) | valid mIoU4(one) |",
            "|---|---:|---:|",
            f"| `current` | **{valid['current']['miou']:.3f}** | {valid['current']['miou_one']:.3f} |",
            f"| `learned` | **{valid['learned']['miou']:.3f}** | {valid['learned']['miou_one']:.3f} |",
            "",
            "| 방법 | Al 기지 n | Al₃Ni n | Primary Si n | Eutectic Si n |",
            "|---|---:|---:|---:|---:|",
            f"| `current` | {valid['current']['per_class_n']['0']} | {valid['current']['per_class_n']['1']} | "
            f"{valid['current']['per_class_n']['2']} | {valid['current']['per_class_n']['3']} |",
            f"| `learned` | {valid['learned']['per_class_n']['0']} | {valid['learned']['per_class_n']['1']} | "
            f"{valid['learned']['per_class_n']['2']} | {valid['learned']['per_class_n']['3']} |",
            "",
            "### Primary Si 평균의 분모 변화",
            "",
            f"단순 집계값은 Primary Si가 {valid['current']['per_class']['2']:.3f}에서 "
            f"{valid['learned']['per_class']['2']:.3f}으로 **{primary_reported_gain:+.3f}** 변한다. 그러나 "
            f"평균 대상이 {valid['current']['per_class_n']['2']}장에서 "
            f"{valid['learned']['per_class_n']['2']}장으로 달라졌다. 양쪽 모두 Primary IoU가 존재하는 "
            f"{primary_common['n_images']}장만 비교하면 {primary_common['current']:.3f}에서 "
            f"{primary_common['learned']:.3f}으로 **{primary_common['difference']:+.3f}**다.",
            "",
            f"`A71_step2_sample1`은 정답 Primary가 없는데 current가 Primary를 예측해 IoU 0으로 "
            f"포함됐고, learned는 그 예측을 멈춰 Primary의 합집합 자체가 없어졌다. 즉 A71의 개선은 "
            f"정답에 없는 클래스를 예측하던 것을 중단한 효과이며, 같은 분모에서의 분류 정확도 "
            f"향상과는 구분해야 한다. 이미지 mIoU는 {a71['current_miou']:.3f}에서 "
            f"{a71['learned_miou']:.3f}으로 {a71['difference']:+.3f} 변했다.",
            "",
            "### 이미지별 이득 집중도",
            "",
            f"평균 이득은 {comparison['mean_difference']:+.6f}이고, 양수 "
            f"{comparison['counts']['positive']}장·음수 {comparison['counts']['negative']}장·무변화 "
            f"{comparison['counts']['unchanged']}장이다. 가장 큰 이득은 `{top['image']}`의 "
            f"{top['difference']:+.3f}으로 순증가 합계(하락 차감) 대비 {top['gain_share_pct']:.2f}%를 차지한다. "
            f"이 이미지를 제외한 나머지 평균 이득은 "
            f"{comparison['mean_difference_excluding_top_gain_image']:+.6f}이다. 이득이 소수 이미지에 "
            "집중되어 있으므로 test 20장에 유사한 이미지가 없으면 개선폭이 크게 줄 수 있다.",
            "",
            "| 이미지 | current | learned | 차이 | 순증가 대비 기여율 |",
            "|---|---:|---:|---:|---:|",
        ]
        for row in sorted(
            comparison["images"], key=lambda item: item["difference"], reverse=True
        ):
            lines.append(
                f"| `{row['image']}` | {row['current_miou']:.3f} | "
                f"{row['learned_miou']:.3f} | {row['difference']:+.3f} | "
                f"{row['gain_share_pct']:.2f}% |"
            )
        lines += [
            "",
            "### 독립성의 범위와 전역 설정 도출",
            "",
            "이 평가는 대회 규정 위반이 아니다. valid는 라벨이 제공된 데이터이고 test 정답은",
            "사용하지 않았다. 또한 이 경로는 재분류기가 새로 만든 문제가 아니라 기존 두께 규칙과",
            "기준선에도 이미 존재한다. 다만 `analysis/si_rule_discovery.py`가 train 70장과 valid 20장",
            "마스크를 합쳐 분석했고, 여기서 도출된 `THICKNESS_K`와 `PIXEL_SCALE`이 세그멘테이션",
            "리샘플링·기준선·재분류 특징에 쓰인다. 따라서 valid 정답은 재분류기 fit/threshold에는",
            "쓰이지 않았지만 전역 설정 도출에는 간접 사용됐으므로 완전 독립 또는 완전 무누출",
            "평가라고 부를 수 없다. 엄밀한 무누출 검증에는 설정 도출에도 쓰이지 않은 새 평가셋이 필요하다.",
        ]
    lines += [
        "",
        "`majority_all`은 정답 다수결 참조값이며 실제 추론법이 아니다. 학습형 점수는",
        "다수결과 목적함수가 다르므로 다수결 이득 중 회수 비율로 해석하지 않는다.",
        "",
    ]
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODEL_NAMES, default=None)
    parser.add_argument("--limit", type=int, default=0, help="앞에서 N장만(빠른 확인용)")
    parser.add_argument("--runs-dir", default=None)
    parser.add_argument("--split", choices=("train", "valid"), default="train")
    parser.add_argument(
        "--prob4-exp",
        default=PROB4_DIR,
        help="재분류기 입력으로 쓸 4클래스 실험명. 기본값이 아니면 모델·문서를 별도 경로에 저장한다",
    )
    args = parser.parse_args(argv)

    runs_dir = Path(args.runs_dir) if args.runs_dir else RUNS_DIR
    model_names = (args.model,) if args.model else MODEL_NAMES
    result = measure(
        runs_dir=runs_dir,
        limit=args.limit,
        model_names=model_names,
        split=args.split,
        prob4_exp=args.prob4_exp,
    )
    if not args.limit:
        DOCS_DIR.mkdir(parents=True, exist_ok=True)
        json_path, markdown_path = result_output_paths(prob4_exp=args.prob4_exp)
        json_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        markdown_path.write_text(render(result), encoding="utf-8")
    print(
        json.dumps(
            {
                "n_images": result["n_images"],
                "current": result["current"]["miou"],
                "majority_all": result["majority_all"]["miou"],
                "learned": {
                    "model": result["learned"]["model"],
                    "miou": result["learned"]["miou"],
                    "miou_one": result["learned"]["miou_one"],
                },
                "valid": (
                    {
                        "current": result["valid"]["current"]["miou"],
                        "learned": result["valid"]["learned"]["miou"],
                    }
                    if "current" in result["valid"]
                    else result["valid"]["status"]
                ),
            },
            ensure_ascii=False,
        )
    )
    return result


if __name__ == "__main__":
    main()
