"""최종 추론 진입점. fold·실험 앙상블 → 원본 해상도 → 후처리 → 대회 RGB PNG.

척도는 둘이며 서로 바꿔 쓸 수 없다.

- OOF4: train 70장을 각 이미지의 비학습 fold로 예측한 설정 선택 지표다.
  ``komap.experiments.cheap_wins``가 계산하며, 이 명령으로는 재현할 수 없다.
- valid4: valid 20장을 모든 fold 모델의 평균으로 예측한 제출 경로 지표다. 이 CLI가 계산한다.

`python -m komap.predict --exp w2_res_s15,w2_res_s20 --split valid`를 플래그 없이
실행하면 OOF로 고른 설정(후광 끔·k 60/80·TTA)이 그대로 적용되고 그 설정의 valid 점수가
나온다. 단, k_fit.json이 있는 과거 실험은 호환성을 위해 그 저장값을 우선한다. 재현 대상은
OOF 점수가 아니라 OOF로 선택한 추론 설정이다.

순서가 중요하다. 확률을 먼저 원본 해상도로 되돌린 뒤 argmax를 한다. 리샘플 격자에서
argmax부터 하면 라벨 이미지를 최근접 보간으로 늘리게 되어 경계가 계단처럼 깨진다.
배율이 다른 실험도 각 확률을 원본 좌표계로 되돌린 뒤에만 평균한다. 후광 제거와 두께 공식은
평균 뒤 한 번만 적용한다 — 두께 기준값이 원본 폭에서 나오기 때문이다.
"""
import argparse
import json
import os
import re
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch
from PIL import Image

from komap import hint as Hn
from komap import io
from komap.combine import combine_learned
from komap.config import N_FOLDS, RECLASS_MODEL_PATH, RUNS_DIR, THICKNESS_K
from komap.cv import overlap_tags
from komap.dataset import load_full
from komap.infer import predict_probs
from komap.metrics import competition_score, OFFICIAL_ABSENT
from komap.model import DEVICE_CHOICES, build_model, pick_device
from komap.oof import restore_full_prob
from komap.postprocess import halo_fix, load_k, thickness_rule
from komap.preprocess import anchor_normalize
from komap.reclass import RECLASS_FIT_COMMAND, ReclassModel
from komap.si2 import expand_prob, output_channels


class LoadedModels(NamedTuple):
    """체크포인트에서 복원한 앙상블과 그 학습 설정. 필드가 늘어도 호출부는 속성으로 읽으므로
    수정할 필요가 없다 — 위치 언패킹이 길어질수록 순서 실수가 조용히 숨는다."""

    models: list
    tile: int
    n_classes: int
    hint_thr: tuple | None
    resample_scale: float
    ridge: bool
    folds: tuple[int, ...] = ()
    norm: str = "anchor3"
    size_correction: bool = True
    head: str = "full"


def load_models(exp_dir, folds=None, device=None):
    """`runs/<exp>/fold*/best.pt`를 전부 읽어 eval 모드 모델 목록을 만든다.
    체크포인트에 학습 당시의 인코더·구조·타일이 들어 있으므로 추론 인자를 따로 받지 않는다.

    tile·n_classes·hint_thr·resample_scale·ridge는 앙상블 전체에 하나씩만 쓰이므로 fold마다
    다르면 마지막 값이 조용히 이겨 버린다. 그러면 나머지 fold는 학습 때와 다른 타일이나 다른
    힌트 임계값, 다른 입력 해상도로 추론되어 점수가 설명 없이 떨어진다. 첫 체크포인트를
    기준으로 삼고 어긋나면 즉시 멈춘다.

    반환: LoadedModels — 호출부는 위치가 아니라 속성(ld.tile, ld.ridge …)으로 읽는다.
    hint_thr는 (t1,t2) 또는 None(힌트 미사용)."""
    checkpoints = sorted(Path(exp_dir).glob("fold*/best.pt"))
    found_folds = {int(ck.parent.name[4:]) for ck in checkpoints}
    expected_folds = set(range(N_FOLDS)) if folds is None else set(folds)
    missing_folds = sorted(expected_folds - found_folds)
    if missing_folds:
        raise FileNotFoundError(
            f"{exp_dir}: 전체 {len(expected_folds)}-fold 앙상블에 필요한 체크포인트 중 "
            f"{len(found_folds & expected_folds)}개를 찾았고, 없는 fold는 {missing_folds}다"
        )
    extra_folds = sorted(found_folds - expected_folds)
    if folds is None and extra_folds:
        raise ValueError(
            f"{exp_dir}: 기본 전체 앙상블 범위 밖의 여분 fold {extra_folds}가 있다. "
            "--folds로 사용할 fold를 명시해 고르라"
        )
    models, tile, n_classes, hint_thr, resample_scale, ridge = [], None, None, None, 1.0, False
    norm, size_correction, head, ref = "anchor3", True, "full", None
    loaded_folds = []
    for ck in checkpoints:
        k = int(ck.parent.name[4:])
        if k not in expected_folds:
            continue
        # weights_only=True: 체크포인트는 텐서와 평범한 파이썬 값만 담는다(임의 객체 역직렬화 차단).
        st = torch.load(ck, map_location="cpu", weights_only=True)
        a = st["args"]
        thr = tuple(a["hint_thr"]) if a.get("hint_thr") else None
        # 옛 체크포인트에는 resample_scale이 없다 → 당시 동작인 1.0으로 읽는다.
        rs = float(a.get("resample_scale", 1.0))
        rg = bool(a.get("ridge", False))
        nm = a.get("norm", "anchor3")
        sc = not bool(a.get("no_size_correction", False))
        hd = a.get("head", "full")
        if tile is None:
            tile, n_classes, hint_thr, resample_scale, ridge, norm, size_correction, head, ref = (
                a["tile"], a["n_classes"], thr, rs, rg, nm, sc, hd, ck)
        elif (a["tile"], a["n_classes"], thr, rs, rg, nm, sc, hd) != (
            tile, n_classes, hint_thr, resample_scale, ridge, norm, size_correction, head
        ):
            raise ValueError(
                f"fold 간 설정이 다르다: {ck} 는 tile={a['tile']} n_classes={a['n_classes']} "
                f"hint={thr} resample={rs} ridge={rg} norm={nm} size_correction={sc} head={hd} 인데 "
                f"기준 {ref} 는 tile={tile} n_classes={n_classes} hint={hint_thr} "
                f"resample={resample_scale} ridge={ridge} norm={norm} "
                f"size_correction={size_correction} head={head} 다. "
                "같은 설정으로 학습한 fold만 앙상블할 수 있다(--folds 로 골라내라)."
            )
        # 가중치는 체크포인트에서 덮어쓰므로 사전학습 가중치를 다시 받을 필요가 없다("none").
        m = build_model(a["encoder"], "none", output_channels(a["n_classes"], hd), a.get("arch", "unetplusplus"),
                        in_channels=Hn.input_channels(bool(a.get("hint")), rg))
        m.load_state_dict(st["model"])
        models.append(m.to(device).eval())
        loaded_folds.append(k)
    if not models:
        raise FileNotFoundError(f"{exp_dir} 에 fold*/best.pt 없음")
    if len(models) != len(expected_folds) or set(loaded_folds) != expected_folds:
        raise RuntimeError(
            f"{exp_dir}: 기대 fold {sorted(expected_folds)}와 실제 적재 fold "
            f"{sorted(loaded_folds)}가 다르다"
        )
    return LoadedModels(
        models, tile, n_classes, hint_thr, resample_scale, ridge,
        tuple(sorted(loaded_folds)), norm, size_correction, head
    )


def predict_full_prob(models, image_path, tile, tta, device, hint_thr=None,
                      resample_scale=1.0, ridge=False, norm="anchor3", size_correction=True,
                      head="full"):
    """한 실험의 fold 확률을 평균한 뒤 원본 좌표계로 복원한다."""
    # 배율 리샘플은 load_full이 이미 했다. hint_thr가 있으면 힌트 4채널까지 붙여 온다.
    xf, _, meta = load_full(
        image_path, n_classes=3, hint_thr=hint_thr, resample_scale=resample_scale,
        ridge=ridge, norm=norm, size_correction=size_correction
    )
    prob = np.mean([predict_probs(m, xf, tile, 0.5, tta, device) for m in models], axis=0)
    H, W = meta["height"], meta["width"]
    return expand_prob(restore_full_prob(prob, H, W), head)


def prob_to_mask(prob_full, image_path, n_classes, halo, k):
    """원본 좌표계 확률 하나에 argmax·후광·두께 판정을 각각 한 번 적용한다.

    4클래스 모델 판정은 예측값이 아니라 체크포인트의 n_classes로 한다. 예측값으로 재면
    4클래스 모델이 마침 초정 Si를 한 픽셀도 찍지 않은 이미지에서 판정이 뒤집힌다.
    """
    p3 = prob_full.argmax(0).astype(np.int64)
    if n_classes == 4:  # 4클래스 모델이면 Si를 하나로 병합한 뒤 두께 공식으로 다시 가른다
        p3 = io.to3(p3)
    if halo:
        # 후광 제거는 anchor3 회색조를 쓴다. 제출은 후광을 끄며 대조군을 후광 켜고 추론하지 않는다.
        p3 = halo_fix(anchor_normalize(io.load_gray(image_path)), p3)
    return thickness_rule(p3, width=prob_full.shape[2], k=k)


def predict_image(models, image_path, tile, tta, halo, device, n_classes=3, k=None,
                  hint_thr=None, resample_scale=1.0, ridge=False, norm="anchor3",
                  size_correction=True, head="full"):
    """단일 실험의 이미지 한 장을 기존 순서 그대로 원본 해상도 4클래스로 예측한다."""
    prob_full = predict_full_prob(models, image_path, tile, tta, device, hint_thr=hint_thr,
                                  resample_scale=resample_scale, ridge=ridge, norm=norm,
                                  size_correction=size_correction, head=head)
    return prob_to_mask(prob_full, image_path, n_classes, halo, k)


def predict_ensemble_prob(loaded_models, image_path, tta, device):
    """실험별 확률을 원본 해상도로 복원한 뒤 평균한다."""
    n_classes = {ld.n_classes for ld in loaded_models}
    if len(n_classes) != 1:
        raise ValueError(f"실험 간 클래스 수가 다르다: {sorted(n_classes)}")
    probs = [
        predict_full_prob(ld.models, image_path, ld.tile, tta, device,
                          hint_thr=ld.hint_thr, resample_scale=ld.resample_scale, ridge=ld.ridge,
                          norm=ld.norm, size_correction=ld.size_correction, head=ld.head)
        for ld in loaded_models
    ]
    # 단일 실험은 불필요한 평균 연산도 피하고 기존 수치 경로를 그대로 보존한다.
    return probs[0] if len(probs) == 1 else np.mean(probs, axis=0)


def predict_ensemble_image(loaded_models, image_path, tta, halo, device, k):
    """실험별 원본 해상도 확률을 평균하고 공통 후처리를 한 번만 적용한다."""
    prob_full = predict_ensemble_prob(loaded_models, image_path, tta, device)
    n_classes = {ld.n_classes for ld in loaded_models}
    return prob_to_mask(prob_full, image_path, next(iter(n_classes)), halo, k)


def resolve_k(exp_dirs, k=None):
    """명시한 k 또는 모든 실험이 공유하는 k_fit을 실제 추론값으로 확정한다."""
    if k is not None:
        return k
    dirs = [Path(exp_dirs)] if isinstance(exp_dirs, (str, Path)) else [Path(p) for p in exp_dirs]
    if not dirs:
        raise ValueError("실험을 하나 이상 지정해야 한다")
    fitted = [load_k(d) or dict(THICKNESS_K) for d in dirs]
    if any(value != fitted[0] for value in fitted[1:]):
        raise ValueError("실험별 k_fit이 다르다 — --k로 공통 값을 명시해야 한다")
    return fitted[0]


def metric_scope_name(split: str, folds=None) -> str:
    """평가 split과 실제 fold 선택을 산출물에 기록할 범위 이름으로 만든다."""
    if folds is None or sorted(folds) == list(range(N_FOLDS)):
        return f"{split}_full_fold_ensemble"
    # fold를 정렬해 이름을 짓는다. 모델 선택은 순서와 무관한 포함 여부 검사이므로
    # [1,0]과 [0,1]은 같은 앙상블이다. 정렬하지 않으면 같은 구성에 두 이름이 생긴다.
    encoded = "-".join(str(fold) for fold in sorted(folds))
    return f"{split}_folds{encoded}"


def _actual_fold_selection(loaded_models, requested_folds):
    """모든 실험에서 실제 적재한 fold가 같은지 확인하고 그 집합을 반환한다."""
    fallback = set(range(N_FOLDS)) if requested_folds is None else set(requested_folds)
    selections = [set(loaded.folds) for loaded in loaded_models if loaded.folds]
    actual = selections[0] if selections else fallback
    if any(selection != actual for selection in selections[1:]):
        raise ValueError(
            "실험별 실제 적재 fold가 다르다: "
            + ", ".join(str(sorted(selection)) for selection in selections)
        )
    return tuple(sorted(actual))


def run(exp_dir, image_paths, out_dir, folds=None, tta=True, halo=True, k=None, exp_names=None,
        metric_scope=None, save_prob=False, combiner="thickness", prob4_exp_dir=None,
        prob4_exp_names=None, reclass_model_path=RECLASS_MODEL_PATH):
    """하나 이상의 실험을 추론한다. 경로별 설정은 섞지 않고 원본 확률만 평균한다."""
    if combiner == "learned" and halo:
        raise ValueError(
            "learned 재분류기는 halo 보정 없는 덩어리로 학습되어, halo를 켜면 "
            "학습 때와 다른 특징이 들어간다. --no-halo를 사용하라"
        )
    device = pick_device()
    exp_dirs = [Path(exp_dir)] if isinstance(exp_dir, (str, Path)) else [Path(p) for p in exp_dir]
    if not exp_dirs:
        raise ValueError("실험을 하나 이상 지정해야 한다")
    loaded = [load_models(d, folds, device) for d in exp_dirs]
    prob4_loaded = []
    prob4_names_used = []
    reclass_model = None
    if combiner == "learned":
        if prob4_exp_dir is None:
            raise ValueError("learned 결합기에는 --prob4-exp로 4클래스 실험을 지정해야 한다")
        prob4_dirs = (
            [Path(prob4_exp_dir)]
            if isinstance(prob4_exp_dir, (str, Path))
            else [Path(p) for p in prob4_exp_dir]
        )
        if not prob4_dirs:
            raise ValueError("learned 결합기에는 4클래스 실험을 하나 이상 지정해야 한다")
        prob4_loaded = [load_models(d, folds, device) for d in prob4_dirs]
        wrong_classes = [
            f"{path.name}={models.n_classes}"
            for path, models in zip(prob4_dirs, prob4_loaded, strict=True)
            if models.n_classes != 4
        ]
        if wrong_classes:
            raise ValueError(
                "--prob4-exp 체크포인트는 4클래스여야 한다: " + ", ".join(wrong_classes)
            )
        prob4_names_used = (
            list(prob4_exp_names) if prob4_exp_names is not None else [d.name for d in prob4_dirs]
        )
        if len(prob4_names_used) != len(prob4_dirs):
            raise ValueError("4채널 실험 이름 수와 경로 수가 다르다")
        reclass_model_path = Path(reclass_model_path)
        if not reclass_model_path.exists():
            raise FileNotFoundError(
                f"재분류 모델이 없다: {reclass_model_path}\n"
                "다음 명령으로 재생성하라: "
                f"{RECLASS_FIT_COMMAND}"
            )
        reclass_model = ReclassModel.load(reclass_model_path)
    elif combiner != "thickness":
        raise ValueError(f"지원하지 않는 결합기: {combiner}")
    actual_folds = _actual_fold_selection([*loaded, *prob4_loaded], folds)
    k = resolve_k(exp_dirs, k)
    names_used = list(exp_names) if exp_names is not None else [d.name for d in exp_dirs]
    if len(names_used) != len(exp_dirs):
        raise ValueError("실험 이름 수와 경로 수가 다르다")
    # 범위 이름 검증은 **출력을 하나도 만들기 전에** 끝낸다. 저장 뒤에 거부하면 PNG는 새 결과인데
    # score.json은 이전 결과로 남아, 한 폴더 안에서 둘이 어긋난다.
    metric_scope = metric_scope or metric_scope_name("valid", actual_folds)
    match = re.fullmatch(r"(valid|test)_(?:full_fold_ensemble|folds\d+(?:-\d+)*)", metric_scope)
    if match is None or metric_scope != metric_scope_name(match.group(1), actual_folds):
        raise ValueError(f"알 수 없는 metric_scope: {metric_scope}")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prob_dir = out_dir / "probs"
    if save_prob:
        prob_dir.mkdir(parents=True, exist_ok=True)
    preds, targets, names = [], [], []
    for ip in image_paths:
        prob_full = predict_ensemble_prob(loaded, ip, tta, device)
        if save_prob:
            np.save(prob_dir / f"{ip.stem}.npy", prob_full.astype(np.float16))
        if combiner == "learned":
            prob4_full = predict_ensemble_prob(prob4_loaded, ip, tta, device)
            gray_norm = anchor_normalize(io.load_gray(ip)) if halo else None
            p4 = combine_learned(
                prob_full,
                width=prob_full.shape[2],
                gray_norm=gray_norm,
                k=k,
                model=reclass_model,
                prob4=prob4_full,
            )
        else:
            p4 = prob_to_mask(prob_full, ip, loaded[0].n_classes, halo, k)
        if not save_prob:
            Image.fromarray(io.to_rgba(p4)).save(
                out_dir / ip.name.replace("_image.png", "_mask.png")
            )
        preds.append(p4)
        names.append(ip.name)
        mp = io.mask_path_for(ip)
        if mp.exists():
            targets.append(io.load_mask(mp))

    def settings(values):
        """단일 실험은 기존 스칼라 형식을, 여러 실험은 exps와 같은 순서의 목록을 쓴다."""
        return values[0] if len(values) == 1 else values

    result = {"n": len(preds), "folds": list(actual_folds), "tta": tta, "halo": halo, "k": k,
              "metric_scope": metric_scope,
              "exps": names_used, "combiner": combiner, "prob4_exps": prob4_names_used,
              "hint_thr": settings([list(ld.hint_thr) if ld.hint_thr else None for ld in loaded]),
              # 격자를 여러 개 돌리면 score.json만 남는다. 그 파일 하나로 어떤 설정의 점수인지
              # 알 수 있어야 하므로 체크포인트에서 읽은 학습 설정을 전부 적는다.
              "resample_scale": settings([ld.resample_scale for ld in loaded]),
              "ridge": settings([ld.ridge for ld in loaded]),
              "norm": settings([ld.norm for ld in loaded]),
              "size_correction": settings([ld.size_correction for ld in loaded])}
    if preds and len(targets) == len(preds):
        # 부재 클래스 처리는 대회 규정이 명시하지 않아 두 산식을 병기한다(skip=제외, one=1.0으로 계산).
        s1 = competition_score(preds, targets, 4, "skip")
        s2 = competition_score(preds, targets, 4, "one")
        # 공식 산식(부재 클래스 IoU 0, 2026-09-29 주최 측 답변)을 함께 남긴다.
        s0 = competition_score(preds, targets, 4, OFFICIAL_ABSENT)
        result.update(miou_official=s0["miou"], miou_skip=s1["miou"], miou_one=s2["miou"], per_class=s1["per_class_mean"],
                      per_image=dict(zip(names, s1["per_image"], strict=True)))
        # 합금 ID가 test에도 나오는 이미지와 처음 보는 합금을 갈라 본다. 일반화 성능의 대리 지표다.
        test_names = [p.name for p in io.list_split("test")]
        if test_names:
            tags = overlap_tags(names, test_names)
            for flag, key in [(True, "miou_alloy_in_test"), (False, "miou_alloy_new")]:
                sel = [v for n, v in zip(names, s1["per_image"], strict=True) if tags[n] == flag]
                result[key] = float(np.mean(sel)) if sel else None
    # 확률 생성 모드는 기존 PNG·score.json 기준자를 건드리지 않고 probs/만 추가한다.
    if not save_prob:
        (out_dir / "score.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def default_out_name(split: str, tta: bool, halo: bool, k=None, folds=None,
                     combiner="thickness") -> str:
    """결과 폴더 이름. 기존 접미사 규칙을 유지해 과거 산출물과 충돌하지 않게 한다.

    변형 설정도 같은 이름을 쓰면 심사자가 --no-tta로 한 번 돌린 순간 보고한 score.json이
    덮어써져 원래 점수를 되찾을 수 없다. 설정을 이름에 박아 서로 침범하지 않게 한다."""
    name = f"pred_{split}" + ("" if tta else "_notta") + ("" if halo else "_nohalo")
    if k is not None and k != THICKNESS_K:
        name += f"_k{k['A']}-{k['B']}"
    if folds is not None and sorted(folds) != list(range(N_FOLDS)):
        encoded = "".join(str(fold) for fold in folds)
        name += f"_fold{encoded}" if len(folds) == 1 else f"_folds{encoded}"
    if combiner != "thickness":
        name += f"_{combiner}"
    return name


def parse_exps(value: str) -> list[str]:
    """쉼표 구분 실험명을 순서대로 읽고 빈 이름은 거부한다."""
    exps = [v.strip() for v in value.split(",")]
    if not exps or any(not v for v in exps):
        raise ValueError("--exp에는 쉼표로 구분한 실험명을 하나 이상 지정해야 한다")
    return exps


def parse_k(value: str) -> dict[str, int]:
    """`A,B` 순서의 두께 상수 두 개를 설정 사전으로 바꾼다."""
    parts = value.split(",")
    if len(parts) != 2:
        raise ValueError("--k는 A,B 순서의 두 정수여야 한다(예: 60,80)")
    try:
        ka, kb = (int(v) for v in parts)
    except ValueError as exc:
        raise ValueError("--k는 A,B 순서의 두 정수여야 한다(예: 60,80)") from exc
    if ka <= 0 or kb <= 0:
        raise ValueError("--k의 두 값은 양수여야 한다")
    return {"A": ka, "B": kb}


def parse_cli(argv=None):
    """OOF로 고른 추론 설정을 valid/test 전체-fold 앙상블에 적용한다."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True, help="실험명. 여러 개면 쉼표로 구분해 확률 앙상블")
    ap.add_argument("--split", default="valid", choices=["valid", "test"])
    ap.add_argument("--folds", default=None)
    ap.add_argument("--tta", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--halo", action=argparse.BooleanOptionalAction, default=False,
                    help="후광 보정(기본: 끔; OOF 선택 설정)")
    ap.add_argument("--k", type=parse_k, default=None, metavar="A,B",
                    help="배율군 A/B의 두께 상수(미지정 시 실험의 k_fit.json, 없으면 60/80)")
    ap.add_argument("--device", default=None, choices=list(DEVICE_CHOICES),
                    help="장치 강제(미지정 시 mps→cuda→cpu 자동)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--combiner", choices=["thickness", "learned"], default="thickness")
    ap.add_argument("--prob4-exp", default=None,
                    help="learned 결합기에 쓸 4클래스 실험명. 여러 개면 쉼표로 구분")
    ap.add_argument(
        "--reclass-model",
        type=Path,
        default=None,
        help="learned 결합기가 읽을 재분류 모델 경로. 기본값은 <runs-dir>/reclass/model.pkl",
    )
    ap.add_argument("--runs-dir", type=Path, default=RUNS_DIR,
                    help="실험 체크포인트와 기본 출력의 기준 디렉터리")
    ap.add_argument("--save-prob", action="store_true", help="원본 해상도 확률을 probs/*.npy로 저장")
    args = ap.parse_args(argv)
    if args.combiner == "learned" and args.prob4_exp is None:
        ap.error("--combiner learned에는 --prob4-exp가 필수다")
    return args


def main(argv=None):
    a = parse_cli(argv)
    if a.device:                      # pick_device()가 읽기 전에 걸어야 효력이 있다
        os.environ["KOMAP_DEVICE"] = a.device
    folds = [int(v) for v in a.folds.split(",")] if a.folds else None
    exps = parse_exps(a.exp)
    runs_dir = Path(a.runs_dir)
    exp_dirs = [runs_dir / exp for exp in exps]
    prob4_exps = parse_exps(a.prob4_exp) if a.prob4_exp is not None else []
    prob4_exp_dirs = [runs_dir / exp for exp in prob4_exps]
    reclass_model_path = a.reclass_model or runs_dir / "reclass" / "model.pkl"
    k = resolve_k(exp_dirs, a.k)
    # 앙상블은 별도 디렉터리를 써 단일 실험 산출물과 섞이지 않게 한다.
    result_dir = runs_dir / "+".join(exps)
    out = a.out or result_dir / default_out_name(
        a.split, a.tta, a.halo, k, folds, a.combiner
    )
    r = run(exp_dirs, io.list_split(a.split), out, folds, a.tta, a.halo, k, exps,
            metric_scope=metric_scope_name(a.split, folds), save_prob=a.save_prob,
            combiner=a.combiner, prob4_exp_dir=prob4_exp_dirs or None,
            prob4_exp_names=prob4_exps, reclass_model_path=reclass_model_path)
    print(json.dumps({k: v for k, v in r.items() if k != "per_image"}, ensure_ascii=False, indent=2))
    return r


if __name__ == "__main__":
    main()
