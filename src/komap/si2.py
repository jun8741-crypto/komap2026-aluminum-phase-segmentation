"""Si 전용 2채널 헤드와 기존 4채널 확률 형식의 변환.

출력 0은 Primary Si, 1은 Eutectic Si다. 저장·평가·추론에서는
[0, 0, p_primary, p_eutectic]으로 전개해 기존 재분류 경로를 유지한다.
"""
import json
from pathlib import Path

import numpy as np
import torch

from komap.config import CLS_EUTSI, CLS_PRISI, IGNORE_LABEL

OUTPUT_HEADS = ("full", "si2")
SI2_CHANNELS = 2
SI2_SOURCE_CLASSES = (CLS_PRISI, CLS_EUTSI)


def output_channels(n_classes: int, head: str = "full") -> int:
    """정답 라벨 수와 헤드 종류로 실제 출력 채널 수를 정한다."""
    if head not in OUTPUT_HEADS:
        raise ValueError(f"알 수 없는 출력 헤드: {head}")
    if head == "full":
        return n_classes
    if n_classes != 4:
        raise ValueError("si2 헤드는 4클래스 정답 전용이다")
    return SI2_CHANNELS


def remap_target_si2(target: torch.Tensor) -> torch.Tensor:
    """4클래스 정답 2→0, 3→1, 나머지→무시 라벨로 사본을 만든다."""
    out = torch.full_like(target, IGNORE_LABEL)
    out[target == CLS_PRISI] = 0
    out[target == CLS_EUTSI] = 1
    return out


def expand_prob_si2(prob2: np.ndarray) -> np.ndarray:
    """2채널 확률을 기존 4채널 [0, 0, Primary, Eutectic] 형식으로 전개한다."""
    if prob2.shape[0] != SI2_CHANNELS:
        raise ValueError(f"si2 확률은 2채널이어야 한다: {prob2.shape}")
    out = np.zeros((4, *prob2.shape[1:]), dtype=prob2.dtype)
    out[2:] = prob2
    return out


def expand_prob(prob: np.ndarray, head: str = "full") -> np.ndarray:
    """full은 원본 객체를 그대로, si2는 4채널 사본을 돌려준다."""
    if head == "full":
        return prob
    if head == "si2":
        return expand_prob_si2(prob)
    raise ValueError(f"알 수 없는 출력 헤드: {head}")


# 체인 점검: 학습 산출물을 읽기만 하고 재학습·수정은 하지 않는다.
def args_differences(saved: dict, expected: dict,
                     skip=("exp", "fold", "runs_dir", "hint_thr")) -> dict:
    """재개 체크포인트와 예정 학습 인자의 차이를 키별로 돌려준다."""
    return {key: (saved.get(key), expected.get(key))
            for key in sorted((saved.keys() | expected.keys()) - set(skip))
            if key not in saved or key not in expected or saved[key] != expected[key]}


def oof_problems(exp_dir: Path, fold: int, expected_names: list[str],
                 reference_dir: Path | None = None) -> list[str]:
    """한 fold의 결과·체크포인트·확률 파일 완비성 문제를 한국어 목록으로 돌려준다."""
    out = Path(exp_dir) / f"fold{fold}"
    issues = []
    result_path = out / "result.json"
    if not result_path.exists():
        issues.append(f"fold{fold}: result.json 누락")
    else:
        try:
            result = json.loads(result_path.read_text())
            if result.get("head") != "si2" or result.get("select_metric") != "si_pair":
                issues.append(f"fold{fold}: result.json head·select_metric 불일치")
        except (OSError, ValueError) as exc:
            issues.append(f"fold{fold}: result.json 읽기 실패: {exc}")
    if not (out / "best.pt").exists():
        issues.append(f"fold{fold}: best.pt 누락")
    prob_dir = out / "oof_probs"
    expected = {name.replace(".png", "") for name in expected_names}
    actual = {path.stem for path in prob_dir.glob("*.npy")}
    if missing := sorted(expected - actual):
        issues.append(f"fold{fold}: OOF 누락 {missing}")
    if extra := sorted(actual - expected):
        issues.append(f"fold{fold}: OOF 여분 {extra}")
    for stem in sorted(expected & actual):
        path = prob_dir / f"{stem}.npy"
        try:
            prob = np.load(path, allow_pickle=False)
        except (OSError, ValueError, EOFError) as exc:
            issues.append(f"fold{fold} {stem}: 확률 읽기 실패: {exc}")
            continue
        if prob.dtype != np.float16 or prob.ndim != 3 or prob.shape[0] != 4:
            issues.append(f"fold{fold} {stem}: dtype·shape 불일치 {prob.dtype} {prob.shape}")
            continue
        if prob[:2].any():
            issues.append(f"fold{fold} {stem}: 채널 0·1이 0이 아니다")
        if not np.isfinite(prob).all():
            issues.append(f"fold{fold} {stem}: 유한하지 않은 확률")
        if not np.allclose(prob[2].astype(np.float32) + prob[3].astype(np.float32), 1, atol=0.01):
            issues.append(f"fold{fold} {stem}: Si 채널 합이 1이 아니다")
        if reference_dir is not None:
            ref_path = Path(reference_dir) / f"fold{fold}" / "oof_probs" / f"{stem}.npy"
            if not ref_path.exists():
                issues.append(f"fold{fold} {stem}: 기준 확률 누락")
            else:
                try:
                    ref_shape = np.load(ref_path, mmap_mode="r", allow_pickle=False).shape
                    if prob.shape != ref_shape:
                        issues.append(f"fold{fold} {stem}: 기준 확률 shape 불일치")
                except (OSError, ValueError) as exc:
                    issues.append(f"fold{fold} {stem}: 기준 확률 읽기 실패: {exc}")
    return issues
