"""규칙 지도 캐스케이드의 힌트 채널(공모안 메모 §5).

규칙 파이프라인(임계값 → 후광 제거 → 두께 공식)의 거친 4클래스 지도를 원-핫 4채널로 회색조 옆에
붙여 모델에 준다. 모델은 경계만 다듬으면 되므로 학습이 쉬워지리라 기대한다.
힌트 dropout: 규칙 지도의 Eutectic Si IoU가 49%로 낮아 모델이 힌트를 맹신하면 규칙의 오류를
그대로 베낀다. 학습 표본의 30%는 힌트를 0으로 지워 회색조만으로도 풀게 강제한다.
"""
import json
from pathlib import Path

import numpy as np
import torch

from komap import io
from komap.baselines.rule_pipeline import fit_thresholds, segment3
from komap.postprocess import halo_fix, thickness_rule
from komap.preprocess import anchor_normalize
from komap.ridge import RIDGE_CH

HINT_CH = 4
HINT_DROP_P = 0.3
THRESHOLDS_FILE = Path("rule") / "thresholds.json"


def input_channels(hint: bool, ridge: bool) -> int:
    """모델 입력 채널 수. train과 predict가 같은 식을 쓰도록 한곳에서만 정의한다.
    채널 순서는 [회색조 1, 힌트 4, 능선 1]이다."""
    return 1 + (HINT_CH if hint else 0) + (RIDGE_CH if ridge else 0)


def fit_thresholds_cached(runs_dir) -> tuple[int, int]:
    """runs/rule/thresholds.json이 있으면 읽고, 없으면 train 전체에서 맞춰 저장한다.

    임계값은 학습·추론이 반드시 같아야 하므로 파일 한 곳을 단일 진실 공급원으로 둔다."""
    p = Path(runs_dir) / THRESHOLDS_FILE
    if p.exists():
        d = json.loads(p.read_text())
        return int(d["t1"]), int(d["t2"])
    pairs = [(anchor_normalize(io.load_gray(ip)), io.to3(io.load_mask(io.mask_path_for(ip))))
             for ip in io.list_split("train")]
    t1, t2 = fit_thresholds(pairs)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"t1": t1, "t2": t2}))
    return t1, t2


def rule_hint(gray_norm: np.ndarray, t1: int, t2: int, width: int) -> np.ndarray:
    """앵커 정규화된 원본 해상도 회색조 → 규칙 4클래스 지도(uint8).

    두께 공식이 원본 폭을 기준으로 초정/공정을 가르므로 리샘플 이전 해상도에서 만들어야 한다."""
    p3 = halo_fix(gray_norm, segment3(gray_norm, t1, t2))
    return thickness_rule(p3, width=width).astype(np.uint8)


def onehot(hint4: np.ndarray) -> np.ndarray:
    """4클래스 지도 → float32 [4,H,W] 원-핫. 0~3 밖의 값(예: 여백 라벨)은 0벡터가 된다."""
    return np.stack([(hint4 == k) for k in range(HINT_CH)]).astype(np.float32)


def drop(x: torch.Tensor, p: float = HINT_DROP_P, rng=None) -> torch.Tensor:
    """x[C,H,W]의 채널 1.. (힌트)를 확률 p로 전부 0으로. 채널 0(회색조)은 건드리지 않는다."""
    rng = np.random.default_rng() if rng is None else rng
    if rng.random() < p:
        x = x.clone()
        x[1:] = 0
    return x
