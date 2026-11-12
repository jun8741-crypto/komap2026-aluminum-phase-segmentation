"""경계 보조 헤드(다중 과제 학습). 분할 특징에서 상 경계를 함께 예측한다."""
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from komap.config import BOUNDARY_BAND_PX, IGNORE_LABEL
from komap.si2 import args_differences

EDGE_AUX_DEFAULT_WEIGHT = 0.0
LEGACY_DEFAULT_KEYS = (
    'anchor_jitter', 'no_interp_geo', 'norm', 'no_size_correction', 'photometric',
    'train_list', 'val_list', 'select_epoch', 'oof_res', 'loss_scope', 'select_metric', 'head',
)
CLAIM_SCOPE = (
    'exploratory candidate selection vs the actually submitted models; '
    'not evidence that edge supervision itself causes the gain; '
    'baseline trained with 2026-09-19 code, current-code equivalence assumed from commit history, not measured; '
    'no lambda=0 control run'
)
BASELINE_EQUIVALENCE = {
    'measured': False,
    'assumed_from': 'commit history 2026-09-19..2026-10-13',
    'legacy_default_keys': list(LEGACY_DEFAULT_KEYS),
}


def _dilate_l1(mask: torch.Tensor, band_px: int) -> torch.Tensor:
    """좌표차 L1 반경만큼 팽창한다. 가장자리에서 반대편으로 넘기지 않는다."""
    expanded = mask
    for _ in range(band_px):
        up = F.pad(expanded[:, 1:], (0, 0, 0, 1))
        down = F.pad(expanded[:, :-1], (0, 0, 1, 0))
        left = F.pad(expanded[:, :, 1:], (0, 1))
        right = F.pad(expanded[:, :, :-1], (1, 0))
        expanded = expanded | up | down | left | right
    return expanded


def boundary_target(target: torch.Tensor, band_px: int = BOUNDARY_BAND_PX,
                    n_classes: int = 3) -> tuple[torch.Tensor, torch.Tensor]:
    """유효한 타 클래스 픽셀이 L1 반경 안에 있으면 경계로 표시한다."""
    if band_px < 0:
        raise ValueError('경계 폭은 음수일 수 없다')
    valid = target != IGNORE_LABEL
    if torch.any(valid & ((target < 0) | (target >= n_classes))):
        raise ValueError('범위 밖 정답 라벨')
    edge = torch.zeros_like(valid)
    for k in range(n_classes):
        own = target == k
        other = valid & ~own
        edge |= own & _dilate_l1(other, band_px)
    return edge, valid


def edge_bce_loss(edge_logits: torch.Tensor, edge: torch.Tensor,
                  valid: torch.Tensor) -> torch.Tensor:
    """무시 라벨을 뺀 픽셀의 로짓 BCE 평균을 계산한다."""
    if not valid.any():
        return edge_logits.sum() * 0.0
    per_pixel = F.binary_cross_entropy_with_logits(edge_logits, edge[:, None].float(), reduction='none')
    return per_pixel[valid[:, None]].mean()


class EdgeAuxModel(torch.nn.Module):
    """추론 출력은 분할 텐서 하나로 유지하는 경계 보조 래퍼."""

    def __init__(self, base: torch.nn.Module, edge_head: torch.nn.Conv2d):
        super().__init__()
        self.base = base
        self.edge_head = edge_head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x)

    def forward_with_edge(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self.base.check_input_shape(x)
        features = self.base.encoder(x)
        decoded = self.base.decoder(features)
        return self.base.segmentation_head(decoded), self.edge_head(decoded)


def attach_edge_head(base: torch.nn.Module, seed: int) -> EdgeAuxModel:
    """전용 CPU 생성기로 헤드를 초기화해 전역 RNG를 보존한다."""
    dec_ch = base.segmentation_head[0].in_channels
    head = torch.nn.utils.skip_init(torch.nn.Conv2d, dec_ch, 1, 3, padding=1)
    generator = torch.Generator(device='cpu')
    generator.manual_seed(seed + 1)
    bound = 1 / math.sqrt(dec_ch * 9)
    with torch.no_grad():
        head.weight.uniform_(-bound, bound, generator=generator)
        head.bias.uniform_(-bound, bound, generator=generator)
    return EdgeAuxModel(base, head)


def checkpoint_state_dict(model: torch.nn.Module) -> dict:
    """기존 분할 모델과 동일한 키만 체크포인트에 저장한다."""
    return model.base.state_dict() if isinstance(model, EdgeAuxModel) else model.state_dict()


def unexplained_arg_differences(saved: dict, expected: dict, defaults: dict,
                                legacy_keys: tuple[str, ...] = LEGACY_DEFAULT_KEYS) -> dict:
    """구형 체크포인트에 없는 기본 인자만 설명된 차이로 제외한다."""
    diffs = args_differences(saved, expected)
    return {key: pair for key, pair in diffs.items()
            if not (key in legacy_keys and key not in saved and key in expected
                    and key in defaults and expected[key] == defaults[key])}


def fold_completeness(fold_dir: Path, expected_names: list[str], expected_args: dict,
                      defaults: dict) -> str:
    """재개 가능한 완비 fold인지, 비어 있거나 부분 산출물인지 판정한다."""
    fold_dir = Path(fold_dir)
    if not fold_dir.exists() or (fold_dir.is_dir() and not any(fold_dir.iterdir())):
        return 'absent'
    for name in ('result.json', 'best.pt'):
        if not (fold_dir / name).is_file():
            return f'partial: {name} 누락'
    expected = {Path(name).stem for name in expected_names}
    actual = {p.stem for p in (fold_dir / 'oof_probs').glob('*.npy')}
    if missing := sorted(expected - actual):
        return f'partial: OOF 누락 {missing}'
    if extra := sorted(actual - expected):
        return f'partial: OOF 여분 {extra}'
    try:
        saved = torch.load(fold_dir / 'best.pt', map_location='cpu', weights_only=True)['args']
        diffs = unexplained_arg_differences(saved, expected_args, defaults)
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        return f'partial: best.pt 읽기 실패 {exc}'
    if diffs:
        return f'partial: args 불일치 {diffs}'
    return 'complete'
