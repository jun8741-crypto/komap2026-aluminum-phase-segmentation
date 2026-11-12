"""손실 함수. 경계 가중과 내부 가중은 반대 방향이므로 비교 실험으로 결정한다(스펙 4.5).

- 경계 띠 무시(ignore_band_px): 정답 마스크의 경계 1~2px는 라벨 자체가 불확실하므로
  학습에서 빼 잡음을 줄인다.
- Si 내부 가중(ce_interior): 반대로 경계가 아닌 Si 내부를 더 중히 본다.
두 방향이 상충하므로 어느 쪽이 대회 점수를 올리는지는 실험으로 가린다.
"""
import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage as ndi
from segmentation_models_pytorch.losses import DiceLoss, LovaszLoss

from komap.config import BOUNDARY_BAND_PX, CLS_EUTSI, CLS_PRISI, CLS_SI3, IGNORE_LABEL
from komap.si2 import OUTPUT_HEADS, output_channels, remap_target_si2

# 무시 라벨은 config가 단일 진실 공급원이다. 증강 여백도 같은 값을 쓴다(augment.py).
IGNORE = IGNORE_LABEL
INTERIOR_WEIGHT = 2.0
CE_DICE_RATIO = 0.75  # CE 비중. 나머지가 Dice.
LOSS_SCOPES = ("all", "si")


def _band_mask(target: torch.Tensor, band_px: int) -> torch.Tensor:
    """각 클래스 영역에서 다른 클래스와 band_px 이내로 맞닿은 경계 띠를 True로 표시한다.
    배치별 scipy 루프라 느리지만 타일 크기에서는 문제되지 않는다(계획 승인)."""
    out = torch.zeros_like(target, dtype=torch.bool)
    for b in range(target.shape[0]):
        t = target[b].cpu().numpy()
        band = np.zeros(t.shape, bool)
        for k in np.unique(t):
            m = t == k
            band |= m & ndi.binary_dilation(~m, iterations=band_px)
        out[b] = torch.from_numpy(band)
    return out.to(target.device)


def _si_classes(n_classes: int) -> tuple[int, ...]:
    """가중을 걸 Si 클래스. 4클래스 라벨에서는 Si가 초정(CLS_PRISI)·공정(CLS_EUTSI) 둘로 갈린다.
    3클래스만 가정하면 4클래스 학습에서 공정 Si 내부가 통째로 가중을 못 받는다."""
    return (CLS_PRISI, CLS_EUTSI) if n_classes == 4 else (CLS_SI3,)


def _interior_weight(target, si_classes=(CLS_SI3,), band_px=BOUNDARY_BAND_PX, w=INTERIOR_WEIGHT):
    """Si 내부(경계에서 band_px 이상 떨어진) 픽셀에만 가중 w, 나머지는 1.0.
    si_classes는 튜플이다(3클래스: Si 하나 / 4클래스: 초정·공정 둘)."""
    band = _band_mask(target, band_px)
    weight = torch.ones_like(target, dtype=torch.float32)
    is_si = torch.zeros_like(target, dtype=torch.bool)
    for c in si_classes:
        is_si |= target == c
    weight[is_si & ~band] = w
    return weight


def build_loss(name, n_classes, class_weights=None, ignore_band_px=0, loss_scope="all", head="full"):
    """name: "ce" | "ce_dice" | "lovasz" | "ce_interior".
    범위 si는 정답이 Primary/Eutectic인 픽셀만 손실에 넣고 나머지는 IGNORE로 칠한다.
    si2 헤드는 2채널 logits를 받아 정답 2→0, 3→1, 나머지를 IGNORE로 바꾼다.
    경계 띠는 재매핑 전 4클래스 원본 경계에서 계산한다. Si 없는 배치는 0 손실이다.
    반환: callable(logits [B,C,H,W], target [B,H,W]) -> 스칼라 텐서"""
    if loss_scope not in LOSS_SCOPES:
        raise ValueError(f"알 수 없는 손실 범위: {loss_scope}")
    if loss_scope == "si" and n_classes != 4:
        raise ValueError("Si 손실 범위는 4클래스 전용이다")
    if loss_scope == "si" and name not in ("ce", "ce_dice", "lovasz", "ce_interior"):
        raise ValueError(f"알 수 없는 손실 이름: {name}")
    if head not in OUTPUT_HEADS:
        raise ValueError(f"알 수 없는 출력 헤드: {head}")
    if head == "si2" and n_classes != 4:
        raise ValueError("si2 헤드는 4클래스 정답 전용이다")
    if head == "si2" and loss_scope != "si":
        raise ValueError("si2 헤드는 Si 손실 범위와만 쓴다")
    channels = output_channels(n_classes, head)
    if head == "si2" and class_weights is not None and len(class_weights) != channels:
        raise ValueError("si2 클래스 가중치는 출력 채널 수 2와 같아야 한다")
    cw = None if class_weights is None else torch.tensor(class_weights, dtype=torch.float32)
    # ignore_band_px와 무관하게 항상 무시 인덱스를 건넨다. 증강 여백(255)이 언제나 들어올 수
    # 있으므로 조건부로 켜면 Dice·Lovasz가 255를 실제 클래스로 읽어 학습이 망가진다.
    dice = DiceLoss("multiclass", ignore_index=IGNORE)
    lov = LovaszLoss("multiclass", ignore_index=IGNORE)
    si_classes = _si_classes(n_classes)

    def _prep(target):
        """경계 띠를 먼저 무시한 뒤 Si 외 픽셀을 무시한다. 원본은 보존한다."""
        if ignore_band_px or loss_scope == "si":
            target = target.clone()
        if ignore_band_px:
            target[_band_mask(target, ignore_band_px)] = IGNORE
        if loss_scope == "si":
            keep = (target == si_classes[0]) | (target == si_classes[1])
            target[~keep] = IGNORE
        if head == "si2":
            target = remap_target_si2(target)
        return target

    def fn(logits, target):
        if logits.shape[1] != channels:
            raise ValueError(f"logits 채널 {logits.shape[1]} ≠ n_classes 출력 채널 {channels}")
        t = _prep(target)
        if loss_scope == "si" and not (t != IGNORE).any():
            return logits.sum() * 0.0
        w = None if cw is None else cw.to(logits.device)
        if name == "ce":
            return F.cross_entropy(logits, t, weight=w, ignore_index=IGNORE)
        if name == "ce_dice":
            ce = F.cross_entropy(logits, t, weight=w, ignore_index=IGNORE)
            return CE_DICE_RATIO * ce + (1 - CE_DICE_RATIO) * dice(logits, t)
        if name == "lovasz":
            return lov(logits, t)
        if name == "ce_interior":
            per = F.cross_entropy(logits, t, weight=w, ignore_index=IGNORE, reduction="none")
            wt = _interior_weight(target, si_classes)
            valid = (t != IGNORE).float()
            # class_weights와 내부 가중은 곱으로 겹친다: 클래스 가중 c, 내부 가중 w인 픽셀의
            # 실효 가중은 c*w다. 정규화 분모는 내부 가중만 반영하므로(F.cross_entropy가
            # 분자에 이미 c를 곱해 둔다) 두 가중을 같이 켜면 Si 내부가 의도보다 더 세진다.
            return (per * wt * valid).sum() / (wt * valid).sum().clamp(min=1)
        raise ValueError(f"알 수 없는 손실 이름: {name}")

    return fn
