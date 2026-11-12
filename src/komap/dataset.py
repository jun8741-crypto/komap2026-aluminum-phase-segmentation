"""학습·추론용 데이터셋. 광도 증강 → 앵커 정규화 → 선택적 앵커 흔들기 → 배율 리샘플
→ 기하 증강 순서를 지킨다.

순서가 중요한 이유: 앵커 정규화는 이미지의 밝기 분포를 (50,140,226)에 맞추는 연산이라
기존 광도 증강은 정규화 '이전' 원본 회색조에만 건다. 선택 플래그를 켠 경우에만 정규화가
맞춘 세 앵커를 별도 난수열로 흔들고, 기본값에서는 이 경로를 전혀 실행하지 않는다.
"""
import albumentations as A
import numpy as np
import torch
from torch.utils.data import Dataset

from komap import hint as Hn
from komap import io
from komap import ridge as Rg
from komap.augment import (
    PHOTOMETRIC_NAMES,
    anchor_jitter as jitter_anchors,
    build_photometric,
    build_train_aug,
    build_valid_aug,
)
from komap.config import SEED
from komap.preprocess import effective_factor, normalize, resample

# 정규화 이전 원본에 거는 광도 증강. 범위는 augment 모듈 한 곳에서만 정의한다.
# seed를 주는 이유: albumentations 2.x의 Compose는 전역 np.random/random이 아니라 자체 난수를
# 쓴다. 시드가 없으면 train.seed_all을 불러도 광도 증강이 실행마다 달라져 학습이 재현되지 않는다.
_PHOTO = A.Compose(build_photometric(), seed=SEED)


def _geo_only(tile: int, train: bool, with_hint: bool = False, with_ridge: bool = False,
              interp_geo: bool = True) -> A.Compose:
    """광도 변환을 걷어낸 기하 전용 파이프라인. 광도는 _PHOTO가 정규화 전에 이미 적용했다."""
    aug = build_train_aug(tile, interp_geo=interp_geo) if train else build_valid_aug(tile)
    ts = [t for t in aug.transforms if t.__class__.__name__ not in PHOTOMETRIC_NAMES]
    extra = {}
    # 힌트는 마스크와 같은 규칙(최근접 보간·여백 IGNORE_LABEL)으로 변환해야 회색조와 어긋나지 않는다
    if with_hint:
        extra["hint"] = "mask"
    # 능선은 라벨이 아니라 연속값 지도다. 최근접 보간을 쓰면 얇은 판의 응답이 끊기므로
    # 회색조와 똑같이 '이미지'로 태워 선형 보간을 받게 한다.
    if with_ridge:
        extra["ridge"] = "image"
    # seed는 build_train_aug가 Compose에 걸어 둔 것이다. 변환 목록만 옮기고 seed를 흘리면
    # 증강이 매 실행 달라져 학습·테스트 재현성이 깨진다(이 태스크 이전부터 있던 누락).
    return A.Compose(ts, additional_targets=extra or None, seed=SEED)


def _ridge_u8(gray_norm: np.ndarray) -> np.ndarray:
    """능선 응답(0~1) → uint8 0~255. 회색조와 같은 형식으로 맞춰 리샘플·기하 증강을
    똑같은 경로(선형 보간)로 통과시키기 위해서다. 8비트 정수화(quantization) 손실은 모델 입력에 무해하다."""
    return np.clip(np.round(Rg.ridge_response(gray_norm) * 255.0), 0, 255).astype(np.uint8)


def _u8_to_ch(a: np.ndarray) -> np.ndarray:
    """uint8 [H,W] → float32 [1,H,W] in [0,1]."""
    return a.astype(np.float32)[None] / 255.0


def load_full(image_path, n_classes: int = 3, hint_thr=None, resample_scale: float = 1.0,
              ridge: bool = False, norm: str = "anchor3", size_correction: bool = True):
    """추론용 전체 이미지. 정규화·리샘플만 적용하고 크롭하지 않는다.
    반환: (x uint8 HxW | float32 [C,H,W], y int8 HxW | None, meta{width, height, factor})

    hint_thr=(t1,t2)면 규칙 지도 힌트 4채널을, ridge=True면 능선 1채널을 붙인다.
    채널 순서는 [회색조, 힌트 4, 능선 1]로 고정한다.
    resample_scale은 폭별 기본 배수에 공통으로 얹는 전역 배율이다(1.0이면 현행 그대로)."""
    gray = io.load_gray(image_path)
    h, w = gray.shape
    f = effective_factor(w, scale=resample_scale, size_correction=size_correction)
    gn = normalize(gray, norm)
    x = resample(gn, f, is_mask=False)
    if hint_thr is not None or ridge:
        chans = [_u8_to_ch(x)]
        if hint_thr is not None:
            # 힌트는 원본 해상도에서 만든다(두께 공식이 원본 폭을 쓴다) → 그 뒤에 같은 배율로 리샘플
            chans.append(Hn.onehot(resample(Hn.rule_hint(gn, *hint_thr, width=w), f, is_mask=True)))
        if ridge:
            # 능선도 원본 해상도의 정규화 회색조에서 만들고 이미지 규칙으로 같이 리샘플한다.
            chans.append(_u8_to_ch(resample(_ridge_u8(gn), f, is_mask=False)))
        x = np.concatenate(chans)
    mp = io.mask_path_for(image_path)
    y = None
    if mp.exists():
        y = io.load_mask(mp)
        if n_classes == 3:
            y = io.to3(y)
        y = resample(y, f, is_mask=True)
    return x, y, {"width": w, "height": h, "factor": f}


class SegDataset(Dataset):
    """이미지 한 장에서 samples_per_image번 무작위 타일을 뽑는 학습용 데이터셋.
    __getitem__ → (x float32 [1,tile,tile] in [0,1], y int64 [tile,tile])
    hint_thr=(t1,t2)면 x는 [5,tile,tile](회색조 + 규칙 지도 원-핫 4채널)이 되고,
    ridge=True면 능선 1채널이 뒤에 더 붙는다(순서는 [회색조, 힌트 4, 능선 1])."""

    def __init__(self, image_paths, tile, train, n_classes=3, samples_per_image=8, hint_thr=None,
                 resample_scale: float = 1.0, ridge: bool = False, anchor_jitter: float = 0.0,
                 interp_geo: bool = True, norm: str = "anchor3", size_correction: bool = True,
                 photometric: str = "measured"):
        self.paths = list(image_paths)
        self.tile = tile
        self.train = train
        self.n_classes = n_classes
        self.spi = samples_per_image if train else 1
        self.hint_thr = hint_thr
        self.resample_scale = resample_scale
        self.ridge = ridge
        self.anchor_jitter = anchor_jitter
        self.norm = norm
        self.size_correction = size_correction
        self.photo = (
            _PHOTO if photometric == "measured"
            else A.Compose(build_photometric(photometric), seed=SEED)
        )
        self.geo = _geo_only(tile, train, with_hint=hint_thr is not None, with_ridge=ridge,
                             interp_geo=interp_geo)
        self.rng = np.random.default_rng(SEED)      # 힌트 dropout 전용(재현성)
        self.anchor_jitter_rng = np.random.default_rng(SEED + 1)
        self.cache = {}

    def __len__(self):
        return len(self.paths) * self.spi

    def _base(self, i):
        """원본 회색조와 마스크를 메모리에 캐시한다(정규화 전 상태로 둬야 광도 증강이 매번 달라진다)."""
        if i not in self.cache:
            p = self.paths[i]
            gray = io.load_gray(p)
            y = io.load_mask(io.mask_path_for(p))
            if self.n_classes == 3:
                y = io.to3(y)
            self.cache[i] = (gray, y)
        return self.cache[i]

    def __getitem__(self, idx):
        gray, y = self._base(idx % len(self.paths))
        if self.train:
            gray = self.photo(image=gray)["image"]
        f = effective_factor(
            gray.shape[1], scale=self.resample_scale, size_correction=self.size_correction
        )
        gn = normalize(gray, self.norm)
        if self.train and self.anchor_jitter > 0.0:
            if self.anchor_jitter_rng.random() < self.anchor_jitter:
                gn = jitter_anchors(gn, self.anchor_jitter_rng)
        x = resample(gn, f, is_mask=False)
        # 마스크는 io에서 int8로 온다(미매핑 색을 -1로 잡기 위해). 그대로 두면 증강 여백에
        # 칠할 IGNORE_LABEL(255)이 int8 범위를 넘어 127로 포화되어 엉뚱한 라벨이 된다.
        yy = resample(y, f, is_mask=True).astype(np.uint8)
        extra = {}
        if self.hint_thr is not None:
            # 힌트는 정규화된 '원본 해상도'에서 만든다(두께 공식이 원본 폭을 쓴다). 광도 증강이
            # 걸린 gray로 만들므로 힌트도 그만큼 흔들린다 — 추론 때 규칙 지도의 오류를 닮게 하는 효과.
            extra["hint"] = resample(Hn.rule_hint(gn, *self.hint_thr, width=gray.shape[1]),
                                     f, is_mask=True).astype(np.uint8)
        if self.ridge:
            # 능선도 정규화된 원본 해상도에서 만든 뒤 회색조와 같은 배율·같은 보간으로 리샘플한다.
            extra["ridge"] = resample(_ridge_u8(gn), f, is_mask=False)
        out = self.geo(image=x, mask=yy, **extra)
        if self.hint_thr is None:
            xt = torch.from_numpy(out["image"].astype(np.float32) / 255.0)[None]
        else:
            # 기하 증강 여백은 힌트에도 IGNORE_LABEL(255)로 칠해진다. 원-핫 4채널에 그 자리가
            # 없으므로 '힌트 없음'을 뜻하는 0벡터로 둔다 — onehot이 0~3 밖의 값을 이미 그렇게 만든다.
            xt = torch.from_numpy(np.concatenate([_u8_to_ch(out["image"]), Hn.onehot(out["hint"])]))
            if self.train:
                xt = Hn.drop(xt, Hn.HINT_DROP_P, self.rng)
        if self.ridge:
            # 힌트 dropout은 규칙 지도의 오류를 모델이 그대로 베끼지 않게 맹신을 끊는 장치다.
            # 능선은 규칙 파이프라인이 아니라 회색조에서 직접 계산한 결정적 특징이라 베낄 오류가
            # 없으므로 지울 이유가 없다. drop을 마친 뒤에 붙여 채널 순서 [회색조, 힌트 4, 능선 1]을 지킨다.
            xt = torch.cat([xt, torch.from_numpy(_u8_to_ch(out["ridge"]))])
        yt = torch.from_numpy(out["mask"].astype(np.int64))
        return xt, yt
