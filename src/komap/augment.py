"""증강. 기하는 이미지·마스크 동시, 광도는 이미지만.
범위는 실측(감마 0.71~1.36, Si 내부 23~108)에서 정했다.

albumentations 2.x 인자명 주의 (설치 2.0.8 기준):
- PadIfNeeded/Affine의 `value`·`mask_value`는 `fill`·`fill_mask`로 바뀌었고 `mode`는 `border_mode`다.
- GaussNoise는 분산(var_limit)이 아니라 0~1로 정규화된 표준편차 범위(std_range)를 받는다.
  계획의 분산 5.0~40.0 → 표준편차 √5~√40 (0~255 스케일) → 255로 나눠 GAUSS_STD_RANGE.
  이름만 바꾸면 노이즈 세기가 달라지므로 단위까지 환산한다.

여백 라벨 주의: 기하 변환이 만든 빈 여백을 마스크에서 0으로 채우면 0 == CLS_AL이라
'여기는 알루미늄'이라는 가짜 정답이 생긴다. 마스크 여백은 반드시 IGNORE_LABEL(255)로
칠해 손실에서 빠지게 한다. 이미지 여백은 그대로 0이다(검은 배경).
"""
import albumentations as A
import numpy as np

from komap.config import IGNORE_LABEL, SEED

# 광도 증강 범위(계획 실측값). dataset이 정규화 전 원본에 적용할 때도 같은 값을 쓴다.
GAMMA_LIMIT = (71, 136)          # 감마 γ 0.71~1.36 (albumentations는 %로 받음)
BRIGHTNESS_LIMIT = 0.15
CONTRAST_LIMIT = 0.25
GAUSS_VAR_LIMIT = (5.0, 40.0)    # 0~255 스케일 분산
GAUSS_STD_RANGE = (GAUSS_VAR_LIMIT[0] ** 0.5 / 255.0, GAUSS_VAR_LIMIT[1] ** 0.5 / 255.0)
BLUR_LIMIT = (3, 3)

# 광도 변환 클래스 이름. dataset이 기하 전용 파이프라인을 만들 때 이 목록으로 걸러낸다.
PHOTOMETRIC_NAMES = ("RandomGamma", "RandomBrightnessContrast", "GaussNoise", "GaussianBlur")
PHOTOMETRIC_KINDS = ("measured", "default")

# 앵커 정규화 뒤 밝기 재매핑에 쓰는 고정점과 변위 범위다.
ANCHOR_JITTER_XS = (0.0, 50.0, 140.0, 226.0, 255.0)
ANCHOR_JITTER_DARK_RANGE = (-10.0, 25.0)
ANCHOR_JITTER_MID_RANGE = (-45.0, 15.0)
ANCHOR_JITTER_BRIGHT_RANGE = (-8.0, 4.0)
ANCHOR_JITTER_MIN_GAP = 8.0


def anchor_jitter(gray: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """정규화 앵커를 단조 구간 선형 재매핑한다.

    뽑힌 출력 앵커의 이웃 간격이 너무 좁으면 밝기 순서가 불안정해지므로 항등을 반환한다.
    """
    ys = np.array([
        0.0,
        50.0 + rng.uniform(*ANCHOR_JITTER_DARK_RANGE),
        140.0 + rng.uniform(*ANCHOR_JITTER_MID_RANGE),
        226.0 + rng.uniform(*ANCHOR_JITTER_BRIGHT_RANGE),
        255.0,
    ])
    if np.any(np.diff(ys) < ANCHOR_JITTER_MIN_GAP):
        return gray.copy()
    mapped = np.interp(gray, ANCHOR_JITTER_XS, ys)
    return np.clip(np.round(mapped), 0, 255).astype(np.uint8)


def build_photometric(kind: str = "measured") -> list:
    """선택한 광도 변환 목록을 만든다. 모든 값은 버전과 무관하도록 명시한다."""
    if kind == "measured":
        return [
            A.RandomGamma(gamma_limit=GAMMA_LIMIT, p=0.7),
            A.RandomBrightnessContrast(
                brightness_limit=BRIGHTNESS_LIMIT, contrast_limit=CONTRAST_LIMIT, p=0.7
            ),
            A.GaussNoise(std_range=GAUSS_STD_RANGE, p=0.3),
            A.GaussianBlur(blur_limit=BLUR_LIMIT, p=0.15),
        ]
    if kind == "default":
        return [
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.RandomGamma(gamma_limit=(80, 120), p=0.5),
        ]
    raise ValueError(f"알 수 없는 광도 증강 종류: {kind}")


def build_train_aug(tile: int, interp_geo: bool = True,
                    photometric: str = "measured") -> A.Compose:
    """학습용 전체 파이프라인(기하 + 광도). 타일보다 작은 이미지는 먼저 패딩한다."""
    transforms = [
        A.PadIfNeeded(tile, tile, border_mode=0, fill=0, fill_mask=IGNORE_LABEL),
        A.RandomCrop(tile, tile),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
    ]
    if interp_geo:
        transforms.extend([
            A.Affine(scale=(0.8, 1.25), rotate=(-15, 15), border_mode=0,
                     fill=0, fill_mask=IGNORE_LABEL, p=0.5),
            A.ElasticTransform(alpha=20, sigma=6, border_mode=0,
                               fill=0, fill_mask=IGNORE_LABEL, p=0.2),
        ])
    transforms.extend(build_photometric(photometric))
    return A.Compose(transforms, seed=SEED)


def build_valid_aug(tile: int) -> A.Compose:
    """검증용. 중앙 크롭만 한다(광도·기하 무작위 없음)."""
    return A.Compose([
        A.PadIfNeeded(tile, tile, border_mode=0, fill=0, fill_mask=IGNORE_LABEL),
        A.CenterCrop(tile, tile),
    ], seed=SEED)
