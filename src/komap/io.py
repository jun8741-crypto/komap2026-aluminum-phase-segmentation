"""파일명 파싱, 이미지·마스크 입출력, 색상↔클래스 변환."""
import re
from pathlib import Path

import numpy as np
from PIL import Image

from komap.config import CLASS_RGB, DATA_DIR, RGB_TO_CLASS, SPLITS

_NAME = re.compile(r"A(\d+)_step(\d)_sample(\d)_(image|mask)\.png$")


def parse_name(name: str) -> dict:
    m = _NAME.search(Path(name).name)
    if not m:
        raise ValueError(f"파일명 규칙에 맞지 않음: {name}")
    return {"alloy": int(m[1]), "step": int(m[2]), "sample": int(m[3])}


def list_split(split: str) -> list[Path]:
    d = DATA_DIR / SPLITS[split] / "images"
    return sorted(d.glob("*.png"))


def mask_path_for(image_path: Path) -> Path:
    p = Path(image_path)
    return p.parent.parent / "masks" / p.name.replace("_image.png", "_mask.png")


def load_gray(path) -> np.ndarray:
    return np.array(Image.open(path).convert("L"), dtype=np.uint8)


def rgb_to_mask(rgb: np.ndarray) -> np.ndarray:
    y = np.full(rgb.shape[:2], -1, dtype=np.int8)
    for color, k in RGB_TO_CLASS.items():
        y[(rgb == color).all(-1)] = k
    if (y < 0).any():
        bad = np.unique(rgb[y < 0].reshape(-1, 3), axis=0)[:5]
        raise ValueError(f"매핑되지 않은 마스크 색 {bad.tolist()}")
    return y


def load_mask(path) -> np.ndarray:
    return rgb_to_mask(np.array(Image.open(path).convert("RGB")))


def to_rgb(mask4: np.ndarray) -> np.ndarray:
    out = np.zeros((*mask4.shape, 3), np.uint8)
    for k, c in CLASS_RGB.items():
        out[mask4 == k] = c
    return out


def to_rgba(mask4: np.ndarray) -> np.ndarray:
    """제출용 4채널 변환. 대회 정답 마스크가 RGBA(알파 전부 255)라 형식을 그대로 맞춘다.
    채점기가 alpha를 떼지 않고 배열을 비교할 경우 3채널이면 shape가 어긋나기 때문이다."""
    rgb = to_rgb(mask4)
    return np.concatenate([rgb, np.full((*mask4.shape, 1), 255, np.uint8)], axis=-1)


def to3(mask4: np.ndarray) -> np.ndarray:
    y = mask4.copy()
    y[y == 3] = 2
    return y
