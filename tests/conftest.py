"""테스트 공용 fixture. 실데이터 없이도 돌아가는 합성 이미지·마스크를 만든다."""
import numpy as np
import pytest

from komap.config import ANCHOR_TARGETS, CLASS_RGB


def make_synth(width=320, height=240, seed=0):
    """밝은 Al 바탕(226)에 중간 밝기 Al3Ni 띠(140)와 어두운 Si 덩어리(50) 두 개(큰 것·작은 것)를 놓는다.
    반환: gray(uint8 HxW), mask4(int HxW, 0~3), mask_rgb(uint8 HxWx3)"""
    rng = np.random.default_rng(seed)
    y4 = np.zeros((height, width), int)
    y4[:, 100:120] = 1                       # Al3Ni 세로 띠 (폭 20)
    y4[40:120, 20:100] = 2                   # 큰 Si 블록 80x80 → 두께 80 > width/60 → 초정
    y4[150:154, 200:260] = 3                 # 얇은 Si 판 4x60 → 두께 4 → 공정
    gray = np.full((height, width), float(ANCHOR_TARGETS[2]))
    gray[y4 == 1] = ANCHOR_TARGETS[1]
    gray[(y4 == 2) | (y4 == 3)] = ANCHOR_TARGETS[0]
    gray += rng.normal(0, 3, gray.shape)
    gray = np.clip(np.round(gray), 0, 255).astype(np.uint8)
    rgb = np.zeros((height, width, 3), np.uint8)
    for k, c in CLASS_RGB.items():
        rgb[y4 == k] = c
    return gray, y4, rgb


@pytest.fixture
def synth_pair():
    return make_synth()
