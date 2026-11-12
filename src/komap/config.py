"""프로젝트 전역 상수. 다른 곳에서 숫자를 정의하지 않는다."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
VENV_PYTHON = ROOT / ".venv" / "bin" / "python"
DATA_DIR = ROOT / "data"
RUNS_DIR = ROOT / "runs"
SPLITS = {"train": "1train", "valid": "2valid", "test": "3test"}

# 대회 마스크 색상 ↔ 클래스
CLASS_NAMES = ["Al", "Al3Ni", "PriSi", "EutSi"]
CLASS_RGB = {0: (153, 127, 76), 1: (25, 76, 153), 2: (76, 178, 76), 3: (204, 204, 204)}
RGB_TO_CLASS = {v: k for k, v in CLASS_RGB.items()}
NAMES3 = ["Al", "Al3Ni", "Si"]

# 3점 앵커 정규화 목표 밝기(어두운 군 · 중간 군 · 밝은 군)
ANCHOR_TARGETS = (50, 140, 226)

# Multi-Otsu 3분할의 중간 군이 실재하는지 판정하는 군 분리 거리 기준.
# rel_sep = min(mid-dark, bright-mid) / (bright-dark). 실데이터 100장 최소 0.336(중앙값 0.465) vs
# 2군 합성(두 평탄 블록) 0.027 (2026-09-17 실측). 이 값보다 작으면 중간 군이 진짜가 아니라
# 한쪽 덩어리를 쪼갠 것으로 보고 2점 매핑으로 폴백한다.
ANCHOR_MIN_REL_SEP = 0.15

# 4클래스 대회 마스크의 클래스 id (CLASS_NAMES 순서와 일치)
CLS_AL, CLS_AL3NI, CLS_PRISI, CLS_EUTSI = 0, 1, 2, 3
CLS_SI3 = 2  # 3클래스 작업에서 Si(초정+공정)

# 배율군과 초정/공정 두께 공식: 최대 내접원 지름 > width / k → 초정
GROUP_BY_WIDTH = {675: "A", 1440: "A", 1096: "B", 1230: "B", 1278: "B"}
THICKNESS_K = {"A": 60, "B": 80}
# 배율 정규화: 각 폭의 픽셀 크기(1440px 이미지 = 1.0). 두께 기준값 비에서 도출.
# 출처: docs/research/2026-09-16_정답마스크_생성방식_포렌식.md "픽셀당 실제 크기" 절 — 두께 기준값 비(1440px=24.2px 기준)에서 도출
PIXEL_SCALE = {1440: 1.00, 675: 2.14, 1278: 1.51, 1230: 1.59, 1096: 1.73}

# 손실에서 제외하는 라벨. 증강 여백(회전·패딩으로 생긴 빈 공간)과 경계 띠 무시에 같이 쓴다.
# 0은 CLS_AL이라 여백을 0으로 채우면 가짜 Al 정답이 되므로 반드시 이 값으로 칠한다.
IGNORE_LABEL = 255

SEED = 42
N_FOLDS = 5
BOUNDARY_BAND_PX = 2

# Si 덩어리 학습형 재분류기의 저장 위치와 mIoU 임계값 탐색 격자.
RECLASS_MODEL_PATH = RUNS_DIR / "reclass" / "model.pkl"
RECLASS_THRESHOLDS = tuple(round(0.10 + 0.05 * index, 2) for index in range(17))

# 배율 정규화 방향. "upsample" = 모든 이미지를 1440px 픽셀 크기로 확대(계획 기본) /
# "downsample_to_1440" = A군 675는 그대로 두고 1440과 B군을 675 기준으로 축소
RESAMPLE_MODE = "upsample"
