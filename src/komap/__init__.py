"""komap 패키지 초기화.

albumentations는 임포트되는 순간 최신 버전을 확인하려 네트워크를 친다. 오프라인 실행에서
지연되므로 꺼야 하는데, 환경변수는 albumentations가 임포트되기 '전에' 설정돼 있어야 한다.
dataset 모듈은 import 정렬 규칙상 albumentations를 komap.augment보다 먼저 임포트하므로
augment.py에 두면 때가 늦는다. 모든 경로에서 가장 먼저 실행되는 패키지 초기화에 둔다.
"""
import os

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

# MPS에 아직 구현되지 않은 연산이 나오면 그 연산만 CPU로 흘려보낸다(전체를 CPU로 내리지 않는다).
# torch import 이전에 설정해야 효력이 있는데, train.py에 두면 predict·테스트 등 다른 진입점이
# 놓친다. 모든 경로가 거쳐 가는 패키지 초기화가 유일하게 안전한 자리다.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
