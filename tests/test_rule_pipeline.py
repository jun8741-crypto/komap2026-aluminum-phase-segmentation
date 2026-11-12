"""규칙 파이프라인 테스트. 합성 데이터로 동작을 보고, 실데이터로 재현 점수를 지킨다."""
import numpy as np
import pytest

from komap.baselines import rule_pipeline as R
from komap.config import DATA_DIR


def test_segment3_orders_classes():
    g = np.array([[10, 120, 230]], np.uint8)
    assert R.segment3(g, 100, 180).tolist() == [[2, 1, 0]]


def test_fit_thresholds_on_synthetic(synth_pair):
    gray, y4, _ = synth_pair
    from komap.preprocess import anchor_normalize
    from komap.io import to3
    t1, t2 = R.fit_thresholds([(anchor_normalize(gray), to3(y4))])
    assert 50 < t1 < 140 < t2 < 226


@pytest.mark.skipif(not (DATA_DIR / "2valid").exists(), reason="실데이터 없음")
def test_reproduces_71_percent():
    r = R.run("valid")
    assert r["miou"] >= 70.5, r   # 분석 단계 실측 71.0%
