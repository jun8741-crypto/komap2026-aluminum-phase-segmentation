"""Si 연결성분의 특징을 만들고 Primary/Eutectic 이진 분류기를 학습한다."""
import pickle
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from komap.config import CLS_EUTSI, CLS_PRISI, CLS_SI3, SEED, THICKNESS_K
from komap.preprocess import group_of

FEATURE_NAMES = (
    "norm_thickness",
    "log_area",
    "elongation",
    "p_primary",
    "p_si3",
    "thickness_std",
    "perimeter_ratio",
)
FEATURE_VERSION = 3
RECLASS_FIT_COMMAND = ".venv/bin/python -m komap.experiments.reclass_fit"


def blob_feature_table(prob3, prob4, lab, thick, width, k=None):
    """라벨 1..n 순서로 Si 덩어리 특징 배열을 만든다."""
    n_components = int(lab.max())
    if n_components == 0:
        return {name: np.zeros(0, dtype=np.float64) for name in FEATURE_NAMES}
    if thick.shape != (n_components,):
        raise ValueError("thick 길이는 lab의 덩어리 수와 같아야 한다")
    if prob4.shape[0] != 4:
        raise ValueError(f"특징 추출에는 4채널 확률이 필요하다(받은 채널 {prob4.shape[0]})")

    indices = np.arange(1, n_components + 1)
    si_mask = lab > 0
    area = np.asarray(ndi.sum(np.ones(lab.shape), lab, indices), dtype=np.float64)
    mean_primary = np.asarray(ndi.mean(prob4[CLS_PRISI], lab, indices))
    mean_eutectic = np.asarray(ndi.mean(prob4[CLS_EUTSI], lab, indices))
    p_primary = mean_primary / np.maximum(mean_primary + mean_eutectic, 1e-9)
    p_si3 = np.asarray(ndi.mean(prob3[CLS_SI3], lab, indices))

    distance = ndi.distance_transform_edt(si_mask)
    thickness_std = 2.0 * np.asarray(ndi.standard_deviation(distance, lab, indices))
    boundary = si_mask & ~ndi.binary_erosion(si_mask)
    perimeter = np.asarray(ndi.sum(boundary, lab, indices), dtype=np.float64)
    k = THICKNESS_K if k is None else k
    threshold = width / k[group_of(width)]
    norm_thickness = thick / threshold

    values = (
        norm_thickness,
        np.log1p(area),
        area / np.maximum(thick, 1.0) ** 2,
        p_primary,
        p_si3,
        thickness_std,
        perimeter / np.maximum(area, 1.0),
    )
    return {name: np.asarray(value, dtype=np.float64) for name, value in zip(FEATURE_NAMES, values, strict=True)}


def majority_target(lab, target):
    """덩어리별 정답 다수결과 동률·정답 Si 없음 제외 마스크를 반환한다."""
    n_components = int(lab.max())
    if n_components == 0:
        empty = np.zeros(0, dtype=bool)
        return empty, empty
    primary = np.bincount(lab[target == CLS_PRISI], minlength=n_components + 1)[1:]
    eutectic = np.bincount(lab[target == CLS_EUTSI], minlength=n_components + 1)[1:]
    has_truth_si = (primary + eutectic) > 0
    usable = has_truth_si & (primary != eutectic)
    return primary > eutectic, usable


def feature_matrix(table):
    """FEATURE_NAMES 순서로 특징 표를 2차원 행렬로 변환한다."""
    if not FEATURE_NAMES:
        return np.empty((0, 0), dtype=np.float64)
    return np.column_stack([table[name] for name in FEATURE_NAMES])


class ReclassModel:
    """scikit-learn 이진 분류기의 학습·저장·임계값 예측 래퍼."""

    def __init__(self, name="logreg", threshold=0.5):
        if name == "logreg":
            estimator = make_pipeline(
                StandardScaler(), LogisticRegression(random_state=SEED, max_iter=2000)
            )
        elif name == "hgb":
            estimator = HistGradientBoostingClassifier(random_state=SEED)
        else:
            raise ValueError(f"알 수 없는 재분류 모델: {name}")
        self.name = name
        self.estimator = estimator
        self.constant_probability = None
        self.threshold = float(threshold)
        self.feature_version = FEATURE_VERSION
        self.feature_names = FEATURE_NAMES

    def fit(self, x, y):
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=bool)
        if x.ndim != 2 or x.shape[1] != len(FEATURE_NAMES):
            raise ValueError(f"X는 (n, {len(FEATURE_NAMES)}) 모양이어야 한다")
        if x.shape[0] != y.size or y.size == 0:
            raise ValueError("X와 y는 같은 양의 표본 수를 가져야 한다")
        classes = np.unique(y)
        if classes.size == 1:
            self.constant_probability = float(classes[0])
        else:
            self.estimator.fit(x, y)
            self.constant_probability = None
        return self

    def predict_proba(self, x):
        x = np.asarray(x, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != len(FEATURE_NAMES):
            raise ValueError(f"X는 (n, {len(FEATURE_NAMES)}) 모양이어야 한다")
        if x.shape[0] == 0:
            return np.zeros(0, dtype=np.float64)
        if self.constant_probability is not None:
            return np.full(x.shape[0], self.constant_probability, dtype=np.float64)
        return self.estimator.predict_proba(x)[:, 1]

    def predict(self, x, threshold=None):
        selected = self.threshold if threshold is None else threshold
        return self.predict_proba(x) > selected

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            pickle.dump(self, handle)

    @classmethod
    def load(cls, path):
        with Path(path).open("rb") as handle:
            model = pickle.load(handle)
        if not isinstance(model, cls):
            raise TypeError(f"저장 파일이 {cls.__name__} 모델이 아니다")
        saved_version = getattr(model, "feature_version", 1)
        if saved_version != FEATURE_VERSION:
            raise ValueError(
                f"재분류 모델 특징 버전이 맞지 않는다(저장 버전 {saved_version}, "
                f"현재 버전 {FEATURE_VERSION}). 다음 명령으로 재생성하라: "
                f"{RECLASS_FIT_COMMAND}"
            )
        saved_names = tuple(getattr(model, "feature_names", ()))
        if saved_names != FEATURE_NAMES:
            max_length = max(len(saved_names), len(FEATURE_NAMES))
            differences = []
            for index in range(max_length):
                saved = saved_names[index] if index < len(saved_names) else "<없음>"
                current = FEATURE_NAMES[index] if index < len(FEATURE_NAMES) else "<없음>"
                if saved != current:
                    differences.append(
                        f"인덱스 {index}: 저장={saved!r}, 현재={current!r}"
                    )
            raise ValueError(
                "재분류 모델 특징 이름 또는 순서가 맞지 않는다(" + "; ".join(differences)
                + f"). 다음 명령으로 재생성하라: {RECLASS_FIT_COMMAND}"
            )
        if not hasattr(model, "threshold"):
            raise ValueError("저장 모델에 결정 임계값이 없다. reclass_fit을 다시 실행해야 한다")
        return model
