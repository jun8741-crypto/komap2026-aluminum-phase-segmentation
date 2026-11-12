"""확률 맵 → 4클래스 대회 마스크 결합기. 스펙 §3의 세 후보를 같은 서명으로 둔다.

- thickness: 3클래스(또는 4클래스를 Si로 합친) argmax → 후광 제거 → 두께 공식. W1 기본 경로.
- direct4:   4클래스 모델의 argmax 그대로. 두께 공식이 틀리는 4~5%를 모델이 배울 수 있는지 본다.
- blob:      4클래스 모델의 Si 덩어리별 평균 확률과 두께 투표를 alpha로 섞는다. 덩어리 단위
             판정은 유지하면서(정답 마스크 순도 99.975%) 모델의 형태 판단을 보탠다.
- shape:     두께로 초정을 고른 뒤, 그중 길쭉한 덩어리를 공정으로 되돌린다. 확률맵의 채널
             수를 가리지 않으므로 3클래스 모델에도 쓸 수 있다(shape.py의 근거 참조).
- learned:   3·4클래스 확률과 덩어리 형태 특징을 학습한 이진 분류기로 이름을 정한다.

모든 결합기는 (prob[C,H,W] 원본 해상도, width, gray_norm, k) 를 받는다. 쓰지 않는 인자도
받아서 grid.py가 이름만 바꿔 호출할 수 있게 한다.
"""
import numpy as np
from scipy import ndimage as ndi

from komap import io
from komap.config import CLS_EUTSI, CLS_PRISI, CLS_SI3, RECLASS_MODEL_PATH, THICKNESS_K
from komap.postprocess import component_thickness, halo_fix, thickness_rule
from komap.preprocess import group_of
from komap.shape import ELONG_MIN, blob_features, classify


def paint_si(pred3, lab, is_primary):
    """덩어리별 Primary 여부(라벨 1..n 순서)를 픽셀로 칠한다.

    덩어리 단위 판정을 픽셀로 옮기는 자리를 한 곳으로 모은다 — 결합기마다 따로 쓰면
    라벨 0(배경) 자리를 채우는 실수가 각각 생긴다."""
    out = pred3.copy()
    if is_primary.size == 0:
        return out
    si = pred3 == CLS_SI3
    out[si] = np.where(np.r_[False, is_primary][lab[si]], CLS_PRISI, CLS_EUTSI)
    return out


def pred3_from_prob(prob, gray_norm):
    """확률맵을 argmax해 3클래스 예측으로 만든다. 4채널이면 Si를 합치고, gray_norm이 있으면 후광을 없앤다."""
    p = prob.argmax(0).astype(np.int64)
    if prob.shape[0] == 4:
        p = io.to3(p)
    if gray_norm is not None:
        p = halo_fix(gray_norm, p)
    return p


def _require4(prob, who):
    if prob.shape[0] != 4:
        raise ValueError(f"{who}: 4채널 확률이 필요하다(받은 채널 {prob.shape[0]})")


def combine_thickness(prob, width, gray_norm=None, k=None):
    return thickness_rule(pred3_from_prob(prob, gray_norm), width=width, k=k)


def combine_direct4(prob, width=None, gray_norm=None, k=None):
    _require4(prob, "direct4")
    return prob.argmax(0).astype(np.int64)


def combine_blob(prob, width, gray_norm=None, k=None, alpha=0.5):
    _require4(prob, "blob")
    k = THICKNESS_K if k is None else k
    thr = width / k[group_of(width)]
    p3 = pred3_from_prob(prob, gray_norm)
    lab, thick = component_thickness(p3 == CLS_SI3)
    n = int(lab.max())
    if n == 0:
        return p3.copy()
    idx = np.arange(1, n + 1)
    # 덩어리별 평균 확률: ndimage.mean은 라벨별 평균을 한 번에 낸다(파이썬 루프 없음).
    m_pri = np.asarray(ndi.mean(prob[CLS_PRISI], lab, idx))
    m_eut = np.asarray(ndi.mean(prob[CLS_EUTSI], lab, idx))
    p_pri = m_pri / np.maximum(m_pri + m_eut, 1e-9)
    vote = (thick > thr).astype(np.float64)
    score = alpha * p_pri + (1 - alpha) * vote
    return paint_si(p3, lab, score > 0.5)


def combine_shape(prob, width, gray_norm=None, k=None, rule="thickness_elong", elong_min=ELONG_MIN):
    """두께로 초정을 고르되 길쭉한 덩어리는 공정으로 되돌린다. rule="thickness"면 기존 공식과 같다."""
    p3 = pred3_from_prob(prob, gray_norm)
    lab, feats = blob_features(p3 == CLS_SI3)
    return paint_si(p3, lab, classify(feats, width=width, k=k, rule=rule, elong_min=elong_min))


def combine_learned(
    prob,
    width,
    gray_norm=None,
    k=None,
    model=None,
    prob4=None,
    threshold=None,
):
    """학습된 덩어리 분류기로 고정된 3클래스 Si 분할에 이름을 칠한다."""
    if prob4 is None:
        raise ValueError("learned: 별도의 4채널 확률이 필요하다")
    _require4(prob4, "learned")
    from komap.reclass import ReclassModel, blob_feature_table, feature_matrix

    p3 = pred3_from_prob(prob, gray_norm)
    lab, thick = component_thickness(p3 == CLS_SI3)
    if thick.size == 0:
        return p3.copy()
    table = blob_feature_table(prob, prob4, lab, thick, width)
    classifier = ReclassModel.load(RECLASS_MODEL_PATH) if model is None else model
    return paint_si(p3, lab, classifier.predict(feature_matrix(table), threshold=threshold))


COMBINERS = {
    "thickness": combine_thickness,
    "direct4": combine_direct4,
    "blob": combine_blob,
    "shape": combine_shape,
    "learned": combine_learned,
}


def combine(name, prob, width, gray_norm=None, k=None, **kw):
    return COMBINERS[name](prob, width=width, gray_norm=gray_norm, k=k, **kw)
