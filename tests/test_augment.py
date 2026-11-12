"""증강 테스트: 기하 변환이 마스크 라벨을 보존하고 타일 크기를 맞추는지 본다."""
import os

import numpy as np

from komap import augment as A


def test_anchor_jitter_preserves_shape_dtype_and_range():
    gray = np.arange(256, dtype=np.uint8).reshape(16, 16)
    out = A.anchor_jitter(gray, np.random.default_rng(0))

    assert out.shape == gray.shape
    assert out.dtype == np.uint8
    assert int(out.min()) >= 0 and int(out.max()) <= 255


def test_anchor_jitter_is_monotonic_for_many_seeds():
    ramp = np.arange(256, dtype=np.uint8)

    for seed in range(200):
        out = A.anchor_jitter(ramp, np.random.default_rng(seed))
        assert np.all(np.diff(out.astype(np.int16)) >= 0)


def test_anchor_jitter_returns_identity_for_too_small_gap(monkeypatch):
    ramp = np.arange(256, dtype=np.uint8)
    monkeypatch.setattr(A, "ANCHOR_JITTER_DARK_RANGE", (0.0, 0.0))
    monkeypatch.setattr(A, "ANCHOR_JITTER_MID_RANGE", (100.0, 100.0))
    monkeypatch.setattr(A, "ANCHOR_JITTER_BRIGHT_RANGE", (0.0, 0.0))

    out = A.anchor_jitter(ramp, np.random.default_rng(0))

    assert np.array_equal(out, ramp)


def test_train_aug_without_interp_geo_keeps_only_non_interpolating_geometry():
    names = [t.__class__.__name__ for t in A.build_train_aug(128, interp_geo=False).transforms]

    assert "Affine" not in names
    assert "ElasticTransform" not in names
    assert {"RandomRotate90", "HorizontalFlip", "VerticalFlip", "RandomCrop"} <= set(names)


def test_train_aug_default_transform_list_is_unchanged():
    names = [t.__class__.__name__ for t in A.build_train_aug(128).transforms]

    assert names == [
        "PadIfNeeded",
        "RandomCrop",
        "HorizontalFlip",
        "VerticalFlip",
        "RandomRotate90",
        "Affine",
        "ElasticTransform",
        "RandomGamma",
        "RandomBrightnessContrast",
        "GaussNoise",
        "GaussianBlur",
    ]


def test_train_aug_keeps_mask_labels_and_shape(synth_pair):
    gray, y4, _ = synth_pair
    aug = A.build_train_aug(tile=128)
    for _ in range(5):
        out = aug(image=gray, mask=y4)
        assert out["image"].shape == (128, 128) and out["mask"].shape == (128, 128)
        # 255는 기하 변환 여백(IGNORE)이라 정상이다. test_geometric_fill_uses_ignore_label 참고.
        assert set(np.unique(out["mask"])) <= {0, 1, 2, 3, 255}
        assert out["image"].dtype == np.uint8


def test_valid_aug_is_crop_only(synth_pair):
    gray, y4, _ = synth_pair
    out = A.build_valid_aug(tile=128)(image=gray, mask=y4)
    assert out["image"].shape == (128, 128)


def test_albumentations_update_check_disabled():
    """패키지 초기화가 albumentations 버전 확인용 네트워크 호출을 꺼야 한다."""
    import komap  # noqa: F401

    assert os.environ.get("NO_ALBUMENTATIONS_UPDATE") == "1"


def test_geometric_fill_uses_ignore_label(synth_pair):
    """기하 변환이 만든 여백은 마스크에서 IGNORE(255)여야 한다.

    여백을 0으로 채우면 0 == CLS_AL이라 '여기는 알루미늄'이라는 가짜 정답이 생긴다.
    광도 변환은 0을 0이 아닌 값으로 밀어내므로(밝기 -0.15 오프셋 등) 여백을 정확히
    식별하기 위해 기하 변환만 남긴 파이프라인으로 확인한다.
    """
    import albumentations as Alb

    from komap.config import IGNORE_LABEL

    gray, y4, _ = synth_pair
    full = A.build_train_aug(tile=64)
    geo = Alb.Compose([t for t in full.transforms if t.__class__.__name__ not in A.PHOTOMETRIC_NAMES])

    saw_fill = False
    for _ in range(30):
        out = geo(image=gray, mask=y4)
        m = out["mask"]
        assert set(np.unique(m)) <= {0, 1, 2, 3, IGNORE_LABEL}
        fill = out["image"] == 0
        assert np.all(m[fill] == IGNORE_LABEL)   # 여백 픽셀이 0(Al)으로 날조되면 안 된다
        saw_fill |= bool(fill.any())
    assert saw_fill, "30회 중 여백이 한 번도 생기지 않아 검증이 무의미하다"


def test_full_train_aug_labels_stay_in_known_set(synth_pair):
    """광도 변환까지 포함한 전체 파이프라인에서도 마스크 라벨 집합은 {0..3, 255}뿐이다."""
    from komap.config import IGNORE_LABEL

    gray, y4, _ = synth_pair
    aug = A.build_train_aug(tile=64)
    for _ in range(10):
        out = aug(image=gray, mask=y4)
        assert set(np.unique(out["mask"])) <= {0, 1, 2, 3, IGNORE_LABEL}


def test_anchor_jitter_keeps_minimum_gap_between_output_anchors():
    """출력 앵커의 이웃 간격이 기준보다 좁으면 항등을 돌려줘야 한다.

    단조성만 검사하면 이 보호 장치가 사라져도 통과한다. 간격이 좁아지면 밝기 순서가
    수치적으로는 유지되어도 상끼리 구분이 뭉개지므로, 간격 자체를 직접 본다.
    실제로 ANCHOR_JITTER_MIN_GAP을 0으로 바꾸면 이 테스트만 실패한다.
    """
    # 기준값을 A.ANCHOR_JITTER_MIN_GAP에서 읽지 않고 직접 적는다. 상수를 읽으면 그 상수를
    # 0으로 바꿨을 때 검사 기준도 함께 0이 되어 아무것도 막지 못한다. 간격 8은 설계 결정이다.
    REQUIRED_GAP = 8.0
    assert A.ANCHOR_JITTER_MIN_GAP == REQUIRED_GAP, "설계값이 바뀌었다면 이 테스트도 함께 검토한다"

    ramp = np.arange(256, dtype=np.uint8)
    jittered = 0
    for seed in range(300):
        out = A.anchor_jitter(ramp, np.random.default_rng(seed))
        if np.array_equal(out, ramp):
            continue
        jittered += 1
        # 입력이 0~255 램프라 출력의 고정점 값이 곧 그 앵커의 출력 위치다.
        ys = [float(out[int(x)]) for x in (0.0, 50.0, 140.0, 226.0, 255.0)]
        gaps = np.diff(ys)
        assert gaps.min() >= REQUIRED_GAP - 1.0, (
            f"seed {seed}: 최소 간격 {gaps.min():.2f} < 기준 {REQUIRED_GAP}"
        )
    # 항등만 돌려주면 위 단언이 한 번도 실행되지 않으므로, 실제 변형이 일어났는지 확인한다.
    # 나아가 변형이 드물면 증강이 사실상 꺼진 것이다. 흔들림 폭을 넓히면 간격 보호가 자주
    # 걸려 항등이 늘어나는데(하한 -120에서 30%), 그 상태는 설정만 켜 두고 효과는 없는 함정이다.
    assert jittered >= 285, (
        f"300개 시드 중 {jittered}개만 실제로 흔들렸다 — 흔들림 폭이 간격 기준에 너무 자주 걸린다"
    )


def test_anchor_jitter_fixed_points_follow_module_constants():
    """재매핑의 고정점이 모듈 상수를 따라야 한다.

    앵커 목표값(50·140·226)은 전처리의 정규화 기준과 같은 값이라 임의로 바꾸면 증강이
    정규화와 어긋난다. 상수를 고쳤을 때 이 테스트가 잡도록 값 자체를 검사한다.
    """
    assert A.ANCHOR_JITTER_XS == (0.0, 50.0, 140.0, 226.0, 255.0)

    ramp = np.arange(256, dtype=np.uint8)
    # 흔들림 폭을 0으로 고정하면 재매핑이 항등이 되어야 한다. 고정점이 상수와 다르면
    # 구간 선형 보간이 어긋나 항등이 나오지 않는다.
    import pytest as _pytest

    with _pytest.MonkeyPatch.context() as mp:
        mp.setattr(A, "ANCHOR_JITTER_DARK_RANGE", (0.0, 0.0))
        mp.setattr(A, "ANCHOR_JITTER_MID_RANGE", (0.0, 0.0))
        mp.setattr(A, "ANCHOR_JITTER_BRIGHT_RANGE", (0.0, 0.0))
        out = A.anchor_jitter(ramp, np.random.default_rng(0))
    assert np.array_equal(out, ramp), "폭 0인데 항등이 아니다 — 고정점이 상수와 어긋났다"
