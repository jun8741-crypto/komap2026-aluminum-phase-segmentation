"""Dataset 테스트: 타일 텐서 규격과 전체 이미지 리샘플, 리샘플 모드 두 가지."""
import numpy as np
import pytest
import torch
from PIL import Image

from komap import dataset as D
from komap.io import to_rgb
from komap.preprocess import effective_factor


@pytest.fixture
def tmp_split(tmp_path, synth_pair):
    gray, y4, _ = synth_pair
    (tmp_path / "images").mkdir()
    (tmp_path / "masks").mkdir()
    ip = tmp_path / "images" / "A1_step1_sample1_image.png"
    Image.fromarray(gray).save(ip)
    Image.fromarray(to_rgb(y4)).save(tmp_path / "masks" / "A1_step1_sample1_mask.png")
    return [ip]


def test_item_shapes_and_classes(tmp_split):
    ds = D.SegDataset(tmp_split, tile=128, train=True, n_classes=3, samples_per_image=4)
    assert len(ds) == 4
    x, y = ds[0]
    assert x.shape == (1, 128, 128) and x.dtype == torch.float32 and 0 <= x.min() and x.max() <= 1
    # 255는 기하 증강 여백(IGNORE)이라 정상이다. 클리핑하면 여백이 Al(0)로 둔갑한다.
    assert y.shape == (128, 128) and y.dtype == torch.int64
    assert set(int(v) for v in torch.unique(y)) <= {0, 1, 2, 255}


def test_anchor_jitter_zero_matches_legacy_default(tmp_split, monkeypatch):
    """새 플래그가 꺼지면 기존 기본 호출과 텐서가 완전히 같아야 한다."""
    monkeypatch.setattr(D, "_PHOTO", lambda **kwargs: kwargs)
    monkeypatch.setattr(
        D,
        "jitter_anchors",
        lambda *_args, **_kwargs: pytest.fail("확률 0에서는 앵커 흔들기를 호출하면 안 된다"),
    )
    legacy = D.SegDataset(tmp_split, tile=128, train=True, n_classes=3, samples_per_image=1)
    explicit = D.SegDataset(
        tmp_split, tile=128, train=True, n_classes=3, samples_per_image=1, anchor_jitter=0.0
    )

    x_legacy, y_legacy = legacy[0]
    x_explicit, y_explicit = explicit[0]

    assert np.array_equal(x_legacy.numpy(), x_explicit.numpy())
    assert np.array_equal(y_legacy.numpy(), y_explicit.numpy())


def test_anchor_jitter_one_changes_image_but_not_mask(tmp_split, monkeypatch):
    monkeypatch.setattr(D, "_PHOTO", lambda **kwargs: kwargs)
    base = D.SegDataset(tmp_split, tile=128, train=True, n_classes=3, samples_per_image=1)
    jittered = D.SegDataset(
        tmp_split, tile=128, train=True, n_classes=3, samples_per_image=1, anchor_jitter=1.0
    )

    x_base, y_base = base[0]
    x_jittered, y_jittered = jittered[0]

    assert not np.array_equal(x_base.numpy(), x_jittered.numpy())
    assert np.array_equal(y_base.numpy(), y_jittered.numpy())


def test_load_full_resamples(tmp_split):
    x, y, meta = D.load_full(tmp_split[0], n_classes=3)
    f = meta["factor"]
    assert x.shape == (round(240 * f), round(320 * f)) and y.shape == x.shape


def test_effective_factor_modes():
    """675px는 확대 모드에서 1440 기준으로 2.14배, 축소 모드에서는 기준 자신이라 1.0배."""
    assert effective_factor(675, mode="upsample") == pytest.approx(2.14)
    assert effective_factor(675, mode="downsample_to_1440") == pytest.approx(1.0)


def test_padded_tile_marks_fill_as_ignore(tmp_split):
    """타일이 이미지보다 크면 PadIfNeeded가 여백을 만든다. 그 여백은 타깃에서 255여야 하고,
    255가 아닌 라벨로 클리핑되어서도 안 된다(0은 CLS_AL이므로 가짜 정답이 된다)."""
    import numpy as np

    from komap.config import IGNORE_LABEL

    ds = D.SegDataset(tmp_split, tile=800, train=True, n_classes=3, samples_per_image=1)
    x, y = ds[0]
    yn = y.numpy()
    assert set(np.unique(yn)) <= {0, 1, 2, IGNORE_LABEL}
    assert (yn == IGNORE_LABEL).any(), "패딩이 일어났는데 IGNORE 여백이 없다"
    # 이미지 여백은 0으로 채워진다. 그 자리는 전부 IGNORE여야 한다.
    fill = x[0].numpy() == 0.0
    assert fill.any() and np.all(yn[fill] == IGNORE_LABEL)


def test_dataset_with_hint_returns_5ch_and_hint_aligned(tmp_path, synth_pair, monkeypatch):
    """힌트가 켜지면 x는 [5,tile,tile]이고, 원-핫 합은 여백에서 0, 그 밖에서는 1이다.

    기하 증강(Affine·Elastic)이 만든 여백은 마스크와 힌트에 IGNORE_LABEL로 칠해지고, 원-핫
    4채널에는 그 자리가 없어 0벡터가 된다. 여백이 생기는 표본도 정상이므로 '합이 언제나 1'이
    아니라 '0 아니면 1이고, 여백이 아닌 곳에서는 1'을 계약으로 검사한다."""
    import numpy as np
    import torch
    from PIL import Image
    from komap import hint as Hn
    from komap.config import IGNORE_LABEL
    from komap.dataset import SegDataset, load_full
    from komap.io import to_rgb
    gray, y4, _ = synth_pair
    (tmp_path / "images").mkdir(); (tmp_path / "masks").mkdir()
    ip = tmp_path / "images" / "A1_step1_sample1_image.png"
    Image.fromarray(gray).save(ip); Image.fromarray(to_rgb(y4)).save(tmp_path / "masks" / "A1_step1_sample1_mask.png")
    monkeypatch.setattr(Hn, "HINT_DROP_P", 0.0)
    # 표본을 여러 개 뽑아 여백이 있는 것과 없는 것을 둘 다 지나가게 한다(고정 시드라 결정적이다).
    ds = SegDataset([ip], tile=128, train=True, n_classes=3, samples_per_image=6, hint_thr=(95, 183))
    n_margin = 0
    for i in range(6):
        x, y = ds[i]
        assert x.shape == (5, 128, 128) and y.shape == (128, 128)
        s = x[1:].sum(0)
        assert torch.all((s == 0) | (s == 1))                      # 원-핫(또는 여백의 0벡터)
        assert torch.all(s[y != IGNORE_LABEL] == 1)                # 여백이 아닌 곳에는 힌트가 있다
        if (y == IGNORE_LABEL).any():
            n_margin += 1
            assert (s[y == IGNORE_LABEL] == 0).all()               # 여백에는 힌트가 남지 않는다
    assert n_margin > 0, "여백이 생기는 표본이 하나도 없어 여백 경로가 검증되지 않았다"
    xf, yf, meta = load_full(ip, 3, hint_thr=(95, 183))
    assert xf.dtype == np.float32 and xf.shape[0] == 5 and xf.shape[1:] == yf.shape


def test_dataset_hint_dropout_zeroes_channels(tmp_path, synth_pair, monkeypatch):
    from PIL import Image
    from komap import hint as Hn
    from komap.dataset import SegDataset
    from komap.io import to_rgb
    gray, y4, _ = synth_pair
    (tmp_path / "images").mkdir(); (tmp_path / "masks").mkdir()
    ip = tmp_path / "images" / "A1_step1_sample1_image.png"
    Image.fromarray(gray).save(ip); Image.fromarray(to_rgb(y4)).save(tmp_path / "masks" / "A1_step1_sample1_mask.png")
    monkeypatch.setattr(Hn, "HINT_DROP_P", 1.0)
    x, _ = SegDataset([ip], tile=128, train=True, n_classes=3, samples_per_image=1, hint_thr=(95, 183))[0]
    assert (x[1:] == 0).all() and (x[0] > 0).any()


def test_load_full_resample_scale_changes_size(tmp_path, synth_pair):
    """배수를 2배로 주면 리샘플 해상도가 2배가 되고, 마스크도 같이 커진다."""
    from PIL import Image
    from komap.dataset import load_full
    from komap.io import to_rgb
    gray, y4, _ = synth_pair
    (tmp_path / "images").mkdir(); (tmp_path / "masks").mkdir()
    ip = tmp_path / "images" / "A1_step1_sample1_image.png"
    Image.fromarray(gray).save(ip)
    Image.fromarray(to_rgb(y4)).save(tmp_path / "masks" / "A1_step1_sample1_mask.png")
    x1, y1, m1 = load_full(ip, 3)
    x2, y2, m2 = load_full(ip, 3, resample_scale=2.0)
    assert abs(x2.shape[0] / x1.shape[0] - 2.0) < 0.02
    assert abs(x2.shape[1] / x1.shape[1] - 2.0) < 0.02
    assert y2.shape == x2.shape
    # meta의 원본 크기는 배수와 무관하게 원본 그대로여야 한다(두께 공식이 쓴다)
    assert m1["width"] == m2["width"] == gray.shape[1]
    assert m2["factor"] == m1["factor"] * 2.0


def test_dataset_with_ridge_adds_one_channel(tmp_path, synth_pair):
    from PIL import Image
    from komap.dataset import SegDataset
    from komap.io import to_rgb
    gray, y4, _ = synth_pair
    (tmp_path / "images").mkdir(); (tmp_path / "masks").mkdir()
    ip = tmp_path / "images" / "A1_step1_sample1_image.png"
    Image.fromarray(gray).save(ip)
    Image.fromarray(to_rgb(y4)).save(tmp_path / "masks" / "A1_step1_sample1_mask.png")
    x, y = SegDataset([ip], tile=128, train=True, n_classes=3, samples_per_image=1, ridge=True)[0]
    assert x.shape == (2, 128, 128)          # 회색조 + 능선
    assert 0.0 <= float(x[1].min()) and float(x[1].max()) <= 1.0
    # 힌트와 함께 켜면 6채널
    x2, _ = SegDataset([ip], tile=128, train=True, n_classes=3, samples_per_image=1,
                       hint_thr=(95, 183), ridge=True)[0]
    assert x2.shape == (6, 128, 128)


def test_load_full_with_ridge_matches_channel_count(tmp_path, synth_pair):
    """추론 경로도 학습과 같은 채널 순서 [회색조, 힌트 4, 능선 1]을 내놓아야 한다."""
    import numpy as np
    from PIL import Image
    from komap import hint as Hn
    from komap.dataset import load_full
    from komap.io import to_rgb
    gray, y4, _ = synth_pair
    (tmp_path / "images").mkdir(); (tmp_path / "masks").mkdir()
    ip = tmp_path / "images" / "A1_step1_sample1_image.png"
    Image.fromarray(gray).save(ip)
    Image.fromarray(to_rgb(y4)).save(tmp_path / "masks" / "A1_step1_sample1_mask.png")
    x, _, _ = load_full(ip, 3, ridge=True)
    assert x.dtype == np.float32 and x.shape[0] == Hn.input_channels(False, True) == 2
    x2, _, _ = load_full(ip, 3, hint_thr=(95, 183), ridge=True)
    assert x2.shape[0] == Hn.input_channels(True, True) == 6
    # 능선은 마지막 채널이다: 힌트를 켜도 끄도 같은 지도가 맨 뒤에 온다.
    assert np.allclose(x[-1], x2[-1])


def test_ridge_channel_survives_hint_dropout(tmp_path, synth_pair, monkeypatch):
    """힌트 dropout은 규칙 지도만 지운다. 능선은 회색조에서 직접 계산한 특징이라 남아야 한다."""
    from PIL import Image
    from komap import hint as Hn
    from komap.dataset import SegDataset
    from komap.io import to_rgb
    gray, y4, _ = synth_pair
    (tmp_path / "images").mkdir(); (tmp_path / "masks").mkdir()
    ip = tmp_path / "images" / "A1_step1_sample1_image.png"
    Image.fromarray(gray).save(ip); Image.fromarray(to_rgb(y4)).save(tmp_path / "masks" / "A1_step1_sample1_mask.png")
    monkeypatch.setattr(Hn, "HINT_DROP_P", 1.0)
    x, _ = SegDataset([ip], tile=128, train=True, n_classes=3, samples_per_image=1,
                      hint_thr=(95, 183), ridge=True)[0]
    assert (x[1:5] == 0).all()               # 힌트 4채널은 지워졌다
    assert (x[5] > 0).any()                  # 능선은 살아 있다
