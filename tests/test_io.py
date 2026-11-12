import numpy as np
import pytest
from komap import io


def test_parse_name():
    d = io.parse_name("A147_step4_sample4_image.png")
    assert d == {"alloy": 147, "step": 4, "sample": 4}


def test_mask_roundtrip(synth_pair):
    _, y4, rgb = synth_pair
    assert np.array_equal(io.rgb_to_mask(rgb), y4)
    assert np.array_equal(io.to_rgb(y4), rgb)
    assert io.rgb_to_mask(rgb).dtype == np.int8


def test_unmapped_color_raises():
    rgb = np.zeros((4, 4, 3), np.uint8)
    rgb[0, 0] = (1, 2, 3)
    with pytest.raises(ValueError):
        io.rgb_to_mask(rgb)


def test_to3():
    y4 = np.array([[0, 1], [2, 3]])
    assert io.to3(y4).tolist() == [[0, 1], [2, 2]]


def test_to_rgba_matches_competition_format(synth_pair, tmp_path):
    """제출 마스크는 대회 정답과 같은 RGBA여야 한다.

    알파를 뺀 3채널은 색상값이 같아도 배열 shape가 달라, 채점기가 convert 없이
    비교하면 어긋난다. 저장·재적재 왕복에서 클래스가 보존되는지도 함께 본다.
    """
    from PIL import Image

    _, y4, rgb = synth_pair
    rgba = io.to_rgba(y4)
    assert rgba.shape == (*y4.shape, 4)
    assert rgba.dtype == np.uint8
    assert np.array_equal(rgba[..., :3], rgb)
    assert (rgba[..., 3] == 255).all()

    p = tmp_path / "A1_step1_sample1_mask.png"
    Image.fromarray(rgba).save(p)
    with Image.open(p) as im:
        assert im.mode == "RGBA"
    assert np.array_equal(io.load_mask(p), y4)
