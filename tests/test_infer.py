"""추론 테스트: 겹치는 타일 블렌딩이 전체 확률맵을 빠짐없이 재구성하는지 본다."""
import numpy as np
import torch

from komap import infer as I


class Const(torch.nn.Module):
    """입력 밝기가 128 이상이면 클래스 0, 아니면 2를 강하게 내는 가짜 모델."""

    def forward(self, x):
        bright = (x[:, :1] >= 0.5).float()
        return torch.cat([bright * 10, torch.zeros_like(bright), (1 - bright) * 10], 1)


def test_sliding_window_reconstructs_full_map():
    x = np.full((300, 500), 226, np.uint8)
    x[100:200, 100:300] = 40
    p = I.predict_probs(Const(), x, tile=128, overlap=0.5, tta=True, device=torch.device("cpu"))
    assert p.shape == (3, 300, 500)
    assert np.allclose(p.sum(0), 1, atol=1e-4)
    assert p.argmax(0)[150, 200] == 2 and p.argmax(0)[10, 10] == 0


def test_image_smaller_than_tile():
    """타일보다 작은 이미지도 패딩 후 원래 크기로 잘려 나와야 한다."""
    x = np.full((40, 70), 226, np.uint8)
    p = I.predict_probs(Const(), x, tile=128, overlap=0.5, tta=False, device=torch.device("cpu"))
    assert p.shape == (3, 40, 70)
    assert np.allclose(p.sum(0), 1, atol=1e-4)
    assert (p.argmax(0) == 0).all()


def test_every_pixel_is_covered():
    """가장자리 픽셀까지 최소 한 타일에 덮여야 한다. 안 덮이면 확률 합이 1이 아니다."""
    x = np.full((301, 507), 40, np.uint8)
    p = I.predict_probs(Const(), x, tile=128, overlap=0.5, tta=False, device=torch.device("cpu"))
    assert p.shape == (3, 301, 507)
    assert np.allclose(p.sum(0), 1, atol=1e-4)
    assert (p.argmax(0) == 2).all()


def test_tta_and_no_tta_agree_on_flip_invariant_model():
    x = np.full((200, 200), 226, np.uint8)
    x[50:150, 50:150] = 40
    kw = dict(tile=128, overlap=0.5, device=torch.device("cpu"))
    a = I.predict_probs(Const(), x, tta=False, **kw)
    b = I.predict_probs(Const(), x, tta=True, **kw)
    assert np.allclose(a, b, atol=1e-5)


def test_predict_probs_accepts_multichannel_float():
    import numpy as np
    import torch
    from komap.infer import predict_probs
    from komap.model import build_model
    m = build_model("resnet18", "none", 3, in_channels=5)
    x = np.random.default_rng(0).random((5, 70, 90), np.float32)
    p = predict_probs(m, x, tile=64, overlap=0.5, tta=False, device=torch.device("cpu"))
    assert p.shape == (3, 70, 90) and np.allclose(p.sum(0), 1, atol=1e-4)
