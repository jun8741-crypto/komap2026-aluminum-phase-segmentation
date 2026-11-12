"""모델 빌더 테스트: 구조·출력 크기, 장치 선택, 인코더 가중치 3종."""
import pytest
import torch

from komap import model as Mo


def test_build_and_forward_cpu():
    m = Mo.build_model("resnet18", weights="none", n_classes=3)
    x = torch.zeros(2, 1, 64, 64)
    y = m(x)
    assert y.shape == (2, 3, 64, 64)


def test_pick_device_returns_device():
    assert isinstance(Mo.pick_device(), torch.device)


def test_imagenet_weights_load():
    m = Mo.build_model("resnet18", weights="imagenet", n_classes=3)
    assert m(torch.zeros(1, 1, 64, 64)).shape == (1, 3, 64, 64)


def _micronet_available():
    """pmm 패키지 임포트와 가중치 다운로드는 네트워크·구버전 의존성에 걸릴 수 있다."""
    try:
        import pretrained_microscopy_models  # noqa: F401
    except Exception:
        return False
    return True


@pytest.mark.skipif(not _micronet_available(), reason="pretrained_microscopy_models 미설치")
def test_micronet_url_is_built():
    """URL 생성은 순수 문자열 연산이라 네트워크 없이 확인할 수 있다."""
    url = Mo.micronet_url("resnet18")
    assert url.startswith("https://") and "resnet18" in url and "microscopynet" in url


@pytest.mark.skipif(not _micronet_available(), reason="pretrained_microscopy_models 미설치")
def test_micronet_weights_load():
    """가중치 다운로드가 필요하므로 네트워크 실패 시 건너뛴다."""
    try:
        m = Mo.build_model("resnet18", weights="micronet", n_classes=3)
    except OSError as e:  # 다운로드 실패(네트워크·S3 접근)
        pytest.skip(f"MicroNet 가중치 다운로드 불가: {e}")
    assert m(torch.zeros(1, 1, 64, 64)).shape == (1, 3, 64, 64)
    # 1채널로 합산된 첫 conv가 살아 있는지
    assert m.encoder.conv1.weight.shape[1] == 1


def test_pick_device_honours_env(monkeypatch):
    """KOMAP_DEVICE가 있으면 자동 캐스케이드를 덮는다. 심사·디버깅에서 CPU를 강제하려면 필요하다."""
    monkeypatch.setenv("KOMAP_DEVICE", "cpu")
    assert Mo.pick_device() == torch.device("cpu")


def test_pick_device_rejects_unknown_env(monkeypatch):
    monkeypatch.setenv("KOMAP_DEVICE", "tpu")
    with pytest.raises(ValueError, match="KOMAP_DEVICE"):
        Mo.pick_device()


def test_mps_fallback_env_set_by_package_import():
    """MPS 미구현 연산을 CPU로 흘리는 설정은 torch import 이전에 걸려야 하므로 패키지 초기화가 맡는다."""
    import os

    import komap  # noqa: F401

    assert os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1"


def test_local_checkpoint_with_bogus_keys_raises(tmp_path):
    """인코더와 맞는 키가 절반도 안 되면 조용히 무작위 초기값으로 학습하지 말고 멈춰야 한다.
    strict=False는 아무것도 안 실린 체크포인트도 성공으로 보이게 만든다."""
    ck = tmp_path / "bogus.pt"
    torch.save({"encoder": {"bogus.weight": torch.zeros(1)}}, ck)
    with pytest.raises(RuntimeError, match="키 불일치"):
        Mo.build_model("resnet18", weights=str(ck), n_classes=3)


def test_local_checkpoint_with_real_encoder_loads(tmp_path):
    ref = Mo.build_model("resnet18", weights="none", n_classes=3)
    ck = tmp_path / "enc.pt"
    torch.save({"encoder": ref.encoder.state_dict()}, ck)
    m = Mo.build_model("resnet18", weights=str(ck), n_classes=3)
    assert m(torch.zeros(1, 1, 64, 64)).shape == (1, 3, 64, 64)


def test_build_model_in_channels():
    from komap.model import build_model
    import torch
    m = build_model("resnet18", "none", 3, in_channels=5)
    assert m(torch.zeros(1, 5, 64, 64)).shape == (1, 3, 64, 64)


def test_micronet_rejects_multichannel_before_download(monkeypatch):
    """5채널로 MicroNet을 요청하면 내려받기 전에 멈춰야 한다.

    첫 conv의 3→1 채널 합산은 1채널 입력에서만 옳다. 그대로 진행하면 조용히 틀린 모양의
    가중치를 싣게 된다. micronet_url을 폭탄으로 갈아 끼워 가드가 네트워크 접근 이전에
    걸리는지까지 확인한다."""
    import pytest
    from komap import model as Mo

    def boom(*a, **k):
        raise AssertionError("가드보다 먼저 네트워크에 접근했다")

    monkeypatch.setattr(Mo, "micronet_url", boom)
    with pytest.raises(ValueError, match="1채널"):
        Mo.build_model("resnet18", "micronet", 3, in_channels=5)
