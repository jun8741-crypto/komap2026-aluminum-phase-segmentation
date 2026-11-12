"""분할 모델 빌더. 구조는 smp, 인코더 가중치는 ImageNet / MicroNet / 없음 / 로컬 체크포인트.

MicroNet은 NASA가 현미경 이미지 수십만 장으로 사전학습한 인코더 가중치다. 광학현미경
조직 사진은 ImageNet의 자연 이미지와 통계가 다르므로 도메인이 맞는 초기값을 비교 실험한다.
"""
import os

import segmentation_models_pytorch as smp
import torch

MICRONET_REPO = "https://github.com/nasa/pretrained-microscopy-models"
ARCHS = {"unetplusplus": smp.UnetPlusPlus, "unet": smp.Unet, "fpn": smp.FPN}

DEVICE_ENV = "KOMAP_DEVICE"
DEVICE_CHOICES = ("cpu", "mps", "cuda")


def pick_device() -> torch.device:
    """KOMAP_DEVICE가 있으면 그 값을 쓰고, 없으면 Apple Silicon(mps) → CUDA → CPU 순으로 고른다.

    환경변수 우선순위를 두는 이유: 심사·디버깅에서 MPS 수치 차이를 배제하려면 CPU를 강제할 수
    있어야 한다. 오타를 조용히 무시하면 원하지 않은 장치로 돌아가므로 알 수 없는 값은 막는다."""
    want = os.environ.get(DEVICE_ENV)
    if want:
        if want not in DEVICE_CHOICES:
            raise ValueError(
                f"알 수 없는 {DEVICE_ENV}: {want!r} (가능: {', '.join(DEVICE_CHOICES)})"
            )
        return torch.device(want)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _check_encoder_keys(model, state, source) -> None:
    """체크포인트가 이 인코더와 실제로 맞는지 본다.

    load_state_dict(strict=False)는 키가 하나도 안 맞아도 조용히 통과한다. 그러면 사전학습
    가중치를 썼다고 믿으면서 실제로는 무작위 초기값으로 학습하게 된다. 절반 미만이 맞으면
    사람 실수(인코더 이름 불일치·엉뚱한 파일)로 보고 즉시 멈춘다."""
    enc_keys = set(model.encoder.state_dict())
    matched = enc_keys & set(state)
    if len(matched) < len(enc_keys) // 2:
        raise RuntimeError(
            f"인코더 가중치 키 불일치: {source} 의 키 중 인코더와 맞는 것이 "
            f"{len(matched)}개뿐이다(인코더 키 {len(enc_keys)}개). "
            "인코더 이름과 체크포인트 파일이 맞는지 확인하라."
        )


def build_model(encoder="resnet34", weights="imagenet", n_classes=3, arch="unetplusplus",
                in_channels=1):
    """weights: "imagenet" | "micronet" | "none" | 로컬 체크포인트 경로.

    in_channels는 기본 1(회색조)이고, 규칙 지도 캐스케이드는 힌트 4채널을 붙여 5를 쓴다.
    ImageNet 가중치의 3채널 첫 conv는 smp가 요청 채널 수로 알아서 바꿔 준다."""
    if arch not in ARCHS:
        raise ValueError(f"알 수 없는 arch: {arch} (가능: {sorted(ARCHS)})")
    enc_w = "imagenet" if weights == "imagenet" else None
    m = ARCHS[arch](encoder_name=encoder, encoder_weights=enc_w, in_channels=in_channels,
                    classes=n_classes)
    if weights == "micronet":
        load_micronet(m, encoder, in_channels)
    elif weights not in ("imagenet", "none"):
        state = torch.load(weights, map_location="cpu", weights_only=True)
        state = state.get("encoder", state)
        _check_encoder_keys(m, state, weights)
        m.encoder.load_state_dict(state, strict=False)
    return m


def micronet_url(encoder: str, encoder_weights: str = "micronet") -> str:
    """pretrained_microscopy_models가 만들어 주는 S3 가중치 URL. 순수 문자열 연산이다."""
    try:
        import pretrained_microscopy_models as pmm
    except ImportError as e:
        raise RuntimeError(f"pretrained_microscopy_models 미설치. {MICRONET_REPO} 참고해 설치") from e
    return pmm.util.get_pretrained_microscopynet_url(encoder, encoder_weights)


def load_micronet(model, encoder, in_channels=1) -> None:
    """MicroNet 인코더 가중치 로드.
    사전학습 첫 conv는 RGB 3채널이지만 우리 입력은 회색조 1채널이라 채널 축으로 합산해 옮긴다.
    (합산은 회색 입력을 R=G=B로 복제해 넣은 것과 수학적으로 같은 응답을 준다.)

    합산은 1채널 입력에서만 옳다. 다른 채널 수에서 조용히 틀린 모양의 가중치를 싣느니 멈춘다."""
    if in_channels != 1:
        raise ValueError(f"MicroNet은 1채널 입력만 지원한다(in_channels={in_channels}).")
    url = micronet_url(encoder)
    # weights_only=True: 체크포인트는 외부 S3에서 받으므로 임의 객체 역직렬화를 막는다.
    state = torch.hub.load_state_dict_from_url(url, map_location="cpu", weights_only=True)
    state = state.get("state_dict", state)
    first = [k for k in state if k.endswith("conv1.weight") or k.endswith("conv_stem.weight")]
    for k in first:
        if state[k].ndim == 4 and state[k].shape[1] == 3:
            state[k] = state[k].sum(1, keepdim=True)
    # smp 0.5.0의 인코더 load_state_dict는 분류용 fc 키를 떼어내며 super() 결과를 반환하지 않아
    # None이 돌아온다. 반환값 대신 키 집합을 직접 비교해 실제로 실린 가중치를 검증한다.
    _check_encoder_keys(model, state, f"MicroNet({url})")
    model.encoder.load_state_dict(state, strict=False)
