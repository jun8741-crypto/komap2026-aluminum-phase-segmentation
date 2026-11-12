"""전체 이미지 추론: 겹치는 타일 + 가우시안 가중 블렌딩 + 플립 TTA. 확률(softmax) 맵을 돌려준다.

가우시안 가중을 쓰는 이유: 타일 가장자리 픽셀은 문맥이 반쯤 잘려 예측이 불안하다.
타일 중심에 큰 가중을 주고 겹친 영역을 가중평균하면 타일 경계의 이음매가 사라진다.
가중평균은 확률 벡터의 볼록결합이라 합이 1인 성질도 그대로 보존된다.
"""
import numpy as np
import torch
import torch.nn.functional as F

FLIPS_TTA = [(False, False), (True, False), (False, True), (True, True)]


def _gauss_window(tile: int, sigma_ratio: float = 0.25) -> np.ndarray:
    """중심이 1.0, 가장자리로 갈수록 작아지는 2차원 가우시안 가중창."""
    ax = np.arange(tile) - (tile - 1) / 2
    g = np.exp(-(ax ** 2) / (2 * (tile * sigma_ratio) ** 2))
    w = np.outer(g, g).astype(np.float32)
    return w / w.max()


@torch.no_grad()
def predict_probs(model, x_norm, tile, overlap=0.5, tta=True, device=None):
    """x_norm: 앵커 정규화·리샘플까지 끝난 입력.

    2D uint8 HxW면 회색조 한 장으로 보고 0~1로 나눈다. 3D float32 [C,H,W]면 이미 0~1로
    맞춰진 다채널 입력(회색조 + 규칙 지도 힌트)으로 보고 채널을 그대로 유지한다.
    반환 채널 수는 모델 출력(n_classes)이며 입력 채널 수와 무관하다.
    반환: float32 [C,H,W] 확률맵."""
    device = device or torch.device("cpu")
    model = model.to(device).eval()
    if x_norm.ndim == 2:                         # uint8 HxW 회색조(기존 경로)
        x = x_norm.astype(np.float32)[None] / 255.0
    else:                                        # float32 [C,H,W], 이미 0~1
        x = x_norm.astype(np.float32)
    _, H, W = x.shape
    stride = max(1, int(tile * (1 - overlap)))
    # 타일이 이미지 끝까지 정확히 닿도록 패딩한다. 이미지가 타일보다 작으면 타일 크기까지 채운다.
    pad_h = (-(H - tile) % stride) if H > tile else tile - H
    pad_w = (-(W - tile) % stride) if W > tile else tile - W
    x = np.pad(x, ((0, 0), (0, pad_h), (0, pad_w)), mode="reflect")
    hp, wp = x.shape[1:]
    win = torch.from_numpy(_gauss_window(tile)).to(device)
    flips = FLIPS_TTA if tta else [(False, False)]
    acc = None
    norm = torch.zeros(1, hp, wp, device=device)
    for top in range(0, hp - tile + 1, stride):
        for left in range(0, wp - tile + 1, stride):
            patch = torch.from_numpy(x[:, top:top + tile, left:left + tile])[None].to(device)
            prob = 0
            for fh, fv in flips:
                p = patch
                if fh:
                    p = p.flip(-1)
                if fv:
                    p = p.flip(-2)
                out = F.softmax(model(p), 1)
                # 예측을 원래 방향으로 되돌린 뒤 평균낸다.
                if fv:
                    out = out.flip(-2)
                if fh:
                    out = out.flip(-1)
                prob = prob + out
            prob = prob[0] / len(flips)
            if acc is None:
                acc = torch.zeros(prob.shape[0], hp, wp, device=device)
            acc[:, top:top + tile, left:left + tile] += prob * win
            norm[:, top:top + tile, left:left + tile] += win
    acc = acc / norm.clamp(min=1e-6)
    return acc[:, :H, :W].cpu().numpy()
