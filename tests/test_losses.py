"""손실 함수 테스트: 유한성·역전파, 완벽 예측의 낮은 손실, 경계 띠 무시의 실제 효과."""
import pytest
import torch

from komap import losses as L


def _batch():
    y = torch.zeros(2, 32, 32, dtype=torch.long)
    y[:, 8:24, 8:24] = 2
    y[:, :, 28:] = 1
    logits = torch.randn(2, 3, 32, 32)
    return logits, y


def test_all_losses_are_finite_and_backprop():
    for name in ["ce", "ce_dice", "lovasz", "ce_interior"]:
        logits, y = _batch()
        logits.requires_grad_(True)
        loss = L.build_loss(name, 3)(logits, y)
        assert torch.isfinite(loss)
        loss.backward()
        assert logits.grad is not None and torch.isfinite(logits.grad).all()


def test_perfect_prediction_low_loss():
    _, y = _batch()
    logits = torch.nn.functional.one_hot(y, 3).permute(0, 3, 1, 2).float() * 20
    assert L.build_loss("ce_dice", 3)(logits, y) < 0.05


def test_ignore_band_masks_boundary():
    logits, y = _batch()
    l_all = L.build_loss("ce", 3)(logits, y)
    l_ign = L.build_loss("ce", 3, ignore_band_px=2)(logits, y)
    assert l_all != l_ign


def test_ignore_band_does_not_mutate_target():
    """경계 띠를 255로 바꾸는 건 사본이어야 한다. 원본이 오염되면 다음 스텝이 망가진다."""
    logits, y = _batch()
    before = y.clone()
    L.build_loss("ce", 3, ignore_band_px=2)(logits, y)
    assert torch.equal(y, before)


def test_interior_weight_marks_si_interior_only():
    """Si(클래스 2) 내부에만 2.0, 경계 띠와 다른 클래스는 1.0이어야 한다."""
    _, y = _batch()
    w = L._interior_weight(y)
    assert w[0, 16, 16] == 2.0          # Si 블록 한가운데
    assert w[0, 8, 8] == 1.0            # Si 경계 모서리
    assert w[0, 0, 0] == 1.0            # Al 배경


def test_class_weights_change_loss():
    logits, y = _batch()
    l_plain = L.build_loss("ce", 3)(logits, y)
    l_weighted = L.build_loss("ce", 3, class_weights=[1.0, 1.0, 5.0])(logits, y)
    assert not torch.isclose(l_plain, l_weighted)


def test_channel_mismatch_raises():
    """logits 채널 수가 n_classes와 다르면 조용히 학습되지 않고 즉시 터져야 한다."""
    _, y = _batch()
    bad = torch.randn(2, 4, 32, 32)
    with pytest.raises(ValueError, match="n_classes"):
        L.build_loss("ce", 3)(bad, y)


def _batch_with_ignore():
    """오른쪽 위 8x8 구석을 IGNORE(255)로 칠한 배치. 증강 여백이 들어온 상황을 흉내낸다."""
    logits, y = _batch()
    y = y.clone()
    y[:, :8, :8] = L.IGNORE
    return logits, y


def test_all_losses_finite_with_ignore_pixels():
    """마스크에 255가 섞여도 네 손실 모두 유한하고 역전파된다.

    Dice·Lovasz가 ignore_index를 받지 못하면 255를 실제 클래스 인덱스로 해석해
    one-hot 변환에서 터지거나 NaN이 된다.
    """
    for name in ["ce", "ce_dice", "lovasz", "ce_interior"]:
        logits, y = _batch_with_ignore()
        logits.requires_grad_(True)
        loss = L.build_loss(name, 3)(logits, y)
        assert torch.isfinite(loss), name
        loss.backward()
        assert torch.isfinite(logits.grad).all(), name


def test_ignore_pixels_do_not_influence_loss():
    """255 픽셀의 logits를 아무리 흔들어도 손실값이 변하면 안 된다.

    '그 픽셀을 예측 클래스로 바꿔도 손실이 같다'로는 검증할 수 없다. 255는 분모(유효
    픽셀 수)에서도 빠지므로 실제 라벨을 넣으면 분모가 늘어 평균이 달라지기 때문이다.
    무시가 제대로 걸렸는지는 '무시 영역의 입력을 바꿔도 출력이 불변인가'로 재는 게 옳다.
    """
    for name in ["ce", "ce_dice", "lovasz", "ce_interior"]:
        logits, y = _batch_with_ignore()
        other = logits.clone()
        other[:, :, :8, :8] = torch.randn(2, 3, 8, 8) * 10
        fn = L.build_loss(name, 3)
        assert torch.allclose(fn(logits, y), fn(other, y), atol=1e-6), name


def test_interior_weight_covers_both_si_classes_when_4():
    """4클래스에서는 초정 Si(2)와 공정 Si(3) 내부 모두에 가중이 걸려야 한다."""
    from komap.config import CLS_EUTSI, CLS_PRISI

    y = torch.zeros(1, 40, 40, dtype=torch.long)
    y[:, 4:20, 4:20] = CLS_PRISI
    y[:, 24:38, 24:38] = CLS_EUTSI

    w3 = L._interior_weight(y, si_classes=(CLS_PRISI,))
    assert w3[0, 12, 12] == L.INTERIOR_WEIGHT
    assert w3[0, 31, 31] == 1.0          # 3클래스 기준에서는 공정 Si 라벨이 존재하지 않는다

    w4 = L._interior_weight(y, si_classes=(CLS_PRISI, CLS_EUTSI))
    assert w4[0, 12, 12] == L.INTERIOR_WEIGHT
    assert w4[0, 31, 31] == L.INTERIOR_WEIGHT
    assert w4[0, 0, 0] == 1.0


def test_build_loss_passes_si_classes_by_n_classes():
    """build_loss(n_classes=4)의 ce_interior는 공정 Si 내부도 가중해야 한다."""
    assert L._si_classes(3) == (2,)
    assert L._si_classes(4) == (2, 3)


def _batch4():
    y = torch.zeros(2, 32, 32, dtype=torch.long)
    y[:, 8:22, 8:22] = 2
    y[:, 24:32, :24] = 3
    y[:, :, 28:] = 1
    y[:, :4, :4] = L.IGNORE
    return torch.randn(2, 4, 32, 32), y


def test_loss_scope_all_equals_default():
    logits, y = _batch4()
    for name in ("ce", "ce_dice", "lovasz", "ce_interior"):
        for band in (0, 2):
            default = L.build_loss(name, 4, ignore_band_px=band)(logits, y)
            explicit = L.build_loss(name, 4, ignore_band_px=band, loss_scope="all")(logits, y)
            assert torch.equal(default, explicit), (name, band)


def test_loss_scope_si_ignores_non_si_pixels():
    logits, y = _batch4()
    for name in ("ce", "ce_dice", "lovasz", "ce_interior"):
        fn = L.build_loss(name, 4, loss_scope="si")
        base = fn(logits, y)
        other = logits.clone()
        mask = ((y == 0) | (y == 1) | (y == L.IGNORE)).unsqueeze(1).expand_as(other)
        other[mask] = torch.randn_like(other[mask]) * 10
        assert torch.allclose(base, fn(other, y), atol=1e-6), name
        for cls, row, col in ((2, 15, 15), (3, 28, 12)):
            assert y[0, row, col] == cls
            changed = logits.clone()
            changed[0, cls, row, col] -= 40
            assert not torch.isclose(base, fn(changed, y), atol=1e-6), (name, cls)


def test_loss_scope_si_with_ignore_band_still_ignores_non_si():
    logits, y = _batch4()
    mask = ((y == 0) | (y == 1) | (y == L.IGNORE)).unsqueeze(1).expand_as(logits)
    other = logits.clone()
    other[mask] = torch.randn_like(other[mask]) * 10
    for name in ("ce", "ce_dice", "lovasz", "ce_interior"):
        fn = L.build_loss(name, 4, ignore_band_px=2, loss_scope="si")
        assert torch.allclose(fn(logits, y), fn(other, y), atol=1e-6), name


def test_loss_scope_si_no_si_tile_is_finite_scalar_and_backprops():
    y0 = torch.zeros(2, 16, 16, dtype=torch.long)
    y0[:, :, 8:] = 1
    y1 = torch.full_like(y0, L.IGNORE)
    for name in ("ce", "ce_dice", "lovasz", "ce_interior"):
        for y in (y0, y1):
            logits = torch.randn(2, 4, 16, 16, requires_grad=True)
            loss = L.build_loss(name, 4, loss_scope="si")(logits, y)
            assert loss.dim() == 0 and torch.isfinite(loss) and loss.item() == 0
            loss.backward()
            assert torch.isfinite(logits.grad).all()


def test_loss_scope_si_does_not_mutate_target():
    logits, y = _batch4()
    before = y.clone()
    L.build_loss("ce", 4, ignore_band_px=2, loss_scope="si")(logits, y)
    assert torch.equal(y, before)


def test_loss_scope_rejects_bad_values():
    with pytest.raises(ValueError):
        L.build_loss("ce", 4, loss_scope="foo")
    with pytest.raises(ValueError):
        L.build_loss("ce", 3, loss_scope="si")
    with pytest.raises(ValueError):
        L.build_loss("nope", 4, loss_scope="si")


def test_head_full_equals_default():
    for n in (3, 4):
        logits, y = _batch4() if n == 4 else _batch_with_ignore()
        for name in ("ce", "ce_dice", "lovasz", "ce_interior"):
            for band in (0, 2):
                a = L.build_loss(name, n, ignore_band_px=band)(logits, y)
                b = L.build_loss(name, n, ignore_band_px=band, head="full")(logits, y)
                assert torch.equal(a, b), (n, name, band)


def _remapped_after_band(target, band, primary=0, eutectic=1):
    original = target.clone()
    if band:
        original[L._band_mask(original, band)] = L.IGNORE
    mapped = torch.full_like(original, L.IGNORE)
    mapped[original == 2] = primary
    mapped[original == 3] = eutectic
    return mapped


def test_head_si2_equals_two_class_loss_on_remapped_target():
    _, y = _batch4()
    torch.manual_seed(1)
    logits = torch.randn(2, 2, 32, 32)
    for name in ("ce", "ce_dice", "lovasz"):
        for band in (0, 2):
            got = L.build_loss(name, 4, ignore_band_px=band, loss_scope="si", head="si2")(logits, y)
            ref = L.build_loss(name, 2)(logits, _remapped_after_band(y, band))
            swapped = L.build_loss(name, 2)(logits, _remapped_after_band(y, band, 1, 0))
            assert torch.allclose(got, ref, atol=1e-6), (name, band)
            assert not torch.isclose(got, swapped, atol=1e-6), (name, band)


def test_head_si2_ignores_non_si_pixels():
    _, y = _batch4()
    logits = torch.randn(2, 2, 32, 32)
    non_si = ((y == 0) | (y == 1) | (y == L.IGNORE)).unsqueeze(1).expand_as(logits)
    for name in ("ce", "ce_dice", "lovasz", "ce_interior"):
        for band in (0, 2):
            fn = L.build_loss(name, 4, ignore_band_px=band, loss_scope="si", head="si2")
            base = fn(logits, y)
            other = logits.clone()
            other[non_si] = torch.randn_like(other[non_si]) * 10
            assert torch.allclose(base, fn(other, y), atol=1e-6), (name, band)
            for cls, row, col in ((2, 15, 15), (3, 28, 12)):
                assert y[0, row, col] == cls and not L._band_mask(y, 2)[0, row, col]
                changed = logits.clone()
                changed[0, :, row, col] += torch.tensor([-8.0, 8.0] if cls == 2 else [8.0, -8.0])
                assert not torch.isclose(base, fn(changed, y), atol=1e-6), (name, band, cls)


def test_head_si2_no_si_tile_is_zero_scalar_and_backprops():
    y0 = torch.zeros(2, 16, 16, dtype=torch.long)
    y0[:, :, 8:] = 1
    for name in ("ce", "ce_dice", "lovasz", "ce_interior"):
        for y in (y0, torch.full_like(y0, L.IGNORE)):
            logits = torch.randn(2, 2, 16, 16, requires_grad=True)
            loss = L.build_loss(name, 4, loss_scope="si", head="si2")(logits, y)
            assert loss.dim() == 0 and torch.isfinite(loss) and loss.item() == 0
            loss.backward()
            assert torch.isfinite(logits.grad).all()


def test_head_si2_rejects_bad_combinations():
    for name in ("ce", "ce_dice", "lovasz", "ce_interior"):
        with pytest.raises(ValueError):
            L.build_loss(name, 4, head="foo")
        with pytest.raises(ValueError):
            L.build_loss(name, 3, loss_scope="si", head="si2")
        with pytest.raises(ValueError):
            L.build_loss(name, 4, loss_scope="all", head="si2")
        with pytest.raises(ValueError):
            L.build_loss(name, 4, class_weights=[1, 1, 1, 1], loss_scope="si", head="si2")
        fn = L.build_loss(name, 4, loss_scope="si", head="si2")
        with pytest.raises(ValueError):
            fn(torch.randn(2, 4, 32, 32), _batch4()[1])
