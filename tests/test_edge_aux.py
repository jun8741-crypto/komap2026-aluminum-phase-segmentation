"""경계 보조 학습의 정답·손실·저장 계약 검증."""
import math

import pytest
import torch
import torch.nn.functional as F

from komap.model import build_model
from komap.edge_aux import (attach_edge_head, boundary_target, checkpoint_state_dict,
                            edge_bce_loss, fold_completeness, unexplained_arg_differences)
from komap.train import parse_args, step_loss


def test_boundary_target_matches_l1_definition():
    g = torch.Generator().manual_seed(0)
    target = torch.randint(0, 3, (2, 24, 24), generator=g)
    target[:, :2, :5] = 255
    target[:, 8:13, 10:15] = 255
    for band in (1, 2, 3):
        edge, valid = boundary_target(target, band)
        want = torch.zeros_like(edge)
        for b in range(2):
            for y in range(24):
                for x in range(24):
                    if target[b, y, x] == 255:
                        continue
                    for dy in range(-band, band + 1):
                        for dx in range(-band, band + 1):
                            yy, xx = y + dy, x + dx
                            if (abs(dy) + abs(dx) <= band and 0 <= yy < 24 and 0 <= xx < 24
                                    and target[b, yy, xx] != 255
                                    and target[b, yy, xx] != target[b, y, x]):
                                want[b, y, x] = True
        assert torch.equal(edge, want)
        assert torch.equal(valid, target != 255)


def test_boundary_target_single_class_all_zero():
    target = torch.zeros(1, 8, 24, dtype=torch.long)
    for band in (0, 2):
        edge, valid = boundary_target(target, band)
        assert not edge.any() and valid.all()


def test_boundary_target_two_class_band_width():
    target = torch.zeros(1, 8, 24, dtype=torch.long)
    target[:, :, 12:] = 1
    edge, _ = boundary_target(target, 2)
    assert {x for x in range(24) if edge[0, :, x].all()} == {10, 11, 12, 13}
    assert edge.sum() == 8 * 4


def test_boundary_target_ignore_handling():
    target = torch.zeros(1, 8, 24, dtype=torch.long)
    target[:, :, 10] = 255
    edge, valid = boundary_target(target, 2)
    assert not edge.any() and not valid[:, :, 10].any()
    target[:, :, 11:] = 1
    edge, _ = boundary_target(target, 2)
    assert {x for x in range(24) if edge[0, :, x].all()} == {9, 11}
    target[0, 0, 0] = 3
    with pytest.raises(ValueError):
        boundary_target(target)


def test_edge_bce_excludes_ignore():
    logits = torch.zeros(1, 1, 2, 2, requires_grad=True)
    edge = torch.tensor([[[True, False], [False, True]]])
    valid = torch.tensor([[[True, False], [True, False]]])
    original = edge_bce_loss(logits, edge, valid)
    changed = logits.detach().clone()
    changed[0, 0, 0, 1], changed[0, 0, 1, 1] = -1e4, 1e4
    assert torch.equal(original, edge_bce_loss(changed, edge, valid))
    zero = edge_bce_loss(logits, edge, torch.zeros_like(valid))
    assert zero == 0
    zero.backward()
    assert logits.grad is not None
    assert torch.allclose(edge_bce_loss(logits, edge, torch.ones_like(valid)),
                          F.binary_cross_entropy_with_logits(logits, edge[:, None].float()))


def test_wrapper_forward_is_single_tensor_equal_to_base():
    base = build_model('resnet18', 'none', 3)
    model = attach_edge_head(base, 0).eval()
    x = torch.randn(1, 1, 32, 32)
    with torch.no_grad():
        out = model(x)
        seg, edge = model.forward_with_edge(x)
        assert isinstance(out, torch.Tensor) and torch.equal(out, base(x))
        assert torch.equal(seg, base(x)) and edge.shape == (1, 1, 32, 32)


def test_checkpoint_state_dict_strict_loads_into_build_model():
    base = build_model('resnet18', 'none', 3)
    model = attach_edge_head(base, 0)
    sd = checkpoint_state_dict(model)
    plain = build_model('resnet18', 'none', 3)
    assert list(sd) == list(plain.state_dict())
    assert not any(k.startswith('edge_head') for k in sd)
    plain.load_state_dict(sd, strict=True)
    assert list(checkpoint_state_dict(plain)) == list(plain.state_dict())


def test_attach_edge_head_leaves_global_rng_untouched():
    torch.manual_seed(7)
    base = build_model('resnet18', 'none', 3)
    before = {k: v.clone() for k, v in base.state_dict().items()}
    cpu = torch.get_rng_state().clone()
    mps = torch.mps.get_rng_state() if torch.backends.mps.is_available() else None
    a = attach_edge_head(base, 7)
    b = attach_edge_head(base, 7)
    c = attach_edge_head(base, 8)
    assert torch.equal(torch.get_rng_state(), cpu)
    if mps is not None:
        assert torch.equal(torch.mps.get_rng_state(), mps)
    assert all(torch.equal(v, before[k]) for k, v in base.state_dict().items())
    assert torch.equal(a.edge_head.weight, b.edge_head.weight)
    assert not torch.equal(a.edge_head.weight, c.edge_head.weight)
    assert a.edge_head.weight.abs().max() <= 1 / math.sqrt(16 * 9)


def test_step_loss_weight_zero_is_plain_seg_loss_and_calls_nothing_else(monkeypatch):
    import komap.train as train
    model = build_model('resnet18', 'none', 3).eval()
    x, y = torch.randn(1, 1, 32, 32), torch.zeros(1, 32, 32, dtype=torch.long)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=255)
    monkeypatch.setattr(train, 'boundary_target', lambda *a, **k: (_ for _ in ()).throw(AssertionError()))
    with torch.no_grad():
        loss, parts = step_loss(model, loss_fn, x, y, 0)
        assert torch.equal(loss, loss_fn(model(x), y)) and parts == {}
    monkeypatch.undo()
    wrapped = attach_edge_head(model, 1)
    y[:, :3] = 255
    with torch.no_grad():
        loss, parts = step_loss(wrapped, loss_fn, x, y, 0.5)
        assert torch.isclose(loss, torch.tensor(parts['seg'] + 0.5 * parts['edge']))
        assert set(parts) == {'seg', 'edge'}


def test_unexplained_arg_differences_by_definition():
    saved = {'a': 1}
    expected = {'a': 1, 'norm': 'anchor3', 'edge_aux_weight': 0.5}
    defaults = {'a': 1, 'norm': 'anchor3', 'edge_aux_weight': 0.0}
    assert unexplained_arg_differences(saved, expected, defaults) == {'edge_aux_weight': (None, 0.5)}
    expected['norm'] = 'none'
    assert 'norm' in unexplained_arg_differences(saved, expected, defaults)
    saved['a'] = 2
    assert 'a' in unexplained_arg_differences(saved, expected, defaults)
    expected['zzz'] = defaults['zzz'] = 3
    assert 'zzz' in unexplained_arg_differences(saved, expected, defaults)
    saved['norm'] = 'wrong'
    assert 'norm' in unexplained_arg_differences(saved, expected, defaults)


def test_checkpoint_matches_trained_base_and_reloads_identically():
    model = attach_edge_head(build_model('resnet18', 'none', 3), 1).eval()
    before = {k: v.clone() for k, v in model.base.state_dict().items()}
    x = torch.randn(1, 1, 32, 32)
    y = torch.zeros(1, 32, 32, dtype=torch.long)
    y[:, :, 16:] = 1
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    loss, _ = step_loss(model, torch.nn.CrossEntropyLoss(), x, y, 0.5)
    loss.backward()
    opt.step()
    sd = checkpoint_state_dict(model)
    assert all(torch.equal(v, model.base.state_dict()[k]) for k, v in sd.items())
    assert any(not torch.equal(v, before[k]) for k, v in sd.items())
    plain = build_model('resnet18', 'none', 3).eval()
    plain.load_state_dict(sd, strict=True)
    with torch.no_grad():
        assert torch.equal(plain(x), model(x))


def test_edge_loss_gradient_reaches_shared_trunk():
    model = attach_edge_head(build_model('resnet18', 'none', 3), 1).train()
    x = torch.randn(2, 1, 32, 32)
    y = torch.zeros(2, 32, 32, dtype=torch.long)
    y[:, :, 16:] = 1
    before = model.base.encoder.conv1.weight.detach().clone()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    _, edge = model.forward_with_edge(x)
    loss = edge_bce_loss(edge, *boundary_target(y))
    loss.backward()
    for module in (model.base.encoder, model.base.decoder, model.edge_head):
        assert any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
                   for p in module.parameters())
    assert all(p.grad is None for p in model.base.segmentation_head.parameters())
    opt.step()
    assert not torch.equal(before, model.base.encoder.conv1.weight)


def test_edge_aux_weight_cli():
    base = ['--exp', 'x', '--fold', '0']
    assert parse_args(base).edge_aux_weight == 0.0
    for bad in ('-0.1', 'nan', 'inf'):
        with pytest.raises(SystemExit):
            parse_args([*base, '--edge-aux-weight', bad])
    with pytest.raises(SystemExit):
        parse_args([*base, '--edge-aux-weight', '0.5', '--n-classes', '4'])
    assert parse_args([*base, '--edge-aux-weight', '0.5']).edge_aux_weight == 0.5
    assert parse_args([*base, '--edge-aux-weight', '0']).edge_aux_weight == 0


def test_fold_completeness_states(tmp_path):
    fold = tmp_path / 'fold0'
    expected = {'exp': 'x', 'fold': 0, 'bs': 4}
    assert fold_completeness(fold, ['A.png'], expected, expected) == 'absent'
    (fold / 'oof_probs').mkdir(parents=True)
    (fold / 'result.json').write_text('{}')
    torch.save({'model': {}, 'args': expected}, fold / 'best.pt')
    (fold / 'oof_probs' / 'A.npy').write_bytes(b'')
    assert fold_completeness(fold, ['A.png'], expected, expected) == 'complete'
    (fold / 'best.pt').unlink()
    assert fold_completeness(fold, ['A.png'], expected, expected).startswith('partial:')
    torch.save({'model': {}, 'args': expected}, fold / 'best.pt')
    (fold / 'oof_probs' / 'B.npy').write_bytes(b'')
    assert fold_completeness(fold, ['A.png'], expected, expected).startswith('partial:')
    (fold / 'oof_probs' / 'B.npy').unlink()
    torch.save({'model': {}, 'args': {**expected, 'bs': 2}}, fold / 'best.pt')
    assert fold_completeness(fold, ['A.png'], expected, expected).startswith('partial:')
