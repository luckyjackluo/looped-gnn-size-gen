import numpy as np
import pytest
import torch

from sizegen.models import FiLMController, FixedDepthGNN, LoopedGNN
from sizegen.schemes.tiers import _clone_looped
from sizegen.training import make_dataset
from sizegen.training.train import _nodes_per_graph, evaluate_risk, train_model, make_k_sampler

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _tiny_batch(n=60, in_dim=2, seed=0):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    x = torch.randn(n, in_dim)
    # random sparse symmetric edges
    m = 3 * n
    src = torch.from_numpy(rng.integers(0, n, m))
    dst = torch.from_numpy(rng.integers(0, n, m))
    edge_index = torch.stack(
        [torch.cat([src, dst]), torch.cat([dst, src])]
    ).long()
    batch = torch.zeros(n, dtype=torch.long)
    npg = torch.tensor([n])
    return x, edge_index, batch, npg


def test_zero_init_controller_is_identity():
    """FS at phi = 0 must reproduce the frozen Tier-1 loop bit-exactly (A7)."""
    torch.manual_seed(0)
    x, ei, b, npg = _tiny_batch()
    base = LoopedGNN(2, hidden_dim=32)
    steered = _clone_looped(base, with_controller=True)
    base.eval(), steered.eval()
    with torch.no_grad():
        for K in (1, 4, 9):
            out0 = base(x, ei, b, npg, K=K)
            out1 = steered(x, ei, b, npg, K=K)
            assert torch.equal(out0, out1)


def test_clone_shares_no_storage():
    base = LoopedGNN(2, hidden_dim=32)
    clone = _clone_looped(base, with_controller=False)
    with torch.no_grad():
        next(clone.parameters()).add_(1.0)
    assert not torch.equal(next(base.parameters()), next(clone.parameters()))


def test_deployment_depth_is_free_knob():
    """Same weights, different K -> different receptive field/output."""
    torch.manual_seed(1)
    x, ei, b, npg = _tiny_batch()
    model = LoopedGNN(2, hidden_dim=32).eval()
    with torch.no_grad():
        o2 = model(x, ei, b, npg, K=2)
        o8 = model(x, ei, b, npg, K=8)
    assert not torch.allclose(o2, o8)


def test_controller_freeze_partition():
    # experiment-size model: d_phi must be a small fraction of d_R
    model = LoopedGNN(2, hidden_dim=128, controller=FiLMController(128))
    from sizegen.training.train import _set_trainable

    n_ctrl = _set_trainable(model, "controller")
    assert n_ctrl == model.controller.num_params()
    assert all(not p.requires_grad for p in model.backbone_parameters())
    n_all = _set_trainable(model, "all")
    assert n_all > 10 * n_ctrl  # d_phi << d_R


def test_fixed_gnn_ignores_K():
    torch.manual_seed(2)
    x, ei, b, npg = _tiny_batch()
    model = FixedDepthGNN(2, hidden_dim=32, k_fix=3).eval()
    with torch.no_grad():
        assert torch.equal(model(x, ei, b, npg, K=2), model(x, ei, b, npg, K=64))


def test_training_reduces_loss_on_pagerank():
    train_set = make_dataset("rgg_d2", "pagerank", [80, 100], 12, seed=0)
    val_set = make_dataset("rgg_d2", "pagerank", [100], 4, seed=1)
    model = LoopedGNN(2, hidden_dim=48)
    r0 = evaluate_risk(model, val_set, K=6, device=DEVICE)["mse"]
    train_model(
        model, train_set, epochs=30, k_sampler=make_k_sampler(2, 6),
        device=DEVICE, val_dataset=val_set, val_K=6, verbose=False,
    )
    r1 = evaluate_risk(model, val_set, K=6, device=DEVICE)["mse"]
    assert r1 < 0.5 * r0


def test_anchored_unroll_is_depth_stable():
    """Anchored update must stay BOUNDED at extreme depth (K=200).

    The anchor pull gives ||h|| a geometrically contracting norm recursion
    (r <- (1-a) r + const), so no blow-up at any depth — the guarantee that
    the plain loop lacked (risk ~1.5 at K=50 in smoke2).  Full convergence
    (deltas -> 0) is shaped by training + the contraction penalty, not init.
    """
    torch.manual_seed(3)
    x, ei, b, npg = _tiny_batch()
    from sizegen.models import LoopedGNN as LG

    model = LG(2, hidden_dim=32, anchored=True).eval()
    with torch.no_grad():
        outs, deltas = model(x, ei, b, npg, K=200, return_all=True,
                             return_deltas=True)
    d = torch.stack([torch.as_tensor(float(v)) for v in deltas])
    # no growth: late steps no larger than early steps
    assert float(d[-20:].mean()) < 3.0 * float(d[:20].mean())
    assert torch.isfinite(outs[-1]).all()
    # bounded outputs across the whole unroll
    assert max(float(o.abs().max()) for o in outs) < 1e3


def test_anchored_clone_preserves_flag():
    from sizegen.models import LoopedGNN as LG

    base = LG(2, hidden_dim=32, anchored=True)
    clone = _clone_looped(base, with_controller=True)
    assert clone.anchored and hasattr(clone, "anchor_logit")
    assert torch.equal(clone.anchor_logit, base.anchor_logit)
