"""The four-tier scheme family (paper §4), as explicit recipes.

Tier 0  : fixed-depth network, deployed as-is                (§4.1)
Tier 1  : looped, zero-shot deployment depth K               (§4.2)
Tier 2/3a (FT): Tier-1 checkpoint, fine-tune ALL on D_adapt  (§4.3)
Tier 3b   (FS): Tier-1 checkpoint FROZEN + train controller  (§4.4)

FT and FS share the pretrained checkpoint, adaptation set, depth rule, and
training budget — they differ ONLY in the freeze sub-choice, which is the
paper's whole point (Prop. 18).
"""

import copy
import math
from typing import Dict, List, Optional, Tuple

import torch

from ..models import FiLMController, FixedDepthGNN, LoopedGNN
from ..training.train import make_k_sampler, train_model


def deploy_depth_rule(n: int, rho_eff: float = 0.8, tau_scale: float = 1.0) -> int:
    """K(N) = ceil(log N / (2 |log rho_eff|)) * tau_scale — the paper's
    closed-form depth schedule (Lemma 14), using the measured rate from E3."""
    return max(2, int(math.ceil(tau_scale * math.log(n) / (2.0 * abs(math.log(rho_eff))))))


def pretrain_tier0(
    train_set: List,
    val_set: List,
    in_dim: int,
    hidden_dim: int = 128,
    k_fix: int = 4,
    epochs: int = 100,
    device: str = "cuda",
    seed: int = 0,
    traj_sup: bool = False,
) -> Dict:
    model = FixedDepthGNN(in_dim, hidden_dim=hidden_dim, k_fix=k_fix)
    # operator-aligned pretraining of the fixed model: each distinct layer's
    # decoded output is supervised toward the k-step partial computation
    k_sampler = (lambda rng: k_fix) if traj_sup else None
    info = train_model(
        model, train_set, epochs=epochs, k_sampler=k_sampler,
        trainable="all", device=device, val_dataset=val_set, val_K=None,
        seed=seed, traj_sup=traj_sup,
    )
    return {"model": model, "info": info, "scheme": "tier0", "k_fix": k_fix}


def pretrain_tier1(
    train_set: List,
    val_set: List,
    in_dim: int,
    hidden_dim: int = 128,
    k_min: int = 2,
    k_max: int = 8,
    epochs: int = 100,
    device: str = "cuda",
    seed: int = 0,
    anchored: bool = False,
    traj_sup: bool = False,
) -> Dict:
    """Looped pretraining with K sampled per step: weight sharing across depth
    turns the process depth into a deployment-time hyperparameter."""
    model = LoopedGNN(in_dim, hidden_dim=hidden_dim, anchored=anchored)
    info = train_model(
        model, train_set, epochs=epochs,
        k_sampler=make_k_sampler(k_min, k_max),
        trainable="all", device=device, val_dataset=val_set, val_K=k_max, seed=seed,
        traj_sup=traj_sup,
    )
    return {"model": model, "info": info, "scheme": "tier1",
            "k_range": (k_min, k_max)}


def _clone_looped(model: LoopedGNN, with_controller: bool) -> LoopedGNN:
    """Fresh LoopedGNN sharing NO storage with `model`, same weights."""
    in_dim = model.encoder[0].in_features
    hidden_dim = model.encoder[0].out_features
    out_dim = model.decoder[-1].out_features
    controller = (
        FiLMController(hidden_dim) if with_controller else None
    )
    clone = LoopedGNN(in_dim, hidden_dim=hidden_dim, out_dim=out_dim,
                      controller=controller, anchored=model.anchored)
    missing, unexpected = clone.load_state_dict(model.state_dict(), strict=False)
    assert not unexpected, unexpected
    # only controller params may be missing from the source state dict
    assert all(k.startswith("controller.") for k in missing), missing
    return clone


def _adapt_k_sampler(k_adapt: int, k_range: Optional[Tuple[int, int]]):
    """Constant depth, or sampled over the range matched to the spread of
    adaptation sizes (so the size-conditioned controller sees a size/depth
    gradient rather than a single point — cf. UnifiedLearning's loop_peft
    training on N=500..1000, not one size)."""
    if k_range is None:
        return lambda rng: k_adapt
    return make_k_sampler(*k_range)


def adapt_ft(
    pretrained: LoopedGNN,
    adapt_set: List,
    val_set: List,
    k_adapt: int,
    k_range: Optional[Tuple[int, int]] = None,
    epochs: int = 50,
    lr: float = 2.5e-4,
    device: str = "cuda",
    seed: int = 0,
    batch_size: int = 32,
    traj_sup: bool = False,
) -> Dict:
    """Scheme FT: fine-tune all d_R + d_phi parameters on D_adapt."""
    model = _clone_looped(pretrained, with_controller=False)
    info = train_model(
        model, adapt_set, epochs=epochs, lr=lr, batch_size=batch_size,
        k_sampler=_adapt_k_sampler(k_adapt, k_range),
        trainable="all", device=device, val_dataset=val_set, val_K=k_adapt, seed=seed,
        traj_sup=traj_sup,
    )
    return {"model": model, "info": info, "scheme": "ft", "k_adapt": k_adapt}


def adapt_fs(
    pretrained: LoopedGNN,
    adapt_set: List,
    val_set: List,
    k_adapt: int,
    k_range: Optional[Tuple[int, int]] = None,
    epochs: int = 50,
    lr: float = 2.5e-4,
    device: str = "cuda",
    seed: int = 0,
    batch_size: int = 32,
    traj_sup: bool = False,
) -> Dict:
    """Scheme FS / LFS: freeze encoder+processor; train the zero-init
    size-conditioned FiLM controller AND the readout (d_phi + d_D << d_R)."""
    model = _clone_looped(pretrained, with_controller=True)
    info = train_model(
        model, adapt_set, epochs=epochs, lr=lr, batch_size=batch_size,
        k_sampler=_adapt_k_sampler(k_adapt, k_range),
        trainable="controller+decoder", device=device, val_dataset=val_set,
        val_K=k_adapt, seed=seed, traj_sup=traj_sup,
    )
    d_r = sum(p.numel() for p in model.backbone_parameters())
    d_phi = model.controller.num_params()
    d_dec = sum(p.numel() for p in model.decoder.parameters())
    return {"model": model, "info": info, "scheme": "fs",
            "k_adapt": k_adapt, "d_R": d_r, "d_phi": d_phi, "d_dec": d_dec}


def _clone_fixed(model: FixedDepthGNN, with_controller: bool) -> FixedDepthGNN:
    """Fresh FixedDepthGNN sharing NO storage with `model`, same weights."""
    in_dim = model.encoder[0].in_features
    hidden_dim = model.encoder[0].out_features
    out_dim = model.decoder[-1].out_features
    controller = FiLMController(hidden_dim) if with_controller else None
    clone = FixedDepthGNN(in_dim, hidden_dim=hidden_dim, out_dim=out_dim,
                          k_fix=model.k_fix, controller=controller)
    missing, unexpected = clone.load_state_dict(model.state_dict(), strict=False)
    assert not unexpected, unexpected
    assert all(k.startswith("controller.") for k in missing), missing
    return clone


def adapt_ft_fix(
    pretrained: FixedDepthGNN,
    adapt_set: List,
    val_set: List,
    epochs: int = 50,
    lr: float = 2.5e-4,
    device: str = "cuda",
    seed: int = 0,
    batch_size: int = 32,
    traj_sup: bool = False,
) -> Dict:
    """Tier-0 variant FT-fix: fine-tune ALL parameters of the fixed-depth
    model on D_adapt (the standard transfer-learning practice)."""
    model = _clone_fixed(pretrained, with_controller=False)
    info = train_model(
        model, adapt_set, epochs=epochs, lr=lr, batch_size=batch_size,
        k_sampler=(lambda rng: model.k_fix) if traj_sup else None,
        trainable="all", device=device,
        val_dataset=val_set, val_K=None, seed=seed, traj_sup=traj_sup,
    )
    return {"model": model, "info": info, "scheme": "ft_fix"}


def adapt_fs_fix(
    pretrained: FixedDepthGNN,
    adapt_set: List,
    val_set: List,
    epochs: int = 50,
    lr: float = 2.5e-4,
    device: str = "cuda",
    seed: int = 0,
    batch_size: int = 32,
    traj_sup: bool = False,
) -> Dict:
    """Fixed-Steer: freeze the fixed-depth model's encoder and layers; train
    a zero-init FiLM adapter inserted between its layers AND the readout."""
    model = _clone_fixed(pretrained, with_controller=True)
    info = train_model(
        model, adapt_set, epochs=epochs, lr=lr, batch_size=batch_size,
        k_sampler=(lambda rng: model.k_fix) if traj_sup else None,
        trainable="controller+decoder", device=device,
        val_dataset=val_set, val_K=None, seed=seed, traj_sup=traj_sup,
    )
    d_r = sum(p_.numel() for p_ in model.encoder.parameters()) +           sum(p_.numel() for p_ in model.blocks.parameters()) +           sum(p_.numel() for p_ in model.decoder.parameters())
    return {"model": model, "info": info, "scheme": "fs_fix",
            "d_R": d_r, "d_phi": model.controller.num_params()}
