"""Shared utilities for partial finetuning / PEFT-style component selection."""

from __future__ import annotations

from typing import Callable, Iterable, List, Optional

import torch.nn as nn


def normalize_component_list(value, key: str) -> List[str]:
    """Normalize a component config value into a deduplicated list of names."""
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        normalized = []
        for item in value:
            if not isinstance(item, str) or not item:
                raise ValueError(f"{key} entries must be non-empty strings")
            normalized.append(item)
        return list(dict.fromkeys(normalized))
    raise ValueError(f"{key} must be a string or list of strings")


def resolve_component_module(model: nn.Module, component_path: str) -> nn.Module:
    """Resolve a dotted component path against a module hierarchy."""
    current = model
    for part in component_path.split("."):
        if not part:
            raise ValueError(f"Invalid empty component path segment in '{component_path}'")

        if hasattr(current, part):
            current = getattr(current, part)
            continue

        if part.isdigit() and isinstance(current, (nn.ModuleList, nn.Sequential, list, tuple)):
            idx = int(part)
            try:
                current = current[idx]
            except IndexError as exc:
                raise ValueError(
                    f"Component path '{component_path}' index {idx} is out of range"
                ) from exc
            continue

        available = []
        if isinstance(current, nn.Module):
            available.extend(name for name, _ in current.named_children())
        raise ValueError(
            f"Could not resolve component path '{component_path}' at segment '{part}'. "
            f"Available children: {sorted(set(available))[:20]}"
        )

    if not isinstance(current, nn.Module):
        raise ValueError(f"Component path '{component_path}' does not resolve to an nn.Module")
    return current


def set_module_requires_grad(module: nn.Module, requires_grad: bool) -> int:
    """Set requires_grad for all parameters in a module and return the parameter count."""
    num_params = 0
    for param in module.parameters():
        param.requires_grad = requires_grad
        num_params += param.numel()
    return num_params


def count_parameters(model: nn.Module) -> tuple[int, int, int]:
    """Return total, trainable, and frozen parameter counts."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    return total_params, trainable_params, frozen_params


def configure_trainable_components(
    model: nn.Module,
    config: dict,
    log_print: Callable[..., None],
    *,
    force_trainable_components: Optional[Iterable[str]] = None,
) -> dict:
    """Apply component allowlist/blocklist rules from a config to a module tree."""
    freeze_components = normalize_component_list(
        config.get("freeze_components"), "freeze_components"
    )
    trainable_components = normalize_component_list(
        config.get("trainable_components"), "trainable_components"
    )
    only_train_components = normalize_component_list(
        config.get("only_train_components"), "only_train_components"
    )

    if trainable_components and only_train_components and trainable_components != only_train_components:
        raise ValueError(
            "Use either trainable_components or only_train_components, or make them identical"
        )

    selected_trainable = trainable_components or only_train_components
    forced = list(dict.fromkeys(force_trainable_components or []))
    for component_name in forced:
        if component_name in freeze_components:
            raise ValueError(
                f"{component_name} cannot be forced trainable while also frozen; "
                f"remove it from freeze_components"
            )
    if forced:
        selected_trainable = list(dict.fromkeys([*selected_trainable, *forced]))

    has_partial_training = bool(freeze_components or selected_trainable)
    if not has_partial_training:
        total_params, trainable_params, frozen_params = count_parameters(model)
        return {
            "freeze_components": freeze_components,
            "trainable_components": selected_trainable,
            "has_partial_training": False,
            "total_params": total_params,
            "trainable_params": trainable_params,
            "frozen_params": frozen_params,
        }

    log_print("\n" + "=" * 80)
    log_print("CONFIGURING TRAINABLE COMPONENTS")
    log_print("=" * 80)

    if selected_trainable:
        set_module_requires_grad(model, False)
        log_print(
            "  only/trainable components specified: froze all parameters first, "
            "then re-enabled the selected components."
        )
        if forced:
            log_print(
                "  forced trainable components: "
                + ", ".join(sorted(forced))
            )
        for component_name in selected_trainable:
            component = resolve_component_module(model, component_name)
            num_params = set_module_requires_grad(component, True)
            log_print(f"  ✓ Trainable '{component_name}': {num_params:,} parameters")

    if freeze_components:
        if selected_trainable:
            log_print("  Applying additional component freezes after the allowlist.")
        else:
            log_print("  Freezing selected components.")
        for component_name in freeze_components:
            component = resolve_component_module(model, component_name)
            num_params = set_module_requires_grad(component, False)
            log_print(f"  ✓ Frozen '{component_name}': {num_params:,} parameters")

    total_params, trainable_params, frozen_params = count_parameters(model)
    if trainable_params == 0:
        raise ValueError("No trainable parameters remain after applying component selection")

    log_print("-" * 80)
    log_print(f"  Total parameters:     {total_params:>12,}")
    log_print(f"  Frozen parameters:    {frozen_params:>12,}")
    log_print(f"  Trainable parameters: {trainable_params:>12,}")
    log_print("=" * 80 + "\n")

    return {
        "freeze_components": freeze_components,
        "trainable_components": selected_trainable,
        "has_partial_training": True,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "frozen_params": frozen_params,
    }
