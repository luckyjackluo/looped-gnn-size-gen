from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Literal, Optional, Sequence, Tuple

import torch


@dataclass
class TimeGridTechniqueConfig:
    enabled: bool = False
    strategy: Literal["uniform", "late_power", "quadratic", "karras", "piecewise"] = "uniform"
    power: float = 2.0
    rho: float = 7.0
    late_fraction: float = 0.1
    late_step_fraction: float = 0.3


@dataclass
class TimeRemapTechniqueConfig:
    enabled: bool = False
    strategy: Literal["none", "constant_sigma_scale", "adaptive_sigma_scale"] = "none"
    sigma_scale: float = 1.0
    adaptive_n0: float = 300.0
    adaptive_beta: float = 0.1
    adaptive_s_min: float = 0.7
    adaptive_t_start: float = 0.6


@dataclass
class PredictorCorrectorTechniqueConfig:
    enabled: bool = False
    method: Literal["euler", "heun"] = "euler"


@dataclass
class VelocityCalibrationTechniqueConfig:
    enabled: bool = False
    mode: Literal["constant", "piecewise_linear", "nearest"] = "piecewise_linear"
    constant_scale: float = 1.0
    time_points: List[float] = field(default_factory=list)
    scale_values: List[float] = field(default_factory=list)
    calibration_json_path: Optional[str] = None
    scale_multiplier: float = 1.0
    clamp_min: float = 0.0
    clamp_max: float = 3.0


@dataclass
class SDENoiseTechniqueConfig:
    enabled: bool = False
    sigma: float = 0.0
    schedule: Literal["constant", "late_gate", "linear_gate"] = "late_gate"
    t_start: float = 0.9
    t_end: float = 1.0


@dataclass
class RestartSamplingTechniqueConfig:
    enabled: bool = False
    t_start: float = 0.8
    t_end: float = 1.0
    num_restarts: int = 1
    noise_std: float = 0.02
    inner_steps: int = 16


@dataclass
class AdaptiveStepTechniqueConfig:
    enabled: bool = False
    atol: float = 1e-4
    rtol: float = 1e-3
    min_step_size: float = 0.0025
    max_step_size: float = 0.05
    safety: float = 0.9
    max_substeps_per_interval: int = 64


@dataclass
class SamplingTechniqueConfig:
    time_grid: TimeGridTechniqueConfig = field(default_factory=TimeGridTechniqueConfig)
    time_remap: TimeRemapTechniqueConfig = field(default_factory=TimeRemapTechniqueConfig)
    predictor_corrector: PredictorCorrectorTechniqueConfig = field(default_factory=PredictorCorrectorTechniqueConfig)
    velocity_calibration: VelocityCalibrationTechniqueConfig = field(default_factory=VelocityCalibrationTechniqueConfig)
    sde_noise: SDENoiseTechniqueConfig = field(default_factory=SDENoiseTechniqueConfig)
    restart: RestartSamplingTechniqueConfig = field(default_factory=RestartSamplingTechniqueConfig)
    adaptive_step: AdaptiveStepTechniqueConfig = field(default_factory=AdaptiveStepTechniqueConfig)


def _as_list(values: Optional[Sequence[float]]) -> List[float]:
    if values is None:
        return []
    return [float(v) for v in values]


def build_sampling_techniques_config(
    diffusion_cfg_dict: dict,
    sampling_cfg_dict: Optional[dict] = None,
) -> SamplingTechniqueConfig:
    sampling_cfg_dict = sampling_cfg_dict or {}

    legacy_warp = diffusion_cfg_dict.get("fm_time_warp", "none")
    legacy_warp_power = float(diffusion_cfg_dict.get("fm_time_warp_power", 1.0))
    time_grid_dict = sampling_cfg_dict.get("time_grid", {})
    if time_grid_dict:
        time_grid = TimeGridTechniqueConfig(
            enabled=bool(time_grid_dict.get("enabled", time_grid_dict.get("strategy", "uniform") != "uniform")),
            strategy=time_grid_dict.get("strategy", "uniform"),
            power=float(time_grid_dict.get("power", legacy_warp_power if legacy_warp == "late_power" else 2.0)),
            rho=float(time_grid_dict.get("rho", 7.0)),
            late_fraction=float(time_grid_dict.get("late_fraction", 0.1)),
            late_step_fraction=float(time_grid_dict.get("late_step_fraction", 0.3)),
        )
    else:
        strategy = "uniform" if legacy_warp == "none" else legacy_warp
        time_grid = TimeGridTechniqueConfig(
            enabled=strategy != "uniform",
            strategy=strategy,
            power=legacy_warp_power if legacy_warp_power > 0.0 else 1.0,
        )

    legacy_sigma_scale = float(diffusion_cfg_dict.get("sigma_scale", 1.0))
    legacy_use_adaptive = bool(diffusion_cfg_dict.get("use_adaptive_sigma_scale", False))
    time_remap_dict = sampling_cfg_dict.get("time_remap", {})
    if time_remap_dict:
        time_remap = TimeRemapTechniqueConfig(
            enabled=bool(time_remap_dict.get("enabled", time_remap_dict.get("strategy", "none") != "none")),
            strategy=time_remap_dict.get("strategy", "none"),
            sigma_scale=float(time_remap_dict.get("sigma_scale", legacy_sigma_scale)),
            adaptive_n0=float(time_remap_dict.get("adaptive_n0", diffusion_cfg_dict.get("adaptive_sigma_n0", 300.0))),
            adaptive_beta=float(time_remap_dict.get("adaptive_beta", diffusion_cfg_dict.get("adaptive_sigma_beta", 0.1))),
            adaptive_s_min=float(time_remap_dict.get("adaptive_s_min", diffusion_cfg_dict.get("adaptive_sigma_s_min", 0.7))),
            adaptive_t_start=float(time_remap_dict.get("adaptive_t_start", diffusion_cfg_dict.get("adaptive_sigma_t_start", 0.6))),
        )
    else:
        strategy = "adaptive_sigma_scale" if legacy_use_adaptive else ("constant_sigma_scale" if legacy_sigma_scale != 1.0 else "none")
        time_remap = TimeRemapTechniqueConfig(
            enabled=strategy != "none",
            strategy=strategy,
            sigma_scale=legacy_sigma_scale,
            adaptive_n0=float(diffusion_cfg_dict.get("adaptive_sigma_n0", 300.0)),
            adaptive_beta=float(diffusion_cfg_dict.get("adaptive_sigma_beta", 0.1)),
            adaptive_s_min=float(diffusion_cfg_dict.get("adaptive_sigma_s_min", 0.7)),
            adaptive_t_start=float(diffusion_cfg_dict.get("adaptive_sigma_t_start", 0.6)),
        )

    pc_dict = sampling_cfg_dict.get("predictor_corrector", {})
    legacy_solver = diffusion_cfg_dict.get("ode_solver", "euler")
    predictor_corrector = PredictorCorrectorTechniqueConfig(
        enabled=bool(pc_dict.get("enabled", legacy_solver == "heun")),
        method=pc_dict.get("method", legacy_solver if legacy_solver in ("euler", "heun") else "euler"),
    )

    velocity_dict = sampling_cfg_dict.get("velocity_calibration", {})
    velocity_calibration = VelocityCalibrationTechniqueConfig(
        enabled=bool(velocity_dict.get("enabled", False)),
        mode=velocity_dict.get("mode", "piecewise_linear"),
        constant_scale=float(velocity_dict.get("constant_scale", 1.0)),
        time_points=_as_list(velocity_dict.get("time_points")),
        scale_values=_as_list(velocity_dict.get("scale_values")),
        calibration_json_path=velocity_dict.get("calibration_json_path"),
        scale_multiplier=float(velocity_dict.get("scale_multiplier", 1.0)),
        clamp_min=float(velocity_dict.get("clamp_min", 0.0)),
        clamp_max=float(velocity_dict.get("clamp_max", 3.0)),
    )

    sde_dict = sampling_cfg_dict.get("sde_noise", {})
    sde_noise = SDENoiseTechniqueConfig(
        enabled=bool(sde_dict.get("enabled", False)),
        sigma=float(sde_dict.get("sigma", 0.0)),
        schedule=sde_dict.get("schedule", "late_gate"),
        t_start=float(sde_dict.get("t_start", 0.9)),
        t_end=float(sde_dict.get("t_end", 1.0)),
    )

    restart_dict = sampling_cfg_dict.get("restart", {})
    restart = RestartSamplingTechniqueConfig(
        enabled=bool(restart_dict.get("enabled", False)),
        t_start=float(restart_dict.get("t_start", 0.8)),
        t_end=float(restart_dict.get("t_end", 1.0)),
        num_restarts=int(restart_dict.get("num_restarts", 1)),
        noise_std=float(restart_dict.get("noise_std", 0.02)),
        inner_steps=int(restart_dict.get("inner_steps", 16)),
    )

    adaptive_dict = sampling_cfg_dict.get("adaptive_step", {})
    adaptive_step = AdaptiveStepTechniqueConfig(
        enabled=bool(adaptive_dict.get("enabled", False)),
        atol=float(adaptive_dict.get("atol", 1e-4)),
        rtol=float(adaptive_dict.get("rtol", 1e-3)),
        min_step_size=float(adaptive_dict.get("min_step_size", 0.0025)),
        max_step_size=float(adaptive_dict.get("max_step_size", 0.05)),
        safety=float(adaptive_dict.get("safety", 0.9)),
        max_substeps_per_interval=int(adaptive_dict.get("max_substeps_per_interval", 64)),
    )

    return SamplingTechniqueConfig(
        time_grid=time_grid,
        time_remap=time_remap,
        predictor_corrector=predictor_corrector,
        velocity_calibration=velocity_calibration,
        sde_noise=sde_noise,
        restart=restart,
        adaptive_step=adaptive_step,
    )


@lru_cache(maxsize=32)
def _load_velocity_calibration_table(calibration_json_path: str) -> Tuple[List[float], List[float]]:
    path = Path(calibration_json_path).expanduser()
    with path.open("r") as f:
        payload = json.load(f)

    if isinstance(payload, dict) and "per_t" in payload:
        entries = payload["per_t"]
    elif isinstance(payload, list):
        entries = payload
    else:
        raise ValueError(f"Unsupported calibration JSON format: {path}")

    time_points: List[float] = []
    scale_values: List[float] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        t_value = entry.get("t")
        if t_value is None:
            continue
        if "vector_vs_gt" in entry and isinstance(entry["vector_vs_gt"], dict):
            c_value = entry["vector_vs_gt"].get("c")
        else:
            c_value = entry.get("c")
        if c_value is None:
            continue
        time_points.append(float(t_value))
        scale_values.append(float(c_value))

    if not time_points or len(time_points) != len(scale_values):
        raise ValueError(f"No valid calibration table found in: {path}")

    pairs = sorted(zip(time_points, scale_values), key=lambda item: item[0])
    sorted_t = [item[0] for item in pairs]
    sorted_c = [item[1] for item in pairs]
    return sorted_t, sorted_c


class FlowMatchingTestTimeController:
    def __init__(
        self,
        techniques: SamplingTechniqueConfig,
        *,
        num_nodes: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        self.cfg = techniques
        self.num_nodes = int(num_nodes)
        self.device = device
        self.dtype = dtype

    def resolve_solver(self, legacy_solver: str) -> str:
        if self.cfg.predictor_corrector.enabled:
            return self.cfg.predictor_corrector.method
        return legacy_solver

    def build_time_grid(self, num_steps: int) -> torch.Tensor:
        u = torch.linspace(0.0, 1.0, num_steps + 1, device=self.device, dtype=self.dtype)
        cfg = self.cfg.time_grid
        if not cfg.enabled or cfg.strategy == "uniform":
            t_grid = u
        elif cfg.strategy == "late_power":
            power = max(cfg.power, 1.0)
            t_grid = 1.0 - (1.0 - u) ** power
        elif cfg.strategy == "quadratic":
            t_grid = 1.0 - (1.0 - u) ** 2
        elif cfg.strategy == "karras":
            rho = max(cfg.rho, 1.0)
            t_grid = 1.0 - (1.0 - u) ** rho
        elif cfg.strategy == "piecewise":
            late_fraction = min(max(cfg.late_fraction, 1e-4), 1.0 - 1e-4)
            late_step_fraction = min(max(cfg.late_step_fraction, 1e-4), 1.0 - 1e-4)
            split_idx = max(1, min(num_steps - 1, int(round(num_steps * (1.0 - late_step_fraction)))))
            early_grid = torch.linspace(
                0.0,
                1.0 - late_fraction,
                split_idx + 1,
                device=self.device,
                dtype=self.dtype,
            )
            late_grid = torch.linspace(
                1.0 - late_fraction,
                1.0,
                num_steps - split_idx + 1,
                device=self.device,
                dtype=self.dtype,
            )
            t_grid = torch.cat([early_grid[:-1], late_grid], dim=0)
        else:
            t_grid = u
        t_grid[0] = torch.tensor(0.0, device=self.device, dtype=self.dtype)
        t_grid[-1] = torch.tensor(1.0, device=self.device, dtype=self.dtype)
        return t_grid

    def remap_time(self, t_value: float) -> float:
        cfg = self.cfg.time_remap
        if not cfg.enabled or cfg.strategy == "none":
            return t_value

        if cfg.strategy == "adaptive_sigma_scale":
            log_ratio = torch.log(
                torch.clamp(
                    torch.tensor(self.num_nodes / cfg.adaptive_n0, device=self.device, dtype=torch.float32),
                    min=1e-6,
                )
            )
            s_n = torch.clamp(1.0 - cfg.adaptive_beta * log_ratio, min=cfg.adaptive_s_min, max=1.0)
            gate_raw = (t_value - cfg.adaptive_t_start) / max(1.0 - cfg.adaptive_t_start, 1e-6)
            gate = max(0.0, min(1.0, gate_raw))
            sigma_scale = float(1.0 - gate * (1.0 - float(s_n.item())))
        else:
            sigma_scale = cfg.sigma_scale

        denom = t_value + sigma_scale * (1.0 - t_value)
        if abs(denom) <= 1e-9:
            return t_value
        return float(t_value / denom)

    def velocity_scale(self, t_value: float) -> float:
        cfg = self.cfg.velocity_calibration
        if not cfg.enabled:
            return 1.0

        if cfg.calibration_json_path:
            time_points, scale_values = _load_velocity_calibration_table(cfg.calibration_json_path)
        else:
            time_points, scale_values = cfg.time_points, cfg.scale_values

        if cfg.mode == "constant" or not time_points or not scale_values:
            scale = cfg.constant_scale
        else:
            pairs = sorted(zip(time_points, scale_values), key=lambda item: item[0])
            t_sorted = [item[0] for item in pairs]
            c_sorted = [item[1] for item in pairs]
            if t_value <= t_sorted[0]:
                scale = c_sorted[0]
            elif t_value >= t_sorted[-1]:
                scale = c_sorted[-1]
            else:
                scale = c_sorted[-1]
                for idx in range(len(t_sorted) - 1):
                    t_lo, t_hi = t_sorted[idx], t_sorted[idx + 1]
                    if t_lo <= t_value <= t_hi:
                        c_lo, c_hi = c_sorted[idx], c_sorted[idx + 1]
                        if cfg.mode == "nearest":
                            scale = c_lo if abs(t_value - t_lo) <= abs(t_value - t_hi) else c_hi
                        else:
                            alpha = (t_value - t_lo) / max(t_hi - t_lo, 1e-8)
                            scale = (1.0 - alpha) * c_lo + alpha * c_hi
                        break

        scaled = cfg.scale_multiplier * scale
        return float(max(cfg.clamp_min, min(cfg.clamp_max, scaled)))

    def apply_velocity_scale(self, velocity: torch.Tensor, t_value: float) -> torch.Tensor:
        scale = self.velocity_scale(t_value)
        if scale == 1.0:
            return velocity
        return velocity * scale

    def sde_sigma(self, t_value: float) -> float:
        cfg = self.cfg.sde_noise
        if not cfg.enabled or cfg.sigma <= 0.0:
            return 0.0

        if cfg.schedule == "constant":
            return cfg.sigma

        if t_value < cfg.t_start or t_value > cfg.t_end:
            return 0.0

        if cfg.schedule == "late_gate":
            return cfg.sigma

        alpha = (t_value - cfg.t_start) / max(cfg.t_end - cfg.t_start, 1e-8)
        alpha = max(0.0, min(1.0, alpha))
        return cfg.sigma * alpha

    def apply_sde_noise(self, z_t: torch.Tensor, *, t_value: float, dt_value: float, mask: Optional[torch.Tensor]) -> torch.Tensor:
        sigma = self.sde_sigma(t_value)
        if sigma <= 0.0 or abs(dt_value) <= 0.0:
            return z_t
        noise = torch.randn_like(z_t) * (sigma * (abs(dt_value) ** 0.5))
        if mask is not None and mask.dtype == torch.bool and mask.shape[0] == z_t.shape[0]:
            z_next = z_t.clone()
            z_next[mask] = z_next[mask] + noise[mask]
            return z_next
        return z_t + noise
