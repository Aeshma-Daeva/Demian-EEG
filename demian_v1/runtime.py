"""Deterministic runtime and capsule boundary for Demian v1."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch

from development.demian_v1_gate_state import V1_CHANNELS, DemianV1GateState, V1State

DEMIAN_V1_ID = "demian-v1"
TRISEQUENCE_TENSION_VARIANTS = {
    "demian_v1_trisequence_tension",
    "demian_v1_trisequence_tension_v2",
    "demian_v1_trisequence_tension_v3",
    "demian_v1_trisequence_tension_v3_no_trisequence",
    "demian_v1_trisequence_tension_v3_no_self_other",
    "demian_v1_trisequence_tension_v3_no_tension",
    "demian_v1_trisequence_tension_v3_no_global_compass",
    "demian_v1_trisequence_tension_v3_no_slow_boost",
    "demian_v1_trisequence_tension_v3_no_fast_suppression",
}
ROUTE_EFFICIENCY_VARIANTS = {
    "demian_v1_trisequence_tension_v2",
    "demian_v1_trisequence_tension_v3",
    "demian_v1_trisequence_tension_v3_no_trisequence",
    "demian_v1_trisequence_tension_v3_no_self_other",
    "demian_v1_trisequence_tension_v3_no_tension",
    "demian_v1_trisequence_tension_v3_no_global_compass",
    "demian_v1_trisequence_tension_v3_no_slow_boost",
    "demian_v1_trisequence_tension_v3_no_fast_suppression",
}
GLOBAL_ROUTE_VARIANTS = ROUTE_EFFICIENCY_VARIANTS - {
    "demian_v1_trisequence_tension_v2",
    "demian_v1_trisequence_tension_v3_no_global_compass",
}


@dataclass(frozen=True)
class DemianV1Config:
    """Stable construction parameters for a Demian v1 runtime."""

    hidden_size: int = 32
    seed: int = 0
    variant: str = "demian_v1_baseline"
    gate_disabled: bool = False
    gate_frozen: bool = False
    binding_start_step: int = 1
    fast_coupling_scale: float = 0.35
    slow_coupling_scale: float = 0.12
    slow_coupling_alpha: float = 0.08
    predictive_gate_scale: float = 0.18
    topographic_region_count: int = 4
    route_compass_window: int = 8
    route_tortuosity_target: float = 2.0
    route_compass_gain: float = 0.12
    min_fast_coupling_scale: float = 0.08
    max_fast_coupling_scale: float = 0.60
    min_slow_coupling_scale: float = 0.03
    max_slow_coupling_scale: float = 0.24
    trisequence_memory_alpha: float = 0.21
    trisequence_self_alpha: float = 0.13
    trisequence_future_scale: float = 0.34
    trisequence_self_other_scale: float = 0.08
    tension_uncertainty_alpha: float = 0.21
    tension_decay: float = 0.89
    tension_release_gain: float = 0.11
    tension_precision_alpha: float = 0.13
    tension_fast_damping: float = 0.55
    tension_route_release_suppression: float = 0.65
    tension_route_fast_suppression: float = 0.80
    tension_route_slow_boost: float = 0.35
    tension_global_tortuosity_target: float = 1.8
    tension_global_compass_gain: float = 0.35


@dataclass(frozen=True)
class DemianV1Snapshot:
    """JSON-compatible recurrent state capsule."""

    runtime_id: str
    config: dict[str, Any]
    step_index: int
    channels: dict[str, list[list[float]]]
    model_state: dict[str, list[float]]
    variant_state: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def serialize_state(state: V1State) -> dict[str, list[list[float]]]:
    """Serialize all six recurrent channels without losing continuity."""

    return {
        channel: tensor.detach().cpu().tolist()
        for channel, tensor in zip(V1_CHANNELS, state, strict=True)
    }


def deserialize_state(
    payload: dict[str, list[list[float]]],
    *,
    device: torch.device | None = None,
) -> V1State:
    """Restore all six recurrent channels from a capsule payload."""

    target = device or torch.device("cpu")
    missing = [channel for channel in V1_CHANNELS if channel not in payload]
    if missing:
        raise ValueError(f"demian_v1_channels_missing:{','.join(missing)}")
    return tuple(
        torch.tensor(payload[channel], dtype=torch.float32, device=target)
        for channel in V1_CHANNELS
    )  # type: ignore[return-value]


class DemianV1Runtime:
    """Small deterministic wrapper suitable for embedding in another runtime."""

    def __init__(self, config: DemianV1Config | None = None) -> None:
        self.config = config or DemianV1Config()
        if self.config.hidden_size < 2:
            raise ValueError("demian_v1_hidden_size_must_be_at_least_two")
        if self.config.variant not in DEMIAN_V1_VARIANTS:
            raise ValueError(f"demian_v1_unknown_variant:{self.config.variant}")
        torch.manual_seed(self.config.seed)
        self.model = DemianV1GateState(
            hidden_size=self.config.hidden_size,
            gate_disabled=self.config.gate_disabled,
            gate_frozen=self.config.gate_frozen,
            binding_start_step=self.config.binding_start_step,
        )
        self.model.eval()
        self.state = self.model.initial_state(1, torch.device("cpu"))
        self.step_index = 0
        self._previous_coupling: torch.Tensor | None = None
        self._slow_coupling: torch.Tensor | None = None
        self._route_surfaces: list[list[float]] = []
        self._adaptive_fast_scale = float(self.config.fast_coupling_scale)
        self._adaptive_slow_scale = float(self.config.slow_coupling_scale)
        self._route_compass_correction = 0.0
        self._past_coupling: torch.Tensor | None = None
        self._future_coupling: torch.Tensor | None = None
        self._self_trace: torch.Tensor | None = None
        self._other_trace: torch.Tensor | None = None
        self._uncertainty_trace: torch.Tensor | None = None
        self._tension_reservoir: torch.Tensor | None = None
        self._precision_trace = 0.5
        self._previous_uncertainty_norm: float | None = None
        self._route_origin_surface: torch.Tensor | None = None
        self._route_previous_surface: torch.Tensor | None = None
        self._route_cumulative_path_length = 0.0
        self._route_global_tortuosity = 1.0
        self._route_global_compass_correction = 0.0

    def step(self, coupling: torch.Tensor | None = None, *, strength: float = 1.0) -> dict[str, Any]:
        """Advance once, optionally injecting a bounded coupling vector."""

        variant_metrics = self._zero_variant_metrics()
        if coupling is not None:
            vector = coupling.detach().to(dtype=torch.float32, device=torch.device("cpu")).view(1, -1)
            if vector.shape[-1] != self.config.hidden_size:
                raise ValueError("demian_v1_coupling_size_mismatch")
            self.state, variant_metrics = self._inject_variant_coupling(vector, float(strength))
        with torch.no_grad():
            self.state = self.model.step(self.state)
        self.step_index += 1
        surface = self.model.state_vector(self.state).view(-1).detach().cpu()
        compass_metrics = self._update_route_compass(surface)
        metrics = dict(self.model.step_aux())
        metrics.update(variant_metrics)
        metrics.update(compass_metrics)
        return {
            "runtime_id": DEMIAN_V1_ID,
            "variant": self.config.variant,
            "step_index": self.step_index,
            "surface": surface.tolist(),
            "metrics": metrics,
        }

    def snapshot(self) -> DemianV1Snapshot:
        """Capture model parameters and full recurrent state."""

        return DemianV1Snapshot(
            runtime_id=DEMIAN_V1_ID,
            config=asdict(self.config),
            step_index=self.step_index,
            channels=serialize_state(self.state),
            model_state={
                name: tensor.detach().cpu().reshape(-1).tolist()
                for name, tensor in self.model.state_dict().items()
            },
            variant_state={
                "previous_coupling": self._previous_coupling.detach().cpu().tolist()
                if self._previous_coupling is not None
                else None,
                "slow_coupling": self._slow_coupling.detach().cpu().tolist()
                if self._slow_coupling is not None
                else None,
                "route_surfaces": self._route_surfaces,
                "adaptive_fast_scale": self._adaptive_fast_scale,
                "adaptive_slow_scale": self._adaptive_slow_scale,
                "route_compass_correction": self._route_compass_correction,
                "past_coupling": self._past_coupling.detach().cpu().tolist()
                if self._past_coupling is not None
                else None,
                "future_coupling": self._future_coupling.detach().cpu().tolist()
                if self._future_coupling is not None
                else None,
                "self_trace": self._self_trace.detach().cpu().tolist()
                if self._self_trace is not None
                else None,
                "other_trace": self._other_trace.detach().cpu().tolist()
                if self._other_trace is not None
                else None,
                "uncertainty_trace": self._uncertainty_trace.detach().cpu().tolist()
                if self._uncertainty_trace is not None
                else None,
                "tension_reservoir": self._tension_reservoir.detach().cpu().tolist()
                if self._tension_reservoir is not None
                else None,
                "precision_trace": self._precision_trace,
                "previous_uncertainty_norm": self._previous_uncertainty_norm,
                "route_origin_surface": self._route_origin_surface.detach().cpu().tolist()
                if self._route_origin_surface is not None
                else None,
                "route_previous_surface": self._route_previous_surface.detach().cpu().tolist()
                if self._route_previous_surface is not None
                else None,
                "route_cumulative_path_length": self._route_cumulative_path_length,
                "route_global_tortuosity": self._route_global_tortuosity,
                "route_global_compass_correction": self._route_global_compass_correction,
            },
        )

    def restore(self, snapshot: DemianV1Snapshot | dict[str, Any], *, surface_only: bool = False) -> None:
        """Restore full continuity or an explicit surface-only control."""

        payload = snapshot.to_dict() if isinstance(snapshot, DemianV1Snapshot) else dict(snapshot)
        if payload.get("runtime_id") != DEMIAN_V1_ID:
            raise ValueError("demian_v1_runtime_id_mismatch")
        channels = deserialize_state(dict(payload.get("channels") or {}))
        if surface_only:
            surface = self.model.state_vector(channels).detach()
            channels = (
                surface,
                *(torch.zeros_like(channel) for channel in channels[1:]),
            )
        self.state = channels
        self.step_index = max(0, int(payload.get("step_index") or 0))
        self.model._step_index = self.step_index
        variant_state = dict(payload.get("variant_state") or {})
        self._previous_coupling = None if surface_only else _optional_tensor(variant_state.get("previous_coupling"))
        self._slow_coupling = None if surface_only else _optional_tensor(variant_state.get("slow_coupling"))
        self._route_surfaces = [] if surface_only else [list(row) for row in variant_state.get("route_surfaces") or []]
        self._adaptive_fast_scale = (
            float(self.config.fast_coupling_scale)
            if surface_only
            else float(variant_state.get("adaptive_fast_scale", self.config.fast_coupling_scale))
        )
        self._adaptive_slow_scale = (
            float(self.config.slow_coupling_scale)
            if surface_only
            else float(variant_state.get("adaptive_slow_scale", self.config.slow_coupling_scale))
        )
        self._route_compass_correction = (
            0.0 if surface_only else float(variant_state.get("route_compass_correction", 0.0))
        )
        self._past_coupling = None if surface_only else _optional_tensor(variant_state.get("past_coupling"))
        self._future_coupling = None if surface_only else _optional_tensor(variant_state.get("future_coupling"))
        self._self_trace = None if surface_only else _optional_tensor(variant_state.get("self_trace"))
        self._other_trace = None if surface_only else _optional_tensor(variant_state.get("other_trace"))
        self._uncertainty_trace = None if surface_only else _optional_tensor(variant_state.get("uncertainty_trace"))
        self._tension_reservoir = None if surface_only else _optional_tensor(variant_state.get("tension_reservoir"))
        self._precision_trace = 0.5 if surface_only else float(variant_state.get("precision_trace", 0.5))
        previous_uncertainty = variant_state.get("previous_uncertainty_norm")
        self._previous_uncertainty_norm = None if surface_only or previous_uncertainty is None else float(previous_uncertainty)
        self._route_origin_surface = None if surface_only else _optional_tensor(variant_state.get("route_origin_surface"))
        self._route_previous_surface = None if surface_only else _optional_tensor(variant_state.get("route_previous_surface"))
        self._route_cumulative_path_length = (
            0.0 if surface_only else float(variant_state.get("route_cumulative_path_length", 0.0))
        )
        self._route_global_tortuosity = (
            1.0 if surface_only else float(variant_state.get("route_global_tortuosity", 1.0))
        )
        self._route_global_compass_correction = (
            0.0 if surface_only else float(variant_state.get("route_global_compass_correction", 0.0))
        )

    def _inject_variant_coupling(self, vector: torch.Tensor, strength: float) -> tuple[V1State, dict[str, float]]:
        variant = self.config.variant
        previous = self._previous_coupling
        if previous is None:
            fast_component = vector
        else:
            fast_component = vector - previous
        if self._slow_coupling is None:
            self._slow_coupling = vector.clone()
        else:
            alpha = _clamp(float(self.config.slow_coupling_alpha), 0.0, 1.0)
            self._slow_coupling = (1.0 - alpha) * self._slow_coupling + alpha * vector

        if variant == "demian_v1_baseline":
            state = self.model.inject_coupling_message(self.state, vector, strength)
            metrics = self._zero_variant_metrics()
        elif variant == "demian_v1_multitimescale":
            fast, slow, control, message, carrier, gate = self.state
            fast_delta = strength * self.config.fast_coupling_scale * fast_component
            slow_delta = strength * self.config.slow_coupling_scale * self._slow_coupling
            state = fast + fast_delta, slow + slow_delta, control, message, carrier, gate
            metrics = {
                **self._zero_variant_metrics(),
                "variant_fast_injection_norm": float(torch.norm(fast_delta).item()),
                "variant_slow_injection_norm": float(torch.norm(slow_delta).item()),
                "variant_fast_slow_ratio": _ratio(torch.norm(fast_delta), torch.norm(slow_delta)),
            }
        elif variant == "demian_v1_multitimescale_v2":
            fast, slow, control, message, carrier, gate = self.state
            fast_delta = strength * self._adaptive_fast_scale * fast_component
            slow_delta = strength * self._adaptive_slow_scale * self._slow_coupling
            state = fast + fast_delta, slow + slow_delta, control, message, carrier, gate
            metrics = {
                **self._zero_variant_metrics(),
                "variant_fast_injection_norm": float(torch.norm(fast_delta).item()),
                "variant_slow_injection_norm": float(torch.norm(slow_delta).item()),
                "variant_fast_slow_ratio": _ratio(torch.norm(fast_delta), torch.norm(slow_delta)),
                "variant_adaptive_fast_scale": self._adaptive_fast_scale,
                "variant_adaptive_slow_scale": self._adaptive_slow_scale,
                "variant_route_compass_correction": self._route_compass_correction,
            }
        elif variant == "demian_v1_predictive_gate":
            fast, slow, control, message, carrier, gate = self.model.inject_coupling_message(
                self.state, vector, strength
            )
            error = torch.zeros_like(vector) if previous is None else vector - previous
            gate_delta = strength * self.config.predictive_gate_scale * torch.tanh(error)
            state = fast, slow, control, message, carrier, gate + gate_delta
            metrics = {
                **self._zero_variant_metrics(),
                "variant_prediction_error_norm": float(torch.norm(error).item()),
                "variant_gate_pressure_norm": float(torch.norm(gate_delta).item()),
            }
        elif variant == "demian_v1_topographic_state":
            fast, slow, control, message, carrier, gate = self.state
            regions = _region_means(vector, self.config.topographic_region_count)
            expanded = _expand_regions(regions, self.config.hidden_size)
            centered = vector - expanded
            state = (
                fast + strength * centered,
                slow + strength * 0.10 * expanded,
                control,
                message + strength * 0.08 * centered,
                carrier + strength * 0.05 * expanded,
                gate,
            )
            metrics = {
                **self._zero_variant_metrics(),
                "variant_region_count": float(regions.shape[-1]),
                "variant_topographic_mean_norm": float(torch.norm(expanded).item()),
                "variant_topographic_residual_norm": float(torch.norm(centered).item()),
            }
        elif variant == "demian_v1_trisequence_fibonacci":
            state, metrics = self._inject_trisequence_fibonacci(vector, fast_component, strength)
        elif variant in TRISEQUENCE_TENSION_VARIANTS:
            state, metrics = self._inject_trisequence_tension(vector, fast_component, strength)
        else:  # pragma: no cover - guarded in __init__
            raise ValueError(f"demian_v1_unknown_variant:{variant}")

        self._previous_coupling = vector.clone()
        return state, metrics

    def _zero_variant_metrics(self) -> dict[str, float]:
        return {
            "variant_fast_injection_norm": 0.0,
            "variant_slow_injection_norm": 0.0,
            "variant_fast_slow_ratio": 0.0,
            "variant_prediction_error_norm": 0.0,
            "variant_gate_pressure_norm": 0.0,
            "variant_region_count": 0.0,
            "variant_topographic_mean_norm": 0.0,
            "variant_topographic_residual_norm": 0.0,
            "variant_adaptive_fast_scale": float(self.config.fast_coupling_scale),
            "variant_adaptive_slow_scale": float(self.config.slow_coupling_scale),
            "variant_route_recent_path_length": 0.0,
            "variant_route_recent_net_displacement": 0.0,
            "variant_route_recent_tortuosity": 1.0,
            "variant_route_compass_correction": 0.0,
            "variant_trisequence_past_norm": 0.0,
            "variant_trisequence_current_norm": 0.0,
            "variant_trisequence_future_norm": 0.0,
            "variant_trisequence_future_error_norm": 0.0,
            "variant_trisequence_self_norm": 0.0,
            "variant_trisequence_other_norm": 0.0,
            "variant_trisequence_self_other_l2": 0.0,
            "variant_trisequence_self_other_cosine": 0.0,
            "variant_tension_uncertainty_norm": 0.0,
            "variant_tension_reservoir_norm": 0.0,
            "variant_tension_release_gate": 0.0,
            "variant_tension_damping_gate": 1.0,
            "variant_tension_precision": 0.5,
            "variant_tension_stored_norm": 0.0,
            "variant_tension_released_norm": 0.0,
            "variant_tension_route_efficiency_gate": 1.0,
            "variant_tension_route_slow_boost": 0.0,
            "variant_route_global_path_length": 0.0,
            "variant_route_global_net_displacement": 0.0,
            "variant_route_global_tortuosity": 1.0,
            "variant_route_global_compass_correction": 0.0,
        }

    def _update_route_compass(self, surface: torch.Tensor) -> dict[str, float]:
        if self.config.variant not in {"demian_v1_multitimescale_v2", *ROUTE_EFFICIENCY_VARIANTS}:
            return {
                "variant_route_recent_path_length": 0.0,
                "variant_route_recent_net_displacement": 0.0,
                "variant_route_recent_tortuosity": 1.0,
                "variant_route_compass_correction": 0.0,
                "variant_adaptive_fast_scale": self._adaptive_fast_scale,
                "variant_adaptive_slow_scale": self._adaptive_slow_scale,
                "variant_route_global_path_length": 0.0,
                "variant_route_global_net_displacement": 0.0,
                "variant_route_global_tortuosity": 1.0,
                "variant_route_global_compass_correction": 0.0,
            }

        global_metrics = self._update_global_route_compass(surface)
        self._route_surfaces.append(surface.tolist())
        window = max(2, int(self.config.route_compass_window))
        self._route_surfaces = self._route_surfaces[-window:]
        recent = torch.tensor(self._route_surfaces, dtype=torch.float32)
        if recent.shape[0] < 2:
            return {
                "variant_route_recent_path_length": 0.0,
                "variant_route_recent_net_displacement": 0.0,
                "variant_route_recent_tortuosity": 1.0,
                "variant_route_compass_correction": self._route_compass_correction,
                "variant_adaptive_fast_scale": self._adaptive_fast_scale,
                "variant_adaptive_slow_scale": self._adaptive_slow_scale,
                **global_metrics,
            }

        steps = torch.norm(torch.diff(recent, dim=0), dim=1)
        path_length = float(torch.sum(steps).item())
        net_displacement = float(torch.norm(recent[-1] - recent[0]).item())
        tortuosity = path_length / max(net_displacement, 1.0e-12)
        target = max(1.0, float(self.config.route_tortuosity_target))
        gain = max(0.0, float(self.config.route_compass_gain))
        excess = max(0.0, tortuosity - target) / target
        collapse = path_length < 1.0e-6

        if collapse:
            correction = -gain
        else:
            correction = min(0.5, gain * excess)
        if self.config.variant in GLOBAL_ROUTE_VARIANTS:
            correction = self._route_global_compass_correction
        self._route_compass_correction = float(correction)

        if collapse:
            next_fast = self._adaptive_fast_scale * (1.0 + gain)
            next_slow = self._adaptive_slow_scale
        else:
            next_fast = self._adaptive_fast_scale * (1.0 - correction)
            next_slow = self._adaptive_slow_scale * (1.0 + 0.5 * correction)
        self._adaptive_fast_scale = _clamp(
            next_fast,
            float(self.config.min_fast_coupling_scale),
            float(self.config.max_fast_coupling_scale),
        )
        self._adaptive_slow_scale = _clamp(
            next_slow,
            float(self.config.min_slow_coupling_scale),
            float(self.config.max_slow_coupling_scale),
        )
        return {
            "variant_route_recent_path_length": path_length,
            "variant_route_recent_net_displacement": net_displacement,
            "variant_route_recent_tortuosity": tortuosity,
            "variant_route_compass_correction": self._route_compass_correction,
            "variant_adaptive_fast_scale": self._adaptive_fast_scale,
            "variant_adaptive_slow_scale": self._adaptive_slow_scale,
            **global_metrics,
        }

    def _update_global_route_compass(self, surface: torch.Tensor) -> dict[str, float]:
        if self.config.variant not in GLOBAL_ROUTE_VARIANTS:
            return {
                "variant_route_global_path_length": 0.0,
                "variant_route_global_net_displacement": 0.0,
                "variant_route_global_tortuosity": 1.0,
                "variant_route_global_compass_correction": 0.0,
            }
        current = surface.detach().cpu()
        if self._route_origin_surface is None:
            self._route_origin_surface = current.clone()
            self._route_previous_surface = current.clone()
            self._route_cumulative_path_length = 0.0
            self._route_global_tortuosity = 1.0
            self._route_global_compass_correction = 0.0
        else:
            previous = self._route_previous_surface if self._route_previous_surface is not None else current
            self._route_cumulative_path_length += float(torch.norm(current - previous).item())
            self._route_previous_surface = current.clone()
            net_displacement = float(torch.norm(current - self._route_origin_surface).item())
            self._route_global_tortuosity = self._route_cumulative_path_length / max(net_displacement, 1.0e-12)
            target = max(1.0, float(self.config.tension_global_tortuosity_target))
            gain = max(0.0, float(self.config.tension_global_compass_gain))
            excess = max(0.0, self._route_global_tortuosity - target) / target
            self._route_global_compass_correction = min(1.0, gain * excess)
        net = (
            0.0
            if self._route_origin_surface is None
            else float(torch.norm(current - self._route_origin_surface).item())
        )
        return {
            "variant_route_global_path_length": self._route_cumulative_path_length,
            "variant_route_global_net_displacement": net,
            "variant_route_global_tortuosity": self._route_global_tortuosity,
            "variant_route_global_compass_correction": self._route_global_compass_correction,
        }

    def _inject_trisequence_fibonacci(
        self,
        vector: torch.Tensor,
        fast_component: torch.Tensor,
        strength: float,
    ) -> tuple[V1State, dict[str, float]]:
        fast, slow, control, message, carrier, gate = self.state
        if self._past_coupling is None:
            self._past_coupling = vector.clone()
        else:
            alpha = _clamp(float(self.config.trisequence_memory_alpha), 0.0, 1.0)
            self._past_coupling = (1.0 - alpha) * self._past_coupling + alpha * vector

        surface = self.model.state_vector(self.state).detach()
        if self._self_trace is None:
            self._self_trace = surface.clone()
        else:
            alpha = _clamp(float(self.config.trisequence_self_alpha), 0.0, 1.0)
            self._self_trace = (1.0 - alpha) * self._self_trace + alpha * surface

        previous_forecast = torch.zeros_like(vector) if self._future_coupling is None else self._future_coupling
        phi_inverse = 0.6180339887498948
        phi_inverse_squared = 1.0 - phi_inverse
        trend = phi_inverse * fast_component + phi_inverse_squared * (vector - self._past_coupling)
        future = vector + float(self.config.trisequence_future_scale) * trend
        future_error = vector - previous_forecast
        self._future_coupling = future.clone()

        self_signal = torch.tanh(self._self_trace)
        other = vector - self_signal
        if self._other_trace is None:
            self._other_trace = other.clone()
        else:
            alpha = _clamp(float(self.config.trisequence_memory_alpha), 0.0, 1.0)
            self._other_trace = (1.0 - alpha) * self._other_trace + alpha * other

        past_weight, current_weight, future_weight = _normalized_fibonacci_triplet()
        past_delta = strength * self.config.slow_coupling_scale * past_weight * self._past_coupling
        current_delta = strength * self.config.fast_coupling_scale * current_weight * fast_component
        future_delta = strength * self.config.predictive_gate_scale * future_weight * torch.tanh(future_error)
        self_other_delta = (
            strength
            * float(self.config.trisequence_self_other_scale)
            * torch.tanh(self._other_trace - self_signal)
        )
        control_delta = _match_width(self_other_delta, control.shape[-1])
        carrier_self = _match_width(self_signal, carrier.shape[-1])
        state = (
            fast + current_delta + 0.5 * future_delta,
            slow + past_delta,
            control + 0.25 * control_delta,
            message + self_other_delta,
            carrier + strength * float(self.config.trisequence_self_other_scale) * carrier_self,
            gate + future_delta,
        )
        self_norm = torch.norm(self_signal)
        other_norm = torch.norm(self._other_trace)
        cosine_den = float((self_norm * other_norm).item())
        cosine = (
            float(torch.sum(self_signal * self._other_trace).item()) / cosine_den
            if abs(cosine_den) > 1.0e-12
            else 0.0
        )
        metrics = {
            **self._zero_variant_metrics(),
            "variant_fast_injection_norm": float(torch.norm(current_delta).item()),
            "variant_slow_injection_norm": float(torch.norm(past_delta).item()),
            "variant_fast_slow_ratio": _ratio(torch.norm(current_delta), torch.norm(past_delta)),
            "variant_prediction_error_norm": float(torch.norm(future_error).item()),
            "variant_gate_pressure_norm": float(torch.norm(future_delta).item()),
            "variant_trisequence_past_norm": float(torch.norm(self._past_coupling).item()),
            "variant_trisequence_current_norm": float(torch.norm(vector).item()),
            "variant_trisequence_future_norm": float(torch.norm(future).item()),
            "variant_trisequence_future_error_norm": float(torch.norm(future_error).item()),
            "variant_trisequence_self_norm": float(self_norm.item()),
            "variant_trisequence_other_norm": float(other_norm.item()),
            "variant_trisequence_self_other_l2": float(torch.norm(self_signal - self._other_trace).item()),
            "variant_trisequence_self_other_cosine": cosine,
        }
        return state, metrics

    def _inject_trisequence_tension(
        self,
        vector: torch.Tensor,
        fast_component: torch.Tensor,
        strength: float,
    ) -> tuple[V1State, dict[str, float]]:
        fast, slow, control, message, carrier, gate = self.state
        components = self._update_trisequence_components(vector, fast_component)
        if self.config.variant == "demian_v1_trisequence_tension_v3_no_trisequence":
            components["past_coupling"] = vector
            components["future_error"] = torch.zeros_like(vector)
            components["metrics"]["variant_trisequence_past_norm"] = 0.0
            components["metrics"]["variant_trisequence_future_norm"] = 0.0
            components["metrics"]["variant_trisequence_future_error_norm"] = 0.0
        if self.config.variant == "demian_v1_trisequence_tension_v3_no_self_other":
            components["other_trace"] = components["self_signal"]
            components["self_other_cosine"] = 1.0
            components["metrics"]["variant_trisequence_other_norm"] = float(torch.norm(components["self_signal"]).item())
            components["metrics"]["variant_trisequence_self_other_l2"] = 0.0
            components["metrics"]["variant_trisequence_self_other_cosine"] = 1.0
        uncertainty_raw = torch.tanh(components["future_error"]) + torch.tanh(
            components["other_trace"] - components["self_signal"]
        )
        alpha = _clamp(float(self.config.tension_uncertainty_alpha), 0.0, 1.0)
        if self._uncertainty_trace is None:
            self._uncertainty_trace = uncertainty_raw.clone()
        else:
            self._uncertainty_trace = (1.0 - alpha) * self._uncertainty_trace + alpha * uncertainty_raw

        uncertainty_norm = float(torch.norm(self._uncertainty_trace).item())
        previous_norm = uncertainty_norm if self._previous_uncertainty_norm is None else self._previous_uncertainty_norm
        resolving = _clamp((previous_norm - uncertainty_norm) / max(previous_norm, 1.0e-6), 0.0, 1.0)
        alignment = _clamp(0.5 * (components["self_other_cosine"] + 1.0), 0.0, 1.0)
        precision_observation = _clamp(0.65 * resolving + 0.35 * alignment, 0.0, 1.0)
        precision_alpha = _clamp(float(self.config.tension_precision_alpha), 0.0, 1.0)
        self._precision_trace = (
            (1.0 - precision_alpha) * self._precision_trace + precision_alpha * precision_observation
        )
        release_gate = _clamp(self._precision_trace * (0.25 + 0.75 * alignment), 0.0, 1.0)
        damping_gate = 1.0 / (1.0 + max(0.0, float(self.config.tension_fast_damping)) * uncertainty_norm)
        route_efficiency_gate = 1.0
        route_slow_boost = 0.0
        if self.config.variant in ROUTE_EFFICIENCY_VARIANTS:
            route_correction = _clamp(self._route_compass_correction, 0.0, 1.0)
            release_gate *= 1.0 / (
                1.0 + max(0.0, float(self.config.tension_route_release_suppression)) * route_correction
            )
            if self.config.variant == "demian_v1_trisequence_tension_v3_no_fast_suppression":
                route_efficiency_gate = 1.0
            else:
                route_efficiency_gate = 1.0 / (
                    1.0 + max(0.0, float(self.config.tension_route_fast_suppression)) * route_correction
                )
            damping_gate *= route_efficiency_gate
            if self.config.variant == "demian_v1_trisequence_tension_v3_no_slow_boost":
                route_slow_boost = 0.0
            else:
                route_slow_boost = max(0.0, float(self.config.tension_route_slow_boost)) * route_correction

        if self.config.variant == "demian_v1_trisequence_tension_v3_no_tension":
            release_gate = 1.0
            self._precision_trace = 1.0
            self._tension_reservoir = torch.zeros_like(vector)
        if self._tension_reservoir is None:
            self._tension_reservoir = torch.zeros_like(vector)
        decay = _clamp(float(self.config.tension_decay), 0.0, 1.0)
        if self.config.variant == "demian_v1_trisequence_tension_v3_no_tension":
            stored = torch.zeros_like(vector)
            release_delta = strength * float(self.config.tension_release_gain) * torch.tanh(self._uncertainty_trace)
            self._tension_reservoir = torch.zeros_like(vector)
        else:
            stored = (1.0 - release_gate) * self._uncertainty_trace
            self._tension_reservoir = decay * self._tension_reservoir + stored
            release_delta = (
                strength
                * float(self.config.tension_release_gain)
                * release_gate
                * torch.tanh(self._tension_reservoir)
            )
            self._tension_reservoir = (1.0 - 0.5 * release_gate) * self._tension_reservoir
        self._previous_uncertainty_norm = uncertainty_norm

        past_weight, current_weight, future_weight = _normalized_fibonacci_triplet()
        past_delta = (
            strength
            * self.config.slow_coupling_scale
            * past_weight
            * (1.0 + route_slow_boost)
            * components["past_coupling"]
        )
        current_delta = (
            strength
            * self.config.fast_coupling_scale
            * current_weight
            * damping_gate
            * fast_component
        )
        forecast_delta = (
            strength
            * self.config.predictive_gate_scale
            * future_weight
            * release_gate
            * torch.tanh(components["future_error"])
        )
        self_other_delta = release_delta
        control_delta = _match_width(self_other_delta, control.shape[-1])
        carrier_self = _match_width(components["self_signal"], carrier.shape[-1])
        state = (
            fast + current_delta + 0.25 * forecast_delta,
            slow + past_delta + 0.25 * release_delta,
            control + 0.25 * control_delta,
            message + self_other_delta,
            carrier + strength * float(self.config.trisequence_self_other_scale) * release_gate * carrier_self,
            gate + forecast_delta,
        )
        metrics = {
            **self._zero_variant_metrics(),
            **components["metrics"],
            "variant_fast_injection_norm": float(torch.norm(current_delta).item()),
            "variant_slow_injection_norm": float(torch.norm(past_delta).item()),
            "variant_fast_slow_ratio": _ratio(torch.norm(current_delta), torch.norm(past_delta)),
            "variant_prediction_error_norm": components["metrics"]["variant_trisequence_future_error_norm"],
            "variant_gate_pressure_norm": float(torch.norm(forecast_delta).item()),
            "variant_tension_uncertainty_norm": uncertainty_norm,
            "variant_tension_reservoir_norm": float(torch.norm(self._tension_reservoir).item()),
            "variant_tension_release_gate": release_gate,
            "variant_tension_damping_gate": damping_gate,
            "variant_tension_precision": self._precision_trace,
            "variant_tension_stored_norm": float(torch.norm(stored).item()),
            "variant_tension_released_norm": float(torch.norm(release_delta).item()),
            "variant_tension_route_efficiency_gate": route_efficiency_gate,
            "variant_tension_route_slow_boost": route_slow_boost,
        }
        return state, metrics

    def _update_trisequence_components(
        self,
        vector: torch.Tensor,
        fast_component: torch.Tensor,
    ) -> dict[str, Any]:
        if self._past_coupling is None:
            self._past_coupling = vector.clone()
        else:
            alpha = _clamp(float(self.config.trisequence_memory_alpha), 0.0, 1.0)
            self._past_coupling = (1.0 - alpha) * self._past_coupling + alpha * vector

        surface = self.model.state_vector(self.state).detach()
        if self._self_trace is None:
            self._self_trace = surface.clone()
        else:
            alpha = _clamp(float(self.config.trisequence_self_alpha), 0.0, 1.0)
            self._self_trace = (1.0 - alpha) * self._self_trace + alpha * surface

        previous_forecast = torch.zeros_like(vector) if self._future_coupling is None else self._future_coupling
        phi_inverse = 0.6180339887498948
        phi_inverse_squared = 1.0 - phi_inverse
        trend = phi_inverse * fast_component + phi_inverse_squared * (vector - self._past_coupling)
        future = vector + float(self.config.trisequence_future_scale) * trend
        future_error = vector - previous_forecast
        self._future_coupling = future.clone()

        self_signal = torch.tanh(self._self_trace)
        other = vector - self_signal
        if self._other_trace is None:
            self._other_trace = other.clone()
        else:
            alpha = _clamp(float(self.config.trisequence_memory_alpha), 0.0, 1.0)
            self._other_trace = (1.0 - alpha) * self._other_trace + alpha * other

        self_norm = torch.norm(self_signal)
        other_norm = torch.norm(self._other_trace)
        cosine_den = float((self_norm * other_norm).item())
        cosine = (
            float(torch.sum(self_signal * self._other_trace).item()) / cosine_den
            if abs(cosine_den) > 1.0e-12
            else 0.0
        )
        metrics = {
            "variant_trisequence_past_norm": float(torch.norm(self._past_coupling).item()),
            "variant_trisequence_current_norm": float(torch.norm(vector).item()),
            "variant_trisequence_future_norm": float(torch.norm(future).item()),
            "variant_trisequence_future_error_norm": float(torch.norm(future_error).item()),
            "variant_trisequence_self_norm": float(self_norm.item()),
            "variant_trisequence_other_norm": float(other_norm.item()),
            "variant_trisequence_self_other_l2": float(torch.norm(self_signal - self._other_trace).item()),
            "variant_trisequence_self_other_cosine": cosine,
        }
        return {
            "past_coupling": self._past_coupling,
            "future": future,
            "future_error": future_error,
            "self_signal": self_signal,
            "other_trace": self._other_trace,
            "self_other_cosine": cosine,
            "metrics": metrics,
        }


DEMIAN_V1_VARIANTS = (
    "demian_v1_baseline",
    "demian_v1_multitimescale",
    "demian_v1_multitimescale_v2",
    "demian_v1_predictive_gate",
    "demian_v1_topographic_state",
    "demian_v1_trisequence_fibonacci",
    "demian_v1_trisequence_tension",
    "demian_v1_trisequence_tension_v2",
    "demian_v1_trisequence_tension_v3",
    "demian_v1_trisequence_tension_v3_no_trisequence",
    "demian_v1_trisequence_tension_v3_no_self_other",
    "demian_v1_trisequence_tension_v3_no_tension",
    "demian_v1_trisequence_tension_v3_no_global_compass",
    "demian_v1_trisequence_tension_v3_no_slow_boost",
    "demian_v1_trisequence_tension_v3_no_fast_suppression",
)


def _optional_tensor(payload: Any) -> torch.Tensor | None:
    if payload is None:
        return None
    return torch.tensor(payload, dtype=torch.float32, device=torch.device("cpu"))


def _ratio(numerator: torch.Tensor, denominator: torch.Tensor) -> float:
    den = float(denominator.item())
    return float(numerator.item()) / den if abs(den) > 1.0e-12 else 0.0


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, float(value)))


def _region_means(vector: torch.Tensor, region_count: int) -> torch.Tensor:
    chunks = torch.tensor_split(vector, max(1, min(int(region_count), vector.shape[-1])), dim=-1)
    return torch.cat([chunk.mean(dim=-1, keepdim=True) for chunk in chunks], dim=-1)


def _expand_regions(regions: torch.Tensor, hidden_size: int) -> torch.Tensor:
    repeats = (hidden_size + regions.shape[-1] - 1) // regions.shape[-1]
    return regions.repeat_interleave(repeats, dim=-1)[..., :hidden_size]


def _normalized_fibonacci_triplet() -> tuple[float, float, float]:
    total = 1.0 + 2.0 + 3.0
    return 1.0 / total, 2.0 / total, 3.0 / total


def _match_width(vector: torch.Tensor, width: int) -> torch.Tensor:
    if vector.shape[-1] == width:
        return vector
    if vector.shape[-1] > width:
        chunks = torch.tensor_split(vector, width, dim=-1)
        return torch.cat([chunk.mean(dim=-1, keepdim=True) for chunk in chunks], dim=-1)
    repeats = (width + vector.shape[-1] - 1) // vector.shape[-1]
    return vector.repeat_interleave(repeats, dim=-1)[..., :width]
