"""Run Demian v1 as a zero-authority EEG observer."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from demian_eeg.features import bandpower_features, fold_features, robust_normalize, topographic_bandpower_features, window_eeg
from demian_v1 import DemianV1Config, DemianV1Runtime


@dataclass(frozen=True)
class EEGObserverConfig:
    hidden_size: int = 32
    seed: int = 0
    runtime_variant: str = "demian_v1_baseline"
    strength: float = 0.25
    window_seconds: float = 2.0
    hop_seconds: float = 1.0
    snapshot_interval: int = 25
    shuffle_seed: int = 17
    feature_mode: str = "global"


@dataclass(frozen=True)
class ObservationRecord:
    window_index: int
    start_time: float
    stop_time: float
    surface: list[float]
    metrics: dict[str, float]


@dataclass(frozen=True)
class EEGObserverResult:
    config: dict[str, Any]
    feature_names: list[str]
    records: list[ObservationRecord]
    summary: dict[str, Any]
    snapshots: dict[str, dict[str, Any]]
    couplings: np.ndarray


def run_observer_from_array(
    data: np.ndarray,
    sfreq: float,
    *,
    config: EEGObserverConfig | None = None,
    label: str | None = None,
    channel_names: list[str] | None = None,
) -> EEGObserverResult:
    """Window EEG, feed Demian, and compute observer diagnostics."""

    cfg = config or EEGObserverConfig()
    windows = window_eeg(data, sfreq, window_seconds=cfg.window_seconds, hop_seconds=cfg.hop_seconds)
    if not windows:
        raise ValueError("eeg_recording_too_short_for_windowing")

    raw_features = []
    masks = []
    feature_names: list[str] | None = None
    for eeg_window in windows:
        if cfg.feature_mode == "global":
            features, mask, names = bandpower_features(eeg_window.data, sfreq)
        elif cfg.feature_mode == "topographic":
            features, mask, names = topographic_bandpower_features(
                eeg_window.data,
                sfreq,
                channel_names=channel_names,
            )
        else:
            raise ValueError(f"eeg_unknown_feature_mode:{cfg.feature_mode}")
        raw_features.append(features)
        masks.append(mask)
        feature_names = names

    normalized = robust_normalize(np.vstack(raw_features))
    couplings = np.vstack(
        [
            fold_features(features, mask, hidden_size=cfg.hidden_size, seed=cfg.seed)
            for features, mask in zip(normalized, masks, strict=True)
        ]
    )

    records, snapshots = _run_sequence(couplings, windows, cfg)
    ordered_surface = np.asarray(records[-1].surface, dtype=np.float32)
    shuffled_surface = _final_surface(couplings[_shuffle_indices(len(couplings), cfg.shuffle_seed)], cfg)
    frozen_surface = _final_surface(couplings, cfg, gate_frozen=True)
    resume_summary = _resume_summary(couplings, cfg)

    summary: dict[str, Any] = {
        "label": label,
        "window_count": len(windows),
        "sampling_rate": float(sfreq),
        "channel_count": int(np.asarray(data).shape[0]),
        "ordered_vs_shuffled_final_l2": _l2(ordered_surface, shuffled_surface),
        "live_vs_gate_frozen_final_l2": _l2(ordered_surface, frozen_surface),
        **resume_summary,
    }
    return EEGObserverResult(
        config=asdict(cfg),
        feature_names=feature_names or [],
        records=records,
        summary=summary,
        snapshots=snapshots,
        couplings=couplings,
    )


def write_observer_outputs(result: EEGObserverResult, out_dir: str | Path) -> None:
    """Persist observer metrics, couplings, snapshots, and summary."""

    target = Path(out_dir)
    target.mkdir(parents=True, exist_ok=True)
    (target / "snapshots").mkdir(exist_ok=True)

    with (target / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(result.summary, handle, indent=2, sort_keys=True)
    with (target / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(result.config, handle, indent=2, sort_keys=True)
    with (target / "feature_names.json").open("w", encoding="utf-8") as handle:
        json.dump(result.feature_names, handle, indent=2)
    np.save(target / "couplings.npy", result.couplings)

    metric_keys = sorted({key for record in result.records for key in record.metrics})
    with (target / "metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["window_index", "start_time", "stop_time", *metric_keys],
        )
        writer.writeheader()
        for record in result.records:
            writer.writerow(
                {
                    "window_index": record.window_index,
                    "start_time": record.start_time,
                    "stop_time": record.stop_time,
                    **record.metrics,
                }
            )

    for name, snapshot in result.snapshots.items():
        with (target / "snapshots" / f"{name}.json").open("w", encoding="utf-8") as handle:
            json.dump(snapshot, handle, indent=2, sort_keys=True)


def _run_sequence(
    couplings: np.ndarray,
    windows: list[Any],
    cfg: EEGObserverConfig,
) -> tuple[list[ObservationRecord], dict[str, dict[str, Any]]]:
    runtime = _runtime_from_config(cfg)
    records: list[ObservationRecord] = []
    snapshots: dict[str, dict[str, Any]] = {}
    for index, coupling in enumerate(couplings):
        result = runtime.step(torch.from_numpy(coupling), strength=cfg.strength)
        window = windows[index]
        records.append(
            ObservationRecord(
                window_index=window.index,
                start_time=float(window.start_time),
                stop_time=float(window.stop_time),
                surface=list(result["surface"]),
                metrics={key: float(value) for key, value in result["metrics"].items()},
            )
        )
        if cfg.snapshot_interval > 0 and (index + 1) % cfg.snapshot_interval == 0:
            snapshots[f"step_{index + 1:06d}"] = runtime.snapshot().to_dict()
    snapshots["final"] = runtime.snapshot().to_dict()
    return records, snapshots


def _final_surface(couplings: np.ndarray, cfg: EEGObserverConfig, *, gate_frozen: bool = False) -> np.ndarray:
    runtime = _runtime_from_config(cfg, gate_frozen=gate_frozen)
    result: dict[str, Any] | None = None
    for coupling in couplings:
        result = runtime.step(torch.from_numpy(coupling), strength=cfg.strength)
    if result is None:
        raise ValueError("eeg_no_couplings_to_observe")
    return np.asarray(result["surface"], dtype=np.float32)


def _resume_summary(couplings: np.ndarray, cfg: EEGObserverConfig) -> dict[str, float | int]:
    if len(couplings) < 3:
        return {
            "resume_split_index": 0,
            "full_vs_source_resume_final_l2": 0.0,
            "surface_only_vs_source_resume_final_l2": 0.0,
        }

    split = len(couplings) // 2
    source = _runtime_from_config(cfg)
    for coupling in couplings[:split]:
        source.step(torch.from_numpy(coupling), strength=cfg.strength)
    snapshot = source.snapshot()

    full = _runtime_from_config(cfg)
    full.restore(snapshot)
    surface_only = _runtime_from_config(cfg)
    surface_only.restore(snapshot, surface_only=True)

    source_surface = full_surface = surface_surface = None
    for coupling in couplings[split:]:
        tensor = torch.from_numpy(coupling)
        source_surface = source.step(tensor, strength=cfg.strength)["surface"]
        full_surface = full.step(tensor, strength=cfg.strength)["surface"]
        surface_surface = surface_only.step(tensor, strength=cfg.strength)["surface"]

    return {
        "resume_split_index": split,
        "full_vs_source_resume_final_l2": _l2(source_surface, full_surface),
        "surface_only_vs_source_resume_final_l2": _l2(source_surface, surface_surface),
    }


def _shuffle_indices(size: int, seed: int) -> np.ndarray:
    indices = np.arange(size)
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)
    return indices


def _l2(left: Any, right: Any) -> float:
    return float(np.linalg.norm(np.asarray(left, dtype=np.float32) - np.asarray(right, dtype=np.float32)))


def _runtime_from_config(cfg: EEGObserverConfig, *, gate_frozen: bool = False) -> DemianV1Runtime:
    return DemianV1Runtime(
        DemianV1Config(
            hidden_size=cfg.hidden_size,
            seed=cfg.seed,
            variant=cfg.runtime_variant,
            gate_frozen=gate_frozen,
        )
    )
