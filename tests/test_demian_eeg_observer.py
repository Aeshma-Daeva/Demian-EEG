"""Tests for the Demian EEG observer bridge."""

from __future__ import annotations

import json

import numpy as np

from demian_eeg import (
    EEGObserverConfig,
    bandpower_features,
    fold_features,
    robust_normalize,
    run_observer_from_array,
    topographic_bandpower_features,
    window_eeg,
    write_observer_outputs,
)


def _synthetic_eeg(channels: int = 4, sfreq: float = 100.0, seconds: float = 8.0) -> np.ndarray:
    t = np.arange(int(sfreq * seconds)) / sfreq
    rows = []
    for channel in range(channels):
        envelope = 1.0 + 0.15 * channel + 0.4 * (t > seconds / 2.0)
        rows.append(
            envelope * np.sin(2 * np.pi * (8 + channel) * t)
            + (0.2 + 0.1 * (t > seconds / 3.0)) * np.sin(2 * np.pi * (16 + channel) * t)
            + 0.01 * channel * t
        )
    return np.asarray(rows, dtype=np.float64)


def test_window_eeg_preserves_chronological_fixed_windows() -> None:
    windows = window_eeg(_synthetic_eeg(seconds=5.0), 100.0, window_seconds=2.0, hop_seconds=1.0)

    assert len(windows) == 4
    assert [window.index for window in windows] == [0, 1, 2, 3]
    assert [window.start_sample for window in windows] == [0, 100, 200, 300]


def test_bandpower_features_are_finite_and_named() -> None:
    window = _synthetic_eeg(seconds=2.0)

    features, mask, names = bandpower_features(window, 100.0)

    assert features.shape == mask.shape
    assert features.shape[0] == len(names)
    assert np.all(np.isfinite(features))
    assert "alpha_rel_mean" in names


def test_topographic_features_preserve_regional_channel_structure() -> None:
    sfreq = 100.0
    data = _synthetic_eeg(channels=4, sfreq=sfreq, seconds=2.0)
    names = ["F3", "F4", "O1", "O2"]
    swapped = data[[2, 3, 0, 1]]

    topo, _, topo_names = topographic_bandpower_features(data, sfreq, channel_names=names)
    topo_swapped, _, _ = topographic_bandpower_features(swapped, sfreq, channel_names=names)
    global_features, _, _ = bandpower_features(data, sfreq)
    global_swapped, _, _ = bandpower_features(swapped, sfreq)

    assert "frontal_alpha_rel_mean" in topo_names
    assert not np.allclose(topo, topo_swapped)
    assert np.allclose(global_features, global_swapped)


def test_feature_folding_is_deterministic() -> None:
    matrix = np.vstack([bandpower_features(_synthetic_eeg(seconds=2.0), 100.0)[0] for _ in range(3)])
    normalized = robust_normalize(matrix)
    mask = np.zeros(normalized.shape[1], dtype=np.float32)

    left = fold_features(normalized[0], mask, hidden_size=8, seed=11)
    right = fold_features(normalized[0], mask, hidden_size=8, seed=11)

    assert left.shape == (8,)
    assert np.allclose(left, right)


def test_observer_reports_temporal_and_resume_diagnostics() -> None:
    result = run_observer_from_array(
        _synthetic_eeg(seconds=8.0),
        100.0,
        config=EEGObserverConfig(hidden_size=8, seed=22, snapshot_interval=3),
        label="synthetic",
    )

    assert result.summary["label"] == "synthetic"
    assert result.summary["window_count"] == 7
    assert result.summary["ordered_vs_shuffled_final_l2"] > 0.0
    assert result.summary["full_vs_source_resume_final_l2"] == 0.0
    assert result.summary["surface_only_vs_source_resume_final_l2"] > 0.0
    assert "final" in result.snapshots


def test_topographic_observer_accepts_channel_names() -> None:
    result = run_observer_from_array(
        _synthetic_eeg(channels=4, seconds=8.0),
        100.0,
        config=EEGObserverConfig(hidden_size=8, seed=22, snapshot_interval=3, feature_mode="topographic"),
        label="synthetic",
        channel_names=["F3", "F4", "O1", "O2"],
    )

    assert result.summary["full_vs_source_resume_final_l2"] == 0.0
    assert any(name.startswith("frontal_") for name in result.feature_names)


def test_write_observer_outputs(tmp_path) -> None:
    result = run_observer_from_array(
        _synthetic_eeg(seconds=4.0),
        100.0,
        config=EEGObserverConfig(hidden_size=8, seed=23, snapshot_interval=2),
    )

    write_observer_outputs(result, tmp_path)

    assert (tmp_path / "summary.json").exists()
    assert (tmp_path / "metrics.csv").exists()
    assert (tmp_path / "couplings.npy").exists()
    assert (tmp_path / "snapshots" / "final.json").exists()
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert summary["window_count"] == 3
