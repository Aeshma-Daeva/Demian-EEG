"""EEG windowing and coupling-vector feature extraction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

DEFAULT_BANDS: tuple[tuple[str, float, float], ...] = (
    ("delta", 1.0, 4.0),
    ("theta", 4.0, 8.0),
    ("alpha", 8.0, 13.0),
    ("beta", 13.0, 30.0),
    ("gamma", 30.0, 45.0),
)

TOPOGRAPHIC_REGIONS: dict[str, tuple[str, ...]] = {
    "frontal": ("fp", "af", "f"),
    "central": ("fc", "c", "cz"),
    "temporal": ("ft", "t", "tp"),
    "parietal": ("cp", "p", "po"),
    "occipital": ("o", "oz", "iz"),
    "left": ("1", "3", "5", "7", "9"),
    "right": ("2", "4", "6", "8", "10"),
    "midline": ("z",),
}


@dataclass(frozen=True)
class EEGWindow:
    """A chronological EEG window in samples and seconds."""

    index: int
    start_sample: int
    stop_sample: int
    start_time: float
    stop_time: float
    data: np.ndarray


def window_eeg(
    data: np.ndarray,
    sfreq: float,
    *,
    window_seconds: float = 2.0,
    hop_seconds: float = 1.0,
) -> list[EEGWindow]:
    """Split channel-first EEG into fixed chronological windows."""

    array = np.asarray(data, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError("eeg_data_must_be_channel_by_sample")
    if sfreq <= 0:
        raise ValueError("eeg_sampling_rate_must_be_positive")
    window_samples = int(round(window_seconds * sfreq))
    hop_samples = int(round(hop_seconds * sfreq))
    if window_samples <= 0 or hop_samples <= 0:
        raise ValueError("eeg_window_and_hop_must_be_positive")
    if array.shape[1] < window_samples:
        return []

    windows: list[EEGWindow] = []
    for index, start in enumerate(range(0, array.shape[1] - window_samples + 1, hop_samples)):
        stop = start + window_samples
        windows.append(
            EEGWindow(
                index=index,
                start_sample=start,
                stop_sample=stop,
                start_time=start / sfreq,
                stop_time=stop / sfreq,
                data=array[:, start:stop],
            )
        )
    return windows


def bandpower_features(
    window: np.ndarray,
    sfreq: float,
    *,
    bands: Iterable[tuple[str, float, float]] = DEFAULT_BANDS,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Return compact log-relative bandpower and quality features."""

    data = np.asarray(window, dtype=np.float64)
    if data.ndim != 2:
        raise ValueError("eeg_window_must_be_channel_by_sample")
    if data.shape[1] < 2:
        raise ValueError("eeg_window_must_have_at_least_two_samples")

    finite = np.isfinite(data)
    clean = np.where(finite, data, 0.0)
    centered = clean - np.nanmean(np.where(finite, data, np.nan), axis=1, keepdims=True)
    centered = np.where(np.isfinite(centered), centered, 0.0)

    spectrum = np.abs(np.fft.rfft(centered, axis=1)) ** 2
    freqs = np.fft.rfftfreq(centered.shape[1], d=1.0 / sfreq)
    total_mask = (freqs >= 1.0) & (freqs <= min(45.0, sfreq / 2.0))
    total_power = spectrum[:, total_mask].sum(axis=1) + 1e-12

    values: list[float] = []
    names: list[str] = []
    for name, low, high in bands:
        band_mask = (freqs >= low) & (freqs < high)
        power = spectrum[:, band_mask].sum(axis=1) if np.any(band_mask) else np.zeros(data.shape[0])
        relative = np.log1p(power / total_power)
        values.extend(
            [
                float(np.mean(relative)),
                float(np.std(relative)),
                float(np.max(relative) - np.min(relative)),
            ]
        )
        names.extend([f"{name}_rel_mean", f"{name}_rel_std", f"{name}_rel_range"])

    channel_std = np.std(centered, axis=1)
    missing_fraction = 1.0 - float(np.mean(finite))
    flat_fraction = float(np.mean(channel_std < 1e-12))
    rms = float(np.sqrt(np.mean(centered**2)))
    robust_scale = np.median(np.abs(centered - np.median(centered))) + 1e-12
    artifact_fraction = float(np.mean(np.abs(centered) > 8.0 * robust_scale))
    values.extend([missing_fraction, flat_fraction, np.log1p(rms), artifact_fraction])
    names.extend(["missing_fraction", "flat_fraction", "log_rms", "artifact_fraction"])

    features = np.asarray(values, dtype=np.float32)
    missing_mask = (~np.isfinite(features)).astype(np.float32)
    return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0), missing_mask, names


def topographic_bandpower_features(
    window: np.ndarray,
    sfreq: float,
    *,
    channel_names: Iterable[str] | None = None,
    bands: Iterable[tuple[str, float, float]] = DEFAULT_BANDS,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Return global plus scalp-region bandpower features."""

    global_features, _, global_names = bandpower_features(window, sfreq, bands=bands)
    data = np.asarray(window, dtype=np.float64)
    names_in = list(channel_names or [f"ch{index}" for index in range(data.shape[0])])
    if len(names_in) != data.shape[0]:
        names_in = [f"ch{index}" for index in range(data.shape[0])]

    finite = np.isfinite(data)
    clean = np.where(finite, data, 0.0)
    centered = clean - np.nanmean(np.where(finite, data, np.nan), axis=1, keepdims=True)
    centered = np.where(np.isfinite(centered), centered, 0.0)
    spectrum = np.abs(np.fft.rfft(centered, axis=1)) ** 2
    freqs = np.fft.rfftfreq(centered.shape[1], d=1.0 / sfreq)
    total_mask = (freqs >= 1.0) & (freqs <= min(45.0, sfreq / 2.0))
    total_power = spectrum[:, total_mask].sum(axis=1) + 1e-12

    values = list(global_features)
    feature_names = [f"global_{name}" for name in global_names]
    region_masks = _region_masks(names_in)
    band_values_by_region: dict[tuple[str, str], float] = {}
    for band_name, low, high in bands:
        band_mask = (freqs >= low) & (freqs < high)
        power = spectrum[:, band_mask].sum(axis=1) if np.any(band_mask) else np.zeros(data.shape[0])
        relative = np.log1p(power / total_power)
        for region, mask in region_masks.items():
            if np.any(mask):
                value = float(np.mean(relative[mask]))
                coverage = float(np.mean(mask))
            else:
                value = 0.0
                coverage = 0.0
            band_values_by_region[(region, band_name)] = value
            values.extend([value, coverage])
            feature_names.extend([f"{region}_{band_name}_rel_mean", f"{region}_coverage"])

        left = band_values_by_region.get(("left", band_name), 0.0)
        right = band_values_by_region.get(("right", band_name), 0.0)
        frontal = band_values_by_region.get(("frontal", band_name), 0.0)
        occipital = band_values_by_region.get(("occipital", band_name), 0.0)
        values.extend([left - right, frontal - occipital])
        feature_names.extend([f"{band_name}_left_right_delta", f"{band_name}_frontal_occipital_delta"])

    features = np.asarray(values, dtype=np.float32)
    missing_mask = (~np.isfinite(features)).astype(np.float32)
    return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0), missing_mask, feature_names


def _region_masks(channel_names: list[str]) -> dict[str, np.ndarray]:
    normalized = [_normalize_channel_name(name) for name in channel_names]
    masks: dict[str, np.ndarray] = {}
    for region, prefixes_or_suffixes in TOPOGRAPHIC_REGIONS.items():
        values = []
        for name in normalized:
            if region in {"left", "right"}:
                values.append(any(name.endswith(suffix) for suffix in prefixes_or_suffixes))
            elif region == "midline":
                values.append(any(name.endswith(suffix) for suffix in prefixes_or_suffixes))
            else:
                values.append(any(name.startswith(prefix) for prefix in prefixes_or_suffixes))
        masks[region] = np.asarray(values, dtype=bool)
    return masks


def _normalize_channel_name(name: str) -> str:
    return "".join(character for character in name.lower() if character.isalnum())


def robust_normalize(features: np.ndarray) -> np.ndarray:
    """Normalize feature rows with median/IQR statistics."""

    matrix = np.asarray(features, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError("eeg_features_must_be_two_dimensional")
    if matrix.shape[0] == 0:
        return matrix.copy()
    median = np.median(matrix, axis=0)
    q75 = np.percentile(matrix, 75, axis=0)
    q25 = np.percentile(matrix, 25, axis=0)
    scale = np.where((q75 - q25) > 1e-6, q75 - q25, 1.0)
    normalized = (matrix - median) / scale
    return np.clip(normalized, -8.0, 8.0).astype(np.float32)


def fold_features(
    features: np.ndarray,
    missing_mask: np.ndarray | None,
    *,
    hidden_size: int,
    seed: int = 0,
) -> np.ndarray:
    """Fold arbitrary EEG features into Demian's fixed coupling size."""

    vector = np.asarray(features, dtype=np.float32).reshape(-1)
    mask = np.zeros_like(vector) if missing_mask is None else np.asarray(missing_mask, dtype=np.float32).reshape(-1)
    if mask.shape != vector.shape:
        raise ValueError("eeg_missing_mask_shape_mismatch")
    if hidden_size < 2:
        raise ValueError("demian_hidden_size_must_be_at_least_two")

    source = np.concatenate([np.nan_to_num(vector), np.nan_to_num(mask)]).astype(np.float32)
    rng = np.random.default_rng(seed)
    projection = rng.normal(0.0, 1.0 / np.sqrt(max(1, source.size)), size=(hidden_size, source.size))
    return np.tanh(projection @ source).astype(np.float32)
