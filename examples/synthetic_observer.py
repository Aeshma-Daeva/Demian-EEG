"""Run the public EEG observer on a deterministic synthetic signal."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from demian_eeg import EEGObserverConfig, run_observer_from_array, write_observer_outputs


def synthetic_eeg(*, channels: int = 4, sfreq: float = 100.0, seconds: float = 8.0) -> np.ndarray:
    t = np.arange(int(sfreq * seconds)) / sfreq
    return np.asarray(
        [
            (1.0 + 0.15 * channel + 0.4 * (t > seconds / 2.0))
            * np.sin(2 * np.pi * (8 + channel) * t)
            + (0.2 + 0.1 * (t > seconds / 3.0)) * np.sin(2 * np.pi * (16 + channel) * t)
            for channel in range(channels)
        ],
        dtype=np.float64,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("artifacts/example"))
    args = parser.parse_args()
    result = run_observer_from_array(
        synthetic_eeg(),
        100.0,
        config=EEGObserverConfig(hidden_size=8, seed=22, snapshot_interval=3),
        label="synthetic-eeg",
    )
    write_observer_outputs(result, args.output)
    print(args.output / "summary.json")


if __name__ == "__main__":
    main()
