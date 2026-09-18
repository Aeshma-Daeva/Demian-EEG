"""Public EEG observer adapter for the Demian recurrent substrate."""

from demian_eeg.features import (
    bandpower_features,
    fold_features,
    robust_normalize,
    topographic_bandpower_features,
    window_eeg,
)
from demian_eeg.observer import (
    EEGObserverConfig,
    EEGObserverResult,
    run_observer_from_array,
    write_observer_outputs,
)

__all__ = [
    "EEGObserverConfig",
    "EEGObserverResult",
    "bandpower_features",
    "fold_features",
    "robust_normalize",
    "run_observer_from_array",
    "topographic_bandpower_features",
    "window_eeg",
    "write_observer_outputs",
]
