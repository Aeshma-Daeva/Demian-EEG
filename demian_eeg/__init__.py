"""Public EEG observer adapter for the Demian recurrent substrate."""

from demian_eeg.features import (
    bandpower_features,
    fold_features,
    robust_normalize,
    topographic_bandpower_features,
    window_eeg,
)
from demian_eeg.observer import (
    EEGObservationResult,
    EEGObserverConfig,
    run_observer_from_array,
    write_observer_outputs,
)

__all__ = [
    "EEGObservationResult",
    "EEGObserverConfig",
    "bandpower_features",
    "fold_features",
    "robust_normalize",
    "run_observer_from_array",
    "topographic_bandpower_features",
    "window_eeg",
    "write_observer_outputs",
]
