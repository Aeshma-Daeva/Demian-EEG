# Demian EEG

A small, reproducible probe of how a recurrent state substrate behaves when driven by EEG-like temporal observations.

This is not a classifier and does not make clinical claims. It turns deterministic synthetic multichannel signals into windowed spectral features, feeds them through the Demian runtime, and measures three concrete properties:

1. whether temporal order changes accumulated state;
2. whether a complete snapshot restores the same trajectory;
3. whether a surface-only snapshot is insufficient.

## Why it exists

The broader research question is how systems can maintain state and integrate sequential evidence without confusing observations with conclusions. Here the observation is the signal-derived feature stream; the recurrent state is an internal transformation; the reported distances are diagnostics, not interpretations of a person.

## Run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[test]'
pytest -q
python examples/synthetic_observer.py
```

The example writes a JSON summary, per-window metrics, coupling vectors, and snapshots under `artifacts/example/`.

The repository vendors the exact minimal Demian v1 runtime dependency needed by the adapter so that a checkout is independently runnable. The canonical runtime and its broader controls remain in [Demian-Substrate](https://github.com/Aeshma-Daeva/Demian-Substrate).

## Evidence boundary

See [CLAIMS.md](CLAIMS.md). No raw EEG, subject identifiers, or private datasets are included. The fixture is generated locally and deterministically.

## Research line

- [Demian-Substrate](https://github.com/Aeshma-Daeva/Demian-Substrate): recurrent state and deterministic restore
- [Demian-Lab](https://github.com/Aeshma-Daeva/Demian-Lab): experiments, controls, failures, and aggregate evidence
- [Demian-Geo](https://github.com/Aeshma-Daeva/Demian-Geo): the same substrate tested on a different sequential domain
- [Zenith Epistemic Runtime](https://github.com/Aeshma-Daeva/Zenith-Epistemic-Runtime): explicit belief/justification state and action authority
