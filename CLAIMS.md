# Claim boundary

## Supported by this repository

- Fixed-window EEG-derived features can drive the Demian recurrent runtime without changing the source signal.
- Chronological and shuffled windows produce different final recurrent states on the deterministic synthetic fixture.
- Restoring the complete source snapshot reproduces the uninterrupted final state exactly in the tested path.
- Restoring only the exposed surface state does not reproduce the uninterrupted trajectory.
- Feature folding, runtime initialization, and the included fixture are seed-controlled and testable.

## Not established here

- Clinical validity, diagnosis, cognition decoding, consciousness measurement, or subject-level generalization.
- Superiority over standard EEG models.
- A learned representation: the public adapter uses deterministic signal features and an untrained recurrent substrate.
- Biological correspondence between Demian's internal variables and neural mechanisms.

The aggregate report from the broader study is published in [Demian-Lab](https://github.com/Aeshma-Daeva/Demian-Lab). Raw participant data is intentionally excluded.
