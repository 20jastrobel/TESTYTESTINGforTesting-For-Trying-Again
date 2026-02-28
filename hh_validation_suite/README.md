Hubbard-Holstein Validation Suite
===============================

This directory contains an isolated test workspace focused on Hubbard-Holstein
model correctness and algorithm behavior.

Run tests in increasing depth:

- Level 1 (smoke): basic mathematical and mapping checks.
- Level 2 (algorithm): ansatz + VQE consistency checks.
- Level 3 (abstract): randomized and edge-case consistency checks against
  sector-restricted exact diagonalization.
- Level 4 (energy): Hubbard-Holstein VQE vs restricted ED energy targets for
  `L=2` and `L=3` in weak-coupling regimes.

Run commands from repository root:

- `bash hh_validation_suite/run_hh_validation.sh 1`
- `bash hh_validation_suite/run_hh_validation.sh 2`
- `bash hh_validation_suite/run_hh_validation.sh 3`
- `bash hh_validation_suite/run_hh_validation.sh 4`
- `bash hh_validation_suite/run_hh_validation.sh`
