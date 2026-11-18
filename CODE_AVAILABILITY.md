# Code Availability Statement

This repository provides a selected public subset of the Python source code developed for the study-specific Gated Cross-Attention (GCA) implementation. The publicly released files are listed in the repository README and located in `code/`: `encoders.py` and `model_core.py`. They expose selected modality encoders, quality-biased cross-attention, GCA V1, the `ResidualGCA` variant, audited-input checks, and related model-training/refit components.

The repository is **not a complete release of the internal research codebase**. Some scripts and files from the original `code/` directory have intentionally been withheld because they belong to the internal experimental workflow, including experiment-specific orchestration and publication/result-generation procedures. Therefore, the public files should not be interpreted as a complete, self-contained implementation of every experiment or as sufficient to regenerate all reported tables, figures, and numerical results.

MIMIC-IV data, derived patient-level HRV data, patient-level predictions, model checkpoints, internal split manifests, archived training-protocol/configuration records, and other restricted or study-specific outputs are not redistributed. Running the released components requires independently authorized MIMIC-IV data and a compatible completed ECG/HRV audit export. Additional study-specific files are required for the internal workflows that are not part of this public release. The exact numerical results reported in the paper cannot be reproduced from this repository alone.

MIMIC-IV access is governed by PhysioNet credentialing and the applicable Data Use Agreement: https://physionet.org/content/mimiciv/

The implementation was not directly adapted from a particular GMU source-code repository. GMU is cited as conceptual background only; no third-party repository is claimed as the direct code source.

No open-source license has been assigned. Reuse should not be assumed to be unrestricted unless the authors add an explicit license.
