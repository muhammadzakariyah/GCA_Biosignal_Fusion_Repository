# Reproducibility and Usage Limitations

## 1. Materials provided

This repository is a selected public code release. It contains `code/encoders.py` and `code/model_core.py`, together with architectural documentation, mathematical descriptions, model-variant notes, references, and dependency specifications.

## 2. Materials intentionally not provided

Some scripts and files from the original internal `code/` directory have intentionally been withheld because they are part of the study's experimental workflow. The public repository therefore does not contain the complete experiment orchestration or every script used to generate, aggregate, replay, and analyze the study results.

The repository also does not provide MIMIC-IV data, derived patient-level HRV data, patient-level predictions, model weights/checkpoints, internal split manifests, archived training-protocol/configuration records, or complete study output directories. The original notebook with embedded execution outputs is not included.

## 3. Reproduction implications

The released source files can be inspected and used as selected implementation components, but the repository alone is insufficient to reproduce the exact numerical results, all experiments, or every publication figure and table. Full reproduction requires authorized local clinical data, an approved hourly HRV export and its audit manifest, as well as the study-specific scripts, provenance records, predictions, and configurations that are not included in this public release.

Differences in library versions, hardware, preprocessing, data access, and training settings may also affect numerical results. The presence of model source files should not be interpreted as a guarantee of end-to-end reproducibility.

## 4. Data governance

Do not upload patient-level or otherwise restricted data to this repository. Researchers seeking access to MIMIC-IV must follow the official PhysioNet credentialing procedures and applicable Data Use Agreement.

## 5. Attribution and licensing

GMU literature is cited as conceptual background for gated multimodal fusion. This repository does not claim that a specific GMU source-code implementation is the direct source of the study code. No open-source license is currently assigned; reuse rights should be clarified by the authors before public release.
