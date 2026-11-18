# Gated Cross-Attention for Clinical–HRV Biosignal Fusion

This repository provides a **selected public release** of the Python implementation and technical documentation for the Gated Cross-Attention (GCA) biosignal-fusion study. It is not the complete internal research codebase, experiment workspace, or a turnkey reproduction package.

Some scripts and files from the original `code/` directory have intentionally been withheld because they are part of the internal experimental workflow. The files present in this repository are the components selected for public inspection. Their presence should not be interpreted as a statement that every experiment, result-generation step, or analysis script is publicly available.

## Repository structure

```text
GCA_Biosignal_Fusion_Repository/
|-- README.md
|-- CODE_AVAILABILITY.md
|-- requirements.txt
|-- .gitignore
|-- code/
|   |-- encoders.py
|   |-- model_core.py
|-- documentation/
|   |-- 01_architecture_overview.md
|   |-- 02_mathematical_formulation.md
|   |-- 03_model_variants_and_selection.md
|   |-- 04_reproducibility_limitations.md
|-- references.bib
```

## Publicly included source files

- `code/encoders.py`: shared experiment configuration and data structures, input validation, fold-local preprocessing, clinical and HRV encoders, quality-biased cross-attention, baseline models, and Gated Cross-Attention V1.
- `code/model_core.py`: candidate/model configuration definitions, the `ResidualGCA` implementation, audited-input validation, patient-grouped training/refit utilities, and related model orchestration.

The public release is intentionally limited to these selected source modules. Other internal scripts used for experiment management, result aggregation, archived replay, or publication-specific analysis are not included. The exact scope of the public release is the file list shown above; no unlisted script should be assumed to be part of this repository.

The implementation contains two distinct GCA fusion forms: GCA V1 uses gate-weighted interpolation between clinical and HRV representations; `ResidualGCA` retains the clinical representation as the base and adds a gate-controlled residual correction. These formulations should not be treated as interchangeable. See `documentation/` for the architecture and equations.

## Installation

Use Python 3.10 or later. Install the dependencies in an environment appropriate for your hardware:

```bash
pip install -r requirements.txt
```

For GPU execution, install the PyTorch build that matches the local CUDA setup by following the official PyTorch installation selector.

## Data and execution requirements

The included modeling components expect authorized local input files and a completed ECG/HRV audit export. In the current implementation, `model_core.load_inputs()` reads:

- `clinical_features_per_admission.csv`
- `readmission_labels.csv`
- `audit_run.json`
- `audited_hrv_hourly.csv`

The clinical table must include the feature columns defined in `model_core.py` (`CLINICAL`); the labels table must include `subject_id`, `hadm_id`, and `y_30day`; and the audited HRV table must include the identifiers, hourly index, HRV features, quality indicators, coverage, and audit flags checked by the loader. The audit JSON must identify a completed audit and the checksum of the approved HRV export. Do not bypass these checks or substitute unaudited waveform-derived features.

The public repository does not include the complete internal experiment orchestration, archived protocol/configuration records, source prediction exports, or publication-analysis workflow. Consequently, the included files support inspection of selected model components and related implementation logic, but the repository alone is insufficient to reproduce the paper's exact numerical results or every reported experiment.

## Data governance

No MIMIC-IV data, derived patient-level HRV data, patient-level identifiers, prediction exports, model checkpoints, or internal experiment outputs are included. Users must obtain MIMIC-IV access through PhysioNet and comply with the applicable credentialing requirements and Data Use Agreement: https://physionet.org/content/mimiciv/

Do not upload restricted or patient-level materials to this repository.

## Code provenance and attribution

The Python modules in `code/` are the selected source files released from the study implementation. The repository intentionally excludes some internal experimental scripts and artifacts. It does not claim that a specific third-party GMU repository was the direct source of this implementation. GMU is cited only as conceptual background for gated multimodal fusion; see `references.bib`.

## License

No open-source license has been assigned in this repository. Unless the authors add a license, the code remains subject to the authors' applicable rights and should not be assumed to be available for unrestricted reuse.
