# Model Variants and Selection Procedure

## 1. Identified variants

The research implementation contains two GCA implementations that should be distinguished.

### 1.1 GatedCrossAttentionModelV1

Characteristics:

- Uses two directions of cross-attention.
- Uses a log-quality bias in the clinical-from-HRV attention scores.
- Generates a context-dependent gate through a linear layer and sigmoid activation.
- Uses a feature-dimensional gate in configurations with `scalar_gate=False`.
- Combines clinical and HRV representations through gate-weighted interpolation.
- Disables the HRV gate contribution when all HRV tokens are invalid.

### 1.2 ResidualGCA

Characteristics:

- Includes cross-attention and contextual gating.
- Retains the initial clinical representation as the base.
- Generates a residual correction through a small network with a tanh activation at the correction output.
- Uses a learnable residual scale.
- Can enable reliability components to adjust the effective gate and residual correction.

## 2. Scalar gate versus feature-wise gate

The candidate configurations documented for the study use `scalar_gate=False`. Therefore, the gate in those configurations is feature-wise (element-wise), rather than a single scalar value for the entire representation.

## 3. Configuration selection

The study's GCA configurations were selected through an internal validation procedure within cross-validation. This public repository does not include the complete archived configuration-selection records or fold-level experiment manifests.

Accordingly, this document describes the model variants and the general selection approach, but does not publish internal fold-by-fold configuration choices or the detailed experiment records used to produce the reported results. The reported performance should be interpreted as the outcome of the study's selected GCA procedure, not as evidence that one fixed configuration was imposed identically in every fold.

The primary selection criterion used mean inner-fold average precision, with Brier score as a tie-breaker.

## 4. Reporting recommendations

When describing the results, state that GCA configurations were selected through an internal validation procedure. Avoid claiming that every selected configuration used residual or reliability components. Detailed fold-specific configuration information may be provided separately if it is approved for release by the authors and requested by the journal.
