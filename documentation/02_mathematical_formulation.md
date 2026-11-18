# Mathematical Formulation

This document presents a high-level formulation consistent with the components in the research package. The notation explains the architectural flow and is not intended to replace the complete tensor-level definitions in the implementation.

## 1. Modality representations

Let:

- `H_c` denote the sequence of clinical representations after the clinical encoder.
- `H_h` denote the sequence of HRV representations after the HRV encoder.
- `M_h` denote the validity mask for HRV tokens.
- `Q_h` denote HRV signal-quality information.

## 2. Quality-biased attention

For query `Q`, key `K`, and value `V`, the base attention score is:

\[
S = \frac{QK^T}{\sqrt{d_k}}
\]

In the clinical-from-HRV direction, HRV key quality is added as a bias:

\[
S_q = S + \operatorname{softplus}(\rho)\,\log(\operatorname{clip}(q,\epsilon,1))
\]

The validity mask is applied to the scores before softmax normalization. The resulting attention weights are used to compute a weighted context from the values.

## 3. Gate context

After cross-attention processing and token aggregation, admission-level representations are obtained:

- `z_c`: clinical representation.
- `z_h`: HRV representation.
- `q`: quality and availability context representation.

The feature-wise gate is:

\[
g = \sigma\left(W_g [z_c;z_h;q] + b_g\right)
\]

where `σ` is the sigmoid function and `[;]` denotes concatenation. In configurations with `scalar_gate=False`, the output dimension of the gate equals the latent dimension.

## 4. Fusion in GCA V1

Admission-level fusion in GCA V1 is expressed as:

\[
z_f = g \odot z_h + (1-g) \odot z_c
\]

where `⊙` denotes element-wise multiplication. The gate is multiplied by an HRV-availability mask so that the HRV contribution is zero when no valid HRV token is present.

## 5. Residual fusion

In the `ResidualGCA` variant, the fusion form is different. The initial clinical representation is retained as the base:

\[
z_f = z_c + \alpha\, g_{eff} \odot r
\]

where:

- `r` is a learned correction derived from the initial clinical representation, the attended clinical representation, and HRV information.
- `g_eff` is the gate after availability adjustment and, when enabled, reliability adjustment.
- `α = sigmoid(α_logit)` is a learnable residual scale.

In the implementation examined, the initial value of `α` is set through a logit corresponding to 0.1.

## 6. Distinction between the two fusion forms

The GCA V1 and ResidualGCA equations are not interchangeable. GCA V1 uses element-wise interpolation between clinical and HRV representations, whereas ResidualGCA retains the clinical representation as the base and adds a residual correction.
