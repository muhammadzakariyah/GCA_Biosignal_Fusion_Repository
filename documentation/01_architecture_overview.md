# Gated Cross-Attention Architecture Overview

## 1. Purpose

The Gated Cross-Attention (GCA) architecture integrates information from two representation sources: clinical data and heart rate variability (HRV) features. HRV signal-quality information is also considered during attention processing and gate generation.

This document summarizes the architecture components identified in the research code package. It is not a substitute for the source code and does not imply that all GCA variants use an identical computational pathway.

## 2. Modality encoders

### 2.1 Clinical encoder

`ClinicalEncoder` projects clinical features into a latent space and forms sequential representations. These representations are used in cross-attention and clinical aggregation.

### 2.2 HRV encoder

`HRVEncoder` combines HRV features with quality-related information and indicators, then forms HRV token representations. Invalid tokens are handled using a validity mask.

## 3. Quality-biased cross-attention

The `QualityBiasedCrossAttention` module uses query, key, and value projections, splits representations into multiple heads, and computes scaled attention scores.

In the attention direction that receives HRV key information, a log-quality bias is added to the attention scores and multiplied by a learned positive scale. A validity mask is applied to prevent invalid HRV tokens from contributing to the context.

In GCA V1, attention is computed in two directions:

- **Clinical-from-HRV:** clinical representations act as queries, while HRV representations act as keys and values. HRV signal quality is used as an attention-score bias.
- **HRV-from-clinical:** HRV representations act as queries, while clinical representations act as keys and values. A clinical validity mask is used; this direction does not receive the HRV quality-bias argument.

## 4. Context-dependent element-wise sigmoid gate

In `GatedCrossAttentionModelV1`, the clinical representation after attention processing, the HRV representation after attention processing, and a summary of the quality context are concatenated. The concatenated representation is projected by a linear layer and a sigmoid function to produce a latent-dimension gate.

The gate regulates the contribution of HRV and clinical representations during admission-level fusion. If no HRV token is valid, the HRV contribution controlled by the gate is disabled through an HRV-availability mask.

## 5. Residual variant

The research package also contains `ResidualGCA`. This variant retains the initial clinical representation as the base and adds a residual correction controlled by a gate and a learnable residual scale.

In configurations where reliability options are enabled, the gate context is extended with HRV coverage and availability information. A reliability factor is also used to adjust the contribution of the residual correction.

## 6. Interpretation boundaries

The term *context-dependent element-wise sigmoid gating* describes the form of the gate: its coefficients are generated from a combined context and have a feature dimension. It does not imply that sigmoid gating itself is a novel mechanism.

The methodological elements that should be described specifically are the integration of bidirectional cross-attention, the quality bias in the HRV-to-clinical attention pathway, contextual gating, and the residual and reliability options in the corresponding model variants.
