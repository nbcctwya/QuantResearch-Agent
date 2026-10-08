---
paper_id: PRISM_VQ_2026
title: "Vector-Quantized Discrete Latent Factors Meet Financial Priors: Dynamic Cross-Sectional Stock Ranking Prediction for Portfolio Construction"
model_name: "PRISM-VQ"
authors:
  - Namhyoung Kim
  - Jae Wook Song
affiliations:
  - RiskX
  - Hanyang University
year: 2026
publication_status: "arXiv preprint v1"
arxiv: "2605.13407"
arxiv_date: "2026-05-13"
paper_type: baseline
subtype:
  - finance_ml
  - dynamic_factor_model
  - discrete_latent_factor
  - vector_quantization
  - mixture_of_experts
  - financial_priors
task:
  - cross_sectional_stock_ranking
  - dynamic_factor_loading_generation
  - portfolio_construction
markets:
  - CSI300
  - S&P500
dataset:
  source: "Qlib"
  stock_features: "Alpha158"
  expert_prior_factors: "13 JKP Global Factor Library factor-return series"
  lookback_T: 20
  primary_target: "5-day forward return"
  auxiliary_targets: "1- to 9-day forward returns"
  train: "2009-2019"
  validation: "2020-2021"
  test: "2022-2024"
preprocessing:
  features:
    - RobustZScoreNorm
    - Fillna
    - RevIN_in_stage1
  target:
    - CSRankNorm_on_training_targets
core_modules:
  stage1:
    - GRU_stock_encoder
    - cross_asset_transformer
    - vector_quantization_codebook
    - contrastive_codebook_learning
    - FiLM_conditioned_reconstruction_decoder
    - multi_horizon_auxiliary_predictor
  stage2:
    - optional_trend_residual_decomposition
    - structure_token_temporal_transformer
    - RoPE
    - code_conditioned_sparse_MoE
    - dynamic_factor_loading_head
    - prior_factor_and_latent_factor_pricing_equation
core_mechanisms:
  - cross_sectional_discrete_information_bottleneck
  - financial_prior_anchoring
  - contrastive_semantic_codebook_learning
  - code_conditioned_expert_routing
  - structure_conditioned_temporal_modeling
  - dynamic_factor_loadings
  - load_balanced_sparse_MoE
  - two_stage_frozen_codebook_training
main_metrics:
  - RankIC
  - RankICIR
  - Annualized_Return
  - Maximum_Drawdown
  - Sharpe_Ratio
main_results:
  CSI300_RankIC: 0.0646
  CSI300_RankICIR: 0.4224
  CSI300_AR: 0.3077
  CSI300_MDD: 0.1924
  CSI300_SR: 1.5694
  SP500_RankIC: 0.0141
  SP500_RankICIR: 0.1208
  SP500_AR: 0.1442
  SP500_MDD: 0.1616
  SP500_SR: 0.6701
code: "https://github.com/finxlab/PRISM-VQ"
relevance_to_agent:
  baseline_understanding: very_high
  discrete_representation: very_high
  financial_priors: very_high
  mixture_of_experts: very_high
  structure_conditioning: very_high
  robustness_reference: very_high
  interpretability_reference: very_high
  reproducibility: very_high
priority: flagship
---

# PRISM-VQ: Vector-Quantized Discrete Latent Factors Meet Financial Priors

## 0. Document Purpose

This document is a high-density, Agent-oriented reconstruction of the 2026 PRISM-VQ paper:

**Vector-Quantized Discrete Latent Factors Meet Financial Priors: Dynamic Cross-Sectional Stock Ranking Prediction for Portfolio Construction.**

The model is called:

\[
\textbf{PRISM-VQ}
\]

for:

> **PR**ior-**I**nformed **S**tock **M**odel with **V**ector **Q**uantization.

This is not merely another VQ-VAE-style stock model.

PRISM-VQ combines four major ideas that had largely appeared separately in earlier literature:

1. **financial prior factors** from established asset-pricing research;
2. **cross-sectional discrete latent structure** learned by vector quantization;
3. **Transformer-based temporal modeling**;
4. **structure-conditioned sparse Mixture-of-Experts (MoE)** for dynamic factor loading generation.

The paper's central modeling principle is:

> Use discrete codes to represent reusable cross-sectional stock structure, use expert-designed factors as economically interpretable anchors, and let the discrete structure determine which temporal experts should generate the time-varying factor loadings.

The model therefore separates:

\[
\text{What structural type is this stock currently in?}
\]

from:

\[
\text{How should factor exposures evolve through time for this structural type?}
\]

This separation is implemented through a two-stage learning procedure:

```text
Stage 1:
historical stock features
      ↓
cross-sectional encoder
      ↓
vector quantization
      ↓
discrete structural code
      ↓
learned latent factor value

Stage 2:
historical stock sequence
+ discrete structural code
      ↓
temporal Transformer
      ↓
code-conditioned MoE routing
      ↓
dynamic factor loadings
      ↓
prior factors + learned latent factor
      ↓
return score
```

For a research Agent, PRISM-VQ is especially important because it unifies:

- **representation learning**;
- **asset-pricing structure**;
- **financial domain priors**;
- **conditional computation**;
- **interpretability**;
- **realistic cost-aware portfolio evaluation**.

The paper also contains a substantial technical appendix, so it is significantly more reproducible than many earlier financial deep-learning papers.

---

# 1. Metadata

- **Title:** Vector-Quantized Discrete Latent Factors Meet Financial Priors: Dynamic Cross-Sectional Stock Ranking Prediction for Portfolio Construction
- **Model:** PRISM-VQ
- **Authors:** Namhyoung Kim, Jae Wook Song
- **Affiliations:** RiskX; Hanyang University
- **arXiv:** 2605.13407v1
- **arXiv date:** 13 May 2026
- **Primary task:** cross-sectional stock ranking / return prediction
- **Secondary task:** portfolio construction
- **Markets:** CSI300 and S&P500
- **Code:** https://github.com/finxlab/PRISM-VQ
- **Model family:** dynamic factor model + vector quantization + Transformer + sparse MoE + expert financial priors

---

# 2. Research Problem

## 2.1 Core Financial Problem

At each decision time \(t\), the model observes a cross-section of:

\[
N_t
\]

stocks.

For stock \(i\), historical features are:

\[
x_i
\in
\mathbb R^{T\times C}.
\]

The objective is to predict a ranking score / return over the next:

\[
\Delta
\]

days.

The paper focuses on ranking stocks for portfolio construction.

---

## 2.2 Why Existing Methods Are Not Enough

The paper identifies three recurring limitations.

### Limitation 1 — Continuous Latents Are Weakly Regularized Under Low SNR

Financial data contain:

- idiosyncratic noise;
- transient fluctuations;
- regime shifts.

Continuous latent representations can fit unstable local variation too easily.

---

### Limitation 2 — Temporal Modeling Is Often Structure-Agnostic

A generic temporal model applies the same transformation to all stocks regardless of their current cross-sectional structural state.

But two stocks in different latent market structures may require different temporal dynamics.

---

### Limitation 3 — Deep Models Underuse Established Financial Knowledge

Financial research already provides interpretable signals such as:

- value;
- size;
- momentum;
- profitability;
- low risk.

Pure end-to-end neural models often ignore these priors.

The paper argues that such priors can act as stabilizing anchors under distribution shift.

---

# 3. Conceptual Research Gap

The paper positions prior work into two broad families.

## 3.1 Factor-Based Neural Models

Examples:

- CAE;
- FactorVAE;
- FactorVQVAE.

Strength:

> preserve factor-model interpretation.

Weakness:

> many remain centered on autoencoder-style representation learning and have weaker temporal conditioning.

---

## 3.2 Architecture-Centric Stock Models

Examples:

- DTML;
- MASTER;
- MATCC.

Strength:

> powerful temporal / cross-stock modeling.

Weakness:

> factor interpretation and explicit financial priors are weaker.

---

## 3.3 PRISM-VQ's Intended Bridge

PRISM-VQ aims to combine:

\[
\text{factor interpretation}
+
\text{modern representation learning}
+
\text{temporal specialization}
+
\text{financial priors}.
\]

---

# 4. Main Contributions

The paper states four principal contributions.

## Contribution 1 — Prior-Informed Discrete Factor Framework

A unified factor-based model combining:

- 13 expert prior factors;
- VQ discrete latent factors;
- temporal modeling.

---

## Contribution 2 — Vector Quantization as Financial Inductive Bias

The paper argues VQ provides:

- denoising;
- capacity control;
- reusable cross-sectional prototypes;
- more stable structure under low SNR.

---

## Contribution 3 — Discrete-Code-Gated MoE

The discrete code is not only a factor representation.

It is also used as:

> a routing signal.

Thus the learned structural class determines which temporal experts participate in dynamic factor-loading generation.

---

## Contribution 4 — Dual-Market Empirical Validation

The model is evaluated on:

- CSI300;
- S&P500;

with reported headline performance:

\[
RankIC_{CSI}=0.0646,
\]

\[
RankIC_{SP}=0.0141,
\]

and Sharpe ratios:

\[
SR_{CSI}=1.57,
\]

\[
SR_{SP}=0.67.
\]

---

# 5. Dynamic Factor Framework

PRISM-VQ retains an explicit factor pricing structure.

For stock \(i\):

\[
y_i
=
\alpha_i
+
\beta_{p,i}^{\top}f_p
+
\beta_{l,i}^{\top}f_{l,i}
+
\epsilon_i.
\tag{1}
\]

Definitions:

- \(\alpha_i\): intercept / idiosyncratic component;
- \(f_p\in\mathbb R^P\): expert prior factors;
- \(\beta_{p,i}\in\mathbb R^P\): dynamic loading on prior factors;
- \(f_{l,i}\): learned latent factor;
- \(\beta_{l,i}\): dynamic loading on learned factors;
- \(\epsilon_i\): idiosyncratic noise.

---

# 6. Dynamic Factor Generation

Unlike a classical factor model, both loadings and learned factors are nonlinear functions.

The paper writes:

\[
(\alpha_i,\beta_{p,i},\beta_{l,i})
=
\Phi_{\text{loading}}
(x_i,f_p;\theta),
\tag{2}
\]

\[
f_{l,i}
=
\Phi_{\text{factor}}
(x_i,f_p;\theta).
\tag{3}
\]

Thus:

> the factor form is interpretable, but the factor values and exposures are dynamically learned.

---

# 7. Two-Stage Learning Overview

PRISM-VQ explicitly decouples:

1. cross-sectional structure discovery;
2. temporal factor-loading generation.

---

## Stage 1 — Spatial Learning

Learn:

- stock temporal embedding;
- cross-asset structure;
- codebook;
- discrete code assignments;
- learned latent factor values;
- reconstruction auxiliary task;
- multi-horizon auxiliary return task.

---

## Stage 2 — Temporal Learning

Freeze the Stage-1 structural system / codebook.

Train:

- temporal Transformer;
- MoE routing;
- expert networks;
- dynamic loading generation;
- final return prediction.

---

# 8. Why Two Stages?

The stated design rationale is:

> avoid codebook collapse and avoid co-adaptation between representation discovery and temporal dynamics.

Stage 1 forces the model to compress cross-sectional variation into a stable structural vocabulary.

Stage 2 then learns how temporal dynamics interact with that already learned structure.

---

# 9. Stage 1 Probabilistic / Structural Formulation

At date \(t\), each stock receives a discrete code conditioned on the entire cross-section and expert priors:

\[
p
\left(
\{z_{q,i}\}_{i=1}^{N_t}
\mid
\{x_i\}_{i=1}^{N_t},
f_p
\right)
=
\prod_{i=1}^{N_t}
q
\left(
z_{q,i}
\mid
\{x_j\}_{j=1}^{N_t},
f_p
\right).
\tag{4}
\]

Each code is mapped to a learned factor value:

\[
f_{l,i}
=
\psi_{\text{factor}}
(z_{q,i})
\in
\mathbb R^{d_s}.
\tag{5}
\]

---

# 10. Stage 2 MoE Formulation

Define all loadings as:

\[
\beta_i
=
(\alpha_i,\beta_{p,i},\beta_{l,i}).
\]

The temporal stage models:

\[
p(\beta_i\mid h_{\text{temp},i},z_{q,i})
=
\sum_{j=1}^{M_e}
p(e_j\mid z_{q,i})
p_{\theta_j}
(\beta_i\mid h_{\text{temp},i}).
\tag{6}
\]

Interpretation:

- temporal features determine expert computation;
- discrete cross-sectional structure determines expert routing.

This is the essence of **structure-conditioned temporal modeling**.

---

# 11. Stage 1 — GRU Stock Encoder

For each stock:

\[
x_i
\in
\mathbb R^{T\times C}.
\]

A GRU maps the temporal sequence to:

\[
h_i
=
\phi_{\text{gru}}(x_i)
\in
\mathbb R^{d_s}.
\tag{7}
\]

Default:

\[
T=20,
\]

\[
C=158,
\]

\[
d_s=128.
\]

---

# 12. Stage 1 — Cross-Asset Transformer

Stock embeddings are stacked:

\[
H
=
[h_1;\dots;h_{N_t}]
\in
\mathbb R^{N_t\times d_s}.
\]

A cross-asset Transformer:

\[
E
\]

produces:

\[
z
=
E(H)
\in
\mathbb R^{N_t\times d_s}.
\tag{8}
\]

Each:

\[
z_i
\]

therefore contains information from the same-date cross-section.

---

# 13. Importance of Cross-Sectional Encoding Before Quantization

PRISM-VQ does not quantize a stock's isolated time-series embedding.

Instead:

\[
GRU
\rightarrow
cross\text{-}asset\ Transformer
\rightarrow
VQ.
\]

Thus the code is intended to represent:

> the stock's current role / structural state **relative to the rest of the cross-section**.

This is a major difference from generic time-series tokenization.

---

# 14. Vector Quantization

The codebook is:

\[
\mathcal Z
=
\{c_k\}_{k=1}^{K},
\]

where:

\[
c_k
\in
\mathbb R^{d_s}.
\]

For stock \(i\):

\[
k_i
=
\arg\min_{k\in\{1,\dots,K\}}
\|z_i-c_k\|_2^2,
\]

\[
z_{q,i}
=
c_{k_i}.
\tag{9}
\]

Default:

\[
K=512.
\]

---

# 15. Why Vector Quantization?

The paper gives three explicit motivations.

## 15.1 Noise Suppression

Small idiosyncratic perturbations can map to the same prototype.

Thus:

\[
z_i+\delta
\]

may still satisfy:

\[
q(z_i+\delta)=q(z_i).
\]

---

## 15.2 Capacity Control

The model is restricted to a finite prototype vocabulary.

This acts as an information bottleneck.

---

## 15.3 Sample Sharing

Stocks assigned to the same code share structural representation.

This encourages reuse and stabilizes estimation.

---

# 16. Structural Stability Across Time

The codebook itself is fixed after Stage 1.

But stock assignments can change over time.

Thus:

\[
\text{prototype vocabulary}
=
\text{stable},
\]

while:

\[
\text{stock-to-code assignment}
=
\text{dynamic}.
\]

This explicitly separates:

- persistent structural prototypes;
- time-varying stock memberships.

---

# 17. VQ Loss

The paper uses:

\[
\mathcal L_{\text{VQ}}
=
\|sg(z)-z_q\|_2^2
+
\lambda_{\text{commit}}
\|z-sg(z_q)\|_2^2.
\tag{10}
\]

where:

\[
sg(\cdot)
\]

denotes stop gradient.

Default:

\[
\lambda_{\text{commit}}=0.25.
\]

---

# 18. Straight-Through Estimator

Because nearest-neighbor assignment is non-differentiable, Stage 1 uses the standard straight-through estimator.

This allows encoder gradients to pass through the quantization operation approximately.

---

# 19. Codebook Maintenance Details

The technical appendix adds important implementation details.

PRISM-VQ maintains:

> an exponential moving average (EMA) of code assignments.

Underutilized codes are periodically re-initialized through:

> probability-based anchoring.

Purpose:

- prevent dead codes;
- improve codebook utilization;
- reduce index / code collapse.

This is a critical implementation detail absent from a bare VQ formula.

---

# 20. Contrastive Codebook Learning

Nearest-neighbor assignment alone may create arbitrary geometric partitions.

PRISM-VQ therefore adds:

\[
\mathcal L_{\text{contra}}
=
-
\log
\frac{
\exp(sim(z_i,c_{k_i})/\tau)
}{
\sum_{k=1}^{K}
\exp(sim(z_i,c_k)/\tau)
}.
\tag{11}
\]

with:

\[
sim(u,v)
=
-\|u-v\|_2.
\]

Default contrastive temperature:

\[
\tau=0.07.
\]

---

# 21. Why the Contrastive Loss Matters

The objective explicitly:

- attracts \(z_i\) toward the assigned code;
- pushes it away from competing codes.

Thus codebook learning becomes:

> semantic / discriminative clustering,

not merely nearest-neighbor geometry.

---

# 22. Contrastive VQ vs. FactorVQVAE

FactorVQVAE primarily relies on:

- VQ reconstruction;
- commitment/codebook optimization.

PRISM-VQ adds an explicit:

\[
\text{prototype contrastive objective}.
\]

This is one of the main representation-learning differences.

---

# 23. Stage 1 Reconstruction Task

A reconstruction decoder:

\[
G
\]

uses:

- quantized code;
- expert prior factors.

The paper writes:

\[
\hat x_i
=
G(z_{q,i},f_p),
\]

\[
\mathcal L_{\text{recon}}
=
\frac{1}{N_t}
\sum_{i=1}^{N_t}
\|x_i-\hat x_i\|_2^2.
\tag{12}
\]

---

# 24. Why Condition Reconstruction on Prior Factors?

The decoder is conditioned on:

\[
f_p.
\]

The stated goal is:

> encourage discrete codes to remain compatible with established financial structure.

Thus the code does not need to encode information already explained by known priors in an unconstrained manner.

---

# 25. FiLM Conditioning

The technical appendix states the decoder uses:

> Feature-wise Linear Modulation (FiLM).

For channel representation \(X\):

\[
X
\leftarrow
X\odot(1+\gamma)+\beta.
\]

The FiLM parameters are generated from:

\[
f_p.
\]

This makes reconstruction explicitly prior-conditioned.

---

# 26. Stage 1 Decoder Architecture

The technical appendix describes a multi-resolution 1D upsampling architecture.

The quantized code is projected to:

\[
X_i^{(0)}
=
reshape
(
GELU(W_{\text{in}}z_{q,i})
).
\]

Default:

\[
H=128,
\]

\[
T_0=5.
\]

With:

\[
T=20,
\]

there are:

\[
K_u=\log_2(T/T_0)=2
\]

upsampling blocks.

---

# 27. Reconstruction Upsampling Block

Each block includes:

- channel expansion;
- PixelShuffle upsampling;
- FiLM prior conditioning;
- transposed-convolution skip connection.

Finally a:

\[
1\times1
\]

convolution maps to:

\[
C=158
\]

feature channels.

---

# 28. Multi-Horizon Auxiliary Prediction

PRISM-VQ also predicts multiple future horizons from the discrete code.

The paper writes:

\[
[\hat y_{i,1},\dots,\hat y_{i,N_h}]
=
\psi_{\text{pred}}
(z_{q,i},f_p).
\]

Loss:

\[
\mathcal L_{\text{pred}}
=
\frac{1}{N_tN_h}
\sum_{i=1}^{N_t}
\sum_{h=1}^{N_h}
(y_{i,h}-\hat y_{i,h})^2.
\tag{13}
\]

---

# 29. Auxiliary Horizons

The experimental section states that auxiliary targets include:

\[
1\text{-day},2\text{-day},\dots,9\text{-day}
\]

forward returns.

Thus:

\[
N_h=9.
\]

The main evaluation target remains the 5-day forward return.

---

# 30. Why Multi-Horizon Prediction?

The paper's rationale is:

> purely reconstructive codes can be stable but weakly predictive.

The auxiliary return task forces discrete codes to retain forward-looking predictive structure.

---

# 31. Multi-Horizon Predictor Implementation

The technical appendix uses:

> a GRU-based autoregressive decoder.

Input:

\[
[z_{q,i};f_p].
\]

An MLP maps this concatenation into the initial GRU hidden state.

A learnable start token is used to autoregressively generate the \(N_h\) horizon targets.

---

# 32. Stage 1 Total Objective

The complete objective is:

\[
\mathcal L_{\text{spatial}}
=
\mathcal L_{\text{recon}}
+
\mathcal L_{\text{VQ}}
+
\lambda_{\text{contra}}
\mathcal L_{\text{contra}}
+
\lambda_{\text{pred}}
\mathcal L_{\text{pred}}.
\tag{14}
\]

Defaults:

\[
\lambda_{\text{contra}}=1,
\]

\[
\lambda_{\text{pred}}=10^{-4}.
\]

---

# 33. Stage 1 Learning Objective Decomposition

Each term solves a distinct failure mode.

## Reconstruction

Preserve information.

## VQ / Commitment

Create stable discrete prototypes.

## Contrastive

Separate prototypes semantically.

## Multi-Horizon Prediction

Preserve financially predictive information.

This is a notably richer codebook-learning objective than FactorVQVAE.

---

# 34. Stage 1 Input Normalization — RevIN

The technical appendix states Stage 1 applies:

> Reversible Instance Normalization (RevIN)

to each stock sequence before encoding.

Purpose:

- mitigate stock/time distribution shift;
- normalize per-sample statistics;
- preserve reversibility for reconstruction.

The paper explicitly states this does not use future information.

---

# 35. Stage 2 — Temporal Learning

After Stage 1:

- GRU encoder fixed;
- cross-asset encoder fixed;
- codebook fixed;
- reconstruction decoder fixed;
- auxiliary predictor fixed.

Train only Stage 2 components.

This parameter-freezing schedule is explicitly summarized in Technical Appendix Table A.1.

---

# 36. Stage 2 Input

The temporal stage consumes:

\[
x_i
\in
\mathbb R^{T\times C}
\]

and the fixed-codebook assignment:

\[
z_{q,i}
\in
\mathbb R^{d_s}.
\]

The discrete code provides cross-sectional structure.

The historical sequence provides temporal information.

---

# 37. Optional Trend / Residual Decomposition

The main method section states the temporal input may optionally be decomposed:

\[
x_{i,\text{trend}}
=
AvgPool_w(Pad(x_i)),
\]

\[
x_{i,\text{seasonal}}
=
x_i-x_{i,\text{trend}},
\]

\[
\tilde x_i
=
\phi_{\text{seasonal}}(x_{i,\text{seasonal}})
+
\phi_{\text{trend}}(x_{i,\text{trend}}).
\tag{15}
\]

---

# 38. Important Decomposition Ambiguity

The paper says the decomposition is:

> optional.

Figure 2 visually includes a decomposition layer.

However, the main experimental implementation description does not separately state whether decomposition is enabled in every reported configuration.

Therefore:

> exact reproduction should verify the released code.

Do not assume the optional decomposition is always active solely from the conceptual diagram.

---

# 39. Structure Token

The discrete code is prepended to the temporal sequence.

After projection when necessary:

\[
[z_{q,i};\tilde x_i].
\]

This is passed through a temporal Transformer:

\[
h_{\text{temp},i}
=
T([z_{q,i};\tilde x_i])
\in
\mathbb R^{d_t}.
\tag{16}
\]

---

# 40. Meaning of the Structure Token

The structure token makes temporal representation learning conditional on cross-sectional structure.

The Transformer therefore answers:

> Given this stock's current discrete structural type, how should its recent temporal sequence be summarized?

This is structurally analogous to conditioning tokens used in NLP / multimodal Transformers.

---

# 41. RoPE

The temporal Transformer uses:

> Rotary Position Embeddings (RoPE).

RoPE applies rotations to query/key channels so attention scores depend on relative time distance.

The technical appendix gives:

\[
RoPE(u,m)
=
BlockDiag(R_{m,1},\dots,R_{m,d_h/2})u.
\]

This is applied to:

\[
Q,K
\]

before attention.

---

# 42. Temporal Transformer Configuration

Default temporal dimension:

\[
d_t=64.
\]

The appendix specifies:

- one encoder layer;
- FFN dimension \(128\);
- dropout \(0.1\).

Heads:

## CSI300

\[
2.
\]

## S&P500

\[
4.
\]

---

# 43. Temporal Summary

The output corresponding to the structure token is used as:

\[
h_{\text{temp},i}.
\]

Thus the structure token functions as a learned query / summary token combining:

- temporal history;
- structural code.

---

# 44. Structure-Conditioned MoE

The MoE separates two roles.

## Expert Selection

Depends on:

\[
z_{q,i}.
\]

## Expert Computation

Depends on:

\[
h_{\text{temp},i}
\]

or the fused expert input.

This is the key architectural principle.

---

# 45. Stochastic Gate

The main text computes mean and scale:

\[
\mu_i
=
\phi_\mu(z_{q,i})
\in
\mathbb R^{M_e},
\]

\[
\sigma_i
=
Softplus(\phi_\sigma(z_{q,i}))
\in
\mathbb R^{M_e}.
\tag{17}
\]

Then:

\[
\tilde g_i
=
\mu_i
+
\epsilon_i\odot\sigma_i,
\]

\[
\epsilon_i
\sim
\mathcal N(0,I_{M_e}).
\tag{18}
\]

This injects stochasticity during training.

---

# 46. Sparse Top-k Routing

Gate logits:

\[
L_i
=
\psi_g(\tilde g_i).
\]

Top experts:

\[
K_i
=
Topk(L_i).
\]

Weights:

\[
G_{i,j}
=
\begin{cases}
\frac{\exp(L_{i,j})}
{\sum_{\ell\in K_i}\exp(L_{i,\ell})},
&
j\in K_i
\\
0,
&
\text{otherwise}.
\end{cases}
\tag{19}
\]

Constraints:

\[
\sum_jG_{i,j}=1,
\]

\[
\|G_i\|_0=k.
\]

---

# 47. Inference-Time Routing

The technical appendix clarifies:

- training uses stochastic routing;
- inference is deterministic.

At inference:

\[
\ell_i=\mu_i.
\]

Thus stochastic exploration is a training mechanism rather than deployment randomness.

---

# 48. Expert Input Fusion

The technical appendix refines the main-text description.

First normalize:

\[
\tilde h_i
=
LN(h_{\text{temp},i}),
\]

\[
\tilde z_i
=
LN(z_{q,i}).
\]

Then:

\[
u_i
=
\phi_{\text{proj}}
([\tilde h_i;\tilde z_i])
\in
\mathbb R^{d_t}.
\tag{A.10}
\]

The projection uses:

- residual FFN;
- GEGLU-style gating;
- dropout \(0.1\).

---

# 49. Important Main-Text vs. Appendix Refinement

The main text describes expert computation as:

\[
\xi_j(h_{\text{temp},i}).
\]

The appendix states actual implementation uses a fused input:

\[
u_i
=
\phi_{\text{proj}}
([h_{\text{temp},i};z_{q,i}]).
\]

This is best interpreted as:

> the main text gives the conceptual abstraction; the appendix provides the actual richer implementation.

For reproduction, follow the appendix / code.

---

# 50. Expert Networks

Each expert is a lightweight MLP.

Main text:

\[
\xi_j(h)
=
W_j^{(2)}
GELU
(
W_j^{(1)}h+b_j^{(1)}
)
+
b_j^{(2)}.
\tag{20}
\]

Appendix:

- expert hidden/output dimension:
  \[
  d_{\text{moe}}=64
  \]
- dropout:
  \[
  0.1.
  \]

---

# 51. Expert Aggregation

The weighted expert representation is:

\[
h_{\text{MoE},i}
=
\sum_{j=1}^{M_e}
G_{i,j}
\xi_j(h_{\text{temp},i})
\tag{21}
\]

conceptually.

Appendix implementation uses:

\[
m_i
=
\sum_{j=1}^{M_e}
G_{i,j}\xi_j(u_i).
\tag{A.12}
\]

---

# 52. Market-Specific MoE Capacity

## CSI300

\[
M_e=2,
\]

top-1 routing.

## S&P500

\[
M_e=8,
\]

top-4 routing.

The authors interpret this as:

> the more efficient / weaker-signal S&P500 benefits from greater expert specialization.

---

# 53. Dynamic Factor Loadings

The main text describes a projection:

\[
(\alpha_i,\beta_{p,i},\beta_{l,i})
=
\psi_{\text{load}}
(h_{\text{MoE},i}).
\]

The appendix gives a more structured implementation.

---

# 54. Base–Modulation Loading Decomposition

Base loadings from temporal information:

\[
\beta^{base}_{p,i}
=
W^{base}_p h_{\text{temp},i},
\]

\[
\beta^{base}_{l,i}
=
W^{base}_l h_{\text{temp},i}.
\tag{A.13}
\]

Then the MoE predicts modulation parameters:

\[
(\gamma_{p,i},\delta_{p,i})
=
\phi_p(m_i),
\]

\[
(\gamma_{l,i},\delta_{l,i})
=
\phi_l(m_i).
\tag{A.14}
\]

Final loadings:

\[
\beta_{p,i}
=
\gamma_{p,i}
\odot
\beta^{base}_{p,i}
+
\delta_{p,i},
\]

\[
\beta_{l,i}
=
\gamma_{l,i}
\odot
\beta^{base}_{l,i}
+
\delta_{l,i}.
\tag{A.15}
\]

---

# 55. Why Base–Modulation Decomposition?

The technical appendix states this improves:

- stability;
- interpretability.

Interpretation:

> temporal dynamics determine a base exposure pattern; structural code / MoE only modulates that base rather than generating all exposures from scratch.

This is a useful inductive bias.

---

# 56. Intercept

The intercept is predicted from the MoE output:

\[
\alpha_i
=
w_\alpha^\top m_i.
\]

---

# 57. Learned Latent Factor

The discrete code is mapped to a latent factor vector:

\[
f_{l,i}
=
\psi_{\text{factor}}(z_{q,i}).
\]

This learned factor is separate from the code itself.

The code:

> indexes structural type.

The factor mapping:

> turns that structural representation into a factor value for the return equation.

---

# 58. Final Return Prediction

The factor pricing equation is:

\[
\hat y_i
=
\alpha_i
+
\beta_{p,i}^{\top}f_p
+
\beta_{l,i}^{\top}f_{l,i}.
\tag{22}
\]

This is the final bridge between deep representation learning and interpretable factor modeling.

---

# 59. Stage 2 Load-Balancing Regularization

The paper defines expert usage frequency:

\[
f_j
=
\frac{1}{N_t}
\sum_{i=1}^{N_t}
I[j\in K_i],
\]

and average gate weight:

\[
P_j
=
\frac{1}{N_t}
\sum_{i=1}^{N_t}
G_{i,j}.
\]

The temporal loss is:

\[
\mathcal L_{\text{temporal}}
=
\frac{1}{N_t}
\sum_{i=1}^{N_t}
(\hat y_i-y_i)^2
+
\lambda_{\text{balance}}
M_e
\sum_{j=1}^{M_e}
f_jP_j.
\tag{23}
\]

---

# 60. Why Load Balancing?

Without balancing, sparse MoE can collapse onto a small number of experts.

The balancing term encourages broader expert utilization.

This is important because the paper later claims interpretable specialization.

---

# 61. Additional Loading Regularization

The technical appendix adds an \(\ell_2\) penalty:

\[
\|\beta_{p,i}\|_2
+
\|\beta_{l,i}\|_2
\]

to discourage excessively large factor exposures.

This detail is not prominent in the main-text Eq. (23).

For exact reproduction, follow the appendix / code.

---

# 62. Two-Stage Parameter Freeze Schedule

Technical Appendix Table A.1:

## Stage 1 Trained

- GRU encoder;
- cross-asset Transformer;
- codebook;
- reconstruction decoder;
- auxiliary predictor.

## Stage 2 Fixed

All above modules.

## Stage 2 Trained

- temporal Transformer;
- MoE gate;
- MoE experts;
- loading head.

This decoupling is foundational to the model.

---

# 63. Expert Prior Factors

PRISM-VQ uses:

\[
P=13
\]

expert-designed factor-return series from the:

> JKP Global Factor Library.

They are **market-wide factor returns**, not stock-specific characteristics.

This distinction matters.

---

# 64. List of 13 Expert Prior Factors

The paper includes:

1. **Accruals**
2. **Debt issuance**
3. **Investment**
4. **Low leverage**
5. **Low risk**
6. **Momentum**
7. **Profit growth**
8. **Profitability**
9. **Quality**
10. **Seasonality**
11. **Short-term reversal**
12. **Size**
13. **Value**

---

# 65. Prior Factor Interpretation

## Accruals

High accounting accruals often predict lower future returns.

## Debt Issuance

High net debt issuance tends to associate with underperformance.

## Investment

High investment / asset growth tends to associate with lower expected returns.

## Low Leverage

Captures lower distress exposure.

## Low Risk

Captures low-volatility / low-beta effects.

## Momentum

Recent winners tend to continue outperforming over intermediate horizons.

## Profit Growth

Improving operating profitability may receive positive revaluation.

## Profitability

Highly profitable firms tend to earn stronger average returns.

## Quality

Combines multiple robust firm-characteristic dimensions.

## Seasonality

Captures recurring calendar return patterns.

## Short-Term Reversal

Captures very recent mean reversion.

## Size

Captures the small-firm effect.

## Value

Captures cheap-vs-expensive valuation effects.

---

# 66. No-Look-Ahead Construction of Prior Factors

The paper is explicit:

> only information available through \(t-1\) is used at decision time \(t\).

For factor \(j\):

\[
f_{p,j}(t)
=
\exp
\left(
\sum_{\tau=t-T}^{t-1}
\log(1+r_{j,\tau})
\right)
-
1.
\tag{B.1}
\]

Equivalent to:

\[
\prod_{\tau=t-T}^{t-1}
(1+r_{j,\tau})
-
1.
\]

---

# 67. Prior-Factor Window

The cumulative factor window matches:

\[
T=20.
\]

Thus prior factors summarize the preceding 20 trading days.

The current day is excluded.

---

# 68. Prior-Factor Normalization

All factor-return series are standardized using:

> training-period statistics only.

This prevents test leakage.

---

# 69. How Prior Factors Enter the Model

They enter both stages.

## Stage 1

Condition:

- reconstruction;
- auxiliary prediction.

Purpose:

> align discrete structure with known economics.

## Stage 2

Enter directly into the pricing equation:

\[
\beta_{p,i}^{\top}f_p.
\]

Purpose:

> preserve transparent interpretable contribution.

---

# 70. Important Conceptual Point: Priors Are Anchors, Not Labels

PRISM-VQ does **not** force each learned code to correspond to one prior factor.

Instead:

- prior factors stabilize representation;
- latent codes can capture additional structure.

Later interpretability experiments show only moderate code-prior correlations.

This is intentional.

---

# 71. Dataset Protocol

Two markets:

## CSI300

China.

## S&P500

United States.

Data source:

> Qlib.

At each trading date, the paper uses:

> contemporaneous index constituents.

This explicitly avoids survivorship bias.

---

# 72. Stock Feature Set

Input stock features:

\[
Alpha158.
\]

Dimension:

\[
C=158.
\]

Lookback:

\[
T=20.
\]

---

# 73. Primary Prediction Target

The paper defines the 5-day forward return as:

\[
y_{i,t}
=
\frac{
price_{i,t+5}
-
price_{i,t+1}
}{
price_{i,t+1}
}.
\]

The text further states:

- entry reference at \(t+1\): open price;
- exit reference at \(t+5\): close price.

This exact execution/label convention should be preserved when reproducing the paper.

---

# 74. Auxiliary Targets

Stage 1 additionally uses:

\[
1\text{-}9
\]

day forward returns.

These are only auxiliary labels for codebook learning.

The main evaluation target remains 5-day.

---

# 75. Train / Validation / Test Split

Chronological split:

## Training

\[
2009
\rightarrow
2019.
\]

## Validation

\[
2020
\rightarrow
2021.
\]

## Testing

\[
2022
\rightarrow
2024.
\]

This is a strict out-of-sample test period.

---

# 76. Main Data Preprocessing

The experimental section reports:

## Feature normalization

\[
RobustZScoreNorm.
\]

## Missing values

\[
Fillna
\]

using forward filling.

## Target normalization

Training targets are cross-sectionally rank-normalized per date:

\[
CSRankNorm.
\]

---

# 77. Important Additional Stage-1 Normalization

The technical appendix also applies:

> RevIN

inside Stage 1 before the GRU/cross-asset encoder.

Thus there are two conceptually distinct normalization layers:

1. dataset-level Qlib preprocessing;
2. per-sample RevIN inside the Stage-1 architecture.

A reproduction should preserve this distinction.

---

# 78. Baseline Categories

PRISM-VQ compares against three groups.

---

## 78.1 General ML

- XGBoost
- GRU
- TCN
- Transformer

---

## 78.2 Factor-Based Models

- CAE
- VAE / FactorVAE
- VQVAE / FactorVQVAE

---

## 78.3 Architecture-Centric Deep Models

- DTML
- MASTER
- MATCC

This is a particularly strong baseline set because it spans both major stock-ML paradigms.

---

# 79. Fairness Protocol

The paper states:

> all baselines use identical data splits and preprocessing pipelines.

Hyperparameters are tuned on the validation set.

This is important because original-paper headline results from earlier baselines are **not** used directly.

They are re-evaluated under the same 2009–2024 protocol.

---

# 80. Evaluation Metrics

Two categories.

## Ranking

- RankIC
- RankICIR

## Portfolio

- Annualized Return
- Maximum Drawdown
- Sharpe Ratio

The model output is explicitly interpreted as a:

> ranking score,

not necessarily a calibrated expected-return forecast.

---

# 81. RankIC

Per date:

\[
RankIC_t
=
\frac{
(r_{\hat y_t}-\bar r_{\hat y_t})^\top
(r_{y_t}-\bar r_{y_t})
}{
N_t
\sigma_{\hat y_t}
\sigma_{y_t}
}.
\tag{C.1}
\]

Reported:

\[
RankIC
=
\frac{1}{T_{\text{test}}}
\sum_t
RankIC_t.
\tag{C.2}
\]

---

# 82. RankICIR

\[
RankICIR
=
\frac{
E[RankIC_t]
}{
std(RankIC_t)
}.
\tag{C.3}
\]

This measures temporal consistency of ranking quality.

---

# 83. Portfolio Strategy

The paper uses Qlib:

> TopkDropoutStrategy.

Default:

\[
K_{\text{port}}=30,
\]

\[
N_{\text{drop}}=5.
\]

Equal weights are used.

---

# 84. TopK-DropN Procedure

At each date:

1. rank current universe;
2. identify top \(K\) candidates;
3. remove names leaving the universe;
4. sell up to bottom \(N\) currently held stocks;
5. replace them with top-ranked candidates not already held;
6. equal-weight final holdings.

This enforces turnover control.

---

# 85. Transaction Costs

Main experiments include:

- entry/open cost:
  \[
  5\ \text{bp}
  \]
- exit/close cost:
  \[
  15\ \text{bp}.
  \]

Thus portfolio results are explicitly **net of transaction costs** under this protocol.

This is a major improvement over many earlier papers.

---

# 86. Portfolio Daily Log Return

\[
g_{p,t}
=
\log
\left(
1+
\sum_{i\in P_t}
w_{i,t}y_{i,t}
\right).
\tag{C.4}
\]

---

# 87. Annualized Return

\[
AR
=
\exp
(
252E[g_{p,t}]
)
-
1.
\tag{C.5}
\]

---

# 88. Maximum Drawdown

Let:

\[
W_t
=
\exp
\left(
\sum_{\tau=1}^{t}
g_{p,\tau}
\right).
\]

Then:

\[
MDD
=
\max_t
\left(
1-
\frac{
W_t
}{
\max_{s\le t}W_s
}
\right).
\tag{C.6}
\]

Note:

> MDD is reported as a positive loss magnitude.

---

# 89. Sharpe Ratio

\[
SR
=
\sqrt{252}
\frac{
E[g_{p,t}]
}{
std(g_{p,t})
}.
\tag{C.7}
\]

Risk-free rate:

\[
0.
\]

---

# 90. Implementation — Stage 1

Defaults:

\[
d_s=128.
\]

GRU stock encoder.

Codebook:

\[
K=512.
\]

Cross-asset Transformer:

- 1 layer;
- 2 heads.

Training:

> up to 50 epochs.

---

# 91. Implementation — Stage 2

Temporal dimension:

\[
d_t=64.
\]

Transformer:

- 1 layer;
- FFN dimension 128;
- dropout 0.1.

Heads:

- CSI300: 2
- S&P500: 4

Training:

> up to 50 epochs.

---

# 92. MoE Configuration

## CSI300

\[
M_e=2,
\]

top-1.

## S&P500

\[
M_e=8,
\]

top-4.

---

# 93. Optimization

Optimizer:

\[
AdamW.
\]

Learning rate:

\[
10^{-4}.
\]

Scheduler:

> LambdaLR.

Gradient clipping:

\[
1.0.
\]

Early stopping patience:

\[
15.
\]

---

# 94. Loss Hyperparameters

\[
\lambda_{\text{commit}}
=
0.25,
\]

\[
\lambda_{\text{contra}}
=
1,
\]

\[
\lambda_{\text{pred}}
=
10^{-4}.
\]

Load-balance coefficient:

## CSI300

\[
\lambda_{\text{balance}}
=
10^{-2}.
\]

## S&P500

\[
\lambda_{\text{balance}}
=
10^{-3}.
\]

---

# 95. Random Seeds

Five seeds:

\[
\{0,1,2,3,4\}.
\]

This is explicitly stated in the technical appendix.

---

# 96. Hardware

Training is performed on:

> one NVIDIA RTX 3090 GPU.

This is useful for practical reproducibility.

---

# 97. Main Results — CSI300

| Model | RankIC | RankICIR | AR | MDD | SR |
|---|---:|---:|---:|---:|---:|
| XGB | 0.0496 | 0.3363 | 0.2430 | 0.2353 | 1.1549 |
| TCN | 0.0518 | 0.3160 | 0.2281 | 0.2118 | 1.1879 |
| GRU | 0.0590 | 0.3756 | 0.2347 | 0.2288 | 1.2221 |
| Transformer | 0.0495 | 0.3100 | 0.2384 | 0.2478 | 1.1339 |
| CAE | 0.0444 | 0.2843 | 0.2194 | 0.2322 | 1.0393 |
| VAE | 0.0502 | 0.3360 | 0.1811 | **0.1906** | 1.0265 |
| VQVAE | 0.0552 | 0.3641 | 0.2129 | 0.2074 | 1.0645 |
| DTML | 0.0625 | 0.4076 | 0.2589 | 0.2069 | 1.1228 |
| MASTER | 0.0401 | 0.2453 | 0.2193 | 0.2481 | 0.9101 |
| MATCC | 0.0493 | 0.3295 | 0.2316 | 0.2299 | 1.0656 |
| **PRISM-VQ** | **0.0646** | **0.4224** | **0.3077** | 0.1924 | **1.5694** |

Important nuance:

> VAE has a slightly lower MDD than PRISM-VQ on CSI300.

PRISM-VQ dominates the other key metrics but does not have the absolute minimum CSI300 MDD.

---

# 98. Main Results — S&P500

| Model | RankIC | RankICIR | AR | MDD | SR |
|---|---:|---:|---:|---:|---:|
| XGB | 0.0077 | 0.0713 | 0.1088 | 0.2079 | 0.4323 |
| TCN | 0.0084 | 0.0722 | 0.0880 | 0.2528 | 0.3841 |
| GRU | 0.0057 | 0.0445 | 0.0867 | 0.3123 | 0.3425 |
| Transformer | 0.0054 | 0.0436 | 0.1004 | 0.2634 | 0.4025 |
| CAE | 0.0063 | 0.0527 | 0.0825 | 0.2104 | 0.3952 |
| VAE | 0.0043 | 0.0402 | 0.0734 | 0.2571 | 0.3702 |
| VQVAE | 0.0046 | 0.0346 | 0.1109 | 0.1868 | 0.5370 |
| DTML | 0.0089 | 0.0570 | 0.0667 | 0.3145 | 0.2726 |
| MASTER | 0.0051 | 0.0321 | 0.1084 | 0.1861 | 0.4662 |
| MATCC | 0.0061 | 0.0478 | 0.1147 | 0.2283 | 0.5164 |
| **PRISM-VQ** | **0.0141** | **0.1208** | **0.1442** | **0.1616** | **0.6701** |

On S&P500, PRISM-VQ is best in all five reported metrics.

---

# 99. Important Portfolio Reporting Detail

Table 1 notes:

- RankIC / RankICIR are averaged over 5 random seeds;
- AR / MDD / SR are computed from portfolios constructed using **ensemble predictions**.

This means:

> portfolio metrics are not simply the average of five independent seed backtests.

A local reproduction should match this ensemble protocol before claiming an exact comparison.

---

# 100. Statistical Significance

The paper tests daily RankIC with:

> block bootstrap, 10,000 resamples.

Significance notation:

- \(***\):
  \[
  p<0.001
  \]
- \(**\):
  \[
  p<0.01
  \]
- \(*\):
  \[
  p<0.05.
  \]

---

# 101. Significance Summary

The paper states PRISM-VQ significantly outperforms baselines in:

\[
17/20
\]

market-baseline comparisons at:

\[
p<0.05.
\]

The comparisons without stars in Table 1 are the cases that do not meet that threshold.

---

# 102. CSI300 Headline Result

\[
RankIC=0.0646,
\]

\[
RankICIR=0.4224,
\]

\[
AR=30.77\%,
\]

\[
MDD=19.24\%,
\]

\[
SR=1.5694.
\]

Best competing RankIC:

\[
DTML=0.0625.
\]

Thus the RankIC improvement is modest but consistent.

The portfolio improvement is larger.

---

# 103. S&P500 Headline Result

\[
RankIC=0.0141,
\]

\[
RankICIR=0.1208,
\]

\[
AR=14.42\%,
\]

\[
MDD=16.16\%,
\]

\[
SR=0.6701.
\]

The paper reports:

\[
58.4\%
\]

RankIC improvement over DTML:

\[
0.0089
\rightarrow
0.0141.
\]

---

# 104. Cross-Market Interpretation

PRISM-VQ's absolute RankIC remains much smaller on S&P500 than on CSI300.

But the relative gain from structure / priors / experts is especially large in S&P500.

The paper interprets this as:

> weaker-signal efficient markets benefit more from explicit regularization and expert specialization.

---

# 105. Core Ablation — Full Model

| Configuration | CSI300 RankIC | CSI300 RankICIR | S&P500 RankIC | S&P500 RankICIR |
|---|---:|---:|---:|---:|
| Full | 0.0646 | 0.4224 | 0.0141 | 0.1208 |
| w/o Prior | 0.0548 | 0.3756 | 0.0031 | 0.0314 |
| w/o MoE | 0.0586 | 0.3958 | 0.0081 | 0.0736 |
| w/o Codebook | 0.0472 | 0.2938 | -0.0024 | -0.0179 |

This is one of the most informative experiments in the paper.

---

# 106. Removing the Codebook

CSI300:

\[
0.0646
\rightarrow
0.0472.
\]

Relative drop:

\[
26.9\%.
\]

S&P500:

\[
0.0141
\rightarrow
-0.0024.
\]

The signal becomes negative.

This is the strongest ablation effect.

---

# 107. Interpretation of Codebook Ablation

The paper concludes:

> discrete latent structure is foundational.

A critical interpretation is:

Removing the codebook likely changes multiple things simultaneously:

- discrete bottleneck;
- structure token;
- routing signal;
- latent factor representation.

Therefore the ablation proves the importance of the **codebook-centered structural mechanism as a whole** more directly than one isolated VQ sub-property.

---

# 108. Removing Expert Prior Factors

CSI300:

\[
0.0646
\rightarrow
0.0548.
\]

Relative drop:

\[
15.2\%.
\]

S&P500:

\[
0.0141
\rightarrow
0.0031.
\]

Relative drop:

\[
78.0\%.
\]

---

# 109. Prior-Factor Interpretation

The dramatic S&P500 drop suggests:

> external financial priors are especially valuable where purely data-driven signals are weak.

This is one of the strongest empirical arguments in the paper.

---

# 110. Removing MoE

CSI300:

\[
0.0646
\rightarrow
0.0586.
\]

Relative drop:

\[
9.3\%.
\]

S&P500:

\[
0.0141
\rightarrow
0.0081.
\]

Relative drop:

\[
42.6\%.
\]

This supports structure-conditioned expert specialization.

---

# 111. Ablation Hierarchy

Approximate importance:

\[
\text{Codebook}
>
\text{Prior Factors}
>
\text{MoE}
\]

on CSI300.

On S&P500:

> all three become highly important.

The paper emphasizes synergistic behavior in the weaker-signal market.

---

# 112. Hyperparameter Sensitivity

The paper analyzes:

- codebook size \(K\);
- temporal dimension \(d_t\);
- number of experts \(M_e\).

---

# 113. Codebook Size

Performance peaks at:

\[
K=512
\]

in both CSI300 and S&P500.

This is interesting because FactorVQVAE found market-specific codebook optima.

PRISM-VQ instead reports a shared optimum at 512.

---

# 114. Temporal Dimension

Performance peaks around:

\[
d_t=64.
\]

Interpretation:

> moderate temporal capacity is preferable to either under-capacity or unnecessarily large hidden state.

---

# 115. Expert Count

## CSI300

Best:

\[
M_e=2.
\]

## S&P500

Improves up to:

\[
M_e=8.
\]

Interpretation:

> more efficient / heterogeneous signal environments benefit from more experts.

---

# 116. Important Capacity Split

PRISM-VQ shows a nuanced cross-market capacity result:

## Shared

Codebook:

\[
K=512
\]

for both.

## Market-specific

MoE capacity:

- CSI300:
  \[
  2
  \]
- S&P500:
  \[
  8.
  \]

Thus:

> structural vocabulary may be shared in required granularity, while temporal specialization needs differ by market.

This is a useful research hypothesis.

---

# 117. Portfolio Size Robustness

The paper evaluates larger portfolio sizes:

\[
K_{\text{port}}
\in
\{40,50\}.
\]

Default is:

\[
30.
\]

---

# 118. CSI300 Top-K Robustness

## \(K=40\)

PRISM-VQ:

\[
AR\approx0.28,
\]

\[
MDD\approx0.18,
\]

\[
SR\approx1.48.
\]

## \(K=50\)

\[
AR\approx0.25,
\]

\[
MDD\approx0.20,
\]

\[
SR\approx1.37.
\]

Performance degrades gradually rather than collapsing.

---

# 119. S&P500 Top-K Robustness

## \(K=40\)

\[
AR\approx0.14,
\]

\[
MDD\approx0.16,
\]

\[
SR\approx0.67.
\]

## \(K=50\)

\[
AR\approx0.12,
\]

\[
MDD\approx0.16,
\]

\[
SR\approx0.61.
\]

Signals remain useful beyond the top-30 stocks.

---

# 120. Interpretation of Top-K Sensitivity

As \(K\) grows, weaker-ranked stocks enter the portfolio.

Performance declines smoothly.

This supports:

> signal quality is strongest at the top but not concentrated in only a tiny handful of names.

---

# 121. Ndrop Sensitivity

Technical Appendix F fixes:

\[
K_{\text{port}}=30
\]

and varies:

\[
N_{\text{drop}}
\in
\{1,3,5,7,10,15\}.
\]

---

# 122. CSI300 Ndrop Results

| N | AR | SR | MDD | Cumulative Return | Turnover |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.1931 | 1.1479 | 0.1886 | 0.5549 | 0.0658 |
| 3 | 0.2636 | 1.4411 | 0.1905 | 0.7574 | 0.1973 |
| **5** | **0.3077** | **1.5694** | 0.1924 | 0.8841 | 0.3259 |
| 7 | 0.3035 | 1.5140 | 0.1879 | 0.8720 | 0.4443 |
| 10 | 0.3179 | 1.5635 | 0.1833 | 0.9133 | 0.5662 |
| 15 | 0.3006 | 1.4864 | 0.1940 | 0.8637 | 0.6183 |

The raw optimum in AR is not exactly the default \(N=5\).

The default balances performance against turnover / trading friction.

---

# 123. S&P500 Ndrop Results

| N | AR | SR | MDD | Cumulative Return | Turnover |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.0172 | 0.0798 | 0.2974 | 0.0514 | 0.0662 |
| 3 | 0.1135 | 0.5268 | 0.1605 | 0.3383 | 0.2005 |
| **5** | **0.1442** | **0.6701** | 0.1616 | 0.4299 | 0.3310 |
| 7 | 0.1772 | 0.8354 | 0.1656 | 0.5280 | 0.4592 |
| 10 | 0.1913 | 0.9149 | 0.1600 | 0.5700 | 0.6511 |
| 15 | 0.2044 | 1.0016 | 0.1487 | 0.6091 | 0.9392 |

Higher turnover materially raises raw performance.

But this comes with rapidly increasing trading intensity.

---

# 124. Why Default Ndrop=5?

The appendix notes turnover rises roughly:

\[
14\times
\]

from \(N=1\) to \(N=15\).

At round-trip cost regimes of roughly:

\[
20-50\ \text{bps},
\]

the marginal gains of very high \(N\) are largely offset.

Thus:

\[
N=5
\]

is chosen as a practical cost-performance trade-off.

---

# 125. Transaction Cost Robustness

The technical appendix evaluates six cost regimes.

1. No Cost
2. Low:
   \[
   2/3\ \text{bp}
   \]
3. Default:
   \[
   5/15\ \text{bp}
   \]
4. High:
   \[
   10/20\ \text{bp}
   \]
5. Very High:
   \[
   15/30\ \text{bp}
   \]
6. Extreme:
   \[
   20/40\ \text{bp}.
   \]

Entry / exit costs are asymmetric.

---

# 126. Transaction-Cost Result

Sharpe and cumulative return decline smoothly as cost increases.

The paper argues:

> gains are not solely driven by fragile excessive-turnover behavior.

This is stronger economic evidence than a single no-cost backtest.

---

# 127. Representation Interpretation — t-SNE

The paper visualizes S&P500 2024 discrete representations.

Four panels color observations by:

1. top 10 most frequent codes;
2. sector;
3. market capitalization;
4. forward return.

---

# 128. Representation Findings

## Code Labels

Top codes form well-separated clusters.

## Sector

Stocks from different sectors mix within clusters.

Therefore code assignments are not merely industry taxonomy.

## Market Cap

Some local concentration exists but no clean partition.

## Future Return

No clean separation.

Therefore code assignments are not simply "high-return vs. low-return" bins.

---

# 129. Interpretation of Learned Codes

The paper argues codes capture:

> richer cross-sectional structure beyond simple sector, size, or future-return partitions.

This is a stronger interpretability claim than t-SNE separation alone.

But it is still qualitative.

---

# 130. Factor Exposure Analysis

The paper computes Spearman correlations between:

- code-specific returns;
- 13 JKP prior factor returns.

Selected codes show moderate systematic associations.

---

# 131. Selected S&P500 Code Exposures

## Code 72

Top factors:

- Profitability
- Low Risk
- Momentum

Correlations:

\[
+0.118,
+0.109,
+0.096.
\]

Interpretation:

> quality / growth-like profile.

---

## Code 34

Top factors:

- Quality
- Size
- Profitability

Correlations:

\[
-0.113,
+0.097,
-0.077.
\]

---

## Code 224

Top factors:

- Profitability
- Low Leverage
- Low Risk

Correlations:

\[
-0.113,
+0.112,
-0.110.
\]

---

# 132. Selected CSI300 Code Exposures

## Code 482

- Quality:
  \[
  -0.118
  \]
- Accruals:
  \[
  +0.115
  \]
- Investment:
  \[
  +0.113
  \]

## Code 247

- Low Leverage:
  \[
  -0.117
  \]
- Value:
  \[
  +0.117
  \]
- Investment:
  \[
  +0.103
  \]

## Code 179

- Size:
  \[
  +0.117
  \]
- Profitability:
  \[
  -0.081
  \]
- Quality:
  \[
  -0.077
  \]

---

# 133. Factor-Correlation Magnitude

Maximum reported correlations are only around:

\[
0.12.
\]

This is important.

The codes are not duplicates of individual JKP factors.

They appear to represent:

> multi-dimensional factor combinations.

This supports complementarity between:

- expert priors;
- learned discrete factors.

---

# 134. Code Persistence

The paper evaluates transition persistence.

Monthly same-code persistence:

## CSI300

\[
4.9\%.
\]

## S&P500

\[
7.4\%.
\]

Uniform top-100 baseline:

\[
1\%.
\]

---

# 135. Quarterly Persistence

## CSI300

\[
11.7\%.
\]

## S&P500

\[
13.8\%.
\]

Persistence increases at longer horizon.

The authors interpret this as evidence codes capture stable structure rather than transient noise.

---

# 136. Important Persistence Interpretation

Higher quarterly than monthly persistence may initially appear counterintuitive if interpreted as ordinary Markov "same next state."

The appendix is reporting empirical transition matrices at different sampling horizons / definitions.

A reproduction should follow the code's exact transition construction before giving a stronger probabilistic interpretation.

The source itself concludes:

> quarterly persistence is higher and supports stable structural representation.

---

# 137. Code Transition Entropy

Monthly transition entropy:

## CSI300

\[
2.27.
\]

## S&P500

\[
2.99.
\]

Interpretation:

### CSI300

- lower persistence;
- more concentrated successor states.

### S&P500

- higher persistence;
- more dispersed alternatives when transition occurs.

---

# 138. Cross-Market Structural Interpretation

The authors connect transition behavior to MoE design.

## CSI300

Fewer experts:

\[
2.
\]

Interpretation:

> stronger signal / more concentrated transition structure.

## S&P500

More experts:

\[
8.
\]

Interpretation:

> weaker signal and broader conditional dynamics benefit from expert diversity.

---

# 139. Expert Specialization Analysis

The paper studies MoE routing in S&P500.

Weekly mean activation shows non-uniform expert usage.

Examples:

- Experts 6 and 7:
  \[
  \approx16\%-20\%
  \]
- Expert 2:
  \[
  \approx5\%-7\%.
  \]

---

# 140. Expert Activation Spikes

Daily mean gate weight for expert \(e\):

\[
P_{e,t}
=
\frac{1}{N_t}
\sum_{i=1}^{N_t}
G_{i,e,t}.
\]

Spike threshold:

\[
P_{e,t}
>
\mu_e+1.5\sigma_e.
\]

This identifies regime-specific expert activation episodes.

---

# 141. Expert Specialization Statistics

All:

\[
\binom{8}{2}=28
\]

expert pairs are compared with Wilcoxon signed-rank tests.

All pairwise activation distributions have:

\[
p<0.001.
\]

This supports statistically distinct expert utilization.

---

# 142. Strongest Expert Separation

Expert 2 mean activation:

\[
7.09\%.
\]

Expert 6:

\[
16.46\%.
\]

Interpretation:

- Expert 2: rare specialist;
- Expert 6: broadly active generalist-like expert.

---

# 143. Spike-Date Overlap

Jaccard overlap of activation spike dates:

\[
6.8\%.
\]

Reported baseline:

\[
8.11\%.
\]

The paper interprets low overlap as evidence experts respond to distinct market conditions.

---

# 144. Expert Specialization Caveat

Distinct activation distributions do not automatically prove each expert has a unique economically interpretable function.

Stronger analysis could examine:

- expert-specific factor-loading patterns;
- market-regime labels;
- portfolio attribution;
- expert intervention / ablation;
- expert stability across seeds.

The paper establishes routing specialization more strongly than economic semantic specialization.

---

# 145. Research Questions

The empirical section explicitly asks five questions.

## RQ1 — Overall Effectiveness

Does PRISM-VQ improve prediction and portfolios?

## RQ2 — Component Validity

How much do:

- priors;
- codebook;
- MoE;

contribute?

## RQ3 — Robustness

Sensitivity to:

- codebook size;
- expert count;
- Top-K selection.

## RQ4 — Interpretability

Do codes capture structure beyond standard factors?

## RQ5 — Expert Specialization

Do experts activate differently across market conditions?

---

# 146. Inductive Biases

PRISM-VQ encodes several strong assumptions.

## 146.1 Finite Reusable Cross-Sectional Structures Exist

A bounded codebook can represent meaningful recurring stock states.

## 146.2 Structural Type Should Be More Stable Than Temporal Exposure

Codebook is fixed after Stage 1.

Loadings remain dynamic.

## 146.3 Financial Priors Are Useful but Incomplete

Expert factors provide anchors, but latent codes must capture residual structure.

## 146.4 Temporal Dynamics Depend on Structural Type

Different codes should route to different experts.

## 146.5 Efficient Markets Require Greater Conditional Specialization

More experts are used on S&P500.

---

# 147. Why PRISM-VQ May Work

Separate evidence from interpretation.

## Evidence 1 — Codebook removal causes largest degradation

Strong support for discrete structure.

## Evidence 2 — Prior removal destroys much of S&P500 signal

Strong support for financial priors as regularizers.

## Evidence 3 — MoE removal hurts especially S&P500

Support for conditional temporal specialization.

## Evidence 4 — codes are not simple sector / size / return clusters

Support for richer latent structure.

## Evidence 5 — expert routing distributions are statistically distinct

Support for specialization.

---

# 148. Plausible Mechanism: VQ as Cross-Sectional Denoising

The cross-asset Transformer first creates context-aware embeddings.

Quantization then maps:

\[
z_i
\rightarrow
c_{k_i}.
\]

This means idiosyncratic local variation is removed unless it is large enough to change the assigned prototype.

---

# 149. Plausible Mechanism: Priors Reduce Search Space

Expert factors already explain known dimensions of return variation.

Therefore the learned factor does not need to rediscover everything from scratch.

Conceptually:

\[
\text{prediction}
=
\text{known structure}
+
\text{learned residual structure}.
\]

This can reduce overfitting under low SNR.

---

# 150. Plausible Mechanism: Code-Based Routing Aligns Structure with Computation

The discrete code is both:

- representation;
- routing key.

Thus stocks sharing structure can share the same temporal experts.

This encourages sample sharing inside expert subnetworks.

---

# 151. Plausible Mechanism: Base + Modulation Stabilizes Loadings

Instead of generating all factor exposures directly from the MoE:

\[
\text{temporal base}
+
\text{structure-conditioned modulation}
\]

separates:

- time-driven component;
- structure-driven adjustment.

This may improve stability.

---

# 152. Potential Failure Modes

These are analytical risks, not all observed failures.

## 152.1 Codebook Freeze Can Become Stale

Market structure can change after Stage 1.

A fixed vocabulary may eventually become outdated.

## 152.2 Two-Stage Error Propagation

A poor Stage-1 codebook constrains Stage 2.

## 152.3 Hard Nearest-Neighbor Assignment

Boundary observations can switch codes discontinuously.

## 152.4 Prior-Factor Availability / Quality

The model relies on external JKP factor-return inputs.

Poor factor construction or missing-market coverage can weaken portability.

## 152.5 Market-Specific Expert Count

MoE capacity must be tuned by market.

## 152.6 MoE Interpretability Can Be Overstated

Different routing statistics do not automatically imply economically distinct experts.

---

# 153. Potential Failure Mode: Structure Token Overdominance

The structure token conditions the entire temporal encoder.

If the code is wrong or unstable:

> temporal modeling can become biased toward the wrong structural prototype.

A useful future diagnostic is sensitivity to code perturbation.

---

# 154. Potential Failure Mode: Codebook + Priors Redundancy

The paper shows moderate code-factor correlations, which is desirable.

But in another market, learned codes could collapse toward prior-factor combinations.

This would reduce incremental information.

A useful diagnostic is:

\[
R^2
(
\text{code representation}
\sim
\text{prior factors}
).
\]

---

# 155. Potential Failure Mode: No Explicit Rank Loss

The model is evaluated primarily by RankIC and ranking portfolios.

But Stage 2 optimizes:

\[
MSE
+
load\ balancing.
\]

There is no explicit pairwise/listwise ranking objective in the main formulation.

This leaves an objective-alignment opportunity.

---

# 156. Evidence Strength

## Strong aspects

- two markets;
- contemporary constituents;
- survivorship-bias control;
- strict 2022–2024 out-of-sample period;
- five seeds;
- common preprocessing across baselines;
- strong baseline set;
- block-bootstrap significance;
- cost-aware portfolios;
- transaction-cost sensitivity;
- Top-K / Ndrop sensitivity;
- component ablation;
- hyperparameter sensitivity;
- representation analysis;
- factor-exposure analysis;
- code-transition analysis;
- expert-specialization statistics;
- technical appendix with architecture details.

This is unusually comprehensive for a stock-prediction paper.

---

# 157. Remaining Evidence Gaps

- no third market;
- no crisis-specific regime table in the main text;
- no explicit rolling retraining analysis in the provided version;
- no direct soft-vs-hard quantization ablation;
- no contrastive-loss ablation reported in the main component table;
- no reconstruction-loss / multi-horizon auxiliary-loss ablations reported;
- no expert count × compute fairness analysis;
- no direct ranking-loss comparison.

---

# 158. Baseline Fairness

The paper's fairness protocol is relatively strong.

Positive:

- same splits;
- same preprocessing;
- validation tuning;
- same test period;
- transaction-cost protocol shared;
- broad baseline families.

Potential concerns:

## Baseline implementation details

The paper does not fully specify whether every baseline uses original official code vs. reimplementation.

Exact implementation sources should be checked in repository.

## Priors are unique extra information

PRISM-VQ receives 13 expert prior factors.

Ablation shows these priors materially improve performance.

Thus the full-model gain is due to both:

- architecture;
- additional economically meaningful inputs.

This is legitimate, but should be acknowledged in architecture-only comparisons.

---

# 159. Reproducibility Strength

PRISM-VQ is highly reproducible from the provided paper because it reports:

- code;
- exact dates;
- exact features;
- prior factor list;
- no-look-ahead construction;
- preprocessing;
- model dimensions;
- codebook size;
- Transformer layers/heads;
- expert counts/top-k;
- optimizer;
- scheduler;
- gradient clipping;
- early stopping;
- random seeds;
- transaction costs;
- portfolio algorithm;
- cost sensitivity;
- transition analysis.

---

# 160. Important Remaining Reproduction Details to Verify in Code

Even with the strong appendix, exact code should still be checked for:

- exact batch size;
- exact `Fillna` implementation;
- exact Qlib constituent provider;
- exact `CSRankNorm` application scope;
- exact LambdaLR schedule;
- exact Stage-1 early stopping / selection metric;
- exact probability-based code anchoring implementation;
- whether optional trend decomposition is enabled by default;
- exact loading \(\ell_2\) coefficient;
- exact baseline implementations / hyperparameter grids.

---

# 161. Relationship to FactorVAE

FactorVAE:

- continuous Gaussian latent factors;
- training-time future-return posterior;
- history-only prior;
- KL matching;
- uncertainty output.

PRISM-VQ:

- historical-feature-derived discrete codes;
- no future-return posterior branch in the core representation learner;
- expert prior financial factors;
- cross-sectional Transformer before VQ;
- code-conditioned MoE;
- factor pricing equation;
- no primary predictive uncertainty output.

---

# 162. Relationship to FactorVQVAE

This comparison is especially important.

| Dimension | FactorVQVAE | PRISM-VQ |
|---|---|---|
| Code source | future-return encoder | historical stock features + cross-section |
| Codebook | VQ financial latent state | cross-sectional structural prototypes |
| Stage 1 supervision | return reconstruction | feature reconstruction + VQ + contrastive + multi-horizon prediction |
| Priors | none explicit | 13 JKP factor returns |
| Stage 2 | AR Transformer over factor tokens | temporal Transformer + code-gated MoE |
| Routing | token prediction | expert routing |
| Dynamic loadings | factor decoder | explicit prior + latent factor loadings |
| Ranking loss | yes in FactorVQVAE | no explicit rank loss |
| Interpretability | code/macros | code/factors + expert routing |
| Codebook size | market-specific in paper | 512 shared optimum |
| Uncertainty | not central | not central |

---

# 163. Crucial Difference from FactorVQVAE

FactorVQVAE learns discrete factors primarily from:

> future returns.

PRISM-VQ learns its discrete cross-sectional structure from:

> historical stock features + cross-asset context.

This means PRISM-VQ's structural code is deployable without a future-return encoder.

That is a major conceptual shift.

---

# 164. Relationship to MASTER

MASTER contributes:

- market-aware feature selection;
- cross-time / cross-stock Transformer structure.

PRISM-VQ instead uses:

- explicit financial factor priors;
- cross-sectional structural codebook;
- code-conditioned temporal specialization.

Both use Transformers, but they solve different problems.

MASTER:

> who influences whom across stock-time?

PRISM-VQ:

> what structural type is each stock in, and which temporal expert should model its factor exposures?

---

# 165. Relationship to MATCC

MATCC focuses on:

- trend decomposition;
- market trend guidance;
- RWKV;
- cross-stock correlations.

PRISM-VQ focuses on:

- discrete cross-sectional structure;
- financial priors;
- MoE specialization;
- factor pricing.

The optional Stage-2 trend decomposition in PRISM-VQ creates one architectural connection to MATCC-style decomposition, but it is not the primary novelty.

---

# 166. Relationship to Classical Asset Pricing

PRISM-VQ explicitly preserves:

\[
\text{return}
=
\alpha
+
\text{prior factor exposures}
+
\text{learned factor exposures}.
\]

This is much closer to classical factor thinking than MASTER/MATCC.

It can be interpreted as:

\[
\text{known factors}
+
\text{learned residual factors}.
\]

This provides a natural bridge between:

- Fama/French-style priors;
- deep latent representations.

---

# 167. Novelty Decomposition

## Existing Components

- GRU;
- Transformer;
- VQ;
- contrastive learning;
- FiLM;
- MoE;
- RoPE;
- classical factor pricing;
- expert factors.

## Main Novel Combination

\[
\text{financial priors}
+
\text{cross-sectional VQ codes}
+
\text{code-conditioned sparse MoE}
+
\text{dynamic factor loadings}.
\]

## Representational Novelty

VQ code = both:

- learned factor structure;
- routing signal.

## Financial Novelty

Explicit expert factor returns coexist with learned latent factors in the final pricing equation.

---

# 168. Mechanism Primitives

For a research Agent, PRISM-VQ decomposes into reusable primitives.

## Primitive A — Prior Factor Anchor

\[
f_p(t)
\]

from expert finance knowledge.

## Primitive B — Per-Stock Temporal Encoder

\[
x_i
\rightarrow
h_i.
\]

## Primitive C — Cross-Asset Contextualization

\[
\{h_i\}
\rightarrow
\{z_i\}.
\]

## Primitive D — Discrete Structural Prototype

\[
z_i
\rightarrow
z_{q,i}.
\]

## Primitive E — Contrastive Prototype Separation

\[
z_i
\leftrightarrow
c_k.
\]

## Primitive F — Prior-Conditioned Reconstruction

\[
(z_q,f_p)
\rightarrow
\hat x.
\]

## Primitive G — Predictive Code Regularization

\[
(z_q,f_p)
\rightarrow
\text{multi-horizon returns}.
\]

## Primitive H — Structure Token

\[
z_q
\rightarrow
\text{temporal Transformer conditioning}.
\]

## Primitive I — Code-Gated Experts

\[
z_q
\rightarrow
G.
\]

## Primitive J — Temporal Expert Computation

\[
h_{\text{temp}}
\rightarrow
\xi_j.
\]

## Primitive K — Base + Modulation Loadings

\[
\beta^{base}
\rightarrow
\gamma\odot\beta^{base}+\delta.
\]

---

# 169. What the Ablation Says Is Most Important

Evidence hierarchy:

\[
\text{Codebook}
>
\text{Expert Priors}
>
\text{MoE}
\]

on CSI300.

But on S&P500:

> all three are close to essential.

This indicates PRISM-VQ behaves more like a tightly coupled system in weaker-signal markets.

---

# 170. How to Beat This Baseline

PRISM-VQ is already much more complete than earlier baselines.

A new model must therefore target real structural limitations rather than adding generic modules.

---

# 171. What PRISM-VQ Already Does Well

- discrete denoising;
- cross-sectional structure learning;
- explicit financial priors;
- temporal Transformer;
- sparse MoE;
- dynamic factor loadings;
- cost-aware portfolio testing;
- survivorship-bias control;
- code interpretability;
- expert specialization;
- strong reproducibility.

---

# 172. Structural Weakness 1 — Frozen Codebook

After Stage 1:

\[
\mathcal Z
\]

is fixed.

But market structure can evolve.

This creates a natural tension:

> stable prototypes improve robustness, but excessive stability can become stale.

---

# 173. Research Direction — Adaptive / Slowly Evolving Codebook

Possible design:

\[
c_k(t)
=
c_k(t-1)
+
\Delta c_k(t)
\]

with strong temporal regularization.

Goal:

> allow gradual regime adaptation without losing prototype identity.

---

# 174. Structural Weakness 2 — Hard Assignment

Current:

\[
z_i
\rightarrow
\arg\min_k.
\]

This ignores assignment uncertainty.

Possible alternatives:

- soft VQ;
- Gumbel-softmax;
- top-m code mixture;
- uncertainty-aware prototype assignment.

---

# 175. Research Direction — Uncertainty-Aware Structural Routing

Use:

\[
p(k\mid z_i)
\]

instead of one code.

Then expert routing can marginalize over:

\[
p(e\mid k).
\]

This may be especially useful near regime boundaries.

---

# 176. Structural Weakness 3 — Expert Routing Uses Only Code

Routing is intentionally separated:

\[
z_q
\rightarrow
gate.
\]

Temporal context affects expert computation, but not expert selection.

This is interpretable, but may be restrictive.

---

# 177. Research Direction — Shared-Routed Hybrid Experts

Use:

- shared experts;
- code-routed experts;
- optionally regime-modulated router.

This preserves common knowledge while allowing specialization.

Hypothesis:

> shared experts improve stability, routed experts improve conditional adaptation.

---

# 178. Research Direction — Temporal-Aware Router

Current router:

\[
G_i
=
g(z_{q,i}).
\]

Potential extension:

\[
G_i
=
g(z_{q,i},h_{\text{temp},i}).
\]

This sacrifices some clean separation but may improve adaptability.

A compromise is:

\[
G_i
=
g(z_q)
+
\delta g(h_{\text{temp}})
\]

with a small residual router.

---

# 179. Structural Weakness 4 — Flat Codebook

All prototypes live in one flat set.

Possible richer structures:

- hierarchical codebook;
- global + local codebooks;
- market + sector + stock codes;
- multi-scale codebooks.

---

# 180. Research Direction — Hierarchical Discrete Factors

For example:

\[
z^{market}
\rightarrow
z^{sector}
\rightarrow
z^{stock}.
\]

This could preserve interpretability while modeling multiple levels of cross-sectional structure.

---

# 181. Structural Weakness 5 — One Structural Code per Stock

A stock may simultaneously express multiple latent factors.

Hard single-code assignment may be overly exclusive.

Potential extension:

\[
\text{stock}
\rightarrow
\{code_1,\dots,code_m\}.
\]

This becomes sparse compositional factor representation.

---

# 182. Research Direction — Compositional VQ Factors

Represent:

\[
f_{l,i}
=
\sum_{r=1}^{m}
w_{i,r}
\psi(c_{k_{i,r}}).
\]

This could model multi-factor stock identity while retaining discrete structure.

---

# 183. Structural Weakness 6 — Prior Factor Set Is Fixed

The 13 factors are selected manually.

Potential problems:

- market-specific relevance;
- time-varying factor usefulness;
- missing emerging anomalies.

---

# 184. Research Direction — Dynamic Prior Selection

Add a prior-factor gate:

\[
a_p(t)
=
g(f_p,\text{market state}).
\]

Then:

\[
\beta_p^\top
(a_p\odot f_p).
\]

This allows the model to determine which expert priors matter in each regime.

---

# 185. Structural Weakness 7 — No Explicit Ranking Loss

Evaluation is rank-centric.

Training is MSE-centric.

A strong extension could add:

\[
\mathcal L_{\text{rank}}.
\]

Examples:

- pairwise rank hinge;
- differentiable Spearman surrogate;
- listwise objective;
- top-k utility surrogate.

---

# 186. Structural Weakness 8 — One Temporal Scale

Default:

\[
T=20.
\]

Markets contain:

- short-term reversal;
- medium-term momentum;
- longer regime effects.

A multi-scale temporal model may improve factor loading stability.

---

# 187. Research Direction — Multi-Scale Temporal Experts

Parallel temporal streams:

\[
T_{short},
T_{mid},
T_{long}.
\]

Then either:

- fuse before MoE;
- route scales by code;
- assign experts per horizon.

---

# 188. Structural Weakness 9 — Cross-Sectional Structure Is Same-Date Only

Stage 1 cross-asset Transformer sees the current same-date cross-section.

It does not explicitly model cross-time stock relations like MASTER.

Potential extension:

> combine discrete structural codes with lead–lag relational modeling.

---

# 189. Research Direction — Cross-Time Structural Codes

Learn transition-aware codes:

\[
z_{q,i,t}
\rightarrow
z_{q,j,t+\Delta}.
\]

Potential mechanisms:

- code transition graph;
- relational VQ;
- graph-conditioned MoE;
- temporal prototype transitions.

---

# 190. Experiment Hook — Adaptive Codebook

Hypothesis:

> codebook drift improves late-test robustness without destroying interpretability.

Evaluate:

- yearly RankIC;
- code persistence;
- code turnover;
- prototype drift;
- expert usage stability.

---

# 191. Experiment Hook — Shared Experts

Compare:

1. routed experts only;
2. shared expert only;
3. shared + routed experts.

Measure:

- RankIC;
- RankICIR;
- seed variance;
- expert collapse;
- regime robustness.

---

# 192. Experiment Hook — Router Inputs

Compare routing by:

1. code only;
2. temporal only;
3. code + temporal;
4. code + market regime.

This directly tests PRISM-VQ's strict structural routing assumption.

---

# 193. Experiment Hook — Ranking Objective

Compare:

\[
MSE
\]

vs:

\[
MSE+\lambda L_{\text{rank}}.
\]

Monitor:

- RankIC;
- RankICIR;
- Top-K AR;
- turnover;
- calibration of predicted returns.

---

# 194. Experiment Hook — Prior Factor Ablation by Group

Instead of removing all priors, remove groups:

- valuation;
- quality;
- risk;
- momentum/reversal;
- investment/accounting.

This would reveal which domains provide the largest stabilizing effect.

---

# 195. Experiment Hook — Codebook Quality Metrics

Add quantitative VQ diagnostics:

- active code ratio;
- perplexity;
- assignment entropy;
- dead-code rate;
- prototype occupancy;
- within-code variance;
- between-code separation;
- transition entropy.

The paper already provides transition analysis, but not all standard VQ utilization metrics.

---

# 196. Experiment Hook — Code Perturbation Robustness

Randomly perturb:

- code assignment;
- prototype vector;
- structure token.

Measure how prediction degrades.

This directly quantifies how strongly the temporal model depends on discrete structure.

---

# 197. Agent Critique: Claim vs. Evidence

## Claim A

Vector quantization provides effective inductive bias.

### Evidence

Largest ablation drop, including negative S&P500 RankIC without codebook.

### Assessment

Very strongly supported at the full-mechanism level.

---

## Claim B

Financial priors stabilize weak-signal prediction.

### Evidence

S&P500 RankIC:

\[
0.0141
\rightarrow
0.0031
\]

without priors.

### Assessment

Very strongly supported.

---

## Claim C

MoE specialization improves temporal modeling.

### Evidence

MoE ablation plus statistically different expert usage.

### Assessment

Strong support for routing specialization.

Economic meaning of each expert remains less established.

---

## Claim D

Codes are interpretable and complementary to known factors.

### Evidence

- sector mixing;
- weak size/return partitioning;
- moderate JKP factor correlations.

### Assessment

Reasonably supported.

Codes appear complementary rather than direct replicas.

---

## Claim E

Model is robust economically.

### Evidence

- default transaction costs included;
- cost-sweep appendix;
- Top-K sensitivity;
- Ndrop/turnover analysis.

### Assessment

Stronger than many prior financial ML baselines.

---

# 198. Strengths

## Method

- coherent factor-model interpretation;
- clear separation of spatial structure and temporal dynamics;
- strong VQ objective;
- domain priors;
- conditional computation;
- interpretable loading equation.

## Experiments

- strong baseline coverage;
- dual-market evaluation;
- cost-aware backtests;
- significance tests;
- ablations;
- robustness;
- interpretability;
- expert analysis.

## Reproducibility

- technical appendix is unusually detailed;
- code is available;
- seeds explicitly listed.

---

# 199. Limitations

The main paper does not contain a long explicit "Limitations" section.

The following are derived from the architecture and reported experiments.

## Fixed Codebook

Potential long-term staleness.

## Manual Prior Selection

13 factors are curated rather than learned.

## Hard Assignment

No structural uncertainty.

## Market-Specific MoE Tuning

Expert count / top-k differ by market.

## No Explicit Rank Loss

Training-evaluation mismatch remains.

## Two Markets Only

Generalization beyond CSI300 / S&P500 remains open.

---

# 200. Future Work — Explicit vs. Implied

The paper's conclusion emphasizes the design principle rather than listing detailed future work.

The source supports continued investigation of:

- discrete structural modeling;
- prior-informed conditional computation;
- expert specialization.

The following are **inferred research opportunities**, not explicit author promises:

- adaptive codebooks;
- probabilistic assignments;
- dynamic prior selection;
- multi-scale temporal experts;
- cross-market shared prototypes;
- rank-aware training.

---

# 201. Writing / Presentation Lessons

## 201.1 Three Recurring Weaknesses → Three Core Mechanisms

The introduction identifies:

1. continuous-latent weakness;
2. structure-agnostic temporal modeling;
3. underused financial priors.

Then maps them to:

1. VQ;
2. code-gated MoE;
3. JKP factors.

This creates very strong narrative closure.

---

# 202. Writing Lesson — One Representation, Two Roles

A particularly elegant contribution is:

> the discrete code is both a latent representation and a routing signal.

This lets one mechanism support:

- interpretability;
- denoising;
- temporal specialization.

This is efficient innovation packaging.

---

# 203. Writing Lesson — Keep Factor Interpretation Explicit

The final prediction is not a generic MLP.

The paper returns to:

\[
\alpha
+
\beta_p^\top f_p
+
\beta_l^\top f_l.
\]

This makes the architecture easier to motivate to a finance audience.

---

# 204. Writing Lesson — Robustness Is Multi-Dimensional

The paper does not stop at:

- main RankIC.

It examines:

- hyperparameters;
- portfolio size;
- transaction cost;
- turnover;
- representation structure;
- factor correlations;
- expert specialization;
- code dynamics.

This creates a much stronger baseline paper.

---

# 205. Writing Lesson — Use Technical Appendix Aggressively

Many implementation details are moved out of the main method:

- RevIN;
- EMA code usage;
- code reinitialization;
- FiLM decoder;
- GEGLU fusion;
- base-modulation loadings;
- exact seeds;
- portfolio algorithm.

This preserves main-paper clarity without sacrificing reproducibility.

---

# 206. Reproduction Checklist

## Data

- Qlib CSI300 / S&P500
- contemporaneous constituents
- Alpha158
- T=20
- 13 JKP factor returns

## Priors

- rolling 20-day cumulative return
- end at \(t-1\)
- train-only normalization

## Labels

- main 5-day forward return
- 1–9 day auxiliary returns
- verify open \(t+1\) / close \(t+5\)

## Preprocessing

- RobustZScoreNorm
- Fillna
- CSRankNorm
- Stage-1 RevIN

## Stage 1

- GRU
- cross-asset Transformer
- \(d_s=128\)
- 1 layer / 2 heads
- codebook 512
- VQ commitment 0.25
- contrastive temperature 0.07
- \(\lambda_{contra}=1\)
- \(\lambda_{pred}=10^{-4}\)
- FiLM reconstruction
- 9-horizon predictor

## Stage 2

- codebook fixed
- \(d_t=64\)
- 1 Transformer layer
- RoPE
- FFN 128
- dropout 0.1
- CSI: 2 heads, 2 experts, top-1
- SP: 4 heads, 8 experts, top-4
- balance coefficients market-specific
- loading \(\ell_2\) penalty

## Optimization

- AdamW
- LR 1e-4
- LambdaLR
- clip 1.0
- early stopping patience 15
- seeds 0–4

## Evaluation

- RankIC / RankICIR per seed
- portfolio ensemble predictions
- K=30 / N=5
- 5/15 bp costs
- AR / MDD / SR

---

# 207. Local Reproduction

This section should be filled with the local baseline implementation.

## Paper-Reported — CSI300

\[
RankIC=0.0646
\]

\[
RankICIR=0.4224
\]

\[
AR=0.3077
\]

\[
MDD=0.1924
\]

\[
SR=1.5694.
\]

---

## Paper-Reported — S&P500

\[
RankIC=0.0141
\]

\[
RankICIR=0.1208
\]

\[
AR=0.1442
\]

\[
MDD=0.1616
\]

\[
SR=0.6701.
\]

---

## Our Reproduction — CSI300

- data period:
- constituent protocol:
- prior factor pipeline:
- seeds:
- IC:
- ICIR:
- RankIC:
- RankICIR:
- AR:
- MDD:
- Sharpe:
- Calmar:
- Sortino:
- Omega:

## Our Reproduction — S&P500

- data period:
- constituent protocol:
- prior factor pipeline:
- seeds:
- IC:
- ICIR:
- RankIC:
- RankICIR:
- AR:
- MDD:
- Sharpe:
- Calmar:
- Sortino:
- Omega:

---

# 208. Common Causes of Reproduction Gap

- static vs. contemporaneous index membership;
- JKP factor timing;
- forward-return label implementation;
- CSRankNorm implementation;
- Stage-1 RevIN;
- codebook dead-code handling;
- optional trend decomposition;
- ensemble-vs.-per-seed backtest;
- transaction costs;
- Qlib version;
- exact Fillna behavior;
- Stage-1 / Stage-2 checkpoint selection.

---

# 209. Baseline Comparison Rules

Before comparing a new model against PRISM-VQ, verify:

1. same constituent protocol;
2. same train/valid/test split;
3. same Alpha158;
4. same 13 JKP priors or clearly remove them from both models;
5. same 5-day target;
6. same CSRankNorm;
7. same seeds;
8. same cost-aware TopK-DropN;
9. same ensemble prediction protocol for portfolios;
10. same validation tuning budget.

---

# 210. Implementation Hooks

Suggested code organization:

```text
prism_vq/
├── data/
│   ├── stock_features.py
│   └── prior_factors.py
├── stage1/
│   ├── revin.py
│   ├── gru_encoder.py
│   ├── cross_asset_transformer.py
│   ├── quantizer.py
│   ├── codebook_manager.py
│   ├── contrastive_loss.py
│   ├── film_decoder.py
│   └── multihorizon_predictor.py
├── stage2/
│   ├── decomposition.py
│   ├── temporal_transformer.py
│   ├── structure_token.py
│   ├── moe_router.py
│   ├── experts.py
│   ├── loading_head.py
│   └── factor_model.py
├── losses/
│   ├── spatial_loss.py
│   └── temporal_loss.py
└── model.py
```

---

# 211. Minimal Interfaces

## Stage 1 Encoder

```text
Input:
x: [N, T, C]

Output:
h: [N, ds]
```

## Cross-Asset Transformer

```text
Input:
h: [N, ds]

Output:
z: [N, ds]
```

## Quantizer

```text
Input:
z: [N, ds]

Output:
z_q: [N, ds]
code_id: [N]
```

## Prior Factors

```text
Input:
factor returns through t-1

Output:
f_p: [P=13]
```

## Temporal Encoder

```text
Input:
x: [N, T, C]
z_q: [N, ds]

Output:
h_temp: [N, dt]
```

## Router

```text
Input:
z_q: [N, ds]

Output:
G: [N, Me]
```

## Loading Head

```text
Input:
h_temp
MoE output

Output:
alpha: [N]
beta_prior: [N, P]
beta_latent: [N, ds]
```

---

# 212. Conceptual Stage 1 Pseudocode

```text
for date in train_dates:

    x = stock_history(date)
    fp = prior_factor_vector(date)

    x_norm = RevIN(x)

    h = GRU(x_norm)

    z = CrossAssetTransformer(h)

    code_id, zq = VQ(z)

    x_recon = FiLMDecoder(zq, fp)

    y_multi = MultiHorizonPredictor(zq, fp)

    loss =
        recon_loss(x, x_recon)
        + vq_loss(z, zq)
        + lambda_contra * contrastive_code_loss(z, zq)
        + lambda_pred * multihorizon_loss(y_multi)

    update_stage1(loss)
    update_codebook_ema()
    maybe_reanchor_dead_codes()
```

---

# 213. Conceptual Stage 2 Pseudocode

```text
freeze(stage1)

for date in train_dates:

    x = stock_history(date)
    fp = prior_factor_vector(date)

    zq = fixed_stage1_code(x)

    x_temporal = maybe_decompose(x)

    h_temp = TemporalTransformer(
        structure_token=zq,
        temporal_tokens=x_temporal
    )

    routing_weights = Router(zq)

    expert_input = Fuse(
        LN(h_temp),
        LN(zq)
    )

    moe_output =
        sparse_weighted_sum(
            experts(expert_input),
            routing_weights
        )

    beta_prior_base =
        W_prior_base @ h_temp

    beta_latent_base =
        W_latent_base @ h_temp

    gamma_prior, delta_prior =
        prior_modulation(moe_output)

    gamma_latent, delta_latent =
        latent_modulation(moe_output)

    beta_prior =
        gamma_prior * beta_prior_base
        + delta_prior

    beta_latent =
        gamma_latent * beta_latent_base
        + delta_latent

    alpha = alpha_head(moe_output)

    latent_factor =
        factor_map(zq)

    y_hat =
        alpha
        + beta_prior · fp
        + beta_latent · latent_factor

    loss =
        MSE(y_hat, y)
        + load_balance_loss
        + loading_L2

    update_stage2(loss)
```

---

# 214. Agent Research Instructions

When using PRISM-VQ:

1. **Treat the codebook as the central structural mechanism.**
2. **Do not interpret the 13 JKP factors as stock characteristics; they are market-wide factor-return priors.**
3. **Preserve the no-look-ahead \(t-1\) prior-factor construction.**
4. **Remember Stage 1 and Stage 2 are deliberately decoupled.**
5. **Do not jointly update the codebook during Stage 2 unless intentionally testing a new method.**
6. **Use the appendix implementation details rather than only the simplified main-text equations.**
7. **Preserve ensemble-vs.-seed differences when reproducing portfolio metrics.**
8. **Do not claim code-economic-factor identity; reported correlations are deliberately moderate.**
9. **Treat expert routing specialization and expert economic semantics as different claims.**
10. **If improving PRISM-VQ, target real limitations such as fixed codebook, hard assignment, routing rigidity, flat code hierarchy, or objective mismatch.**
11. **Match transaction costs and TopK-DropN before claiming portfolio superiority.**
12. **Always verify whether optional trend decomposition is enabled in the exact experimental code path.**

---

# 215. Agent Takeaways

## Most Important Research Problem

How to combine:

- noisy learned factors;
- established finance priors;
- cross-sectional structure;
- time-varying dynamics;

without sacrificing interpretability.

## Most Important Representation Idea

\[
\text{cross-sectional stock embedding}
\rightarrow
\text{discrete structural prototype}.
\]

## Most Important Training Idea

Learn structure first, then freeze it:

\[
\text{Stage 1 structure}
\rightarrow
\text{Stage 2 dynamics}.
\]

## Most Important Routing Idea

\[
\text{code}
\rightarrow
\text{expert selection}.
\]

## Most Important Financial Equation

\[
\hat y_i
=
\alpha_i
+
\beta_{p,i}^{\top}f_p
+
\beta_{l,i}^{\top}f_{l,i}.
\]

## Most Important Ablation

Removing codebook:

\[
0.0646
\rightarrow
0.0472
\]

on CSI300 and:

\[
0.0141
\rightarrow
-0.0024
\]

on S&P500.

## Most Important Cross-Market Insight

S&P500 benefits from:

- priors more strongly;
- more experts;
- larger routing diversity.

## Biggest Structural Limitation

The codebook is stable but frozen.

## Most Promising Research Direction

Combine:

> stable discrete structure + adaptive structure evolution + shared/specialized expert routing.

---

# 216. Compact Architecture Summary

```text
                            PRISM-VQ
=================================================================

FINANCIAL PRIOR STREAM
----------------------
13 JKP factor returns
      │
20-day cumulative return through t-1
      │
      └────────────────────────────────────────────┐
                                                   │

STAGE 1 — SPATIAL / STRUCTURE LEARNING             │
---------------------------------------            │
Alpha158 history x_i [T=20, C=158]                 │
      │                                            │
      ▼                                            │
RevIN                                              │
      │                                            │
      ▼                                            │
GRU stock encoder                                  │
      │                                            │
      ▼                                            │
h_i [ds=128]                                       │
      │                                            │
Cross-Asset Transformer                            │
      │                                            │
      ▼                                            │
z_i                                                │
      │                                            │
      ▼                                            │
Vector Quantizer / Codebook [K=512]                │
      │                                            │
      ├───────────────┐                            │
      │               │                            │
      ▼               ▼                            │
discrete code z_q    code_id                       │
      │                                            │
      ├── contrastive code objective               │
      ├── VQ commitment objective                  │
      ├── FiLM reconstruction ◄────────────────────┘
      └── multi-horizon return prediction ◄────────┘
      │
      ▼
latent factor f_l = ψ_factor(z_q)

[freeze Stage 1 / codebook]


STAGE 2 — TEMPORAL / LOADING LEARNING
-------------------------------------
Alpha158 history
      │
(optional trend-residual decomposition)
      │
      ├────────────── z_q structure token
      │
      ▼
Temporal Transformer + RoPE
      │
      ▼
h_temp
      │
      ├──────────────┐
      │              │
      │              ▼
      │      code-conditioned router
      │              │
      │              ▼
      │         sparse Top-k experts
      │              │
      └──────────────┤
                     ▼
                  MoE output
                     │
          ┌──────────┴──────────┐
          ▼                     ▼
 temporal base loadings    structure modulation
          │                     │
          └──────────┬──────────┘
                     ▼
        β_prior, β_latent, α
                     │
                     ▼

FINAL FACTOR MODEL
------------------
ŷ_i
 =
 α_i
 + β_prior,i^T · f_prior
 + β_latent,i^T · f_latent,i
```

---

# 217. Source Location Map

Useful paper locations:

- **pp. 1–2:** motivation, contributions, relation to FactorVAE / FactorVQVAE / MASTER / MATCC.
- **pp. 2–3:** factor-model formulation, two-stage overview, cross-asset encoder, vector quantization, contrastive objective.
- **pp. 3–4:** reconstruction, multi-horizon prediction, Stage-1 loss, temporal structure token, MoE routing, factor pricing.
- **p. 5:** load balancing, two-stage training procedure, dataset protocol, prior factors, baselines, transaction-cost portfolio setup.
- **pp. 6–7:** main results, significance, component ablations, hyperparameter sensitivity, Top-K robustness, representation interpretation, expert specialization.
- **pp. 10–12:** technical appendix architecture, RoPE, freeze schedule, RevIN, codebook maintenance, FiLM decoder, exact MoE implementation, base-modulation loading head.
- **pp. 12–13:** full JKP prior factor definitions and no-look-ahead rolling construction.
- **pp. 13–15:** exact RankIC / RankICIR / AR / MDD / SR equations, TopK-DropN algorithm, transaction-cost robustness, Ndrop sensitivity.
- **pp. 15–17:** code transition dynamics and market-specific persistence / entropy analysis.

---

# 218. Compact Retrieval Summary

PRISM-VQ (Kim and Song, 2026) is a two-stage dynamic factor model for cross-sectional stock ranking that integrates vector-quantized discrete stock structure, 13 expert financial prior factors from the JKP Global Factor Library, and a code-conditioned sparse Mixture-of-Experts. Stage 1 encodes each stock's Alpha158 history with a GRU, contextualizes all stocks through a cross-asset Transformer, and quantizes the resulting representations into a 512-code discrete vocabulary. The codebook is trained with VQ commitment, contrastive prototype separation, prior-conditioned FiLM reconstruction, and 1–9 day auxiliary return prediction. Stage 2 freezes the structural codebook, prepends each stock's discrete code as a structure token to a RoPE temporal Transformer, and uses the code to route temporal representations through sparse experts. Dynamic loadings on both expert prior factors and learned latent factors are then produced through a base-plus-modulation loading head, yielding the explicit factor model \(\hat y_i=\alpha_i+\beta_{p,i}^\top f_p+\beta_{l,i}^\top f_{l,i}\). The model is evaluated on contemporaneous CSI300 and S&P500 constituents with Qlib Alpha158, training on 2009–2019, validation on 2020–2021, and strict out-of-sample testing on 2022–2024. Main RankIC/RankICIR results are 0.0646/0.4224 on CSI300 and 0.0141/0.1208 on S&P500; cost-aware TopK-DropN portfolios achieve Sharpe ratios 1.5694 and 0.6701. Ablations show the codebook is the most critical component, prior factors are especially important on S&P500, and structure-conditioned MoE contributes strongly in weaker-signal markets. The paper also provides transaction-cost sensitivity, Top-K/Ndrop robustness, code-to-factor interpretation, code transition dynamics, and statistically distinct expert routing behavior. The main research opportunities are adaptive/evolving codebooks, uncertainty-aware assignments, shared-plus-routed experts, dynamic prior selection, multi-scale temporal modeling, explicit rank objectives, and cross-time relational structure.
