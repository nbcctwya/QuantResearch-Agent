---
paper_id: FactorVQVAE_2025
title: "FactorVQVAE: Discrete latent factor model via Vector Quantized Variational Autoencoder"
authors:
  - Namhyoung Kim
  - Seung Eun Ock
  - Jae Wook Song
venue: "Knowledge-Based Systems"
volume: 318
article_number: 113460
year: 2025
doi: "10.1016/j.knosys.2025.113460"
paper_type: baseline
subtype:
  - finance_ml
  - dynamic_factor_model
  - discrete_latent_model
  - vector_quantization
  - autoregressive_transformer
task:
  - cross_sectional_stock_return_prediction
  - latent_factor_learning
  - portfolio_investment
markets:
  - CSI300
  - S&P500
dataset:
  stock_features: "Qlib Alpha158"
  market_features_cn: "63 features from CSI300, CSI500, CSI800 indices"
  market_features_us: "63 features from GSPC, DJI, NDX indices"
  sequence_length: 20
  train: "2009-01-01 to 2019-06-30"
  validation: "2019-07-01 to 2019-12-31"
  test: "2020-01-01 to 2023-06-30"
core_modules:
  - GRU_feature_extractor
  - future_return_factor_encoder
  - vector_quantizer_codebook
  - factor_decoder_alpha_beta
  - latent_factor_GRU
  - autoregressive_transformer_prior
  - market_cross_attention
  - rank_loss
core_mechanisms:
  - discrete_latent_factors
  - vector_quantization
  - codebook_learning
  - two_stage_training
  - autoregressive_token_prediction
  - market_conditioning
  - ranking_aligned_objective
  - factor_interpretability_via_codeword_usage
main_metrics:
  - RankIC
  - RankICIR
  - Annualized_Return
  - Annualized_STD
  - Maximum_Drawdown
  - Sharpe
  - Sortino
  - Calmar
main_results:
  CSI300_RankIC: 0.0508
  CSI300_RankICIR: 0.3775
  SP500_RankIC: 0.0122
  SP500_RankICIR: 0.0982
  CSI300_TopK_AR: 0.2140
  CSI300_TopK_Sharpe: 1.0777
  CSI300_LongShort_AR: 0.2760
  CSI300_LongShort_Sharpe: 3.1669
  SP500_TopK_AR: 0.2588
  SP500_TopK_Sharpe: 0.8827
  SP500_LongShort_AR: 0.1465
  SP500_LongShort_Sharpe: 1.4956
relevance_to_agent:
  baseline_understanding: very_high
  architecture_transfer: very_high
  finance_specific_adaptation: very_high
  discrete_representation_reference: very_high
  reproducibility: high
  interpretability_reference: high
priority: very_high
---

# FactorVQVAE: Discrete Latent Factor Model via Vector Quantized Variational Autoencoder

## 0. Document Purpose

This document is an Agent-oriented reconstruction of **FactorVQVAE (Kim, Ock, Song, Knowledge-Based Systems, 2025)**.

The purpose is not merely to summarize the paper. It is designed so a research Agent can use FactorVQVAE as:

1. a **baseline to reproduce and beat**;
2. a **mechanism library for discrete latent-factor modeling**;
3. a **reference for two-stage training**;
4. a **case study of adapting VQ-VAE + autoregressive Transformer to cross-sectional stock prediction**;
5. a **source of experimental protocols, ablations, regime tests, and interpretability analyses**;
6. a **direct successor / alternative to FactorVAE-style continuous latent factor modeling**.

The paper's central thesis is:

> Continuous latent factor models may suffer from noisy, redundant, or collapsed representations.  
> FactorVQVAE replaces the continuous latent factor space with a discrete codebook learned through vector quantization, and then learns a causal autoregressive Transformer prior over the resulting factor tokens.

The full system is deliberately separated into two stages:

- **Stage 1:** learn a discrete factor representation and a factor decoder;
- **Stage 2:** freeze Stage 1 and learn to predict future discrete factor tokens from previous tokens and market information.

This paper is especially important for a research Agent because it provides substantially more experimental detail than FactorVAE:

- two markets;
- multiple baseline families;
- five random seeds;
- rank metrics;
- two portfolio strategies;
- subperiod analysis;
- bull/bear/sideways regime analysis;
- VQ ablation;
- component ablation;
- hyperparameter sensitivity;
- codeword interpretability analysis.

---

# 1. Metadata

- **Title:** FactorVQVAE: Discrete latent factor model via Vector Quantized Variational Autoencoder
- **Authors:** Namhyoung Kim, Seung Eun Ock, Jae Wook Song
- **Affiliation:** Department of Industrial Engineering, Hanyang University
- **Journal:** Knowledge-Based Systems
- **Volume:** 318
- **Article:** 113460
- **Year:** 2025
- **DOI:** 10.1016/j.knosys.2025.113460
- **Received:** 30 December 2024
- **Revised:** 23 March 2025
- **Accepted:** 28 March 2025
- **Available online:** 10 April 2025
- **Primary task:** Cross-sectional stock return prediction
- **Secondary tasks:** Portfolio construction, regime robustness, latent-factor interpretation
- **Model family:** Dynamic factor model + VQ-VAE + autoregressive Transformer

---

# 2. Research Problem

## 2.1 Financial Problem

The model begins from a dynamic factor view of returns:

\[
\mathbf y_{t+\delta}
=
\boldsymbol\alpha_t
+
\sum_{k=1}^{K}
\boldsymbol\beta_t^{(k)}
\mathbf z_t^{(k)}
+
\boldsymbol\epsilon_t.
\]

The goal is to predict future cross-sectional returns for a changing set of stocks.

The desired learned predictor is:

\[
\hat{\mathbf y}_{t+\delta}
=
f(\mathbf x_t;\Theta)
=
E(\mathbf y_{t+\delta}\mid \mathbf x_t),
\]

with decomposition:

\[
\hat{\mathbf y}_{t+\delta}
=
\boldsymbol\alpha(\mathbf x_t)
+
\boldsymbol\beta(\mathbf x_t)
\mathbf z(\mathbf x_t).
\]

Here:

- \(\boldsymbol\alpha\): stock-specific return component;
- \(\boldsymbol\beta\): dynamic stock-factor loading matrix;
- \(\mathbf z\): latent market factor representation.

---

## 2.2 Representation Problem

The authors argue that existing latent-factor approaches face three related difficulties.

### Problem A — Continuous latent factors can be noisy and unstable

Continuous factor vectors can react to small fluctuations in noisy financial data.

### Problem B — Posterior collapse / redundant latent dimensions

The paper argues that VAE-style continuous latent spaces can become overly homogeneous or redundant.

### Problem C — Temporal factor structure is not modeled explicitly enough

Many latent-factor models emphasize the cross-section but treat latent factors across time as approximately independent.

The paper instead wants:

\[
z_1,z_2,\dots,z_T
\]

to form a structured temporal token sequence.

---

# 3. Motivation

The paper's motivation can be summarized as:

\[
\text{Continuous latent factors}
\rightarrow
\text{noise sensitivity + redundancy + weak factor selection}
\]

then:

\[
\text{Vector quantization}
\rightarrow
\text{finite discrete codebook}
\rightarrow
\text{stable / distinct latent states}
\]

and finally:

\[
\text{discrete factor sequence}
+
\text{autoregressive Transformer}
\rightarrow
\text{predictable temporal factor dynamics}.
\]

The intended benefits are:

1. reduce sensitivity to minor latent perturbations;
2. mitigate posterior collapse;
3. impose a structured discrete latent representation;
4. enable explicit token-level temporal modeling;
5. make codeword usage inspectable and potentially interpretable as market states / risk regimes.

---

# 4. Main Contributions

The paper states three primary contributions.

## Contribution 1 — Discrete Latent Factor Modeling

FactorVQVAE applies VQ-VAE to financial latent factor modeling.

Continuous factor embeddings are mapped to a finite codebook.

---

## Contribution 2 — Time-Varying Discrete Factors

The paper models factor evolution through a sequence of discrete latent vectors / code indices.

The latent sequence is modeled with an autoregressive Transformer.

---

## Contribution 3 — Empirical Validation Across Two Markets

The model is evaluated on:

- CSI300;
- S&P500.

The authors report improvements in:

- RankIC;
- RankICIR;
- Top-\(k\) Drop-\(n\) portfolios;
- Long–Short portfolios;
- risk-adjusted performance;
- market-regime robustness.

---

# 5. FactorVQVAE vs. FactorVAE: Core Conceptual Difference

This distinction is central for a research Agent.

## FactorVAE

Uses a continuous probabilistic latent factor distribution:

\[
z
\sim
\mathcal N(\mu,\sigma^2).
\]

Training is based on a future-informed posterior factor distribution and a history-only prior factor distribution.

Core mechanism:

\[
KL(q_{\text{post}}\|q_{\text{prior}}).
\]

---

## FactorVQVAE

Uses a discrete latent codebook:

\[
\mathcal C
=
\{z_1,z_2,\dots,z_K\}.
\]

Continuous factor embeddings are quantized:

\[
\hat z
\rightarrow
z_q
\in
\mathcal C.
\]

Then Stage 2 predicts **code indices / discrete tokens** with an autoregressive Transformer.

Core mechanism:

\[
\text{future-return factor encoder}
\rightarrow
\text{VQ code}
\rightarrow
\text{token sequence model}.
\]

---

## Agent-level Interpretation

FactorVAE asks:

> What Gaussian latent factor distribution should explain future returns, and can history predict that distribution?

FactorVQVAE asks:

> Which discrete latent factor state best represents the return structure, and can a sequential token model predict the next factor state?

This is a major change in the inductive bias.

---

# 6. Overall Architecture

FactorVQVAE uses a two-stage architecture.

```text
Stage 1: Learn discrete factor vocabulary
-----------------------------------------
Historical stock characteristics x
        │
        ▼
Feature Extractor (GRU)
        │
        ▼
Stock features e

Future return sequence y
        │
        ▼
Factor Encoder (causal multi-head attention)
        │
        ▼
Continuous factor embedding z
        │
        ▼
Vector Quantizer / Codebook
        │
        ▼
Discrete factor z_q / code index s

Stock features e + z_q
        │
        ▼
Factor Decoder
(alpha + beta × latent-factor temporal state)
        │
        ▼
Reconstructed / predicted returns


Stage 2: Learn discrete-factor prior
------------------------------------
Previous discrete factor tokens s_<t
        +
Market features m_<t
        │
        ▼
Autoregressive Transformer
(causal self-attention + market attention)
        │
        ▼
P(s_t | s_<t, m_<t)
        │
        ▼
Greedy next-token prediction
        │
        ▼
Codebook vector
        │
        ▼
Frozen Factor Decoder
        │
        ▼
Future cross-sectional returns
```

---

# 7. Two-Stage Training Principle

## Stage 1

Learn:

- factor encoder;
- vector quantizer / codebook;
- factor decoder;
- stock feature extractor.

Objective:

> reconstruct future cross-sectional returns while learning a useful discrete latent vocabulary.

## Stage 2

Freeze:

- encoder;
- decoder;
- codebook.

Train only the autoregressive prior / Transformer.

Objective:

> predict the discrete latent factor token sequence.

This separation is crucial.

The Stage 2 Transformer is **not jointly updating the Stage 1 codebook** according to the paper's described protocol.

---

# 8. VQ-VAE Preliminaries

The encoder produces a continuous latent representation:

\[
\hat z
\in
\mathbb R^{l\times n_d}.
\]

The codebook is:

\[
\mathcal C
=
\{z_k\}_{k=1}^{K},
\qquad
z_k
\in
\mathbb R^{n_z}.
\]

The quantizer selects the nearest code:

\[
z_q
=
q(\hat z)
=
\arg\min_{z_k\in\mathcal C}
\|\hat z-z_k\|.
\]

Small movements in \(\hat z\) need not change the selected code.

This is the main robustness intuition.

---

# 9. Decoder in Generic VQ-VAE

The quantized representation is decoded as:

\[
\hat y
=
D(z_q)
=
D(q(E(y))).
\]

Because nearest-neighbor quantization is non-differentiable, VQ-VAE uses straight-through / stop-gradient style training.

The paper writes the generic VQ loss:

\[
\mathcal L_{\text{VQ}}
=
\|y-\hat y\|_2^2
+
\|sg[E(y)]-z_q\|_2^2
+
\|sg[z_q]-E(y)\|_2^2.
\]

Here:

\[
sg[\cdot]
\]

is the stop-gradient operator.

---

# 10. Codebook Optimization Enhancements

The paper does not use only the simplest vanilla VQ-VAE update.

It follows techniques from Huh et al. (2023), including:

- shared affine parameterization of code vectors;
- alternating optimization between the quantizer and encoder-decoder;
- a synchronized update rule / modified commitment strategy.

Purpose:

> reduce index collapse and improve codebook utilization.

This is an important implementation detail.

A research Agent should not assume the codebook is trained with a completely standard VQ-VAE implementation.

---

# 11. Dynamic Factor Problem Formulation

Future cross-sectional returns:

\[
\mathbf y_t
=
\frac{
\mathbf{price}_{t+1}
-
\mathbf{price}_{t}
}{
\mathbf{price}_{t}
}
\in
\mathbb R^{N_t}.
\]

Dynamic factor model:

\[
\mathbf y_{t+\delta}
=
\boldsymbol\alpha_t
+
\sum_{k=1}^{K}
\boldsymbol\beta_t^{(k)}
\mathbf z_t^{(k)}
+
\boldsymbol\epsilon_t.
\]

The paper interprets latent factors as encoding things such as:

- market regimes;
- macroeconomic drivers;
- risk exposures.

This interpretation is an intended semantic reading of the latent space, not a hard identification constraint.

---

# 12. Input Tensor

Historical stock characteristics:

\[
x
\in
\mathbb R^{N\times T\times C}.
\]

Feature extractor output:

\[
e
=
\phi_{\text{feat}}(x)
\in
\mathbb R^{N\times T\times H}.
\]

This differs from FactorVAE's simpler final-time stock feature representation.

FactorVQVAE retains a temporal feature sequence.

---

# 13. Stage 1 — Feature Extractor

The paper uses a GRU.

Purpose:

> model temporal interdependencies in stock characteristics and produce dynamic stock feature representations.

Formally:

\[
e=\phi_{\text{feat}}(x).
\]

The paper does not provide all internal GRU equations because this component uses a standard GRU architecture.

---

# 14. Stage 1 — Factor Encoder

The factor encoder extracts the latent factor from future stock returns.

The paper applies a multi-head self-attention mechanism to future-return representations.

First:

\[
\tilde y
=
Linear(y).
\]

Then for attention head \(i\):

\[
q_i
=
\tilde y W_i^q,
\]

\[
k_i
=
\tilde y W_i^k,
\]

\[
v_i
=
\tilde y W_i^v.
\]

The latent factor representation is:

\[
z(y)
=
Concat[
head_1,
head_2,
\dots,
head_h
]W^o.
\]

Each head is:

\[
head_i
=
softmax
\left(
\frac{
q_i k_i^\top
+
mask\cdot(-\infty)
}{
\sqrt{d_k}
}
\right)
v_i.
\]

---

# 15. Causal Mask in the Stage-1 Encoder

The paper uses:

\[
mask
=
tril(\mathbf 1).
\]

Purpose stated by the authors:

> ensure each temporal position attends only to current and previous time steps.

This introduces temporal causality into the encoded factor sequence.

Important nuance:

- the factor encoder is still a **training-time future-return encoder**;
- the causal mask prevents within-sequence look-ahead beyond the current token position;
- the exact mapping between "prediction date," "future-return sequence," and token time index should be checked in code for an exact reproduction.

The paper gives the conceptual rule clearly, but not every indexing implementation detail.

---

# 16. Stage 1 — Vector Quantizer

The factor embedding \(z\) is assigned to a nearest code vector:

\[
z_q
=
z_i,
\qquad
i
=
\arg\min_k
d(z,z_k).
\]

The original VQ-VAE uses Euclidean distance.

The codebook is:

\[
\mathcal C
=
\{z_k\}_{k=1}^{K}.
\]

Each codeword has dimension:

\[
H.
\]

---

# 17. Meaning of the Codeword

In the paper's intended interpretation, a codeword can represent:

- a latent factor;
- a market state;
- a risk regime;
- a recurring return-structure pattern.

However, the codebook is learned purely from the predictive/reconstruction objective.

No economic label is imposed during training.

Thus:

> economic meaning is discovered post hoc, not supervised.

---

# 18. Stage 1 — Factor Decoder

The decoder predicts returns using:

\[
\hat y
=
\phi_{\text{dec}}
(z_q,e).
\]

The factor-model decomposition is:

\[
\hat y
=
\alpha
+
\beta\cdot h_t.
\]

Here:

\[
\alpha
=
AlphaLayer(e)
=
eW_\alpha+b_\alpha,
\]

\[
\beta
=
BetaLayer(e)
=
eW_\beta+b_\beta,
\]

and:

\[
h_t
=
GRU(h_{t-1},z_q).
\]

---

# 19. Alpha Layer

The paper describes \(\alpha\) as similar to Jensen's alpha.

It is intended to represent the stock-specific intercept component.

The paper states that a linear transformation followed by GELU is used.

Conceptually:

\[
e
\rightarrow
\alpha.
\]

---

# 20. Beta Layer

The beta layer produces latent-factor sensitivities.

Conceptually:

\[
e
\rightarrow
\beta.
\]

The paper states that the transformation is followed by GELU.

The factor loading is dynamic because it is a function of the current stock feature representation.

---

# 21. Temporal Dynamics in the Decoder

A notable difference from FactorVAE is that the quantized latent factors themselves are processed through a GRU:

\[
h_t
=
GRU(h_{t-1},z_q).
\]

Thus the decoder does not simply use:

\[
\beta z_q.
\]

It uses a temporally updated latent-factor state:

\[
\beta h_t.
\]

This provides one additional temporal modeling layer already in Stage 1.

---

# 22. Stage 1 Objective

The paper defines:

\[
\mathcal L_{\text{VQ}}
=
\mathcal L_{\text{rec}}
+
\mathcal L_{\text{codebook}}.
\]

---

# 23. Stage 1 Reconstruction Loss

The reconstruction loss is a negative log-likelihood:

\[
\mathcal L_{\text{rec}}
=
-
\frac{1}{N}
\sum_{i=1}^{N}
\log
P_{\phi_{\text{dec}}}
\left(
\hat y_{\text{rec}}^{(i)}
=
\hat y^{(i)}
\mid
x,z_q
\right).
\]

The notation in the paper is somewhat awkward because the prediction and target symbols are close.

Operationally, the role is clear:

> train the decoder so discrete factor codes reconstruct / explain future stock returns.

---

# 24. Stage 1 Codebook Loss

The paper writes:

\[
\mathcal L_{\text{codebook}}
=
\frac{1}{N}
\sum_{i=1}^{N}
\left(
\|
sg[z^{(i)}]-z_q^{(i)}
\|_2^2
+
\beta
\|
sg[z_q^{(i)}]-z^{(i)}
\|_2^2
\right).
\]

where:

\[
\beta\in[0,1]
\]

controls the update trade-off.

Do not confuse this \(\beta\) with the **financial factor-loading matrix** \(\boldsymbol\beta\).

They are different symbols serving different roles.

---

# 25. Stage 2 — Prior Learning

After Stage 1:

- factor encoder is fixed;
- factor decoder is fixed;
- codebook is fixed.

The quantizer converts latent factors into code indices:

\[
s_t
\in
\{0,1,\dots,K-1\}.
\]

A token sequence is:

\[
s
=
[s_t]_{t=1}^{T}.
\]

The Stage 2 Transformer learns:

\[
p_\theta(s_t\mid s_{<t}).
\]

With market conditioning, the actual model is:

\[
p_\theta
(
s_i
\mid
s_{<i},m_{<i}
).
\]

---

# 26. Autoregressive Transformer

The paper uses a causal Transformer in a GPT-like manner.

At each step:

\[
s_t
\]

is predicted from the preceding token sequence.

The generic autoregressive objective is:

\[
\mathcal L_{\text{AR}}
=
-
E_s
\left[
\sum_{i=1}^{N}
\log
p(s_i\mid s_{<i})
\right].
\]

With market information:

\[
\mathcal L_{\text{AR}}
=
-
E_s
\left[
\sum_i
\log
p_\theta
(
s_i
\mid
s_{<i},m_{<i}
)
\right].
\]

---

# 27. Market Attention Mechanism

FactorVQVAE adds market-wide information to the Transformer.

Market features:

\[
m
\in
\mathbb R^{T\times d_k}.
\]

Keys:

\[
k
=
mW_k.
\]

Values:

\[
v
=
mW_v.
\]

The Stage 2 architecture therefore contains:

1. causal token self-attention;
2. market cross-attention;
3. feed-forward network;
4. residual Add & Norm structure.

This block is repeated multiple layers.

---

# 28. Purpose of Market Attention

The paper argues that latent factor transitions should not depend only on previous factor tokens.

They should also depend on the state of the broader market.

Thus:

\[
p(s_t\mid s_{<t})
\]

becomes:

\[
p(s_t\mid s_{<t},m_{<t}).
\]

This is a finance-specific modification of a generic autoregressive prior.

---

# 29. Rank Loss

FactorVQVAE explicitly adds a ranking-aligned objective.

The paper defines:

\[
\mathcal L_P
=
\sum_{i=1}^{N}
\|
p_i^t-y_i^t
\|^2
+
\lambda
\sum_{i=1}^{N}
\sum_{j=1}^{N}
\max
\left[
0,
-
(p_i^t-p_j^t)
(y_i^t-y_j^t)
\right].
\]

This contains two pieces.

## Pointwise term

\[
\sum_i
\|p_i-y_i\|^2.
\]

Purpose:

> align predicted return levels with realized returns.

## Pairwise ranking term

\[
\sum_{i,j}
\max
[
0,
-(p_i-p_j)(y_i-y_j)
].
\]

Purpose:

> penalize predicted orderings inconsistent with true cross-sectional ordering.

---

# 30. Stage 2 Total Objective

The final Stage 2 objective is:

\[
\mathcal L_{\text{RP}}
=
\mathcal L_{\text{AR}}
+
\eta
\mathcal L_P.
\]

Here:

- \(\mathcal L_{\text{AR}}\): token prediction objective;
- \(\mathcal L_P\): return/ranking objective;
- \(\eta\): weight on rank loss.

This creates a useful hybrid target:

> predict the correct latent factor token **and** optimize the downstream financial ranking objective.

---

# 31. One-Step Prediction Protocol

The paper focuses on:

\[
\delta=1.
\]

It predicts only the immediate next interval.

The authors explicitly contrast this with long autoregressive generation.

The preceding historical sequence is observed and used as context.

---

# 32. Greedy Decoding

At each prediction step, the model selects:

\[
\hat s_t
=
\arg\max_s
p_\theta(s\mid s_{<t},m_{<t})
\]

rather than sampling.

The paper argues greedy decoding is preferable for short-horizon stock prediction because stochastic sampling could introduce additional prediction error.

---

# 33. Dataset

The paper uses two markets.

## CSI300

Large and liquid China A-share stocks.

## S&P500

Large U.S. stocks.

This is a major expansion over FactorVAE, which is evaluated only on the China A-share setting in its original paper.

---

# 34. Stock Features

The paper uses Qlib's:

\[
Alpha158.
\]

Unlike the original FactorVAE paper, this paper states that it uses Alpha158 for comparability and additionally creates separate market information.

The exact treatment of all 158 stock features should be checked in code if reproducing dimensional details, but the source states Alpha158 as the stock-feature dataset.

---

# 35. Market Features

The paper constructs:

\[
63
\]

market information features for each market.

## China

Derived from:

- CSI300;
- CSI500;
- CSI800.

## United States

Derived from:

- GSPC;
- DJI;
- NDX.

These market features are used by Stage 2 market attention.

---

# 36. Missing-Value Handling

The paper states missing values are:

1. filled with values from prior time points where available;
2. otherwise replaced by zero.

A reproduction should preserve this order.

---

# 37. Sequence Length

\[
T=20.
\]

---

# 38. Prediction Label

For stock \(i\):

\[
y^{(i)}
=
\frac{
price_{t+2}^{(i)}
-
price_{t+1}^{(i)}
}{
price_{t+1}^{(i)}
}.
\]

This is the same one-day-ahead label convention used in the FactorVAE paper.

---

# 39. Data Split

Both CSI300 and S&P500 use:

## Training

\[
2009\text{-}01\text{-}01
\rightarrow
2019\text{-}06\text{-}30
\]

## Validation

\[
2019\text{-}07\text{-}01
\rightarrow
2019\text{-}12\text{-}31
\]

## Test

\[
2020\text{-}01\text{-}01
\rightarrow
2023\text{-}06\text{-}30.
\]

The test period spans 3.5 years.

The paper additionally analyzes it in **seven six-month subperiods**.

---

# 40. Baselines

The paper evaluates nine baseline models.

## XGB

Tree-based gradient boosting.

Role:

> strong non-deep-learning financial baseline.

## GRU

Sequential recurrent model.

## TCN

Temporal convolutional model.

## Transformer

Standard self-attention time-series model.

## GAT

Graph Attention Network.

## iTransformer

Transformer with inverted time/variable representation.

## MASTER

Market-aware Stock Transformer.

Especially important because it explicitly uses market information.

## CAE

Conditional Autoencoder dynamic factor model.

## FactorVAE

Continuous probabilistic dynamic factor model.

This is the most important latent-factor predecessor.

---

# 41. Baseline Implementation Protocol

The paper is unusually explicit.

## Reimplemented

- CAE;
- FactorVAE.

Reason:

> official code unavailable.

## Official code adapted

- MASTER.

The official MASTER implementation is adapted to the Qlib-based dataset.

## Qlib implementations used directly

- XGB;
- Transformer;
- GAT;
- GRU;
- TCN.

The paper says these are used without modification.

This matters when evaluating baseline fairness and reproducibility.

---

# 42. Data Preprocessing

FactorVQVAE is implemented inside Qlib.

The paper explicitly states that preprocessing follows Qlib's default pipeline, including:

\[
CSRankNorm.
\]

This is a critical reproduction detail because cross-sectional normalization can materially affect RankIC.

---

# 43. Ranking Metrics

## Rank IC

For day \(s\):

\[
RankIC_s
=
\frac{1}{N_s}
\frac{
(r_{\hat y_s}-mean(r_{\hat y_s}))^\top
(r_{y_s}-mean(r_{y_s}))
}{
std(r_{\hat y_s})
\cdot
std(r_{y_s})
}.
\]

Average RankIC:

\[
RankIC
=
\frac{1}{T_{\text{test}}}
\sum_{s=1}^{T_{\text{test}}}
RankIC_s.
\]

---

## Rank ICIR

The paper describes RankICIR as a z-score / volatility-adjusted RankIC measure used to assess consistency.

The exact explicit equation is not printed in the provided section.

Do not silently substitute a formula if reproducing from paper only.

In practice, it is clearly intended as a stability-normalized RankIC statistic consistent with the financial literature.

---

# 44. Top-\(k\) Drop-\(n\) Strategy

The paper uses:

\[
k=10\%
\]

of the asset universe.

Therefore:

## CSI300

\[
k=30.
\]

## S&P500

\[
k=50.
\]

And:

\[
n=5.
\]

The strategy keeps high-ranked stocks while replacing only a limited number each rebalance.

---

# 45. Long–Short Strategy

Stocks are sorted into ten quantiles.

The strategy:

- long highest-ranked decile;
- short lowest-ranked decile.

This provides a more direct market-neutral test of the cross-sectional signal.

---

# 46. Portfolio Metrics

The paper reports:

- Annualized Return (AR);
- annualized / return Standard Deviation (STD);
- Maximum Drawdown (MDD);
- Sharpe Ratio;
- Sortino Ratio;
- Calmar Ratio.

Risk-free rate:

\[
R_f=0.
\]

---

# 47. Sharpe Ratio

\[
Sharpe
=
\frac{
E[R_p-R_f]
}{
STD
}.
\]

With:

\[
R_f=0,
\]

the numerator is treated as annualized return.

---

# 48. Sortino Ratio

\[
Sortino
=
\frac{
E[R_p-R_f]
}{
DD
},
\]

where \(DD\) is the standard deviation of negative portfolio returns.

---

# 49. Calmar Ratio

\[
Calmar
=
\frac{
E[R_p-R_f]
}{
MDD
}.
\]

---

# 50. Implementation Details

## Framework

- PyTorch
- Qlib

## Hardware

- AMD Ryzen Threadripper PRO 3975WX
- 32 cores
- 256 GB RAM
- NVIDIA RTX 3090
- 24 GB VRAM

---

# 51. Optimizer

Both stages use:

\[
AdamW.
\]

Learning rate:

\[
1\times10^{-4}.
\]

Scheduler:

> cosine annealing.

---

# 52. Stage 1 Training

Reported settings:

- model size: 128;
- latent dimension: 64;
- epochs: 10;
- approximate runtime: 20 minutes per run.

The paper does not give a single concise Stage-1 codebook size here because codebook size is treated as a market-specific sensitivity parameter later.

---

# 53. Stage 2 Training

The implementation section reports exploration of:

- layers: 1, 2;
- attention heads: 2, 4;
- model dimension: 32, 64.

Optimizer-specific settings:

\[
\beta_{\text{AdamW}}
=
(0.9,0.98),
\]

\[
\epsilon
=
10^{-6},
\]

\[
weight\_decay
=
10^{-3}.
\]

Gradient clipping:

\[
3.
\]

Maximum epochs:

\[
100.
\]

Early stopping:

> based on validation loss.

Approximate Stage 2 runtime:

> about one hour.

---

# 54. Total Training Cost

Overall FactorVQVAE training time:

> approximately 80 minutes.

The paper compares this favorably against:

- FactorVAE: 2–3 hours;
- larger Transformer: about 1.5 hours.

This makes the computational-cost argument more nuanced:

> FactorVQVAE has a structurally complex two-stage design, but the chosen dimensions keep training relatively manageable.

---

# 55. Random Seeds

All experiments are repeated:

\[
5
\]

times with different random seeds.

The paper reports means and standard deviations for robust comparisons.

However, the main Table 2 displays only mean values, not the explicit ± standard deviations.

---

# 56. Main Prediction Results — Table 2

| Model | CSI300 RankIC | CSI300 RankICIR | S&P500 RankIC | S&P500 RankICIR |
|---|---:|---:|---:|---:|
| XGB | 0.0363 | 0.2649 | 0.0117 | 0.0946 |
| GRU | 0.0372 | 0.2647 | 0.0110 | 0.0802 |
| TCN | 0.0027 | 0.0199 | 0.0017 | 0.0169 |
| Trans | 0.0394 | 0.2881 | 0.0091 | 0.0504 |
| GAT | 0.0364 | 0.2353 | 0.0070 | 0.0384 |
| iTransformer | 0.0234 | 0.1928 | 0.0024 | 0.0251 |
| MASTER | 0.0447 | 0.2806 | 0.0027 | 0.0166 |
| CAE | 0.0352 | 0.2353 | 0.0069 | 0.0395 |
| FactorVAE | 0.0340 | 0.2620 | 0.0061 | 0.0657 |
| **FactorVQVAE** | **0.0508*** | **0.3775*** | **0.0122** | **0.0982** |

The asterisk indicates statistically significant improvement over the second-best model using a t-test at 5% significance.

The table marks significance for CSI300, not for S&P500.

---

# 57. CSI300 Main Result

FactorVQVAE:

\[
RankIC=0.0508,
\]

\[
RankICIR=0.3775.
\]

Best competing RankIC:

\[
MASTER=0.0447.
\]

Relative RankIC improvement:

\[
\frac{0.0508-0.0447}{0.0447}
\approx
13.6\%.
\]

This aligns with the paper's "about 13%" statement.

---

# 58. Important Table/Text Inconsistency

The prose states that MASTER is the second-best model with:

\[
RankICIR=0.2806.
\]

But Table 2 shows:

\[
Transformer\ RankICIR=0.2881,
\]

which is larger than MASTER's:

\[
0.2806.
\]

Therefore, for CSI300:

- second-best **RankIC** = MASTER;
- second-best **RankICIR** = Transformer, according to the table.

A research Agent should prefer the explicit table values and record this discrepancy.

---

# 59. S&P500 Main Result

FactorVQVAE:

\[
RankIC=0.0122,
\]

\[
RankICIR=0.0982.
\]

XGB:

\[
RankIC=0.0117,
\]

\[
RankICIR=0.0946.
\]

The improvement is much smaller than in CSI300.

This suggests:

> the model's advantage is market-dependent.

That conclusion is also consistent with later hyperparameter-sensitivity analysis.

---

# 60. Interpretation of Cross-Market Gap

The paper interprets the smaller S&P500 gains as consistent with a more mature / efficient market.

For a research Agent, the more useful observation is:

> FactorVQVAE does not dominate all baselines by a large margin in every market.

CSI300 is where the architecture shows its strongest advantage.

This distinction matters when using the model as a baseline.

---

# 61. Top-\(k\) Drop-\(n\) Results — CSI300

| Model | AR | STD | MDD | Sharpe | Sortino | Calmar |
|---|---:|---:|---:|---:|---:|---:|
| XGB | 0.2010 | 0.2344 | -0.2742 | 0.8575 | 0.8352 | 0.7330 |
| GRU | 0.1895 | 0.2145 | -0.2442 | 0.8834 | 0.8670 | 0.7762 |
| TCN | 0.1456 | 0.2228 | -0.2692 | 0.6532 | 0.6325 | 0.5407 |
| Trans | 0.1832 | 0.2203 | -0.2740 | 0.8315 | 0.8147 | 0.6686 |
| GAT | 0.2056 | 0.2185 | -0.2665 | 0.9410 | 0.9370 | 0.7714 |
| iTransformer | 0.0826 | 0.1946 | -0.2590 | 0.4243 | 0.4182 | 0.3187 |
| MASTER | 0.1772 | 0.2176 | -0.2865 | 0.8144 | 0.8150 | 0.6184 |
| CAE | 0.1894 | 0.1991 | -0.1936 | 0.9516 | 0.9150 | 0.9784 |
| FactorVAE | 0.1641 | 0.1937 | -0.2111 | 0.8468 | 0.8341 | 0.7770 |
| **FactorVQVAE** | **0.2140** | 0.1985 | -0.2259 | **1.0777** | **1.0465** | 0.9473 |

FactorVQVAE has:

- highest AR;
- highest Sharpe;
- highest Sortino;
- second-level / competitive Calmar;
- not the lowest drawdown.

CAE has the best MDD and slightly better Calmar.

---

# 62. Top-\(k\) Drop-\(n\) Results — S&P500

| Model | AR | STD | MDD | Sharpe | Sortino | Calmar |
|---|---:|---:|---:|---:|---:|---:|
| XGB | **0.2619** | 0.3090 | -0.3915 | 0.8476 | 0.8524 | 0.6689 |
| GRU | 0.2324 | 0.2658 | -0.3634 | 0.8742 | 0.8786 | 0.6396 |
| TCN | 0.1399 | 0.2730 | -0.3757 | 0.5123 | 0.4988 | 0.3723 |
| Trans | 0.2357 | 0.3166 | -0.4251 | 0.7445 | 0.7531 | 0.5545 |
| GAT | 0.1820 | 0.2739 | -0.4133 | 0.6647 | 0.6521 | 0.4405 |
| iTransformer | 0.1631 | 0.2715 | -0.3904 | 0.6005 | 0.5824 | 0.4177 |
| MASTER | 0.1283 | 0.2396 | -0.3705 | 0.5355 | 0.5070 | 0.3463 |
| CAE | 0.2425 | 0.2678 | **-0.3164** | **0.9056** | **0.9389** | **0.7664** |
| FactorVAE | 0.1836 | 0.2503 | -0.3775 | 0.7336 | 0.6956 | 0.4864 |
| FactorVQVAE | **0.2588** | 0.2932 | -0.3638 | **0.8827** | **0.9126** | **0.7113** |

FactorVQVAE is competitive but not best on every metric.

Specifically:

- XGB has higher raw AR;
- CAE has the best Sharpe, Sortino, Calmar, and MDD;
- FactorVQVAE is generally second-tier / near-top.

This is important because the abstract's broad language should not be read as "best on every portfolio metric in both markets."

---

# 63. Long–Short Results — CSI300

| Model | AR | STD | MDD | Sharpe | Sortino | Calmar |
|---|---:|---:|---:|---:|---:|---:|
| XGB | 0.2392 | 0.1097 | -0.0954 | 2.1812 | 2.3487 | 2.5063 |
| GRU | 0.2002 | 0.0978 | -0.0923 | 2.0476 | 2.1475 | 2.1700 |
| TCN | 0.1151 | 0.1013 | -0.1085 | 1.1371 | 1.1712 | 1.0608 |
| Trans | 0.2308 | 0.1030 | -0.1088 | 2.2409 | 2.3856 | 2.1215 |
| GAT | 0.1700 | 0.1057 | -0.0935 | 1.6082 | 1.6972 | 1.8179 |
| iTransformer | 0.0822 | **0.0608** | -0.0762 | 1.3510 | 1.4033 | 1.0782 |
| MASTER | 0.2036 | 0.0986 | -0.0868 | 2.0653 | 2.2005 | 2.3462 |
| CAE | 0.2532 | 0.0987 | -0.0662 | 2.5661 | 2.6532 | 3.8244 |
| FactorVAE | 0.1363 | 0.0768 | **-0.0656** | 1.7750 | 1.8240 | 2.0789 |
| **FactorVQVAE** | **0.2760** | 0.0871 | -0.0674 | **3.1669** | **3.5867** | **4.0932** |

This is one of the strongest result tables for the model.

FactorVQVAE achieves the best:

- AR;
- Sharpe;
- Sortino;
- Calmar.

Its drawdown is also near the best.

---

# 64. Long–Short Results — S&P500

| Model | AR | STD | MDD | Sharpe | Sortino | Calmar |
|---|---:|---:|---:|---:|---:|---:|
| XGB | **0.1513** | 0.1073 | -0.0718 | 1.4105 | 1.6641 | **2.1081** |
| GRU | 0.1012 | 0.0656 | -0.0774 | **1.5429** | 1.6751 | 1.3068 |
| TCN | -0.0001 | 0.0590 | -0.0835 | -0.0010 | -0.0010 | -0.0007 |
| Trans | 0.1208 | 0.1130 | -0.1228 | 1.0688 | 1.2045 | 0.9839 |
| GAT | 0.0703 | 0.0792 | -0.0907 | 0.8872 | 0.9838 | 0.7745 |
| iTransformer | 0.0262 | 0.0473 | **-0.0486** | 0.5548 | 0.6366 | 0.5398 |
| MASTER | -0.0086 | **0.0403** | -0.0895 | -0.2143 | -0.2032 | -0.0964 |
| CAE | 0.1249 | 0.0942 | -0.1513 | 1.3255 | 1.5859 | 0.8251 |
| FactorVAE | 0.0213 | 0.0378 | -0.0594 | 0.5631 | 0.5703 | 0.3589 |
| **FactorVQVAE** | **0.1465** | 0.0980 | -0.1006 | **1.4956** | **1.8316** | **1.4571** |

FactorVQVAE has:

- second-highest AR;
- second-highest Sharpe;
- highest Sortino;
- strong but not best Calmar.

Again, the model is strong but not universally dominant.

---

# 65. Subperiod Robustness

The test period is divided into seven semiannual intervals.

The paper visualizes RankIC distributions across these intervals.

## CSI300

FactorVQVAE:

- median RankIC above roughly 0.04;
- relatively narrow spread;
- strong temporal stability.

## S&P500

FactorVQVAE has the strongest median RankIC among the displayed models, with XGB close behind.

The paper argues this supports robustness across different market conditions.

---

# 66. Average Portfolio Rank Across Subperiods

Table 5 converts each model's metric performance into an average rank across the semiannual periods.

This is useful because it evaluates **persistence of relative performance**, not only full-period aggregates.

For CSI300, FactorVQVAE ranks especially well in risk-adjusted metrics.

For S&P500, FactorVQVAE remains competitive and often strong in Sharpe / Sortino / Calmar, though it is not uniformly first.

---

# 67. Vector Quantization Representation Analysis

The paper addresses:

- RQ4: does VQ mitigate posterior collapse / create distinct representations?
- RQ5: what happens if VQ is removed?

Two types of evidence are provided:

1. t-SNE visualization;
2. direct VQ ablation.

---

# 68. t-SNE Visualization

The paper compares latent representations from:

- FactorVAE;
- FactorVQVAE.

Reported visual pattern:

## FactorVAE

Dense, relatively homogeneous latent cluster.

## FactorVQVAE

Structured, separated, topologically distinct manifold-like representation.

The authors interpret this as evidence that vector quantization preserves heterogeneous factor states and mitigates posterior collapse.

---

# 69. Critical Interpretation of t-SNE Evidence

t-SNE is a nonlinear visualization tool.

Therefore:

> visually separated clusters are suggestive, but they are not by themselves a rigorous statistical proof of posterior collapse avoidance.

A stronger future analysis could use:

- codebook perplexity;
- active-code ratio;
- mutual information;
- cluster stability;
- entropy of code usage;
- explained-return variance per code;
- factor diversity / orthogonality;
- code transition entropy.

The paper's t-SNE result is useful qualitative evidence, not a complete diagnosis.

---

# 70. Removing Vector Quantization

The paper constructs a continuous-only ablation.

Changes:

- Stage 1 uses a continuous latent space;
- no discrete quantizer;
- Stage 2 Transformer consumes continuous latent embeddings instead of token indices.

Reported CSI300 result:

\[
RankIC:
0.0505
\rightarrow
0.0257
\]

when VQ is removed.

RankICIR:

\[
0.3824
\rightarrow
0.2402.
\]

This is a large degradation.

---

# 71. Important Numerical Discrepancy in VQ Ablation Section

The main Table 2 reports full FactorVQVAE as:

\[
RankIC=0.0508,
\qquad
RankICIR=0.3775.
\]

But the VQ ablation discussion uses:

\[
0.0505,
\qquad
0.3824
\]

as the full reference.

The paper does not explicitly explain the small discrepancy.

Possible reasons could include a separately rerun ablation setting, but this should **not be assumed as fact**.

For reproduction:

> preserve both reported values and verify against code / supplementary materials if available.

---

# 72. What the VQ Ablation Proves

Strong supported conclusion:

> within the FactorVQVAE pipeline, discrete quantization materially improves CSI300 RankIC and RankICIR relative to the continuous counterpart.

Less strongly supported claim:

> the entire gain is specifically due to "posterior collapse mitigation."

Why?

Because the ablation simultaneously changes:

- representation type;
- codebook bottleneck;
- tokenization;
- Transformer input representation;
- effective regularization.

Therefore the performance gain demonstrates the utility of the discrete design as a whole more directly than it proves one isolated causal mechanism.

---

# 73. Component Ablation

The paper removes three components separately.

## FactorVQVAE-\(\alpha\)

Remove market information attention.

\[
RankIC=0.0495.
\]

## FactorVQVAE-\(\beta\)

Remove rank loss.

\[
RankIC=0.0458.
\]

## FactorVQVAE-\(\gamma\)

Replace Transformer with GRU.

\[
RankIC=0.0430.
\]

## Full

\[
RankIC=0.0508.
\]

---

# 74. Ablation Contribution Magnitudes

Absolute drops relative to full model:

## Market attention

\[
0.0508-0.0495
=
0.0013.
\]

Small but positive.

## Rank loss

\[
0.0508-0.0458
=
0.0050.
\]

Material.

## Transformer → GRU

\[
0.0508-0.0430
=
0.0078.
\]

Largest among these ablations.

---

# 75. What the Component Ablation Suggests

The paper concludes:

1. Transformer temporal modeling is the most critical of the three tested Stage 2 components;
2. rank-aligned optimization contributes meaningfully;
3. market attention provides a smaller incremental gain.

This is important for future modifications.

A research Agent should not over-invest in market-attention complexity before considering:

- token-sequence modeling;
- ranking objective;
- discrete representation quality.

---

# 76. Hyperparameter Sensitivity

The paper studies:

- codebook size;
- Transformer hidden/model dimension;
- number of heads;
- number of layers.

This is done for both markets.

Model selection primarily targets highest RankIC.

---

# 77. Best CSI300 Configuration

Reported best configuration:

- codebook size:
  \[
  512
  \]
- Transformer dimension:
  \[
  64
  \]
- attention heads:
  \[
  2
  \]
- layers:
  \[
  2
  \]

Performance:

\[
RankIC=0.0508.
\]

Long–Short Sharpe:

\[
3.1711
\]

in the sensitivity discussion.

The 128-code codebook gives a slightly higher Top-\(k\) Drop-\(n\) Sharpe:

\[
1.1021
\]

versus:

\[
1.0777
\]

for codebook 512.

The authors still prioritize the configuration with highest RankIC.

---

# 78. Best S&P500 Configuration

Reported best configuration:

- codebook size:
  \[
  128
  \]
- Transformer dimension:
  \[
  32
  \]
- heads:
  \[
  4
  \]
- layers:
  \[
  2
  \]

Performance:

\[
RankIC=0.0122.
\]

Long–Short Sharpe:

\[
1.4974.
\]

Top-\(k\) Sharpe:

\[
0.8827.
\]

---

# 79. Market-Specific Codebook Interpretation

The paper argues that:

## CSI300

Larger codebook helps because the market is more diverse / volatile.

## S&P500

Increasing codebook size from 128 to 256 or 512 worsens performance.

Possible interpretation offered by the authors:

> excessive discretization can introduce unnecessary complexity / noise in a mature market.

This is a valuable design lesson:

> codebook capacity should be matched to market complexity rather than assumed universal.

---

# 80. Important Implementation/Sensitivity Inconsistency

The implementation section says Stage 2 explores:

- layers: 1, 2;
- heads: 2, 4.

But the sensitivity discussion / Figure 9 includes experiments with:

- 8 heads;
- more than 2 layers (the figure includes a 4-layer setting).

Therefore the paper's hyperparameter-search description is not fully aligned with the later sensitivity figure.

Possible interpretation:

> the sensitivity analysis may use an expanded grid beyond the main tuning grid.

However, because the paper does not explicitly state this distinction, an exact reproduction should verify the actual sweep.

---

# 81. Codeword Interpretability Study

The paper analyzes codeword usage on ten representative S&P500 stocks from different sectors.

Stocks:

- XOM — Energy
- LIN — Materials
- UNP — Industrials
- KO — Consumer Staples / Discretionary wording in paper
- JNJ — Health Care
- JPM — Financials
- MSFT — Information Technology
- META — Communication Services
- NEE — Utilities
- PLD — Real Estate

---

# 82. Temporal Codeword Activation

The paper plots stacked area charts of normalized codeword activation proportions over time together with stock closing prices.

The authors observe:

- some codewords dominate during high-volatility periods;
- others dominate during stable periods;
- usage changes through time.

Interpretation:

> codewords may represent time-varying latent regimes / shocks rather than static factors.

---

# 83. Macro Indicator Correlation Analysis

The paper correlates codeword usage with:

- VIX;
- U.S. Dollar Index (DXY);
- U.S. 10-year Treasury yield (US10Y).

Pearson correlation is used.

The goal is not to train on these variables.

They are used **after training** to interpret learned codewords.

---

# 84. Correlation Ranges

Reported examples / ranges:

## VIX

Strong positive codeword correlations:

\[
0.211
\text{ to }
0.482.
\]

Strong negative correlations:

\[
-0.151
\text{ to }
-0.360.
\]

## DXY

Positive correlations up to about:

\[
0.425.
\]

## US10Y

Positive correlations up to about:

\[
0.410.
\]

---

# 85. Specific Codeword Examples

## Codeword 11

XOM / VIX correlation:

\[
0.482.
\]

LIN / DXY correlation:

\[
-0.287.
\]

The paper interprets this as potentially associated with high-volatility / dollar-weakness conditions.

---

## Codeword 30

US10Y correlations include:

\[
0.410
\]

for UNP,

\[
0.402
\]

for JNJ,

and:

\[
0.322
\]

for KO.

This suggests interest-rate sensitivity.

---

## Codeword 108

The paper notes:

- positive VIX association in multiple stocks;
- negative DXY association.

It suggests codeword 108 may act like a broad:

> "risk-off" latent factor.

---

## Codeword 107

Appears with strong positive US10Y correlations across multiple stocks.

Interpretation:

> interest-rate-sensitive factor.

---

# 86. Authors' Interpretability Caveat

The paper explicitly warns:

> a codeword should not be treated as exactly equal to one macroeconomic variable.

Codewords are described as multifaceted abstract market signals.

The correlations should be interpreted as:

- broad associations;
- market-condition relationships;

not exact economic identities.

The authors propose:

- econometric analysis;
- event studies;

as future work for stronger interpretation.

---

# 87. Market Regime Analysis

The appendix defines bull, bear, and sideways periods.

## CSI300

### Bull

\[
2020\text{-}01\text{-}01
\rightarrow
2021\text{-}02\text{-}10
\]

### Bear

\[
2021\text{-}02\text{-}10
\rightarrow
2022\text{-}04\text{-}26
\]

### Sideways

\[
2022\text{-}04\text{-}26
\rightarrow
2023\text{-}06\text{-}30.
\]

---

## S&P500

### Bull

\[
2020\text{-}04\text{-}01
\rightarrow
2021\text{-}12\text{-}31
\]

### Bear

\[
2022\text{-}01\text{-}01
\rightarrow
2022\text{-}10\text{-}30
\]

### Sideways

\[
2022\text{-}11\text{-}01
\rightarrow
2023\text{-}06\text{-}30.
\]

---

# 88. Regime Performance — CSI300 Top-\(k\)

FactorVQVAE:

| Regime | AR | STD | MDD | Sharpe | Sortino | Calmar |
|---|---:|---:|---:|---:|---:|---:|
| Bull | 0.5319 | 0.2181 | -0.1412 | 2.4385 | 2.2831 | 3.7669 |
| Bear | -0.1034 | 0.2134 | -0.2259 | -0.4844 | -0.4416 | -0.4578 |
| Sideways | 0.3219 | 0.1587 | -0.1214 | 2.0285 | 2.3303 | 2.6518 |

Interpretation:

- not highest raw return in bull market;
- relatively strong downside control in bear period;
- strong risk-adjusted performance in sideways market.

---

# 89. Regime Performance — S&P500 Top-\(k\)

FactorVQVAE:

| Regime | AR | STD | MDD | Sharpe | Sortino | Calmar |
|---|---:|---:|---:|---:|---:|---:|
| Bull | 0.8253 | 0.2683 | -0.1639 | 3.0757 | 3.5304 | 5.0361 |
| Bear | -0.1034 | 0.2555 | -0.2093 | -0.4048 | -0.4122 | -0.4942 |
| Sideways | 0.2051 | 0.2126 | -0.0994 | 0.9645 | 1.1242 | 2.0627 |

Notably, FactorVQVAE is not the best model in every S&P500 regime.

This is a useful reminder:

> aggregate full-period superiority does not imply regime-by-regime dominance.

---

# 90. Regime Performance — CSI300 Long–Short

FactorVQVAE:

| Regime | AR | STD | MDD | Sharpe | Sortino | Calmar |
|---|---:|---:|---:|---:|---:|---:|
| Bull | 0.3367 | 0.0899 | -0.0674 | 3.7446 | 4.7175 | 4.9944 |
| Bear | 0.2516 | 0.0978 | -0.0362 | 2.5735 | 2.7795 | 6.9461 |
| Sideways | 0.2448 | 0.0720 | -0.0543 | 3.4013 | 3.7653 | 4.5109 |

This is especially strong because the long–short strategy remains positive even during the defined bear regime.

---

# 91. Regime Performance — S&P500 Long–Short

FactorVQVAE:

| Regime | AR | STD | MDD | Sharpe | Sortino | Calmar |
|---|---:|---:|---:|---:|---:|---:|
| Bull | 0.2536 | 0.0954 | -0.0545 | 2.6590 | 3.7921 | 4.6562 |
| Bear | 0.0408 | 0.0712 | -0.0671 | 0.5727 | 0.5458 | 0.6075 |
| Sideways | 0.0064 | 0.0723 | -0.0579 | 0.0885 | 0.1038 | 0.1105 |

The S&P500 sideways period is difficult.

FactorVQVAE remains slightly positive, but performance is weak in absolute terms.

---

# 92. Inductive Biases

FactorVQVAE encodes several strong assumptions.

## 92.1 Discrete Market-Factor States Exist

Financial latent structure can be usefully represented by a finite codebook.

## 92.2 Small Latent Perturbations Are Often Noise

If small movements in continuous embedding do not change the underlying regime, snapping to a shared codeword can improve robustness.

## 92.3 Factor States Have Sequential Dependence

\[
p(s_t\mid s_{<t})
\]

is meaningful.

## 92.4 Market Context Conditions Factor Transitions

\[
p(s_t\mid s_{<t},m_{<t})
\]

should outperform a token-only prior.

## 92.5 Ranking Is a Primary Financial Objective

The model explicitly adds pairwise rank supervision because pointwise prediction alone is not enough.

## 92.6 Market Complexity Determines Optimal Codebook Capacity

Different markets require different discrete capacity.

---

# 93. Why FactorVQVAE May Work

Separate evidence from interpretation.

## Evidence 1 — VQ removal causes a large performance drop

Supports discrete representation within this architecture.

## Evidence 2 — Transformer → GRU causes the largest component-ablation drop

Supports powerful temporal modeling over token sequences.

## Evidence 3 — Removing rank loss hurts RankIC

Supports objective alignment.

## Evidence 4 — Market attention gives a small positive gain

Supports conditioning factor transitions on market-level information.

## Evidence 5 — Codeword usage correlates with macro states

Supports some degree of interpretable latent organization.

---

# 94. Plausible Mechanism: Quantization as Denoising

If two nearby continuous states:

\[
z_a
\approx
z_b
\]

map to the same codeword:

\[
q(z_a)=q(z_b),
\]

then small input perturbations do not propagate as continuously varying factor values.

This creates a piecewise-constant latent bottleneck.

For noisy finance data, this can act as a form of representation stabilization.

---

# 95. Plausible Mechanism: Discrete Factor Selection

Continuous latent models can activate all dimensions simultaneously.

A codebook instead forces each latent position to select from a finite vocabulary.

This can create:

- reuse;
- specialization;
- latent-state competition.

However, FactorVQVAE still requires empirical checks on:

- code usage entropy;
- code collapse;
- specialization stability.

---

# 96. Plausible Mechanism: Easier Temporal Modeling

A discrete token sequence is well suited to autoregressive modeling.

Instead of predicting a continuous high-dimensional latent vector:

\[
z_{t+1}\in\mathbb R^d,
\]

the Transformer predicts:

\[
s_{t+1}
\in
\{1,\dots,K\}.
\]

This converts latent-factor dynamics into a classification / language-model-like problem.

---

# 97. Potential Failure Modes

These are analytical concerns, not necessarily failures observed by the authors.

## 97.1 Codebook Overcapacity

Too many codewords can fragment data into unstable rare states.

The S&P500 sensitivity results provide empirical support for this concern.

## 97.2 Codebook Undercapacity

Too few codes can merge distinct regimes and lose predictive structure.

## 97.3 Token Semantics Can Drift

A codeword may not preserve identical economic meaning across long periods.

## 97.4 Discrete Boundary Instability

Although small perturbations inside one Voronoi cell are ignored, points near code boundaries can switch abruptly.

## 97.5 Two-Stage Error Propagation

Stage 2 performance depends on Stage 1 code quality.

A weak codebook cannot be repaired by the Transformer prior.

## 97.6 Greedy Decoding Ignores Predictive Uncertainty

Choosing only the most probable token discards the full token distribution.

## 97.7 Fixed Codebook Across Regimes

The paper itself identifies this as a limitation.

## 97.8 Quadratic Pairwise Rank Loss Cost

The pairwise term:

\[
\sum_i\sum_j
\]

is potentially \(O(N^2)\) in the cross-section.

The paper does not provide a detailed complexity discussion for this term.

---

# 98. Evidence Strength

## Stronger aspects

- two markets;
- five seeds;
- broad baseline set;
- same Qlib environment;
- two portfolio strategies;
- statistical significance marker for main CSI300 comparison;
- VQ ablation;
- component ablation;
- hyperparameter sensitivity;
- subperiod analysis;
- explicit regime analysis;
- codeword interpretability study.

## Weaker / missing aspects

- no complete reporting of seed-wise standard deviations in main tables;
- significance methodology is only briefly described;
- no formal codebook perplexity / utilization statistics;
- no explicit compute-matched baseline analysis;
- no transaction-cost sensitivity table;
- no out-of-sample market transfer without retuning;
- no adaptive codebook experiment despite being a stated limitation;
- codeword interpretability remains correlational.

Overall:

> empirical evidence is substantially stronger than FactorVAE's original paper, especially for robustness and ablation, but causal claims about "posterior collapse mitigation" and economic factor identity remain only partially established.

---

# 99. Baseline Fairness

Positive points:

- common Qlib environment;
- explicit preprocessing;
- baseline implementation sources documented;
- five seeds;
- two markets.

Potential concerns:

## Reimplementation gap

CAE and FactorVAE are reimplemented by the FactorVQVAE authors.

Exact reproduction quality may affect baseline strength.

## Hyperparameter budgets

The paper does not provide a fully symmetric tuning budget for every baseline.

## Model-specific inputs

FactorVQVAE uses 63 explicit market-information features.

MASTER also uses market information conceptually, but not necessarily the identical market feature construction.

Other baselines may not receive this extra information.

Therefore:

> some performance gain may arise from both architecture and additional market-conditioned input.

The market-attention ablation helps quantify part of this effect, and its RankIC contribution is relatively small on CSI300.

---

# 100. Reproducibility Strength

FactorVQVAE is more reproducible from the paper than FactorVAE because it reports:

- optimizer;
- learning rate;
- scheduler;
- epochs;
- early stopping;
- optimizer betas;
- epsilon;
- weight decay;
- gradient clipping;
- hardware;
- approximate training time;
- random seed count;
- data split;
- market features;
- preprocessing;
- hyperparameter sensitivity.

Still missing / not fully explicit:

- exact Stage 1 codebook update implementation details;
- exact shared affine parameterization formula;
- exact synchronized-update implementation;
- batch size;
- exact value of rank-loss \(\lambda\);
- exact weight \(\eta\);
- exact Stage-1 commitment \(\beta\);
- all random seed values;
- exact transaction-cost assumptions for portfolio backtests;
- full baseline tuning grids.

---

# 101. Known Hyperparameters

| Setting | Reported Value / Search |
|---|---|
| Sequence length | 20 |
| Stage 1 model size | 128 |
| Stage 1 latent dimension | 64 |
| Stage 1 epochs | 10 |
| Stage 2 dimension | 32 / 64 sensitivity |
| Stage 2 heads | 2 / 4 in implementation section; sensitivity also includes 8 |
| Stage 2 layers | 1 / 2 in implementation section; sensitivity figure also includes 4 |
| Codebook size | 128 / 256 / 512 sensitivity |
| AdamW LR | 0.0001 |
| Scheduler | cosine annealing |
| AdamW betas Stage 2 | (0.9, 0.98) |
| AdamW epsilon | \(10^{-6}\) |
| Weight decay Stage 2 | \(10^{-3}\) |
| Gradient clip | 3 |
| Stage 2 max epochs | 100 |
| Early stopping | validation loss |
| Seeds | 5 |
| Top-k CSI300 | 30 |
| Top-k S&P500 | 50 |
| Drop-n | 5 |
| Risk-free rate | 0 |

---

# 102. Hyperparameters Not Explicitly Reported

- batch size;
- Stage 1 commitment weight \(\beta\);
- rank-loss pairwise weight \(\lambda\);
- Stage 2 rank-loss weight \(\eta\);
- exact codebook affine transformation details;
- exact early-stopping patience;
- exact cosine schedule parameters;
- exact attention dropout;
- exact Transformer FFN multiplier;
- exact GRU hidden size if distinct from listed model/latent dimension;
- exact transaction fees and slippage;
- exact seeds.

These should be obtained from code if exact reproduction is required.

---

# 103. Complexity Notes

## Stage 1

Approximate runtime:

> 20 minutes.

## Stage 2

Approximate runtime:

> 1 hour.

## Total

> 80 minutes.

Potential computational hotspots:

- Transformer attention;
- cross-sectional rank loss;
- codebook nearest-neighbor search;
- hyperparameter sweeps across codebook / dimension / heads / layers.

The authors explicitly identify computational cost during hyperparameter optimization as a limitation.

---

# 104. Novelty Decomposition

## Existing Components

- dynamic factor model;
- GRU feature encoder;
- VQ-VAE;
- discrete codebook;
- autoregressive Transformer;
- cross-attention;
- pairwise rank loss.

## Finance-Specific Adaptations

- factor decoder with \(\alpha\) and \(\beta\);
- future-return latent factor encoder;
- market-index feature cross-attention;
- cross-sectional ranking objective;
- financial portfolio evaluation;
- codeword-macro interpretation.

## Main Novel Combination

\[
\text{Dynamic Factor Model}
+
\text{VQ-VAE}
+
\text{Autoregressive Token Prior}.
\]

## Main Representational Innovation

Continuous latent factor:

\[
z\in\mathbb R^d
\]

becomes discrete factor token:

\[
s\in\{1,\dots,K\}.
\]

---

# 105. Mechanism Primitives

For a research Agent, decompose FactorVQVAE into reusable primitives.

## Primitive A — Temporal Stock Encoder

\[
x_{i,1:T}
\rightarrow
e_{i,1:T}.
\]

## Primitive B — Future-Return Factor Encoder

\[
y_{1:T}
\rightarrow
z_{1:T}.
\]

## Primitive C — Nearest-Code Quantization

\[
z_t
\rightarrow
s_t
\rightarrow
z_{q,t}.
\]

## Primitive D — Learned Financial Factor Vocabulary

\[
\mathcal C
=
\{z_1,\dots,z_K\}.
\]

## Primitive E — Factor-State Temporal Decoder

\[
z_{q,1:t}
\rightarrow
h_t.
\]

## Primitive F — Dynamic Stock Exposure

\[
e_i
\rightarrow
\beta_i.
\]

## Primitive G — Autoregressive Factor Prior

\[
s_{<t}
\rightarrow
p(s_t).
\]

## Primitive H — Market-Conditioned Prior

\[
(s_{<t},m_{<t})
\rightarrow
p(s_t).
\]

## Primitive I — Ranking-Aware Training

\[
\mathcal L_{\text{token}}
+
\eta
\mathcal L_{\text{rank}}.
\]

---

# 106. What Is Actually Most Important?

Based on the paper's ablations:

## Highest-impact component among Stage 2 components

Transformer temporal model.

## Second

Rank loss.

## Third

Market attention.

## Foundational representation change

Vector quantization itself.

Therefore, an Agent should understand the hierarchy:

\[
\text{VQ representation}
>
\text{token temporal modeling}
>
\text{ranking alignment}
>
\text{market attention increment}
\]

as a rough evidence-based ordering, not an exact universal ranking.

---

# 107. How to Beat This Baseline

## 107.1 What FactorVQVAE Already Does Well

- explicit discrete latent-factor learning;
- strong CSI300 RankIC;
- dual-market evaluation;
- long test period;
- market conditioning;
- rank-aware objective;
- comprehensive portfolio evaluation;
- regime tests;
- strong ablation coverage;
- interpretable codeword analysis.

---

## 107.2 Structural Weaknesses

### Fixed Codebook

\[
K
\]

is static.

### Single Flat Codebook

No hierarchy or factor families.

### Hard Nearest-Neighbor Assignment

No explicit uncertainty over code assignments.

### Greedy Decoding

Ignores multimodal next-factor uncertainty.

### Stage Separation

Stage 2 cannot reshape a poor Stage 1 codebook.

### No Explicit Factor Diversity Objective

Distinct codewords need not correspond to economically distinct factor roles.

### No Cross-Market Shared Structure

CSI300 and S&P500 are trained/tuned separately.

### Static Token Semantics

Code meaning may drift with regime change.

---

# 108. Safe Improvement Directions

Relatively incremental:

- improved market attention;
- better ranking loss;
- stronger temporal encoder;
- sampled / beam token decoding;
- better codebook update rules;
- codebook utilization regularization.

These may improve performance but may not be enough alone for a strong new paper.

---

# 109. Higher-Novelty Directions

## Adaptive Codebook

Let codebook evolve with market regime.

## Hierarchical Codebook

Separate:

- global market code;
- sector code;
- stock-specific code.

## Shared + Specialized Codebooks

One shared factor vocabulary plus regime/market-specific vocabularies.

## Dynamic Sparse Factor Activation

Predict multiple selected codewords instead of one fixed representation.

## Mixture-of-Experts Prior

Route different market states to different token-transition experts.

## Regime-Aware Quantization

Condition code selection on a regime encoder.

## Multi-Scale Discrete Factors

Separate daily / weekly / monthly latent token streams.

## Probabilistic Token Assignment

Represent uncertainty over codewords instead of only nearest-code hard assignment.

## Cross-Market Transfer

Learn shared discrete factors across CSI300 and S&P500 with market-specific adapters.

---

# 110. Experimental Hooks for Innovation

## E1 — Adaptive Codebook

Hypothesis:

> fixed codebooks become stale when market structure shifts.

Measure:

- RankIC by year / half-year;
- code usage entropy;
- code turnover;
- regime performance.

---

## E2 — Shared + Specialized Codewords

Hypothesis:

> some factors are market-wide and persistent, while others are regime-specific.

Architecture:

```text
shared codebook
+
specialized codebook bank
+
router
```

Ablation:

- shared only;
- specialized only;
- shared + routed specialized.

---

## E3 — Factor Transition Experts

Replace one Transformer prior with:

\[
\text{Router}
\rightarrow
\text{multiple transition experts}.
\]

Hypothesis:

> latent factor transition dynamics differ across regimes.

---

## E4 — Multi-Scale Tokenization

Construct:

- short-horizon factor tokens;
- medium-horizon factor tokens;
- long-horizon factor tokens.

Test whether this improves:

- RankIC persistence;
- IC decay;
- regime stability.

---

## E5 — Token Semantics Regularization

Add macro / market-state contrastive supervision.

Goal:

> make factor codes more stable and interpretable.

---

## E6 — Uncertainty-Aware Decoding

Instead of greedy:

\[
\arg\max p(s_t),
\]

use:

- expected return over token distribution;
- top-\(k\) token mixture;
- entropy-aware confidence;
- risk-adjusted token distribution.

This restores some uncertainty information lost by hard decoding.

---

# 111. Agent Critique: Claim vs. Evidence

## Claim A

Vector quantization mitigates posterior collapse.

### Evidence

- t-SNE latent separation;
- large performance drop in continuous ablation.

### Assessment

Strong evidence that VQ helps representation quality in this pipeline.

But posterior-collapse mitigation itself is not directly quantified with standard latent-utilization diagnostics.

---

## Claim B

Discrete factors improve robustness to noise.

### Evidence

- cross-market performance;
- regime analysis;
- sensitivity to codebook size.

### Assessment

Plausible and moderately supported, but there is no direct controlled noise-injection experiment.

---

## Claim C

Market attention improves prediction.

### Evidence

\[
RankIC:
0.0495
\rightarrow
0.0508.
\]

### Assessment

Supported, but effect is relatively small.

---

## Claim D

Transformer is important for factor temporal dynamics.

### Evidence

Replacing Transformer with GRU:

\[
0.0508
\rightarrow
0.0430.
\]

### Assessment

Strong within the tested alternatives.

---

## Claim E

Codewords are interpretable financial factors.

### Evidence

Temporal code usage and correlations with VIX/DXY/US10Y.

### Assessment

Suggestive and useful, but codeword-economic-variable relationships are correlational and non-unique.

The authors acknowledge this limitation.

---

# 112. Explicit Limitations from the Paper

The authors state three main limitations.

## Limitation 1 — Computational Cost

The complex architecture increases training and hyperparameter optimization cost.

### Proposed future direction

More efficient training or approximation.

---

## Limitation 2 — Limited Interpretability

Discrete factors help, but the full network remains difficult to interpret.

### Proposed future direction

- SHAP;
- attention heatmaps.

---

## Limitation 3 — Fixed Codebook Size

A fixed codebook may be insufficient for highly dynamic markets.

### Proposed future direction

- adaptive codebook;
- evolving codebook.

This is one of the clearest research opportunities in the paper.

---

# 113. Additional Explicit Future Work

The paper suggests:

## Multimodal Inputs

- news sentiment;
- macroeconomic indicators.

## Other Financial Tasks

- derivative pricing;
- broader risk prediction.

## More Markets / Assets

- additional emerging markets;
- alternative asset classes.

## Stronger Interpretability

- SHAP-based attributions;
- attention-based explanation;
- deeper codeword analysis.

---

# 114. Agent-Identified Limitations

These are additional analytical observations, not direct author claims.

## Hard tokenization discards uncertainty

Only one codeword is selected.

## Model selection is market-specific

Best codebook and Transformer settings differ sharply across markets.

## Extra market inputs complicate architecture-only attribution

FactorVQVAE has explicit 63-dimensional market information.

## Stage 1 future-return representation details are complex

Exact temporal indexing should be verified in code.

## Long–Short S&P500 performance remains regime-sensitive

Especially the sideways regime.

## Main-table seed variation is not fully displayed

The paper says five seeds are used, but table cells show means only.

---

# 115. Writing / Presentation Lessons

## 115.1 Strong Problem-to-Mechanism Mapping

The introduction says:

- continuous latent representations are noisy/redundant;
- latent temporal structure is weak;
- factor selection is insufficient.

The method then directly introduces:

- quantization;
- autoregressive Transformer;
- discrete codebook.

This is good research narrative design.

---

## 115.2 Two-Stage Story Is Easy to Explain

Stage 1:

> learn the latent vocabulary.

Stage 2:

> learn the grammar / dynamics of that vocabulary.

This analogy is powerful and intuitive.

---

## 115.3 Experiments Are Organized Around Explicit RQs

RQ1–RQ7 cover:

- prediction;
- portfolio value;
- market regimes;
- VQ utility;
- VQ ablation;
- component contribution;
- hyperparameter sensitivity.

This creates a much stronger experimental narrative than a single benchmark table.

---

## 115.4 Interpretability Is Added After Performance

The paper first establishes:

- predictive gains;
- portfolio gains;
- robustness.

Then investigates latent meaning.

This ordering prevents interpretability analysis from carrying the main efficacy claim.

---

# 116. Important Writing Pattern for Future Papers

A reusable structure is:

```text
Problem:
continuous latent factors are unstable / redundant.

Mechanism:
discretize latent space.

Second problem:
factor dynamics evolve through time.

Mechanism:
autoregressive token model.

Finance-specific problem:
market state affects transitions.

Mechanism:
market cross-attention.

Evaluation mismatch:
stock selection depends on ranking.

Mechanism:
rank loss.

Evidence:
main results → portfolios → regimes → VQ ablation →
component ablation → sensitivity → interpretability.
```

This is a strong "closed-loop" paper structure.

---

# 117. Relationship to FactorVAE in Detail

| Dimension | FactorVAE | FactorVQVAE |
|---|---|---|
| Latent space | Continuous Gaussian | Discrete codebook |
| Core latent regularization | KL prior-posterior matching | VQ codebook / commitment |
| Training style | Joint prior-posterior | Two-stage |
| Future-return role | posterior factor teacher | Stage-1 factor encoder |
| Prior prediction | multi-head global attention | autoregressive Transformer |
| Temporal latent dynamics | limited / implicit | explicit token sequence |
| Market information | derived from stock cross-section | explicit 63 market features + attention |
| Ranking loss | no | yes |
| Uncertainty output | yes | not a central probabilistic output |
| Interpretability | limited | codeword usage / macro correlation |
| Markets | China A-shares | CSI300 + S&P500 |
| Test horizon | 2019–2020 | 2020–2023H1 |

---

# 118. Key Conceptual Trade-Off vs. FactorVAE

FactorVAE preserves a continuous probabilistic distribution:

\[
\mu,\sigma.
\]

This naturally provides uncertainty.

FactorVQVAE gains:

- discrete structure;
- stable token states;
- autoregressive token modeling;

but loses a direct continuous predictive uncertainty representation.

A strong future model could try to combine both:

> discrete regime/factor identity + probabilistic uncertainty inside each discrete state.

---

# 119. Implementation Hooks

A clean implementation can be organized as:

```text
factorvqvae/
├── feature_extractor.py
├── factor_encoder.py
├── quantizer.py
├── codebook.py
├── factor_decoder.py
├── latent_gru.py
├── market_feature_extractor.py
├── ar_transformer.py
├── market_attention.py
├── rank_loss.py
├── stage1_trainer.py
├── stage2_trainer.py
└── model.py
```

---

# 120. Minimal Module Interfaces

## Feature Extractor

```text
Input:
x: [N, T, C]

Output:
e: [N, T, H]
```

## Factor Encoder

```text
Input:
future returns y / embedded future-return sequence

Output:
continuous latent factors z
```

## Quantizer

```text
Input:
z: [..., H]

Output:
z_q: [..., H]
token_ids: [...]
```

## Factor Decoder

```text
Input:
e
z_q sequence

Output:
predicted returns y_hat
alpha
beta
latent hidden state h_t
```

## AR Transformer

```text
Input:
previous token ids
market features m

Output:
next-token logits
```

---

# 121. Stage 1 Training Pseudocode

```text
for batch / cross-sectional sequence:

    e = feature_extractor(x)

    z = factor_encoder(future_returns)

    token_ids, z_q = quantizer(z)

    h = latent_factor_gru(z_q)

    alpha = alpha_layer(e)
    beta  = beta_layer(e)

    y_hat = alpha + beta * h

    loss_rec = return_reconstruction_nll(y_hat, y)

    loss_codebook =
        codebook_alignment(z, z_q)
        + commitment(z, z_q)

    loss = loss_rec + loss_codebook

    alternating / synchronized VQ update
```

Exact update ordering should follow the paper's Huh et al.-based codebook procedure.

---

# 122. Stage 2 Training Pseudocode

```text
freeze(
    feature_extractor,
    factor_encoder,
    quantizer,
    codebook,
    factor_decoder
)

for token_sequence, market_sequence:

    logits = AR_transformer(
        previous_tokens,
        market_features
    )

    loss_ar = cross_entropy(
        logits,
        next_token
    )

    predicted_token = teacher-forced / decoded token
    predicted_factor = codebook[predicted_token]

    predicted_returns = frozen_decoder(
        predicted_factor,
        stock_features
    )

    loss_rank =
        pointwise_return_loss
        + lambda * pairwise_rank_loss

    loss =
        loss_ar
        + eta * loss_rank

    update(transformer_only)
```

---

# 123. Prediction Pseudocode

```text
historical_tokens = encode_known_history(...)
market_context = build_market_features(...)

next_logits = transformer(
    historical_tokens,
    market_context
)

next_token = argmax(next_logits)

next_factor = codebook[next_token]

predicted_returns =
    factor_decoder(
        next_factor,
        current_stock_features
    )

rank stocks by predicted_returns
```

---

# 124. Reproduction Checklist

A faithful reproduction should confirm:

## Data

- CSI300 / S&P500 constituent definition;
- Alpha158;
- 63 market features;
- missing-value fill;
- T=20;
- exact label.

## Split

- 2009-01-01 to 2019-06-30 train;
- 2019H2 validation;
- 2020–2023H1 test.

## Preprocessing

- Qlib default;
- CSRankNorm.

## Stage 1

- GRU feature extractor;
- future-return attention encoder;
- VQ codebook;
- synchronized / affine VQ update;
- factor decoder;
- 10 epochs.

## Stage 2

- frozen Stage 1;
- AR Transformer;
- market attention;
- rank loss;
- AdamW;
- LR \(10^{-4}\);
- cosine anneal;
- 100 max epochs;
- validation early stopping.

## Evaluation

- five seeds;
- RankIC / RankICIR;
- Top-\(k\) Drop-\(n\);
- Long–Short;
- AR / STD / MDD / Sharpe / Sortino / Calmar.

---

# 125. Local Reproduction

This section should be filled with local experiment results.

## Paper-Reported — CSI300

\[
RankIC=0.0508
\]

\[
RankICIR=0.3775.
\]

## Paper-Reported — S&P500

\[
RankIC=0.0122
\]

\[
RankICIR=0.0982.
\]

## Our Reproduction

### CSI300

- data period:
- feature set:
- market information:
- seeds:
- IC:
- ICIR:
- RankIC:
- RankICIR:
- AR:
- Sharpe:
- MDD:

### S&P500

- data period:
- feature set:
- market information:
- seeds:
- IC:
- ICIR:
- RankIC:
- RankICIR:
- AR:
- Sharpe:
- MDD:

## Differences From Paper

- TBD.

## Possible Causes

- dataset period;
- constituent handling;
- market features;
- Qlib preprocessing;
- codebook implementation;
- baseline implementation;
- random seeds;
- transaction costs;
- hyperparameter search.

---

# 126. Baseline Comparison Rules

When comparing a new model with FactorVQVAE, verify:

1. same market;
2. same date split;
3. same stock universe;
4. same Alpha158 preprocessing;
5. same market information;
6. same label horizon;
7. same RankIC computation;
8. same number of seeds;
9. same model-selection criterion;
10. same portfolio strategy and fees.

A model using different market features or a different 2020–2023 test universe should not be compared only by headline RankIC.

---

# 127. Agent Research Instructions

When using FactorVQVAE:

1. **Treat vector quantization as the core representational innovation.**
2. **Treat the autoregressive Transformer as the strongest Stage-2 component according to ablation.**
3. **Preserve the two-stage freeze/train protocol unless intentionally testing an alternative.**
4. **Do not claim codewords are exact economic factors without additional evidence.**
5. **Do not infer universal codebook size; CSI300 and S&P500 prefer very different capacities.**
6. **Record market-information inputs when comparing against baselines.**
7. **Check the paper's small numerical inconsistencies before using values as ground truth.**
8. **Use the main tables, not prose alone, when rankings conflict.**
9. **Do not treat t-SNE alone as proof of posterior collapse mitigation.**
10. **If proposing a new model, target the fixed-codebook, static-token-semantics, hard-assignment, or single-prior limitations rather than only swapping GRU/Transformer backbones.**
11. **Use regime and subperiod diagnostics, not only full-period RankIC.**
12. **For a stronger paper, evaluate code utilization, token stability, transition entropy, and factor semantics quantitatively.**

---

# 128. Agent Takeaways

## Most Important Research Problem

How to learn robust, non-redundant, temporally structured latent financial factors from noisy and non-stationary cross-sectional stock data.

## Most Important Representation Change

\[
\text{continuous factor}
\rightarrow
\text{discrete codeword}.
\]

## Most Important Temporal Mechanism

\[
p(s_t\mid s_{<t},m_{<t})
\]

with an autoregressive Transformer.

## Most Important Finance-Specific Objective

\[
\mathcal L_{\text{AR}}
+
\eta
\mathcal L_{\text{rank}}.
\]

## Most Important Ablation

Removing vector quantization:

\[
RankIC:
0.0505
\rightarrow
0.0257.
\]

## Most Important Component Ablation

Transformer \(\rightarrow\) GRU:

\[
0.0508
\rightarrow
0.0430.
\]

## Most Important Cross-Market Insight

CSI300 benefits from a large codebook:

\[
512,
\]

while S&P500 prefers:

\[
128.
\]

The optimal latent capacity is market-specific.

## Biggest Structural Limitation

Fixed codebook with static capacity and hard token assignment.

## Most Promising Research Direction

Adaptive / routed / hierarchical discrete factor vocabularies that change with market regime.

---

# 129. Compact Architecture Summary

```text
                         FACTORVQVAE
============================================================

STAGE 1 — DISCRETE FACTOR LEARNING

Historical stock features x
        │
        ▼
GRU Feature Extractor
        │
        ▼
Stock feature sequence e
        │
        ├──────────────────────────────┐
        │                              │
Future return sequence y              │
        │                              │
        ▼                              │
Causal Multi-Head Factor Encoder      │
        │                              │
        ▼                              │
Continuous factor embedding z         │
        │                              │
        ▼                              │
Vector Quantizer / Codebook           │
        │                              │
        ▼                              │
Discrete factor z_q / token id s      │
        │                              │
        ▼                              │
Latent-factor GRU                     │
        │                              │
        ▼                              │
h_t                                   │
        │                              │
        └──────────────┬───────────────┘
                       │
             e → Alpha / Beta
                       │
                       ▼
               y_hat = α + β h_t

Loss:
return reconstruction
+
codebook / commitment loss


STAGE 2 — FACTOR TOKEN PRIOR

Previous factor tokens s_<t
        │
        ▼
Causal Self Attention
        │
Market features m ──► Market Attention
        │
        ▼
Feed Forward
        │
        ▼
Next-token distribution
        │
        ▼
Greedy token
        │
        ▼
Frozen codebook
        │
        ▼
Predicted discrete factor
        │
        ▼
Frozen factor decoder
        │
        ▼
Cross-sectional return forecast

Loss:
AR token cross-entropy
+
η × rank loss
```

---

# 130. Source Location Map

Useful paper locations:

- **pp. 1–2:** motivation, contributions, discrete-factor rationale, related work.
- **pp. 3–4:** model overview, VQ-VAE preliminaries, autoregressive Transformer preliminaries, problem formulation.
- **pp. 5–6:** Stage 1 encoder, quantizer, decoder, codebook loss, Stage 2 market attention and objectives.
- **pp. 7–8:** prediction protocol, datasets, splits, baselines, metrics, implementation details.
- **pp. 9–12:** main RankIC results, Top-\(k\) and Long–Short portfolio results, subperiod robustness.
- **pp. 13–15:** VQ ablation, component ablations, sensitivity analysis.
- **pp. 16–17:** codeword usage and macroeconomic interpretability.
- **p. 18:** conclusion, explicit limitations, future work.
- **pp. 19–21:** bull/bear/sideways regime appendix.

---

# 131. Compact Retrieval Summary

FactorVQVAE (Kim, Ock, Song, KBS 2025) replaces the continuous latent factor space of VAE-style financial factor models with a vector-quantized codebook and learns factor dynamics through a two-stage pipeline. Stage 1 uses a GRU stock-feature extractor, a causal multi-head factor encoder over realized return sequences, a VQ codebook, and a factor-model decoder with alpha/beta structure to learn discrete latent factors. Stage 2 freezes the Stage-1 components and trains an autoregressive Transformer to predict the next factor token from previous tokens while conditioning on 63 market-index features through market attention; training also includes a pairwise rank loss. The model uses Qlib Alpha158 with \(T=20\), trains on 2009–2019H1, validates on 2019H2, and tests on 2020–2023H1 for CSI300 and S&P500. It reports RankIC/RankICIR of 0.0508/0.3775 on CSI300 and 0.0122/0.0982 on S&P500. In CSI300, FactorVQVAE strongly outperforms baselines in both prediction and long–short portfolio metrics; in S&P500, gains are smaller and XGB/CAE/GRU win some portfolio metrics. Removing vector quantization causes a large CSI300 drop, and component ablations show the autoregressive Transformer and rank loss are especially important. Hyperparameter analysis finds a large 512-code codebook works best for CSI300 while a 128-code codebook works best for S&P500, indicating market-specific latent capacity. Codeword usage correlates with VIX, DXY, and US10Y, providing suggestive interpretability. The paper's key limitations are computational tuning cost, incomplete interpretability, and a fixed codebook; explicit future directions include adaptive/evolving codebooks, multimodal data, more markets/assets, and stronger explainability.
