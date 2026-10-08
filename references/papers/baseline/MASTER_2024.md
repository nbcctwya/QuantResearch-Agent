---
paper_id: MASTER_2024
title: "MASTER: Market-Guided Stock Transformer for Stock Price Forecasting"
authors:
  - Tong Li
  - Zhaoyang Liu
  - Yanyan Shen
  - Xue Wang
  - Haokun Chen
  - Sen Huang
venue: "AAAI-24 / The Thirty-Eighth AAAI Conference on Artificial Intelligence"
year: 2024
code: "https://github.com/SJTU-Quant/MASTER"
paper_type: baseline
subtype:
  - finance_ml
  - cross_sectional_stock_prediction
  - stock_correlation_modeling
  - market_guided_transformer
task:
  - normalized_future_return_prediction
  - cross_sectional_stock_ranking
  - portfolio_selection
markets:
  - CSI300
  - CSI800
dataset:
  stock_features: "Qlib Alpha158"
  market_features: "63 index-derived market-status features"
  lookback_tau: 8
  prediction_interval_d: 5
  train: "2008-Q1 to 2020-Q1"
  validation: "2020-Q2"
  test: "2020-Q3 to 2022-Q4"
core_modules:
  - market_status_representation
  - market_guided_gating
  - intra_stock_transformer_aggregation
  - inter_stock_attention_aggregation
  - temporal_attention_aggregation
  - linear_prediction_head
core_mechanisms:
  - market_conditioned_feature_selection
  - alternating_intra_inter_attention
  - momentary_stock_correlation
  - cross_time_stock_correlation
  - attention_field_factorization
main_metrics:
  - IC
  - ICIR
  - RankIC
  - RankICIR
  - Excess_Annualized_Return
  - Information_Ratio
main_results:
  CSI300_IC: "0.064 ± 0.006"
  CSI300_RankIC: "0.076 ± 0.005"
  CSI300_AR: "0.27 ± 0.05"
  CSI300_IR: "2.4 ± 0.4"
  CSI800_IC: "0.052 ± 0.006"
  CSI800_RankIC: "0.066 ± 0.007"
  CSI800_AR: "0.28 ± 0.02"
  CSI800_IR: "2.3 ± 0.3"
relevance_to_agent:
  baseline_understanding: very_high
  architecture_transfer: very_high
  market_information_modeling: very_high
  cross_time_correlation: very_high
  stock_relation_modeling: very_high
  reproducibility: high
priority: very_high
---

# MASTER: Market-Guided Stock Transformer for Stock Price Forecasting

## 0. Document Purpose

This document is a high-density, Agent-oriented reconstruction of **MASTER (Li et al., AAAI 2024)**.

MASTER is one of the most important baselines for modern Qlib-style cross-sectional stock prediction because it combines two ideas that recur in many later models:

1. **market-aware feature selection**;
2. **momentary and cross-time stock-correlation modeling**.

The paper's key architectural insight is not simply "use a Transformer."

Its central design is:

> Avoid collapsing each stock's full historical sequence into one vector before modeling stock relationships.  
> Instead, preserve time-specific stock embeddings, alternate intra-stock and inter-stock aggregation, and let information travel from any source-stock/time pair \((v,j)\) to any target-stock/time pair \((u,i)\) through a factorized two-step attention path.

This creates an efficient approximation to the conceptually expensive full attention field over:

\[
\tau \times |S|
\]

stock-time tokens.

The second major idea is:

> Market conditions change which input factors are useful.  
> Use a market-status vector to generate feature-wise gates, dynamically emphasizing or suppressing stock features before correlation modeling.

For a research Agent, MASTER should be understood as:

- a baseline to reproduce exactly;
- a reference architecture for factorized stock-time attention;
- a market-information conditioning mechanism;
- the direct intellectual predecessor of several later stock Transformer models, including MATCC-style work.

---

# 1. Metadata

- **Title:** MASTER: Market-Guided Stock Transformer for Stock Price Forecasting
- **Authors:** Tong Li, Zhaoyang Liu, Yanyan Shen, Xue Wang, Haokun Chen, Sen Huang
- **Affiliations:** Shanghai Jiao Tong University; Alibaba Group
- **Venue:** AAAI-24
- **Year:** 2024
- **Proceedings pages:** 162–170
- **Code:** https://github.com/SJTU-Quant/MASTER
- **Task:** cross-sectional stock return forecasting
- **Markets:** CSI300 and CSI800
- **Data platform:** Qlib
- **Primary inputs:** Alpha158 stock features + market-index-derived status features
- **Primary modeling target:** normalized future return ratio

---

# 2. Research Problem

## 2.1 Standard Stock-Correlation Pipeline

The paper argues that many earlier methods follow:

\[
\text{stock time series}
\rightarrow
\text{one representation per stock}
\rightarrow
\text{cross-stock correlation}
\rightarrow
\text{prediction}.
\]

This is problematic because temporal details are compressed before stock relationships are learned.

---

## 2.2 Problem 1 — Momentary and Cross-Time Correlations

The authors argue real stock correlations are often:

- momentary;
- sparse;
- asymmetric;
- cross-time rather than time-aligned.

A relationship may exist between:

\[
(stock_v,\ time_j)
\]

and:

\[
(stock_u,\ time_i),
\]

with:

\[
i\neq j.
\]

Example intuition:

> upstream and downstream firms can react to the same shock with different delays.

Therefore:

\[
\text{same-time correlation only}
\]

is insufficient.

---

## 2.3 Problem 2 — Feature Effectiveness Changes with Market Status

The usefulness of technical / price-volume features is not stationary.

The paper argues that feature effectiveness changes with:

- bull vs. bear conditions;
- market activity;
- volatility;
- broader environment.

Therefore a fixed feature representation may overuse stale / ineffective features.

---

# 3. Why Not Full Stock-Time Attention?

Suppose:

- \(|S|=M\) stocks;
- lookback length \(\tau\).

A direct solution is to build all:

\[
M\tau
\]

stock-time tokens and attend among all of them.

This allows arbitrary:

\[
(v,j)\rightarrow(u,i)
\]

relations.

But pairwise attention would scale as:

\[
O(M^2\tau^2).
\]

The authors argue this is problematic for finance because there are relatively few trading days compared with the enormous attention field.

Thus full stock-time attention is:

- computationally expensive;
- data hungry;
- difficult to optimize.

---

# 4. MASTER's Core Solution

Instead of one huge attention field, MASTER factorizes the information flow.

## Step 1 — Intra-stock aggregation

Within each stock:

\[
(v,j)
\rightarrow
(v,i).
\]

Information from time \(j\) is transported into time-specific local embedding \(h_{v,i}\).

## Step 2 — Inter-stock aggregation

At target time \(i\):

\[
(v,i)
\rightarrow
(u,i).
\]

Information from stock \(v\)'s temporally enriched state reaches stock \(u\).

Thus the two-hop path:

\[
(v,j)
\rightarrow
h_{v,i}
\rightarrow
z_{u,i}
\]

allows MASTER to model a cross-time cross-stock relation without direct full attention between all stock-time pairs.

---

# 5. Main Contributions

The paper states three contributions.

## Contribution 1 — Specialized Stock Transformer

MASTER alternates:

- intra-stock temporal aggregation;
- inter-stock aggregation;

to model momentary and cross-time stock correlation.

---

## Contribution 2 — Market-Guided Feature Gating

Market status generates feature-wise scaling coefficients.

The gating mechanism performs automatic feature selection that changes with the market environment.

---

## Contribution 3 — Empirical and Visual Validation

The model is tested on CSI300 / CSI800 using:

- IC;
- RankIC;
- ICIR;
- RankICIR;
- portfolio AR / IR;
- architecture ablations;
- hyperparameter sensitivity;
- correlation visualizations.

---

# 6. Formal Task Definition

For stock:

\[
u\in S
\]

and historical time:

\[
t\in[1,\tau],
\]

stock feature vector:

\[
x_{u,t}\in\mathbb R^F.
\]

The raw future return target is:

\[
\tilde r_u
=
\frac{
c_{u,\tau+d}-c_{u,\tau+1}
}{
c_{u,\tau+1}
},
\]

where:

- \(c_{u,t}\): closing price;
- \(d\): prediction interval.

The paper then applies daily cross-sectional normalization:

\[
r_u
=
Norm_S(\tilde r_u).
\]

The task is:

\[
\{x_{u,t}\}
\rightarrow
\{r_u\}_{u\in S}.
\]

---

# 7. Important Label Interpretation

MASTER does not optimize raw unnormalized return directly.

The label is cross-sectionally normalized each day.

This means the model is trained to predict a **relative cross-sectional return score**.

For comparison with another implementation, verify:

- exact return horizon;
- exact \(T+d\) / \(T+1\) indexing;
- exact normalization operator.

---

# 8. Overall Architecture

MASTER has five steps:

1. Market-Guided Gating
2. Intra-Stock Aggregation
3. Inter-Stock Aggregation
4. Temporal Aggregation
5. Prediction

Conceptually:

```text
Market Status m_tau
      │
      ▼
Feature Gate α(m_tau)
      │
      ▼
Stock Sequence x[u,t]
      │
      ▼
Gated Stock Features
      │
      ▼
Feature Projection
      │
      ▼
Intra-Stock Transformer
      │
      ▼
Local embeddings h[u,t]
      │
      ▼
Inter-Stock Attention at each t
      │
      ▼
Temporal embeddings z[u,t]
      │
      ▼
Temporal Attention
      │
      ▼
Stock embedding e[u]
      │
      ▼
Linear Prediction
```

---

# 9. Figure 2: Information-Flow Interpretation

The paper's main architecture figure is important because it explicitly illustrates a cross-time path.

For source:

\[
(v,j)
\]

and target:

\[
(u,i),
\]

information travels:

\[
y_{v,j}
\rightarrow
h_{v,i}
\rightarrow
z_{u,i}.
\]

This is MASTER's central mechanism.

---

# 10. Market Status Representation

MASTER constructs:

\[
m_\tau
\]

to represent the latest market environment.

The paper uses two classes of market information.

## 10.1 Market Index Price

Includes:

- current index price;
- historical average;
- historical standard deviation.

Historical windows:

\[
d'.
\]

---

## 10.2 Market Index Volume

Includes historical:

- average volume;
- standard deviation of volume.

The intended meaning is:

- price statistics → market movement / volatility;
- volume statistics → investor activity / market scale.

---

# 11. Market Indices

For the Chinese market, the paper constructs market features from:

- CSI300;
- CSI500;
- CSI800.

Total market-status feature dimension:

\[
F'=63.
\]

Reference windows:

\[
d'
\in
\{5,10,20,30,60\}.
\]

---

# 12. Market-Guided Gating

Market status is transformed into one scaling coefficient per stock feature.

The paper writes:

\[
\alpha(m_\tau)
=
F\cdot
softmax_\beta
(
W_\alpha m_\tau+b_\alpha
).
\]

Here:

- \(F\): stock-feature dimension;
- \(\beta\): temperature;
- \(W_\alpha,b_\alpha\): trainable parameters.

---

# 13. Why Multiply by \(F\)?

Softmax outputs sum to:

\[
1.
\]

The uniform value per dimension is:

\[
1/F.
\]

Multiplying by \(F\) makes a uniform gate equal:

\[
1.
\]

Therefore:

- coefficient \(>1\): feature emphasized;
- coefficient \(<1\): feature suppressed.

This makes the gate easy to interpret as a scaling mechanism.

---

# 14. Temperature Parameter

The temperature controls feature-selection sharpness.

## Smaller \(\beta\)

More concentrated distribution.

Stronger feature selection.

## Larger \(\beta\)

More uniform distribution.

Weaker feature selection.

Thus \(\beta\) controls how aggressively market status selects features.

---

# 15. Market-Conditioned Feature Selection

The gate is shared across:

- all stocks;
- all time steps;

for the same current market-status vector.

The gated feature is:

\[
\tilde x_{u,t}
=
\alpha(m_\tau)
\circ
x_{u,t}.
\]

This encodes the assumption:

> market condition determines which feature dimensions are globally effective now.

---

# 16. Important Inductive Bias of the Gate

The gate is:

- market-dependent;
- feature-specific;
- stock-independent.

Thus the paper assumes:

> feature relevance changes with the market, but the same market-wide relevance weighting can be applied to all stocks in the cross-section.

A future model could relax this assumption.

---

# 17. Feature Projection

The gated feature is projected into model dimension \(D\):

\[
y_{u,t}
=
f(\tilde x_{u,t}).
\]

The paper uses a single linear layer for \(f\).

Then fixed sinusoidal positional encoding:

\[
p_t
\]

is added.

---

# 18. Intra-Stock Aggregation

MASTER preserves the entire stock time sequence instead of immediately collapsing it.

For stock \(u\):

\[
Y_u
=
\|_{t\in[1,\tau]}
LN(f(\tilde x_{u,t})+p_t).
\]

The paper then applies one Transformer encoder layer.

---

# 19. Intra-Stock Attention

For stock \(u\):

\[
Q_u^1
=
W_Q^1Y_u,
\]

\[
K_u^1
=
W_K^1Y_u,
\]

\[
V_u^1
=
W_V^1Y_u.
\]

Then:

\[
H_u^1
=
\|_{t\in[1,\tau]}
h_{u,t}
=
FFN_1
(
MHA_1(Q_u^1,K_u^1,V_u^1)
+
Y_u
).
\]

The multi-head attention uses:

\[
N_1
\]

heads.

---

# 20. Role of the Intra-Stock Block

Each:

\[
h_{u,t}
\]

retains local time-\(t\) detail while absorbing information from all time steps in stock \(u\)'s lookback window.

Thus the sequence:

\[
h_{u,1},\dots,h_{u,\tau}
\]

acts as a set of **time-specific relays**.

These relays carry cross-time information into the later cross-stock attention block.

---

# 21. Bidirectionality of Intra-Stock Transformer

The paper describes the sequential encoder as bi-directional.

Therefore, within the historical lookback window:

\[
h_{u,t}
\]

can aggregate information from both earlier and later positions inside the already observed historical window.

This does not automatically constitute prediction-time leakage because all \(\tau\) lookback positions are historical relative to the forecast date.

However:

> it means MASTER's internal time aggregation is not a causal decoder-style Transformer.

This is relevant when comparing with RWKV / autoregressive variants.

---

# 22. Inter-Stock Aggregation

For each historical time index \(t\), MASTER collects local embeddings from all stocks:

\[
H_t^2
=
\|_{u\in S}
h_{u,t}.
\]

Then:

\[
Q_t^2
=
W_Q^2H_t^2,
\]

\[
K_t^2
=
W_K^2H_t^2,
\]

\[
V_t^2
=
W_V^2H_t^2.
\]

The stock-attention output is:

\[
Z_t
=
\|_{u\in S}
z_{u,t}
=
FFN_2
(
MHA_2(Q_t^2,K_t^2,V_t^2)
+
H_t^2
).
\]

The number of heads is:

\[
N_2.
\]

---

# 23. Meaning of \(z_{u,t}\)

The temporal embedding:

\[
z_{u,t}
\]

contains:

1. stock \(u\)'s own temporally enriched information;
2. information from stocks that are correlated with \(u\) at time \(t\).

The residual connection preserves the stock's personal information.

---

# 24. How MASTER Models Cross-Time Cross-Stock Correlation

This is the most important architectural derivation.

Suppose we want information flow:

\[
(v,j)
\rightarrow
(u,i).
\]

## Intra-stock step

Stock \(v\)'s time-\(j\) feature contributes to:

\[
h_{v,i}
\]

through temporal self-attention.

## Inter-stock step

At time \(i\):

\[
h_{v,i}
\]

contributes to:

\[
z_{u,i}
\]

through stock self-attention.

Therefore:

\[
(v,j)
\rightarrow
h_{v,i}
\rightarrow
z_{u,i}.
\]

This factorization enables arbitrary conceptual cross-time correlation without direct global stock-time attention.

---

# 25. Important Nuance: Cross-Time Correlation Is Factorized

MASTER does **not** directly compute a full attention matrix between all:

\[
(stock,time)
\]

pairs.

Instead it factorizes the relation.

This gives:

- lower complexity;
- stronger inductive structure;
- reduced attention-field size.

But it also constrains how relations are represented.

A direct stock-time graph could potentially capture structures MASTER's two-hop factorization cannot.

---

# 26. Temporal Aggregation

After inter-stock attention, every stock still has a sequence:

\[
z_{u,1},\dots,z_{u,\tau}.
\]

MASTER uses the latest embedding:

\[
z_{u,\tau}
\]

as query.

Attention score:

\[
\lambda_{u,t}
=
\frac{
\exp
(
z_{u,t}^TW_\lambda z_{u,\tau}
)
}{
\sum_{i\in[1,\tau]}
\exp
(
z_{u,i}^TW_\lambda z_{u,\tau}
)
}.
\]

Final stock embedding:

\[
e_u
=
\sum_{t\in[1,\tau]}
\lambda_{u,t}z_{u,t}.
\]

---

# 27. Interpretation of Temporal Aggregation

The most recent temporal embedding represents the current stock state.

Historical embeddings are weighted according to relevance to this current state.

Therefore the final embedding is a:

> current-state-conditioned historical aggregation.

---

# 28. Prediction Head

The final stock embedding is fed into:

\[
\hat r_u
=
g(e_u).
\]

The predictor \(g\) is one linear layer.

Loss:

\[
\mathcal L
=
\sum_{u\in S}
MSE(r_u,\hat r_u).
\]

---

# 29. Batch Structure

Each training batch corresponds to:

> one prediction date containing all stocks in the current universe.

This is important because the inter-stock attention operates jointly across the cross-section.

An epoch consists of multiple date-level batches.

---

# 30. Why the Final Prediction Head Is Simple

MASTER places almost all modeling capacity in:

- market gating;
- intra-stock attention;
- inter-stock attention;
- temporal aggregation.

The output head is deliberately simple.

This supports the paper's claim that performance comes from representation / correlation modeling, not from a complicated task head.

---

# 31. Computational Complexity

Let:

\[
M=|S|.
\]

The paper decomposes complexity into:

- market gating:
  \[
  O(FM\tau)
  \]
- intra-stock attention:
  \[
  O(N_1M\tau^2D^2)
  \]
- inter-stock attention:
  \[
  O(N_2M^2\tau D^2)
  \]
- temporal aggregation:
  \[
  O(M\tau D^2).
  \]

Overall:

\[
O
\left(
FM\tau
+
N_1M\tau^2D^2
+
N_2M^2\tau D^2
+
M\tau D^2
\right).
\]

Because:

\[
M\gg\tau,
\]

the dominant term is:

\[
O(N_2M^2\tau D^2).
\]

---

# 32. Comparison with Full Stock-Time Attention

Direct attention over:

\[
M\tau
\]

tokens costs approximately:

\[
O(NM^2\tau^2D^2).
\]

MASTER reduces one factor of \(\tau\), achieving roughly:

\[
\tau
\]

times lower attention complexity.

This is one of the key architectural justifications.

---

# 33. Parameter Structure

The paper identifies trainable transformation matrices including:

\[
W_Q^1,W_K^1,W_V^1,
\]

\[
W_Q^2,W_K^2,W_V^2,
\]

\[
W_\lambda.
\]

These are:

\[
D\times D.
\]

Additional parameters appear in:

- gating network \(\alpha\);
- feature projection \(f\);
- FFN1;
- FFN2;
- prediction head \(g\).

---

# 34. Datasets

MASTER uses:

- CSI300;
- CSI800.

Daily data range:

\[
2008\rightarrow2022.
\]

---

# 35. Train / Validation / Test Split

## Training

Q1 2008 to Q1 2020.

Equivalent date interpretation:

\[
2008\text{-}01\text{-}01
\rightarrow
2020\text{-}03\text{-}31.
\]

## Validation

Q2 2020:

\[
2020\text{-}04\text{-}01
\rightarrow
2020\text{-}06\text{-}30.
\]

## Test

Q3 2020 to Q4 2022:

\[
2020\text{-}07\text{-}01
\rightarrow
2022\text{-}12\text{-}31.
\]

---

# 36. Stock Features

MASTER uses:

\[
Alpha158
\]

Qlib indicators.

Stock feature dimension:

\[
F=158
\]

under the standard Alpha158 setup.

---

# 37. Lookback and Horizon

Lookback length:

\[
\tau=8.
\]

Prediction interval:

\[
d=5.
\]

This differs from FactorVAE / FactorVQVAE protocols, which use \(T=20\) and different date splits.

---

# 38. Market Features

Market-status features are constructed from:

- CSI300;
- CSI500;
- CSI800.

Total:

\[
63
\]

features.

Reference windows:

\[
d'
=
5,10,20,30,60.
\]

---

# 39. Data Preprocessing Explicitly Stated in the Main Paper

The main paper explicitly specifies:

- Alpha158 features;
- future return ratio;
- daily cross-sectional Z-score label normalization.

It does **not** fully specify every stock-feature preprocessing step in the main text.

Exact Qlib processing should be taken from the released code / supplementary materials.

Do not silently assume a preprocessing operator such as CSRankNorm unless verified from code.

---

# 40. Baselines

The paper compares MASTER with:

## XGBoost

Tree-based baseline.

The paper notes it is one of Qlib's strongest classical baselines.

## LSTM

Sequential recurrent baseline.

## GRU

Sequential recurrent baseline.

## TCN

Temporal convolution baseline.

## Transformer

Vanilla sequential Transformer baseline.

## GAT

Graph-attention stock-correlation baseline.

## DTML

Dynamic stock-correlation model using attention and market information.

DTML is the structurally closest baseline.

---

# 41. Baseline Implementation

## DTML

Reimplemented from the original paper because no public official implementation is available.

## Other baselines

Use Qlib implementations.

---

# 42. Baseline Hyperparameter Tuning

For every baseline:

## Layer count

\[
\{1,2,3\}.
\]

## Model size

\[
\{128,256,512\}.
\]

## Learning rate

\[
10^{-3},10^{-4},10^{-5},10^{-6}.
\]

Best hyperparameters are selected by:

\[
validation\ IC.
\]

This is a relatively explicit and fair tuning protocol.

---

# 43. MASTER Hyperparameters

MASTER uses the same search range for:

- model size \(D\);
- learning rate.

Selected for both datasets:

\[
D=256,
\]

\[
lr=10^{-5}.
\]

Heads:

\[
N_1=4,
\]

\[
N_2=2.
\]

---

# 44. Gating Temperature

The paper reports:

## CSI300

\[
\beta=5.
\]

## CSI800

\[
\beta=2.
\]

Interpretation:

- CSI300 requires weaker feature selection;
- CSI800 benefits from stronger feature selection.

---

# 45. Training Protocol

Each model is trained:

> at most 40 epochs.

Early stopping is used.

The exact:

- patience;
- optimizer;
- scheduler;
- batch-size implementation;

are not fully described in the main paper excerpt.

These should be retrieved from code / supplementary materials for exact reproduction.

---

# 46. Hardware

Experiments use:

- Intel Xeon Platinum 8163 CPU;
- 128 GB memory;
- Tesla V100-SXM2 GPU;
- 16 GB VRAM.

---

# 47. Random Seeds

Every experiment is repeated:

\[
5
\]

times with random initialization.

Average performance is reported.

Table 1 also reports standard deviations.

---

# 48. Evaluation Metrics

Four prediction metrics:

- IC;
- ICIR;
- RankIC;
- RankICIR.

Two portfolio metrics:

- Excess Annualized Return (AR);
- Information Ratio (IR).

---

# 49. IC

IC is daily Pearson correlation:

\[
IC_t
=
corr_{\text{Pearson}}
(
\hat r_t,r_t
).
\]

The reported IC is averaged across dates.

---

# 50. RankIC

RankIC is daily Spearman correlation:

\[
RankIC_t
=
corr_{\text{Spearman}}
(
\hat r_t,r_t
).
\]

Then averaged over time.

---

# 51. ICIR / RankICIR

The paper describes:

\[
ICIR
=
\frac{
mean(IC_t)
}{
std(IC_t)
},
\]

and analogously:

\[
RankICIR
=
\frac{
mean(RankIC_t)
}{
std(RankIC_t)
}.
\]

The paper describes these as normalized performance metrics.

---

# 52. Portfolio Strategy

The paper simulates daily trading by selecting the:

\[
30
\]

stocks with highest predicted return ratio.

It reports:

- Excess AR;
- IR.

The main paper does not explicitly state transaction-cost handling in the provided text.

Therefore:

> do not assume either zero cost or nonzero cost without checking code / supplementary materials.

This is different from MATCC, which explicitly states its portfolio metrics are without cost.

---

# 53. Overall Results — CSI300

| Model | IC | ICIR | RankIC | RankICIR | AR | IR |
|---|---:|---:|---:|---:|---:|---:|
| XGBoost | 0.051 ± 0.001 | 0.37 ± 0.01 | 0.050 ± 0.001 | 0.36 ± 0.01 | 0.23 ± 0.03 | 1.9 ± 0.3 |
| LSTM | 0.049 ± 0.001 | 0.41 ± 0.01 | 0.051 ± 0.002 | 0.41 ± 0.03 | 0.20 ± 0.04 | 2.0 ± 0.4 |
| GRU | 0.052 ± 0.004 | 0.35 ± 0.04 | 0.052 ± 0.005 | 0.34 ± 0.04 | 0.19 ± 0.04 | 1.5 ± 0.3 |
| TCN | 0.050 ± 0.002 | 0.33 ± 0.04 | 0.049 ± 0.002 | 0.31 ± 0.04 | 0.18 ± 0.05 | 1.4 ± 0.5 |
| Transformer | 0.047 ± 0.007 | 0.39 ± 0.04 | 0.051 ± 0.002 | 0.42 ± 0.04 | 0.22 ± 0.06 | 2.0 ± 0.4 |
| GAT | 0.054 ± 0.002 | 0.36 ± 0.02 | 0.041 ± 0.002 | 0.25 ± 0.02 | 0.19 ± 0.03 | 1.3 ± 0.3 |
| DTML | 0.049 ± 0.006 | 0.33 ± 0.04 | 0.052 ± 0.005 | 0.33 ± 0.04 | 0.21 ± 0.03 | 1.7 ± 0.3 |
| **MASTER** | **0.064 ± 0.006** | **0.42 ± 0.04** | **0.076 ± 0.005** | **0.49 ± 0.04** | **0.27 ± 0.05** | **2.4 ± 0.4** |

Asterisks in the paper indicate statistically significant improvement for:

- CSI300 IC;
- CSI300 RankIC;

over all baselines using a t-test with:

\[
p<0.01.
\]

---

# 54. Overall Results — CSI800

| Model | IC | ICIR | RankIC | RankICIR | AR | IR |
|---|---:|---:|---:|---:|---:|---:|
| XGBoost | 0.040 ± 0.000 | 0.37 ± 0.01 | 0.047 ± 0.000 | 0.42 ± 0.01 | 0.08 ± 0.02 | 0.6 ± 0.2 |
| LSTM | 0.028 ± 0.002 | 0.32 ± 0.02 | 0.039 ± 0.002 | 0.41 ± 0.03 | 0.09 ± 0.02 | 0.9 ± 0.2 |
| GRU | 0.039 ± 0.002 | 0.36 ± 0.05 | 0.044 ± 0.003 | 0.39 ± 0.07 | 0.07 ± 0.04 | 0.6 ± 0.3 |
| TCN | 0.038 ± 0.002 | 0.33 ± 0.04 | 0.045 ± 0.002 | 0.38 ± 0.05 | 0.05 ± 0.04 | 0.4 ± 0.3 |
| Transformer | 0.040 ± 0.003 | **0.43 ± 0.03** | 0.048 ± 0.003 | **0.51 ± 0.05** | 0.13 ± 0.04 | 1.1 ± 0.3 |
| GAT | 0.043 ± 0.002 | 0.39 ± 0.02 | 0.042 ± 0.002 | 0.35 ± 0.02 | 0.10 ± 0.04 | 0.7 ± 0.3 |
| DTML | 0.039 ± 0.004 | 0.29 ± 0.03 | 0.053 ± 0.008 | 0.37 ± 0.06 | 0.16 ± 0.03 | 1.3 ± 0.2 |
| **MASTER** | **0.052 ± 0.006** | 0.40 ± 0.06 | **0.066 ± 0.007** | 0.48 ± 0.06 | **0.28 ± 0.02** | **2.3 ± 0.3** |

Important nuance:

MASTER is **not best on all CSI800 ranking stability metrics**.

Transformer has:

\[
ICIR=0.43
\]

vs MASTER:

\[
0.40,
\]

and:

\[
RankICIR=0.51
\]

vs MASTER:

\[
0.48.
\]

This explains the paper's wording that MASTER is best on:

\[
6/8
\]

ranking metrics across the two datasets.

---

# 55. Main Empirical Claim

The paper states that MASTER improves over second-best results by approximately:

- **13%** on ranking metrics;
- **47%** on portfolio metrics;

on average.

The exact percentage depends on metric and dataset.

A research Agent should use table values rather than a single aggregate improvement claim.

---

# 56. Cross-Universe Pattern

All models perform better on CSI300 than CSI800.

The authors attribute this to CSI300 containing larger-cap companies whose prices are more predictable.

The gating-temperature results are consistent with this interpretation:

- CSI300:
  \[
  \beta=5
  \]
  → weaker gating;
- CSI800:
  \[
  \beta=2
  \]
  → stronger gating.

---

# 57. Architecture Study — Table 2

The paper evaluates four architecture variants on CSI300.

| Model | IC | ICIR | RankIC | RankICIR | AR | IR |
|---|---:|---:|---:|---:|---:|---:|
| (MA)STER | 0.064 ± 0.003 | 0.43 ± 0.02 | 0.074 ± 0.004 | 0.48 ± 0.04 | 0.25 ± 0.03 | 2.1 ± 0.3 |
| (MA)STER-Bi | 0.058 ± 0.005 | 0.38 ± 0.04 | 0.066 ± 0.008 | 0.41 ± 0.05 | 0.19 ± 0.03 | 1.6 ± 0.2 |
| Naive | 0.041 ± 0.008 | 0.30 ± 0.05 | 0.046 ± 0.007 | 0.32 ± 0.04 | 0.18 ± 0.05 | 1.6 ± 0.6 |
| Clustering | 0.044 ± 0.003 | 0.36 ± 0.02 | 0.049 ± 0.005 | 0.39 ± 0.04 | 0.18 ± 0.04 | 1.7 ± 0.3 |

---

# 58. What is (MA)STER?

The notation:

> (MA)STER

refers to MASTER **without the market-guided gating**.

Thus it isolates the stock-transformer architecture.

It still contains:

- intra-stock aggregation;
- inter-stock aggregation;
- temporal aggregation.

---

# 59. (MA)STER-Bi

This replaces the intra-stock Transformer with a bidirectional LSTM.

Purpose:

> show the architecture is not dependent on an extremely strong Transformer encoder.

The performance remains decent but lower than the Transformer-based stock architecture.

---

# 60. Naive Variant

The Naive baseline directly attends over:

\[
\tau\times|S|
\]

tokens.

This is conceptually the full stock-time attention field.

Despite its expressiveness, it performs badly.

The authors interpret this as:

- too large an attention field;
- insufficient financial training data;
- difficult optimization.

This is one of the most important architecture results.

---

# 61. Clustering Variant

This uses Local Sensitive Hashing to divide all stock-time tokens into 10 buckets before attention.

Purpose:

> reduce the global attention field using a generic approximation.

It outperforms Naive slightly but remains well below MASTER.

The authors argue domain-specific factorization is superior to task-agnostic attention compression.

---

# 62. Architectural Lesson from Table 2

The evidence suggests:

\[
\text{structured factorization}
>
\text{full global attention}
\]

for this low-data financial setting.

This is a strong inductive-bias result.

---

# 63. Gating Contribution

Compare:

## Full MASTER

CSI300:

\[
IC=0.064\pm0.006,
\]

\[
RankIC=0.076\pm0.005,
\]

\[
AR=0.27\pm0.05.
\]

## (MA)STER without gating

\[
IC=0.064\pm0.003,
\]

\[
RankIC=0.074\pm0.004,
\]

\[
AR=0.25\pm0.03.
\]

The gating contribution on CSI300 mean prediction is relatively modest.

Its value is more pronounced in other settings, especially CSI800 according to the temperature analysis and the paper's motivation.

---

# 64. Head-Number Sensitivity

The paper varies:

\[
(N_1,N_2)
\]

where:

- \(N_1\): intra-stock attention heads;
- \(N_2\): inter-stock attention heads.

Tested values include:

\[
1,2,4.
\]

The paper reports no dramatic difference across combinations relative to seed variance.

This suggests MASTER is not highly sensitive to exact head counts in the studied range.

---

# 65. Selected Head Configuration

Final choice:

\[
N_1=4,
\]

\[
N_2=2.
\]

This is used for both datasets.

---

# 66. Gating Temperature Sensitivity

The paper studies:

\[
\beta
\in
\{1,2,5,10,20\}.
\]

Interpretation:

- smaller \(\beta\): stronger selection;
- larger \(\beta\): weaker selection.

---

# 67. CSI300 Temperature Pattern

CSI300 is interpreted as a relatively easy / stable universe.

More features remain effective.

Therefore optimal gating is weaker.

Selected:

\[
\beta=5.
\]

---

# 68. CSI800 Temperature Pattern

CSI800 is more heterogeneous.

The paper argues stronger feature-selection intervention is more useful.

Selected:

\[
\beta=2.
\]

---

# 69. Attention Visualization — Momentary Correlation

The paper chooses three target stocks on August 19, 2022:

- CNPC (SH601857)
- ICBC (SH601398)
- CATL (SZ300750)

and samples 100 random source stocks.

The inter-stock attention map is:

- sparse;
- scattered;
- not organized as long horizontal/vertical strips.

The authors interpret this as evidence that stock correlations are:

> momentary rather than persistent across the entire lookback window.

---

# 70. Sparse Correlation

Only a few source stocks have high attention toward each target.

This supports the idea:

> economically relevant stock correlations are sparse and dynamic.

MASTER itself still uses dense attention computationally, but learned weights are sparse in effect.

---

# 71. Cross-Time Pair Correlation

The paper reconstructs a \(\tau\times\tau\) correlation map between source stock \(v\) and target stock \(u\):

\[
I_{u\leftarrow v}[i,j]
=
S_v^1[i,j]
S_i^2[u,v].
\]

Here:

- \(S^1\): intra-stock attention map;
- \(S^2\): inter-stock attention map.

Interpretation:

- \(S_v^1[i,j]\): how source stock \(v\)'s time \(j\) influences its relay at time \(i\);
- \(S_i^2[u,v]\): how stock \(v\)'s relay at time \(i\) influences stock \(u\) at time \(i\).

Their product reconstructs the effective cross-time influence.

---

# 72. Cross-Time Attention Visualization Findings

The paper highlights three observations.

## Observation 1 — Off-diagonal emphasis

Attention blocks are not centered on the diagonal.

Therefore:

\[
i\neq j
\]

relations are important.

This supports cross-time correlation.

---

## Observation 2 — Asymmetry

The maps for:

\[
u\leftarrow v
\]

and:

\[
v\leftarrow u
\]

are different.

Thus stock correlation is asymmetric.

---

## Observation 3 — Slow evolution

When the lookback window shifts to another prediction date, some absolute-time patterns remain similar.

Thus correlation structure is dynamic but not completely unstable.

---

# 73. What the Visualization Does Not Prove

Attention maps are model-internal explanations.

They do not prove causal economic relations.

Stronger validation could include:

- lead–lag correlation tests;
- industry / supply-chain relationships;
- event studies;
- cross-seed stability;
- Granger-style validation.

The paper uses visualization as qualitative mechanism evidence.

---

# 74. Research Questions

The paper organizes experiments around four RQs.

## RQ1

Does MASTER outperform SOTA?

## RQ2

Is the specialized stock-transformer architecture effective?

## RQ3

How do hyperparameters affect performance?

## RQ4

What stock-correlation insights can attention visualizations reveal?

This creates a clear experiment-method alignment.

---

# 75. Inductive Biases

MASTER encodes several important assumptions.

## 75.1 Market-Conditioned Feature Relevance

Feature importance changes with current market environment.

## 75.2 Global Feature Gate

The same market-driven feature gate is useful for all stocks.

## 75.3 Stock-Local Temporal Continuity

Within-stock temporal patterns are simpler and should be modeled before cross-stock interactions.

## 75.4 Momentary Cross-Stock Dependence

Stock relationships should be modeled at each historical time step rather than once for the whole lookback window.

## 75.5 Cross-Time Relations Can Be Factorized

Arbitrary:

\[
(v,j)\rightarrow(u,i)
\]

influence can be represented through:

\[
(v,j)\rightarrow(v,i)\rightarrow(u,i).
\]

---

# 76. Why MASTER May Work

## Evidence A — Large gain over DTML

DTML also models dynamic correlations and uses market information.

MASTER still wins across all six CSI300 metrics.

This supports the architecture rather than simply "using market information."

---

## Evidence B — (MA)STER beats Naive and Clustering

This supports factorized attention design.

---

## Evidence C — (MA)STER-Bi remains strong

This suggests the high-level intra→inter architecture matters independently of the exact temporal encoder.

---

## Evidence D — attention visualizations show off-diagonal, asymmetric patterns

Supports the intended cross-time relation structure qualitatively.

---

# 77. Potential Failure Modes

These are analytical concerns beyond direct author claims.

## 77.1 Dense Inter-Stock Attention

Dominant complexity:

\[
O(M^2\tau D^2).
\]

This can become expensive for larger universes.

## 77.2 One Global Market Gate

Different stocks may respond differently to market conditions.

A single gate may be too coarse.

## 77.3 Market Status Is Hand-Engineered

Market information depends on:

- selected indices;
- price/volume summaries;
- fixed reference windows.

## 77.4 Full Historical Bidirectional Attention

Intra-stock Transformer uses all lookback positions symmetrically.

This is valid for historical context but may not encode directional lead–lag structure explicitly.

## 77.5 Cross-Time Relation Is Restricted to Two-Hop Factorization

A relation that cannot be well represented as:

\[
(v,j)\rightarrow(v,i)\rightarrow(u,i)
\]

may be difficult to capture.

## 77.6 MSE Objective

Training loss is not directly aligned with:

- RankIC;
- ranking;
- portfolio performance.

---

# 78. Evidence Strength

## Strong points

- official code released;
- two stock universes;
- five seeds;
- strong baseline tuning;
- significance tests;
- architecture variants;
- attention-field alternatives;
- hyperparameter sensitivity;
- visual interpretation.

## Missing / weaker evidence

- only Chinese markets;
- no explicit realistic-cost backtest description in the main paper text;
- no direct lead–lag numerical benchmark;
- no runtime comparison to all baselines;
- no stock-specific gating ablation;
- no rank-loss objective comparison;
- no out-of-distribution market test;
- no detailed regime/subperiod analysis.

Overall:

> strong baseline paper with convincing architecture evidence, but narrower robustness coverage than later works such as FactorVQVAE / MATCC.

---

# 79. Reproducibility Assessment

MASTER is relatively reproducible because it provides:

- official code;
- dataset split;
- Alpha158;
- market feature construction sources;
- lookback/horizon;
- baseline tuning grid;
- MASTER tuning grid;
- head counts;
- temperature;
- early stopping;
- hardware;
- five seeds.

Still missing from the main paper:

- optimizer;
- exact batch construction code;
- exact Qlib preprocessing;
- exact normalization implementation;
- exact early-stopping patience;
- dropout;
- FFN expansion ratio;
- exact portfolio-cost assumptions.

Use the repository / supplementary material for exact reproduction.

---

# 80. Known Hyperparameters

| Setting | Value |
|---|---|
| Lookback \(\tau\) | 8 |
| Prediction interval \(d\) | 5 |
| Model dimension \(D\) | 256 |
| Learning rate | \(10^{-5}\) |
| Intra-stock heads \(N_1\) | 4 |
| Inter-stock heads \(N_2\) | 2 |
| CSI300 gate temperature | 5 |
| CSI800 gate temperature | 2 |
| Max epochs | 40 |
| Model selection | early stopping |
| Seeds | 5 |
| Market features | 63 |
| Market reference windows | 5, 10, 20, 30, 60 |
| Portfolio selection | top 30 |

---

# 81. Hyperparameters Not Fully Reported in Main Paper

- optimizer;
- optimizer betas;
- weight decay;
- dropout;
- batch size;
- early-stopping patience;
- FFN hidden size;
- exact positional encoding implementation details beyond standard sinusoid;
- exact stock-feature normalization;
- exact transaction-cost setup;
- exact random seed values.

---

# 82. Novelty Decomposition

## Existing Components

- Transformer encoder;
- multi-head attention;
- temporal attention;
- feature gating;
- market-index features.

## Main Novel Architecture

Alternating:

\[
\text{intra-stock attention}
\rightarrow
\text{inter-stock attention}.
\]

This factorizes the stock-time attention field.

## Main Finance-Specific Mechanism

Market-status-conditioned feature gate.

## Main Interpretability Mechanism

Reconstruct effective:

\[
\tau\times\tau
\]

cross-time pair correlation from intra- and inter-stock attention maps.

---

# 83. Mechanism Primitives

For a research Agent, MASTER can be decomposed into reusable primitives.

## Primitive A — Market Status Vector

\[
\text{index price + volume statistics}
\rightarrow
m_\tau.
\]

## Primitive B — Market-Conditioned Feature Gate

\[
m_\tau
\rightarrow
\alpha(m_\tau)
\rightarrow
\tilde x.
\]

## Primitive C — Time-Specific Relay Embeddings

\[
x_{u,1:\tau}
\rightarrow
h_{u,1:\tau}.
\]

## Primitive D — Momentary Cross-Stock Attention

\[
\{h_{u,t}\}_{u\in S}
\rightarrow
\{z_{u,t}\}_{u\in S}.
\]

## Primitive E — Two-Hop Cross-Time Correlation

\[
(v,j)
\rightarrow
(v,i)
\rightarrow
(u,i).
\]

## Primitive F — Latest-State Query Aggregation

\[
z_{u,\tau}
\rightarrow
attention(z_{u,1:\tau}).
\]

---

# 84. Most Important Component According to Evidence

MASTER does not contain a simple "remove each module" table like MATCC.

But architecture experiments imply the most important factor is:

> the structured intra-stock → inter-stock factorization.

Evidence:

- (MA)STER strongly beats Naive;
- (MA)STER strongly beats Clustering;
- (MA)STER-Bi remains reasonably strong.

The market gate provides additional benefit but is not the sole source of performance.

---

# 85. Relationship to MATCC

MATCC is a direct conceptual successor in several respects.

| Dimension | MASTER | MATCC |
|---|---|---|
| Market information | 63 features | 63 features |
| Market usage | feature gating | explicit market trend addition |
| Time model | Transformer | RWKV |
| Stock trend decomposition | no | yes |
| Intra-stock first | yes | yes |
| Inter-stock second | yes | yes |
| Cross-time correlation | two-hop attention | temporally enriched same-time attention |
| Temporal aggregation | latest-state query | latest-state query |
| Objective | MSE | MSE |
| Lookback | 8 | 8 |
| Prediction horizon | 5 | 5 |

MATCC preserves much of MASTER's overall problem framing but changes:

- market conditioning;
- temporal backbone;
- trend decomposition.

---

# 86. Important Difference: MASTER vs. MATCC Cross-Time Story

MASTER explicitly derives:

\[
I_{u\leftarrow v}[i,j]
=
S_v^1[i,j]
S_i^2[u,v],
\]

which gives an explicit effective cross-time pair correlation map.

MATCC uses RWKV to embed time history before same-time cross-stock attention.

Thus MASTER's cross-time relation is easier to reconstruct explicitly from attention matrices.

This is an important structural distinction.

---

# 87. Relationship to FactorVAE / FactorVQVAE

MASTER is not a latent factor model.

It predicts returns directly from stock and market representations.

| Dimension | FactorVAE | FactorVQVAE | MASTER |
|---|---|---|---|
| Latent factor variable | yes | yes | no |
| Cross-stock representation | latent factor aggregation | discrete factors | direct stock attention |
| Market conditioning | implicit/global | explicit market features | explicit market gate |
| Temporal backbone | GRU | GRU + AR Transformer | Transformer |
| Main innovation | posterior guidance | VQ tokens | factorized stock-time attention |
| Uncertainty | yes | limited | no |
| Ranking loss | no | yes | no |

---

# 88. How to Beat This Baseline

## 88.1 What MASTER Already Does Well

- explicit market condition input;
- adaptive feature selection;
- time-specific stock states;
- dynamic stock relationships;
- efficient cross-time correlation factorization;
- strong Qlib compatibility;
- good multi-seed results;
- official implementation.

---

# 89. Structural Weaknesses

## Global Gate

One market gate for all stocks.

## Static Market Feature Engineering

Market context is manually defined from selected index statistics.

## Dense Inter-Stock Attention

Quadratic in stock count.

## Single Temporal Scale

One lookback window:

\[
\tau=8.
\]

## MSE Objective

No direct ranking alignment.

## No Explicit Regime Specialization

Market variation only changes feature scales.

## Two-Hop Correlation Restriction

Cross-time influence follows a fixed route structure.

---

# 90. Safe Improvement Directions

- stock-specific market gates;
- ranking loss;
- sparse stock attention;
- stronger market encoder;
- longer / multi-scale lookback;
- alternative temporal backbone;
- industry-aware attention prior.

These are useful but may be incremental alone.

---

# 91. Higher-Novelty Directions

## Regime-Routed Market Gating

\[
m
\rightarrow
regime
\rightarrow
different\ feature\ gates.
\]

## Multi-Scale Relay Attention

Build:

- short-horizon relay;
- medium-horizon relay;
- long-horizon relay.

## Explicit Directed Lead–Lag Graph

Learn:

\[
A_{u,v,\Delta t}.
\]

## Sparse Relation Routing

Route each stock to only a few relevant stocks.

## Shared + Specialized Stock Experts

Use common and regime-specific relation experts.

## Market-Conditioned Relation Network

Let market status change not only feature weights but also:

\[
inter\text{-}stock\ attention.
\]

This is a particularly natural extension because MASTER's market information currently only acts before the attention blocks.

---

# 92. Research Opportunity: Market Information Should Affect Relations Too

MASTER assumes market state changes feature effectiveness.

But market state also plausibly changes:

- which stocks are correlated;
- correlation strength;
- lead–lag delay.

A stronger formulation could condition:

\[
S_t^2
\]

directly on market state:

\[
S_t^2
=
Attention(H_t,m_t).
\]

This extends market guidance from:

\[
\text{feature selection}
\]

to:

\[
\text{relationship selection}.
\]

---

# 93. Research Opportunity: Dynamic Stock-Specific Gating

Current MASTER:

\[
\alpha(m_\tau)
\]

is shared by all stocks.

Possible extension:

\[
\alpha_u
=
g(m_\tau,x_u).
\]

Hypothesis:

> the same market regime should affect different stocks differently.

---

# 94. Research Opportunity: Ranking-Aligned MASTER

MASTER trains:

\[
MSE.
\]

But evaluates:

\[
IC,\ RankIC,\ AR,\ IR.
\]

A natural extension is:

\[
\mathcal L
=
\mathcal L_{\text{MSE}}
+
\lambda
\mathcal L_{\text{rank}}.
\]

This directly addresses objective mismatch.

---

# 95. Research Opportunity: Sparse Cross-Time Relations

The attention visualization suggests strong relations are sparse.

Yet MASTER uses dense inter-stock attention.

Therefore:

> the empirical finding suggests a computational improvement opportunity.

Possible mechanism:

- top-k attention;
- differentiable sparse routing;
- learned graph pruning;
- expert-based relation selection.

---

# 96. Research Opportunity: Uncertainty / Reliability

MASTER outputs one deterministic score.

A future model could estimate:

- predictive mean;
- confidence;
- relation uncertainty.

This could help reject low-confidence relation edges or adapt portfolio exposure.

---

# 97. Experiment Hooks

## E1 — Market-Conditioned Relation Attention

Compare:

- feature gate only;
- relation conditioning only;
- both.

Metrics:

- IC;
- RankIC;
- regime stability.

---

## E2 — Sparse Relation Selection

Compare dense inter-stock attention with:

- top-k;
- learned sparse graph;
- routing network.

Measure:

- predictive performance;
- runtime;
- relation stability.

---

## E3 — Direct Cross-Time Attention

Compare MASTER's two-hop factorization against:

\[
(u,t)\leftrightarrow(v,t').
\]

Use a sparse direct mechanism to avoid full cost.

---

## E4 — Multi-Scale Time

Use:

\[
\tau
\in
\{8,20,60\}
\]

or parallel temporal scales.

Test whether cross-time correlation persists at different horizons.

---

## E5 — Rank Loss

Add pairwise / listwise ranking objective.

Check whether gains appear in:

- RankIC;
- portfolio AR / IR;

without harming IC.

---

# 98. Agent Critique: Claim vs. Evidence

## Claim A

MASTER captures momentary and cross-time stock correlation.

### Evidence

- architecture factorization;
- visual off-diagonal attention maps;
- strong performance vs. DTML.

### Assessment

Well supported at the representational / predictive level.

Not causally validated economically.

---

## Claim B

Market-guided gating improves adaptation to market changes.

### Evidence

- full MASTER vs. (MA)STER;
- temperature sensitivity;
- stronger gating in CSI800.

### Assessment

Supported, though the CSI300 mean gain is modest.

---

## Claim C

Structured factorization is better than global attention.

### Evidence

(MA)STER:

\[
IC=0.064
\]

vs Naive:

\[
0.041.
\]

### Assessment

Strong within this financial-data regime.

---

## Claim D

Cross-time stock correlation is important.

### Evidence

Cross-time maps are off-diagonal and model significantly outperforms time-aligned correlation baselines.

### Assessment

Good qualitative + indirect empirical support, but no explicit "remove cross-time only" ablation.

---

# 99. Strengths

## Method

- elegant factorization of a difficult 2D stock-time attention problem;
- clear finance-specific motivation;
- simple market gate;
- modular design.

## Experiments

- strong baselines;
- five seeds;
- statistical significance;
- architecture comparisons;
- interpretable attention visualizations.

## Engineering

- official code;
- manageable complexity relative to full stock-time attention;
- Qlib-compatible.

## Writing

- the paper clearly explains why naive full attention is undesirable;
- every architectural component addresses a stated limitation.

---

# 100. Limitations

## Explicit / Directly Supported

The conclusion states future work should:

- mine higher-quality stock correlations;
- explore other uses of market information.

---

## Agent-Identified Limitations

### Only Chinese stock universes

No U.S. market in the paper.

### Market gate only acts at feature input

It does not directly alter relation attention.

### No explicit relation sparsification

Dense \(M^2\) stock attention remains expensive.

### No uncertainty model

Predictions and correlations are deterministic.

### Objective mismatch

MSE vs ranking-based evaluation.

### Limited temporal horizon

\[
\tau=8.
\]

### Limited robustness analysis

No market-regime decomposition or noise injection in the original MASTER paper.

---

# 101. Future Work

## Explicit Author Future Work

1. mine higher-quality stock correlations;
2. explore other uses of market information.

These are the two future directions explicitly stated in the conclusion.

---

## Implied Future Directions

- market-conditioned relation attention;
- sparse correlation learning;
- dynamic stock-specific gating;
- ranking-aware objectives;
- multi-market generalization;
- multi-scale temporal modeling.

These are inferred research opportunities, not direct author statements.

---

# 102. Writing / Presentation Lessons

## 102.1 Start from Architectural Failure Mode

The introduction does not say only:

> existing models are inaccurate.

It identifies a concrete design flaw:

> collapse time first, correlate stocks second.

That makes the new architecture easy to justify.

---

## 102.2 Complexity Is Part of the Motivation

The paper anticipates the obvious solution:

> just attend over every stock-time token.

Then explains why it is computationally and statistically unattractive.

This strengthens the case for factorization.

---

## 102.3 Use an Intuitive Real-World Example

The upstream/downstream reaction-delay example makes cross-time stock correlation easy to understand.

---

## 102.4 Each Main Claim Has a Matching Experiment

- architecture → Naive / Clustering / Bi variants;
- gating → temperature analysis;
- correlation → visual maps;
- overall performance → baseline table.

This creates a strong research loop.

---

# 103. Important Academic-Writing Pattern

A reusable structure from MASTER:

```text
Observation:
real stock correlations are momentary and cross-time.

Naive solution:
full stock-time attention.

Problem with naive solution:
too expensive + too data hungry.

Structured solution:
alternate intra-stock and inter-stock aggregation.

Second observation:
feature effectiveness changes with market condition.

Solution:
market-conditioned gating.

Experiments:
main benchmark → architecture variants → sensitivity → visualization.
```

This is an excellent architecture-paper narrative.

---

# 104. Local Reproduction

This section should be populated with local experimental results.

## Paper-Reported — CSI300

\[
IC=0.064\pm0.006
\]

\[
ICIR=0.42\pm0.04
\]

\[
RankIC=0.076\pm0.005
\]

\[
RankICIR=0.49\pm0.04
\]

\[
AR=0.27\pm0.05
\]

\[
IR=2.4\pm0.4.
\]

## Paper-Reported — CSI800

\[
IC=0.052\pm0.006
\]

\[
ICIR=0.40\pm0.06
\]

\[
RankIC=0.066\pm0.007
\]

\[
RankICIR=0.48\pm0.06
\]

\[
AR=0.28\pm0.02
\]

\[
IR=2.3\pm0.3.
\]

---

## Our Reproduction — CSI300

- dataset:
- train:
- valid:
- test:
- feature set:
- market features:
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

MASTER original paper does not report S&P500.

If a local S&P500 reproduction exists, it should be clearly labeled as:

> local extension, not paper-reported result.

---

## Differences From Paper

- TBD.

## Possible Causes

- date range;
- Qlib version;
- stock-universe construction;
- preprocessing;
- market-feature implementation;
- label normalization;
- hyperparameters;
- backtest strategy;
- transaction costs.

---

# 105. Baseline Comparison Rules

Before comparing a new model to MASTER, verify:

1. same market;
2. same date split;
3. same Alpha158 definition;
4. same market features;
5. same \(\tau=8\);
6. same \(d=5\);
7. same label normalization;
8. same seed count;
9. same validation-based tuning;
10. same portfolio strategy.

MASTER numbers from its paper are not directly comparable to later Qlib studies with:

- different test period;
- \(T=20\);
- unnormalized labels;
- different backtests.

---

# 106. Implementation Hooks

Suggested implementation structure:

```text
master/
├── market_status.py
├── market_gate.py
├── feature_encoder.py
├── intra_stock_transformer.py
├── inter_stock_attention.py
├── temporal_aggregation.py
├── predictor.py
└── model.py
```

---

# 107. Minimal Interfaces

## Market Status

```text
Input:
index prices / volume statistics

Output:
m_tau: [F_market]
```

## Gate

```text
Input:
m_tau: [F_market]

Output:
alpha: [F_stock]
```

## Feature Encoder

```text
Input:
x: [M, tau, F_stock]

Output:
Y: [M, tau, D]
```

## Intra-Stock

```text
Input:
Y: [M, tau, D]

Output:
H: [M, tau, D]
```

## Inter-Stock

```text
for each time i:
    Input H[:, i, :] : [M, D]
    Output Z[:, i, :] : [M, D]
```

## Temporal Aggregation

```text
Input:
Z: [M, tau, D]

Output:
E: [M, D]
```

## Predictor

```text
Input:
E: [M, D]

Output:
score: [M]
```

---

# 108. Conceptual Forward Pass

```text
market_status = build_market_status(market_data)

gate = F * softmax(
    linear(market_status) / temperature
)

x_gated = stock_features * gate

Y = linear(x_gated) + positional_encoding

for each stock:
    H[stock] = intra_stock_transformer(Y[stock])

for each historical time:
    Z[:, time] = inter_stock_attention(H[:, time])

E = temporal_attention(
    query = Z[:, -1],
    keys = Z,
    values = Z
)

prediction = linear(E)
```

---

# 109. Agent Research Instructions

When using MASTER:

1. **Do not reduce MASTER to "a Transformer baseline."**
2. **Treat intra-stock → inter-stock factorization as the central architectural contribution.**
3. **Preserve time-specific embeddings until after stock correlation.**
4. **Remember the market gate is global across stocks but feature-specific.**
5. **Do not claim MASTER directly attends over arbitrary stock-time pairs; cross-time influence is factorized through a two-hop path.**
6. **Match label normalization and horizon before comparing IC/RankIC.**
7. **Use the official repository for exact preprocessing and optimizer details.**
8. **When extending MASTER, consider market-conditioned relation modeling, not only market-conditioned feature gating.**
9. **Distinguish paper-reported CSI300/CSI800 results from local S&P500 extensions.**
10. **If comparing against MATCC, separate trend-decomposition gains from correlation-architecture gains.**
11. **If designing a stronger model, report regime / subperiod diagnostics because MASTER itself does not.**
12. **Use the attention visualization as qualitative evidence, not causal proof.**

---

# 110. Agent Takeaways

## Most Important Research Problem

How to efficiently model realistic stock relations that are:

- momentary;
- asymmetric;
- cross-time;

without full stock-time attention.

## Most Important Architecture Idea

\[
\text{intra-stock aggregation}
\rightarrow
\text{inter-stock aggregation}.
\]

## Most Important Market Mechanism

\[
m_\tau
\rightarrow
\alpha(m_\tau)
\rightarrow
\text{feature-wise scaling}.
\]

## Most Important Efficiency Result

MASTER reduces direct stock-time attention by roughly one factor of:

\[
\tau.
\]

## Most Important Architecture Experiment

\[
(MA)STER
>
Naive
\]

and:

\[
(MA)STER
>
Clustering.
\]

This supports domain-specific factorization.

## Most Important Interpretation Result

The effective cross-time correlation map is:

\[
I_{u\leftarrow v}[i,j]
=
S_v^1[i,j]S_i^2[u,v].
\]

## Biggest Structural Limitation

Market information only controls feature selection; it does not directly control stock relationships.

## Best Research Direction

Condition the relation structure itself on:

- market regime;
- stock state;
- lead–lag dynamics;

while keeping correlation modeling sparse and scalable.

---

# 111. Compact Architecture Summary

```text
                       MASTER
========================================================

Market index statistics
        │
        ▼
Market status m_tau
        │
        ▼
Feature-wise Gate α(m_tau)
        │
        ▼
Alpha158 stock features x[u,t]
        │
        ▼
Gated features x_tilde[u,t]
        │
        ▼
Linear Feature Projection + Positional Encoding
        │
        ▼
Sequence Y[u,1:tau]
        │
        ▼
Intra-Stock Transformer
        │
        ▼
Local relay sequence h[u,1:tau]
        │
        ▼
for each historical time i:
    Cross-Stock MHA over h[:,i]
        │
        ▼
Temporal embedding z[u,i]
        │
        ▼
Latest-state-query Temporal Attention
        │
        ▼
Final stock embedding e[u]
        │
        ▼
Linear regression head
        │
        ▼
Normalized future return score


Effective cross-time influence:

(v,j)
  │
  ▼
intra-stock attention
  │
  ▼
h[v,i]
  │
  ▼
inter-stock attention at i
  │
  ▼
z[u,i]

Thus:
(v,j) → (u,i)
```

---

# 112. Source Location Map

Useful paper locations:

- **pp. 162–163:** problem motivation, momentary / cross-time stock correlation, market-guided feature selection.
- **pp. 163–164:** problem formulation, architecture overview, market-status representation, gating, intra-stock aggregation.
- **pp. 164–165:** intra-stock attention equations, inter-stock aggregation, temporal aggregation, prediction loss, complexity analysis.
- **pp. 166–167:** datasets, baselines, metrics, implementation, main results, architecture variants.
- **p. 168:** head sensitivity, gate-temperature sensitivity, correlation visualizations, conclusion.

---

# 113. Compact Retrieval Summary

MASTER (Li et al., AAAI 2024) is a market-guided stock Transformer designed to model momentary and cross-time stock correlations without directly attending over the full stock-time token field. It first builds a 63-dimensional market-status vector from CSI300/CSI500/CSI800 index price and volume statistics and maps it through a temperature-controlled softmax gate to dynamically rescale Alpha158 feature dimensions. MASTER then preserves time-specific stock states through a single-layer intra-stock Transformer, producing local embeddings \(h_{u,t}\). At each historical time step, inter-stock multi-head attention operates across stocks to generate \(z_{u,t}\). This factorization allows information from a source pair \((v,j)\) to reach a target pair \((u,i)\) through \((v,j)\to h_{v,i}\to z_{u,i}\), approximating full cross-time stock attention at roughly one factor of \(\tau\) lower complexity. A final temporal-attention layer uses the latest embedding as query to aggregate the sequence, and a linear head predicts the daily cross-sectionally normalized future return. MASTER uses Qlib Alpha158, \(\tau=8\), \(d=5\), trains on 2008Q1–2020Q1, validates on 2020Q2, and tests on 2020Q3–2022Q4. On CSI300 it reports IC/RankIC of 0.064/0.076 and on CSI800 0.052/0.066, outperforming XGBoost, recurrent baselines, Transformer, GAT, and DTML. Architecture experiments show the structured intra→inter factorization substantially outperforms naive full stock-time attention and LSH clustering. Market-guided gating provides additional adaptive feature selection, with stronger gating preferred on the more heterogeneous CSI800 universe. The main future opportunities are market-conditioned relation modeling, sparse dynamic stock routing, stock-specific gating, ranking-aligned objectives, and richer direct lead–lag structures.
