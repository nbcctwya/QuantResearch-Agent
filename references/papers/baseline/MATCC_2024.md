---
paper_id: MATCC_2024
title: "MATCC: A Novel Approach for Robust Stock Price Prediction Incorporating Market Trends and Cross-time Correlations"
authors:
  - Zhiyuan Cao
  - Jiayu Xu
  - Chengqi Dong
  - Peiwen Yu
  - Tian Bai
venue: "CIKM '24"
year: 2024
doi: "10.1145/3627673.3679715"
code: "https://github.com/caozhiy/MATCC"
paper_type: baseline
subtype:
  - finance_ml
  - cross_sectional_stock_prediction
  - market_trend_modeling
  - cross_time_correlation
  - rwkv
task:
  - normalized_future_return_prediction
  - cross_sectional_stock_ranking
  - portfolio_selection
markets:
  - CSI300
  - CSI800
  - S&P500
dataset:
  stock_features: "Qlib Alpha158"
  market_features: "63 index-derived features per market"
  lookback_T: 8
  prediction_interval_d: 5
  train: "2008-01-01 to 2020-03-31"
  validation: "2020-04-01 to 2020-06-30"
  test: "2020-07-01 to 2023-12-31"
core_modules:
  - market_status_representation
  - market_trend_guidance
  - stock_trend_decomposition
  - rwkv_time_correlation
  - self_attention_stock_correlation
  - temporal_embedding_aggregation
  - linear_prediction_head
core_mechanisms:
  - explicit_market_trend_guidance
  - trend_fluctuation_decomposition
  - time_first_cross_time_modeling
  - rwkv_linear_complexity_attention
  - dynamic_stock_correlation
  - market_noise_robustness
main_metrics:
  - IC
  - ICIR
  - RankIC
  - RankICIR
  - Excess_Annualized_Return
  - Information_Ratio
main_results:
  CSI300_IC: "0.117 ± 0.007"
  CSI300_RankIC: "0.086 ± 0.004"
  CSI300_AR: "0.80 ± 0.08"
  CSI300_IR: "8.5 ± 0.8"
  CSI800_IC: "0.118 ± 0.007"
  CSI800_RankIC: "0.083 ± 0.004"
  CSI800_AR: "1.62 ± 0.11"
  CSI800_IR: "13.3 ± 0.8"
  SP500_IC: "0.071 ± 0.001"
  SP500_RankIC: "0.035 ± 0.001"
  SP500_AR: "0.64 ± 0.02"
  SP500_IR: "6.7 ± 0.2"
relevance_to_agent:
  baseline_understanding: very_high
  architecture_transfer: very_high
  market_information_modeling: very_high
  temporal_cross_sectional_modeling: very_high
  robustness_reference: high
  reproducibility: high
priority: very_high
---

# MATCC: A Novel Approach for Robust Stock Price Prediction Incorporating Market Trends and Cross-time Correlations

## 0. Document Purpose

This document is a high-density, Agent-oriented reconstruction of **MATCC (Cao et al., CIKM 2024)**.

It is designed so that a research Agent can use the paper as:

1. a **baseline specification**;
2. an **implementation reference**;
3. a **market-information modeling reference**;
4. a **cross-time correlation modeling reference**;
5. a **source of ablation and robustness experiments**;
6. a **comparison point against MASTER-style market-aware stock Transformers**;
7. a **mechanism library for trend decomposition, RWKV temporal modeling, and dynamic stock correlation**.

The paper's central claim is that previous stock-prediction models underuse two sources of structure:

- **explicit market trends**;
- **cross-time stock correlations**.

MATCC addresses this by combining:

\[
\text{Market Trend Guidance}
+
\text{Stock Trend Decomposition}
+
\text{RWKV Time Correlation}
+
\text{Dynamic Stock Attention}.
\]

The model is not a latent-factor model like FactorVAE or FactorVQVAE.

Instead, MATCC is a **direct cross-sectional return predictor** that builds a richer stock representation before applying a simple linear prediction head.

The most important design idea is:

> Do not collapse the temporal dimension too early.  
> First preserve and enrich the full per-stock temporal sequence, then perform dynamic cross-stock interaction at each time step, and only aggregate time at the end.

---

# 1. Metadata

- **Title:** MATCC: A Novel Approach for Robust Stock Price Prediction Incorporating Market Trends and Cross-time Correlations
- **Authors:** Zhiyuan Cao, Jiayu Xu, Chengqi Dong, Peiwen Yu, Tian Bai
- **Conference:** 33rd ACM International Conference on Information and Knowledge Management
- **Venue shorthand:** CIKM '24
- **Dates:** October 21–25, 2024
- **Location:** Boise, ID, USA
- **DOI:** 10.1145/3627673.3679715
- **Paper length:** 10 pages
- **Code:** https://github.com/caozhiy/MATCC
- **Primary task:** cross-sectional stock return prediction
- **Secondary task:** stock ranking and portfolio selection
- **Core architecture:** market trend guidance + time-series decomposition + RWKV + cross-stock self-attention

---

# 2. Research Problem

## 2.1 High-Level Task

Given historical stock features:

\[
\{x_{u,t}\}_{u\in S,t\in[1,T]},
\]

predict the future normalized return ratio for every stock in the current cross-section:

\[
\{r_u\}_{u\in S}.
\]

Here:

- \(S\): stock universe;
- \(u\): stock index;
- \(t\): historical time index;
- \(T\): lookback length;
- \(F\): stock feature dimension.

---

## 2.2 Two Main Problems Identified by the Authors

The paper argues that existing work has two important limitations.

### Problem A — Market trends are mostly implicit

Many models use only individual stock data and hope the network implicitly reconstructs the global market environment.

But stock data contain substantial:

- random short-term fluctuations;
- sentiment-driven movements;
- event noise;
- idiosyncratic effects.

Therefore:

\[
\text{aggregate stock features}
\not\Rightarrow
\text{clean market trend}
\]

automatically.

The authors instead advocate using explicit market-index information.

---

### Problem B — Stock correlation is modeled too coarsely in time

A common pipeline is:

\[
\text{stock time series}
\rightarrow
\text{one temporal embedding per stock}
\rightarrow
\text{stock correlation}.
\]

This can erase:

- local time-step information;
- lead–lag structure;
- cross-time relations.

MATCC instead preserves:

\[
h_{u,1},h_{u,2},\dots,h_{u,T}
\]

before cross-stock interaction.

---

# 3. Core Motivation

The paper's reasoning can be compressed into the following chain.

## Step 1

Stock prediction is difficult because financial series are:

- non-stationary;
- noisy;
- highly correlated across assets.

## Step 2

Market indices provide cleaner global information than trying to reconstruct the market only from noisy individual-stock inputs.

## Step 3

Each stock itself also has two qualitatively different components:

- long-term trend;
- short-term fluctuation.

These should be modeled separately.

## Step 4

Temporal information should not be compressed too early.

Instead, maintain a full sequence of enriched stock states.

## Step 5

Model:

1. within-stock cross-time dependence;
2. cross-stock dependence at each enriched time step.

## Step 6

Only after these interactions should the temporal sequence be aggregated into the final stock embedding.

---

# 4. Main Contributions

The paper states three main contributions.

## Contribution 1 — Market-Stock Trend Guidance

The model explicitly extracts market movement / trend information from market-index data and injects it into each stock's representation.

It also decomposes each stock representation into:

- trend;
- fluctuation.

The intended benefit is:

> reduce sensitivity to noise while retaining both stable and volatile information.

---

## Contribution 2 — Cross-Time Correlation Modeling

The model retains a temporal sequence for each stock and uses RWKV for within-stock temporal correlation before dynamic inter-stock attention.

The intended benefit is:

> preserve time-step-specific information and capture lead–lag / inter-temporal relations more effectively.

---

## Contribution 3 — Broad Empirical Evaluation

The paper evaluates MATCC on:

- CSI300;
- CSI800;
- S&P500;

with:

- ranking metrics;
- portfolio metrics;
- module ablations;
- RWKV replacement studies;
- hyperparameter sensitivity;
- artificial market-noise robustness;
- attention visualizations.

---

# 5. Relationship to MASTER

MASTER is especially important because MATCC repeatedly positions itself against it.

## MASTER-style design

According to this paper's description, MASTER:

- uses Transformer blocks;
- models momentary and cross-time stock correlations;
- incorporates market information;
- uses market information for feature selection / gating.

---

## MATCC's claimed differences

MATCC emphasizes:

1. **explicit market trend extraction** rather than only market-guided feature scaling;
2. **stock trend / fluctuation decomposition**;
3. **RWKV-based within-stock temporal modeling**;
4. **preservation of the full temporal sequence before stock correlation**;
5. **time-correlation first, stock-correlation second**.

---

## Agent Interpretation

A useful simplified comparison is:

```text
MASTER:
market info
    ↓
feature selection / stock Transformer
    ↓
temporal + cross-stock modeling

MATCC:
market-index sequence
    ↓
explicit market trend
    ↓
stock trend/fluctuation decomposition
    ↓
within-stock RWKV temporal modeling
    ↓
cross-stock attention at every time step
    ↓
temporal aggregation
```

This difference is central when MATCC is used as a baseline.

---

# 6. Overall Architecture

The paper describes four major framework components, followed by prediction.

A more granular implementation view has six steps:

1. Market Status Representation
2. Market Trend Guidance
3. Stock Trend Decomposition
4. Time Correlation
5. Stock Correlation
6. Stock Embedding Aggregation
7. Prediction

The paper's conclusion informally counts six main functional steps by grouping the market-trend construction.

---

# 7. Compact Architecture Diagram

```text
Market index data m
        │
        ▼
Market Status Representation
        │
        ▼
Depthwise 1D Convolution
        │
        ▼
Market Trend m_t
        │
        ├────────────────────────────┐
        │                            │
Stock Alpha158 x_u                   │
        │                            │
        ▼                            │
Feature Projection                   │
        │                            │
        └────────── + m_t ───────────┘
                     │
                     ▼
          Market-guided stock sequence
                     │
                     ▼
           AvgPool decomposition
             ┌───────┴────────┐
             ▼                ▼
           Trend          Fluctuation
             │                │
             └──── Linear ────┘
                     │
                     ▼
          Enriched stock sequence X_u
                     │
                     ▼
          RWKV Time Correlation
                     │
                     ▼
          h_{u,1:T} for every stock
                     │
                     ▼
      Stock Self-Attention per time step
                     │
                     ▼
          z_{u,1:T} for every stock
                     │
                     ▼
        Temporal Attention Aggregation
                     │
                     ▼
                Stock embedding e_u
                     │
                     ▼
              Linear Predictor
                     │
                     ▼
          Normalized future return
```

---

# 8. Problem Formulation

For stock \(u\), at each trading day \(t\):

\[
x_{u,t}\in\mathbb R^F.
\]

The lookback sequence is:

\[
\{x_{u,t}\}_{t=1}^{T}.
\]

The paper predicts a future return ratio rather than absolute price change.

The raw return target is:

\[
\hat r_u
=
\frac{
c_{u,T+d}-c_{u,T+1}
}{
c_{u,T+1}
},
\]

where:

- \(c_{u,t}\): stock \(u\)'s close price at time \(t\);
- \(d\): predefined prediction interval.

Then the paper performs daily cross-sectional normalization:

\[
r_u
=
Norm_S(\hat r_u).
\]

The final task is:

> jointly predict \(\{r_u\}_{u\in S}\) for the stock cross-section.

---

# 9. Important Label Detail

The paper later sets:

\[
d=5.
\]

A research Agent should preserve the exact formula rather than replacing it with a generic "5-day return."

The target is explicitly written as:

\[
\frac{c_{T+d}-c_{T+1}}{c_{T+1}}.
\]

Because indexing conventions can differ across Qlib implementations, exact reproduction should verify the code.

---

# 10. Daily Cross-Sectional Target Normalization

The target is normalized daily across stocks.

This means the model is not trained purely against raw economic returns.

It predicts a relative normalized cross-sectional label.

That aligns naturally with:

- IC;
- RankIC;
- stock selection.

Agent warning:

> headline prediction metrics cannot be compared blindly against models trained on a different target normalization.

---

# 11. Module 1 — Market Status Representation

The paper does not infer market condition solely from the stock universe.

It explicitly uses market-index information.

Representative market variables include:

- market index price;
- market index trading volume.

For historical market context, it uses statistics over multiple reference intervals.

For each reference window \(d'\), it includes statistics such as:

- mean market index price;
- standard deviation of market index price;
- mean market index volume;
- standard deviation of market index volume.

Reference intervals:

\[
d'
\in
\{5,10,20,30,60\}.
\]

The current and historical information together form the market-status representation.

---

# 12. Market Features by Market

Total market feature dimension:

\[
F'=63.
\]

## China

Constructed from:

- CSI300;
- CSI500;
- CSI800.

## United States

Constructed from:

- GSPC;
- DJI;
- NDX.

This setup is closely related to later market-aware Qlib baselines.

---

# 13. Module 2 — Market Trend Guidance

The market-status sequence is:

\[
m
\in
\mathbb R^{T\times F'}.
\]

The paper first applies a linear transformation:

\[
m'
=
(W_\alpha m+b_\alpha)^T.
\]

Then a 1D depthwise convolution:

\[
m_t
=
W_\beta
DepthConv(m')
+
b_\beta.
\]

The paper states:

\[
m_t
\in
\mathbb R^{D\times T}.
\]

---

# 14. Why Depthwise Convolution?

The paper explicitly prefers depthwise convolution over simple pooling.

Reason:

> each market-information channel should preserve its own temporal trend pattern instead of being mixed prematurely with other channels.

A depthwise kernel independently processes channels.

The paper uses:

\[
kernel=stride=k_c,
\]

\[
padding=0.
\]

A linear projection then maps the result into the model feature space.

---

# 15. Market Trend Injection into Each Stock

The individual stock sequence is projected:

\[
W_\theta x_u.
\]

The market trend is then added directly:

\[
\hat x_u
=
W_\theta x_u
+
m_t^T.
\]

with:

\[
\hat x_u
\in
\mathbb R^{T\times D}.
\]

---

# 16. Why Direct Addition?

The authors contrast their method with market-aware gating approaches.

Their intuition is:

> the market affects all stock features, so market trend should directly guide the entire stock representation rather than merely rescale selected dimensions.

The inductive bias is therefore:

\[
\text{stock state}
=
\text{stock-specific information}
+
\text{shared market trend}.
\]

---

# 17. Module 3 — Stock Trend Decomposition

The market-guided stock sequence is decomposed into:

- a smooth trend component;
- a fluctuation component.

Trend:

\[
\hat x_u^t
=
AvgPool
(
Padding(\hat x_u)
).
\]

Fluctuation:

\[
\hat x_u^f
=
\hat x_u
-
\hat x_u^t.
\]

Both are in:

\[
\mathbb R^{T\times D}.
\]

---

# 18. Interpretation of Trend and Fluctuation

## Trend

Represents:

- stable structure;
- long-horizon movement;
- persistent trajectory.

## Fluctuation

Represents:

- short-horizon variation;
- local irregularity;
- volatile movement.

The paper does **not** discard the fluctuation component.

Instead, it explicitly keeps both.

This is important:

> MATCC is a decomposition model, not simply a smoothing model.

---

# 19. Average Pooling as Decomposition

The trend is produced through average pooling.

Kernel size:

\[
k_a.
\]

Padding is used to preserve sequence length.

The paper argues average pooling:

- smooths local irregularities;
- reduces noise sensitivity.

---

# 20. Recombination of Trend and Fluctuation

Each component is passed through a one-layer linear mapping along the time axis.

The paper writes:

\[
X_u^T
=
\left(
W_f(\hat x_u^f)^T+b_f
\right)
+
\left(
W_t(\hat x_u^t)^T+b_t
\right).
\]

The final sequence can be understood as:

\[
X_u
\in
\mathbb R^{T\times D}.
\]

Thus:

\[
\text{final stock sequence}
=
\text{transformed fluctuation}
+
\text{transformed trend}.
\]

---

# 21. Market-Stock Trend Guidance Module: Combined Interpretation

The entire trend stage performs:

\[
x_u
\rightarrow
\hat x_u
\rightarrow
(\hat x_u^t,\hat x_u^f)
\rightarrow
X_u.
\]

This gives every stock representation:

- explicit market context;
- long-term stock trend;
- short-term stock fluctuation.

The paper argues this prepares a cleaner sequence for later correlation mining.

---

# 22. Module 4 — Time Correlation

The first correlation step operates **within each stock across time**.

Input for stock \(u\):

\[
X_u
=
[X_{u,1},\dots,X_{u,T}].
\]

The paper uses RWKV rather than a standard Transformer.

---

# 23. Why RWKV?

The paper gives three primary reasons.

## Reason 1 — Natural Temporal Order

RWKV is RNN-like and inherently respects input order.

It does not require explicit positional encoding in the same way as vanilla Transformer self-attention.

## Reason 2 — Causal Processing

At time \(t\), the model uses only:

- current information;
- earlier information.

This matches financial temporal causality.

## Reason 3 — Linear Complexity

RWKV's WKV operator approximates attention-like aggregation with linear time and space complexity.

---

# 24. RWKV Structure

The paper uses two sub-blocks:

1. Time Mixing
2. Channel Mixing

The input is layer-normalized and processed with residual connections.

---

# 25. Time Mixing

RWKV mixes current and previous inputs using trainable vectors:

\[
\mu_r,\mu_k,\mu_v.
\]

The receptance vector is:

\[
r_t
=
W_r
\left(
\mu_r\odot X_{u,t}
+
(1-\mu_r)\odot X_{u,t-1}
\right).
\]

The key vector is:

\[
k_t
=
W_k
\left(
\mu_k\odot X_{u,t}
+
(1-\mu_k)\odot X_{u,t-1}
\right).
\]

The value vector is:

\[
v_t
=
W_v
\left(
\mu_v\odot X_{u,t}
+
(1-\mu_v)\odot X_{u,t-1}
\right).
\]

---

# 26. WKV Operator

The paper writes the time-dependent WKV update as:

\[
wkv_t
=
\frac{
\sum_{i=1}^{t-1}
e^{-(t-1-i)w+k_i}\odot v_i
+
e^{u+k_t}\odot v_t
}{
\sum_{i=1}^{t-1}
e^{-(t-1-i)w+k_i}
+
e^{u+k_t}
}.
\]

where:

- \(w\): learnable channel-wise time-decay vector;
- \(u\): learnable bonus for the current / newly encountered token.

This gives a decayed history aggregation.

---

# 27. Multi-Head WKV

The paper supports multiple heads:

\[
wkv_t
=
concat
\{
wkv_t^1,\dots,wkv_t^N
\}.
\]

where:

\[
N
\]

is the number of RWKV heads.

The selected value later is:

\[
N=4.
\]

---

# 28. RWKV Time-Mixing Output

The output is:

\[
o_t
=
W_o
\left(
\sigma(r_t)
\odot
wkv_t
\right),
\]

where:

\[
\sigma(\cdot)
\]

is the sigmoid function.

The receptance controls how much of the aggregated state is passed through.

---

# 29. Channel Mixing

The Channel Mixing sub-block mixes feature dimensions using a nonlinear mapping.

The paper gives:

\[
o'_t
=
\sigma(r'_t)
\odot
\left(
W'_v
\cdot
\max(k'_t,0)^2
\right).
\]

The activation is squared ReLU.

---

# 30. Output of Time Correlation

After RWKV:

\[
h_{u,t},
\qquad
t\in[1,T]
\]

is retained for every time step.

This is critical.

MATCC does **not** immediately reduce the stock sequence to:

\[
h_u.
\]

Instead it preserves:

\[
h_{u,1:T}.
\]

---

# 31. Cross-Time Interpretation

Each:

\[
h_{u,t}
\]

contains:

- current-time stock information;
- temporally mixed information from earlier time steps.

This allows later stock attention at time \(t\) to interact using temporally enriched representations.

---

# 32. Important Architectural Nuance

The paper motivates "cross-time stock correlations."

However, the Stock Correlation block itself performs attention across stocks **at the same time index**:

\[
t.
\]

Cross-time cross-stock dependence is therefore mediated through the temporally enriched state:

\[
h_{u,t},
\]

rather than by a single explicit all-pairs attention operation between:

\[
(u,t)
\]

and:

\[
(v,t').
\]

This is a useful distinction for future model design.

---

# 33. Module 5 — Stock Correlation

For each time step:

\[
t,
\]

the representations of all stocks are concatenated:

\[
H_t
=
\|_{u\in S}
h_{u,t}.
\]

Then:

\[
Q_t
=
W_QH_t,
\]

\[
K_t
=
W_KH_t,
\]

\[
V_t
=
W_VH_t.
\]

The cross-stock representation is:

\[
Z_t
=
FFN
\left(
MHA(Q_t,K_t,V_t)
+
H_t
\right).
\]

---

# 34. Dynamic Stock Relationship

The stock-attention block allows each stock to exchange information with all others at a given time step.

The relation is not defined by:

- industry graph;
- predefined concept graph;
- manually specified edges.

Instead:

> correlations are learned dynamically through attention.

The authors emphasize that stocks can autonomously establish or dissolve relationships.

---

# 35. Feed-Forward Network

The stock-correlation FFN is described as:

- two-layer MLP;
- ReLU activation;
- residual connection.

---

# 36. Why Time Correlation Comes Before Stock Correlation

The paper explicitly studies module order.

Full MATCC uses:

\[
\text{Time Correlation}
\rightarrow
\text{Stock Correlation}.
\]

The authors argue:

> stock trend information should first be modeled inside each stock before cross-stock mixing, otherwise stock-to-stock interaction may interfere with distinct temporal trends.

This claim is tested with the "Change order" ablation.

---

# 37. Module 6 — Stock Embedding Aggregation

After Stock Correlation, each stock still has:

\[
z_{u,1},
\dots,
z_{u,T}.
\]

The final aggregation uses the last-day embedding:

\[
z_{u,T}
\]

as the query.

For historical time \(t\):

\[
\lambda_{u,t}
=
\frac{
\exp
(
z_{u,t}^TW_\lambda z_{u,T}
)
}{
\sum_{i=1}^{T}
\exp
(
z_{u,i}^TW_\lambda z_{u,T}
)
}.
\]

Final stock embedding:

\[
e_u
=
\sum_{t\in[1,T]}
\lambda_{u,t}z_{u,t}.
\]

---

# 38. Interpretation of Temporal Aggregation

The last time step acts as a summary of the current state.

Historical embeddings are weighted by relevance to that current state.

Thus the final representation is not:

- simple mean pooling;
- last hidden state only.

It is a current-state-conditioned temporal weighted sum.

---

# 39. Prediction Head

The final stock embedding is passed to a linear predictor:

\[
\hat r_u
=
g(e_u).
\]

The model is trained using MSE:

\[
Loss
=
\sum_{u\in S}
MSE(r_u,\hat r_u).
\]

All stocks on one prediction date are jointly optimized in each batch.

---

# 40. Training Batch Semantics

The paper states:

> one batch corresponds to all stocks on a particular prediction date.

Therefore the effective batch object is a **cross-section**, not an independent stock sample.

This matters because:

- stock attention requires the full cross-section;
- daily label normalization is cross-sectional;
- memory requirements scale with stock count.

A training epoch contains multiple date-based batches.

---

# 41. Experimental Research Questions

The experiments answer four questions.

## RQ1

How does MATCC compare with state-of-the-art models?

## RQ2

Are the proposed stock-trend and inter-temporal correlation components effective?

## RQ3

How do:

- hyperparameters;
- market noise;

affect MATCC?

## RQ4

What can attention maps reveal about stock trends and cross-time correlations?

---

# 42. Datasets

Three real-world stock datasets are used.

## CSI300

\[
300
\]

stocks.

## CSI800

\[
800
\]

stocks.

## S&P500

\[
500
\]

stocks.

The data span:

\[
2008
\rightarrow
2023.
\]

---

# 43. Train / Validation / Test Split

All three markets use the same date split.

## Training

\[
2008\text{-}01\text{-}01
\rightarrow
2020\text{-}03\text{-}31.
\]

## Validation

\[
2020\text{-}04\text{-}01
\rightarrow
2020\text{-}06\text{-}30.
\]

## Testing

\[
2020\text{-}07\text{-}01
\rightarrow
2023\text{-}12\text{-}31.
\]

This is a fixed chronological split.

---

# 44. Input Features

Stock features:

\[
158
\]

Alpha158 features.

Market features:

\[
63.
\]

Total reported feature count:

\[
158+63.
\]

Important nuance:

> the 63 market features are not necessarily concatenated directly to every stock vector in the raw input. They feed the market-trend branch and then guide stock features.

---

# 45. Lookback and Horizon Settings

Lookback:

\[
T=8.
\]

Prediction interval:

\[
d=5.
\]

Reference market-statistic windows:

\[
d'
=
5,10,20,30,60.
\]

This setup differs substantially from FactorVAE / FactorVQVAE protocols that use \(T=20\).

---

# 46. Input Preprocessing — Robust Z-Score

The paper applies robust normalization within stocks.

It replaces:

- mean;
- standard deviation;

with:

- median;
- median absolute deviation.

The printed equation is:

\[
\tilde x_u
=
\frac{
|x_u-MED(X)|
}{
MAD(X)
}.
\]

---

# 47. Important Normalization Caveat

The absolute-value operator printed in Eq. 11 is unusual.

Conventional robust z-score is normally signed:

\[
(x-\text{median})/\text{MAD}.
\]

The source PDF explicitly prints:

\[
|x-\text{median}|/\text{MAD}.
\]

Therefore:

> an exact reproduction should verify the released MATCC code rather than silently replacing Eq. 11 with the conventional formula.

Do not "correct" the paper in the knowledge base without source verification.

---

# 48. Dropping Extreme Labels

The paper drops:

- lowest 2.5%;
- highest 2.5%;

of labels.

Motivation:

Chinese-market limit-up / limit-down stocks can be difficult or impossible to trade at the desired price.

The authors argue that learning a strategy centered on these extreme labels can be impractical.

---

# 49. Extreme-Label Caveat

The paper motivates this preprocessing by tradability.

However, the text does not fully specify whether the 2.5% thresholds are computed:

- globally;
- per date;
- per market;
- within another grouping.

Exact reproduction should verify the code.

---

# 50. Baselines

The baseline set contains seven models.

## LSTM

Standard recurrent stock forecasting baseline.

## ALSTM

LSTM with temporal attentive aggregation.

## GRU

Gated recurrent sequence model.

## GAT

Graph-attention stock-prediction model.

## Transformer

Vanilla Transformer-based forecasting model.

## DTML

Dynamic stock-correlation mining model with market information.

## MASTER

Market-aware stock Transformer modeling momentary and cross-time stock correlation with market-guided feature selection.

---

# 51. Baseline Selection Logic

The baseline set covers:

- classical recurrent sequence models;
- attention models;
- graph-based stock-correlation modeling;
- dynamic cross-stock correlation models;
- market-aware SOTA architectures.

This makes:

- DTML;
- MASTER;

the most structurally important comparators.

---

# 52. Evaluation Metrics

The paper uses four ranking metrics.

## IC

Pearson correlation between:

- model prediction;
- target label.

For date \(t\):

\[
IC_t
=
corr_{\text{Pearson}}
(
\hat r_t,r_t
).
\]

## RankIC

Spearman rank correlation:

\[
RankIC_t
=
corr_{\text{Spearman}}
(
\hat r_t,r_t
).
\]

## ICIR

The paper describes ICIR as IC normalized by its standard deviation over time.

Conceptually:

\[
ICIR
\approx
\frac{
mean(IC_t)
}{
std(IC_t)
}.
\]

## RankICIR

Analogously:

\[
RankICIR
\approx
\frac{
mean(RankIC_t)
}{
std(RankIC_t)
}.
\]

The paper does not print a dedicated equation for these two ratios.

---

# 53. Portfolio Strategy

The paper uses:

> top30-drop30.

Interpretation given in the paper:

- "top30": hold stocks ranked in the top 30;
- "drop30": drop any stock once its score falls outside the top 30.

This is more aggressive than the TopK-Drop-\(n\) setup in FactorVAE / FactorVQVAE where only a limited number of names are replaced.

---

# 54. Portfolio Metrics

The paper reports:

- Excess Annualized Return (AR);
- Information Ratio (IR).

Importantly:

> the portfolio evaluation is explicitly reported **without transaction cost**.

This must be remembered when comparing to backtests with fees.

---

# 55. Portfolio Evaluation Caveat

Because:

- the strategy can replace stocks aggressively;
- transaction costs are excluded;

the reported AR and IR should not be interpreted as net deployable portfolio performance.

For baseline comparison:

> match the paper's exact no-cost protocol first, then evaluate a separate realistic-cost protocol.

---

# 56. Implementation Platform

MATCC is implemented in:

- PyTorch;
- Qlib.

The preprocessing pipeline follows MASTER.

The authors implement the missing `DropExtremeLabel` step themselves according to another cited work.

---

# 57. Baseline Hyperparameters

The paper states that baseline:

- optimal hyperparameters;
- learning rates;

follow those provided by MASTER.

The paper does not reproduce every baseline configuration in this text.

Therefore exact baseline parity should be checked in released code.

---

# 58. MATCC Hyperparameter Search

The paper tunes:

## Model dimension

\[
D
\in
\{128,256,512\}.
\]

## Depthwise convolution kernel

\[
k_c
\in
\{3,5,7\}.
\]

## Average-pool kernel

\[
k_a
\in
\{3,5,7\}.
\]

## RWKV heads

\[
N
\in
\{2,4,8\}.
\]

---

# 59. Selected Hyperparameters

Chosen using validation IC:

\[
D=256,
\]

\[
k_c=5,
\]

\[
k_a=3,
\]

\[
N=4.
\]

---

# 60. Optimizer and Learning Rate Schedule

Optimizer:

\[
Adam.
\]

Scheduler:

\[
CosineAnnealingLR
\]

with warm-up.

Maximum learning rate:

\[
3\times10^{-3}.
\]

Minimum learning rate:

\[
2\times10^{-4}.
\]

Warm-up epochs:

\[
10.
\]

Epochs between restarts:

\[
15.
\]

Total training epochs:

\[
70.
\]

---

# 61. Model Selection

The paper states:

> the final epoch parameters are used for testing.

This is distinct from early-stopping-based selection.

Validation data are used for hyperparameter selection, but not for selecting the best epoch according to the described protocol.

---

# 62. Hardware

Experiments run on:

- Intel Xeon E5-2678 v3;
- 48 CPUs;
- 125 GB RAM;
- NVIDIA RTX 3090;
- 24 GB VRAM.

---

# 63. Random Seeds

Training/testing is repeated:

\[
5
\]

times for all methods.

The paper reports:

- mean;
- standard deviation.

This is a strong feature of the experimental protocol.

---

# 64. Main Results — CSI300

| Model | IC | ICIR | RankIC | RankICIR | AR | IR |
|---|---:|---:|---:|---:|---:|---:|
| LSTM | 0.048 ± 0.008 | 0.34 ± 0.06 | 0.050 ± 0.010 | 0.34 ± 0.05 | 0.13 ± 0.04 | 1.3 ± 0.5 |
| GRU | 0.046 ± 0.002 | 0.32 ± 0.02 | 0.046 ± 0.004 | 0.33 ± 0.02 | 0.11 ± 0.04 | 1.0 ± 0.4 |
| ALSTM | 0.043 ± 0.004 | 0.30 ± 0.03 | 0.041 ± 0.004 | 0.30 ± 0.03 | 0.11 ± 0.03 | 1.1 ± 0.3 |
| Transformer | 0.048 ± 0.002 | 0.32 ± 0.02 | 0.046 ± 0.003 | 0.32 ± 0.02 | 0.11 ± 0.02 | 1.0 ± 0.2 |
| GAT | 0.052 ± 0.001 | 0.38 ± 0.02 | 0.051 ± 0.002 | 0.38 ± 0.02 | 0.17 ± 0.03 | 1.6 ± 0.2 |
| DTML | 0.051 ± 0.004 | 0.35 ± 0.03 | 0.052 ± 0.006 | 0.35 ± 0.03 | 0.15 ± 0.02 | 1.5 ± 0.1 |
| MASTER | 0.053 ± 0.004 | 0.39 ± 0.04 | 0.054 ± 0.006 | 0.39 ± 0.05 | 0.19 ± 0.02 | 1.9 ± 0.1 |
| **MATCC** | **0.117 ± 0.007** | **1.02 ± 0.06** | **0.086 ± 0.004** | **0.87 ± 0.06** | **0.80 ± 0.08** | **8.5 ± 0.8** |

MATCC is far ahead on all reported metrics.

---

# 65. CSI300 Relative Improvement vs. MASTER

From Table 2:

## IC

\[
0.053
\rightarrow
0.117.
\]

Relative improvement:

\[
\approx120.8\%.
\]

## RankIC

\[
0.054
\rightarrow
0.086.
\]

Relative improvement:

\[
\approx59.3\%.
\]

## AR

\[
0.19
\rightarrow
0.80.
\]

Relative improvement:

\[
\approx321.1\%.
\]

## IR

\[
1.9
\rightarrow
8.5.
\]

Relative improvement:

\[
\approx347.4\%.
\]

---

# 66. Important Claim-vs-Table Nuance

The paper states approximately:

- 120% improvement in ranking metrics;
- 300% improvement in portfolio metrics;

relative to MASTER.

The exact percentage depends strongly on the chosen metric.

For example:

- CSI300 IC improves about 121%;
- CSI300 RankIC improves about 59%;
- CSI300 AR improves about 321%;
- CSI300 IR improves about 347%.

Therefore:

> the broad percentage claim should not be treated as one uniform metric improvement.

Use the actual table values for precise comparisons.

---

# 67. Main Results — CSI800

| Model | IC | ICIR | RankIC | RankICIR | AR | IR |
|---|---:|---:|---:|---:|---:|---:|
| LSTM | 0.043 ± 0.005 | 0.36 ± 0.05 | 0.043 ± 0.008 | 0.36 ± 0.05 | 0.15 ± 0.04 | 1.2 ± 0.4 |
| GRU | 0.040 ± 0.002 | 0.35 ± 0.01 | 0.043 ± 0.002 | 0.35 ± 0.01 | 0.07 ± 0.02 | 0.6 ± 0.2 |
| ALSTM | 0.039 ± 0.003 | 0.34 ± 0.02 | 0.041 ± 0.004 | 0.33 ± 0.02 | 0.11 ± 0.03 | 0.9 ± 0.3 |
| Transformer | 0.038 ± 0.003 | 0.32 ± 0.03 | 0.040 ± 0.006 | 0.31 ± 0.04 | 0.10 ± 0.02 | 0.8 ± 0.2 |
| GAT | 0.044 ± 0.004 | 0.39 ± 0.04 | 0.047 ± 0.006 | 0.38 ± 0.04 | 0.15 ± 0.03 | 1.2 ± 0.2 |
| DTML | 0.043 ± 0.002 | 0.35 ± 0.02 | 0.047 ± 0.004 | 0.34 ± 0.02 | 0.14 ± 0.01 | 1.2 ± 0.1 |
| MASTER | 0.046 ± 0.004 | 0.42 ± 0.05 | 0.048 ± 0.007 | 0.41 ± 0.05 | 0.16 ± 0.04 | 1.3 ± 0.3 |
| **MATCC** | **0.118 ± 0.007** | **1.33 ± 0.07** | **0.083 ± 0.004** | **1.07 ± 0.06** | **1.62 ± 0.11** | **13.3 ± 0.8** |

The gap is even larger in portfolio metrics.

---

# 68. CSI800 Relative Improvement vs. MASTER

## IC

\[
0.046
\rightarrow
0.118
\]

approximately:

\[
+156.5\%.
\]

## RankIC

\[
0.048
\rightarrow
0.083
\]

approximately:

\[
+72.9\%.
\]

## AR

\[
0.16
\rightarrow
1.62
\]

approximately:

\[
+912.5\%.
\]

## IR

\[
1.3
\rightarrow
13.3
\]

approximately:

\[
+923.1\%.
\]

These enormous portfolio improvements make the no-transaction-cost setting especially important to remember.

---

# 69. Main Results — S&P500

| Model | IC | ICIR | RankIC | RankICIR | AR | IR |
|---|---:|---:|---:|---:|---:|---:|
| LSTM | 0.006 ± 0.001 | 0.03 ± 0.00 | 0.005 ± 0.001 | 0.03 ± 0.00 | 0.08 ± 0.03 | 0.8 ± 0.2 |
| GRU | 0.012 ± 0.002 | 0.07 ± 0.01 | 0.010 ± 0.002 | 0.07 ± 0.01 | 0.07 ± 0.01 | 0.6 ± 0.1 |
| ALSTM | 0.011 ± 0.002 | 0.06 ± 0.01 | 0.010 ± 0.003 | 0.05 ± 0.01 | 0.07 ± 0.01 | 0.6 ± 0.1 |
| Transformer | 0.015 ± 0.001 | 0.08 ± 0.02 | 0.014 ± 0.001 | 0.08 ± 0.02 | 0.08 ± 0.03 | 0.6 ± 0.1 |
| GAT | 0.014 ± 0.005 | 0.07 ± 0.02 | 0.013 ± 0.004 | 0.07 ± 0.02 | 0.09 ± 0.02 | 0.7 ± 0.2 |
| DTML | 0.012 ± 0.001 | 0.10 ± 0.01 | 0.011 ± 0.001 | 0.10 ± 0.01 | 0.05 ± 0.01 | 0.5 ± 0.1 |
| MASTER | 0.021 ± 0.001 | 0.14 ± 0.01 | 0.019 ± 0.001 | 0.14 ± 0.01 | 0.09 ± 0.02 | 0.9 ± 0.1 |
| **MATCC** | **0.071 ± 0.001** | **0.81 ± 0.02** | **0.035 ± 0.001** | **0.32 ± 0.01** | **0.64 ± 0.02** | **6.7 ± 0.2** |

All models perform worse in raw ranking terms on S&P500 than on the Chinese markets.

MATCC still remains far ahead.

---

# 70. Cross-Market Pattern

The authors attribute lower U.S. predictive performance partly to different:

- trading mechanisms;
- market rules.

A more general research interpretation is:

> model effectiveness and signal strength are strongly market-dependent.

This is consistent with many later cross-market Qlib studies.

---

# 71. Ablation Study Overview

Ablations are run only on Chinese datasets due to space/resource constraints.

Variants:

- w/o MTG
- w/o STD
- w/o TC
- w/o SC
- Change order
- Transformer
- Mamba
- Ours

Definitions:

- MTG = Market Trend Guidance
- STD = Stock Trend Decomposition
- TC = Time Correlation
- SC = Stock Correlation

---

# 72. CSI300 Ablation Results

| Variant | IC | ICIR | RankIC | RankICIR | AR | IR |
|---|---:|---:|---:|---:|---:|---:|
| w/o MTG | 0.114 ± 0.010 | 1.01 ± 0.10 | 0.085 ± 0.004 | 0.96 ± 0.10 | 0.79 ± 0.05 | 8.4 ± 0.6 |
| w/o STD | 0.048 ± 0.003 | 0.37 ± 0.02 | 0.048 ± 0.003 | 0.37 ± 0.02 | 0.12 ± 0.02 | 1.3 ± 0.3 |
| w/o TC | 0.055 ± 0.008 | 0.42 ± 0.08 | 0.052 ± 0.006 | 0.35 ± 0.18 | 0.19 ± 0.09 | 2.0 ± 1.0 |
| w/o SC | 0.049 ± 0.005 | 0.41 ± 0.05 | 0.046 ± 0.004 | 0.41 ± 0.05 | 0.16 ± 0.05 | 1.9 ± 0.5 |
| Change order | 0.054 ± 0.002 | 0.39 ± 0.04 | 0.053 ± 0.003 | 0.39 ± 0.04 | 0.14 ± 0.03 | 1.4 ± 0.3 |
| Transformer | 0.083 ± 0.006 | 0.65 ± 0.05 | 0.071 ± 0.003 | 0.65 ± 0.04 | 0.52 ± 0.08 | 5.3 ± 0.8 |
| Mamba | 0.052 ± 0.003 | 0.38 ± 0.03 | 0.051 ± 0.002 | 0.29 ± 0.16 | 0.13 ± 0.06 | 1.3 ± 0.7 |
| **MATCC** | **0.117 ± 0.007** | **1.02 ± 0.06** | **0.086 ± 0.004** | **0.87 ± 0.06** | **0.80 ± 0.08** | **8.5 ± 0.8** |

---

# 73. CSI800 Ablation Results

| Variant | IC | ICIR | RankIC | RankICIR | AR | IR |
|---|---:|---:|---:|---:|---:|---:|
| w/o MTG | 0.087 ± 0.032 | 0.91 ± 0.42 | 0.068 ± 0.017 | 0.83 ± 0.33 | 1.11 ± 0.72 | 9.1 ± 5.8 |
| w/o STD | 0.043 ± 0.002 | 0.42 ± 0.02 | 0.047 ± 0.002 | 0.42 ± 0.02 | 0.14 ± 0.04 | 1.2 ± 0.4 |
| w/o TC | 0.047 ± 0.003 | 0.43 ± 0.03 | 0.048 ± 0.004 | 0.42 ± 0.03 | 0.14 ± 0.06 | 1.2 ± 0.6 |
| w/o SC | 0.061 ± 0.029 | 0.63 ± 0.33 | 0.051 ± 0.015 | 0.60 ± 0.29 | 0.49 ± 0.60 | 4.8 ± 5.8 |
| Change order | 0.052 ± 0.009 | 0.46 ± 0.11 | 0.051 ± 0.006 | 0.45 ± 0.11 | 0.24 ± 0.11 | 2.3 ± 1.1 |
| Transformer | 0.097 ± 0.015 | 0.98 ± 0.19 | 0.079 ± 0.007 | 0.91 ± 0.13 | 1.25 ± 0.38 | 10.4 ± 2.6 |
| Mamba | 0.046 ± 0.002 | 0.42 ± 0.03 | 0.048 ± 0.001 | 0.42 ± 0.03 | 0.13 ± 0.03 | 1.2 ± 0.2 |
| **MATCC** | **0.118 ± 0.007** | **1.33 ± 0.07** | **0.083 ± 0.004** | **1.07 ± 0.06** | **1.62 ± 0.11** | **13.3 ± 0.8** |

---

# 74. Most Important Ablation Result: Stock Trend Decomposition

Removing Stock Trend Decomposition causes a dramatic collapse.

CSI300:

\[
IC:
0.117
\rightarrow
0.048.
\]

CSI800:

\[
0.118
\rightarrow
0.043.
\]

This is one of the strongest pieces of evidence in the paper.

The decomposition is not a minor helper module.

It is foundational to the reported performance.

---

# 75. Time Correlation Is Also Critical

Removing Time Correlation:

CSI300:

\[
IC:
0.117
\rightarrow
0.055.
\]

CSI800:

\[
0.118
\rightarrow
0.047.
\]

Thus, within-stock temporal correlation modeling is essential.

---

# 76. Stock Correlation Is Critical

Removing Stock Correlation:

CSI300:

\[
IC:
0.117
\rightarrow
0.049.
\]

CSI800:

\[
0.118
\rightarrow
0.061.
\]

Therefore:

> cross-sectional interaction is also fundamental.

---

# 77. Module Order Matters

The "Change order" variant produces:

CSI300:

\[
IC=0.054.
\]

CSI800:

\[
IC=0.052.
\]

This is far below the full model.

The result strongly supports the paper's sequencing:

\[
\text{within-stock time correlation}
\rightarrow
\text{cross-stock correlation}.
\]

---

# 78. Market Trend Guidance Has a Different Role

Removing MTG:

## CSI300

Only a small mean decline:

\[
0.117
\rightarrow
0.114.
\]

## CSI800

Much larger deterioration and much higher variance:

\[
0.118\pm0.007
\rightarrow
0.087\pm0.032.
\]

The paper interprets this as:

> global market trend guidance becomes more important when the stock universe contains more heterogeneous small/mid-cap stocks.

This is one of the most interesting cross-universe findings.

---

# 79. Market Trend Guidance May Be More About Stability Than Raw Mean

Especially on CSI800, w/o MTG produces enormous standard deviation increases.

Example:

\[
AR:
1.11\pm0.72
\]

versus:

\[
1.62\pm0.11.
\]

Thus MTG appears to contribute:

- mean performance;
- stability across seeds.

This supports the paper's "robustness" narrative.

---

# 80. RWKV vs. Transformer

Replacing RWKV with Transformer:

## CSI300

\[
IC=0.083,
\]

\[
RankIC=0.071.
\]

## CSI800

\[
IC=0.097,
\]

\[
RankIC=0.079.
\]

This is the strongest alternative Time Correlation baseline.

Transformer remains substantially better than many ablated versions, but below full MATCC.

---

# 81. RWKV vs. Mamba

Replacing RWKV with Mamba performs much worse:

CSI300:

\[
IC=0.052.
\]

CSI800:

\[
IC=0.046.
\]

According to the paper's implementation, RWKV is clearly superior.

---

# 82. Interpretation of RWKV Ablation

The evidence supports:

> RWKV is a strong temporal backbone in this architecture.

It does **not** fully establish:

> RWKV is universally superior to Transformer or Mamba for stock forecasting.

Why?

The paper states only that replacements use the same heads for comparison.

It does not provide a full compute-matched, tuning-budget-matched study for all alternative backbones.

The claim should remain architecture-specific.

---

# 83. Hyperparameter Sensitivity — DepthConv Kernel

The tested values are:

\[
k_c\in\{3,5,7\}.
\]

Paper interpretation:

## Too small

Market trend receptive field is too short.

The trend changes insufficiently.

## Too large

Important local market information can be blurred or lost.

Chosen:

\[
k_c=5.
\]

---

# 84. Hyperparameter Sensitivity — AvgPool Kernel

Tested:

\[
k_a\in\{3,5,7\}.
\]

Paper interpretation:

> large pooling kernels over-smooth stock features.

Chosen:

\[
k_a=3.
\]

This means the stock trend smoother is intentionally local.

---

# 85. Hyperparameter Sensitivity — RWKV Heads

Tested:

\[
N\in\{2,4,8\}.
\]

The paper reports that:

- too few heads underuse feature diversity;
- too many heads reduce effectiveness.

Chosen:

\[
N=4.
\]

---

# 86. Market Noise Sensitivity

The authors inject Gaussian noise into market features.

Noise distribution:

\[
\epsilon
\sim
\mathcal N(0,\gamma^2).
\]

Noise levels:

\[
\gamma\in\{1.0,2.0,3.0\}.
\]

The market representation is normalized, with most values roughly in:

\[
[-3,3].
\]

Therefore these noise levels are substantial relative to feature scale.

---

# 87. Noise Experiment Result

The paper reports that MATCC continues outperforming under added market noise.

This is used to support robustness under volatile or extreme market conditions.

The paper provides plots rather than a detailed numerical table.

Therefore a knowledge base should preserve the conclusion qualitatively and not invent exact values.

---

# 88. Limits of the Noise Robustness Test

The experiment perturbs:

> market features.

It does not test:

- stock-feature noise;
- missing stocks;
- structural breaks;
- delayed data;
- adversarial microstructure distortions.

Additionally, Gaussian perturbation is only one stylized form of market noise.

Therefore:

> this is evidence for robustness to market-feature perturbation, not general robustness to every financial distribution shift.

---

# 89. Portfolio Visualization — Normal Market Environment

Figure 4 compares cumulative returns during a normal market environment.

The paper reports that MATCC achieves the highest cumulative return for almost the entire displayed interval.

The figure is presented for:

- CSI300;
- CSI800.

Exact period-level numerical portfolio metrics are not separately tabulated here.

---

# 90. Portfolio Visualization — Extreme Market Environment

Figure 5 presents an extreme downturn environment.

The paper reports:

- several baselines experience substantial negative returns;
- MATCC remains positive for much of the downturn;
- the effect is especially clear on CSI800.

This is used as qualitative evidence of robustness.

---

# 91. Time Correlation Attention Visualization

The paper selects:

> CMCC (SH600941)

as an example stock.

Three time periods are visualized.

The attention map axes are:

- x-axis: stock features;
- y-axis: lookback time steps.

The paper highlights two patterns:

## Pattern 1

Some feature regions weaken and then strengthen through time.

Interpretation:

> changing feature trend.

## Pattern 2

Other feature regions remain consistently high.

Interpretation:

> persistent trend.

---

# 92. What the WKV Visualization Supports

The attention map suggests that RWKV's Time Correlation module responds differently to:

- temporally changing features;
- stable features.

This is a useful qualitative mechanism visualization.

However, it does not directly quantify:

- lead–lag prediction gain;
- causal relationships;
- which Alpha158 factors drive returns.

---

# 93. Stock Correlation Attention Visualization

The Stock Correlation map is shown for CSI800.

Each attention cell represents:

> influence from stock \(j\) to stock \(i\).

The authors define an influence score:

\[
f(u)
=
\sum_i S_{iu}.
\]

Stocks are reordered by this score.

The visualization shows concentrated columns associated with highly influential stocks.

---

# 94. Interpretation of Dynamic Stock Attention

The figure supports the claim that:

> learned inter-stock influence is heterogeneous rather than uniform.

A few stocks may exert broad market influence according to the learned attention matrix.

The paper uses this as evidence that dynamic stock correlations are meaningful.

---

# 95. Attention-Interpretability Caveat

Attention magnitude is not automatically equivalent to causal economic influence.

The visualization is useful for model inspection.

But stronger interpretability would require:

- event analysis;
- Granger / lead–lag validation;
- sector / industry alignment;
- causal intervention;
- stability across seeds.

The paper does not perform these analyses.

---

# 96. Inductive Biases

MATCC contains several explicit inductive assumptions.

## 96.1 Market Trend Is Shared

Every stock should be influenced by a common market environment.

## 96.2 Trend and Fluctuation Are Distinct

A useful stock representation can be decomposed into:

\[
\text{smooth trend}
+
\text{short-term fluctuation}.
\]

## 96.3 Temporal Structure Should Precede Cross-Stock Mixing

Within-stock temporal dynamics should be learned before information is mixed across assets.

## 96.4 Stock Relations Are Dynamic

Correlations should be learned through attention rather than fixed concept graphs.

## 96.5 Recent State Determines Historical Relevance

The final time-step representation is an appropriate query for selecting useful historical states.

---

# 97. Why MATCC May Work

Separate paper evidence from interpretation.

## Evidence A — STD removal collapses performance

Strong evidence for trend/fluctuation decomposition.

## Evidence B — TC and SC removal collapse performance

Strong evidence for the two-stage correlation pipeline.

## Evidence C — Changing TC/SC order hurts heavily

Strong evidence for the ordering hypothesis.

## Evidence D — Transformer is weaker than RWKV

Supports the selected temporal backbone.

## Evidence E — MTG improves CSI800 stability

Supports explicit market guidance under a more heterogeneous universe.

---

# 98. Plausible Mechanism: Decomposition as Denoising

The trend:

\[
AvgPool(\hat x)
\]

filters local fluctuations.

The residual:

\[
\hat x-\hat x^t
\]

preserves short-term information.

Because both are retained, the model gets separate channels for:

- slowly varying structure;
- rapid deviation.

This may simplify later temporal modeling.

---

# 99. Plausible Mechanism: Time-First Avoids Cross-Stock Trend Contamination

Suppose two stocks have different underlying trends.

If cross-stock mixing occurs before each stock's own temporal dynamics are encoded, the model may mix incompatible local trajectories.

MATCC instead first builds:

\[
h_{u,t}.
\]

Then stocks communicate.

This preserves stock-local temporal identity before cross-sectional mixing.

The strong "Change order" ablation supports this explanation.

---

# 100. Plausible Mechanism: RWKV Provides Decayed Long Memory

The WKV operator explicitly includes time decay:

\[
e^{-(t-1-i)w}.
\]

This gives an inductive bias that older observations gradually lose influence.

In finance, this is intuitively compatible with:

- recency;
- regime drift;
- decaying predictive relevance.

---

# 101. Potential Failure Modes

These are analytical concerns beyond direct author claims.

## 101.1 Market Indices May Be Incomplete

Three index series may not summarize:

- sector dispersion;
- volatility surface;
- rates;
- credit;
- macro surprises.

## 101.2 Direct Additive Market Guidance May Be Too Rigid

The model uses:

\[
stock+market.
\]

Some stocks may react oppositely or nonlinearly to the same market trend.

## 101.3 Average Pooling Has Fixed Trend Scale

The chosen:

\[
k_a
\]

defines one smoothing scale.

Different stocks may have different trend horizons.

## 101.4 Fixed Market Trend Scale

One:

\[
k_c
\]

may not capture multi-scale market dynamics.

## 101.5 Cross-Stock Attention Remains Time-Aligned

The model does not explicitly attend from:

\[
stock\ u,\ time\ t
\]

to:

\[
stock\ v,\ time\ t'.
\]

Cross-time cross-stock dependence is indirect.

## 101.6 No Explicit Regime Switch

Market trend is continuous, but there is no regime classifier/router.

## 101.7 Expensive Cross-Stock Attention

Self-attention across:

\[
S
\]

stocks is approximately quadratic in the number of stocks per time step.

This matters for large universes.

---

# 102. Evidence Strength

## Strong aspects

- three markets / universes;
- long 3.5-year test;
- five seeds;
- strong baselines;
- full ablation table;
- backbone replacement study;
- sensitivity analysis;
- market-noise perturbation;
- attention visualizations.

## Weak / missing aspects

- no transaction costs;
- no runtime / parameter comparison;
- no formal statistical significance testing;
- ablations only on Chinese markets;
- no realistic structural-noise stress test;
- no direct lead–lag quantitative evaluation;
- no explicit S&P500 ablation;
- no broader market-universe evaluation in the paper body.

Overall:

> experimental breadth is strong, but reported portfolio metrics should be interpreted with caution because costs are omitted and some gains are unusually large.

---

# 103. Baseline Fairness

Positive points:

- baselines follow MASTER's reported hyperparameter setup;
- all methods are repeated five times;
- common data preprocessing is used;
- common Qlib platform.

Potential concerns:

## Baseline recency

No FactorVAE / FactorVQVAE / XGB-style latent-factor baselines are included.

This paper focuses more on direct forecasting architectures.

## Tuning budget

MATCC receives an explicit hyperparameter grid.

The paper does not state equivalent tuning effort for every baseline.

## Additional information

MATCC uses explicit 63-dimensional market-index features.

MASTER and DTML also use market information conceptually, but other baselines may not receive identical market-side information.

---

# 104. Reproducibility Strength

The paper provides:

- code URL;
- exact split;
- feature family;
- market-feature sources;
- \(T\);
- \(d\);
- \(d'\);
- normalization;
- extreme-label preprocessing;
- optimizer;
- LR range;
- warm-up;
- restart interval;
- epochs;
- hardware;
- random seed count;
- hyperparameter search.

This makes MATCC comparatively reproducible.

---

# 105. Important Missing Reproduction Details

The paper text does not fully specify:

- batch size;
- exact Adam betas;
- weight decay;
- dropout;
- exact RWKV version / implementation settings;
- number of RWKV layers;
- number of stock-attention layers;
- exact FFN hidden width;
- exact market-status feature construction formula producing all 63 dimensions;
- exact `NormS` implementation;
- exact extreme-label threshold scope;
- transaction-cost configuration, since costs are omitted;
- exact random seeds;
- exact runtime / parameter count.

Code should be treated as the source of truth for exact reproduction.

---

# 106. Known / Potential Paper-Specification Oddities

## Oddity 1 — Robust Z-Score Equation

Eq. 11 includes:

\[
|x-MED(X)|.
\]

This removes sign information from normalized deviations if implemented literally.

Verify code.

## Oddity 2 — "100% / 120% Ranking Improvement" Language

The paper uses broad improvement percentages that do not match every ranking metric equally.

Use actual table values.

## Oddity 3 — Huge Portfolio AR / IR

Reported AR and IR are extremely large, especially on CSI800.

Since trading cost is explicitly excluded, these should not be compared directly with cost-aware backtests.

---

# 107. Novelty Decomposition

## Existing Components

- market-index features;
- time-series decomposition;
- average pooling;
- RWKV;
- multi-head stock attention;
- temporal attention aggregation;
- MSE regression.

## Main Finance-Specific Innovations

### Market Trend Guidance

Use explicit market indices to generate shared trend context.

### Stock Trend / Fluctuation Separation

Decompose market-guided stock features before correlation modeling.

### Ordered Time → Stock Correlation

Preserve temporal sequence and model stock-local time correlation first.

### RWKV for Stock Forecasting

Use RWKV as the main time-correlation engine inside a stock-correlation pipeline.

---

# 108. Mechanism Primitives

For a research Agent:

## Primitive A — Explicit Market Context

\[
m_{1:T}
\rightarrow
m_t.
\]

## Primitive B — Shared Market Injection

\[
x_u+m_t.
\]

## Primitive C — Trend / Residual Decomposition

\[
x
\rightarrow
x^t+x^f.
\]

## Primitive D — Time-Decay Attention

RWKV:

\[
history
\rightarrow
wkv_t.
\]

## Primitive E — Per-Time Cross-Stock Attention

\[
\{h_{u,t}\}_{u\in S}
\rightarrow
\{z_{u,t}\}_{u\in S}.
\]

## Primitive F — Current-State Temporal Query

\[
z_{u,T}
\rightarrow
attention(z_{u,1:T}).
\]

---

# 109. What the Ablations Say Is Most Important

A rough evidence-based ordering is:

\[
\text{Trend Decomposition}
\approx
\text{Time Correlation}
\approx
\text{Stock Correlation}
>
\text{Module Ordering}
>
\text{MTG mean effect on CSI300}.
\]

But on CSI800:

> Market Trend Guidance becomes much more important for stability.

This distinction is valuable.

---

# 110. How to Beat This Baseline

## 110.1 What MATCC Already Does Well

- explicit market information;
- trend/fluctuation separation;
- strong temporal backbone;
- dynamic cross-stock relationships;
- preservation of time-step-level representations;
- long test window;
- multi-seed reporting;
- broad ablation coverage.

---

## 110.2 Structural Weaknesses

### Single-Scale Trend Decomposition

Only one AvgPool scale is used.

### Single-Scale Market Trend

Depthwise convolution uses one selected kernel.

### Additive Market Injection

No stock-specific market response gate.

### Time-Aligned Stock Attention

Cross-stock attention is performed at the same enriched time step.

### No Dynamic Relation Routing

Every stock participates in dense attention.

### No Explicit Market Regime

Market state is represented continuously without discrete regime specialization.

### MSE-Only Training

Training objective is not directly aligned with:

- RankIC;
- portfolio return.

---

# 111. Safe Improvement Directions

Incremental extensions:

- multi-scale AvgPool;
- multi-scale market convolution;
- ranking loss;
- market trend gate;
- sparse stock attention;
- stronger temporal aggregation;
- cost-aware portfolio objective.

These may improve MATCC but may not constitute large novelty alone.

---

# 112. Higher-Novelty Directions

## Dynamic Market Regime Routing

Use a regime encoder to decide how market trend influences different stocks.

## Multi-Scale Trend Experts

Different experts model:

- short trend;
- medium trend;
- long trend;
- fluctuation.

## Explicit Cross-Time Cross-Stock Attention

Attend directly between:

\[
(u,t)
\leftrightarrow
(v,t').
\]

This goes beyond MATCC's time-first mediated design.

## Sparse Lead–Lag Graph

Learn directed temporal relationships:

\[
u_t
\rightarrow
v_{t+\tau}.
\]

## Regime-Specific Correlation Experts

Different stock-correlation networks for different market conditions.

## Trend / Noise Uncertainty Modeling

Estimate confidence in:

- trend component;
- fluctuation component.

---

# 113. Research Opportunity: Multi-Scale Decomposition

MATCC uses one:

\[
k_a.
\]

But financial signals often operate at multiple scales.

Possible model:

\[
x
=
trend^{(short)}
+
trend^{(medium)}
+
trend^{(long)}
+
residual.
\]

Then route different prediction heads to different scales.

This is a natural extension of MATCC's strongest ablated module.

---

# 114. Research Opportunity: Adaptive Market Guidance

MATCC uses:

\[
\hat x_u
=
W_\theta x_u+m_t^T.
\]

A richer form could be:

\[
\hat x_u
=
W_\theta x_u
+
g_u(m_t,x_u)
\odot
m_t,
\]

where:

\[
g_u
\]

is stock-specific.

Hypothesis:

> different stocks respond differently to the same market trend.

---

# 115. Research Opportunity: Direct Lead–Lag Correlation

The paper's motivating example emphasizes cross-time stock dependence.

A stronger explicit architecture could learn:

\[
A_{u,v,\tau}
\]

representing the influence of stock \(u\) on stock \(v\) after lag \(\tau\).

This would provide:

- more direct lead–lag structure;
- stronger interpretability;
- potentially sparse dynamic graphs.

---

# 116. Research Opportunity: Ranking-Aware MATCC

MATCC optimizes:

\[
MSE.
\]

But evaluates:

- IC;
- RankIC;
- AR;
- IR.

A natural experiment is:

\[
\mathcal L
=
\mathcal L_{\text{MSE}}
+
\lambda
\mathcal L_{\text{rank}}.
\]

This directly tests whether objective mismatch limits MATCC.

---

# 117. Research Opportunity: Dynamic Relation Sparsity

Dense stock attention is:

\[
O(S^2T).
\]

For large universes, use:

- top-k relation selection;
- learned sparse attention;
- graph routing;
- clustered attention;
- sector-aware attention.

This could improve both:

- scalability;
- interpretability.

---

# 118. Experiment Hooks

## E1 — Multi-Scale Trend

Compare:

- single \(k_a\);
- multi-scale \(\{3,5,7\}\);
- adaptive learned smoothing.

Metrics:

- IC;
- RankIC;
- IC decay;
- subperiod stability.

---

## E2 — Explicit Cross-Time Stock Attention

Baseline:

\[
time\rightarrow stock.
\]

New:

\[
(u,t)\leftrightarrow(v,t').
\]

Ablate:

- same-time only;
- fixed lags;
- learned lags;
- full cross-time sparse attention.

---

## E3 — Regime-Conditioned MTG

Test whether market trend guidance becomes more valuable in:

- high volatility;
- bear markets;
- heterogeneous universes.

This directly follows the CSI300 vs CSI800 ablation difference.

---

## E4 — Ranking Objective

Compare MSE with:

- pairwise ranking;
- listwise ranking;
- IC surrogate;
- hybrid loss.

---

## E5 — Cost-Aware Backtest

Re-run top30-drop30 with:

- commissions;
- stamp duty;
- slippage;
- turnover constraints.

This tests whether the massive AR/IR survives realistic execution.

---

# 119. Agent Critique: Claim vs. Evidence

## Claim A

Explicit market trends improve prediction.

### Evidence

Removing MTG hurts CSI800 strongly and increases variance.

### Assessment

Well supported for heterogeneous Chinese universe; weaker mean effect on CSI300.

---

## Claim B

Stock trend decomposition improves robustness and prediction.

### Evidence

Removing STD causes one of the largest performance collapses.

### Assessment

Very strongly supported within the architecture.

---

## Claim C

Cross-time correlation modeling is important.

### Evidence

- removing TC hurts;
- changing TC/SC order hurts;
- replacing RWKV with Transformer or Mamba hurts.

### Assessment

Strong evidence that the proposed temporal pipeline matters.

---

## Claim D

RWKV is superior for this task.

### Evidence

Better than tested Transformer and Mamba replacements.

### Assessment

Supported within this model and tuning setup, but not a universal backbone theorem.

---

## Claim E

MATCC is robust to market noise.

### Evidence

Gaussian perturbation of market features and cumulative-return visualizations under extreme market periods.

### Assessment

Useful but stylized robustness evidence.

---

# 120. Strengths

## Architecture

- every module has a clear finance/time-series motivation;
- module ordering is explicitly justified and ablated;
- market and stock trends are separated from correlation modeling.

## Experiments

- 3 datasets;
- 5 seeds;
- ranking + portfolio metrics;
- broad ablations;
- backbone replacements;
- noise analysis;
- visual interpretation.

## Reproducibility

- code released;
- preprocessing described;
- split and hyperparameters clearly reported.

## Narrative

The problem → module → ablation mapping is unusually clean.

---

# 121. Limitations Explicitly Supported by the Paper

The paper does not present a long limitations section.

However, the Discussion / Conclusion explicitly indicate:

## Market-Specific Modeling Challenge

Different countries have different:

- trading mechanisms;
- market rules.

The authors state that fully capturing market information may require finer market-specific modeling.

## Limited Market Coverage in the Paper

Experiments focus on:

- China;
- United States.

The authors mention broader market tests as future / repository work.

## Future Volatility Reduction

The paper states an intention to explore additional ways to reduce volatility in:

- models;
- data.

Diffusion-based approaches are explicitly mentioned.

---

# 122. Agent-Identified Limitations

These are analytical extensions, not direct author claims.

## No Transaction Cost

Portfolio results are cost-free.

## Extremely Large Portfolio Metrics

Practical robustness requires realistic execution tests.

## Target / Preprocessing Complexity

Robust input normalization + extreme-label dropping + cross-sectional target normalization can make comparisons with other papers difficult.

## Market Features Are Hand-Specified

The market context depends on selected indices and summary statistics.

## Dense Stock Attention

Scalability may be difficult beyond CSI800.

## No Explicit Statistical Significance

Reported mean ± std across seeds is useful, but no formal paired significance test is provided.

## Only Chinese Ablations

S&P500 module dependence is not established.

---

# 123. Future Work

## Explicit Author Future Work

The paper proposes exploring strategies to reduce volatility in:

- model behavior;
- data.

It specifically mentions:

> diffusion-based approaches.

---

## Additional Implied Directions

Based on the design:

- adaptive trend scales;
- adaptive market-state encoding;
- direct lead–lag stock graphs;
- cost-aware portfolio optimization;
- market-specific modules;
- broader international testing;
- sparse correlation structures.

These are inferred directions and should not be attributed directly to the authors.

---

# 124. Writing / Presentation Lessons

## 124.1 Two Clear Deficiencies

The introduction is organized around exactly two shortcomings:

1. insufficient explicit market trend modeling;
2. insufficient cross-time stock-correlation modeling.

This makes the motivation easy to follow.

---

## 124.2 One Main Module per Problem

Problem:

> noisy stock data hide market trend.

Solution:

> Market-Stock Trend Guidance.

Problem:

> temporal aggregation blurs cross-time information.

Solution:

> preserve time sequence + Time Correlation + Stock Correlation.

This creates strong module-function correspondence.

---

## 124.3 Ablations Mirror the Method Section

The paper has:

- MTG;
- STD;
- TC;
- SC.

The ablation table removes exactly these blocks.

This makes the causal story much cleaner.

---

## 124.4 Replacement Ablations Are Stronger Than Removal Alone

Rather than only remove RWKV, the paper also compares:

- Transformer;
- Mamba.

This helps show that the temporal result is not only due to "having any sequence model."

---

## 124.5 Visualization Is Mechanism-Oriented

Visualizations target:

- temporal attention;
- stock influence.

They are linked directly to the paper's two main claims.

---

# 125. Relationship to FactorVAE / FactorVQVAE

MATCC belongs to a different baseline family.

| Dimension | FactorVAE | FactorVQVAE | MATCC |
|---|---|---|---|
| Main paradigm | latent factor | discrete latent factor | direct representation prediction |
| Latent factor model | yes | yes | no explicit latent factor variable |
| Market info | cross-section-derived | explicit index features | explicit index features |
| Temporal backbone | GRU | GRU + AR Transformer | RWKV |
| Cross-stock modeling | global factor aggregation | factor model | stock self-attention |
| Trend decomposition | no | no | yes |
| VQ | no | yes | no |
| Prediction loss | likelihood + KL | token + ranking | MSE |
| Main novelty | posterior guidance | discrete factor tokens | trend + cross-time correlation |
| Markets | China | CSI300 + S&P500 | CSI300 + CSI800 + S&P500 |

---

# 126. Relationship to MASTER

| Dimension | MASTER | MATCC |
|---|---|---|
| Market information | yes | yes |
| Market role | feature guidance / selection | explicit market trend added to stock sequence |
| Temporal model | Transformer | RWKV |
| Trend decomposition | no core decomposition | explicit trend + fluctuation |
| Stock correlation | Transformer-style | MHA at each time step |
| Temporal aggregation | Transformer structure | final temporal attention |
| Lookback | paper-specific MASTER setup | \(T=8\) |
| Objective | baseline-specific | MSE |
| Key MATCC claim | — | time-first + trend decomposition improves robustness |

---

# 127. Local Reproduction

Fill this with local experiment results.

## Paper-Reported — CSI300

\[
IC=0.117\pm0.007
\]

\[
ICIR=1.02\pm0.06
\]

\[
RankIC=0.086\pm0.004
\]

\[
RankICIR=0.87\pm0.06
\]

\[
AR=0.80\pm0.08
\]

\[
IR=8.5\pm0.8.
\]

---

## Paper-Reported — CSI800

\[
IC=0.118\pm0.007
\]

\[
RankIC=0.083\pm0.004.
\]

---

## Paper-Reported — S&P500

\[
IC=0.071\pm0.001
\]

\[
RankIC=0.035\pm0.001.
\]

---

## Our Reproduction

### CSI300

- train:
- validation:
- test:
- features:
- market information:
- seeds:
- IC:
- ICIR:
- RankIC:
- RankICIR:
- AR:
- MDD:
- Sharpe:
- other metrics:

### CSI800

- TBD.

### S&P500

- TBD.

---

## Known Protocol Differences

- TBD.

## Possible Causes of Reproduction Gap

- label convention;
- Robust Z-score implementation;
- extreme-label filtering;
- index-feature construction;
- Qlib version;
- constituent universe;
- RWKV implementation;
- baseline preprocessing;
- no-cost vs cost-aware backtest.

---

# 128. Implementation Hooks

Suggested code organization:

```text
matcc/
├── market_status.py
├── market_trend.py
├── stock_decomposition.py
├── rwkv_time_correlation.py
├── stock_attention.py
├── temporal_aggregation.py
├── predictor.py
├── model.py
└── preprocessing.py
```

---

# 129. Minimal Module Interfaces

## Market Trend

```text
Input:
market sequence: [T, F_market]

Output:
market trend: [T, D]
```

## Stock Projection

```text
Input:
stock x: [S, T, F_stock]

Output:
projected stock: [S, T, D]
```

## Trend Decomposition

```text
Input:
guided stock: [S, T, D]

Output:
trend: [S, T, D]
fluctuation: [S, T, D]
recombined: [S, T, D]
```

## Time Correlation

```text
Input:
X: [S, T, D]

Output:
H: [S, T, D]
```

## Stock Correlation

```text
for t in 1..T:
    Input: H[:, t, :] = [S, D]
    Output: Z[:, t, :] = [S, D]
```

## Temporal Aggregation

```text
Input:
Z: [S, T, D]

Output:
E: [S, D]
```

## Predictor

```text
Input:
E: [S, D]

Output:
return_score: [S]
```

---

# 130. Conceptual Training Pseudocode

```text
for prediction_date in training_dates:

    stock_x = load_stock_features(date)
    market_x = load_market_features(date)

    market_status = build_market_status(market_x)
    market_trend = depthwise_conv(market_status)

    stock_proj = linear(stock_x)
    guided_stock = stock_proj + market_trend

    trend = avg_pool(guided_stock)
    fluctuation = guided_stock - trend

    X = linear_trend(trend) + linear_fluctuation(fluctuation)

    H = RWKV_time_correlation(X)

    for t in range(T):
        Z[:, t] = stock_self_attention(H[:, t])

    E = temporal_attention(
        query=Z[:, -1],
        keys=Z,
        values=Z
    )

    pred = linear_head(E)

    loss = MSE(pred, normalized_labels)

    update(loss)
```

---

# 131. Reproduction Checklist

## Data

- Alpha158 stock features;
- 63 market features;
- same index sources.

## Dates

- 2008Q1–2020Q1 train;
- 2020Q2 validation;
- 2020Q3–2023Q4 test.

## Label

- exact \(c_{T+d}\) / \(c_{T+1}\) formula;
- \(d=5\);
- daily cross-sectional normalization.

## Preprocessing

- Robust Z-score;
- verify Eq. 11 sign / absolute value;
- DropExtremeLabel 2.5% tails.

## Architecture

- D=256;
- kc=5;
- ka=3;
- RWKV heads=4;
- time → stock order.

## Training

- Adam;
- CosineAnnealingLR;
- warm-up 10;
- max LR 3e-3;
- min LR 2e-4;
- restart interval 15;
- total 70 epochs;
- final epoch used.

## Evaluation

- 5 seeds;
- IC/ICIR;
- RankIC/RankICIR;
- no-cost top30-drop30 AR/IR.

---

# 132. Agent Research Instructions

When using MATCC:

1. **Preserve the exact label convention and target normalization before comparing metrics.**
2. **Verify the unusual Robust Z-score equation against code.**
3. **Treat Stock Trend Decomposition as a core module, not an auxiliary feature.**
4. **Preserve Time Correlation → Stock Correlation ordering unless intentionally ablating it.**
5. **Do not interpret the stock-attention block as direct arbitrary cross-time attention; cross-time interaction is mediated through RWKV states.**
6. **Remember that portfolio results exclude transaction costs.**
7. **Do not compare MATCC's AR/IR directly to cost-aware Qlib backtests.**
8. **When claiming RWKV superiority, state that the evidence is within the paper's architecture and tuning setup.**
9. **Use CSI800 results to study the stabilizing value of market trend guidance.**
10. **If building on MATCC, target multi-scale trend modeling, adaptive market guidance, sparse cross-time stock interaction, or ranking-aligned objectives rather than only changing the backbone.**
11. **For a stronger modern study, report subperiod/regime metrics rather than only aggregate 2020–2023 performance.**
12. **Use the released code for exact preprocessing and RWKV implementation details.**

---

# 133. Agent Takeaways

## Most Important Research Problem

How to improve cross-sectional stock prediction by explicitly modeling:

- market trend;
- stock trend/fluctuation;
- inter-temporal stock behavior;
- dynamic stock correlation.

## Most Important Design Principle

**Do not collapse time before modeling stock relationships.**

## Most Important Module

According to ablation, Stock Trend Decomposition is one of the strongest contributors.

## Most Important Correlation Result

Both:

\[
\text{Time Correlation}
\]

and:

\[
\text{Stock Correlation}
\]

are essential.

Their order is also essential.

## Most Important Backbone Result

RWKV outperforms tested Transformer and Mamba replacements inside MATCC.

## Most Important Market-Information Insight

MTG contributes modestly to CSI300 mean performance but strongly improves CSI800 performance and stability.

## Biggest Evaluation Caveat

Portfolio AR / IR are reported **without transaction cost**.

## Biggest Reproduction Caveat

The Robust Z-score equation contains an unusual absolute value and should be verified against code.

## Best Research Direction

Build a more explicit, adaptive, multi-scale version of MATCC's:

\[
\text{trend}
+
\text{cross-time relation}
+
\text{market regime}
\]

pipeline.

---

# 134. Compact Architecture Summary

```text
                     MATCC
================================================

Market Indices
     │
     ▼
Market Status Features
(current + multi-window statistics)
     │
     ▼
Depthwise Conv
     │
     ▼
Market Trend
     │
     ├──────────────────────────┐
     │                          │
Alpha158 Stock Sequence         │
     │                          │
     ▼                          │
Linear Projection               │
     └────────── + ─────────────┘
                │
                ▼
        Market-Guided Stock Sequence
                │
                ▼
        AvgPool Trend Decomposition
          ┌─────┴──────┐
          ▼            ▼
        Trend      Fluctuation
          │            │
          └── Linear ──┘
                │
                ▼
          Recombined Sequence
                │
                ▼
        RWKV Time Correlation
                │
                ▼
        Sequence h[u,t]
                │
                ▼
   Cross-Stock Self-Attention at each t
                │
                ▼
        Sequence z[u,t]
                │
                ▼
 Last-Step-Query Temporal Attention
                │
                ▼
          Final Stock Embedding
                │
                ▼
          Linear Prediction Head
                │
                ▼
   Cross-Sectional Normalized Return
```

---

# 135. Source Location Map

Useful paper locations:

- **pp. 1–2:** motivation, two limitations, contributions, related work.
- **p. 3:** architecture overview, market status, market trend guidance, stock trend decomposition.
- **p. 4:** RWKV Time Correlation, WKV equations, stock correlation, temporal aggregation.
- **p. 5:** prediction objective, datasets, split, preprocessing, baselines, evaluation setup.
- **pp. 6–7:** main performance tables, ablation tables, implementation details, model-study discussion.
- **p. 8:** hyperparameter sensitivity, Gaussian market-noise experiment, normal-market portfolio figure.
- **p. 9:** extreme-market portfolio figure, time-correlation and stock-correlation visualizations, discussion, future work.

---

# 136. Compact Retrieval Summary

MATCC (Cao et al., CIKM 2024) is a direct cross-sectional stock-return predictor designed around explicit market trends and cross-time stock correlations. It uses Qlib Alpha158 stock features plus 63 index-derived market features. A Market-Stock Trend Guidance module extracts a market trend with depthwise 1D convolution, adds that trend to every stock representation, and decomposes each stock sequence into average-pooled trend and fluctuation components. The recombined sequence is passed through RWKV to model within-stock temporal correlations while retaining a representation at every lookback step. At each time step, multi-head self-attention then models dynamic cross-stock relations. A final temporal attention layer uses the last-day embedding as a query to aggregate historical representations, and a linear head predicts a daily cross-sectionally normalized future return. The paper uses \(T=8\), prediction interval \(d=5\), and fixed train/validation/test periods of 2008Q1–2020Q1 / 2020Q2 / 2020Q3–2023Q4 across CSI300, CSI800, and S&P500. MATCC reports very large improvements over MASTER and other baselines, including CSI300 IC/RankIC of 0.117/0.086 and CSI800 0.118/0.083. Ablations show that stock trend decomposition, RWKV time correlation, stock correlation, and the order of time-before-stock correlation are all critical; market trend guidance has a smaller mean effect on CSI300 but strongly improves CSI800 stability. The model also remains strong under Gaussian perturbations to market features. Important caveats are that portfolio AR/IR are reported without transaction costs, the paper's Robust Z-score equation contains an unusual absolute-value operation that should be verified in code, and the cross-stock attention itself is time-aligned rather than an explicit arbitrary cross-time stock-to-stock attention mechanism.
