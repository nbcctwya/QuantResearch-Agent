---
paper_id: FactorVAE_2022
title: "FactorVAE: A Probabilistic Dynamic Factor Model Based on Variational Autoencoder for Predicting Cross-Sectional Stock Returns"
authors:
  - Yitong Duan
  - Lei Wang
  - Qizhong Zhang
  - Jian Li
venue: "AAAI-22 / The Thirty-Sixth AAAI Conference on Artificial Intelligence"
year: 2022
paper_type: baseline
subtype:
  - finance_ml
  - dynamic_factor_model
  - probabilistic_model
  - variational_autoencoder
task:
  - cross_sectional_stock_return_prediction
  - dynamic_factor_learning
  - risk_estimation
market: "China A-shares"
dataset:
  source: "Yahoo Finance + Qlib Alpha158"
  feature_set: "20 Qlib-selected features from Alpha158 for the time-series model"
  sequence_length: 20
  train: "2010-01-01 to 2017-12-31"
  validation: "2018-01-01 to 2018-12-31"
  test: "2019-01-01 to 2020-12-31"
core_modules:
  - GRU_feature_extractor
  - posterior_factor_encoder
  - dynamic_portfolio_layer
  - probabilistic_factor_decoder
  - multi_head_global_attention_factor_predictor
  - prior_posterior_KL_learning
core_mechanisms:
  - future_guided_posterior_factor_learning
  - probabilistic_latent_factors
  - dynamic_factor_exposure
  - cross_sectional_global_attention
  - risk_estimation_from_predictive_distribution
main_metrics:
  - RankIC
  - RankICIR
  - Annualized_Return
  - Sharpe_Ratio
  - Maximum_Drawdown
main_result:
  rank_ic: "0.055 ± 0.004"
  rank_icir: "0.568 ± 0.044"
  backtest_AR: "15.32% excess annualized return"
  backtest_SR: "1.92"
  backtest_MDD: "4.47%"
code_url_in_paper: "Qlib links are provided; FactorVAE implementation URL is not specified in the paper text provided"
relevance_to_agent:
  baseline_understanding: very_high
  reproducibility: medium_low
  architecture_transfer: high
  finance_specific_adaptation: very_high
  innovation_reference: high
priority: high
---

# FactorVAE: A Probabilistic Dynamic Factor Model Based on Variational Autoencoder for Predicting Cross-Sectional Stock Returns

## 0. Document Purpose

This document is a high-density, Agent-oriented reconstruction of **FactorVAE (Duan et al., AAAI 2022)**.

Unlike a short literature note, this Markdown is designed so that a research Agent can use the paper as:

1. a **baseline specification**;
2. an **implementation reference**;
3. a **mechanism library**;
4. a **source of ablation hypotheses**;
5. a **comparison target for new cross-sectional stock models**;
6. a **case study in finance-specific adaptation of VAE / latent-factor modeling**.

The paper's central idea is:

> Use future returns during training to infer a posterior distribution over "optimal" latent market factors, then train a historical-data-only factor predictor to approximate that posterior. At inference time, remove all future-data modules and use only the learned prior factor predictor plus a dynamic factor decoder.

The paper therefore has a clear train/test asymmetry:

- **Training:** historical characteristics + future returns are used.
- **Prediction:** historical characteristics only are used.

The future information is intentionally used as a **training-only oracle signal**, not as a test-time input.

This distinction is essential for correctly reproducing and evaluating the model.

---

# 1. Metadata

- **Title:** FactorVAE: A Probabilistic Dynamic Factor Model Based on Variational Autoencoder for Predicting Cross-Sectional Stock Returns
- **Authors:** Yitong Duan, Lei Wang, Qizhong Zhang, Jian Li
- **Affiliation:** Tsinghua University
- **Venue:** The Thirty-Sixth AAAI Conference on Artificial Intelligence (AAAI-22)
- **Year:** 2022
- **Pages in proceedings:** 4468–4476
- **Primary task:** Cross-sectional stock return prediction
- **Secondary task:** Risk estimation from a predictive distribution
- **Model family:** Dynamic factor model + variational latent-variable model
- **Core training idea:** Prior-posterior learning
- **Primary market:** China A-shares

---

# 2. Research Problem

## 2.1 Financial Problem

The paper starts from the standard dynamic factor-model form:

\[
y_s
=
\alpha_s
+
\sum_{k=1}^{K}
\beta_s^{(k)} z_s^{(k)}
+
\epsilon_s,
\]

where cross-sectional stock returns are decomposed into:

- idiosyncratic return \(\alpha_s\);
- latent factors \(z_s\);
- factor exposures \(\beta_s\);
- idiosyncratic noise \(\epsilon_s\).

The practical goal is:

> Learn an effective dynamic factor model from historical stock characteristics so that future cross-sectional returns can be predicted.

---

## 2.2 Machine-Learning Problem

The paper argues that modern nonlinear ML factor models can automatically discover latent factors, but financial data have a low signal-to-noise ratio.

This creates a learning difficulty:

\[
\text{historical noisy observations}
\rightarrow
\text{weak / unstable learned latent factors}.
\]

The authors' central question is therefore:

> How can one guide latent-factor learning so that the learned factors are more effective despite noisy financial data?

---

# 3. Motivation

The paper's motivation can be written as the following logical chain:

### Step 1 — Dynamic factor models are useful

Dynamic factor exposure is preferable to a static exposure assumption because firm and market conditions vary over time.

### Step 2 — Hand-designed factors may introduce prior bias

Traditional factors are often motivated by empirical finance knowledge and practical experience.

The authors argue that:

\[
\text{human prior knowledge}
\neq
\text{true market-generating structure}
\]

in general.

### Step 3 — Data-driven nonlinear factor learning is attractive

Machine learning can automatically extract latent factors from rich market data.

### Step 4 — Financial data are highly noisy

The low signal-to-noise ratio can prevent end-to-end ML models from discovering useful latent factors.

### Step 5 — Future returns contain direct information about what factors would have explained the realized cross-section well

During training, realized future returns are available.

The paper uses them to construct an "oracle-like" posterior factor model.

### Step 6 — Learn a predictor that imitates this posterior factor distribution from historical information only

This leads to prior-posterior learning.

---

# 4. Main Contributions

The paper states three main contributions.

## Contribution 1 — Probabilistic Dynamic Factor Model

FactorVAE combines:

- dynamic factor modeling;
- variational autoencoder-style latent random variables.

The factors are treated as random variables rather than deterministic hidden vectors.

---

## Contribution 2 — Prior-Posterior Learning

The paper trains:

- a posterior factor encoder with access to future returns;
- a prior factor predictor that only observes historical information.

The prior factor distribution is pushed toward the posterior factor distribution via KL divergence.

This is the central learning mechanism of the paper.

---

## Contribution 3 — Joint Return Prediction and Risk Estimation

Because factors and alpha are probabilistic Gaussian variables, the decoder outputs a predictive return distribution.

The model therefore provides:

- expected return;
- predictive standard deviation.

The standard deviation is later used in a risk-adjusted TopK strategy.

---

# 5. Research Questions in the Experiments

The authors explicitly formulate three experimental research questions.

## RQ1

> Does prior-posterior learning guide model learning effectively?

## RQ2

> Is the model robust to stocks never observed during training?

## RQ3

> Can the model's risk estimate improve stock investment?

These three questions organize the empirical section.

---

# 6. Formal Problem Definition

At cross-sectional time step \(s\):

\[
x_s
\in
\mathbb{R}^{N_s \times T \times C}
\]

contains sequential historical characteristics for \(N_s\) stocks.

Definitions:

- \(N_s\): number of stocks available at cross-section \(s\);
- \(T\): historical sequence length;
- \(C\): number of characteristics/features.

Future cross-sectional returns are:

\[
y_s
\in
\mathbb{R}^{N_s}.
\]

The paper defines return as:

\[
y_s
=
\frac{price_{s+1}-price_s}{price_s}.
\]

In the experimental section, the daily label is written more specifically as:

\[
y_t^{(i)}
=
\frac{
p_{t+2}^{(i)}-p_{t+1}^{(i)}
}{
p_{t+1}^{(i)}
},
\]

where \(t\) is the prediction day and \(p_{t+1}^{(i)}\) is the next trading day's closing price.

The learned model is:

\[
\hat y_s
=
f(x_s;\Theta)
=
\alpha(x_s)
+
\beta(x_s)z(x_s).
\]

---

# 7. Dynamic Factor Interpretation

The prediction decomposes into:

\[
\hat y
=
\alpha
+
\beta z.
\]

Interpretation:

- \(\alpha\): stock-specific / idiosyncratic expected-return component;
- \(\beta\): time-varying factor exposure;
- \(z\): market-level latent factor realization.

This structure is important because FactorVAE is not merely a generic VAE applied to stocks.

It is explicitly constrained by the factor-model decomposition.

---

# 8. Standard VAE Background Used by the Paper

The paper reviews the standard VAE objective:

\[
\max_{\theta,\phi}
\left\{
\mathbb{E}_{z\sim q_\phi(z|x)}
[
\ln p_\theta(x|z)
]
-
KL[q_\phi(z|x)\|p(z)]
\right\}.
\]

Interpretation:

### Reconstruction term

\[
\mathbb{E}_{q_\phi}
[
\ln p_\theta(x|z)
]
\]

encourages \(z\) to contain information useful for reconstructing the observation.

### KL term

\[
KL[q_\phi(z|x)\|p(z)]
\]

regularizes the posterior toward a prior.

FactorVAE adapts this prior-posterior structure to dynamic financial factors.

---

# 9. Overall Architecture

The full model contains four major learnable components:

1. **Feature Extractor**
2. **Factor Encoder**
3. **Factor Predictor**
4. **Factor Decoder**

The training and prediction graphs are different.

---

## 9.1 Training-Time Path

Historical characteristics:

\[
x
\rightarrow
\phi_{\text{feat}}
\rightarrow
e
\]

Future returns plus latent stock features:

\[
(y,e)
\rightarrow
\phi_{\text{enc}}
\rightarrow
q_{\text{post}}(z|x,y)
\]

Historical latent features alone:

\[
e
\rightarrow
\phi_{\text{pred}}
\rightarrow
q_{\text{prior}}(z|x)
\]

Posterior factors are used to reconstruct future returns:

\[
z_{\text{post}}, e
\rightarrow
\phi_{\text{dec}}
\rightarrow
\hat y_{\text{rec}}.
\]

The KL divergence forces:

\[
q_{\text{post}}(z|x,y)
\approx
q_{\text{prior}}(z|x).
\]

---

## 9.2 Test-Time Path

The future-return encoder is removed.

Only:

\[
x
\rightarrow
\phi_{\text{feat}}
\rightarrow
e
\rightarrow
\phi_{\text{pred}}
\rightarrow
z_{\text{prior}}
\rightarrow
\phi_{\text{dec}}
\rightarrow
\hat y_{\text{pred}}
\]

is used.

Therefore, according to the paper's intended protocol, there is no future-return input at prediction time.

---

# 10. Figure 1: High-Level Concept

Figure 1 gives the simplest conceptual view:

- **Encoder:** future information → optimal/posterior factors;
- **Predictor:** historical information → predicted/prior factors;
- **Decoder:** factors → returns;
- predicted prior factors are trained to approximate posterior factors.

Agent interpretation:

> The posterior branch acts as a training-only oracle/teacher-like latent target, while the prior branch acts as the deployable student/predictor.

"Teacher-like" is an interpretation for mechanism understanding; the paper itself calls them posterior and prior factors.

---

# 11. Feature Extractor

The feature extractor maps each stock's historical feature sequence into a stock latent representation.

\[
e
=
\phi_{\text{feat}}(x).
\]

For stock \(i\) at time \(t\):

\[
h_{\text{proj}}^{(i,t)}
=
\operatorname{LeakyReLU}
\left(
W_{\text{proj}}x^{(i,t)}
+
b_{\text{proj}}
\right),
\]

followed by:

\[
h_{\text{gru}}^{(i,t)}
=
GRU
\left(
h_{\text{proj}}^{(i,t)},
h_{\text{gru}}^{(i,t-1)}
\right).
\]

The final feature is:

\[
e^{(i)}
=
h_{\text{gru}}^{(i,T)}.
\]

---

## 11.1 Tensor Shapes

Input:

\[
x
\in
\mathbb{R}^{N\times T\times C}.
\]

Per-stock latent representation:

\[
e^{(i)}
\in
\mathbb{R}^{H}.
\]

Cross-sectional latent feature matrix:

\[
e
\in
\mathbb{R}^{N\times H}.
\]

---

## 11.2 Role

The feature extractor performs the **temporal modeling**.

It converts the past \(T\)-step time series for each stock into one fixed-dimensional stock state.

Cross-sectional interactions are not modeled here; they enter later through:

- dynamic portfolio construction in the posterior encoder;
- global attention in the factor predictor.

---

# 12. Factor Encoder: Posterior Factor Model

The posterior factor encoder has access to:

- future cross-sectional returns \(y\);
- historical stock latent features \(e\).

It outputs a Gaussian posterior factor distribution:

\[
[\mu_{\text{post}},\sigma_{\text{post}}]
=
\phi_{\text{enc}}(y,e),
\]

\[
z_{\text{post}}
\sim
\mathcal N
\left(
\mu_{\text{post}},
\operatorname{diag}
(\sigma_{\text{post}}^2)
\right).
\]

Shapes:

\[
\mu_{\text{post}}
\in
\mathbb{R}^K,
\]

\[
\sigma_{\text{post}}
\in
\mathbb{R}^K.
\]

---

# 13. Why the Posterior Encoder Does Not Directly Consume All Returns

The cross-section contains:

- many stocks;
- a changing number of stocks across time.

Directly feeding raw future-return vector \(y\) into a fixed neural network would create:

- high input dimensionality;
- parameter growth;
- sensitivity to a changing stock universe.

The paper therefore inserts a **dynamic portfolio layer**.

---

# 14. Dynamic Portfolio Layer

For each portfolio \(j\), stock \(i\) receives a learned weight:

\[
a_p^{(i,j)}
=
\frac{
\exp
\left(
(W_p e^{(i)}+b_p)^{(j)}
\right)
}{
\sum_{i=1}^{N}
\exp
\left(
(W_p e^{(i)}+b_p)^{(j)}
\right)
}.
\]

The weights satisfy:

\[
\sum_{i=1}^{N}
a_p^{(i,j)}
=
1.
\]

Portfolio return \(j\) is:

\[
y_p^{(j)}
=
\sum_{i=1}^{N}
y^{(i)}
a_p^{(i,j)}.
\]

Thus:

\[
y_p
\in
\mathbb{R}^{M},
\]

where \(M\) is the number of learned portfolios.

---

# 15. Interpretation of the Dynamic Portfolio Layer

The dynamic portfolio layer performs a learned cross-sectional compression:

\[
\mathbb{R}^{N}
\rightarrow
\mathbb{R}^{M}.
\]

The portfolio weights depend on stock latent features \(e\), so they are not fixed portfolios.

They change with the cross-section.

The paper gives two main motivations:

1. reduce input dimensionality and avoid excessive parameters;
2. remain robust when the available stock universe changes.

This is one of the most finance-specific design choices in FactorVAE.

---

# 16. Posterior Factor Distribution Mapping

The learned portfolio returns are mapped into Gaussian factor parameters:

\[
\mu_{\text{post}}
=
W_{\text{post}\mu} y_p
+
b_{\text{post}\mu},
\]

\[
\sigma_{\text{post}}
=
\operatorname{Softplus}
\left(
W_{\text{post}\sigma} y_p
+
b_{\text{post}\sigma}
\right).
\]

with:

\[
\operatorname{Softplus}(x)
=
\log(1+\exp(x)).
\]

Softplus guarantees non-negative standard deviations.

---

# 17. Factor Decoder

The decoder reconstructs/predicts stock returns using:

\[
\hat y
=
\phi_{\text{dec}}(z,e)
=
\alpha+\beta z.
\]

It has two branches:

1. alpha layer;
2. beta layer.

---

# 18. Alpha Layer

The idiosyncratic return is modeled probabilistically:

\[
\alpha
\sim
\mathcal N
\left(
\mu_\alpha,
\operatorname{diag}(\sigma_\alpha^2)
\right).
\]

For stock \(i\):

\[
h_\alpha^{(i)}
=
\operatorname{LeakyReLU}
\left(
W_\alpha e^{(i)}
+
b_\alpha
\right),
\]

\[
\mu_\alpha^{(i)}
=
W_{\alpha\mu}h_\alpha^{(i)}
+
b_{\alpha\mu},
\]

\[
\sigma_\alpha^{(i)}
=
\operatorname{Softplus}
\left(
W_{\alpha\sigma}h_\alpha^{(i)}
+
b_{\alpha\sigma}
\right).
\]

---

# 19. Beta Layer

Factor exposure is deterministic conditional on the latent stock feature:

\[
\beta^{(i)}
=
\phi_{\beta}(e^{(i)})
=
W_\beta e^{(i)}
+
b_\beta.
\]

Shape:

\[
\beta
\in
\mathbb{R}^{N\times K}.
\]

This makes the factor model dynamic:

\[
\beta
=
\beta(x).
\]

Different stocks and dates can have different factor exposures.

---

# 20. Probabilistic Return Distribution

Because both:

- \(\alpha\);
- factor vector \(z\);

are Gaussian random variables, the predicted stock return is Gaussian.

For stock \(i\):

\[
\hat y^{(i)}
\sim
\mathcal N
\left(
\mu_y^{(i)},
(\sigma_y^{(i)})^2
\right),
\]

where:

\[
\mu_y^{(i)}
=
\mu_\alpha^{(i)}
+
\sum_{k=1}^{K}
\beta^{(i,k)}
\mu_z^{(k)},
\]

and:

\[
\sigma_y^{(i)}
=
\left[
(\sigma_\alpha^{(i)})^2
+
\sum_{k=1}^{K}
(\beta^{(i,k)})^2
(\sigma_z^{(k)})^2
\right]^{1/2}.
\]

This equation is important because it explains how FactorVAE converts latent-factor uncertainty into per-stock return uncertainty.

---

# 21. Factor Predictor: Prior Factor Model

At test time the model cannot use future returns.

The factor predictor therefore learns:

\[
[\mu_{\text{prior}},\sigma_{\text{prior}}]
=
\phi_{\text{pred}}(e),
\]

\[
z_{\text{prior}}
\sim
\mathcal N
\left(
\mu_{\text{prior}},
\operatorname{diag}
(\sigma_{\text{prior}}^2)
\right).
\]

The factor predictor is the deployable factor-generation module.

---

# 22. Why the Predictor Uses Global Cross-Sectional Attention

A factor should represent a market-level risk premium rather than one stock in isolation.

The predictor therefore aggregates stock latent features across the entire cross-section.

The paper uses **multi-head global attention**.

The stated intuition is:

> different factor heads should integrate different global market representations corresponding to different risk premia.

---

# 23. Single-Head Global Attention

For stock \(i\):

\[
k^{(i)}
=
W_{\text{key}}e^{(i)},
\]

\[
v^{(i)}
=
W_{\text{value}}e^{(i)}.
\]

A learnable query vector \(q\in\mathbb{R}^H\) is used.

The attention weight is based on positive cosine similarity:

\[
a_{\text{att}}^{(i)}
=
\frac{
\max
\left(
0,
\frac{
q^\top k^{(i)}
}{
\|q\|_2\|k^{(i)}\|_2
}
\right)
}{
\sum_{i=1}^{N}
\max
\left(
0,
\frac{
q^\top k^{(i)}
}{
\|q\|_2\|k^{(i)}\|_2
}
\right)
}.
\]

The market representation is:

\[
h_{\text{att}}
=
\sum_{i=1}^{N}
a_{\text{att}}^{(i)}
v^{(i)}.
\]

---

# 24. Multi-Head Global Representation

The model uses \(K\) independent heads:

\[
h_{\text{multi}}
=
\operatorname{Concat}
\left(
[
\phi_{\text{att}_1}(e),
\dots,
\phi_{\text{att}_K}(e)
]
\right).
\]

The paper writes:

\[
h_{\text{multi}}
\in
\mathbb{R}^{K\times H}.
\]

A distribution network maps this representation to prior factor parameters:

\[
[\mu_{\text{prior}},\sigma_{\text{prior}}]
=
\pi_{\text{prior}}
(h_{\text{multi}}).
\]

---

# 25. Important Architectural Coupling: Number of Heads and Factors

The paper uses \(K\) attention heads and \(K\) factors.

This creates a natural architectural interpretation:

> each attention head can learn a distinct global market representation associated with one latent factor dimension.

The paper motivates diverse risk premia through multiple heads.

However, it does not provide a head-specialization analysis.

Therefore:

- **paper claim:** multi-head global representations support diverse factor extraction;
- **not demonstrated directly:** each head corresponds to a semantically distinct economic factor.

---

# 26. Prior-Posterior Learning

This is the core innovation.

The authors argue that direct end-to-end prediction from noisy historical data may fail to extract effective factors.

Their solution is:

### Posterior

Use future information to estimate:

\[
q_{\text{post}}(z|x,y).
\]

### Prior

Use only historical information to estimate:

\[
q_{\text{prior}}(z|x).
\]

### Guidance

Train the prior distribution to approximate the posterior distribution.

Conceptually:

\[
\text{future-informed optimal factors}
\rightarrow
\text{guide}
\rightarrow
\text{history-only predicted factors}.
\]

---

# 27. Objective Function

The paper's loss is:

\[
\mathcal L(x,y)
=
-
\frac{1}{N}
\sum_{i=1}^{N}
\log
P_{\phi_{\text{dec}}}
\left(
\hat y_{\text{rec}}^{(i)}
=
y^{(i)}
\mid
x,z_{\text{post}}
\right)
+
\gamma
\cdot
KL
\left[
P_{\phi_{\text{enc}}}(z|x,y),
P_{\phi_{\text{pred}}}(z|x)
\right].
\]

---

## 27.1 Reconstruction / Negative Log-Likelihood Term

The reconstructed return is:

\[
\hat y_{\text{rec}}^{(i)}
=
\alpha^{(i)}
+
\beta^{(i)}z_{\text{post}}.
\]

The first term trains the posterior factor model so that the future cross-sectional returns can be reconstructed well.

---

## 27.2 Prior-Posterior KL Term

The second term:

\[
KL
\left[
q_{\text{post}}(z|x,y),
q_{\text{prior}}(z|x)
\right]
\]

encourages the history-only factor predictor to approximate the future-informed posterior factor distribution.

\(\gamma\) controls the strength of this guidance.

---

# 28. Important Difference from a Vanilla VAE

A standard VAE often regularizes:

\[
q(z|x)
\]

toward a fixed prior such as:

\[
\mathcal N(0,I).
\]

FactorVAE instead uses a **learned conditional prior**:

\[
q_{\text{prior}}(z|x)
\]

and matches the future-informed posterior:

\[
q_{\text{post}}(z|x,y)
\]

to that learned prior/predictor.

This is a key structural difference.

---

# 29. Prediction Phase

Prediction uses:

\[
\hat y_{\text{pred}}
=
\phi_{\text{dec}}
(z_{\text{prior}},x)
=
\alpha
+
\beta z_{\text{prior}}.
\]

where:

\[
e
=
\phi_{\text{feat}}(x),
\]

\[
\alpha
\sim
\pi_{\alpha}(e),
\]

\[
\beta
=
\phi_{\beta}(e),
\]

\[
z_{\text{prior}}
\sim
\phi_{\text{pred}}(e).
\]

The posterior encoder is removed.

---

# 30. Prediction Used for Ranking

The predictive return distribution is:

\[
\hat y_{\text{pred}}^{(i)}
\sim
\mathcal N
\left(
\mu_{\text{pred}}^{(i)},
(\sigma_{\text{pred}}^{(i)})^2
\right).
\]

For RankIC evaluation, the paper uses:

\[
\mu_{\text{pred}}
\]

as the point prediction.

Thus ranking is based on expected return, not on a stochastic sample.

---

# 31. Risk Estimate

The model also produces:

\[
\sigma_{\text{pred}}^{(i)}.
\]

The paper interprets this as an estimate of stock risk.

This estimate is used in Experiment 3 through a risk-adjusted score:

\[
\mu_{\text{pred}}^{(i)}
-
\eta
\sigma_{\text{pred}}^{(i)}.
\]

This is the only direct practical evaluation of predictive uncertainty in the paper.

---

# 32. Figure 3: Full Framework

Figure 3 is the most important architecture figure.

It separates the model into three conceptual regions:

### Encoder — future-informed training branch

\[
(y,e)
\rightarrow
z_{\text{post}}
\]

### Predictor — deployable history-only branch

\[
e
\rightarrow
z_{\text{prior}}
\]

### Decoder

\[
(z,e)
\rightarrow
\hat y.
\]

The dotted modules involving future data are explicitly stated to be training-only and removed at test time.

---

# 33. Figure 4: Encoder and Decoder Internals

Figure 4(a) shows:

\[
e + y
\rightarrow
\text{portfolio layer}
\rightarrow
y_p
\rightarrow
\text{mapping layer}
\rightarrow
z_{\text{post}}.
\]

Figure 4(b) shows:

\[
e
\rightarrow
\alpha
\]

and:

\[
e
\rightarrow
\beta,
\]

combined with:

\[
z
\]

to reconstruct/predict stock returns.

---

# 34. Figure 5: Factor Predictor

Figure 5 shows the prior predictor:

\[
\{e^{(1)},\dots,e^{(N)}\}
\rightarrow
K \text{ global attention heads}
\rightarrow
h_{\text{multi}}
\rightarrow
\pi_{\text{prior}}
\rightarrow
(\mu_{\text{prior}},\sigma_{\text{prior}}).
\]

The design makes the factor prior explicitly cross-sectional.

---

# 35. Dataset

## 35.1 Market

China A-shares.

## 35.2 Raw Data

The paper states that raw day-level price-volume data are collected from Yahoo Finance.

Stocks under suspension or other abnormal conditions are excluded from the raw-data universe.

## 35.3 Feature Dataset

The paper uses Qlib's:

\[
\text{Alpha158}
\]

feature set.

Alpha158 contains 158 technical features derived from price-volume data.

For the time-series model, the authors use:

\[
20
\]

features selected by Qlib.

The paper does **not** list the names of those 20 features in the provided text.

---

# 36. Sequence Length

The historical sequence length is:

\[
T=20.
\]

Thus the feature extractor processes a 20-step historical sequence for each stock.

---

# 37. Label

For stock \(i\), prediction day \(t\):

\[
y_t^{(i)}
=
\frac{
p_{t+2}^{(i)}
-
p_{t+1}^{(i)}
}{
p_{t+1}^{(i)}
}.
\]

This should be reproduced exactly when matching the paper's protocol.

A research Agent should not silently replace the label with another Qlib horizon.

---

# 38. Train / Validation / Test Split

The paper reports:

## Training

- **Dates:** 2010-01-01 to 2017-12-31
- **Number of stocks:** 3432

## Validation

- **Dates:** 2018-01-01 to 2018-12-31
- **Number of stocks:** 3450

## Test

- **Dates:** 2019-01-01 to 2020-12-31
- **Number of stocks:** 3923

The stock count increases because of new listings.

---

# 39. Data-Split Interpretation

This is a fixed chronological split.

There is no rolling cross-validation described.

The paper does not specify in the provided text:

- how often hyperparameters were tuned on the validation set;
- whether model selection was repeated across seeds;
- whether the final test was touched during development.

Therefore these details must be marked as **not specified** rather than inferred.

---

# 40. Baselines

The paper compares against two groups.

## 40.1 Dynamic Factor Models

### Linear

A linear dynamic factor model.

### CA

Conditional autoencoder-based dynamic factor model from Gu, Kelly, and Xiu.

It is the most conceptually similar baseline.

---

## 40.2 ML Prediction Models

### GRU

GRU + linear prediction layer.

### ALSTM

Attention-enhanced LSTM.

### GAT

Graph Attention Network.

The paper describes it as treating stocks as nodes and predicting without requiring the graph structure in advance.

### Trans

Transformer-based stock-return predictor.

### SFM

RNN with frequency decomposition of hidden states.

---

# 41. Baseline Selection Logic

The baseline set spans:

- classical linear factor modeling;
- nonlinear dynamic factor modeling;
- recurrent sequence models;
- attention-based models;
- graph neural networks;
- Transformer;
- multi-frequency RNN.

This gives the authors both:

- same-family factor-model comparisons;
- broader predictive-ML comparisons.

---

# 42. Baseline Fairness: What the Paper Supports

The paper presents all compared models on the same test dataset and reports 5-seed results for prediction.

However, the provided paper text does **not** fully specify:

- whether all baseline hyperparameters received equal tuning budgets;
- whether original-author implementations or reimplementations were used;
- the exact optimizer/training schedule for each baseline;
- model parameter counts.

Therefore baseline fairness is only partially auditable from the paper.

---

# 43. Evaluation Metric: Daily Rank IC

For each test date \(s\), the paper computes cross-sectional rank correlation:

\[
RankIC_s
=
\frac{1}{N_s}
\frac{
(r_{\hat y_s}-\operatorname{mean}(r_{\hat y_s}))^\top
(r_{y_s}-\operatorname{mean}(r_{y_s}))
}{
\operatorname{std}(r_{\hat y_s})
\operatorname{std}(r_{y_s})
}.
\]

Here:

- \(r_{\hat y_s}\): predicted ranks;
- \(r_{y_s}\): true ranks.

The overall RankIC is:

\[
RankIC
=
\frac{1}{T_{\text{test}}}
\sum_{s=1}^{T_{\text{test}}}
RankIC_s.
\]

---

# 44. Rank ICIR

The paper defines Rank ICIR as:

\[
RankICIR
=
\frac{
\operatorname{mean}(RankIC_s)
}{
\operatorname{std}(RankIC_s)
}.
\]

It is intended to measure stability of cross-sectional ranking performance through time.

---

# 45. Experiment 1: Main Prediction Results

Table 1 reports mean and standard deviation over **5 random seeds**.

| Category | Method | Rank IC | Rank ICIR |
|---|---|---:|---:|
| ML prediction | GRU | 0.032 ± 0.002 | 0.398 ± 0.031 |
| ML prediction | ALSTM | 0.031 ± 0.004 | 0.360 ± 0.019 |
| ML prediction | GAT | 0.034 ± 0.002 | 0.390 ± 0.032 |
| ML prediction | Trans | 0.033 ± 0.003 | 0.417 ± 0.032 |
| ML prediction | SFM | 0.037 ± 0.001 | 0.456 ± 0.004 |
| Dynamic factor | Linear | 0.022 ± 0.002 | 0.333 ± 0.033 |
| Dynamic factor | CA | 0.039 ± 0.002 | 0.442 ± 0.036 |
| Dynamic factor | FactorVAE-prior | 0.042 ± 0.003 | 0.384 ± 0.033 |
| Dynamic factor | **FactorVAE** | **0.055 ± 0.004** | **0.568 ± 0.044** |

---

# 46. Main Prediction Result

FactorVAE achieves:

\[
RankIC = 0.055 \pm 0.004
\]

and:

\[
RankICIR = 0.568 \pm 0.044.
\]

It is best among all compared methods in both metrics.

Relative to CA:

\[
0.055-0.039=0.016
\]

absolute RankIC improvement.

Relative to SFM, the strongest ML baseline in RankIC:

\[
0.055-0.037=0.018.
\]

---

# 47. Critical Ablation: FactorVAE-prior

The most important ablation in the paper is **FactorVAE-prior**.

It removes prior-posterior learning and trains the prior factors directly for return prediction.

Results:

\[
RankIC:
0.042
\rightarrow
0.055
\]

when posterior guidance is added.

RankICIR:

\[
0.384
\rightarrow
0.568.
\]

The paper interprets this as evidence that:

> direct learning from noisy market data is insufficient, and future-informed posterior factors provide useful guidance.

---

# 48. What the FactorVAE-prior Ablation Actually Supports

The ablation supports the claim that the full training setup is better than direct prior-factor training.

It does **not** isolate every possible reason for the gain.

For example, it does not separately test:

- posterior reconstruction alone;
- KL guidance with alternative teacher targets;
- deterministic posterior vs. probabilistic posterior;
- fixed Gaussian prior vs. learned conditional prior;
- different values of \(\gamma\).

Thus the evidence supports the overall prior-posterior mechanism, but not every subdesign choice independently.

---

# 49. Experiment 2: Missing-Stock Robustness

The authors test whether models can generalize to stocks never observed during training.

Procedure:

1. randomly remove \(m\) stocks from the training dataset;
2. train models without those stocks;
3. evaluate predictions only on those held-out stock identities during test.

Values:

\[
m\in\{50,100,200\}.
\]

This is an unusual and valuable robustness test for a market with:

- new listings;
- changing universe membership.

---

# 50. Robustness Results

Table 2:

| Method | m=50 RankIC | m=50 RankICIR | m=100 RankIC | m=100 RankICIR | m=200 RankIC | m=200 RankICIR |
|---|---:|---:|---:|---:|---:|---:|
| GRU | 0.031 ± 0.005 | 0.184 ± 0.029 | 0.030 ± 0.004 | 0.234 ± 0.030 | 0.031 ± 0.004 | 0.282 ± 0.032 |
| ALSTM | 0.027 ± 0.004 | 0.162 ± 0.022 | 0.028 ± 0.007 | 0.210 ± 0.045 | 0.026 ± 0.005 | 0.237 ± 0.041 |
| GAT | 0.029 ± 0.008 | 0.166 ± 0.043 | 0.023 ± 0.011 | 0.176 ± 0.085 | 0.025 ± 0.009 | 0.215 ± 0.071 |
| Trans | 0.034 ± 0.007 | 0.201 ± 0.040 | 0.034 ± 0.006 | 0.259 ± 0.043 | 0.033 ± 0.003 | 0.302 ± 0.023 |
| SFM | 0.037 ± 0.007 | 0.220 ± 0.042 | 0.038 ± 0.004 | 0.294 ± 0.035 | 0.038 ± 0.003 | 0.342 ± 0.036 |
| Linear | 0.018 ± 0.005 | 0.138 ± 0.075 | 0.018 ± 0.005 | 0.147 ± 0.044 | 0.018 ± 0.004 | 0.176 ± 0.042 |
| CA | 0.038 ± 0.008 | 0.215 ± 0.046 | 0.039 ± 0.004 | 0.284 ± 0.034 | 0.039 ± 0.003 | 0.328 ± 0.027 |
| FactorVAE-port | 0.043 ± 0.005 | 0.241 ± 0.022 | 0.039 ± 0.003 | 0.272 ± 0.005 | 0.041 ± 0.004 | 0.328 ± 0.011 |
| **FactorVAE** | **0.053 ± 0.007** | **0.299 ± 0.039** | **0.056 ± 0.002** | **0.384 ± 0.044** | **0.050 ± 0.008** | **0.399 ± 0.063** |

FactorVAE is strongest across all reported missing-stock settings.

---

# 51. FactorVAE-port Ablation

FactorVAE-port replaces the proposed dynamic portfolio layer with the portfolio construction from the CA baseline:

\[
y_p
=
(e^\top e)^{-1}e^\top y
\]

as written in the paper.

The full model generally performs better.

The authors interpret this as evidence that the dynamically re-weighted portfolio layer is more effective and robust for changing stock universes.

---

# 52. What Experiment 2 Shows

The strongest supported claim is:

> FactorVAE retains better cross-sectional ranking performance on stock identities excluded from training.

This is relevant for:

- new listings;
- changing stock universes;
- generalization beyond memorized stock identity.

However, the experiment does not prove robustness to all forms of distribution shift.

It specifically tests **missing stock identity**, not:

- new market regimes;
- structural breaks;
- unseen industries;
- unseen feature distributions.

---

# 53. Experiment 3: Portfolio Backtest

The paper uses Qlib's **TopK-Drop** strategy.

Each trading day, it holds:

\[
k=50
\]

stocks.

The turnover constraint uses:

\[
n=5,
\]

meaning at least:

\[
k-n
\]

stocks overlap with the previous portfolio.

Thus:

\[
|P_t\cap P_{t-1}|
\ge
k-n.
\]

---

# 54. Investment Universe and Benchmark

The portfolio selects 50 stocks from the **CSI300** universe each trading day.

Benchmark:

\[
CSI300.
\]

The paper states that the backtest accounts for:

- trading fee;
- stock suspension;
- A-share price limits.

Exact fee values are **not specified in the paper text provided**.

---

# 55. Backtest Metrics

The paper reports:

- annualized return (AR);
- Sharpe ratio (SR);
- maximum drawdown (MDD).

These are computed on cumulative **excess return relative to CSI300**, according to the surrounding text and Table 3 description.

---

# 56. Main Backtest Results

| Method | AR ↑ | SR ↑ | MDD ↓ |
|---|---:|---:|---:|
| GRU | 2.28% | 0.31 | 9.08% |
| ALSTM | 2.20% | 0.27 | 12.19% |
| GAT | 4.49% | 0.56 | 7.20% |
| Trans | 4.79% | 0.62 | 5.01% |
| SFM | 3.33% | 0.42 | 7.32% |
| Linear | 0.01% | 0.02 | 8.02% |
| CA | 3.62% | 0.47 | 7.00% |
| **FactorVAE** | **15.32%** | **1.92** | **4.47%** |
| **FactorVAE(TDrisk)** | **16.32%** | **2.09** | 4.50% |

FactorVAE has a much larger reported excess-return performance than the baselines.

---

# 57. TDrisk

TDrisk modifies stock ranking by using:

\[
score_i
=
\mu_{\text{pred}}^{(i)}
-
\eta
\sigma_{\text{pred}}^{(i)}.
\]

Here:

- \(\mu_{\text{pred}}^{(i)}\): predicted expected return;
- \(\sigma_{\text{pred}}^{(i)}\): predicted risk;
- \(\eta\): risk-aversion weight.

Reported result:

\[
AR: 15.32\% \rightarrow 16.32\%
\]

and:

\[
SR: 1.92 \rightarrow 2.09.
\]

MDD changes slightly:

\[
4.47\%
\rightarrow
4.50\%.
\]

---

# 58. What TDrisk Actually Demonstrates

The result supports:

> the predicted standard deviation contains some useful information for risk-adjusted stock ranking.

But the paper does **not** provide:

- calibration curves;
- volatility prediction error;
- negative log-likelihood comparison;
- coverage probability;
- correlation between \(\sigma_{\text{pred}}\) and realized risk;
- sensitivity analysis over \(\eta\).

Therefore the uncertainty model is only indirectly validated through one portfolio application.

---

# 59. Figure 6

Figure 6 plots:

1. cumulative excess return;
2. cumulative absolute portfolio return;

during the 2019–2020 test period.

FactorVAE and FactorVAE(TDrisk) visually separate from the baseline curves over the test range.

The plot provides trajectory information that Table 3 alone does not show.

---

# 60. Known Hyperparameters Explicitly Reported

The paper explicitly provides the following.

| Hyperparameter / setting | Value |
|---|---|
| Sequence length \(T\) | 20 |
| Input features | 20 Qlib-selected features from Alpha158 |
| Prediction test seeds | 5 random seeds |
| Missing-stock test | \(m=50,100,200\) |
| TopK portfolio size \(k\) | 50 |
| TopK-Drop replacement parameter \(n\) | 5 |
| Main backtest universe | CSI300 |
| Main prediction train period | 2010–2017 |
| Validation period | 2018 |
| Test period | 2019–2020 |

---

# 61. Important Hyperparameters NOT Specified in the Paper Text

The following are required for exact reproduction but are not specified in the provided paper text.

## Architecture

- latent factor dimension \(K\);
- GRU hidden dimension \(H\);
- number of dynamic posterior portfolios \(M\);
- LeakyReLU negative slope \(\zeta\);
- internal widths of distribution networks;
- exact number of layers in mapping/distribution networks beyond shown formulas.

## Training

- optimizer;
- learning rate;
- learning-rate scheduler;
- batch size;
- number of epochs;
- early-stopping rule;
- weight decay;
- gradient clipping;
- initialization;
- exact random seeds.

## Objective

- KL weight \(\gamma\).

## Risk Strategy

- risk-aversion coefficient \(\eta\).

## Backtest

- exact trading-fee value;
- detailed execution assumptions;
- exact rebalance timing;
- slippage configuration.

These omissions materially reduce exact reproducibility from the paper alone.

---

# 62. Reproducibility Assessment

## High reproducibility aspects

- task definition;
- model decomposition;
- major formulas;
- chronological data split;
- sequence length;
- feature family;
- label definition;
- baseline names;
- evaluation metrics;
- main backtest strategy;
- headline results.

## Low reproducibility aspects

- model dimensionalities;
- optimizer/training schedule;
- KL coefficient;
- risk-aversion coefficient;
- feature identities among the selected 20;
- baseline tuning details;
- fee/slippage configuration.

Overall:

> **Conceptual reproducibility: high. Exact numerical reproducibility from the paper alone: medium-low.**

---

# 63. Inductive Biases

The architecture encodes several strong assumptions.

## 63.1 Low-Dimensional Factor Structure

Cross-sectional returns can be approximately decomposed into a small number of common latent factors plus idiosyncratic components.

## 63.2 Time-Varying Factor Exposure

Factor exposure depends on current stock characteristics:

\[
\beta=\beta(x).
\]

## 63.3 Global Cross-Sectional State Matters

Latent market factors should be predicted from the joint state of the cross-section rather than from one stock independently.

## 63.4 Future Returns Reveal Better Training-Time Factors

Realized future returns contain information useful for constructing a superior latent factor target.

## 63.5 Probabilistic Latents Help with Noise

Noise should be modeled through latent distributions rather than only deterministic embeddings.

## 63.6 Stock Identity Should Not Be Essential

The use of feature-driven portfolio weighting and global aggregation should make the model applicable to unseen stock identities.

---

# 64. Why FactorVAE May Work

This section separates paper-supported mechanisms from interpretation.

## 64.1 Paper-Supported Explanation: Posterior Guidance

FactorVAE-prior is weaker than full FactorVAE.

This supports the value of future-informed posterior guidance.

## 64.2 Paper-Supported Explanation: Dynamic Portfolio Layer

FactorVAE outperforms FactorVAE-port in the missing-stock experiment.

This supports the proposed dynamic portfolio compression under changing universes.

## 64.3 Plausible Mechanism: Better Latent Target

The posterior encoder sees the realized cross-sectional outcome.

Therefore it can identify factor realizations that explain the target cross-section more directly than a history-only encoder.

The predictor receives a cleaner learning target in latent distribution space.

## 64.4 Plausible Mechanism: Structured Bottleneck

The factor form:

\[
\alpha+\beta z
\]

restricts the model to a finance-motivated decomposition.

This may improve generalization relative to an unconstrained black-box predictor.

## 64.5 Plausible Mechanism: Cross-Sectional Aggregation

Global attention allows factors to depend on the market-wide state.

This is more appropriate for "factor" prediction than independent per-stock modeling.

---

# 65. Potential Failure Modes

These are analytical concerns inferred from the architecture, not reported failures unless stated otherwise.

## 65.1 Posterior Learns Realized Noise

Because the posterior encoder sees realized future returns, it may encode not only systematic structure but also cross-sectional noise.

The KL-guided prior may then be asked to imitate some unpredictable components.

## 65.2 Factor Semantics Are Unidentified

Latent dimensions are learned.

The paper does not demonstrate that individual factors correspond to stable economic concepts.

## 65.3 Gaussian Assumption

Both alpha and factors are modeled as independent Gaussian variables.

Real stock-return distributions may exhibit:

- skewness;
- heavy tails;
- nonlinear dependence;
- correlated latent uncertainty.

## 65.4 Diagonal Factor Covariance

The formulation uses diagonal covariance for factors.

Cross-factor uncertainty correlations are not modeled explicitly.

## 65.5 Linear Decoder in Factors

Although exposures depend nonlinearly on history, the final factor return decomposition remains:

\[
\alpha+\beta z.
\]

Complex nonlinear factor interactions are not represented in the decoder.

## 65.6 Fixed Number of Factors

The factor count \(K\) is fixed.

The model does not dynamically vary effective factor complexity across regimes.

## 65.7 Short Test Period

The main test period is only:

\[
2019\text{–}2020.
\]

This includes distinctive market conditions and may not establish long-horizon robustness.

---

# 66. Evidence Strength

## Strong evidence

- 5-seed prediction statistics;
- comparison across multiple baseline families;
- clear ablation of prior-posterior learning;
- explicit unseen-stock robustness experiment;
- economic backtest;
- risk-adjusted portfolio variant.

## Missing evidence

- statistical significance tests between models;
- longer multi-regime test period;
- multiple markets;
- detailed uncertainty calibration;
- sensitivity to \(K\), \(M\), \(\gamma\), \(H\);
- parameter-count / compute comparison;
- full module-by-module ablation;
- performance under transaction-cost variation.

Overall evidence strength:

> **Moderate to strong for the paper's main benchmark claim, but incomplete for mechanism isolation and generalization beyond the reported market/time period.**

---

# 67. Ablation Coverage

## Present

### FactorVAE-prior

Tests the value of prior-posterior learning.

### FactorVAE-port

Tests the proposed dynamic portfolio layer against CA-style portfolio construction in the missing-stock setting.

### TDrisk

Tests practical use of predicted uncertainty.

## Missing

No isolated ablations for:

- probabilistic vs. deterministic factors;
- probabilistic vs. deterministic alpha;
- global attention vs. pooling;
- number of heads;
- factor count;
- GRU vs. alternative temporal encoder;
- beta layer design;
- KL weight;
- posterior portfolio count;
- diagonal vs. full covariance.

---

# 68. Negative Results

The paper does not provide a dedicated negative-results section.

The most relevant weak/negative comparison is:

> FactorVAE-prior has lower RankICIR than CA despite higher RankIC.

Specifically:

\[
FactorVAE\text{-prior RankIC}=0.042
\]

vs.

\[
CA=0.039,
\]

but:

\[
FactorVAE\text{-prior RankICIR}=0.384
\]

vs.

\[
CA=0.442.
\]

This suggests direct prior-factor learning can raise mean rank correlation while reducing temporal stability.

The full posterior-guided model improves both.

---

# 69. Novelty Decomposition

## Existing Ingredients

- dynamic factor model;
- GRU temporal encoder;
- variational latent variables;
- Gaussian latent distribution;
- attention;
- encoder-decoder architecture.

## Finance-Specific Adaptations

- factor-model decoder:
  \[
  \alpha+\beta z;
  \]
- dynamic stock-to-portfolio compression;
- market-level factor prediction from the stock cross-section;
- per-stock risk propagation from latent-factor uncertainty.

## Main Novel Mechanism

Training-only future-informed posterior factors supervise a history-only prior factor predictor.

## Novel Combination

\[
\text{Dynamic Factor Model}
+
\text{Conditional VAE-like Prior/Posterior Learning}
+
\text{Cross-Sectional Global Attention}.
\]

---

# 70. Mechanism Primitives

For a research Agent, FactorVAE can be decomposed into reusable primitives.

## Primitive A — Historical Stock Encoder

\[
x_i^{1:T}
\rightarrow
e_i.
\]

## Primitive B — Learned Cross-Sectional Portfolio Compression

\[
\{e_i,y_i\}_{i=1}^{N}
\rightarrow
y_p.
\]

## Primitive C — Future-Informed Posterior Latent Factors

\[
(y_p)
\rightarrow
q_{\text{post}}(z|x,y).
\]

## Primitive D — Market-Wide Global Attention

\[
\{e_i\}_{i=1}^{N}
\rightarrow
h_{\text{market}}.
\]

## Primitive E — Historical Conditional Prior

\[
h_{\text{market}}
\rightarrow
q_{\text{prior}}(z|x).
\]

## Primitive F — Latent Distribution Matching

\[
KL(q_{\text{post}}\|q_{\text{prior}}).
\]

## Primitive G — Dynamic Factor Exposure

\[
e_i
\rightarrow
\beta_i.
\]

## Primitive H — Probabilistic Alpha

\[
e_i
\rightarrow
(\mu_{\alpha_i},\sigma_{\alpha_i}).
\]

## Primitive I — Risk Propagation

\[
(\sigma_\alpha,\beta,\sigma_z)
\rightarrow
\sigma_y.
\]

---

# 71. Task-Specific vs. Task-Agnostic Components

## Strongly Finance-Specific

- \(\alpha+\beta z\) factor decoder;
- factor-exposure interpretation;
- TopK-Drop portfolio backtest;
- risk-adjusted return ranking;
- dynamic portfolio compression over stocks.

## More Task-Agnostic

- posterior-guided conditional latent learning;
- teacher/oracle branch used only during training;
- KL matching between future-informed and history-only latent distributions;
- global attention aggregation;
- probabilistic latent representation.

This distinction is important if a research Agent wants to transfer FactorVAE ideas to another domain.

---

# 72. Implementation Hooks

## 72.1 Suggested Module Boundaries

```text
factorvae/
├── feature_extractor.py
├── factor_encoder.py
├── portfolio_layer.py
├── factor_predictor.py
├── factor_decoder.py
├── distributions.py
├── loss.py
└── model.py
```

---

## 72.2 Minimal Interfaces

### Feature Extractor

```text
Input:
x: [N, T, C]

Output:
e: [N, H]
```

### Factor Encoder

```text
Input:
e: [N, H]
y: [N]

Output:
mu_post: [K]
sigma_post: [K]
```

### Factor Predictor

```text
Input:
e: [N, H]

Output:
mu_prior: [K]
sigma_prior: [K]
```

### Factor Decoder

```text
Input:
e: [N, H]
mu_z / sampled z: [K]
sigma_z: [K]

Output:
mu_y: [N]
sigma_y: [N]
beta: [N, K]
alpha distribution parameters
```

---

# 73. Training Pseudocode

A faithful conceptual training loop is:

```text
for each cross-sectional sample (x, y):

    e = feature_extractor(x)

    # posterior branch: training only
    mu_post, sigma_post = factor_encoder(y, e)
    z_post = reparameterized_sample(mu_post, sigma_post)

    # prior branch: deployable
    mu_prior, sigma_prior = factor_predictor(e)

    # reconstruction from posterior factors
    mu_alpha, sigma_alpha, beta = decoder_parameters(e)
    mu_rec, sigma_rec = return_distribution(
        mu_alpha,
        sigma_alpha,
        beta,
        mu_post,
        sigma_post
    )

    loss_rec = gaussian_negative_log_likelihood(y, mu_rec, sigma_rec)

    loss_kl = KL(
        Normal(mu_post, sigma_post),
        Normal(mu_prior, sigma_prior)
    )

    loss = loss_rec + gamma * loss_kl

    update_parameters(loss)
```

Exact sampling and implementation details should follow the authors' code if available, because the paper does not specify every engineering detail.

---

# 74. Prediction Pseudocode

```text
e = feature_extractor(x)

mu_prior, sigma_prior = factor_predictor(e)

mu_alpha, sigma_alpha, beta = decoder_parameters(e)

mu_pred =
    mu_alpha
    + beta @ mu_prior

sigma_pred =
    sqrt(
        sigma_alpha^2
        + (beta^2) @ (sigma_prior^2)
    )

ranking_score = mu_pred
```

For TDrisk:

```text
ranking_score =
    mu_pred
    - eta * sigma_pred
```

---

# 75. Experiment Hooks for Improving FactorVAE

The following are not in the paper; they are useful research hypotheses for future work.

## E1 — Replace GRU Feature Extractor

Hypothesis:

> A stronger temporal encoder improves stock latent features without changing the factor-learning framework.

Candidates:

- TCN;
- Transformer;
- state-space model;
- multi-scale encoder.

Minimal comparison:

\[
GRU
\rightarrow
\text{new temporal encoder}.
\]

---

## E2 — Improve Cross-Sectional Factor Predictor

Hypothesis:

> Fixed \(K\)-head cosine attention is not the best way to summarize market state.

Possible replacements:

- cross-sectional Transformer;
- set encoder;
- routing network;
- hierarchical attention;
- graph aggregation;
- regime-aware attention.

---

## E3 — Regime-Conditioned Prior

Hypothesis:

> The mapping from historical market state to factor distribution changes by regime.

Add:

\[
r_t
=
g(\text{market state})
\]

and condition:

\[
q_{\text{prior}}(z|x,r_t).
\]

---

## E4 — Richer Posterior

Hypothesis:

> A diagonal Gaussian posterior is too restrictive.

Candidates:

- full covariance;
- low-rank covariance;
- mixture posterior;
- vector quantization;
- normalizing flow.

---

## E5 — Better Distillation Target

Hypothesis:

> Direct KL matching may force the predictor to mimic unpredictable posterior noise.

Possible alternatives:

- factor-mean-only distillation;
- uncertainty-weighted KL;
- symmetric divergence;
- Wasserstein distance;
- predictive-consistency loss.

---

## E6 — Explicit Cross-Sectional Ranking Objective

The model is trained with likelihood but evaluated with RankIC.

Hypothesis:

> Adding a ranking-aware auxiliary objective improves alignment between training and evaluation.

---

# 76. Research Opportunities

## Opportunity 1 — Posterior Quality

The full model assumes posterior factors are "optimal" because they see future returns.

But realized future returns contain noise.

Research question:

> How can a posterior distinguish systematic realized structure from idiosyncratic future noise?

---

## Opportunity 2 — Factor Specialization

The model uses \(K\) attention heads / factors but does not explicitly enforce diversity.

Research question:

> Can factor collapse be reduced with diversity or orthogonality regularization?

---

## Opportunity 3 — Time-Varying Factor Count

Market complexity may vary through time.

Research question:

> Can routing or sparse mixture mechanisms activate a variable number of factors?

---

## Opportunity 4 — Uncertainty Calibration

The model outputs \(\sigma_{\text{pred}}\), but calibration is not directly evaluated.

Research question:

> Does predicted uncertainty correspond quantitatively to realized forecast error / volatility?

---

## Opportunity 5 — Cross-Market Generalization

The paper tests one market.

Research question:

> Do the learned mechanisms transfer to U.S. equities or other universes?

---

# 77. How to Beat This Baseline

## 77.1 What FactorVAE Already Does Well

- uses both temporal and cross-sectional information;
- preserves a finance-motivated factor structure;
- handles variable stock universes;
- uses future information legally during training as supervision;
- produces a probabilistic return distribution;
- reports multi-seed prediction results;
- connects predictive output to a realistic Qlib portfolio strategy.

---

## 77.2 Structural Weaknesses

### Temporal Encoder

GRU is relatively simple and stock-wise.

### Posterior Distribution

Diagonal Gaussian.

### Prior Factor Predictor

Fixed global attention architecture.

### Factor Count

Fixed.

### Factor Independence

Implicit diagonal Gaussian assumption.

### Final Decoder

Linear in factors.

### Objective Mismatch

Likelihood training vs. ranking evaluation.

---

## 77.3 Evaluation Weaknesses

- test period only 2019–2020;
- one country / market;
- limited uncertainty validation;
- limited hyperparameter sensitivity;
- no parameter-count fairness;
- no compute comparison;
- incomplete module ablation.

---

## 77.4 Safe Improvement Directions

Relatively low conceptual risk:

- stronger temporal encoder;
- improved cross-sectional aggregator;
- ranking auxiliary loss;
- better uncertainty calibration;
- regime information;
- factor diversity regularization.

---

## 77.5 Higher-Novelty Directions

Potentially stronger paper contributions:

- discrete / quantized posterior factors;
- regime-conditioned factor routing;
- mixture-of-experts factor predictor;
- sparse dynamic factor activation;
- hierarchical posterior factors;
- shared + specialized factors;
- uncertainty-aware routing;
- factor persistence / temporal transition model.

---

## 77.6 What Would Probably Be Too Weak as Novelty

Examples of changes that may be insufficient alone:

- replace GRU with Transformer with no finance-specific adaptation;
- increase hidden dimension;
- add more attention heads;
- tune \(K\);
- change activation function;
- add a generic MLP layer.

A stronger contribution should alter the **factor-learning mechanism**, **market-state adaptation**, **latent structure**, or **training objective**, not merely the backbone.

---

# 78. Local Reproduction Section

This section should be filled with the user's own experimental pipeline results when available.

## Paper-Reported Results

### Prediction

\[
RankIC=0.055\pm0.004
\]

\[
RankICIR=0.568\pm0.044
\]

### Backtest

\[
AR=15.32\%
\]

\[
SR=1.92
\]

\[
MDD=4.47\%.
\]

## Our Reproduction

- Dataset:
- Market:
- Train:
- Validation:
- Test:
- Feature set:
- Seed count:
- IC:
- ICIR:
- RankIC:
- RankICIR:
- AR:
- Sharpe:
- MDD:

## Differences from Paper

- TBD.

## Possible Causes

- TBD.

## Known Implementation Differences

- TBD.

---

# 79. Baseline Comparison Checklist

When a new model is compared against FactorVAE, an Agent should check:

## Same data?

- same market;
- same universe;
- same date ranges.

## Same features?

- Alpha158;
- same 20 selected features or full Alpha158?

## Same label?

\[
(p_{t+2}-p_{t+1})/p_{t+1}
\]

or another horizon?

## Same evaluation?

- RankIC;
- RankICIR;
- seed averaging.

## Same backtest?

- CSI300;
- TopK-Drop;
- \(k=50\);
- \(n=5\);
- fees;
- suspension;
- price limits.

Without these controls, headline numbers are not directly comparable.

---

# 80. Important Warnings for a Research Agent

## Warning 1 — Do Not Call the Posterior Branch Leakage by Default

The posterior branch intentionally uses future returns **during training only**.

That is analogous to training-time supervision.

Leakage occurs only if future information enters prediction/test features or model selection improperly.

---

## Warning 2 — Do Not Treat Posterior Factors as Ground-Truth Economic Factors

They are learned latent variables optimized to reconstruct returns.

Their economic semantics are not established.

---

## Warning 3 — Do Not Equate Predictive Sigma with Realized Volatility Without Testing

The paper uses \(\sigma_{\text{pred}}\) as a risk signal.

It does not prove that this sigma is calibrated to realized volatility.

---

## Warning 4 — Do Not Compare Paper RankIC Directly to a Different Qlib Protocol

Differences in:

- market;
- feature set;
- horizon;
- stock universe;
- train/test split;
- preprocessing;

can materially change RankIC.

---

# 81. Strengths

## Methodological

- clear financial factor-model structure;
- elegant training-time use of future outcomes;
- probabilistic modeling is integrated with the factor decomposition;
- variable-universe handling is explicitly considered.

## Experimental

- 5-seed reporting;
- multiple baseline families;
- ablation for core prior-posterior mechanism;
- robustness to unseen stocks;
- practical portfolio backtest.

## Writing

- contributions are tightly aligned with experiments;
- RQ1–RQ3 structure creates a clean experimental narrative;
- architecture figures are highly informative;
- formulas map directly to modules.

---

# 82. Limitations

## 82.1 Explicitly Acknowledged / Directly Observable from the Paper

The paper's conclusion only states one explicit future direction:

> explore more portfolio strategies based on the model.

The paper itself does not provide a long limitations section.

Therefore the following should be distinguished as analytical limitations.

---

## 82.2 Agent-Identified Limitations

### Reproducibility gap

Many critical hyperparameters are omitted.

### Limited markets

Only China A-shares are tested.

### Limited test horizon

Two-year test period.

### Uncertainty validation is indirect

No calibration analysis.

### Limited ablation depth

Core modules are not all isolated.

### Latent-factor interpretability

No semantic or economic validation of learned factors.

### Gaussian assumptions

May be restrictive for financial returns.

### No computational analysis

No parameter/FLOP/training-time comparison.

---

# 83. Future Work

## 83.1 Explicit Author Future Work

The authors state:

> explore more portfolio strategies based on FactorVAE.

No broader formal future-work list is given.

---

## 83.2 Implied Future Work

Based on the model and missing evaluations:

- improve portfolio/risk-aware decision strategies;
- calibrate predictive uncertainty;
- test other markets;
- evaluate longer regimes;
- study factor interpretability;
- relax Gaussian assumptions;
- improve cross-sectional aggregation;
- test sensitivity to factor dimension and KL strength.

These are inferred opportunities, not direct quotations from the paper.

---

# 84. Writing / Presentation Lessons

## 84.1 Problem Framing

The paper frames the challenge as:

\[
\text{powerful ML}
+
\text{low-SNR finance}
\rightarrow
\text{need guided latent learning}.
\]

This gives a very direct reason for the method.

---

## 84.2 Architecture Story

The model is easy to explain because every component has a role:

- feature extractor → temporal stock representation;
- encoder → future-informed optimal factor;
- predictor → deployable factor;
- decoder → factor-model return;
- KL → connect posterior and prior.

This is strong module-function correspondence.

---

## 84.3 Experimental Narrative

The RQs align almost one-to-one with experiments:

### RQ1
FactorVAE-prior ablation.

### RQ2
missing-stock robustness.

### RQ3
risk-aware portfolio strategy.

This makes the paper feel closed-loop.

---

## 84.4 Useful Writing Pattern

A reusable method paragraph pattern is:

1. identify a domain difficulty;
2. explain why end-to-end learning is insufficient;
3. introduce a training-time privileged-information branch;
4. remove privileged information at inference;
5. validate each claimed benefit with a dedicated experiment.

---

# 85. Agent Critique: Claim vs. Evidence

## Claim A

Prior-posterior learning helps extract better factors from noisy market data.

### Evidence

FactorVAE significantly outperforms FactorVAE-prior in RankIC and RankICIR.

### Assessment

Well supported at the whole-mechanism level.

---

## Claim B

Dynamic portfolio construction improves robustness to missing stocks.

### Evidence

FactorVAE generally outperforms FactorVAE-port in Experiment 2.

### Assessment

Supported for the specific unseen-stock setup.

---

## Claim C

Probabilistic modeling provides useful risk estimates.

### Evidence

TDrisk improves AR and SR.

### Assessment

Promising but incomplete because uncertainty calibration itself is not measured.

---

## Claim D

FactorVAE models noisy market data better because factors are random variables.

### Evidence

The overall model performs well.

### Assessment

Not cleanly isolated because there is no deterministic-factor ablation.

---

# 86. Agent Research Instructions

When using FactorVAE as a baseline:

1. **Reproduce the exact data split before judging model-level improvements.**
2. **Check whether the local implementation uses the paper's 20 selected features or another Alpha158 setup.**
3. **Check the exact return label/horizon.**
4. **Separate posterior-branch training logic from test-time inference.**
5. **Do not attribute all gains to probabilistic modeling; the paper does not isolate that component.**
6. **Treat prior-posterior learning as the best-supported core innovation.**
7. **Treat the dynamic portfolio layer as a second important finance-specific mechanism.**
8. **If modifying the model, preserve a clear explanation of what replaces each original module.**
9. **Use RankIC/RankICIR and backtest metrics under the same protocol when claiming superiority.**
10. **Add missing sensitivity / calibration / multi-market evidence if targeting a stronger modern paper.**
11. **Do not confuse latent factor semantics with known financial factors unless tested.**
12. **Use the paper's weaknesses to design falsifiable experiments rather than merely adding modules.**

---

# 87. Agent Takeaways

## Most Important Research Problem

How to learn useful latent dynamic factors from low-signal-to-noise financial data.

## Most Important Mechanism

**Future-informed posterior factors guide a history-only prior factor predictor through KL divergence.**

## Most Important Architecture Equation

\[
\hat y
=
\alpha+\beta z.
\]

## Most Important Learning Equation

\[
\mathcal L
=
\mathcal L_{\text{NLL}}
+
\gamma
KL
(q_{\text{post}}\|q_{\text{prior}}).
\]

## Most Important Ablation

FactorVAE vs. FactorVAE-prior.

## Most Important Finance-Specific Design

Dynamic learned portfolio compression of the cross-section.

## Most Important Practical Addition

Predictive standard deviation is used for risk-adjusted stock selection.

## Biggest Reproducibility Problem

Critical architecture and training hyperparameters are omitted from the paper text.

## Biggest Scientific Limitation

The model's probabilistic and latent-factor interpretations are not fully isolated or calibrated experimentally.

## Best Direction for Building Beyond It

Improve the **latent-factor generation and adaptation mechanism** rather than merely replacing the GRU backbone.

---

# 88. Compact Architecture Summary

```text
Historical stock sequences x
        │
        ▼
GRU Feature Extractor
        │
        ▼
Stock latent features e
        │
        ├──────────────────────────────────────┐
        │                                      │
        │ TRAINING POSTERIOR                    │ HISTORY-ONLY PRIOR
        │                                      │
        ▼                                      ▼
Future returns y + e                  Multi-head global attention
        │                                      │
Dynamic portfolio layer                       ▼
        │                              Market representation
        ▼                                      │
Portfolio returns y_p                         ▼
        │                              Prior distribution net
        ▼                                      │
Posterior mapping                              ▼
        │                           (mu_prior, sigma_prior)
        ▼
(mu_post, sigma_post)
        │
        └────────── KL matching ───────────────┘

Stock features e
    │
    ├── Alpha distribution
    │
    └── Dynamic exposures beta

Factor distribution z + alpha + beta
                │
                ▼
Predicted return distribution
(mu_pred, sigma_pred)
```

---

# 89. Source Location Map

Useful paper locations for verification:

- **AAAI p.4468–4469:** problem motivation, contributions, related work.
- **p.4470:** formal DFM problem, VAE background, feature extractor, start of posterior encoder.
- **p.4471:** dynamic portfolio layer, posterior distribution, factor decoder, alpha/beta equations, predictive variance.
- **p.4472:** prior factor predictor, global attention, loss, prediction equations, research questions.
- **p.4473:** dataset split, baselines, RankIC/RankICIR, Table 1, unseen-stock robustness design.
- **p.4474:** Table 2 continuation, Figure 6, TopK-Drop backtest, Table 3, conclusion.

---

# 90. Compact Retrieval Summary

FactorVAE (Duan et al., AAAI 2022) is a probabilistic dynamic factor model for cross-sectional stock-return prediction. Historical Alpha158-derived sequences are encoded with a GRU into stock latent features. During training, a posterior factor encoder uses future realized returns together with those stock features; it first compresses the variable-size cross-section into dynamically weighted portfolio returns and then maps those portfolio returns into a Gaussian distribution over latent factors. A history-only factor predictor uses multi-head global cross-sectional attention to produce a conditional Gaussian prior over factors. Training minimizes posterior reconstruction negative log-likelihood plus a KL divergence that aligns the history-only prior with the future-informed posterior. A finance-structured decoder predicts returns as \(\alpha+\beta z\), where alpha and factors are probabilistic and factor exposures are dynamic functions of stock features. At test time, the future-return encoder is removed. On China A-shares with Qlib Alpha158, FactorVAE reports RankIC \(0.055\pm0.004\) and RankICIR \(0.568\pm0.044\), outperforming Linear, CA, GRU, ALSTM, GAT, Transformer, and SFM baselines. The key ablation, FactorVAE-prior, falls to RankIC 0.042, supporting the prior-posterior learning mechanism. The model also shows stronger performance on stock identities excluded from training and produces a predictive standard deviation used in a risk-adjusted TopK-Drop strategy. Its main limitations are incomplete hyperparameter reporting, a short 2019–2020 test period, one market, limited uncertainty calibration, and incomplete module-level ablation.
