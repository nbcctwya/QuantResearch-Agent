---
paper_id: TimeMixer_2024
title: "TimeMixer: Decomposable Multiscale Mixing for Time Series Forecasting"
short_name: TimeMixer
authors:
  - Shiyu Wang
  - Haixu Wu
  - Xiaoming Shi
  - Tengge Hu
  - Huakun Luo
  - Lintao Ma
  - James Y. Zhang
  - Jun Zhou
year: 2024
venue: "International Conference on Learning Representations (ICLR 2024)"
paper_type: foundations
subtype:
  - time_series_forecasting
  - multiscale_learning
  - seasonal_trend_decomposition
  - mlp_mixing
  - multiscale_ensemble
source_document: "Wang et al., TimeMixer, user-supplied 27-page ICLR 2024 conference PDF"
source_pages: 27
source_repository: "https://github.com/kwuking/TimeMixer"
recommended_directory: "papers/foundations/TimeMixer_2024.md"
original_task:
  - multivariate_long_term_forecasting
  - multivariate_short_term_forecasting
  - univariate_short_term_forecasting
main_mechanisms:
  - average_pooling_multiscale_pyramid
  - seasonal_trend_decomposition
  - fine_to_coarse_seasonal_mixing
  - coarse_to_fine_trend_mixing
  - residual_multiscale_information_flow
  - multiscale_future_prediction_ensemble
  - all_mlp_temporal_processing
main_method_blocks:
  - Past_Decomposable_Mixing
  - Future_Multipredictor_Mixing
original_benchmarks:
  - ETTh1
  - ETTh2
  - ETTm1
  - ETTm2
  - Weather
  - Solar-Energy
  - Electricity
  - Traffic
  - PEMS03
  - PEMS04
  - PEMS07
  - PEMS08
  - M4-Yearly
  - M4-Quarterly
  - M4-Monthly
  - M4-Weekly
  - M4-Daily
  - M4-Hourly
original_metrics:
  - MSE
  - MAE
  - MAPE
  - RMSE
  - SMAPE
  - MASE
  - OWA
primary_finance_relevance:
  - multiscale_short_and_medium_term_signals
  - scale_specific_trend_and_residual_information
  - temporal_encoder_alternatives_to_rwkv_and_transformer
  - market_regime_conditional_mixing
  - multiscale_rank_score_fusion
  - interaction_with_cross_sectional_stock_modules
classification_note: "Foundational reusable time-series method, not an original CSI300 / S&P500 Qlib baseline"
---

# TimeMixer — Decomposable Multiscale Mixing for Time Series Forecasting

> **Role:** General time-series **foundations**, not a finance-specific baseline or a 2026 frontier contribution.  
> **Original source:** Wang et al., *TimeMixer: Decomposable Multiscale Mixing for Time Series Forecasting*, ICLR 2024, supplied PDF pp. 1–27.  
> **Location:** `papers/foundations/TimeMixer_2024.md`  
> **Core:** multiscale sampling → seasonal/trend decomposition → **direction-specific cross-scale mixing** → multiscale forecasts → sum.  
> **Scope warning:** The paper contains **no CSI300, S&P500, Alpha158, IC/RankIC, financial factor model, or portfolio backtest**. All stock-prediction designs later in this note are proposed adaptations, **not evidence from the original paper**.

## Document provenance and evidence convention

This file has four explicitly different levels of authority:

- **`[SOURCE]`**: statement, notation, formula, figure, numerical value, or limitation found in the supplied 27-page ICLR paper; an original page/section/equation/table is attached where useful.
- **`[DERIVATION]`**: mathematical consequence, dimensional trace, or executable-like pseudocode reconstructed from those descriptions; verify code details independently before exact reproduction.
- **`[CRITIQUE]`**: independent assessment of evidence, untested alternatives, hidden assumptions, and comparability.
- **`[EXTENSION]`**: proposed adaptations and falsifiable research hypotheses for cross-sectional stock return ranking; not experiments conducted by Wang et al.

For citation traceability, `p.` always refers to **PDF page number** (which matches the conference pages here), not a guessed web edition. Formula and table numbers are those printed in the provided PDF. Numerical values are transcribed from the supplied conference document. If there is a table typo or ambiguity, record it rather than silently correcting the source.

---

# Part I — Research Identity, Problem and Contributions

## 1. Exact bibliographic identification `[SOURCE, p.1]`

- **Full title:** *TimeMixer: Decomposable Multiscale Mixing for Time Series Forecasting*.
- **Authors:** Shiyu Wang; Haixu Wu; Xiaoming Shi; Tengge Hu; Huakun Luo; Lintao Ma; James Y. Zhang; Jun Zhou.
- **Affiliations:** Ant Group and Tsinghua University.
- **Venue:** ICLR 2024, conference paper.
- **Original code URL:** `https://github.com/kwuking/TimeMixer`, reported in reproducibility statement (p.10).
- **Main task:** time-series forecasting over future horizons, univariate or multivariate.
- **Model family:** fully MLP-based, multiscale, decomposition-based predictor.
- **Not:** Transformer / MoE / vector-quantization / stock-factor model.

## 2. Central research question `[SOURCE, Abstract and §1, pp.1–2]`

How can a prediction model represent **intricate temporal variations** that coexist at distinct temporal resolutions without relying on increasingly elaborate self-attention machinery?

- Fine temporal resolution contains detailed local fluctuations and microscopic patterns.
- Coarse resolution emphasizes macroscopic trends and suppresses some fine details.
- Both kinds of variation matter for predicting future changes.
- Merely decomposing at one scale is not the same as **exchanging information across scales**.
- Predicting from only one scale cannot fully exploit complementary scale-specific forecasting information.

The research target is therefore **multiscale feature extraction AND multiscale future fusion**, not just hierarchical pooling.

## 3. Motivating observation `[SOURCE, §1, pp.1–2]`

The paper uses intuitive multi-resolution phenomena:

1. Hourly traffic reveals variation within a day.
2. Daily traffic suppresses intraday detail but may highlight holiday-related effects.
3. Yearly-averaged macroeconomic variables emphasize slow macro trends.
4. Distinct resolutions emphasize distinct causal timescales or observational regularities.

The authors infer that temporal changes are naturally easier to disentangle after changing sampling scales. This is the conceptual basis, not a mathematical proof of optimal decomposition.

## 4. Limits of preceding paradigms `[SOURCE, §1–2, pp.1–3]`

Two important predecessors:

- **Series decomposition:** Autoformer, FEDformer, DLinear and MICN separate trend and seasonal-like residual fluctuations.
- **Multiperiodicity:** N-BEATS, FiLM and TimesNet characterize multiple cycles/periods.

Prior multiscale structures (Pyraformer, SCINet, and related work) provide temporal resolutions but, according to the authors, do not combine **separate multiscale past extraction and multiscale future prediction** in the way TimeMixer does.

The paper intentionally does **not** claim to invent moving-average decomposition, average pooling, MLP mixing, or multiscale forecasting individually.

## 5. Exact novelty relative to predecessors `[SOURCE/CRITIQUE]`

**Existing building blocks:** downsampling; moving-average decomposition from Autoformer; linear temporal transformations; residual connections; two-layer GELU feedforward networks; ensemble of predictors.

**Novel arrangement claimed by this paper:**

1. Multiscale observations at successive resolutions.
2. Decompose **each scale**, not merely original series.
3. Seasonal-like branch mixed **fine-to-coarse**.
4. Trend branch mixed **coarse-to-fine**.
5. Cross-scale interactions repeated in stacked PDM blocks.
6. Predict the same future from **each scale**, then combine them with FMM.

Thus the innovation is mainly in **information-flow direction, decomposition-aware routing across scales, and two-phase multiscale use**, not a wholly new atomic neural primitive.

## 6. Main contributions from authors `[SOURCE, §1, p.2]`

- Formulate complex forecasting as a **multiscale-mixing** problem.
- Introduce **Past-Decomposable-Mixing (PDM)** and **Future-Multipredictor-Mixing (FMM)**.
- Demonstrate competitiveness over 18 benchmarks and 15 listed main baselines covering long- and short-term forecasting, with lower resource usage.

The third item is an empirical claim **within the paper's benchmark protocols**; it is not a claim of financial-market superiority.

## 7. The paper's research-positioning template `[SOURCE/INTERPRETATION]`

```text
Traditional decomposition / periodicity techniques
                    ↓
Different resolutions reveal different temporal patterns
                    ↓
Past representation needs directional cross-scale exchange
                    ↓
PDM: seasonal bottom-up; trend top-down
                    ↓
Prediction requires contributions from all scales
                    ↓
FMM: scale-specific forecasting and integration
                    ↓
Accuracy + computational efficiency + interpretability studies
```

This is an example of motivating a multi-part architecture with **two distinct deficiencies**, one for representation and one for prediction.

## 8. Original problem setting is not stock ranking `[SOURCE, §3 p.3]`

Observed time series:

$$
\mathbf{x}\in\mathbb{R}^{P\times C},
$$

with:

- $P$: historical input length;
- $C$: number of observed variates/channels;
- $F$: forecast horizon (number of future time points).

Target output:

$$
\hat{\mathbf{x}}\in\mathbb{R}^{F\times C}.
$$

The output is a **sequence of future values** across all channels. It is not a cross-sectional scalar score for each asset.

## 9. Formal notation table `[SOURCE / DERIVATION]`

| Symbol | Meaning | Typical shape |
|---|---|---|
| $P$ | input lookback length | scalar integer |
| $F$ | prediction horizon | scalar integer |
| $C$ | observed variates | scalar integer |
| $M$ | highest downsampling level (levels indexed $0,\dots,M$) | scalar integer |
| $L$ | number of stacked PDM blocks | scalar integer |
| $d_{model}$ | hidden representation dimension | scalar integer |
| $\mathbf{x}_0$ | original finest-scale input | $[P,C]$ |
| $\mathbf{x}_m$ | downsampled input at level $m$ | $[\lfloor P/2^m\rfloor,C]$ |
| $\mathcal X^\ell$ | set of all scale embeddings at block $\ell$ | levels $0..M$ |
| $\mathbf{s}_m^\ell$ | season/residual-like component at scale $m$ | $[P_m,d_{model}]$ |
| $\mathbf{t}_m^\ell$ | smooth trend component at scale $m$ | $[P_m,d_{model}]$ |
| $\hat{\mathbf{x}}_m$ | forecast generated by scale $m$ | $[F,C]$ |
| $\hat{\mathbf{x}}$ | final sum of scale forecasts | $[F,C]$ |

Important collision prevention: **$M$ is a scale-depth index, so there are $M+1$ scale levels**. The paper's parameter table lists `M=3` for long-horizon tasks and `M=1` for short-horizon tasks.

## 10. Architecture overview `[SOURCE, Fig.1 p.4]`

```text
Original observed sequence x [P,C]
           │
           ▼
  Multiscale average pooling
           │
      x0, x1, ..., xM
           │
           ▼
    Scale-wise embeddings
           │
           ▼
     PDM block × L
   ┌───────┴─────────┐
   │                 │
 Season/residual    Trend
   │                 │
 Fine → coarse     Coarse → fine
   │                 │
   └───────┬─────────┘
       Residual + FFN
           │
           ▼
 Mixed features at EVERY scale
           │
           ▼
    FMM: one head per scale
   xhat0, xhat1, ..., xhatM
           │
        Summation
           │
           ▼
 Future prediction xhat [F,C]
```

The model has **no token self-attention in the core computation**.

---

# Part II — Multiscale Representation and Core Formulas

## 11. Average-pooling hierarchy `[SOURCE, §3.1 p.3]`

Input:

$$
\mathbf x\in\mathbb R^{P\times C}.
$$

Downsample at powers of two:

$$
\mathcal X=\{\mathbf x_0,\dots,\mathbf x_M\},\qquad
\mathbf x_m\in\mathbb R^{\lfloor P/2^m\rfloor\times C},
\qquad\mathbf x_0=\mathbf x.
$$

- $m=0$: highest temporal resolution; every original observation retained.
- $m=M$: lowest resolution; coarse/macroscopic variations emphasized.
- The default construction is average pooling, not FFT, a learned convolution, or VQ.

## 12. One-step downsampling intuition `[DERIVATION]`

For a simple non-overlapping average pooling with stride 2, an illustrative step is:

$$
\mathbf x_{m+1}[t,:]
\approx\frac{\mathbf x_m[2t,:]+\mathbf x_m[2t+1,:]}{2}.
$$

This is an **implementation-oriented illustration**, not a verbatim paper equation. Actual padding, handling of odd lengths, and exact pooling options should be checked in the official repository. Crucially, averaging can attenuate high-frequency variation; it does not guarantee all economically relevant low-frequency content is preserved.

## 13. Resolution example `[DERIVATION]`

With $P=96$ and $M=3$:

| Scale level | Approximate sequence length | Relative resolution |
|---|---:|---|
| 0 | 96 | original/fine |
| 1 | 48 | 2× coarser |
| 2 | 24 | 4× coarser |
| 3 | 12 | 8× coarser |

For very short history $P=8$ and $M=3$, the coarsest length is 1. For financial transfer this is a warning, not a recommendation to use the original $M$.

## 14. Scale-wise embedding `[SOURCE, §3.1 p.3]`

The model projects multiscale inputs into deep representations:

$$
\mathcal X^0=\operatorname{Embed}(\mathcal X).
$$

For each scale:

$$
\mathbf x_m^0\in\mathbb R^{P_m\times d_{model}},\qquad
P_m=\lfloor P/2^m\rfloor.
$$

The same scale count remains after PDM layers. The paper does not print a complete embedding-weight equation in the main text; the exact module implementation is best verified in code.

## 15. Stacked PDM recurrence `[SOURCE, Eq.(1), p.4]`

$$
\mathcal X^{\ell}=\operatorname{PDM}(\mathcal X^{\ell-1}).
\tag{1}
$$

Each PDM block performs decomposition, two distinct cross-scale passes, and a residual feature update. Default long-horizon setting has **two PDM blocks** (Appendix A, p.13/Table 7 p.14).

**Index note:** the paper's printed Eq.(1) uses an unconventional layer index range in its prose. For implementation, interpret blocks as $\ell=1,\ldots,L$, starting from $\mathcal X^0=Embed(\mathcal X)$. This is notation clarification, not a correction to printed data.

## 16. Seasonal-trend decomposition at every scale `[SOURCE, Eq.(3), p.4]`

$$
\big(\mathbf s_m^\ell,\mathbf t_m^\ell\big)
=\operatorname{SeriesDecomp}(\mathbf x_m^\ell),
\quad m=0,\ldots,M.
\tag{3a}
$$

Uses an Autoformer-style series decomposition block based on **moving average**.

- $\mathbf t_m^\ell$: smoothed trend estimate.
- $\mathbf s_m^\ell$: residual often called *seasonal* in time-series deep-learning literature.
- The split is repeated **independently at each scale**.
- Seasonal here does **not** imply the residual is demonstrably periodic for every input.

## 17. Moving-average decomposition equations `[DERIVATION]`

A conventional expression consistent with the named Autoformer mechanism is:

$$
\mathbf t_m=\operatorname{MovingAverage}(\mathbf x_m),\qquad
\mathbf s_m=\mathbf x_m-\mathbf t_m.
$$

This is an explanatory expansion of `SeriesDecomp` used in the source. Kernel and boundary/padding implementation require code verification; the paper does not specify every such setting in §3.2.

## 18. PDM composite equation `[SOURCE, Eq.(3), p.4]`

The paper composes seasonal and trend branches and adds a residual:

$$
\mathcal X^\ell
=
\mathcal X^{\ell-1}
+
\operatorname{FeedForward}
\left(
\operatorname{S\!\!\text{-}Mix}\big(\{\mathbf s_m^\ell\}_{m=0}^{M}\big)
+
\operatorname{T\!\!\text{-}Mix}\big(\{\mathbf t_m^\ell\}_{m=0}^{M}\big)
\right).
\tag{3b}
$$

The sum inside the feedforward is interpreted **scale-wise** with compatible dimensions, not as a naive sum of tensors of different sequence lengths.

`FeedForward` consists of two linear layers with intermediate GELU, and mixes channels inside each scale. The main **cross-scale** operators are the temporal-dimension Bottom-Up-Mixing and Top-Down-Mixing MLPs.

## 19. Why two branches? `[SOURCE, §3.2 pp.4–5]`

The paper reasons:

- Fine-scale *seasonal* information encodes useful detailed fluctuations that can improve coarse representations.
- Coarse-scale *trend* information is less contaminated by fine-grained variations and can regularize fine-scale representations.

Therefore one mixing direction for all features is not necessarily appropriate.

**Important distinction:** These are design hypotheses empirically supported by ablations, not a theorem that trends universally must flow top-down.

## 20. Bottom-up seasonal mixing `[SOURCE, Eq.(4), p.5]`

For $m=1,\dots,M$:

$$
\mathbf s_m^\ell
\leftarrow
\mathbf s_m^\ell
+
\operatorname{BottomUpMix}
(\mathbf s_{m-1}^\ell).
\tag{4}
$$

Data flows:

```text
scale 0 (finest)
  ↓ bottom-up MLP
scale 1
  ↓ bottom-up MLP
scale 2
  ↓ bottom-up MLP
scale M (coarsest)
```

Each coarse seasonal component receives transformed information from the previous finer scale. The update is residual. Computational path is sequential across scales.

## 21. Bottom-up operator shapes `[SOURCE, p.5; DERIVATION]`

The Bottom-Up-Mixing operator is **two temporal linear layers separated by GELU**. It maps:

$$
\mathbb R^{P_{m-1}\times d_{model}}
\longrightarrow
\mathbb R^{P_m\times d_{model}}.
$$

For a single hidden channel, an illustrative factorization is:

$$
\operatorname{BUM}_m(\mathbf s)=
W_{m,2}\operatorname{GELU}(W_{m,1}\mathbf s+b_{m,1})+b_{m,2}
$$

with output length $P_m$ and input length $P_{m-1}$ after appropriate axis arrangement. Matrix sizes and implementation transposes depend on code conventions. The exact internal hidden width of the temporal MLP is not specified in the main equations.

## 22. What bottom-up does NOT mean `[CRITIQUE]`

- It does not mean a cross-stock graph with directed arrows.
- It does not imply the original input sequence is directly predicted at all resolutions here; this is the **past mixing** stage.
- It does not mean seasonal information only lives in the fine scale; each scale is decomposed independently.
- It does not prove the final residual is an identifiable economic seasonal factor.

## 23. Top-down trend mixing `[SOURCE, Eq.(5), p.5]`

For $m=M-1,\dots,0$:

$$
\mathbf t_m^\ell
\leftarrow
\mathbf t_m^\ell
+
\operatorname{TopDownMix}
(\mathbf t_{m+1}^\ell).
\tag{5}
$$

Data flow:

```text
scale M (coarsest trend)
  ↓ top-down MLP
scale M-1
  ↓ top-down MLP
scale M-2
  ↓ top-down MLP
scale 0 (finest trend)
```

Each finer trend gains a transformed macro-trend contribution from its next coarser neighbor.

## 24. Top-down operator shapes `[SOURCE, p.5; DERIVATION]`

The Top-Down-Mixing operator uses two temporal-dimension linear layers and GELU:

$$
\mathbb R^{P_{m+1}\times d_{model}}
\longrightarrow
\mathbb R^{P_m\times d_{model}}.
$$

The direction changes temporal length in the **opposite** sense from Bottom-Up-Mixing. This is a learnable cross-resolution projection, not the fixed interpolation or bilinear upsampling of conventional image pyramids.

## 25. PDM output and residual update `[SOURCE/DERIVATION]`

At the end of a PDM block:

1. Each scale still has one sequence embedding.
2. The seasonal pathway has integrated fine-to-coarse information.
3. The trend pathway has integrated coarse-to-fine information.
4. Scale-wise seasonal/trend results are combined and passed through FeedForward.
5. The output is added to the original scale representation by a residual connection.

This is why TimeMixer learns interactions **between resolutions without discarding the finest-resolution stream**.

## 26. Two-stage meaning: extraction vs prediction, not two-stage optimizer `[CRITIQUE]`

The paper speaks of:

- past extraction via PDM;
- future prediction via FMM.

This describes **two processing phases within one forward computation graph**, not FactorVQVAE or PRISM-VQ's *Stage 1 train, freeze, Stage 2 train* regimen. TimeMixer is trained end-to-end under the task loss.

## 27. PDM pseudocode `[DERIVATION; NOT source repository]`

```python
# Input: list of scale features, x_scales[m]: [B, P_m, d_model]
def illustrative_pdm(x_scales):
    season, trend = [], []
    for x in x_scales:
        t = moving_average(x)  # boundary padding must match released code
        s = x - t
        season.append(s)
        trend.append(t)

    # Seasonal: fine -> coarse
    for m in range(1, len(season)):
        season[m] = season[m] + bottom_up_mlp[m](season[m-1])

    # Trend: coarse -> fine
    for m in range(len(trend)-2, -1, -1):
        trend[m] = trend[m] + top_down_mlp[m](trend[m+1])

    return [
        original + feed_forward(s + t)
        for original, s, t in zip(x_scales, season, trend)
    ]
```

This is conceptual and deliberately avoids claiming exact axis placement, normalization, dropout, implementation classes, or missing-layer-specific quirks from the released code.

---

# Part III — Future-Multipredictor-Mixing and Complete Training

## 28. FMM motivation `[SOURCE, §3.3 p.5]`

Even after good past representations are learned, the future may contain complementary:

- fine patterns accessible to fine scale;
- slowly varying macro patterns accessible to coarse scales.

Predicting from only the most detailed representation wastes the other representations' predictive capabilities.

## 29. FMM equation `[SOURCE, Eq.(6), p.5]`

For each scale $m$:

$$
\hat{\mathbf x}_m
=
\operatorname{Predictor}_m(\mathbf x_m^L),
\qquad m=0,\ldots,M.
\tag{6a}
$$

Combine:

$$
\hat{\mathbf x}
=
\sum_{m=0}^{M}\hat{\mathbf x}_m.
\tag{6b}
$$

Each $\hat{\mathbf x}_m$ has **the same target shape** $[F,C]$, irrespective of its input resolution.

## 30. One predictor per scale `[SOURCE, §3.3 p.5]`

A `Predictor_m`:

1. Uses a **single linear layer across the historical temporal dimension** to map $P_m\to F$.
2. Projects deep hidden features into the original $C$ variates.

Hence it is a learned scale-specific time-to-future projection rather than a decoder that autoregressively generates each next time point.

## 31. Important: FMM is sum, not learned softmax gating `[SOURCE/CRITIQUE]`

The official form is **sum ensemble** of scale forecasts. The paper does **not** use a regime-aware router or an explicit per-scale softmax gate in the main design.

Appendix F.4 tests average ensemble and finds similar results. A research Agent must not confuse this with Mixture-of-Experts routing.

## 32. FMM tensor trace `[DERIVATION]`

Example $P=96,M=3,F=24,C=7,d_{model}=16$:

```text
x0: [B,96,16] ---> Predictor0 ---> yhat0: [B,24,7]
x1: [B,48,16] ---> Predictor1 ---> yhat1: [B,24,7]
x2: [B,24,16] ---> Predictor2 ---> yhat2: [B,24,7]
x3: [B,12,16] ---> Predictor3 ---> yhat3: [B,24,7]
                                           │
                  final yhat = yhat0 + ... + yhat3
                                           │
                                        [B,24,7]
```

The shapes illustrate the mathematical architecture and do not imply the model paper used this exact combination in any particular table.

## 33. FMM pseudocode `[DERIVATION]`

```python
# input levels already passed through L PDM blocks
preds = []
for m, features in enumerate(mixed_scales):
    y_m = predictors[m](features)  # output [B, F, C]
    preds.append(y_m)
y_hat = sum(preds)
```

## 34. Sum vs average mathematically `[DERIVATION / SOURCE Appendix F.4 p.20]`

For $M+1$ scales:

$$
\hat x_{sum}=\sum_m\hat x_m,
\qquad
\hat x_{avg}=\frac{1}{M+1}\sum_m\hat x_m.
$$

If predictor outputs can rescale freely, a learned final-layer multiplicative constant can compensate for the difference. The paper finds practically similar outcomes, but there is no guarantee identical optimization dynamics or regularization in every setting.

## 35. Forecasting loss `[SOURCE, §4 and Appendix A pp.6,13]`

TimeMixer trains forecasting models with **L2 / MSE** in general long-horizon and PEMS settings; the Appendix parameter table labels the M4 task loss as **SMAPE**.

A schematic standard pointwise objective is:

$$
\mathcal L_{MSE}
=
\frac{1}{BFC}
\sum_{b=1}^{B}
\sum_{t=1}^{F}
\sum_{c=1}^{C}
(\hat x_{b,t,c}-x_{b,t,c})^2.
$$

This normalization is a standard explicit rendering of MSE, not an independently printed source equation. **Do not report a single uniform MSE training loss for M4**; Table 7 lists SMAPE there.

## 36. End-to-end model pseudocode `[DERIVATION]`

```python
# x: [B, P, C], not a cross-sectional stock batch by construction
levels = downsample_by_2(x, levels=M+1)
levels = [embedding(level) for level in levels]

for _ in range(L):
    levels = illustrative_pdm(levels)

preds = [predictor[m](levels[m]) for m in range(M+1)]
y_hat = sum(preds)  # each predictor produces [B, F, C]

loss = mse(y_hat, target)  # long-term/PEMS; M4 config differs
loss.backward()
optimizer.step()
```

## 37. What can be trained jointly? `[SOURCE/INTERPRETATION]`

The framework's parameters include scale embedding layers, seasonal mixing MLPs, trend mixing MLPs, residual FFNs, and each scale-specific predictor. The paper does not describe freezing one of these groups before training another. End-to-end task supervision couples all contributions through the **final fused prediction**.

## 38. One shared supervision signal for all future predictors `[SOURCE, Appendix F.4 p.20]`

The paper explicitly states that the loss is applied to the **ensemble result**, rather than a separate independent loss for each scale's predictor.

Important implications:

- A scale can learn to contribute a complementary residual rather than fitting the complete target alone.
- Individual predictor outputs in Fig. 4 need not have the absolute amplitude of the full future trajectory.
- A naive adaptation that supervises **each scale separately** changes the original learning objective.

## 39. Architectural economy and temporal linear layers `[SOURCE/CRITIQUE]`

The paper uses simple linear layers for cross-resolution temporal transformations. Consequences:

- no quadratic time-token attention required;
- few primitive operator types;
- weights can nevertheless grow with input temporal length;
- parameter sharing across input lengths is **not inherently guaranteed** by learned fixed-size temporal linear projections.

The last point is also reflected in the explicit Limitations section, Appendix J p.22.

---
# Part IV — Original Experimental Protocol, Data and Comparators

## 40. Complete evaluation coverage `[SOURCE, §4 p.5; Table 1 p.6]`

The paper reports **18 benchmark groupings** across three settings:

- Eight long-term forecasting datasets: four ETT subsets, Weather, Solar-Energy, Electricity, Traffic.
- Four multivariate short-term PeMS traffic datasets: PEMS03, PEMS04, PEMS07, PEMS08.
- Six M4 subsets of differing frequencies: yearly, quarterly, monthly, weekly, daily, hourly.

The 18 are **dataset/subset categories**, not 18 distinct financial markets or 18 different national time-series collections.

## 41. Original long-term benchmark inventory `[SOURCE, Table 1 p.6; Appendix Table 6 p.12]`

| Dataset | Channels $C$ | Frequency | Forecast horizons | Description |
|---|---:|---|---|---|
| ETTh1 | 7 | hourly in standard ETT naming | 96/192/336/720 | electrical transformer temperature |
| ETTh2 | 7 | hourly in standard ETT naming | 96/192/336/720 | electrical transformer temperature |
| ETTm1 | 7 | 15 min | 96/192/336/720 | electrical transformer temperature |
| ETTm2 | 7 | 15 min | 96/192/336/720 | electrical transformer temperature |
| Weather | 21 | 10 min | 96/192/336/720 | weather variables |
| Solar-Energy | 137 | 10 min | 96/192/336/720 | solar-energy series |
| Electricity | 321 | hourly | 96/192/336/720 | electricity consumption |
| Traffic | 862 | hourly | 96/192/336/720 | traffic occupancy |

**Source fidelity:** Appendix Table 6 prints `15 min` even for ETTh rows, despite the standard `h` naming; this looks like a table typo. The discrepancy is recorded here, not silently treated as two independent truths. The task's exact calendar span is not provided in one universal train/valid/test date table in the source.

## 42. Long-term sample sizes / partitions `[SOURCE, Appendix Table 6 p.12]`

The original table reports the following **dataset-size triples**. These are counts, not dates; do not invent date cutoffs from them.

| Dataset | Train | Validation | Test | Exact calendar dates in paper? |
|---|---:|---:|---:|---|
| ETTm1 | 34,465 | 11,521 | 11,521 | not tabulated as dates |
| ETTm2 | 34,465 | 11,521 | 11,521 | not tabulated as dates |
| ETTh1 | 8,545 | 2,881 | 2,881 | not tabulated as dates |
| ETTh2 | 8,545 | 2,881 | 2,881 | not tabulated as dates |
| Electricity | 18,317 | 2,633 | 5,261 | not tabulated as dates |
| Traffic | 12,185 | 1,757 | 3,509 | not tabulated as dates |
| Weather | 36,792 | 5,271 | 10,540 | not tabulated as dates |
| Solar-Energy | 36,601 | 5,161 | 10,417 | not tabulated as dates |

The source does not explain how every count maps to windows after lookback/horizon extraction; **do not reinterpret counts as number of independent forecast origins without verifying code**.

## 43. Short-term PeMS data `[SOURCE, Tables 1/6 pp.6/12]`

| Dataset | Channels | Frequency | Horizon | Train / Validation / Test counts as printed |
|---|---:|---|---:|---|
| PEMS03 | 358 | 5 min | 12 | 15,701 / 5,216 / 434 |
| PEMS04 | 307 | 5 min | 12 | 10,172 / 3,375 / 281 |
| PEMS07 | 883 | 5 min | 12 | 16,911 / 5,622 / 468 |
| PEMS08 | 170 | 5 min | 12 | 10,690 / 3,548 / 265 |

**Verify in code:** the appendix's small third entries are presented as printed; if a reproduction disagrees in dataset size, inspect provider and split logic rather than assuming the paper and code align exactly.

## 44. M4 benchmark and special protocol `[SOURCE, Tables 1/4/6]`

The six M4 subsets contain **univariate** forecasting problems at different frequencies.

| M4 series group | Prediction horizon in source Appendix Table 6 | Frequency | Role |
|---|---:|---|---|
| Yearly | 6 | yearly | short-term univariate |
| Quarterly | 8 | quarterly | short-term univariate |
| Monthly | 18 | monthly | short-term univariate |
| Weekly (source table typesets “Weakly”) | 13 | weekly | short-term univariate |
| Daily | 14 | daily | short-term univariate |
| Hourly | 48 | hourly | short-term univariate |

M4 has its own **SMAPE / MASE / OWA** protocol and model configuration. The M4 overall result is a **weighted average across groups**, not the arithmetic average of the six visible values.

## 45. Forecastability metric `[SOURCE, Table 1 p.6; Appendix Table 6 p.12]`

The authors report a dataset-level property called *forecastability* defined using **one minus an entropy-based measure in the Fourier domain** (Goerg 2013).

- High values indicate more easily forecastable datasets under that definition.
- Examples stated in Table 1: Weather 0.75, Solar-Energy 0.33, Electricity 0.77, Traffic 0.68.
- These are descriptive statistics, **not training/evaluation scores** of TimeMixer.
- The paper does not derive the complete spectral entropy estimator in the main body; do not fabricate its implementation.

## 46. Historical window and forecast horizon choices `[SOURCE, §4 p.6]`

For main long-term benchmark **Table 2**:

$$
P=96,\qquad F\in\{96,192,336,720\}.
$$

The table shows average error across four horizons. Appendix Table 13 gives each horizon separately.

For PeMS:

$$
P=96,\qquad F=12.
$$

For M4, forecast horizon and input lengths differ by frequency, and the authors follow the short-term M4 settings.

## 47. Why unified input lengths matter `[SOURCE, §4 p.6]`

The authors point out that forecasting papers often report numbers under different lookback windows and hyperparameter searches. To control this:

1. Main experiments fix input length for long-term benchmarks.
2. Results reported in the main body are averages of **three experimental runs**.
3. Appendix E separately presents **broader per-model hyperparameter searches** as an alternate comparison protocol.

A research Agent must distinguish these two protocols when interpreting “TimeMixer vs PatchTST.”

## 48. Model comparators `[SOURCE, §4 p.6]`

Main baseline list contains:

1. PatchTST
2. TimesNet
3. SCINet
4. Crossformer
5. MICN
6. FiLM
7. DLinear
8. LightTS
9. FEDformer
10. Stationary Transformer
11. Pyraformer
12. Autoformer
13. Informer
14. N-HiTS
15. N-BEATS

Appendix G adds comparisons with Scaleformer, MTSMixer and TSMixer. “15 baselines” refers to the primary collection; it does not mean exactly 15 baselines are present in every single benchmark table.

## 49. What these baseline families test `[SOURCE/INTERPRETATION]`

| Family | Examples | Counter-argument TimeMixer tests |
|---|---|---|
| Transformer temporal attention | PatchTST, Informer, Autoformer, FEDformer | attention not intrinsically required |
| Multi-resolution Transformer | Crossformer, Pyraformer | resolution alone may be insufficient |
| Decomposition / spectral | MICN, FiLM, FEDformer | decompose + directed mixing may help |
| Efficient MLP/linear | DLinear, LightTS | simple MLP is a serious baseline |
| Multi-periodic patterns | TimesNet | periodicity and multiscale mixing not identical |
| Short-horizon baselines | SCINet, N-HiTS, N-BEATS | evaluate beyond long-horizon forecasting |

These are **functional descriptions**, not a statement that each baseline was recreated identically with the same parameter count.

## 50. Metric families by task `[SOURCE, Appendix A pp.12–13]`

| Task | Metric | Direction | Meaning |
|---|---|---|---|
| Long-term | MSE | lower | squared error over forecast values |
| Long-term | MAE | lower | absolute error over forecast values |
| PeMS | MAE | lower | absolute error |
| PeMS | MAPE | lower | relative absolute error percentage |
| PeMS | RMSE | lower | root mean squared error |
| M4 | SMAPE | lower | symmetric percentage error |
| M4 | MASE | lower | error scaled by naive seasonal forecast |
| M4 | OWA | lower | weighted combination of SMAPE and MASE relative to M4 naive baseline |

**Not present:** stock IC, RankIC, information coefficient decay, excess annualized return, Sharpe, maximum drawdown.

## 51. MSE, MAE, RMSE and percentage metrics `[DERIVATION; SOURCE Appendix A]`

Standard normalized forms:

$$
\mathrm{MSE}=\frac1{FC}\sum_{t,c}(y_{t,c}-\hat y_{t,c})^2,
$$

$$
\mathrm{MAE}=\frac1{FC}\sum_{t,c}|y_{t,c}-\hat y_{t,c}|,
$$

$$
\mathrm{RMSE}=\sqrt{\frac1{FC}\sum_{t,c}(y_{t,c}-\hat y_{t,c})^2}.
$$

$$
\mathrm{MAPE}=\frac{100}{FC}\sum_{t,c}
\left|\frac{y_{t,c}-\hat y_{t,c}}{y_{t,c}}\right|.
$$

$$
\mathrm{SMAPE}=\frac{200}{FC}\sum_{t,c}
\frac{|y_{t,c}-\hat y_{t,c}|}{|y_{t,c}|+|\hat y_{t,c}|}.
$$

The original appendix prints one-dimensional forecast-index notation and some formulas with compressed/misaligned normalizers in PDF text extraction; the above express conventional multi-channel normalized equivalents, **not necessarily byte-for-byte transcription**. Zero denominators require implementation conventions not fully discussed in the paper.

## 52. MASE and OWA `[SOURCE, Appendix A p.13; DERIVATION]`

For seasonal period $s$:

$$
\mathrm{MASE}
=\frac{\mathrm{MAE}_{forecast}}
{\mathrm{MAE}_{naive\ seasonal}},
$$

where the denominator uses in-sample changes at seasonal lag $s$. The exact M4 training series and naive scale must be carried through when reproducing.

The appendix defines:

$$
\mathrm{OWA}
=\frac12\left[
\frac{\mathrm{SMAPE}}{\mathrm{SMAPE}_{\mathrm{Naive2}}}
+
\frac{\mathrm{MASE}}{\mathrm{MASE}_{\mathrm{Naive2}}}
\right].
$$

An OWA below one indicates improvement relative to that specific M4 naive reference under the competition definition.

## 53. Implementation technology `[SOURCE, §4 p.6; Appendix A p.13]`

- PyTorch implementation.
- NVIDIA A100 80GB GPU for paper's primary runs.
- Adam with usual $\beta_1=0.9,\beta_2=0.999$ (Table 7).
- Training repeated **three times**.
- Learning rate selected as $10^{-2}$ or $10^{-3}$ depending on dataset.
- The paper summarizes batch sizes, model size and training epochs per dataset in Table 7.

## 54. Training model configuration — complete Table 7 `[SOURCE, p.14]`

The table reports `M` as scale depth, `Layers` as PDM stack depth, and `d_model` as width.

| Dataset | M | PDM layers | d_model | Initial LR | Loss | Batch size | Epochs |
|---|---:|---:|---:|---:|---|---:|---:|
| ETTh1 | 3 | 2 | 16 | 1e-2 | MSE | 128 | 10 |
| ETTh2 | 3 | 2 | 16 | 1e-2 | MSE | 128 | 10 |
| ETTm1 | 3 | 2 | 16 | 1e-2 | MSE | 128 | 10 |
| ETTm2 | 3 | 2 | 32 | 1e-2 | MSE | 128 | 10 |
| Weather | 3 | 2 | 16 | 1e-2 | MSE | 128 | 20 |
| Electricity | 3 | 2 | 16 | 1e-2 | MSE | 32 | 20 |
| Solar-Energy | 3 | 2 | 128 | 1e-2 | MSE | 32 | 20 |
| Traffic | 3 | 2 | 32 | 1e-2 | MSE | 8 | 20 |
| PeMS | 1 | 5 | 128 | 1e-3 | MSE | 32 | 10 |
| M4 | 1 | 4 | 32 | 1e-2 | SMAPE | 128 | 50 |

**Important:** Do not lazily use the “default 2 PDM layers” statement for PeMS (5) or M4 (4). They differ explicitly in the experimental configuration.

## 55. Known training details that are NOT directly specified `[SOURCE/CRITIQUE]`

The PDF supplies LR, epochs, batch size, width, Adam, GPU, layer count and scale depth. It does **not** provide a full universally applicable code-level recipe for:

- per-dataset padding decisions in every pooling operation;
- every temporal-mixer hidden width and boundary policy;
- exact data-loader shuffle / batching details;
- all preprocessing scalar transformations and provider versions;
- seed integers for the three primary runs;
- every baseline's independently selected scheduler / weight decay / patience;
- full run-specific checkpoint selection details;
- the complete M4 competition aggregation implementation.

For exact reproduction, use the repository; do **not** manufacture missing values.

## 56. Main long-term result is horizon-averaged `[SOURCE, Table 2 p.6]`

**Table 2** reports mean MSE/MAE over four distinct future horizons: 96, 192, 336 and 720. Its row `Weather MSE=0.240` does **not** mean forecast horizon 96; that horizon's error is a different row in Appendix Table 13.

## 57. Long-term results: complete Table 2, 8 datasets × 11 models `[SOURCE, p.6]`

Each cell is `MSE / MAE`; lower is better. All models use the unified-input protocol and the table averages over four forecast horizons. The table contains *one summary pair* per dataset/model.

| Dataset | TimeMixer | PatchTST | TimesNet | Crossformer | MICN | FiLM | DLinear | FEDformer | Stationary | Autoformer | Informer |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Weather | **0.240/0.271** | 0.265/0.285 | 0.251/0.294 | 0.264/0.320 | 0.268/0.321 | 0.271/0.291 | 0.265/0.315 | 0.309/0.360 | 0.288/0.314 | 0.338/0.382 | 0.634/0.548 |
| Solar-Energy | **0.216/0.280** | 0.287/0.333 | 0.403/0.374 | 0.406/0.442 | 0.283/0.358 | 0.380/0.371 | 0.330/0.401 | 0.328/0.383 | 0.350/0.390 | 0.586/0.557 | 0.331/0.381 |
| Electricity | **0.182/0.272** | 0.216/0.318 | 0.193/0.304 | 0.244/0.334 | 0.196/0.309 | 0.223/0.302 | 0.225/0.319 | 0.214/0.327 | 0.193/0.296 | 0.227/0.338 | 0.311/0.397 |
| Traffic | **0.484/0.297** | 0.529/0.341 | 0.620/0.336 | 0.667/0.426 | 0.593/0.356 | 0.637/0.384 | 0.625/0.383 | 0.610/0.376 | 0.624/0.340 | 0.628/0.379 | 0.764/0.416 |
| ETTh1 | **0.447/0.440** | 0.516/0.484 | 0.495/0.450 | 0.529/0.522 | 0.475/0.480 | 0.516/0.483 | 0.461/0.457 | 0.498/0.484 | 0.570/0.537 | 0.496/0.487 | 1.040/0.795 |
| ETTh2 | **0.364/0.395** | 0.391/0.411 | 0.414/0.427 | 0.942/0.684 | 0.574/0.531 | 0.402/0.420 | 0.563/0.519 | 0.437/0.449 | 0.526/0.516 | 0.450/0.459 | 4.431/1.729 |
| ETTm1 | **0.381/0.395** | 0.406/0.407 | 0.400/0.406 | 0.513/0.495 | 0.423/0.422 | 0.411/0.402 | 0.404/0.408 | 0.448/0.452 | 0.481/0.456 | 0.588/0.517 | 0.961/0.734 |
| ETTm2 | **0.275/0.323** | 0.290/0.334 | 0.291/0.333 | 0.757/0.610 | 0.353/0.402 | 0.287/0.329 | 0.354/0.402 | 0.305/0.349 | 0.306/0.347 | 0.327/0.371 | 1.410/0.810 |

**Audit rule:** Values from this table must not be compared directly to PatchTST's original ICLR 2023 table, because input-length and tuning choices are different. Use the matched Table 2 protocol to compare.

## 58. Representative gains vs PatchTST `[SOURCE/DERIVATION]`

Main Table 2 MSE:

- Weather: $0.265\to0.240$; reduction $\approx9.43\%$.
- Solar-Energy: $0.287\to0.216$; reduction $\approx24.74\%$.
- Electricity: $0.216\to0.182$; reduction $\approx15.74\%$.
- Traffic: $0.529\to0.484$; reduction $\approx8.51\%$.
- ETTh1: $0.516\to0.447$; reduction $\approx13.37\%$.
- ETTh2: $0.391\to0.364$; reduction $\approx6.91\%$.
- ETTm1: $0.406\to0.381$; reduction $\approx6.16\%$.
- ETTm2: $0.290\to0.275$; reduction $\approx5.17\%$.

These are calculated from the paper's **rounded table summaries**; they are not new experiments.

## 59. PeMS short-term comparison — TimeMixer vs key baselines `[SOURCE, Table 3 p.7]`

All PeMS forecasts use input length 96 and prediction length 12. A compact exact comparison:

| Dataset | Metric | TimeMixer | SCINet | Crossformer | PatchTST | TimesNet |
|---|---|---:|---:|---:|---:|---:|
| PEMS03 | MAE | **14.63** | 15.97 | 15.64 | 18.95 | 16.41 |
| PEMS03 | MAPE | **14.54** | 15.89 | 15.74 | 17.29 | 15.17 |
| PEMS03 | RMSE | **23.28** | 25.20 | 25.56 | 30.15 | 26.72 |
| PEMS04 | MAE | **19.21** | 20.35 | 20.38 | 24.86 | 21.63 |
| PEMS04 | MAPE | **12.53** | 12.84 | 12.84 | 16.65 | 13.15 |
| PEMS04 | RMSE | **30.92** | 32.31 | 32.41 | 40.46 | 34.90 |
| PEMS07 | MAE | **20.57** | 22.79 | 22.54 | 27.87 | 25.12 |
| PEMS07 | MAPE | **8.62** | 9.41 | 9.38 | 12.69 | 10.60 |
| PEMS07 | RMSE | **33.59** | 35.61 | 35.49 | 42.56 | 40.71 |
| PEMS08 | MAE | **15.22** | 17.38 | 17.56 | 20.35 | 19.01 |
| PEMS08 | MAPE | **9.67** | 10.80 | 10.92 | 13.15 | 11.83 |
| PEMS08 | RMSE | **24.26** | 27.34 | 27.21 | 31.04 | 30.65 |

Full PeMS Table 3 also includes MICN, FiLM, DLinear, FEDformer, Stationary, Autoformer and Informer. Those numbers can be consulted in the exact original Table 3 p.7 if only individual comparisons are needed.

## 60. Why PeMS matters to the conceptual claim `[SOURCE/CRITIQUE]`

PeMS has hundreds of interconnected sensor channels and strong spatiotemporal relationships. The paper argues purely channel-independent PatchTST and DLinear are disadvantaged on these tasks. TimeMixer outperforms them empirically.

**Caution:** TimeMixer does not demonstrate graph-level road connectivity learning or prove that its structure captures explicit sensor causality. The result supports the **combined architectural design** on traffic forecasting, not direct financial cross-stock relevance.

## 61. M4 full weighted-average result `[SOURCE, Table 4 p.7]`

| Model | Weighted SMAPE | Weighted MASE | Weighted OWA |
|---|---:|---:|---:|
| **TimeMixer** | **11.723** | **1.559** | **0.840** |
| TimesNet | 11.829 | 1.585 | 0.851 |
| N-HiTS | 11.927 | 1.613 | 0.861 |
| N-BEATS (ensemble removed in this comparison) | 11.851 | 1.559 | 0.855 |
| PatchTST | 13.152 | 1.945 | 0.998 |
| DLinear | 13.639 | 2.095 | 1.051 |

This table is from **univariate M4**, not a comparison on multivariate Alpha158 or stocks.

## 62. M4 source subgroup results for TimeMixer `[SOURCE, Table 4 p.7]`

| Frequency group | SMAPE | MASE | OWA |
|---|---:|---:|---:|
| Yearly | 13.206 | 2.916 | 0.776 |
| Quarterly | 9.996 | 1.166 | 0.825 |
| Monthly | 12.605 | 0.919 | 0.869 |
| Others | 4.564 | 3.115 | 0.982 |
| Weighted average | **11.723** | **1.559** | **0.840** |

The visible Table 4 groups weekly/daily/hourly components under `Others`; it does not report a separate top-level result row for each such frequency group.

## 63. M4 statistical comparison `[SOURCE, Appendix C Table 11 p.15]`

Weighted average TimeMixer (3 runs):

- SMAPE: $11.723\pm0.011$;
- MASE: $1.559\pm0.022$;
- OWA: $0.840\pm0.001$.

TimesNet:

- SMAPE: $11.829\pm0.120$;
- MASE: $1.585\pm0.017$;
- OWA: $0.851\pm0.003$.

The source prints confidence level information, but not complete test type and dependence treatment in the section; the levels alone should not be over-interpreted as a general significance guarantee for all settings.

## 64. Which benchmark claims are well supported? `[CRITIQUE]`

**Supported within the published protocol:** lower aggregate forecasting error than selected contemporary baselines, strong results on short- and long-horizon tasks, low GPU/time cost, and systematic PDM/FMM ablation gains.

**Not established by this paper:** improved future stock return rankings, better trading Sharpe, improved backtest returns, robustness to equity-market regimes, or mathematical optimality of seasonal bottom-up/trend top-down mixing.

---

# Part V — Ablation Logic and Scientific Evidence

## 65. Ablation design is richer than a simple removal table `[SOURCE, Table 5 p.8]`

The study tests **ten** variants:

1. Official PDM + FMM.
2. Remove FMM multi-predictor fusion.
3. Remove seasonal cross-scale mixing.
4. Remove trend cross-scale mixing.
5. Seasonal and trend both top-down.
6. Seasonal and trend both bottom-up.
7. Seasonal top-down + trend bottom-up (reverse directions).
8. No decomposition; direct bottom-up mixing.
9. No decomposition; direct top-down mixing.
10. No decomposition and no past mixing.

This controls not only whether mixing exists, but also **whether opposite directions are meaningful**.

## 66. Complete main Table 5: M4, PeMS04, ETTm1 `[SOURCE, p.8]`

- `BU`: bottom-up (fine→coarse); `TD`: top-down (coarse→fine); `—`: absent.
- ETTm1 uses predict-336; M4/PeMS04 follow original short-horizon protocols.
- Lower is better.

| Case | Decomp | Seasonal | Trend | FMM | M4 SMAPE | M4 MASE | M4 OWA | PEMS04 MAE | PEMS04 MAPE | PEMS04 RMSE | ETTm1 MSE | ETTm1 MAE |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ① Full | ✓ | BU | TD | ✓ | **11.723** | **1.559** | **0.840** | **19.21** | **12.53** | **30.92** | **0.390** | **0.404** |
| ② w/o FMM | ✓ | BU | TD | × | 12.503 | 1.634 | 0.925 | 21.67 | 13.45 | 34.89 | 0.402 | 0.415 |
| ③ w/o seasonal mixing | ✓ | — | TD | ✓ | 13.051 | 1.676 | 0.962 | 24.49 | 16.28 | 38.79 | 0.411 | 0.427 |
| ④ w/o trend mixing | ✓ | BU | — | ✓ | 12.911 | 1.655 | 0.941 | 22.91 | 15.02 | 37.04 | 0.405 | 0.414 |
| ⑤ both TD | ✓ | TD | TD | ✓ | 12.008 | 1.628 | 0.871 | 20.78 | 13.02 | 32.47 | 0.392 | 0.413 |
| ⑥ both BU | ✓ | BU | BU | ✓ | 11.978 | 1.626 | 0.859 | 21.09 | 13.78 | 33.11 | 0.396 | 0.415 |
| ⑦ reversed | ✓ | TD | BU | ✓ | 13.012 | 1.657 | 0.954 | 22.27 | 15.14 | 34.67 | 0.412 | 0.429 |
| ⑧ no decomposition, BU | × | BU | — | ✓ | 11.975 | 1.617 | 0.851 | 21.51 | 13.47 | 34.81 | 0.395 | 0.408 |
| ⑨ no decomposition, TD | × | TD | — | ✓ | 11.973 | 1.622 | 0.850 | 21.79 | 14.03 | 35.23 | 0.393 | 0.406 |
| ⑩ no past mixing | × | — | — | ✓ | 12.468 | 1.671 | 0.916 | 24.87 | 16.66 | 39.48 | 0.405 | 0.412 |

## 67. Case ②: Is future multiscale fusion useful? `[SOURCE, Table 5]`

Case ① vs. ②:

- M4 OWA: $0.840\to0.925$ without FMM.
- PeMS04 MAE: $19.21\to21.67$ without FMM.
- ETTm1 MSE: $0.390\to0.402$ without FMM.

**Inference supported:** keeping a forecast head for each scale is useful in all three representative settings.

**Not proved:** adaptive softmax-scale gating would be worse; only the specific ablated alternative was tested.

## 68. Case ③: Seasonal mixing matters `[SOURCE, Table 5]`

Deleting seasonal mixing:

- M4 OWA rises to $0.962$.
- PEMS04 MAE rises to $24.49$.
- ETTm1 MSE rises to $0.411$.

This is among the strongest degradations in the table. It supports **fine-to-coarse information propagation for the decomposed residual-like component**.

## 69. Case ④: Trend mixing matters `[SOURCE, Table 5]`

Deleting trend mixing:

- M4 OWA $0.941$;
- PEMS04 MAE $22.91$;
- ETTm1 MSE $0.405$.

Both components provide incremental value; strong seasonal gains do not make the trend branch redundant.

## 70. Case ⑤ / ⑥: Mixing direction must be different by component `[SOURCE]`

Using the same direction for both component types leads to worse values than the official design.

- Case ⑤: both TD; PeMS04 MAE $20.78$.
- Case ⑥: both BU; PeMS04 MAE $21.09$.
- Official: PeMS04 MAE $19.21$.

**Causal isolation:** since decomposition remains enabled, this directly probes which **directional inductive bias** is suitable for each component.

## 71. Case ⑦: Full direction reversal strongly hurts `[SOURCE]`

Reverse to seasonal TD + trend BU:

- M4 OWA $0.954$;
- PEMS04 MAE $22.27$;
- ETTm1 MSE $0.412$.

It is not merely “more cross-scale communication = better”. The information-flow orientation matters.

## 72. Case ⑧ / ⑨: Decomposition is important for directed mixing `[SOURCE]`

Remove decomposition but keep BU/TD direct cross-scale mixing:

- M4 OWA $0.851$ / $0.850$;
- PEMS04 MAE $21.51$ / $21.79$;
- ETTm1 MSE $0.395$ / $0.393$.

Different component types benefit from tailored mixing; no-decomposition alternatives do not match the full model on these tasks.

## 73. Case ⑩: Entire past mixing removed `[SOURCE]`

- M4 OWA: $0.916$;
- PEMS04 MAE: $24.87$;
- ETTm1 MSE: $0.405$.

Note that ⑩ retains FMM, so this is a particularly useful contrast: **multi-scale output ensembling without effective past cross-scale interaction is insufficient**.

## 74. Biggest lessons of the ten-way ablation `[CRITIQUE]`

1. Decomposition is not alone the claimed novelty.
2. Cross-scale transfer helps, but **direction** matters.
3. Seasonal and trend components respond differently.
4. Even with PDM, future fusion supplies independent value.
5. All claims are subject to the original datasets, horizons, and architecture; they are not universal statements about financial regime dynamics.

## 75. Additional ablations over all eight long-term datasets `[SOURCE, Appendix F.1/Table 15 p.18]`

Appendix F extends the ten-case ablation to ETTh1, ETTh2, ETTm1, ETTm2, Weather, Solar, Electricity and Traffic at forecast horizon 336.

For reference, some **exact source MSE values**:

| Dataset | Full ① | w/o FMM ② | w/o seasonal ③ | w/o trend ④ | reversed ⑦ | no mixing ⑩ |
|---|---:|---:|---:|---:|---:|---:|
| ETTh1 | **0.484** | 0.493 | 0.507 | 0.491 | 0.498 | 0.502 |
| ETTh2 | **0.386** | 0.399 | 0.408 | 0.397 | 0.421 | 0.411 |
| ETTm1 | **0.390** | 0.402 | 0.411 | 0.405 | 0.412 | 0.405 |
| ETTm2 | **0.298** | 0.311 | 0.322 | 0.317 | 0.321 | 0.319 |
| Weather | **0.251** | 0.262 | 0.273 | 0.269 | 0.277 | 0.273 |
| Solar | **0.231** | 0.267 | 0.274 | 0.268 | 0.298 | 0.295 |
| Electricity | **0.185** | 0.198 | 0.207 | 0.200 | 0.221 | 0.217 |
| Traffic | **0.498** | 0.518 | 0.532 | 0.525 | 0.564 | 0.558 |

Full Table 15 also contains MAE and the remaining cases; refer to PDF p.18 for those additional exact cells.

## 76. Larger input / more scales change ablation magnitudes `[SOURCE, Appendix F.5 pp.20–21]`

The paper reruns the ten-case study with:

- PEMS04 $M=3$ instead of $M=1$;
- ETTm1 input $P=336$ instead of $P=96$.

Results show larger **relative contributions** from multiple components in many longer/more-scale settings. This is important because **multiscale benefit depends on actual temporal range and number of scales**.

## 77. Larger-scale PEMS04 selected ablation `[SOURCE, Table 21 p.21]`

| Case | PEMS04 MAE, M=3 | MAPE | RMSE |
|---|---:|---:|---:|
| Full ① | **18.10** | **11.73** | **28.51** |
| w/o FMM ② | 21.49 | 13.12 | 33.48 |
| w/o seasonal ③ | 23.68 | 16.01 | 37.42 |
| w/o trend ④ | 22.44 | 14.81 | 36.54 |
| reversed ⑦ | 22.16 | 14.60 | 35.42 |
| w/o past mixing ⑩ | 24.16 | 16.21 | 38.04 |

Increasing scale depth can improve even a short-horizon task with sufficient history, but short *stock histories* with only 8 or 20 daily points differ from the 96-point PeMS input.

## 78. ETTm1 long input selected ablation `[SOURCE, Table 21 p.21]`

| Case | ETTm1 MSE, P=336 | ETTm1 MAE |
|---|---:|---:|
| Full ① | **0.360** | **0.381** |
| w/o FMM ② | 0.375 | 0.398 |
| w/o seasonal ③ | 0.390 | 0.415 |
| w/o trend ④ | 0.386 | 0.410 |
| reversed ⑦ | 0.384 | 0.409 |
| no past mixing ⑩ | 0.401 | 0.414 |

This is another important conditional result: benefits of particular mixing choices may grow with lookback/context duration.

---
# Part VI — Alternative Implementations, Sensitivities and Negative Results

## 79. How should the Agent use appendices? `[SOURCE, Appendix F–J]`

The appendices contain **counterfactuals beyond the model the paper chose to publish as the default**:

- DFT frequency / seasonal-trend decomposition;
- strided convolution rather than average pooling;
- mean instead of sum aggregation;
- different scale counts and PDM depths;
- broader hyperparameter search;
- additional MLP and multiscale baselines;
- spectral-domain visualization;
- explicit computational and theoretical limitations.

These are not incidental: they reveal **which original design choices are convenient rather than uniquely best**.

## 80. Alternative decomposition #1 — DFT high/low split `[SOURCE, Appendix F.2 p.19]`

The authors try directly splitting signals into:

- high-frequency part, treated like seasonal;
- low-frequency part, treated like trend.

Under TimeMixer's existing directed mixing this alternative performs **worse than the official moving-average decomposition** on the three selected benchmarks.

Source examples:

| Method | ETTm1 MSE | M4 OWA | PEMS04 MAE |
|---|---:|---:|---:|
| DFT high/low decomposition | 0.392 | 0.862 | 19.83 |
| Official moving average | **0.390** | **0.840** | **19.21** |

The authors note that the original mixing directions might not be optimal for these purely high-/low-frequency components. They identify further DFT-based study as a research direction.

## 81. Alternative decomposition #2 — DFT seasonal/trend `[SOURCE, Appendix F.2 Table 18 p.19]`

This method:

1. transforms the sequence with DFT;
2. selects important frequencies;
3. reconstructs a seasonal component with inverse DFT;
4. defines trend as raw input minus the seasonal component.

**It outperforms the official moving-average-based decomposition in the reported tests.**

| Method | ETTm1 MSE | ETTm1 MAE | M4 SMAPE | M4 MASE | M4 OWA | PEMS04 MAE | PEMS04 MAPE | PEMS04 RMSE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| DFT high/low | 0.392 | 0.404 | 12.054 | 1.632 | 0.862 | 19.83 | 12.74 | 31.48 |
| **DFT seasonal/trend** | **0.383** | **0.399** | **11.673** | **1.536** | **0.824** | **18.91** | **12.27** | **29.47** |
| Moving average (official) | 0.390 | 0.404 | 11.723 | 1.559 | 0.840 | 19.21 | 12.53 | 30.92 |

The reason the authors retain moving average is not superior accuracy everywhere: they cite **simplicity, PyTorch ease of implementation, and efficiency**.

## 82. Scientific consequence of the DFT experiment `[CRITIQUE]`

Do not conclude “moving-average decomposition is optimal” from TimeMixer.

More defensible conclusions:

- Separate trend/residual mixing with correct directional flow is effective.
- How trend/residual components are extracted is a replaceable design choice.
- One DFT seasonal/trend alternative gives modestly better accuracy in a few reported benchmarks.
- Frequency decomposition may cost more or have different inductive assumptions.
- Whether it would help financial **short return sequences** is untested.

This yields a concrete transfer opportunity: **PDM mixing can be retained while replacing the decomposition operator**.

## 83. Alternative downsampling: strided convolution `[SOURCE, Appendix F.3 Table 19 p.20]`

| Downsampler | ETTm1 MSE | ETTm1 MAE | M4 SMAPE | M4 MASE | M4 OWA | PEMS04 MAE | PEMS04 MAPE | PEMS04 RMSE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Average pooling (official) | 0.390 | 0.404 | 11.723 | 1.559 | 0.840 | 19.21 | 12.53 | 30.92 |
| Strided 1D convolution | **0.387** | **0.401** | **11.682** | **1.542** | **0.831** | **19.04** | **12.17** | **29.88** |

The more elaborate learned downsampling **slightly improves** the displayed scores, but the official model uses average pooling for efficiency and simplicity.

## 84. Distinguish learned downsampling from temporal MLP `[DERIVATION/CRITIQUE]`

- **Downsampling step** chooses how a high-resolution sequence becomes coarse.
- **Cross-scale MLP** chooses how features at adjacent scales interact.

Replacing pooling with conv is not the same intervention as changing bottom-up and top-down mixing. They should have separate ablation switches in a new model.

## 85. Alternative future ensemble — sum vs average `[SOURCE, Appendix F.4 Table 20 p.20]`

| Ensemble | ETTm1 MSE | ETTm1 MAE | M4 SMAPE | M4 MASE | M4 OWA | PEMS04 MAE | PEMS04 MAPE | PEMS04 RMSE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Sum (official) | **0.390** | **0.404** | **11.723** | **1.559** | **0.840** | 19.21 | 12.53 | 30.92 |
| Average | 0.391 | 0.407 | 11.742 | 1.573 | 0.851 | **19.17** | **12.45** | **30.88** |

**Important nuance:** the sum version is not better in every PeMS04 cell: average has slightly better MAE/MAPE/RMSE on PeMS04 in the table. The authors emphasize overall similarity and functional equivalence under rescalable predictors.

## 86. Hidden research opportunity in FMM `[EXTENSION]`

TimeMixer sums scale forecasts but does not learn explicit context-dependent scale reliability. In heterogeneous markets it could be useful to predict:

$$
\hat y_{i,t}=\sum_{m=0}^{M}w_{i,t,m}\hat y_{i,t}^{(m)},
\qquad
\sum_m w_{i,t,m}=1,
$$

where $w$ is conditioned on stock context/market regime. This is a proposed **scale gate**, not the original FMM. It must be compared with both *sum* and *average* baselines to establish incremental value.

## 87. Scale-depth study `[SOURCE, §4.2 Fig.6 p.9; Appendix D/Table 12 p.15]`

The authors find:

- For shorter forecast horizons, additional scale depth gives diminishing improvement.
- For longer horizons, greater scale depth can be more useful.
- Original defaults: $M=3$ long-term and $M=1$ short-term.

**Counter-intuitive warning:** a 5-day finance *target* is not enough to choose $M$; the available **input lookback** also matters, since $P_m$ shrinks geometrically.

## 88. Exact number-of-scale sensitivity on ETTm1 `[SOURCE, Table 12 p.15]`

| Scale-depth M | Predict 96 MSE | Predict 192 | Predict 336 | Predict 720 |
|---|---:|---:|---:|---:|
| 1 | 0.326 | 0.371 | 0.405 | 0.469 |
| 2 | 0.323 | 0.365 | 0.401 | 0.460 |
| 3 | 0.320 | 0.361 | 0.390 | **0.454** |
| 4 | 0.321 | **0.360** | **0.388** | **0.454** |
| 5 | 0.321 | 0.362 | 0.389 | 0.461 |

**Source detail:** for horizon 96, $M=3$ (0.320) is smaller than $M=4$ (0.321); no single number of scales dominates every horizon. Numerical values, not emphasis, should govern model selection.

## 89. PDM layer-count sensitivity `[SOURCE, Appendix D/Table 12 p.15]`

| PDM layers L | Predict 96 MSE | Predict 192 | Predict 336 | Predict 720 |
|---|---:|---:|---:|---:|
| 1 | 0.328 | 0.369 | 0.405 | 0.467 |
| 2 | 0.320 | 0.361 | 0.390 | 0.454 |
| 3 | 0.321 | **0.360** | 0.389 | **0.451** |
| 4 | **0.318** | 0.361 | **0.385** | 0.452 |
| 5 | 0.322 | **0.359** | 0.390 | 0.457 |

The paper selects $L=2$ as an efficiency/performance compromise, not as a proof that more than two layers never help.

## 90. Separating scale depth and network depth `[CRITIQUE]`

Two independent complexity knobs:

- $M$: how many temporal resolutions, $M+1$ streams.
- $L$: how many PDM passes across those streams.

A new experiment should tune them separately. More $M$ may shorten deepest streams too far; more $L$ can repeatedly mix already correlated information.

## 91. Efficiency study methodology `[SOURCE, Fig.5 p.9; Appendix B/Table 8 p.14]`

- Single GPU context.
- ETTh1 benchmark in the efficiency figure.
- Batch size 16.
- Time measured over 102 iterations (as described in Fig. 5).
- Tested input lengths: 192, 384, 768, 1536, 3072.
- Outputs: GPU memory consumption and seconds per iteration.

**Scope:** these are paper-specific implementations/hardware measurements, not guaranteed performance on every GPU or Qlib batch shape.

## 92. Complete TimeMixer vs PatchTST efficiency measurements `[SOURCE, Table 8 p.14]`

| Input length | TimeMixer memory MiB | TimeMixer s/iter | PatchTST memory MiB | PatchTST s/iter |
|---|---:|---:|---:|---:|
| 192 | 1,003 | 0.007 | 1,919 | 0.018 |
| 384 | 1,043 | 0.007 | 2,097 | 0.019 |
| 768 | 1,075 | 0.008 | 2,749 | 0.021 |
| 1,536 | 1,151 | 0.009 | 5,465 | 0.032 |
| 3,072 | 1,411 | 0.016 | 16,119 | 0.094 |

TimeMixer's reported efficiency advantage is substantial **especially for long input sequences**. It does **not** prove the gain will be equally large for $P=8$ or 20 stock windows, where overhead and cross-sectional operations may dominate.

## 93. Other models in the efficiency comparison `[SOURCE, Appendix B/Table 8]`

For input length 3072:

| Model | Memory MiB | s/iter |
|---|---:|---:|
| TimeMixer | **1,411** | 0.016 |
| PatchTST | 16,119 | 0.094 |
| TimesNet | 2,353 | 0.073 |
| Crossformer | 3,759 | 0.035 |
| MICN | 2,239 | 0.020 |
| DLinear | 1,239 | **0.015** |
| FEDformer | 12,485 | 0.288 |
| Autoformer | 10,043 | 0.255 |

DLinear is slightly faster and uses less memory at this input length. Therefore “TimeMixer is absolutely the fastest / smallest model” would be unsupported; the correct claim is a favorable efficiency-accuracy trade-off.

## 94. Source table transcription note `[SOURCE, Appendix B]`

DLinear's runtime entry for length 1536 in Table 8 is visibly mistyped (appears like `0.0.004`). A reasonable number should not be silently inserted into the knowledge base. This document avoids using that cell as if fully verified.

## 95. Paper's explicit computational limitation `[SOURCE, Appendix J p.22]`

The authors acknowledge that **temporal linear-mixing layers can grow in parameter count as input length increases**, which can harm deployment efficiency, especially on resource-constrained/mobile devices.

They propose investigating alternative mixing operators, including **attention-based or CNN-based** designs, as future work.

This qualifies the paper's efficiency claim:

- strong memory/run-time efficiency in evaluated ranges;
- potential weight-count scaling problem at increasing fixed input lengths.

## 96. Hyperparameter search protocol `[SOURCE, Appendix E pp.15–17]`

Beyond the unified-main-table protocol, the authors search over:

- input length: 96, 192, 336, 512;
- learning rate: from $10^{-5}$ to 0.05;
- encoder layers: 1 to 5;
- $d_{model}$: 16 to 512;
- training epochs: 10 to 100.

Table 14 reports outcomes of this broader search and is a different experimental setting from main Table 2.

## 97. A particularly important honesty point `[SOURCE, Appendix E p.15]`

The authors acknowledge that the **relative improvement over PatchTST is smaller** under broad hyperparameter searching than under the unified setting.

This is directly useful for research-Agent methodology: a newly designed architecture should be evaluated against **properly tuned** alternatives, not only default settings that may disadvantage them.

## 98. Tuned-protocol example: Weather `[SOURCE, Appendix Table 14 p.17]`

Selected aggregated MSE/MAE (average over forecast horizons):

| Model | Unified setting (Table 2) | Tuned setting (Table 14) |
|---|---|---|
| TimeMixer | 0.240 / 0.271 | **0.222 / 0.262** |
| PatchTST | 0.265 / 0.285 | 0.241 / 0.264 |

MSE improvement changes from approximately $9.4\%$ to approximately $7.9\%$ under the broader tuned setting.

## 99. Tuned-protocol examples: Traffic, Electricity `[SOURCE, Appendix Table 14 p.17]`

| Dataset | TimeMixer tuned MSE/MAE | PatchTST tuned MSE/MAE | Comment |
|---|---|---|---|
| Traffic | 0.387 / 0.262 | 0.391 / 0.264 | narrow gap |
| Electricity | 0.156 / 0.246 | 0.159 / 0.253 | narrow gap |
| Solar-Energy | 0.192 / 0.244 | 0.256 / 0.298 | larger gap remains |
| ETTh1 | 0.411 / 0.423 | 0.413 / 0.434 | narrow gap |
| ETTh2 | 0.316 / 0.384 | 0.324 / 0.381 | TimeMixer lower MSE; PatchTST slightly lower MAE |
| ETTm1 | 0.348 / 0.375 | 0.353 / 0.382 | narrow gap |
| ETTm2 | 0.256 / 0.315 | 0.256 / 0.317 | near tie in MSE |

**Critical correction to “SOTA on everything”:** even under its own tuned Table 14, TimeMixer is not strictly better than PatchTST for **every individual metric**, notably ETTh2 average MAE and ETTm2 MSE ties.

## 100. Practical research lesson from the tuned table `[CRITIQUE]`

A research Agent should use two kinds of comparison:

1. **Unified-protocol** comparison: controls data lengths/training conditions and isolates effects more directly.
2. **Reasonably tuned** comparison: tests whether improvement survives a competitive search budget.

Neither one alone is sufficient for a publication-grade claim.

## 101. Baseline fairness nuance `[SOURCE/CRITIQUE]`

The paper makes a strong effort to correct mismatched lookback/tuning across literature. Remaining caveats:

- broad hyperparameter search may have different computational cost for each architecture;
- appendix does not provide an independent equal-FLOPs analysis for every competitor;
- multiple hyperparameters and datasets can introduce benchmark-selection bias over development;
- the work uses general-domain forecasting MSE/MAE; downstream stock-ranking metrics would require their own protocol.

## 102. Three-run uncertainty `[SOURCE, Appendix C pp.14–15]`

Original experiments are repeated **three times**. Appendix Tables 9, 10, 11 supply mean ± standard deviation and confidence-level labels for important comparisons.

Selected long-term values:

| Dataset | TimeMixer MSE | PatchTST MSE | Confidence label |
|---|---:|---:|---|
| Weather | 0.240 ± 0.010 | 0.265 ± 0.012 | 99% |
| Solar-Energy | 0.216 ± 0.002 | 0.287 ± 0.020 | 99% |
| Electricity | 0.182 ± 0.017 | 0.216 ± 0.012 | 99% |
| Traffic | 0.484 ± 0.015 | 0.529 ± 0.008 | 99% |
| ETTh1 | 0.447 ± 0.002 | 0.516 ± 0.003 | 99% |
| ETTh2 | 0.364 ± 0.008 | 0.391 ± 0.005 | 99% |
| ETTm1 | 0.381 ± 0.003 | 0.400 ± 0.002 | 99% |
| ETTm2 | 0.275 ± 0.001 | 0.290 ± 0.002 | 99% |

The printed appendix's “confidence” column is not a substitute for detailed reporting of paired-run dependence, test statistic, or multiple-comparison adjustment.

## 103. PeMS three-run uncertainty `[SOURCE, Appendix C Table 10 p.15]`

| Dataset | TimeMixer MAE | SCINet MAE | Confidence label |
|---|---:|---:|---|
| PEMS03 | 14.63 ± 0.112 | 15.97 ± 0.153 | 99% |
| PEMS04 | 19.21 ± 0.217 | 20.35 ± 0.201 | 95% |
| PEMS07 | 20.57 ± 0.158 | 22.79 ± 0.179 | 99% |
| PEMS08 | 15.22 ± 0.311 | 17.38 ± 0.332 | 99% |

## 104. Additional concurrent baselines `[SOURCE, Appendix G pp.22–24]`

The authors also compare:

- Scaleformer: multiscale Transformer framework;
- MTSMixer: MLP-based multivariate mixing;
- TSMixer: all-MLP temporal forecasting.

They report unified and searched settings using available public implementations. This is relevant to **novelty**, because TimeMixer is not the first model using MLP layers, nor the first using multiple time scales.

## 105. Extra baseline selected numbers `[SOURCE, Appendix Table 23]`

| Dataset, unified | TimeMixer MSE/MAE | Scaleformer | MTSMixer | TSMixer |
|---|---|---|---|---|
| Weather (average) | **0.240/0.271** | 0.416/0.423 | 0.258/0.286 | 0.253/0.304 |
| Solar-Energy | **0.216/0.280** | 0.323/0.378 | 0.261/0.300 | 0.280/0.348 |
| Electricity | **0.182/0.272** | 0.203/0.315 | 0.201/0.293 | 0.220/0.327 |
| Traffic | **0.484/0.297** | 0.578/0.352 | 0.558/0.375 | 0.546/0.372 |

These are not financial transfer experiments; they are additional general forecasting baselines.

---

# Part VII — Figures, Interpretation and Evidence Boundaries

## 106. Figure 1 architecture `[SOURCE, PDF p.4]`

Three visual parts:

- (a) multiscale input via downsampling;
- (b) PDM with seasonal BU and trend TD information paths;
- (c) future forecast from multiple scales and summation.

This is the first figure an Agent should consult when checking the **overall graph** and direction of scale interactions.

## 107. Figure 2 temporal linear layer `[SOURCE, PDF p.5]`

- Blue seasonal mixing: fine-to-coarse sequence projection.
- Red trend mixing: coarse-to-fine sequence projection.
- Prediction projection maps each scale representation to the same forecast length.

**Practical purpose:** clarifies that `Bottom-Up-Mixing` and `Top-Down-Mixing` are MLPs acting on temporal sequence length rather than attention matrices or soft-routing weights.

## 108. Figure 3 weights and season/trend predictions `[SOURCE, PDF p.8]`

The authors visualize:

1. Seasonal-mixing temporal linear weights, showing repeated/periodic-like structures.
2. Trend-mixing temporal linear weights, showing a dominant local/diagonal structure.
3. Predictions from fine-scale seasonal and coarse-scale trend features.

They interpret these plots as supporting **distinct seasonal and trend cross-scale roles**. These are weight/prediction visualizations, not identified economic causes.

## 109. Figure 4 scale-wise predictions `[SOURCE, PDF p.9]`

ETTh1 input-96/predict-96:

- finest-scale predictor captures more local fluctuations/seasonal detail;
- coarse-scale predictors emphasize macro trend;
- mixed output follows combined future variation.

The figure provides qualitative support for FMM complementarity. The appendix explains visualization rescaling because individual scale contributions form a sum.

## 110. Figure 5 efficiency `[SOURCE, PDF p.9]`

Plots GPU memory and runtime versus length 192 to 3072. See Tables 8 / Sections 92–93 for exact reported numbers. The comparison is controlled within their hardware and implementation context.

## 111. Figure 6 depth sensitivity `[SOURCE, PDF p.9]`

Number of scales versus error for multiple forecast horizons. Supports choosing depth conditional on forecasting horizon and input length, not a single universal scale configuration.

## 112. Figure 7 direct predictor visualization `[SOURCE, Appendix A p.13]`

The appendix explains that each scale's forecast is **one summand** of the final prediction. It plots both raw scale contributions and rescaled contributions by multiplying by $(M+1)$ for visualization. This guards against misreading a low-amplitude scale output as an inherently weak predictor.

## 113. Figure 8 spectral comparison `[SOURCE, Appendix H p.22]`

Plots ground-truth and model prediction spectrograms for ETTh1 against PatchTST. Authors claim TimeMixer captures diverse frequency components. **Critique:** the figure is qualitative; it does not report frequency-specific explained variance or confidence intervals.

## 114. Figures 9–18 prediction examples `[SOURCE, pp.24–27]`

Representative forecasts from:

- ETTh1, Electricity, Traffic, Weather, Solar-Energy;
- PEMS03, PEMS04, PEMS07, PEMS08;
- M4.

These are useful for assessing fit patterns, scale, under/over-smoothing, and visually inspecting failures. However, a few illustrative sequences should not outweigh aggregate quantitative results.

## 115. How the figures support a scientific narrative `[INTERPRETATION]`

```text
Figure 1  → what is built
Figure 2  → how directional mixing is parametrized
Figure 3  → whether learned weights match intended component roles
Figure 4  → whether scale forecasts are complementary
Figure 5  → whether the new architecture is efficient
Figure 6  → whether multiscale depth behaves sensibly
Figures 8–18 → qualitative spectral and time-domain behavior
```

It is a strong example of **module–motivation–ablation–visualization alignment**.

---
# Part VIII — Inductive Bias, Failure Conditions and Scientific Critique

## 116. Core inductive bias inventory `[INTERPRETATION]`

| ID | Assumption / bias | Implementation evidence | Where it may fail |
|---|---|---|---|
| IB-01 | Important temporal structure exists at multiple resolutions | downsample pyramid | short/irregular series |
| IB-02 | Detailed residual/seasonal information can enrich coarse scale | bottom-up seasonal mixer | residual is mainly unpredictable noise |
| IB-03 | Coarse trends can stabilize fine-scale dynamics | top-down trend mixer | structural breaks or coarse trend bias |
| IB-04 | Trends and residuals require distinct mixing operators | separated PDM branches | trend/residual not separable |
| IB-05 | Averaging is a useful coarse-scale operator | avg pooling | spikes/rare events contain signal |
| IB-06 | Scale-specific forecasts are complementary | FMM sum | all heads learn duplicate predictions |
| IB-07 | Temporal linear layers can capture sufficient dynamics | fully MLP design | variable lengths, complex irregular timing |
| IB-08 | Forecast-value MSE represents task utility well | default training objective | ranking/decision objectives diverge |
| IB-09 | Same decomposition form is useful across datasets | repeated SeriesDecomp | scale- and data-specific filtering needed |

These are an independent formalization of the paper's choices, not a list of claims proven in all domains.

## 117. Why TimeMixer is NOT “just multiscale pooling” `[INTERPRETATION]`

Four separable operations are present:

1. **Construct hierarchy:** raw values → downsampled sequences.
2. **Decompose:** each scale → smooth trend + residual-like season.
3. **Mix:** directed communication across neighboring scales separately for the two components.
4. **Predict and fuse:** independent future heads from all scales.

A model that merely concatenates daily/weekly features implements only part (1), not the full architectural hypothesis.

## 118. Why TimeMixer is NOT “just decomposition” `[INTERPRETATION]`

MATCC and DLinear can split a series into smooth + residual components at one resolution. TimeMixer distinguishes itself by (a) **decomposition at every resolution** and (b) **cross-resolution communication with opposite directions**. Decomposition is a prerequisite to PDM, not its entire contribution.

## 119. Why TimeMixer is NOT Sparse MoE `[CRITIQUE]`

FMM's multiple predictors can resemble experts at a high level, but:

- all scale predictors are ordinarily evaluated;
- routing is not sparse;
- no top-k selection;
- no learned code-gated expert selection;
- no expert load-balancing loss;
- scale identity, rather than learned gate input, identifies each forecast head.

Calling the original model “MoE” risks conflating it with PRISM-VQ's sparse expert routing.

## 120. Why TimeMixer may denoise `[DERIVATION/INTERPRETATION]`

Average pooling acts roughly as a low-pass filter, reducing some high-frequency variance. If noise is short-lived and prediction target depends on persistent components, coarse representations may increase useful signal concentration.

**Counter-case:** if returns are driven by short-lived reversals, jump events, order-flow shocks or volatility spikes, smoothing may destroy precisely the signal needed. The original paper does not test the latter finance-specific cases.

## 121. Why component-specific direction may matter `[INTERPRETATION]`

The basic asymmetry:

- **Residual:** fine-scale changes are available only at high resolution. Passing details **up** can enrich coarser representations.
- **Trend:** slow trends are comparatively easy to see at coarse resolution. Passing macro information **down** can regularize high-resolution embeddings.

The ten-case ablation gives unusually good evidence for the *design as a whole*, but does not prove no other learnable orientation could do better.

## 122. Complementary forecasting vs. specialization `[CRITIQUE]`

Even if each scale produces a distinct-looking forecast, we still need to test whether contributions are genuinely non-redundant.

Missing quantitative diagnostics in the paper include:

- pairwise correlation of scale-head prediction errors;
- head-specific marginal loss reduction;
- dependence of head contributions on regime/horizon;
- entropy or concentration of scale contribution under a learned weighting alternative;
- stability of each head across random seeds.

These diagnostics would strengthen an advanced Agent's adaptation.

## 123. Identifiability of “seasonal” components `[CRITIQUE]`

Under a moving-average residual split:

$$
\mathbf s=\mathbf x-\operatorname{MA}(\mathbf x),
$$

the residual can include irregular noise, sharp jumps, changes in volatility, unmodeled trend breaks and true repeating seasonal behavior.

Therefore financial notes should preferentially say **“high-frequency/residual-like component”** unless periodicity is independently demonstrated. This prevents false economic interpretation of a learned mathematical branch.

## 124. Fixed-kernel / scale assumption `[CRITIQUE]`

PDM inherits a fixed hierarchy of 2× downsampling. Real financial patterns do not necessarily align to powers of two, and Alpha158 variables encode heterogeneous windows already.

Example:

- a 5-day reversal;
- 20-day price momentum;
- 60-day volatility;

are not guaranteed to be optimally represented by pooling a $T=20$ sequence to lengths $20,10,5,2$.

## 125. Boundary effects under short history `[CRITIQUE]`

At $P=8$:

$$
[P_0,P_1,P_2,P_3]=[8,4,2,1].
$$

At the coarsest level, a trend or seasonal decomposition has little meaningful temporal variation. If a moving-average kernel is larger than the current scale's length, padding dominates.

A stock adaptation should cap the scale depth and/or redesign the hierarchy, and verify the actual number of distinct observations after pooling.

## 126. Irregular sampling, holidays and cross-market calendars `[CRITIQUE]`

The paper assumes regularly indexed time-series values. Daily equity bars have irregular calendar gaps caused by weekends/holidays; Chinese and U.S. markets have different trading calendars.

Possible effects:

- two adjacent trading observations may be 1 or 4 calendar days apart;
- “weekly trend” measured in trading steps may not represent fixed elapsed time;
- cross-market factor windows can become misaligned.

A finance adaptation needs clearly stated **trading-day vs calendar-day** semantics.

## 127. Nonstationarity and regime changes `[CRITIQUE]`

The method is motivated by temporal variability but uses learned linear mixers with time-length-specific parameters. The original paper does not demonstrate online updating of temporal mixers under financial regime changes or explicit drift adaptation.

Thus terms such as **regime-aware**, **adaptive market gating**, **online prototype evolution**, or **market-state routing** would be proposed additions, not TimeMixer original results.

## 128. Theoretical optimality was not proved `[SOURCE, Appendix J p.22]`

The authors explicitly list theoretical analysis of the **optimality/completeness** of their design as future work. The ablation performance is empirical support; it is not a proof that bottom-up seasonal/top-down trend is uniquely optimal.

## 129. Variable-dimension mixing remains incomplete `[SOURCE, Appendix J p.22]`

The authors state that their emphasis is **temporal dimension mixing** and explicitly propose incorporating **variate-dimension mixing** as future work.

The model may contain channel projections or feedforward feature interactions, but its principal mixing novelty is not explicit learned cross-variable graph or channel routing. A new model adding financial cross-stock interaction should identify the chosen interaction axis precisely.

## 130. Source's explicit limitations — complete list `[SOURCE, Appendix J p.22]`

1. Temporal linear mixing can grow parameter count as input length grows; this is a challenge for constrained devices.
2. Alternative mixing mechanisms, including attention-based or convolution-based, are suggested.
3. Additional **variate-dimension mixing** is an explicit future direction.
4. Theoretical analysis of why design is optimal/complete remains open.

Do not replace this with generic “future work: stock markets / MoE” language. None of those finance directions is explicitly promised by the authors.

## 131. Additional independent limitations `[CRITIQUE]`

- Mixed frequency components are not necessarily economically identifiable.
- Moving average may remove event-driven signal.
- Fixed hierarchy may not fit short history.
- Sum ensemble may be redundant or scale-miscalibrated.
- Long-horizon benchmark MSE is not a financial ranking metric.
- No study of transaction costs or turnover.
- No CSI300, S&P500, Qlib integration in original paper.
- No factor exposure or risk decomposition model.
- No probabilistic predictive intervals.
- Parameter count can rise with temporal length; arbitrary-length transfer is not automatic.
- No explicit financial distribution-shift or regime robustness study.

## 132. “Negative results” worth retaining `[SOURCE/CRITIQUE]`

The appendix is especially informative because it reports design alternatives that did not become the official implementation:

| Alternative | Observation | Interpretation |
|---|---|---|
| DFT high/low decomposition | worse than MA design on selected tests | decomposition semantics and mixer direction must match |
| DFT seasonal/trend decomposition | better on selected tests | MA chosen for simplicity, not absolute best accuracy |
| strided 1D convolution downsampler | modestly better | pooling is cheap/easy, not strictly optimum |
| average vs sum FMM | nearly tied | explicit scaling is not the essence |
| both mixing branches same direction | weaker | direction-specific bias matters |
| reversed mixing directions | substantially weaker | branch alignment matters |
| more PDM layers | not monotonic | increased complexity can stop helping |
| more scales | diminishing returns for short horizons | scale capacity has task dependence |

A research Agent should mine these negative/comparison findings before proposing what it thinks is a “novel” modification.

## 133. Evidence strength assessment `[CRITIQUE]`

**Stronger evidence:**

- 18 benchmark groupings;
- three-run error bars;
- alternative controlled baseline protocols;
- ten-way ablation with direction reversals;
- multiple decomposition and pooling alternatives;
- efficiency measurements over five sequence lengths;
- direct multi-head visualization and spectral cases.

**Weaker evidence or unresolved aspects:**

- theoretical optimality unspecified;
- benefits in forecasting with very short, noisy financial lookbacks untested;
- no direct economic use case;
- large-scale parameter count and variable-size adaptation unresolved;
- no causal interpretation of seasonal/trend channels;
- ensemble weights not explicitly optimized against regime-dependent utility;
- some main-table SOTA advantages shrink under more tuning.

## 134. Mechanism evidence matrix `[CRITIQUE]`

| Mechanism | Original evidence | Confidence for original datasets | Transfer confidence to equities |
|---|---|---|---|
| Multiscale temporal views | horizon and depth sensitivity | supported | unknown until tested |
| Seasonal BU mixer | specific ablations | supported | hypothesis only |
| Trend TD mixer | ablations, weight maps | supported | hypothesis only |
| Separation of component directions | reversing orientation hurts | supported | hypothesis only |
| FMM sum of forecasts | w/o FMM ablation | supported | hypothesis only |
| MA decomposer itself optimal | DFT-seasonal/trend beats MA in selected tests | **not supported** | untested |
| Average pooling optimal | strided conv slightly wins | **not supported** | untested |
| Low GPU / runtime cost at long P | efficiency table | supported in tested setup | depends on stock use case |
| Robust market-regime prediction | no market experiment | absent | unsupported |

---

# Part IX — Mechanism Inventory for Machine Innovation

## 135. Compact mechanism card format

Each primitive below has five fields:

1. **Primitive** — atomic reusable operation.
2. **Mathematical interface** — input and output semantics.
3. **Inductive bias** — why authors use it.
4. **Dependencies** — what must be in place.
5. **Candidate transfer** — separately labeled hypothesis for finance.

Do not copy the whole TimeMixer architecture by default. Novelty usually lies in making a primitive **solve a specific unsolved task limitation**, and validating the new mechanism.

## 136. Primitive TM-01 — scale pyramid `[SOURCE/INTERPRETATION]`

- Operation: average downsampling at powers of two.
- Interface: $[P,C]\to\{[P_m,C]\}_{m=0}^M$.
- Bias: coarse views reveal macroscopic regularity.
- Dependency: sufficiently long and sensibly sampled input.
- Finance transfer: per-stock multi-resolution sequences, with small $M$ for short $T$.

## 137. Primitive TM-02 — scale-specific embedding `[SOURCE/INTERPRETATION]`

- Operation: embed each resolution into $d_{model}$-dimensional features.
- Interface: $[P_m,C]\to[P_m,d]$.
- Bias: learned feature space enables nonlinear temporal interactions.
- Finance transfer: separate projection for different Alpha158 feature groups or time scales.

## 138. Primitive TM-03 — moving-average trend extractor `[SOURCE]`

- Operation: smooth each scale with an MA operator.
- Interface: $x_m\to t_m$.
- Bias: low-frequency structure can be separated from local variation.
- Dependency: suitable averaging width and padding.
- Finance transfer: trend-specific representations with carefully selected trading-day windows.

## 139. Primitive TM-04 — residual/season extractor `[SOURCE]`

- Operation: subtract trend from original.
- Interface: $s_m=x_m-t_m$.
- Bias: avoid suppressing fluctuations completely.
- Limitation: seasonal name need not imply periodicity.
- Finance transfer: isolate price/volume deviations from smoothed component.

## 140. Primitive TM-05 — bottom-up MLP `[SOURCE]`

- Operation: learnable temporal-length projection from $P_{m-1}$ to $P_m$.
- Interface: residual seasonal update across adjacent scales.
- Bias: detailed fluctuations inform macro-scale seasonal representation.
- Dependency: at least two scale levels.
- Finance transfer: aggregate higher-frequency shock features into lower-frequency context.

## 141. Primitive TM-06 — top-down MLP `[SOURCE]`

- Operation: learnable temporal-length projection from $P_{m+1}$ to $P_m$.
- Interface: residual trend update across adjacent scales.
- Bias: macro trend guides local trend modeling.
- Dependency: sufficiently meaningful coarse scale.
- Finance transfer: contextualize short-term stock signals with market or longer-horizon trend.

## 142. Primitive TM-07 — branch-specific direction `[SOURCE]`

- Operation: different graph topology by component type.
- Evidence: strong direction-reversal ablation.
- Core transferable idea: **information-flow orientation can encode domain assumptions**.
- Finance transfer: directional flow may instead be selected by market regime or feature group.

## 143. Primitive TM-08 — per-scale residual FFN `[SOURCE]`

- Operation: channel mixing after sum of season/trend cross-scale representations, then residual addition.
- Bias: preserve original scale state while learning nonlinear corrections.
- Finance transfer: stable lightweight residual adapter for existing temporal encoders.

## 144. Primitive TM-09 — stacked mixing `[SOURCE]`

- Operation: repeat $L$ PDM blocks.
- Bias: progressively refine multi-resolution representations.
- Limitation: performance does not increase monotonically with $L$.
- Finance transfer: test one vs two PDM layers before extending depth.

## 145. Primitive TM-10 — per-scale future predictor `[SOURCE]`

- Operation: learn a linear mapping from each length $P_m$ to forecast length $F$, then project to channels.
- Bias: resolution-specific structure has predictive utility.
- Finance transfer: one scalar ranking head per scale, with optional regime-conditioned fusion.

## 146. Primitive TM-11 — sum ensemble `[SOURCE]`

- Operation: $\hat x=\sum_m\hat x_m$.
- Bias: complementary heads can jointly explain target.
- Limitation: no explicit learned reliability.
- Finance transfer: sum scale-dependent score contributions under a common target protocol.

## 147. Primitive TM-12 — decomposition replacement socket `[SOURCE/EXTENSION]`

- Operation: replace moving-average with DFT seasonal/trend, learned filter or multi-kernel trend extractor.
- Evidence: alternatives in Appendix F.2.
- Novelty caveat: *simply switching to DFT* was already tested by TimeMixer; it is not automatically a new contribution.

## 148. Primitive TM-13 — downsampling replacement socket `[SOURCE/EXTENSION]`

- Operation: replace pooling by strided convolution or learnable resampler.
- Evidence: Appendix F.3.
- Novelty caveat: authors already tried strided conv; a finance contribution needs new mechanism or rigorous task-specific explanation.

## 149. Primitive TM-14 — scale-depth controller `[SOURCE/EXTENSION]`

- Operation: set or learn effective $M$ based on available history and target.
- Evidence: Figure 6/Table 12.
- Finance transfer: cap depth or gate levels for 8/20-day histories.
- Research requirement: dynamic scale selection must be compared against tuned fixed depth.

## 150. Primitive TM-15 — source transparency `[SOURCE/INTERPRETATION]`

- Operation: ablate **mechanisms and their directional assumptions**, not just remove layers.
- Evidence: 10-way Table 5, Tables 15–17.
- Transfer: explicit falsification of claimed stock-specific multiscale rationale.

## 151. Mechanism dependency graph `[INTERPRETATION]`

```text
Multi-resolution representation
      ├── each scale meaningful enough? ── NO → reduce M
      │
      ▼
Trend / residual decomposition
      ├── chosen operator MA vs DFT / learned filter?
      │
      ▼
Direction-specific exchange
      ├── seasonal BU
      └── trend TD
      │
      ▼
All-resolution representations
      │
      ▼
Independent scale prediction heads
      │
      ▼
Common future target
      │
      ▼
FMM sum
```

## 152. Comparison with PatchTST `[SOURCE/INTERPRETATION]`

| Dimension | PatchTST (ICLR 2023) | TimeMixer (ICLR 2024) |
|---|---|---|
| Main abstraction | local patch tokens | multiple temporal sampling scales |
| Core backbone | Transformer encoder | all-MLP temporal mixing |
| Input resolution | patch size/stride | downsampling pyramid |
| Key temporal bias | patch-level local semantics | seasonal BU + trend TD mixing |
| Channel handling | explicitly channel-independent shared backbone | emphasizes temporal mixing; cross-variate modeling is further work |
| Output | horizon head per channel | per-scale heads summed |
| Pretraining | masked-patch SSL studied | no corresponding masked SSL core |
| Evaluation | mostly long-horizon with SSL transfer | long + PeMS + M4 short horizons |
| Finance relevance | patch granularity and shared weights | directional multiscale decomposition/fusion |

**Not a contradiction:** both can be useful as building blocks. The comparative advantage depends on lookback, feature count, objective and cross-variable structure.

## 153. Comparison with MATCC `[INTERPRETATION]`

| Dimension | MATCC (CIKM 2024) | TimeMixer |
|---|---|---|
| Task | cross-sectional stock return score | general time-series future values |
| Primary decomposition | market-guided stock trend/fluctuation | every-scale seasonal/trend |
| Scale count | one main decomposition scale | hierarchy $0..M$ |
| Market context | explicit index features | no dedicated index feature stream |
| Temporal backbone | RWKV | temporal MLP mixing |
| Cross-stock mixing | multi-head attention | no dedicated stock-attention block |
| Main ordering | time correlation → stock correlation | season BU/trend TD across scales |
| Output objective | stock target MSE | sequence forecast MSE/SMAPE |

Natural orthogonal combination: retain MATCC's **cross-stock relation stage** while testing TimeMixer-style **multiscale temporal encoding**. However, a larger stack does not automatically improve performance and must control parameter/compute budgets.

## 154. Comparison with MASTER `[INTERPRETATION]`

MASTER factorizes stock-time relation modeling as temporal attention then cross-stock attention. TimeMixer factorizes temporal resolution components into two directions then forecasts from each.

These address different interaction axes:

- MASTER: *stock × time* dependency structure.
- TimeMixer: *time resolution × trend/residual* dependency structure.

Possible complementarity exists, but novelty depends on whether a newly designed scale-conditioned cross-stock interaction produces **more than two independent modules concatenated**.

## 155. Comparison with FactorVAE / FactorVQVAE `[INTERPRETATION]`

| | FactorVAE | FactorVQVAE | TimeMixer |
|---|---|---|---|
| Core role | continuous probabilistic factor | discrete token factor | deterministic multiscale feature/forecast processing |
| Training-time future input | posterior branch | Stage-1 future factor encoding | standard supervised target only |
| Factor-model decoder | yes | yes | no |
| VQ codebook | no | yes | no |
| Multi-resolution temporal mixing | not original core | not original core | yes |
| Latent uncertainty | explicit | discrete tokens | no primary probabilistic output |

A research Agent can borrow the **temporal encoder idea** without misrepresenting TimeMixer as a new kind of VAE.

## 156. Comparison with PRISM-VQ `[INTERPRETATION]`

PRISM-VQ's learned discrete code is a **cross-sectional structure condition** and sparse-MoE routing signal. TimeMixer's scales are fixed deterministic **temporal resolutions**.

Potential complementarity:

$$
\text{stock history}
\rightarrow
\text{multiscale temporal features}
\xrightarrow{\text{VQ structural condition}}
\text{factor-loading predictor}.
$$

However, VQ and MoE already create conditional capacity; adding scale-specific heads can overparameterize low-signal stock data. Complexity controls and minimal ablations are essential.

## 157. Research lineage `[SOURCE/INTERPRETATION]`

```text
Moving-average decomposition (Autoformer and later models)
               │
               ├── DLinear: decompose then linear forecast
               │
               └── TimeMixer: decompose at EACH scale
                                ↓
                         seasonal fine→coarse
                         trend coarse→fine
                                ↓
                        scale-wise future heads

PatchTST: independent parallel lineage
  patch tokenization + channel-independent Transformer

MATCC: finance-adapted trend/fluctuation + market info + RWKV

PRISM-VQ: finance-adapted cross-sectional discrete codes + MoE
```

Do not infer direct authorship or inheritance among finance papers merely because they share an operation.

---
# Part X — Finance-Specific Transfer: Task Semantics and Compatibility

> **Every section in Parts X–XII is `[EXTENSION]` unless additionally tagged.** None of the experiments below appears in TimeMixer's ICLR 2024 paper. Their purpose is to give a research Agent actionable, falsifiable hypotheses, **not** ready-made claims of novelty or gains.

## 158. Destination task: cross-sectional ranking is not long-horizon value prediction `[EXTENSION]`

Consider daily stock input:

$$
X_t\in\mathbb R^{N_t\times T\times C},
$$

where:

- $N_t$: number of stocks on trading date $t$;
- $T$: historical trading-day lookback;
- $C$: stock feature count (e.g., 158 technical indicators under Alpha158);
- target $y_t\in\mathbb R^{N_t}$: forward return or rank-normalized target per stock.

Original TimeMixer:

$$
[P,C]\longrightarrow[F,C].
$$

Financial adaptation:

$$
[N_t,T,C]\longrightarrow[N_t],
$$

with **cross-sectional** objective and evaluation rather than only future-value MSE.

## 159. The most conservative adaptation `[EXTENSION]`

Replace only a per-stock temporal encoder with a TimeMixer-style block. Then keep the original downstream cross-stock module and target protocol unchanged.

```text
Stock history [N,T,C]
          │
          ▼
 Per-stock TimeMixer encoder
          │
          ▼
 Stock embedding [N,D]
          │
          ▼
 Existing cross-stock / factor / predictor head
          │
          ▼
 Return score [N]
```

This preserves experiment interpretability: first test whether multiscale time processing adds value before redesigning whole model.

## 160. Direct score-head adaptation `[EXTENSION]`

One can adapt FMM from a sequence forecast to a scalar stock score:

$$
\hat y_{i,t}^{(m)}
=h_m(H_{i,t}^{(m)})\in\mathbb R,
$$

$$
\hat y_{i,t}=\sum_{m=0}^{M}\hat y_{i,t}^{(m)}.
$$

Now each head must predict **the same forward-return target** (or contribute to it), not scale-dependent raw price levels. The loss must be applied to the final combined score, mirroring original FMM.

## 161. Do not silently confuse “time scale” with “forecast horizon” `[EXTENSION]`

Two independent axes:

- **Historical scale:** how past observations are downsampled or grouped.
- **Label horizon:** how future return is measured.

A 20-day historical lookback can feed a 1-day, 5-day or 10-day target. Changing horizon without declaring it breaks protocol comparability.

## 162. Lookback scale selection for $T=8$ `[EXTENSION]`

| Proposed depth M | Levels | Sequence lengths | Expected issue |
|---|---:|---|---|
| 0 | 1 | 8 | no cross-scale operation |
| 1 | 2 | 8, 4 | modest coarse context |
| 2 | 3 | 8, 4, 2 | deepest stream extremely short |
| 3 | 4 | 8, 4, 2, 1 | deepest stream degenerates |

I would start an engineering smoke test with $M=1$, compare against $M=0$ and $M=2$, and not presume $M=3$ remains sensible.

## 163. Lookback scale selection for $T=20$ `[EXTENSION]`

| Proposed depth M | Levels | Sequence lengths | Question |
|---|---:|---|---|
| 0 | 1 | 20 | temporal encoder only |
| 1 | 2 | 20, 10 | is one coarse view useful? |
| 2 | 3 | 20, 10, 5 | is medium-scale structure incremental? |
| 3 | 4 | 20, 10, 5, 2 | does 2-point coarse stream help or hurt? |

An M=2 configuration may be a more reasonable starting hypothesis than the paper's fixed M=3 for general long-range forecasting, but this is **not verified** for stocks.

## 164. Multiple native-horizon features change what pooling means `[EXTENSION]`

Alpha158 may contain features computed over different internal windows (e.g., historical means, momentum, volatility). Downsampling the **time axis of those already aggregated factors** is not equivalent to constructing raw 5/10/20-day price patches or computing factor values at new frequencies.

A research Agent should distinguish:

1. sequence downsampling of technical features;
2. recalculating factors on resampled price-volume data;
3. multi-horizon feature grouping without downsampling;
4. temporal downsampling of learned embeddings.

These are four different mechanisms and should not be conflated.

## 165. Architecture choice A — temporal encoder substitution `[EXTENSION]`

Use TimeMixer to replace GRU / Transformer / RWKV for stock-local temporal feature extraction while preserving the rest of a baseline.

- **Pros:** minimal intervention; easy ablation.
- **Cons:** gains may be small for a very short $T$; feature dimension can be large.
- **Original paper support:** only for general sequence forecasting, not cross-sectional stocks.

## 166. Architecture choice B — TimeMixer + MASTER `[EXTENSION]`

```text
Stocks X [N,T,C]
   │
   ├── Market gate (MASTER unchanged)
   ▼
TimeMixer local multiscale temporal encoder
   │
   ▼
Stock-time relay representations
   │
   ▼
MASTER inter-stock attention
   │
   ▼
Cross-sectional ranking head
```

**Important design tension:** MASTER depends on preserving a suitable **time-indexed embedding** for inter-stock interaction. A naive TimeMixer implementation that collapses to a single $[N,D]$ vector can destroy MASTER's original time-relay mechanism. A fair adaptation should retain the required time axis or explicitly declare a changed relation architecture.

## 167. Architecture choice C — TimeMixer + MATCC `[EXTENSION]`

MATCC already contains:

- market trend guidance;
- trend/fluctuation decomposition;
- RWKV temporal correlation;
- stock self-attention;
- temporal aggregation.

Potential experiment:

- keep MATCC market and stock modules;
- replace only the **single-scale trend decomposition + RWKV** path with a multiscale PDM path;
- check that the experiment is not simply raising parameter count or adding an extra predictor ensemble.

Ablate independently: multiscale decomposition, directional mixing, and backbone replacement.

## 168. Architecture choice D — TimeMixer + PRISM-VQ `[EXTENSION]`

Possible use: TimeMixer provides $h_{\mathrm{temp},i}$ as the time-dependent representation for the Stage-2 factor-loading generator. Keep:

- Stage 1 cross-sectional VQ code;
- finance priors;
- fixed Stage 1 protocol;
- MoE router and expert inventory;
- factor pricing equation;
- datasets and label.

Then test whether TimeMixer's multiscale representation improves loadings or destabilizes VQ-conditioned routing. Do not modify both Stage 1 and Stage 2 at once in the first experiment.

## 169. Architecture choice E — scale-aware cross-stock interaction `[EXTENSION]`

Instead of one cross-stock attention operation after all scales have been fused, each scale can define a distinct relation matrix:

$$
A_t^{(m)}
=\operatorname{StockRelation}
(\{h_{i,t}^{(m)}\}_{i\in\mathcal U_t}).
$$

A fused score could depend on:

$$
\hat y_{i,t}
=\sum_m h_m(A_t^{(m)}H_t^{(m)}).
$$

Hypothesis: peer relations differ at short and medium horizons. This is considerably more expensive and may overfit; a minimal single-scale relation variant should be the primary comparator.

## 170. “Seasonal” component in daily stocks `[EXTENSION]`

Do not assume the original semantic explanation transfers verbatim:

- daily stock returns may have calendar seasonality, but many high-frequency residuals are noise;
- volume and volatility features can have distinct autocorrelation;
- multi-scale trends may be more meaningful for **prices/volume** than for rank-normalized future returns;
- economic significance must be assessed through signed stock ranking and portfolio construction, not only reconstruction smoothness.

## 171. Potential interactions with market regime `[EXTENSION]`

A market-state vector $r_t$ could control **scale importance**, **directional information strength**, or **trend-smoother width**:

$$
\alpha_{t,m}
=\operatorname{softmax}_m
(g(r_t)),
$$

$$
\hat y_{i,t}
=\sum_m\alpha_{t,m}\hat y_{i,t}^{(m)}.
$$

This is not original FMM. A publishable claim requires demonstrating why conditional weights help in a heterogenous market environment and why simpler static weights fail.

## 172. Cross-sectional data availability checklist `[EXTENSION]`

Before evaluating any variant, enforce:

- [ ] Each input timestamp is no later than decision time $t$.
- [ ] Features have correct lookback and publication-time availability.
- [ ] Nontrading days handled consistently.
- [ ] Train/validation/test boundaries prevent look-ahead from forward-return labels.
- [ ] Universe membership (CSI300, S&P500) follows the chosen point-in-time rule.
- [ ] Any scaler is fitted only on allowable training data.
- [ ] Future return horizon and entry/exit price conventions match baselines.
- [ ] Market-regime features only use observable information at decision time.
- [ ] No future cross-section is used in pooling or target-dependent conditioning.
- [ ] Model selection is based on validation only; test is held out for final comparison.

## 173. What a finance evaluation must add `[EXTENSION]`

The original MSE/MAE metrics should be supplemented or replaced with:

- daily Pearson IC;
- daily Spearman RankIC;
- ICIR and RankICIR;
- rolling/monthly/quarterly stability;
- expected long-only or long-short ranking value;
- cost-aware annualized return;
- Sharpe / maximum drawdown and other risk metrics;
- portfolio turnover;
- seed dispersion;
- worst-period performance and regime breakdown.

Simply improving general sequence MSE is **not** proof of financial usefulness.

## 174. Cost / complexity control `[EXTENSION]`

Report at minimum:

| Measure | Why it matters |
|---|---|
| Trainable parameters | larger mixer matrices may explain gains |
| Peak GPU memory | cross-section size and feature dimension vary |
| Training wall time | research throughput for many candidates |
| Inference time per date | practical portfolio generation |
| Total seeds | statistical reliability |
| Number of scale levels M | architecture complexity |
| Number of PDM blocks L | depth confounding |
| Cross-stock attention cost | potential quadratic growth with universe |
| Portfolio turnover | net tradability |

---

# Part XI — Explicit, Falsifiable Experiment Hooks for the Agent

## 175. Experiment design rule `[EXTENSION]`

Every proposed idea below must specify:

1. **one hypothesis**;
2. **one minimally sufficient intervention**;
3. **a matched baseline**;
4. **metrics for success**;
5. **a result that would falsify the hypothesis**.

Do not present a clever-looking diagram as an experiment. Keep a per-seed and per-market record of outcomes.

## 176. E01 — Does TimeMixer help a short-window stock encoder? `[EXTENSION]`

**Hypothesis:** a small multiscale temporal encoder extracts more stable predictive structure than a single-scale MLP or GRU on the same input.

**Minimal test:** keep input features/label/head fixed; compare GRU, simple MLP, TimeMixer-$M=1$, and TimeMixer-$M=2$ as independent temporal modules.

**Success:** reproducible improvement in CSI300 and/or S&P500 RankIC and regime stability without sharply increasing runtime.

**Failure / falsification:** no stable gain across seeds, or gains disappear in a compute-matched comparison.

## 177. E02 — Is bottom-up seasonal mixing useful in stock data? `[EXTENSION]`

**Hypothesis:** detailed short-horizon residual information helps the coarser stock state.

**Compare:** official BU seasonal path vs removed path vs TD seasonal path, while holding trends/heads fixed.

**Measure:** RankIC, ICIR, worst-month RankIC, runtime, and signal concentration by liquidity/volatility bins.

**Failure:** BU consistently provides no incremental performance or increases instability.

## 178. E03 — Is top-down trend guidance useful? `[EXTENSION]`

**Hypothesis:** coarse trend can regularize fine-scale stock prediction in changing markets.

**Compare:** TD trend vs no TD vs BU trend under the same scale hierarchy.

**Measure:** rolling RankIC, 2024–2025-like market shifts, worst-quarter performance, trend scale contributions.

**Failure:** coarse trend consistently degrades RankIC in trend reversals or is dominated by single-scale smoothing.

## 179. E04 — Do opposite directions matter beyond two generic cross-scale MLPs? `[EXTENSION]`

**Hypothesis:** direction-specific inductive bias is beneficial even when total parameter count is matched.

**Compare:** official BU/TD, both BU, both TD, reversed TD/BU, learned bidirectional mixing with equal parameter budget.

**Success:** official or a new conditional-direction design beats all same-capacity alternatives with significance.

**Failure:** a symmetric or no-decomposition mixer matches or exceeds official design.

## 180. E05 — Which decomposition makes sense for stocks? `[EXTENSION]`

**Hypothesis:** price/volume-derived inputs may benefit from a different smoother than general forecast benchmarks.

**Compare:** MA decomposition, DFT seasonal/trend, strided convolution, identity/no decomposition under same mixer.

**Success:** consistent out-of-sample lift linked to reduced residual noise or better regime stability.

**Failure:** changes only improve training loss or a single market seed.

**Novelty caveat:** DFT seasonal/trend and conv alternatives are **already in TimeMixer Appendix F**. Merely using them in finance is not necessarily enough novelty for a strong paper.

## 181. E06 — Is fixed scale count too rigid? `[EXTENSION]`

**Hypothesis:** market regimes need different effective temporal resolution.

**Compare:** static $M=1/2/3$ vs regime-aware scale weight/gate, with the same pool of feature levels.

**Measure:** active-scale entropy, RankIC by month, bear/bull/sideways segment, turnover and parameter cost.

**Failure:** dynamic gate collapses to constant weights or does not beat the best tuned static model.

## 182. E07 — Is scale-gated FMM better than sum? `[EXTENSION]`

**Hypothesis:** some scale-specific forecasts are unreliable in certain volatility regimes.

**Compare:** raw sum, average, fixed learned weights, per-date market gate, per-stock market+feature gate.

**Measure:** RankIC, RankICIR, calibration of scale score contributions, out-of-sample Sharpe.

**Failure:** dynamic weights do not beat constant weights on held-out data or require excessive tuning.

## 183. E08 — Are scale experts truly complementary? `[EXTENSION]`

**Hypothesis:** shorter and longer temporal scales make nonidentical ranking errors.

**Inspect:** per-scale score correlation, per-scale residual after regressing on other heads, leave-one-scale-out rank ablations.

**Measure:** marginal RankIC contribution, pairwise prediction correlation, regime condition dependence.

**Failure:** all heads generate essentially the same ranking and each extra head's contribution is insignificant.

## 184. E09 — Multi-scale representation with PRISM-VQ VQ codes `[EXTENSION]`

**Hypothesis:** fine/coarse temporal views help predict dynamic factor loadings given a fixed cross-sectional structural code.

**Intervention:** change Stage-2 temporal encoder only; keep VQ codebook, prior factors and MoE fixed.

**Measure:** RankIC, market-wide routing distribution, factor-loading stability, seed variance.

**Failure:** gains vanish after matching encoder parameter count or degrade latent code usefulness.

## 185. E10 — Multi-scale temporal state in MATCC `[EXTENSION]`

**Hypothesis:** TimeMixer's cross-scale interactions enrich MATCC's original trend+RWKV sequence.

**Compare:** original MATCC, only decomposition replaced, only RWKV replaced, both replaced, with standard ablations.

**Measure:** RankIC, IC, runtime, temporal interaction stability.

**Failure:** no gain beyond a simple stronger single-scale temporal encoder, or cost-aware returns deteriorate.

## 186. E11 — Scale-conditioned inter-stock attention `[EXTENSION]`

**Hypothesis:** stock dependencies are different at short and medium historical scales.

**Intervention:** compute scale-specific (possibly sparse) stock interactions, then fuse.

**Baseline:** same temporal features with only one relation matrix and same parameter budget.

**Measure:** RankIC, relation stability, regime robustness, inference latency.

**Failure:** cross-scale relation matrices are redundant or gains come entirely from model size.

## 187. E12 — Adaptive trend-residual decomposition by market state `[EXTENSION]`

**Hypothesis:** smoothing width appropriate in low volatility differs from width in shock regimes.

**Compare:** fixed MA window vs gated multi-window MA vs context-adaptive learned filter.

**Measure:** RankIC across volatility quantiles and transient stress events; ensure no future regime labels enter routing.

**Failure:** adaptive version adds no stable gain or worsens regime transitions due to noise amplification.

## 188. E13 — Multi-scale ranking-aware objective `[EXTENSION]`

**Hypothesis:** MSE-based scale heads are suboptimal for a cross-sectional ranking task.

**Compare:** MSE, pairwise rank loss, hybrid MSE+rank surrogate, and a rank-residual head variant.

**Measure:** RankIC, RankICIR, cost-aware TopK Sharpe, predicted-return magnitude stability.

**Failure:** rank objective improves RankIC in-sample only or damages Sharpe/turnover out-of-sample.

## 189. E14 — Do more history and better multiscale encoding substitute for each other? `[EXTENSION]`

**Hypothesis:** additional 40/60 trading days could be more valuable than adding deeper PDM blocks to a 20-day sequence.

**Compare:** $(T,M)\in\{(20,1),(20,2),(40,1),(40,2),(60,1),(60,2)\}$ with matched compute when feasible.

**Measure:** RankIC, performance by return horizon, validation stability, runtime.

**Failure:** longer windows only increase leakage risk/overfitting or do not beat simple 20-day baselines.

## 190. E15 — Do multiscale forecasts react differently to factor regimes? `[EXTENSION]`

**Hypothesis:** short-horizon component dominates mean-reversion environments while coarse component dominates persistent-trend environments.

**Measure:** post-hoc scale contribution conditional on market trend, volatility, and known factor exposures; validate with ablations.

**Failure:** associations disappear after controlling for lookback target autocorrelation or do not survive new periods.

**Economic caution:** post-hoc attribution does not establish causality of learned heads.

## 191. E16 — Adaptive sparse scale routing instead of dense FMM `[EXTENSION]`

**Hypothesis:** evaluating only a few scales per stock/date can improve compute efficiency without hurting rank quality.

**Method:** inexpensive gate selects at most $k$ active scale predictors; measure routing stability.

**Baselines:** all-scale sum; all-scale soft gate; random scale selection; static best-$k$ scales.

**Failure:** sparse model reduces runtime trivially because the original all-MLP FMM is already cheap, while lowering RankIC.

## 192. E17 — Temporal multi-resolution factor posterior `[EXTENSION]`

**Hypothesis:** in continuous/latent-factor models, better time-scale decomposition could improve posterior or prior factor estimation.

**Scope:** begin with the historical factor predictor. Do not automatically mix future-return posterior semantics into deployment features.

**Measure:** RankIC, posterior-prior discrepancy, predictive uncertainty calibration where available, cost-aware returns.

**Failure:** factor variance/mode collapse increases or gains depend on privileged future information at test time.

## 193. E18 — Multi-resolution code evolution `[EXTENSION]`

**Hypothesis:** a stock's discrete structural type can be more stable at medium scale than at daily scale.

**Approach:** train or analyze code transitions at different temporal scales, then condition loading models on a small set of multi-resolution codes.

**Metrics:** code persistence, occupancy/perplexity, transition entropy, RankIC, regime stability.

**Failure:** codebook redundancy explodes, utilization collapses, or apparent persistence reflects downsampling arithmetic alone.

## 194. E19 — Financial task-specific pooling choice `[EXTENSION]`

**Hypothesis:** averaging raw irregular/stressful stock signals is inferior to a robust or learned pooling operator.

**Compare:** mean, median, trimmed mean, strided conv, max, and quantile pooling under consistent time windows.

**Measure:** post-shock behavior, turnover, rank persistence and worst-month IC.

**Failure:** robust pooling merely deletes predictive spikes or becomes too expensive in the full cross-section.

## 195. E20 — True ablation of novelty, not module pile-up `[EXTENSION]`

Suppose a proposed model adds multi-scale PDM, scale MoE and dynamic factor priors together. The minimal test matrix is:

| PDM | Scale gate | Priors | Purpose |
|---|---|---|---|
| off | off | original | baseline |
| on | off | original | pure PDM |
| off | on | original | gate-only with same scales |
| on | on | original | interaction |
| on | on | modified | extra-prior value |

Then add fixed compute/parameter budget controls, separate seed means, and a held-out test. **Do not claim all three modules are proven beneficial from the full model alone.**

## 196. Prioritization for experimental throughput `[EXTENSION]`

**Fast/low-risk:**

- E01 encoder substitution;
- E02/E03 directional ablation;
- E05 MA-vs-DFT decomposition;
- E07 sum-vs-gated FMM.

**Medium:**

- E09 PRISM-VQ Stage-2 temporal replacement;
- E10 MATCC partial replacement;
- E14 lookback-scale grid.

**Higher complexity / novelty potential:**

- E11 scale-dependent stock relations;
- E16 sparse routing;
- E18 multi-resolution structural codes.

These are prioritization suggestions based on controllability, not promised results.

---
# Part XII — Academic Writing, Presentation, Reproduction and Retrieval

## 197. The paper's Introduction strategy `[SOURCE/INTERPRETATION]`

The Introduction is unusually effective because it does not begin with a very large new architecture. It proceeds by:

1. establishing time-series forecasting difficulty from mixed temporal variation;
2. grouping prior work into backbone families and specialized time-series designs;
3. pointing out that fine/coarse sampling scales naturally reveal different patterns;
4. identifying **two missing capabilities**: past cross-scale mixing and future cross-scale prediction combination;
5. proposing PDM and FMM as one-to-one answers;
6. claiming accuracy and efficiency, later supported through controlled benchmarking.

A research Agent should copy the **argument structure**, not stock phrases or unsupported novelty language.

## 198. Problem-to-module mapping `[INTERPRETATION]`

| Problem | Dedicated design | Experiment that probes it |
|---|---|---|
| mixed dynamics | decompose each scale | cases ⑧/⑨ |
| fine seasonal details fail to reach coarse scale | seasonal bottom-up mixer | case ③ / reversed ⑦ |
| coarse trend does not guide fine scale | trend top-down mixer | case ④ / reversed ⑦ |
| future driven by multiple scales | FMM multi-predictor sum | case ② |
| long sequences expensive for attention | all-MLP temporal layers | Fig.5 / Table 8 |
| module design not obviously unique | alternative decompositions/poolings | Appendix F.2/F.3 |

This is a useful blueprint for writing a coherent experimental section.

## 199. Why the ablation story is persuasive `[INTERPRETATION]`

The authors do not merely delete a component; they systematically try **the opposite direction** of information flow. The result is a hypothesis-driven intervention: if the reason for separate mixing is correct, reversing the flow should be harmful. The table in fact shows precisely that tendency within evaluated tasks.

Finance transfer lesson: for any claimed regime-aware direction or expert specialization, provide an **incorrect/neutral opposite** comparator rather than only a “module removed” baseline.

## 200. The paper's use of appendices `[SOURCE/INTERPRETATION]`

Important hidden technical value appears outside the main nine pages:

- exact model configs;
- forecast-metric formulas;
- memory/time tables;
- seed statistics;
- extra baseline comparisons;
- ten-way ablations on more datasets;
- DFT/conv alternatives;
- tuned baseline protocol;
- explicit limitations and future work.

A research Agent should not read only the abstract and main method: **Appendix F and J are especially relevant to innovation**.

## 201. What to emulate in a new research paper `[INTERPRETATION]`

- Clearly state why the previous architecture's information flow is insufficient.
- Show that new directions/interfaces are task-motivated, not arbitrary layers.
- Preserve fair experimental settings and separately show tuned-baseline comparisons.
- Quantify multiple seeds rather than only best checkpoint.
- Use ablations for each module, including reverse-orientation controls.
- Report performance **and cost**.
- Acknowledge if a more complicated alternative performs better in raw accuracy but is rejected for cost/complexity.
- Distinguish qualitative visualization from quantifiable predictive evidence.

## 202. What NOT to emulate without stronger evidence `[CRITIQUE]`

- Universal claim that one direction must be optimal for all financial signals.
- Assuming moving-average “seasonal” residuals have identified periodic economic meaning.
- Reporting relative gains only under weak/tied default hyperparameter baselines.
- Treating long-horizon MSE results as sufficient evidence for ranking or portfolio construction.
- Omitting the direction-specific comparator when claiming a new mixed-scale router.

## 203. Minimal source-to-code reproduction checklist `[SOURCE/DERIVATION]`

```text
[ ] Acquire exact original datasets and official splits / data providers
[ ] Reproduce $P$, $F$, $C$ for each task
[ ] Build levels 0..M (M+1 total)
[ ] Match average pooling and sequence-length truncation/padding
[ ] Match scale embedding and d_model
[ ] Match decomposition kernel / padding
[ ] Decompose at every scale
[ ] Seasonal bottom-up pass with residual connections
[ ] Trend top-down pass with residual connections
[ ] Scale-wise feedforward and original residual
[ ] Repeat L PDM blocks
[ ] Project every final scale to the same future horizon F
[ ] Sum forecasts BEFORE computing loss
[ ] Apply task-specific MSE or SMAPE training objective
[ ] Adam β1=0.9, β2=0.999
[ ] Match Table 7 dataset-specific d_model, M, layers, LR, batch, epochs
[ ] Repeat runs / preserve seed-level results
[ ] Evaluate correct task metrics
[ ] Compare unified setting separately from broad hyperparameter search
[ ] Reproduce critical cases ②,③,④,⑦,⑧,⑨,⑩
[ ] Compare MA vs DFT decomposition and mean vs strided-conv pooling if extending
```

## 204. Suggested modular code organization `[DERIVATION]`

```text
timemixer/
├── data/
│   ├── time_series_loader.py
│   └── multiscale_sampler.py
├── model/
│   ├── embedding.py
│   ├── decomposition.py
│   ├── bottom_up_seasonal_mixer.py
│   ├── top_down_trend_mixer.py
│   ├── past_decomposable_mixing.py
│   ├── future_multipredictor_mixing.py
│   ├── model.py
│   └── output_head.py
├── training/
│   ├── losses.py
│   ├── trainer.py
│   └── configs.py
└── diagnostics/
    ├── ablation.py
    ├── scale_contributions.py
    └── runtime_memory.py
```

This is an Agent-generated logical organization; it is not claimed to be the official repository layout.

## 205. Implementation hooks for a finance codebase `[EXTENSION]`

### Temporal mixing module

```text
Input:
  stock_features: [N, T, C]
Optional:
  market_context: [MktDim]
Output:
  either score per stock: [N]
  OR latent representation: [N,D]
  OR multiscale sequences: list[[N,T_m,D]]
```

### Cross-sectional interaction socket

```text
Input:
  per-stock representations: [N,D] or [N,T_m,D]
Output:
  context-aware stock representations: [N,D] or [N,T_m,D]
```

### Ranking head

```text
Input:
  per-stock representation / scale-head scores
Output:
  y_hat: [N]
Objective:
  same Qlib task label / ranking metric as local baselines
```

## 206. Common implementation mistakes `[CRITIQUE]`

1. Treat `M=3` as three scales instead of four levels $0..3$.
2. Reverse bottom-up seasonal and top-down trend directions.
3. Decompose only original scale instead of every scale.
4. Compute loss on each scale instead of the ensemble.
5. Average outputs in code, then attribute difference to TimeMixer originality rather than tested FMM variant.
6. Shrink a short stock history to unusably small sequences.
7. Treat `C=158` as 158 different stocks instead of 158 features.
8. Assume TimeMixer automatically implements stock-to-stock relations.
9. Use PatchTST's original $P=512$ history and TimeMixer's $P=96$ history as directly comparable under different splits.
10. Misattribute optional finance adaptations as findings of the original paper.
11. Call “seasonal” residual a verified economic seasonal premium without tests.
12. Forget M4 uses a different original training loss/configuration.

## 207. Cross-paper novelty search instructions `[EXTENSION]`

Before claiming publication-level novelty for a TimeMixer-inspired stock method, search:

- `papers/foundations/PatchTST.md`: patch semantics, channel independence and pretraining.
- `papers/baseline/MATCC_2024.md`: stock trend/residual decomposition and RWKV.
- `papers/baseline/MASTER_2024.md`: temporal relay and cross-stock interactions.
- `papers/baseline/PRISM-VQ_2026.md`: structural VQ, finance priors and sparse MoE.
- `papers/related_work/`: other finance applications of multiscale mixing.
- `papers/frontier/`: more recent multiscale, hierarchical temporal and adaptive routing mechanisms.

Do not infer novelty merely because the current personal paper index lacks an implementation. A current literature check and code review are needed.

## 208. Source location map (27-page PDF)

| PDF pages | Source content | Why Agent should read |
|---|---|---|
| 1–2 | abstract, motivation, high-level contributions | conceptual problem |
| 3–4 | overall multiscale formulation and Fig.1 | architecture graph, symbols |
| 4–5 | PDM equations (3–5), FMM equation (6), Fig.2 | exact mechanism |
| 6 | Table 1 benchmarks / Table 2 long-term | dataset and main model comparison |
| 7 | Tables 3–4 short-term outcomes | PeMS and M4 evidence |
| 8 | Table 5 ten variants, Fig.3 | direction-specific ablations |
| 9 | Figs.4–6, conclusion | per-scale prediction and efficiency |
| 10–11 | references, code link | original prior work trace |
| 12–14 | dataset counts, metric definitions, config Table 7, efficiency Table 8 | reproducibility |
| 14–15 | Tables 9–11 statistics; Table 12 depth | uncertainty and sensitivity |
| 16–17 | Tables 13–14 unified vs searched | fairness of comparisons |
| 18–19 | extra full ablations, alternative decompositions | critical negative/counterfactual results |
| 20–21 | convolution vs pooling, sum vs mean, deeper ablations | alternative implementation evidence |
| 22 | extra baselines, spectral comparison, **Appendix J limitations** | novelty and open problems |
| 23–24 | extra-baseline tables | additional comparative results |
| 24–27 | forecasting visualizations | qualitative behavior |

## 209. Suggested targeted retrieval plan

For **architecture understanding**, read §§8–39.

For **dataset/hyperparameter auditing**, read §§40–55 and Table 7.

For **whether it really outperforms other models**, read §§56–64 and §§96–105.

For **component-level innovation / alternative designs**, read §§65–85 and §§116–157.

For **short-window finance transfer**, read §§158–195.

For **reproducing model code**, read §§203–206.

Avoid loading every historical benchmark number for a simple mechanism question: retrieve the relevant original source sections and tables on demand.

## 210. Machine-readable mechanism summary

```yaml
paper: TimeMixer_2024
paper_type: foundations
core_idea: decomposable_multiscale_mixing
primary_primitives:
  - name: multi_resolution_average_pooling
    input: "[P,C]"
    output: "list([P_m,C])"
    paper_evidence: "Sec 3.1, p3"
  - name: seasonal_bottom_up
    direction: "fine_to_coarse"
    paper_evidence: "Eq 4, p5; Table 5, p8"
  - name: trend_top_down
    direction: "coarse_to_fine"
    paper_evidence: "Eq 5, p5; Table 5, p8"
  - name: scale_specific_future_predictors
    aggregation: "sum"
    paper_evidence: "Eq 6, p5"
comparators:
  - PatchTST
  - DLinear
  - TimesNet
  - Autoformer
  - FEDformer
original_stock_backtest: false
original_rankic: false
supports_finance_transfer_as_hypothesis: true
important_original_negative_results:
  - DFT_seasonal_trend_better_than_MA_in_selected_tests
  - strided_conv_slightly_better_than_average_pool_in_selected_tests
  - direction_reversal_degrades_performance
  - broad_hyperparameter_search_shrinks_margin_over_PatchTST
```

## 211. Exact source-result lookup: selected per-horizon Table 13 (TimeMixer vs PatchTST) `[SOURCE, Appendix E p.16]`

Each cell is `MSE/MAE`. Unlike Table 2's average, these are **individual horizons** from the unified input-length protocol. This enables precise comparisons without accidentally mixing forecast horizons.

| Dataset | Future F | TimeMixer | PatchTST |
|---|---:|---|---|
| Weather | 96 | 0.163/0.209 | 0.186/0.227 |
| Weather | 192 | 0.208/0.250 | 0.234/0.265 |
| Weather | 336 | 0.251/0.287 | 0.284/0.301 |
| Weather | 720 | 0.339/0.341 | 0.356/0.349 |
| Solar-Energy | 96 | 0.189/0.259 | 0.265/0.323 |
| Solar-Energy | 192 | 0.222/0.283 | 0.288/0.332 |
| Solar-Energy | 336 | 0.231/0.292 | 0.301/0.339 |
| Solar-Energy | 720 | 0.223/0.285 | 0.295/0.336 |
| Electricity | 96 | 0.153/0.247 | 0.190/0.296 |
| Electricity | 192 | 0.166/0.256 | 0.199/0.304 |
| Electricity | 336 | 0.185/0.277 | 0.217/0.319 |
| Electricity | 720 | 0.225/0.310 | 0.258/0.352 |
| Traffic | 96 | 0.462/0.285 | 0.526/0.347 |
| Traffic | 192 | 0.473/0.296 | 0.522/0.332 |
| Traffic | 336 | 0.498/0.296 | 0.517/0.334 |
| Traffic | 720 | 0.506/0.313 | 0.552/0.352 |
| ETTh1 | 96 | 0.375/0.400 | 0.460/0.447 |
| ETTh1 | 192 | 0.429/0.421 | 0.512/0.477 |
| ETTh1 | 336 | 0.484/0.458 | 0.546/0.496 |
| ETTh1 | 720 | 0.498/0.482 | 0.544/0.517 |
| ETTh2 | 96 | 0.289/0.341 | 0.308/0.355 |
| ETTh2 | 192 | 0.372/0.392 | 0.393/0.405 |
| ETTh2 | 336 | 0.386/0.414 | 0.427/0.436 |
| ETTh2 | 720 | 0.412/0.434 | 0.436/0.450 |
| ETTm1 | 96 | 0.320/0.357 | 0.352/0.374 |
| ETTm1 | 192 | 0.361/0.381 | 0.390/0.393 |
| ETTm1 | 336 | 0.390/0.404 | 0.421/0.414 |
| ETTm1 | 720 | 0.454/0.441 | 0.462/0.449 |
| ETTm2 | 96 | 0.175/0.258 | 0.183/0.270 |
| ETTm2 | 192 | 0.237/0.299 | 0.255/0.314 |
| ETTm2 | 336 | 0.298/0.340 | 0.309/0.347 |
| ETTm2 | 720 | 0.391/0.396 | 0.412/0.404 |

## 212. Notes on transcription of numerical results

- **The source of all table values is the user-uploaded original PDF** and its conference appendix.
- When a source offers only aggregate metrics, the document does not fabricate per-horizon cells.
- When a table contains apparent typos (e.g. frequency label, runtime cell), the text calls them out rather than rewriting source values invisibly.
- All downstream IC/RankIC/Sharpe statements are marked as hypothetical and refer to potential **new financial experiments**, not TimeMixer's reported results.
- The original framework is end-to-end forecasting with MLP temporal mixers; treating it as a two-stage VQ or MoE algorithm is incorrect.

## 213. Final Agent takeaways

**Research problem:** complex temporal variation appears differently at fine and coarse sampling scales.

**Signature mechanism:** decompose each scale; seasonal/residual information flows fine→coarse, trend information flows coarse→fine.

**Signature equations:**

$$
\mathbf s_m\leftarrow\mathbf s_m+\operatorname{BUM}(\mathbf s_{m-1}),
\qquad
\mathbf t_m\leftarrow\mathbf t_m+\operatorname{TDM}(\mathbf t_{m+1}),
$$

$$
\hat{\mathbf x}=\sum_{m=0}^{M}\operatorname{Predictor}_m(\mathbf x_m^L).
$$

**Most important evidence:** reversing the mixing directions and removing seasonal/trend mixing degrades performance; removing FMM also hurts.

**Most important caveat:** authors themselves find DFT seasonal-trend decomposition and strided convolution slightly improve raw accuracy in selected settings, while choosing simpler operators for efficiency.

**Most important comparator:** PatchTST's patched Transformer. The tuned accuracy margin between them can be very small on some benchmarks, even though TimeMixer retains efficiency advantages.

**Most important original future work:** parameter-efficient alternative mixing, explicit variate-dimension mixing, and theoretical analysis of optimality (Appendix J).

**Most important finance-adaptation constraint:** short 8–20-day stock lookbacks have far fewer meaningful resolution levels than 96–3072-point general forecasting histories. Do not blindly import original depths or claim MSE gains imply RankIC gains.

## 214. Compact retrieval summary

TimeMixer (Wang et al., ICLR 2024) is a fully MLP-based model for long- and short-term time-series forecasting. It forms an average-pooled pyramid of input observations across levels $0..M$, decomposes **each** scale into moving-average trend and residual/seasonal components, and processes them with Past-Decomposable-Mixing (PDM). Its key asymmetry is seasonal fine-to-coarse bottom-up mixing and trend coarse-to-fine top-down mixing, implemented with learnable temporal MLPs and residual connections. After stacking PDM blocks, Future-Multipredictor-Mixing (FMM) builds one horizon-matched linear predictor per scale and **sums** the outputs, with the forecasting loss applied to the combined prediction. The original paper evaluates 18 dataset/subset groups across long-term ETT/Weather/Solar/Electricity/Traffic and short-term PeMS/M4 benchmarks, compares with 15 main baselines including PatchTST, and reports strong accuracy with low GPU/time overhead. Ten-way ablations support separation and directionality of seasonal/trend mixing and the value of multi-scale future fusion. Crucially, Appendix F shows that a DFT-based seasonal/trend decomposition and strided convolution downsampling can outperform default moving-average/average pooling on selected tasks; the authors retain simpler defaults for efficiency. Appendix E finds more extensive tuning reduces TimeMixer's margin over PatchTST. Appendix J explicitly lists length-dependent temporal linear parameter growth, variable-dimension mixing, and theoretical analysis as future work. For cross-sectional stock learning, TimeMixer supplies reusable ideas in hierarchical temporal representation and conditional multiscale forecasting, but **does not itself validate any financial ranking model, Qlib protocol, or portfolio result**.
