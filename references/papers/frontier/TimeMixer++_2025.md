---
paper_id: TimeMixerPlusPlus_2025
title: "TimeMixer++: A General Time Series Pattern Machine for Universal Predictive Analysis"
model_name: TimeMixer++
authors:
  - Shiyu Wang
  - Jiawei Li
  - Xiaoming Shi
  - Zhou Ye
  - Baichuan Mo
  - Wenze Lin
  - Shengtong Ju
  - Zhixuan Chu
  - Ming Jin
venue: "International Conference on Learning Representations (ICLR)"
year: 2025
paper_type: frontier
subtype:
  - universal_time_series_analysis
  - multiscale_representation
  - multi_resolution_time_imaging
  - axial_attention
  - time_frequency_modeling
  - hierarchical_mixing
publication_status: published_conference_paper
source_pdf: "ICLR-2025-timemixer-a-general-time-series-pattern-machine-for-universal-predictive-analysis-Paper-Conference.pdf"
source_pages: 37
source_sections: ["1 Introduction", "2 Related Work", "3 TimeMixer++", "4 Experiments", "5 Conclusion", "Appendix A-K"]
repository: "https://github.com/kwuking/TimeMixer"
predecessor: TimeMixer_2024
tasks:
  - long_term_forecasting
  - univariate_short_term_forecasting
  - multivariate_short_term_forecasting
  - imputation
  - classification
  - anomaly_detection
  - few_shot_forecasting
  - zero_shot_forecasting
core_components:
  - coarse_scale_channel_attention
  - input_embedding
  - MRTI
  - TID
  - MCM
  - MRM
  - task_specific_output_heads
mechanisms:
  - strided_convolution_multiscale_downsampling
  - FFT_topK_period_selection
  - multiresolution_1D_to_2D_time_imaging
  - row_column_axis_attention
  - bottom_up_seasonal_2D_convolution
  - top_down_trend_transposed_2D_convolution
  - amplitude_weighted_resolution_mixing
  - scale_head_ensemble
  - residual_mixerblocks
research_agent_priority: very_high
research_agent_use: "Frontier mechanism mining; temporal/feature interaction, multi-scale spectral structure and financial adaptation hypotheses."
tags: [TimeMixer++, MRTI, TID, MCM, MRM, FFT, dual-axis-attention, multiscale, time-image, finance-transfer]
source_convention: "[SOURCE] paper-reported; [DERIVATION] math reconstructed from source; [CRITIQUE] evidence/limitations; [EXTENSION] proposed research, not authored by the paper; [VERIFY] ambiguity or conflicting information"
---

# TimeMixer++: A General Time Series Pattern Machine for Universal Predictive Analysis

> **Classification**: `papers/frontier/TimeMixer++_2025.md`  
> **Source**: Wang et al., *ICLR 2025*, 37-page version (10-page main text + references + technical appendices).  
> **Purpose**: High-density, source-grounded research-agent knowledge artifact, **not** a reproduction of the PDF, and **not** a claim that this model has been validated for cross-sectional stock ranking.

## 00. Reader protocol and evidence tags

- `[SOURCE]` = statement, formula, setting, table value, or explicit limitation **reported by the authors**; identify paper section, printed page, figure, or table.
- `[DERIVATION]` = algebra, tensor bookkeeping, or implementation-level interpretation that follows from the reported equations but is not presented verbatim.
- `[CRITIQUE]` = evidence audit, ambiguity, limitation, comparator, or a distinction between model output and the authors' interpretation.
- `[EXTENSION]` = a new financial/ML hypothesis or experiment suggested for a future research agent; **not a result of this paper**.
- `[VERIFY]` = internally inconsistent reporting, unclear equation, absent training information, or a detail requiring released code.
- **Trust hierarchy**: source formula / table / explicit implementation notes → parsed caption and figure → author interpretation → this document's hypotheses. A Markdown adaptation never replaces the PDF or actual implementation.
- This paper is a general time-series architecture study. It does **not** evaluate Qlib Alpha158, CSI300, S&P500, RankIC, IC, or stock-selection backtests.

## 01. One-screen mechanism summary

[SOURCE; §3, pp. 3–6]

```text
Multivariate series x [T,C]
       │
       ▼
Strided Conv downsampling → scales m = 0..M
       │
       ├── coarsest-scale, variate-wise Channel Attention
       │
       ▼
Embedding → multiscale representations [T/2^m, d_model]
       │
       ▼
Residual MixerBlocks × L
  ┌────────────────────────────────────────────────────────┐
  │ MRTI: FFT(coarsest) → top-K periods                     │
  │       pad/reshape every scale into K time-images        │
  │                                                        │
  │ TID: 2D Conv + dual-axis attention                     │
  │       seasonal image / trend image                     │
  │                                                        │
  │ MCM: seasonal images fine → coarse via 2D Conv         │
  │       trend images coarse → fine via 2D TransConv       │
  │                                                        │
  │ MRM: FFT-amplitude-weighted sum over K resolutions     │
  │       → 1D representation per scale                     │
  └────────────────────────────────────────────────────────┘
       │
       ▼
Scale-specialized task heads → scale ensemble → output
```

**What changes relative to TimeMixer 2024?**

- 2024: 1D average downsampling; moving-average trend/season decomposition; time-axis MLP seasonal/trend mixing; scale-specific forecasting heads.
- 2025: learned strided-convolution downsampling; coarse-scale **cross-variable attention**; Fourier-selected **2D time images**; **latent-space dual-axis attention decomposition**; **2D convolutional multiscale mixing**; **FFT-amplitude adaptive resolution fusion**; scale/task-dependent output heads.
- The bidirectional information-flow bias stays: **fine→coarse seasonality**, **coarse→fine trend**.
- The ambition grows from forecasting to a general **Time Series Pattern Machine (TSPM)** spanning eight task types.

## 02. Bibliographic identification

[SOURCE; title page]

- Full title: *TimeMixer++: A General Time Series Pattern Machine for Universal Predictive Analysis*.
- Venue: **ICLR 2025**; this is the final conference-paper version in the supplied PDF.
- Authors: Shiyu Wang, Jiawei Li, Xiaoming Shi, Zhou Ye, Baichuan Mo, Wenze Lin, Shengtong Ju, Zhixuan Chu, Ming Jin.
- Affiliations include Griffith University, HKUST (Guangzhou), MIT, and Zhejiang University; consult the title page for author-to-institution superscripts.
- Implementation location referenced by the paper: `https://github.com/kwuking/TimeMixer` (Appendix A, p.16).
- Intellectual predecessor: *TimeMixer: Decomposable Multiscale Mixing for Time Series Forecasting* (ICLR 2024).
- **Important**: the paper uses the term **TimeMixer++** (two plus signs), not TimeMixer+.

## 03. Main research problem

[SOURCE; Abstract, Introduction, pp.1–3]

The paper asks how a **single general-purpose time-series pattern extractor** can represent patterns needed by substantially different downstream tasks: long-horizon prediction, short-horizon prediction, imputing missing observations, anomaly detection, classification, and data-scarce/generalization scenarios. Multiple temporal resolutions coexist in real data: fast local fluctuations, periodic repetition, slowly varying trends, and different variable-to-variable relations.

Formally, let observed series be:

\[
X\in\mathbb R^{T\times C},
\]

with history length \(T\) and \(C\) observed variates. An encoder should deliver a rich, task-adaptable set of temporal representations. Task-specific heads produce forecasts, reconstructions, classes, or anomaly scores. The exact target/output depends on the task; the model is **not** one universal scalar prediction function.

## 04. Why generic token attention is not automatically enough

[SOURCE; Introduction, p.2]

- Language tokens usually occupy relatively distinct contexts; the same observation time in time series can simultaneously participate in multiple cycles and trends.
- A single fixed-size token partition can obscure time-varying, overlapping periodic structure.
- Modeling one frequency, one temporal resolution, or one trend decomposition globally may underrepresent task-dependent patterns.
- Different applications value different aspects: e.g., stable recurring structure may matter for forecasting, while fine irregular structure can matter for anomaly detection.

[CRITIQUE] These are architectural motivations, not a formal proof that Transformers cannot represent those functions. Transformer-based models remain important baselines.

## 05. Time Series Pattern Machine (TSPM) framing

[SOURCE; Introduction, Figure 1]

A TSPM is intended to serve many tasks with one adaptable backbone. The authors motivate it using **CKA (centered kernel alignment)** similarity between the first and last network layers. Across examples, layer representation consistency/diversity varies with the task.

[CRITIQUE] CKA is an **observational similarity metric** between learned feature representations. It is not a guarantee of generalization or a direct measurement of financial signal quality. Treat the authors' task-specific CKA discussion as descriptive rather than as a causal explanation.

## 06. Exactly eight evaluation task families

[SOURCE; §4, p.6]

1. Long-term forecasting.
2. Univariate short-term forecasting (M4).
3. Multivariate short-term forecasting (PeMS).
4. Imputation.
5. Time-series classification.
6. Anomaly detection.
7. Few-shot forecasting.
8. Zero-shot forecasting.

[SOURCE] The authors claim approximately **30 datasets / benchmarks** and **27 advanced baselines**, across these task families. Do not confuse the number of task types, datasets, prediction horizons, and baseline implementations.

## 07. Distinguish frequency, period, sampling scale, and resolution

[DERIVATION; based on §3]

| Concept | Paper's role | What not to confuse it with |
|---|---|---|
| Sampling scale \(m\) | Recursive stride-2 downsampling of historical time | Output forecast horizon |
| Fourier bin \(f_k\) | Selected FFT frequency index / spectral component | Stock feature index |
| Period length \(p_k\) | Time-image row/period dimension derived from frequency | Number of historical stocks |
| Resolution \(k\) | One FFT-chosen time-image periodicity | A sampling scale |
| Time image | 2D reshaping of one time-series representation | Actual image modality or raw spectrogram |
| Seasonal image | Learned branch after one axis of attention | Verified calendar seasonality |
| Trend image | Learned branch after the other axis of attention | Statistically identified economic trend |
| \(M\) | Number of downsampling steps; \(M+1\) scales | MoE expert count |
| \(K\) | Number of selected frequency/period resolutions | Top-K stock portfolio size |
| \(L\) | Number of stacked MixerBlocks | Forecast length |

[CRITIQUE] Terms such as “seasonal” and “trend” name learned branches/inductive biases. The learned tensors are not mathematically guaranteed to disentangle the ground-truth components uniquely.

## 08. Full mathematical input/output contract

[SOURCE; §3.1]

\[
X_{\mathrm{init}}=\{x_0,x_1,\ldots,x_M\},
\qquad x_0=x\in\mathbb R^{T\times C}.
\]

\[
x_m\in\mathbb R^{\lfloor T/2^m\rfloor\times C}.
\]

The embedded multiscale pattern set is:

\[
X^0=\operatorname{Embed}(X_{\mathrm{init}}),\qquad
x_m^0\in\mathbb R^{\lfloor T/2^m\rfloor\times d_{\mathrm{model}}}.
\]

The stack of \(L\) MixerBlocks returns \(X^L\), after which task-specific scale heads are ensembled. For a forecasting head, a typical output is \(\hat y\in\mathbb R^{F\times C}\); for classification/anomaly/imputation, shape differs and must be checked in the released implementation.

## 09. Design evolution from TimeMixer 2024

[SOURCE] Previous TimeMixer used Past-Decomposable-Mixing (PDM) and Future-Multipredictor-Mixing (FMM), mainly with simple MLP layers. TimeMixer++ keeps the multiscale philosophy and heterogeneous-scale heads but changes how patterns are discovered.

[DERIVATION] The transition is best viewed as:

\[
\underbrace{\text{multi-scale 1D decomposition/mixing}}_{\text{TimeMixer 2024}}
\quad\longrightarrow\quad
\underbrace{\text{multi-scale}\times\text{multi-period 2D latent modeling}}_{\text{TimeMixer++ 2025}}.
\]

This is **not** a drop-in replacement of one MLP by attention; it changes tokenization, feature interactions, latent decomposition, scale transitions, and task scope together.

## 10. Overall architecture in Figure 2, page 4

[SOURCE; Fig.2] The figure shows: a multiscale one-dimensional input pyramid → embeddings → a MixerBlock repeated \(L\) times. Within a block, (a) **multi-resolution time imaging** generates a grid of 2D representations indexed by both \(m\) and \(k\); (b) **time image decomposition** uses two different attention axes; (c) **multi-scale mixing** exchanges information vertically along the scale axis; (d) **multi-resolution mixing** fuses periodic views at fixed scale; (e) output heads aggregate the scales.

[CRITIQUE] A diagram alone does not specify padding masks, FFT normalization, learned convolution widths, or task-specific loss weighting. Avoid asserting unreported details.

## 11. Multi-scale downsampling: Eq. (1)

[SOURCE; §3, p.4]

\[
x_m=\operatorname{Conv}(x_{m-1};\operatorname{stride}=2),
\qquad m=1,\dots,M.
\tag{1}
\]

- Each subsequent scale has approximately half the temporal length.
- Downsampling uses convolution (unlike **average pooling** in the original TimeMixer 2024).
- The paper sets stride = 2 partly to maximize the number of possible scales.

[VERIFY] Exact kernel size, padding, initialization, and length behavior should be read from code. The idealized floor shape in the paper assumes compatible implementation choices.

## 12. Scale count and variable count are separate axes

[DERIVATION]

For \(T=192\) and \(M=3\), the ideal temporal lengths are 192, 96, 48, 24. The number of input variables stays \(C\) before embedding; \(d_{\mathrm{model}}\) is an **embedding width**, not a number of timescales.

[EXTENSION] A financial Agent should distinguish a hierarchy across **time** from a hierarchy across **stocks**, **sectors**, or **factor families**. TimeMixer++ explicitly defines the first, and mixes channels; it does not natively define a stock-universe graph.

## 13. Coarsest-scale channel attention: Eq. (2)

[SOURCE; §3.1, p.4]

At \(m=M\), attention operates **over variates/channels**:

\[
x_M\leftarrow \operatorname{ChannelAttn}(Q_M,K_M,V_M),
\tag{2}
\]

\[
Q_M,K_M,V_M\in\mathbb R^{C\times \lfloor T/2^M\rfloor}.
\]

- The paper explicitly contrasts its **channel mixing** with channel-independent time-series modeling.
- Coarsest-scale temporal context is used before the embedding process.
- Cross-variable dependencies can be modeled without mixing all variates at every finest-scale time step.

[CRITIQUE] Channel attention over 158 Alpha158 features is conceptually different from attention over 300/500 stocks. Both can be modeled, but the axes and computational burdens differ.

## 14. Why not Channel Independence?

[SOURCE; §3.1 and Appendix B]

PatchTST and related channel-independent methods are cited as approaches that avoid overmixing variables. TimeMixer++ deliberately adopts the opposite inductive bias at the coarsest scale: important variable dependencies should be learned globally.

[EXTENSION] This establishes a useful controlled finance experiment:

\[
\text{CI temporal encoder}
\quad vs.\quad
\text{coarse cross-feature attention}
\quad vs.\quad
\text{fine dense cross-feature attention}.
\]

It does **not** prove that full cross-stock mixing is always superior to stock-wise modeling.

## 15. Input embedding

[SOURCE; Eq.2 and §3.1]

\[
X^0=\operatorname{Embed}(X_{\mathrm{init}}),
\qquad x_m^0\in\mathbb R^{\lfloor T/2^m\rfloor\times d_{\mathrm{model}}}.
\]

The embedding maps observed channels to a common latent width, enabling shared MixerBlock computation. A research Agent should preserve exact shape conventions in code (`[batch,time,channels]` vs `[batch,channels,time]`). The PDF specifies mathematical shapes but not every transpose/permutation.

## 16. MixerBlock residual contract: Eq. (4)

[SOURCE; §3.2, p.5]

\[
X^{l+1}=\operatorname{LayerNorm}
\big(X^l+\operatorname{MixerBlock}(X^l)\big).
\tag{4}
\]

A MixerBlock is a full four-stage operator (**MRTI→TID→MCM→MRM**). LayerNorm stabilizes residual iteration across \(L\) blocks. The residual should be interpreted scale-wise, because its input/output are both sets of multiscale tensors.

## 17. Four internal mechanisms and their roles

| Submodule | Full name | Input → Output | Design intent |
|---|---|---|---|
| MRTI | Multi-Resolution Time Imaging | 1D at all scales → 2D views indexed by scale/period | expose periodic geometry |
| TID | Time Image Decomposition | 2D time image → seasonal/trend images | learn axis-specialized patterns |
| MCM | Multi-Scale Mixing | seasonal/trend image pyramids → mixed pyramids | transfer local and macro context |
| MRM | Multi-Resolution Mixing | K period views per scale → fused 1D scale sequence | combine complementary periodicities |

[SOURCE; §3.2, Eqs.5–11] These mechanisms operate **inside each MixerBlock**, rather than forming four independent models.

## 18. MRTI — select frequencies from the coarsest scale

[SOURCE; §3.2, p.5; Eq.5]

\[
A,\{f_1,\ldots,f_K\},\{p_1,\ldots,p_K\}
=\operatorname{FFT}(x_M^l).
\tag{5}
\]

- FFT is applied to the **coarsest-scale hidden representation**, not independently to every finer scale.
- Select \(K\) spectral components with the largest amplitudes.
- The selected periods are then **shared across all scales**.
- Conceptually, coarsest context decides which periodic patterns deserve repeated examination at all scales.

[CRITIQUE] “Top-K periods” are selected based on spectral magnitude, which is not necessarily predictive relevance to future returns or anomaly risk. FFT features can be dominated by trend, noise, harmonics, and sampling artifacts.

## 19. Period conversion and notation nuance

[SOURCE; Eq.5] The PDF writes a period conversion resembling:

\[
p_k=\left\lfloor\frac{T}{f_k}\right\rfloor,
\]

where \(f_k\) indexes selected spectral components.

[VERIFY] FFT is run on the length-\(\lfloor T/2^M\rfloor\) **coarsest** representation, but the period expression is written with \(T\), the original input length. The exact calibration/rounding and whether frequencies are rescaled across levels must be verified against source code. Repeating the equation without noting this tension risks wrong tensor sizes or mismatched period semantics.

## 20. MRTI — pad and reshape: Eq. (6)

[SOURCE; §3.2, pp.5–6]

For scale \(m\) and selected period \(p_k\):

\[
\widetilde{x}_{m,k}^l
=\operatorname{Padding}_{m,k}(x_m^l),
\]

\[
z_m^{(l,k)}=
\operatorname{Reshape}_{1D\to2D,m,k}(\widetilde{x}_{m,k}^l).
\tag{6}
\]

If \(n_m=\lfloor T/2^m\rfloor\), padded temporal length is:

\[
\widetilde{n}_{m,k}=p_k\Big\lceil\frac{n_m}{p_k}\Big\rceil.
\]

Each resulting time image has form:

\[
z_m^{(l,k)}\in\mathbb R^{p_k\times\lceil n_m/p_k\rceil\times d_{\mathrm{model}}}.
\]

**Important**: a time image is a reshaping of a padded 1D temporal tensor according to an estimated period; it is **not** a spectrogram whose pixel values are FFT coefficients. FFT is used for period selection and fusion weights.

## 21. Correct indexing of scale and resolution

[DERIVATION]

The collection inside one block is:

\[
\big\{z_m^{(l,k)}\;|\;m=0,\dots,M;\ k=1,\dots,K\big\}.
\]

This yields \((M+1)K\) 2D views, ignoring batch/channel axes. With \(M=3,K=3\), there are 12 views per block. A code implementation can flatten (batch×scale×frequency) internally, but must recover correct group membership before MCM and MRM.

## 22. What the image's two axes mean

[SOURCE; §3.2 and Figs.6–7]

- Period dimension has \(p_k\) positions within one cycle.
- The other dimension counts repeated segments/periods after temporal reshaping.
- The paper names *column-axis attention* as the **seasonality** branch and *row-axis attention* as the **trend** branch.

[VERIFY] The printed equation/text alternates descriptions such as “rows/columns correspond to periods” and query/key tensor shapes in ways that can be confusing under row-major image conventions. Preserve the **paper's labels** rather than silently swapping the two operations; validate exact permute/transpose directions in code.

## 23. Why 2D imaging may matter

[DERIVATION/INTERPRETATION]

A 1D pattern like repeated short cycles is not necessarily spatially local when represented as a flat vector. Reshaping by \(p_k\) places corresponding within-cycle positions and adjacent cycles along structured axes. This allows:

- a local spatial kernel to capture repeatable within-cycle motifs;
- axial attention to inspect periodic and cross-cycle dependencies separately;
- different \(k\) to offer different periodic partition hypotheses;
- scale mixing to link micro-patterns with macro-patterns.

[CRITIQUE] This is an inductive bias that may help strongly periodic data; finance may not exhibit stable periodic structure at 8- or 20-day input length.

## 24. MRTI computational and causal boundaries

- Historical FFT, padding and reshaping are permissible at inference if applied **only to the observed lookback window**.
- Do not perform FFT on the full sample including future test labels and feed period selection back into old predictions.
- Peak-period selection is piecewise/discrete and can shift discontinuously as spectral energy ordering changes.
- Zero padding can create synthetic boundaries. Without masking, attention may learn from padded zeros as if they were real samples.
- Source does not provide a full theoretical guarantee of spectral consistency across sampling scales.

## 25. TID — learned decomposition in a latent time-image space

[SOURCE; §3.2, p.6; Eq.7]

Instead of decomposing raw 1D observations by an arithmetic moving average, the authors use **two separate attention axes** on hidden 2D time images. With queries/keys/values produced by shared 2D convolutions:

\[
s_m^{(l,k)}=
\operatorname{Attention}_{\mathrm{col}}(Q_{\mathrm{col}},K_{\mathrm{col}},V_{\mathrm{col}}),
\]

\[
t_m^{(l,k)}=
\operatorname{Attention}_{\mathrm{row}}(Q_{\mathrm{row}},K_{\mathrm{row}},V_{\mathrm{row}}).
\tag{7}
\]

Both outputs retain \(p_k\times\lceil n_m/p_k\rceil\times d_{\mathrm{model}}\) dimensions.

## 26. TID axis specialization

[SOURCE; pp.6,18; Fig.7]

- **Column-axis / seasonal branch:** author-intended dependency extraction within repeated periodic structure.
- **Row-axis / trend branch:** author-intended patterns across periods and slowly evolving structure.
- Only one selected axis receives attention at a time; the other axis is batched/transposed to reduce global 2D attention cost.
- Both branches use Q/K/V projections derived via 2D convolutions, not raw dot products of the original observations.

[CRITIQUE] The two terms need not be orthogonal or statistically independent; no unique disentanglement theorem is given.

## 27. TID versus shallow decomposition

| Aspect | TimeMixer 2024 | TimeMixer++ 2025 |
|---|---|---|
| Decomposition domain | One-dimensional time-series features | Embedded two-dimensional time images |
| Separation operation | Moving average / subtraction | Distinct axial attention branches |
| Which periods? | Fixed aggregation scales | FFT-selected multiple periodic resolutions |
| Representation output | seasonal/trend time sequences | seasonal/trend time images for every \(m,k\) |
| Modeling inductive bias | Smooth trend vs local variation | Learned dependencies aligned to periodic geometry |
| Main cost concern | Length-dependent dense MLP mixing | FFT, 2D conv, axial attention, image storage |

[EXTENSION] Comparisons must hold scale count and model capacity constant; replacing the entire 2024 pipeline with the 2025 architecture does **not** isolate TID by itself.

## 28. MCM — seasonal bottom-up spatial mixing: Eq. (8)

[SOURCE; §3.2, p.6]

At a fixed resolution \(k\), for \(m=1,\dots,M\):

\[
s_m^{(l,k)}\leftarrow
s_m^{(l,k)}+
\operatorname{2DConv}\big(s_{m-1}^{(l,k)}\big).
\tag{8}
\]

The downscale residual message uses two 2D convolution layers and temporal stride 2. Fine-scale periodic detail is propagated to coarser levels.

[DERIVATION] For this addition to be valid, the convolution output must match the target scale's period/segment/channel shape. In a faithful implementation confirm which of the two image axes is strided and how asymmetric/padded lengths are aligned.

## 29. MCM — trend top-down spatial mixing: Eq. (9)

[SOURCE; §3.2, p.6]

For \(m=M-1,\dots,0\):

\[
t_m^{(l,k)}\leftarrow
 t_m^{(l,k)}+
\operatorname{2DTransConv}\big(t_{m+1}^{(l,k)}\big).
\tag{9}
\]

The upsampling message uses two 2D transposed-convolution layers, transferring coarse global context to finer temporal resolution.

## 30. Why the directions are asymmetric

[SOURCE/DERIVATION]

- Seasonal patterns can be composed from fine local repetitions → **bottom-up**.
- Coarse-scale inputs suppress rapid variation and highlight broad movement → trend information can guide **top-down** finer-scale representations.
- The original TimeMixer already established the direction choice experimentally; TimeMixer++ applies it to 2D learned features.

[CRITIQUE] It is a design hypothesis, not a universal statement that every “seasonal” characteristic in a stock return obeys this direction.

## 31. Merge seasonal/trend images and restore temporal shape: Eq. (10)

[SOURCE; §3.2, p.6]

\[
\widetilde z_m^{(l,k)}
=\operatorname{Reshape}_{2D\to1D}
\big(s_m^{(l,k)}+t_m^{(l,k)}\big).
\tag{10}
\]

- The paper sums the two learned branches.
- Inverse reshape returns a 1D sequence with the padded temporal length.
- [VERIFY] For exact original length \(n_m\), any trailing padding should be removed before scale-wise residual addition. The excerpted formula does not independently spell out cropping for all cases; check the released code.

## 32. MRM — period-amplitude adaptive mixing: Eq. (11)

[SOURCE; §3.2, p.6]

\[
(\widehat A_{f_1},\dots,\widehat A_{f_K})
=\operatorname{Softmax}(A_{f_1},\dots,A_{f_K}),
\]

\[
\overline x_m^{l}
=\sum_{k=1}^{K}\widehat A_{f_k}\odot\widetilde z_m^{(l,k)}.
\tag{11}
\]

At each scale, the several periodic views are combined using FFT-amplitude-derived weights. The source denotes this new scale-wise tensor with the same x-like notation as earlier; here we use \(\overline x\) to avoid a self-referential ambiguity.

## 33. MRM is not necessarily learned routing

[CRITIQUE]

The score for periodic resolution \(k\) comes from spectral amplitude, not from a separately described supervised MoE router. It is **data-dependent adaptive fusion**, but should not be called:

- a learned expert gate conditioned on RankIC;
- a validated regime classifier;
- sparse top-k expert routing;
- a probabilistic uncertainty distribution over frequencies.

[EXTENSION] In finance, a reliability-aware resolution selector could use spectral peak confidence or validation-learned weights rather than amplitude alone.

## 34. Output projections: task-conditional heads, Eq. (3)

[SOURCE; §3.1, p.5]

\[
\operatorname{output}
=\operatorname{Ensemble}
\big(\{\operatorname{Head}_m(x_m^L)\}_{m=0}^{M}\big).
\tag{3}
\]

- Separate head per scale.
- Ensemble may take **average or weighted sum**, depending on task/implementation.
- Heads are typically linear in the generic description.
- Do not assume the same forecast-output head is used identically for imputation, classification, and anomaly detection.

## 35. End-to-end formula and axis trace

[DERIVATION]

\[
\begin{aligned}
X_{\mathrm{init}} &= \operatorname{MultiScaleConv}(x),\\
X^0 &=\operatorname{Embed}(\operatorname{ChannelMix}(X_{\mathrm{init}})),\\
X^{l+1} &=\operatorname{LN}\big(X^l+\operatorname{MRM}(\operatorname{MCM}(\operatorname{TID}(\operatorname{MRTI}(X^l))))\big),\\
\hat y &=\operatorname{Ensemble}_m(\operatorname{Head}_m(x_m^L)).
\end{aligned}
\]

This is a compact conceptual composition. A production implementation additionally needs normalization ordering, correctly broadcast FFT weights, padding/cropping, attention heads, task-specific output shapes, and training batch semantics.

## 36. Minimal typed tensor-interface chart

```text
Input x:                 [B, T, C]
Multiscale x[m]:         [B, floor(T/2^m), C]
Embedding x_l[m]:        [B, floor(T/2^m), D]
FFT peaks:               K indices / periods from coarsest scale
Time image z[m,k]:       [B, period_k, ceil(T_m/period_k), D]
Season/trend image:      same shape as z[m,k]
Mixed image:             shape-aligned residual at each scale
Flattened view:          [B, pad_len(T_m,period_k), D]
Cropped 1D view:         [B, T_m, D] (implementation should verify)
MRM output per scale:    [B, T_m, D]
Scale-specific head:     task-dependent
Ensemble output:         task-dependent
```

[VERIFY] The paper does not print an explicit batch-dimension tensor table; this interface reconstructs batch dimensions and common deep-learning conventions.

## 37. Figure 5–9 visual evidence map

[SOURCE; Appendix B, printed pp.17–19]

- **Figure 5:** Channel attention across variables at the coarse temporal scale, before embedding.
- **Figure 6:** Top-3 FFT frequencies from global context, yielding a matrix of 2D time images over multiple scales and resolutions.
- **Figure 7:** Dual-axis attention on 2D images; separate seasonal and trend branches.
- **Figure 8:** 2D conv fine→coarse for seasonal branch, transpose-conv coarse→fine for trend branch.
- **Figure 9:** Period-resolution fusion inside each scale, followed by scale-specific output.

The architecture figures are explanatory; exact convolution hyperparameters are not provided by the diagrams.

## 38. Which mechanics are inherited vs. genuinely added

| Component | Already in TimeMixer 2024? | TimeMixer++ 2025 adaptation |
|---|---|---|
| Multi-scale time inputs | Yes | Learned stride-2 Conv |
| Fine→coarse seasonal mixing | Yes | Operates on 2D images with convolution |
| Coarse→fine trend mixing | Yes | 2D transposed Conv |
| Decomposition | Yes | Deep axis-specific attention instead of shallow moving average |
| Multi-scale forecast ensemble | Yes | Generalizes to task-conditioned scale heads |
| Fourier top-K periodicity | No core mechanism | New MRTI/MRM resolution axis |
| 2D period-based time images | No | New MRTI architecture |
| Inter-variate attention | Not central in 2024 TimeMixer | Channel attention at coarsest scale |
| General eight-task evaluation | No, mainly forecasting | Major expansion of task scope |

[CRITIQUE] The use of FFT/time-to-2D imaging has precedents such as **TimesNet (ICLR 2023)**; the novel contribution is the particular **joint combination** of multiscale and multi-resolution 2D attention/mixing, rather than inventing 2D time-series imaging from nothing.

## 39. Comparison to TimesNet and PatchTST

- **TimesNet:** Uses periods inferred in frequency space to rearrange temporal information into 2D patterns; TimeMixer++ adds a **sampling-scale pyramid**, dual-axis trend/season decomposition, and hierarchical mixing.
- **PatchTST:** Uses fixed temporal patch tokens and channel independence in its canonical form; TimeMixer++ uses FFT-selected periodic views and coarse-scale channel attention.
- **TimeMixer 2024:** Avoids high-cost attention, uses all-MLP temporal scale mixing; TimeMixer++ **reintroduces attention** to learn a richer pattern space.

[CRITIQUE] A cross-paper model improvement may combine several architectural differences; do not assign all gains to FFT, channels, or attention without matched ablations.

## 40. Efficiency is not the same as the 2024 lightweight-MLP story

[SOURCE; Appendix E, Fig.14] The authors report an **efficiency/performance trade-off**, not a simple guarantee that TimeMixer++ is universally smaller and faster than all compared models.

[DERIVATION] Extra FFT selections, \((M+1)K\) image representations, 2D convolution, and dual-axis attention can raise memory use relative to TimeMixer. Performance gains should be benchmarked under a matched GPU and lookback length.


---

# Part II — Original experimental design and numerical evidence

## 41. Evaluation protocol: no financial ranking metrics

[SOURCE; §4, Appendix A/H]

Different tasks use different output metrics:

| Task | Main metric(s) | Direction |
|---|---|---|
| Long-term forecasting | MSE, MAE | lower better |
| Univariate short-term M4 | sMAPE, MASE, OWA | lower better |
| Multivariate short-term PeMS | MAE, MAPE, RMSE | lower better |
| Imputation | MSE, MAE | lower better |
| Time-series classification | Accuracy | higher better |
| Anomaly detection | Precision / Recall / F1 | higher better |
| Few-shot forecasting | MSE, MAE | lower better |
| Zero-shot cross-dataset forecasting | MSE, MAE | lower better |

[CRITIQUE] The main objective in long-term forecasting is **numeric sequence error**, not cross-sectional Spearman RankIC. Migration to stock ranking must be evaluated independently; no source result supports portfolio return or Sharpe improvement.

## 42. Overall dataset collection

[SOURCE; §4, Appendix A, Tables 8–10]

Long-term data include **ETTh1, ETTh2, ETTm1, ETTm2, Electricity, Traffic, Weather, Solar-Energy, and Exchange**. Note that the paper describes “8 datasets” in §4.1.1 while its main **Table 1** additionally includes Exchange, which produces nine named series groups if ETT four subsets are counted separately. Short-term tasks use **M4** and **PeMS03/04/07/08**. Additional tasks draw on ETT/Electricity/Weather, UEA multivariate classification series, and SMD/MSL/SMAP/SWaT/PSM anomaly series.

[VERIFY] Do not hardcode an exact universe count from the abstract's “30 benchmarks”; benchmark-counting conventions vary with task/dataset variants.

## 43. Long-term dataset detail from Appendix Table 8

| Dataset | Channels | Train / validation / test samples | Frequency | Forecastability reported |
|---|---:|---|---|---:|
| ETTh1 | 7 | 8545 / 2881 / 2881 | 15 min (as printed) | 0.38 |
| ETTh2 | 7 | 8545 / 2881 / 2881 | 15 min (as printed) | 0.45 |
| ETTm1 | 7 | 34465 / 11521 / 11521 | 15 min | 0.46 |
| ETTm2 | 7 | 34465 / 11521 / 11521 | 15 min | 0.55 |
| Electricity | 321 | 18317 / 2633 / 5261 | Hourly | 0.77 |
| Traffic | 862 | 12185 / 1757 / 3509 | Hourly | 0.68 |
| Weather | 21 | 36792 / 5271 / 10540 | 10 min | 0.75 |
| Solar-Energy | 137 | 36601 / 5161 / 10417 | 10 min | 0.33 |
| Exchange | 8 | 5120 / 665 / 1422 | Daily | 0.41 |

[SOURCE] The paper defines “forecastability” using one minus the entropy of a Fourier-domain decomposition; higher generally means easier forecasting under this definition.

[VERIFY] Appendix Table 8 prints `15min` for **ETTh** as well as ETTm; conventional ETTh is hourly. The supplied paper's printed value should be preserved as a source-reported discrepancy rather than silently amended. Similarly, Appendix Table 8 lists `Exchange` information category as “Weather,” apparently a copy/editing error. Verify dataset loaders and repository metadata for actual sampling frequency and category.

## 44. Original problem is usually much longer than our Qlib lookback

[SOURCE; §4/Appendix A]

- Standard long-term benchmark lookback in the main original TimeMixer 2024 paper is 96 observations.
- TimeMixer++ examines additional input lengths in sensitivity (§F); common prediction lengths are 96, 192, 336, 720.
- PeMS short-term forecast horizon is 12 steps with input length 96.
- Imputation examples use sequences of length 1024 and several masking ratios.

[CRITIQUE] These time scales are **not equivalent** to stock lookbacks \(T=8\) or \(T=20\). FFT top-frequency selection in an 8-step window can be highly unstable, and stride-2 hierarchies can leave too few observations at coarse levels.

## 45. Main baseline families

[SOURCE; Appendix A]

- TimeMixer (2024) as direct predecessor.
- iTransformer, PatchTST, Crossformer, FEDformer, Non-stationary Transformer, Autoformer, Informer.
- TimesNet, MICN, SCINet, TCN-family.
- TiDE, DLinear, LightTS, and additional MLP/linear approaches.
- N-HiTS, N-BEATS for M4 forecasting.
- Anomaly Transformer and related detection models.
- Rocket, DTW, XGBoost, LSSL, LSTNet etc. for classification.
- LLMTime, ETSformer, Reformer for few-shot/zero-shot comparators.

Authors describe **27 baselines** across tasks; no single table uses all 27 simultaneously. Match a task's actual comparator set rather than assuming “one unified 27-model run.”

## 46. Baseline fairness: reused versus rerun values

[SOURCE; Appendix A, p.16]

The authors state they **reuse results from TimesNet** when the baseline's experimental settings agree, and **reproduce using Time-Series Library** for settings that differ or tasks not previously implemented. Thus the paper does not claim all 27 baselines were freshly run through one entirely identical experiment script in every table.

[CRITIQUE] In a cross-model meta-analysis, separate original-source numbers, benchmark-reused values, and directly rerun values where provenance can be established.

## 47. General experimental implementation

[SOURCE; Appendix A, p.16]

| Item | Reported information |
|---|---|
| Framework | PyTorch |
| GPUs | Multiple NVIDIA A100 80GB |
| Repetitions | 3 experiments / runs |
| Optimizer | Adam |
| Learning rate | Search / choices from \(10^{-3}\) to \(10^{-1}\) |
| Typical batch size | 512 |
| Number of period resolutions \(K\) | 1–5 |
| MixerBlocks \(L\) | 1–3 |
| Long-series downsample stages \(M\) | Usually 3 |
| Short-series downsample stages \(M\) | Usually 1 |
| Optimization | Forecasting uses L2/MSE; task losses follow respective benchmark protocols |
| Schedule | Learning-rate decay after linear warmup |
| Repository | `https://github.com/kwuking/TimeMixer` |

[VERIFY] This appendix does not give a complete per-dataset matrix of learning rates, stopping epoch, patience, heads, convolution kernel size, FFT padding masks and all seeds. It also says “pretrained the model” but the mechanism/meaning of this phrase for each experiment is **not** a clearly established universal cross-dataset pretraining procedure. Check code before asserting that every experiment used a pretrained checkpoint.

## 48. Task-specific training is not stock-market pretraining

[SOURCE/CRITIQUE]

- Few-shot: training with only 10% of time steps from the target dataset.
- Zero-shot: train on source dataset \(D_a\), evaluate on distinct target \(D_b\) **without target training**.
- Classification/imputation/anomaly are separate task formulations with different output heads and losses.

A research Agent must not conflate **zero-shot transfer across ETT series** with **zero-shot transfer from CSI300 to S&P500**. Financial market transfer is an untested hypothesis in this paper.

## 49. Main long-horizon results: Table 1, MSE/MAE

[SOURCE; Table 1, p.7; averages over forecast lengths 96,192,336,720]

| Dataset | TimeMixer++ MSE | TimeMixer++ MAE | TimeMixer MSE | TimeMixer MAE | PatchTST MSE | PatchTST MAE | iTransformer MSE | iTransformer MAE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Electricity | 0.165 | 0.253 | 0.182 | 0.272 | 0.205 | 0.290 | 0.178 | 0.270 |
| ETT (Avg, *as printed*) | 0.349 | **0.399** | 0.367 | 0.388 | 0.381 | 0.397 | 0.383 | **0.377** |
| Exchange | 0.357 | 0.391 | 0.391 | 0.453 | 0.403 | 0.404 | 0.378 | **0.360** |
| Traffic | 0.416 | 0.264 | 0.484 | 0.297 | 0.481 | 0.304 | 0.428 | 0.282 |
| Weather | 0.226 | 0.262 | 0.240 | 0.271 | 0.259 | 0.281 | 0.258 | 0.278 |
| Solar-Energy | 0.203 | 0.238 | 0.216 | 0.280 | 0.270 | 0.307 | 0.233 | 0.262 |

[VERIFY] The `ETT (Avg)` MAE entries in Table 1 do **not** reconcile with the four per-ETT dataset rows in Appendix Table 16. Preserve these numbers as *as printed*, but use the complete per-dataset appendix in exact comparisons (see §51 below).

## 50. Why Table 1 is impressive but not an all-metric clean sweep

[CRITIQUE]

- Table 1 contains many strong wins on MSE; e.g., Solar-Energy improves from TimeMixer's 0.216 to 0.203, and Electricity from 0.182 to 0.165.
- **Exchange MAE:** iTransformer reports 0.360, better than TimeMixer++ 0.391.
- **ETT Avg MAE (as printed):** 0.399 for TimeMixer++ versus 0.388 TimeMixer, although Table 16 conflicts with this summary.
- Even if a paper reports “SOTA across tasks,” it does **not** prove first place on every individual metric, horizon, or dataset.

## 51. IMPORTANT: ETT average MAE inconsistency (Table 1 vs Table 16)

[VERIFY; **critical numerical discrepancy**]

Appendix **Table 16** gives TimeMixer++ ETT MAEs:

| Dataset | TimeMixer++ MAE | iTransformer MAE | TimeMixer MAE |
|---|---:|---:|---:|
| ETTh1 | 0.432 | 0.447 | 0.440 |
| ETTh2 | 0.380 | 0.407 | 0.395 |
| ETTm1 | 0.378 | 0.410 | 0.395 |
| ETTm2 | 0.320 | 0.332 | 0.323 |
| **Arithmetic mean of these four rows** | **0.3775** | **0.3990** | **0.38825** |

Yet Table 1 prints TimeMixer++ **0.399** and iTransformer **0.377** for ETT(Avg) MAE. These two numbers closely match the **opposite models'** arithmetic averages in Table 16. The most parsimonious explanation is a transposition/error in Table 1, but this remains a **verification hypothesis** until confirmed against authors' corrections/code. Do not silently alter original Table 1 in bibliographic metadata or claim the inconsistency is definitively explained.

## 52. Long-term per-horizon TimeMixer++ results (Table 16)

[SOURCE; Appendix H, Table 16]

| Dataset | 96 MSE/MAE | 192 MSE/MAE | 336 MSE/MAE | 720 MSE/MAE | Avg MSE/MAE |
|---|---|---|---|---|---|
| ETTh1 | .361/.403 | .416/.441 | .430/.434 | .467/.451 | .419/.432 |
| ETTh2 | .276/.328 | .342/.379 | .346/.398 | .392/.415 | .339/.380 |
| ETTm1 | .310/.334 | .348/.362 | .376/.391 | .440/.423 | .369/.378 |
| ETTm2 | .170/.245 | .229/.291 | .303/.343 | .373/.399 | .269/.320 |
| Weather | .155/.205 | .201/.245 | .237/.265 | .312/.334 | .226/.262 |
| Solar-Energy | .171/.231 | .218/.263 | .212/.269 | .212/.270 | .203/.238 |
| Electricity | .135/.222 | .147/.235 | .164/.245 | .212/.310 | .165/.253 |
| Traffic | .392/.253 | .402/.258 | .428/.263 | .441/.282 | .416/.264 |
| Exchange | .085/.214 | .175/.313 | .316/.420 | .851/.689 | .357/.391 |

[CRITIQUE] This condensed table retains TimeMixer++ predictions at all four key horizons; the 12-model wide original Table 16 is not reproduced here in full. When interpreting a specific baseline-vs-TimeMixer++ comparison, consult original **Table 16** rather than infer it from dataset averages.

## 53. Long-term horizon dependence and forecasting challenge

[CRITIQUE]

Prediction errors generally rise with horizon, but not always monotonically for each dataset. Increasing forecast horizon simultaneously changes the target, relevant periodic components, uncertainty, and model-selection landscape. Do not infer that a positive result at 96-step Weather predicts gains on 5-day stock returns.

## 54. Short-term multivariate PeMS results: Tables 3 and 18

[SOURCE; §4.1.3 and Appendix H, Table 18]

| Dataset | TimeMixer++ MAE / MAPE / RMSE | TimeMixer MAE / MAPE / RMSE | PatchTST MAE / MAPE / RMSE |
|---|---|---|---|
| PEMS03 | 13.99 / 13.43 / 24.03 | 14.63 / **11.54*** / 23.28 | 18.95 / 17.29 / 30.15 |
| PEMS04 | 17.46 / 11.34 / 28.83 | 19.21 / 12.53 / 30.92 | 24.86 / 16.65 / 40.46 |
| PEMS07 | 18.38 / 7.32 / 31.75 | 20.57 / 8.62 / 33.59 | 27.87 / 12.69 / 42.56 |
| PEMS08 | 13.81 / 8.21 / 23.62 | 15.22 / 9.67 / 24.26 | 20.35 / 13.15 / 31.04 |
| **Reported average** | **15.91 / 10.08 / 27.06** | **17.41 / 10.59 / 28.01** | **23.01 / 14.95 / 36.05** |

[VERIFY] `PEMS03 TimeMixer MAPE = 11.54` is **as printed in Table 18**, but the original TimeMixer 2024 reports **14.54** in its PeMS results, and TimeMixer++ Appendix Table 14 also prints **14.54**. The 11.54 in Table 18 appears inconsistent with the corresponding average and earlier paper. Store both and verify; do not overwrite the paper unilaterally.

[CRITIQUE] Even ignoring that MAPE inconsistency, **PEMS03 RMSE** in TimeMixer++ is **24.03**, worse than TimeMixer's **23.28**. Therefore the strongest accurate conclusion is improved average performance and many individual wins, **not** dominance on every PeMS metric.

## 55. PeMS task structure and financial analogy

[SOURCE] PeMS contains road-sensor traffic time series with 170–883 variables; it is a **multivariate forecasting** benchmark with strong inter-variable dependence.

[EXTENSION] This has a partial analogy to joint forecasting of many stock features or sector-level time series. But sensor channels generally have persistent geographic identities whereas changing stock universes introduce survivorship, missingness, and membership masks. The analogy is method-level, not protocol-equivalent.

## 56. M4 short-term univariate summary: Table 2

[SOURCE; p.7]

| Method | sMAPE | MASE | OWA |
|---|---:|---:|---:|
| **TimeMixer++** | **11.448** | **1.487** | **0.821** |
| TimeMixer | 11.723 | 1.559 | 0.840 |
| TimesNet | 11.829 | 1.585 | 0.851 |
| N-HiTS | 11.927 | 1.613 | 0.861 |
| N-BEATS (paper protocol) | 11.851 | 1.559 | 0.855 |
| PatchTST | 13.152 | 1.945 | 0.998 |

[CRITIQUE] Strong average performance on M4 does not imply its multiscale frequency/image operations should improve cross-sectional equity ranking. M4 is an univariate ungrouped task with different data-generating mechanisms.

## 57. M4 subset detail and important counterexamples

[SOURCE; Appendix H Table 17]

| M4 group | TimeMixer++ sMAPE | TimeMixer sMAPE | TimeMixer++ MASE | TimeMixer MASE | TimeMixer++ OWA | TimeMixer OWA |
|---|---:|---:|---:|---:|---:|---:|
| Yearly | 13.179 | 13.206 | **2.934** | **2.916** | 0.769 | 0.776 |
| Quarterly | 9.755 | 9.996 | 1.159 | 1.166 | **0.865** | **0.825** |
| Monthly | 12.432 | 12.605 | 0.904 | 0.919 | 0.841 | 0.869 |
| Others | **4.698** | **4.564** | 2.931 | 3.115 | **1.010** | **0.982** |
| Weighted average | 11.448 | 11.723 | 1.487 | 1.559 | 0.821 | 0.840 |

[CRITIQUE] TimeMixer 2024 is better in **Yearly MASE, Quarterly OWA, Others sMAPE and OWA**, so the 2025 model is not superior for all frequency groups or metrics.

## 58. Imputation protocol

[SOURCE; §4.1.4, p.8]

- Time-series length: **1024**.
- Randomly mask time points at proportions **12.5%, 25%, 37.5%, 50%**.
- Results are reported as average over four masking ratios.
- ETT (four variants summarized), Electricity/ECL, Weather.
- Metrics: MSE and MAE.

This is not missing-stock-ID robustness; the data task is reconstruction of masked temporal values.

## 59. Imputation major results: Table 4

[SOURCE; p.8]

| Dataset | TimeMixer++ MSE | TimeMixer++ MAE | TimeMixer MSE | TimeMixer MAE | Other standout comparator |
|---|---:|---:|---:|---:|---|
| ETT (Avg) | 0.055 | 0.154 | 0.097 | 0.220 | TimesNet 0.079/0.182 |
| Electricity/ECL | **0.109** | 0.197 | 0.142 | 0.261 | **DLinear MSE 0.080 (better)** |
| Weather | 0.049 | 0.078 | 0.091 | 0.114 | TimesNet 0.061/0.098 |

[CRITIQUE] The authors' broad “consistently best” language is not literally true on Electricity MSE: **0.080 for DLinear** beats **0.109** for TimeMixer++. This counterexample should not be omitted by a research Agent.

## 60. Missingness assumptions and distribution shift

[CRITIQUE]

Random time-point masking is a controlled imputation benchmark. Financial production missingness often correlates with:

- listing/delisting events;
- trading suspension;
- feature vendor gaps;
- calendar/exchange differences;
- genuine zero volume vs. absent observations.

A useful stock-specific imputation study would include structured/missing-not-at-random missingness rather than treat these paper results as evidence for every financial missing-data regime.

## 61. Few-shot forecasting: what “10%” means

[SOURCE; §4.1.5]

The paper trains on **10% of available time steps** for certain datasets and evaluates forecasting error under limited in-domain training data. This is **not** few-shot in the language-prompting sense; it is a reduced training-data setting.

## 62. Few-shot selected comparison: Table 5

[SOURCE; p.8]

| Dataset | TimeMixer++ MSE/MAE | TimeMixer MSE/MAE | PatchTST MSE/MAE | DLinear MSE/MAE |
|---|---|---|---|---|
| ETT (Avg) | 0.396/0.421 | 0.453/0.445 | 0.461/0.446 | 0.506/0.484 |
| Weather | 0.241/0.271 | 0.242/0.281 | 0.242/0.279 | 0.241/0.283 |
| ECL | 0.168/0.271 | 0.187/0.277 | 0.180/0.273 | 0.180/0.280 |

[CRITIQUE] On Weather, TimeMixer++ and DLinear have identical **0.241 MSE**, even though TimeMixer++ has lower MAE. The gain depends on the metric and dataset.

## 63. Zero-shot transfer protocol

[SOURCE; §4.1.6, p.9]

Train on one ETT dataset and evaluate **without additional training** on a different ETT dataset. The reported values average prediction horizons 96,192,336,720. The source/target pairs often differ in sampling intervals or dataset conditions but remain within the ETT family.

[CRITIQUE] This is a meaningful **cross-ETT** transfer test, not evidence for cross-asset-class or cross-country financial generalization.

## 64. Zero-shot full six-pair TimeMixer++ vs relevant comparators

[SOURCE; Table 6]

| Source → Target | TimeMixer++ MSE/MAE | TimeMixer MSE/MAE | PatchTST MSE/MAE | iTransformer MSE/MAE |
|---|---|---|---|---|
| ETTh1 → ETTh2 | .367/.391 | .427/.424 | .380/.405 | .481/.474 |
| ETTh1 → ETTm2 | .301/.357 | .361/.397 | .314/.360 | .311/.361 |
| ETTh2 → ETTh1 | .511/.498 | .679/.577 | .565/.513 | .552/.511 |
| ETTm1 → ETTh2 | .417/.422 | .452/.441 | .439/.438 | .434/.438 |
| ETTm1 → ETTm2 | .291/.331 | .329/.357 | .296/.334 | .324/.331 |
| ETTm2 → ETTm1 | .427/.448 | .554/.478 | .568/.492 | .559/.491 |

[CRITIQUE] The result is strong for this evaluation set; it does **not** establish transferable latent factor semantics or real-market invariance.

## 65. Classification protocol

[SOURCE; §4.1.7, Appendix Tables 9 and 20]

The UEA time-series classification evaluation uses **10 named multivariate datasets**:

1. EthanolConcentration
2. FaceDetection
3. Handwriting
4. Heartbeat
5. JapaneseVowels
6. PEMS-SF
7. SelfRegulationSCP1
8. SelfRegulationSCP2
9. SpokenArabicDigits
10. UWaveGestureLibrary

These are classification datasets, not long-term forecasting samples.

## 66. Classification main numbers and table/text discrepancy

[SOURCE; Appendix H, Table 20]

| Dataset | TimeMixer++ accuracy (%) | TimesNet accuracy (%) | Selected stronger comparator when applicable |
|---|---:|---:|---|
| EthanolConcentration | 39.9 | 35.7 | Rocket 45.2 |
| FaceDetection | 71.8 | 68.6 | TimeMixer++ higher than listed entries |
| Handwriting | 26.5 | 32.1 | Rocket 58.8; TCN 53.3 |
| Heartbeat | 79.1 | 78.0 | Reformer 80.5 |
| JapaneseVowels | 97.9 | 98.4 | Non-stationary Transformer reports 99.2; several others 98.9 |
| PEMS-SF | 91.0 | 89.6 | XGBoost 98.3 |
| SelfRegulationSCP1 | 93.1 | 91.8 | TimeMixer++ high |
| SelfRegulationSCP2 | 65.6 | 57.2 | TimeMixer++ high |
| SpokenArabicDigits | 99.8 | 99.0 | Several methods 100.0 |
| UWaveGestureLibrary | 88.2 | 85.3 | Rocket 94.4 |
| **Arithmetic average as reported Table 20** | **75.3** | **73.6** | Not best on every dataset |

[VERIFY] Main text §4.1.7 says “**75.9% accuracy**,” while Appendix **Table 20** gives **75.3%** and its ten rows average approximately **75.29%**. The table and arithmetic support 75.3 as its tabulated evaluation, but retain 75.9 as an explicit textual discrepancy.

[CRITIQUE] TimeMixer++ is not the top method on several individual UEA datasets (including Handwriting, JapaneseVowels, and PEMS-SF). This is an aggregate-performance claim, not universal class-wise dominance.

## 67. Anomaly detection protocol

[SOURCE; §4.1.7; Appendix A/H]

Five multivariate datasets:

- SMD
- MSL
- SMAP
- SWaT
- PSM

Reported evaluation includes precision, recall and F1, with overall mean F1 used for comparison.

[VERIFY] The paper does not fully describe threshold calibration or possible point-adjustment protocols in the main method. Exact anomaly scoring and adjustment conventions matter substantially; check benchmark code before comparing numbers to a new system.

## 68. Anomaly detection source values, Table 19

[SOURCE; Appendix H, Table 19]

| Dataset | TimeMixer++ precision | TimeMixer++ recall | TimeMixer++ F1 | TimesNet F1 |
|---|---:|---:|---:|---:|
| SMD | 88.59 | 84.50 | 86.50 | 85.81 |
| MSL | 89.73 | 82.23 | 85.82 | 85.15 |
| SMAP | 93.47 | 60.02 | 73.10 | 71.52 |
| SWaT | 92.96 | 94.33 | 94.64 | 91.74 |
| PSM | 98.33 | 96.90 | 97.60 | 97.47 |
| **Mean F1 (%)** | — | — | **87.47** | **86.34** |

[CRITIQUE] Paper narrative says “exceeds TimesNet by 2.59%,” but 87.47−86.34=**1.13 percentage points**, approximately **1.31% relative**. The narrative comparison is not numerically supported by this Table 19 mean. Preserve the raw values, and avoid repeating a percentage without checking its intended comparator/definition.

## 69. Representation CKA analysis

[SOURCE; Fig.1, Fig.12, Appendix D]

CKA is computed between first- and last-layer representations in several task settings. The authors plot CKA against a task metric and describe pattern consistency/diversity as a sign of adaptation.

[VERIFY] The **introduction's verbal description** associates some tasks with lower CKA and others with higher CKA, while the later representation discussion (p.10) uses somewhat inconsistent task pairings. It is safer to record the plotted values/correlations and actual task outcomes than to assert a universal causal law “high CKA is always good for task X.”

## 70. Main long-term ablation: Table 7

[SOURCE; p.10; eight datasets]

| Variant | ETTh1 | ETTh2 | ETTm1 | ETTm2 | ECL | Traffic | Weather | Solar | Printed average MSE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Full** | .419 | .339 | .369 | .269 | .165 | .416 | .226 | .203 | **.300** |
| w/o channel mixing | .424 | .346 | .374 | .271 | .197 | .442 | .233 | .245 | .317 |
| w/o time image decomposition | .441 | .358 | .409 | .291 | .198 | .445 | .251 | .241 | .329 |
| w/o multi-scale mixing | .447 | .361 | .391 | .284 | .172 | .427 | .239 | .234 | .320 |
| w/o multi-resolution mixing | .431 | .350 | .374 | .280 | .181 | .432 | .241 | .233 | .316 |

**Paper's reported average promotions**: channel mixing 5.36%; TID 8.81%; MCM 6.25%; MRM 5.10%. These numbers are not guaranteed to be additive or transferable across tasks, because ablations are separately evaluated models and the components interact.

## 71. Which ablation is largest for long forecasting?

[SOURCE/CRITIQUE]

For the 8-dataset average MSE in Table 7:

- Removing **TID** increases mean MSE from 0.300 to 0.329, the largest listed loss.
- Removing **MCM** increases to 0.320.
- Removing channel mixing increases to 0.317.
- Removing MRM increases to 0.316.

**Evidence-specific reading:** latent-space trend/season decomposition is the highest-impact component in this table, **not** automatically the globally most important model mechanism under every possible financial task.

## 72. Short-term component ablation, Appendix Table 11

[SOURCE; Appendix C]

| Variant | M4 sMAPE | PEMS03 MAPE | PEMS04 MAPE | PEMS07 MAPE | PEMS08 MAPE | Printed average |
|---|---:|---:|---:|---:|---:|---:|
| Full | 11.45 | 13.43 | 11.34 | 7.32 | 8.21 | 10.35 |
| w/o channel mixing | 11.44 | 15.57 | 13.31 | 9.74 | 10.78 | 12.17 |
| w/o TID | 12.37 | 15.59 | 12.97 | 9.65 | 9.97 | 12.11 |
| w/o MCM | 11.98 | 14.97 | 13.02 | 9.17 | 9.69 | 11.79 |
| w/o MRM | 11.87 | 15.02 | 13.14 | 8.72 | 9.53 | 11.68 |

[SOURCE] Appendix C concludes channel mixing's value is especially pronounced in multivariate PeMS, with printed average improvement **14.95%** in its ablation metric summary.

[CRITIQUE] Removing channel mixing slightly **improves** M4 sMAPE (11.45 → 11.44). This is consistent with its purpose: M4 is univariate, and the channel mixer is not invariably useful.

## 73. Anomaly-detection component ablation, Appendix Table 12

[SOURCE; Appendix C]

| Variant | SMD F1 | MSL F1 | SMAP F1 | SWaT F1 | PSM F1 | Avg F1 |
|---|---:|---:|---:|---:|---:|---:|
| **Full** | 86.50 | 85.82 | 73.10 | 94.64 | 97.60 | **87.47** |
| w/o channel mixing | 84.51 | 74.03 | 70.91 | 90.41 | 96.17 | 83.21 |
| w/o TID | 81.21 | 72.43 | 66.02 | 82.41 | 92.53 | 78.92 |
| w/o MCM | 82.37 | 75.12 | **92.79** | 86.48 | 94.53 | 86.26 |
| w/o MRM | 83.37 | 79.24 | **77.49** | 88.46 | 96.02 | 86.26 |

[VERIFY] The `SMAP` value of **92.79** in the “w/o MCM” row is far higher than the full model's **73.10** while the average is still lower overall. This could be a genuine task-specific trade-off or a table error. Preserve it instead of silently replacing it. The 'every component helps every dataset' narrative is not supported literally by this table.

## 74. Ablation mechanism claims: what is and is not isolated

- `w/o channel mixing` tests the impact of the coarse cross-variable attention branch in the authors' implementation.
- `w/o TID` removes the deep axis-based decomposition module.
- `w/o MCM` removes cross-scale exchange.
- `w/o MRM` removes adaptive resolution fusion.
- The baseline-versus-full model comparison simultaneously changes multiple mechanisms and is **not** a single-module ablation.
- Statistical tests for overall model comparisons do not automatically establish the significance of each ablation gain.

## 75. Hyperparameter sensitivity: Appendix F

[SOURCE; Appendix F, Fig.15]

Parameters studied:

- number of scales \(M\);
- number of stacked MixerBlocks \(L\);
- input/history length \(T\);
- number of selected top periods \(K\).

Reported trends:

- Larger \(M\) often helps, but marginal benefit declines; practical range roughly 1–3.
- Depth gains are stronger early, especially from 1 to 2 layers; deeper is not necessarily universally better.
- Larger top-K can help, especially for longer forecast horizons.
- Longer input history often lowers MSE on the tested ETTm1 curves.

[CRITIQUE] This does **not** imply maximizing \(M\) or \(K\) on short stock lookbacks. With \(T=8\), \(M=3\) yields only one coarse observation, undermining meaningful Fourier period selection.

## 76. Appendix G: three-repeat uncertainty

[SOURCE; Tables 13–15]

The paper reports mean ± standard deviation for several models and describes confidence comparisons. Selected examples:

| Case | TimeMixer++ | Comparator | Reported confidence |
|---|---|---|---|
| Weather long MSE | 0.226 ± 0.008 | iTransformer 0.258 ± 0.009 | 99% |
| Solar-Energy long MSE | 0.203 ± 0.027 | iTransformer 0.233 ± 0.009 | 95% |
| Electricity long MSE | 0.165 ± 0.017 | iTransformer 0.178 ± 0.002 | 99% |
| PEMS03 MAE | 13.99 ± 0.271 | TimeMixer 14.63 ± 0.471 | 99% |
| PEMS04 MAE | 17.46 ± 0.951 | TimeMixer 19.21 ± 0.511 | 95% |
| M4 average sMAPE | 11.448 ± 0.007 | TimeMixer 11.723 ± 0.011 | 99% |

[CRITIQUE] Only **three independent runs** are described; the precise statistical-test design (pairing, independent units, autocorrelation adjustment, multiple comparisons) is not completely established by the supplied text. Do not infer a strong causal p-value framework solely from `95%/99%` confidence labels.

## 77. Evaluation reporting contradictions ledger

[VERIFY] Most important paper-internal points to preserve:

1. **Main ETT mean MAE** from Table 1 conflicts with Table 16 per-dataset arithmetic. The 0.399 and 0.377 entries appear transposed between TimeMixer++ and iTransformer.
2. **Classification accuracy:** prose 75.9%, Appendix Table 20 average 75.3%.
3. **PEMS03 TimeMixer MAPE:** Table 18 11.54 vs Appendix G Table 14 14.54, and original TimeMixer 2024 reports 14.54.
4. **Anomaly improvement claim vs TimesNet:** prose 2.59%, Table 19 mean F1 87.47 vs 86.34 (1.13 pp).
5. **SMAP w/o MCM** in Appendix Table 12: 92.79 vs full 73.10, a striking exception to the all-parts-help story.
6. **Electricity imputation MSE:** DLinear 0.080, better than TimeMixer++ 0.109, despite broad narrative of superiority.
7. **ETTh sampling frequencies:** Appendix A lists `15 min` for ETTh; standard dataset convention is hourly; preserve as printed until verified.
8. **Figure/task CKA narrative** has inconsistent verbal task ordering; rely on actual plots and targeted comparisons.
9. **Aggregate vs individual SOTA:** some M4 subsets and classification datasets favor previous methods; avoid translating abstract-style summaries into universal claims.

Each of these warrants `[VERIFY]`, not ad hoc numerical correction.


---

# Part III — Inductive biases, failure modes, evidence audit, and research opportunities

## 78. Scientific contribution decomposition

[SOURCE/CRITIQUE]

**Established ingredients**:

- multi-scale downsampling;
- FFT period selection and 1D→2D time-series reshaping (cf. TimesNet);
- self-attention;
- convolution and transposed convolution;
- residual blocks and normalization;
- per-scale prediction heads;
- temporal decomposition;
- benchmarking on ETT/M4/PeMS/UEA/anomaly tasks.

**TimeMixer++'s distinct proposal**:

- combine **sampling-scale** and **period-resolution** axes in one pattern machine;
- extract seasonal/trend-like structure **within latent time images** using dual-axis attention;
- perform direction-specific **2D hierarchical mixing** across sampling scales;
- fuse multiple FFT-derived resolutions and scale-head outputs;
- validate across eight analytical task types rather than only forecasting.

This is a strong *system-level architectural innovation*. It is not historically accurate to attribute the origin of FFT period imaging or plain axial attention solely to TimeMixer++.

## 79. Inductive bias 1 — shared spectral skeleton from the coarsest scale

[DERIVATION]

The same selected \(p_1,\ldots,p_K\) defines time-image layouts at all scales. This encourages a consistent multiscale account of globally dominant periodic components.

**Potential advantage**: suppress arbitrary local frequency choices and share scale structure.

**Potential downside**: transient high-frequency patterns present only at fine scales may be missed if global spectral top-K is dominated by other energy components.

## 80. Inductive bias 2 — dual-axis factorization

[DERIVATION]

Instead of full joint attention over every pair of 2D time-image pixels, each branch attends mainly along one axis, giving a structured approximation to fully coupled time-period interaction.

**Potential advantage**: lower attention field and more interpretable module roles.

**Potential downside**: cross-axis interactions might require stacking blocks; some joint patterns may be difficult to represent with one axial sweep.

## 81. Inductive bias 3 — geometric locality

The model uses 2D convolutions on time images. A local kernel assumes nearby positions on the chosen period-derived image grid are particularly related. If the chosen period is wrong, image adjacency may be economically spurious.

## 82. Inductive bias 4 — asymmetric season/trend information flow

MCM makes two opposite-direction passes rather than using one symmetric exchange layer. This embeds the prior that local repeated details should aggregate upward while macro trend context should propagate downward.

**Important nuance**: TimeMixer++ does not prove there are precisely two economically unique independent processes; branches can remain statistically entangled.

## 83. Inductive bias 5 — coarse cross-variable interaction

Channel attention is applied on coarse-scale information. This assumes much of the valuable cross-variable interaction can be exposed in lower-resolution / longer-context representations. This may be beneficial for noisy short-term variables, but may underweight high-frequency cross-variable events.

## 84. Inductive bias 6 — amplitude-based resolution reliability

MRM uses FFT amplitude to weight resolutions. This encodes the hypothesis that high-energy periodic components are informative. In financial data, the largest spectral components need not produce the best forward signal. There is no task-supervised frequency selection theorem.

## 85. Inductive bias 7 — diverse scale-head predictions

Different time scales may be complementary and deserve separate heads before output fusion. This can counter the assumption that one pooled latent vector is sufficient for all targets.

[EXTENSION] A cross-sectional system could instead learn multiple scale-specific **ranking** heads, but then needs cross-sectional normalization, comparable score scale, and turnover-aware fusion.

## 86. Failure mode — short histories make the pyramid degenerate

[DERIVATION]

Consider stock lookbacks frequently used by earlier baselines:

```text
T=8:  M=0   8
      M=1   8 → 4
      M=2   8 → 4 → 2
      M=3   8 → 4 → 2 → 1  (coarsest FFT nearly meaningless)

T=20: M=0  20
      M=1  20 → 10
      M=2  20 → 10 → 5
      M=3  20 → 10 → 5 → 2 (few usable spectral bins)
```

[CRITIQUE] This is a decisive transfer constraint. The original long-series \(M=3\) should **not** be copied uncritically into an 8- or 20-day Qlib predictor.

## 87. Failure mode — pseudo-periods in nonstationary financial sequences

Financial signals often contain changing regimes, unbalanced trend components, structural gaps, and point events. The highest-amplitude FFT component of a short window may be a transient artifact. A variable length input or shift of a single event can change the selected top-K periods abruptly.

## 88. Failure mode — frequency aliasing across downsampling

Stride-2 convolution changes spectral content. If the downsampling filter does not sufficiently suppress high-frequency energy, aliases may create false periodic signals. The paper does not supply a theorem guaranteeing alias-free multi-scale imaging for arbitrary input processes.

[EXTENSION] Compare anti-aliased pooling and learned downsampling, with spectral-stability diagnostics under shift and injected noise.

## 89. Failure mode — spurious calendar seasonality labels

Even if a learned attention branch is called “seasonal,” one should not interpret its output as a known trading-day-of-week, month-end, or holiday effect without controlled alignment analysis.

## 90. Failure mode — frequency weights emphasize non-predictive energy

Let \(A_k\) be observed spectral amplitude and \(q_k\) be out-of-sample predictive value of that component. There is no necessary monotonic relationship between \(A_k\) and \(q_k\). MRM emphasizes \(A_k\). In finance, this can make the model focus on visually dominant but non-predictive cycles.

## 91. Failure mode — unstable representation near frequency-rank boundaries

Top-K selection is discrete. Two close spectral peak magnitudes can switch order as the window shifts. The learned 2D geometry can jump even when the original input changes slightly. An Agent should consider differentiable or smoothed peak selection as a **hypothesis**, not established improvement.

## 92. Failure mode — padded zeros and zero-valued market features

If padding is not masked, zeros may be treated like valid observations. In Alpha158, an actual zero value can be meaningful, and zero-filled missing data can have a different meaning. Padding masks and missing-data masks should remain distinct.

## 93. Failure mode — channel mixing with many features

A 158-feature interaction matrix is reasonable to explore, but some feature channels are near duplicates or deterministic transforms of related price-volume statistics. Mixing can amplify collinearity and redundant attention. Agent diagnostics should include effective rank, attention sparsity, and feature-family stability, not only mean IC.

## 94. Failure mode — time-series independence versus changing universes

The original multivariate benchmarks have fixed sensor/variable dimensions. A cross-stock variant faces daily varying \(N_t\), listing changes, suspensions, and index membership. If stocks are treated as channels, architecture must support masks/permutations and should not tie temporal convolution weights to a static company ordering.

## 95. Failure mode — complexity and limited finance sample length

The number of time images per block is \((M+1)K\); each carries an embedded-channel axis and attention/convolution. Unlike a single 1D MLP, this can raise both memory and optimization variance. Small daily cross-sectional datasets may overfit frequency patterns despite large numbers of stock-date observations that are **not independent**.

## 96. Failure mode — likelihood/regression objective does not align with stock ranking

The paper's forecasting experiments evaluate MSE/MAE. A high MSE model can still rank stocks well; conversely low MSE does not ensure RankIC or economic utility. A financial transfer should include losses and diagnostics appropriate to ranking.

## 97. Failure mode — other objectives have different heads

Classification, anomaly detection, and imputation are not achieved by one identical final linear layer without task design. When adapting TimeMixer++ in a modular Agent codebase, treat the generic representation encoder and the head/loss as separate concepts.

## 98. What the paper's evidence establishes strongly

[SOURCE/CRITIQUE]

- A very broad empirical study is reported across 8 task types.
- Multi-scale / multi-resolution architecture produces strong **aggregate** performance in the authors' test settings.
- Explicit removals of TID, MCM, MRM and channel mixing often degrade benchmark metrics.
- Reconstructed time images and attention images exhibit visually differentiated periodic and trend-like patterns (Figs.4,10,11,13).
- Three repeated experiments and confidence reporting offer some variance information.

## 99. What the evidence does not fully establish

- General optimality of the four-module architecture across all time series.
- Uniqueness of latent season/trend decomposition.
- Universal wins on every dataset/metric (several direct counterexamples exist).
- Transfer to stock ranking, risk factors, Qlib, or tradable portfolios.
- Robustness to financial structural breaks, publication delay, or constituent drift.
- Compute-matched and parameter-matched superiority for every comparator across all tasks.
- A verified scaling law for large time-series models.

## 100. Explicit author future work: Appendix K

[SOURCE; Appendix K, printed p.27]

The authors identify a field-level challenge: scaling laws and large data/parameter regimes for **time-series foundation models**. They propose pursuing **large-scale time-series datasets** and studying **scaling laws for TimeMixer++** / TSPMs.

This is the explicit future-work emphasis in the supplied 2025 version. It is **not** a paper-proposed adaptive VQ codebook, rank-loss design, or financial MoE router.

## 101. Explicitly described application reach vs validated experiments

[SOURCE; Appendix J]

The broader-impact text mentions **rapidly changing financial markets** as a potential application, alongside energy, weather, supply chains, etc.

[CRITIQUE] This is a statement of **potential**, not evidence from a stock-return experiment. The only finance-adjacent dataset in its long-term benchmark is the `Exchange` foreign exchange time series; that is not an S&P500 return ranking benchmark.

## 102. Source's claim that TID is first latent decomposition

[SOURCE; Appendix J]

Authors describe their deep-space dual-axis seasonal/trend decomposition as novel relative to shallow moving-average/FFT decomposition. A research Agent should attribute that **claim** to the authors and verify related work before repeating a strong global “first-ever” novelty statement in a new manuscript.

## 103. Technical constraints left to repository

[VERIFY] The supplied PDF does not fully determine:

- 2D convolution kernel sizes, channel widths and exact padding;
- spectral peak selection handling of DC/Nyquist bins;
- FFT normalization and gradients through frequency ranking;
- padding-mask/crop behavior in all scales;
- exact handling of very short sequences or \(K\) exceeding valid Fourier bins;
- task-specific head details and loss weightings for every benchmark;
- per-experiment early stopping selection and exact seeds;
- effective GPU memory scaling in stock-universe settings.

Use the official implementation before writing exact replication code. This document reconstructs the high-level mechanism but does **not** claim byte-for-byte code equivalence.

## 104. Algorithm — conceptual forward pass

[DERIVATION; pseudocode from Eqs.1–11]

```python
# Conceptual pseudocode, NOT copied from official code.
def timemixerpp_forward(x, task):
    # x: [B,T,C] -- observed historical window only
    scales = [x]
    for m in range(1, M + 1):
        scales.append(stride2_conv(scales[-1]))

    scales[-1] = channel_attention_at_coarsest_scale(scales[-1])
    X = [embed(s) for s in scales]

    for layer in range(L):
        peaks, amplitudes, periods = fft_topk_periods(X[-1], K)
        period_weights = softmax(amplitudes)

        multi_resolution = []
        for m, x_m in enumerate(X):
            views = []
            for k, period in enumerate(periods):
                im, padding_metadata = pad_and_reshape_to_time_image(x_m, period)
                season = column_axis_attention(conv2d_qkv(im))
                trend = row_axis_attention(conv2d_qkv(im))
                views.append((season, trend, padding_metadata))
            multi_resolution.append(views)

        # Each k: exchange scale information asymmetrically.
        for k in range(K):
            for m in range(1, M + 1):
                multi_resolution[m][k].season += downsample_2d(
                    multi_resolution[m - 1][k].season
                )
            for m in reversed(range(M)):
                multi_resolution[m][k].trend += upsample_2d(
                    multi_resolution[m + 1][k].trend
                )

        mixed = []
        for m in range(M + 1):
            reconstructed_views = []
            for k in range(K):
                y = multi_resolution[m][k].season + multi_resolution[m][k].trend
                reconstructed_views.append(reshape_and_crop(y))
            mixed.append(weighted_sum(reconstructed_views, period_weights))

        X = [layer_norm(old + new) for old, new in zip(X, mixed)]

    return task.ensemble([task.heads[m](X[m]) for m in range(M + 1)])
```

[VERIFY] This uses Python-like **mutable notation** for multi-resolution branch updates to explain flow. Exact module ordering within the source code, normalization, broadcasting, FFT index scaling, masking, and head semantics should be checked in the official repo.

## 105. Implementation interfaces for a stock-adaptation prototype

[EXTENSION; proposed interfaces, not original code]

```text
Stock temporal input:       x[date, stock, time, feature]
Flatten independent stocks: x_batch[N, T, C]
Market features:            market[T, M] (optional separate stream)
Multiscale feature pyramid: levels [N, T_m, D]
Time image:                 [N, period, repeats, D]
TimeMixer++ encoder output: per-stock multiscale embeddings
Stock-wise score head:      score[N]
Optional cross-stock block: contextual_scores[N] or features[N,D]
Qlib labels:                future_return[N]
Ranking evaluation:         IC / RankIC per date
Trading evaluation:         cost-aware TopK-DropN and realistic equity constraints
```

Do not use `N` as a fixed architecture dimension. In Qlib it changes through time.

## 106. Complexity — qualified reasoning only

[DERIVATION]

With \(M+1\) temporal scales, \(K\) period views, \(L\) blocks, many input channels and separate 2D conv + axial attentions, computational load grows with \(LK(M+1)\) representations and attention lengths along image axes. Channel-wise attention at the coarsest scale can scale quadratically with the number of **variables**, although exact FLOPs depend on projection and head dimensions. FFT is not the dominant cost in all regimes. Do **not** assign a precise asymptotic FLOP formula to TimeMixer++ without deriving the actual implemented kernels/masks; the supplied paper emphasizes empirical efficiency rather than a complete symbolic complexity theorem.

## 107. Reproducibility checklist

- [ ] Source is ICLR 2025 TimeMixer++, not TimeMixer 2024.
- [ ] Downsample with stride-2 convolution, not copied average pooling from 2024.
- [ ] Apply channel mixing at coarsest scale with correct variate attention.
- [ ] Embed all \(M+1\) scales to common \(d_{\mathrm{model}}\).
- [ ] Perform FFT on coarsest **embedded** representation at each MixerBlock.
- [ ] Verify period/frequency indexing and rescaling.
- [ ] Keep top-K periods and period scores associated correctly.
- [ ] Form \((M+1)K\) time images with correct padding, reshape and crop.
- [ ] Confirm column/row axis attention maps to seasonal/trend branch as implemented.
- [ ] Apply fine→coarse 2D convolution mixing for seasonal images.
- [ ] Apply coarse→fine transposed 2D convolution mixing for trends.
- [ ] Sum branches, reshape/crop, FFT-weight mix across periods.
- [ ] Add residuals and LayerNorm exactly as required.
- [ ] Use appropriate task heads, loss, horizon and evaluation protocol.
- [ ] Match original repeats (3) or explicitly distinguish expanded seed runs.
- [ ] Resolve the data/result inconsistencies documented in §77 before citing headline comparisons.

---

# Part IV — Financial research-agent connections and falsifiable migration experiments

## 108. Transfer rule: mechanism, not full architecture

[EXTENSION] For a KBS-oriented cross-sectional model project, treat TimeMixer++ as an **innovation mechanism library**, not an obligation to implement full general time-series analysis. The minimal transferable primitives are:

1. coarsened global cross-variable attention;
2. multi-scale historical views;
3. selected frequency/periodic views;
4. attention-based learned trend/fluctuation decomposition;
5. directional information transfer across scales;
6. scale- and resolution-specific output fusion.

Each primitive should first be evaluated independently to identify useful causal contributions under a fair local protocol.

## 109. Connection to MASTER

[SOURCE for architecture family; EXTENSION for transfer]

MASTER's core problem is cross-time stock relationship modeling through time-first/stock-second attention. TimeMixer++ provides a different way to **encode the stock's historical time series before cross-stock interaction**. A candidate hybrid is:

```text
Alpha158 x[N,T,C]
    ↓
TimeMixer++ lightweight per-stock temporal encoder
    ↓
stock-time multiscale representation
    ↓
MASTER-style cross-stock relational block
    ↓
Stock ranking head
```

Key test: does this improve **within-stock representation** after controlling for cross-stock attention, or merely add parameter capacity?

## 110. Connection to MATCC

MATCC explicitly adds market trends, separates trend/fluctuation by pooling, then uses RWKV and cross-stock attention. TimeMixer++ suggests replacing the **single-scale fixed trend decomposition** with a richer learned multi-scale/time-image representation.

[EXTENSION] However, because MATCC's ablation shows trend decomposition is central, a fair experiment must test: simple AvgPool → TimeMixer-style multi-scale mixing → TimeMixer++-style MRTI+TID, **keeping the downstream RWKV/stock attention and protocol fixed**.

## 111. Connection to PatchTST

PatchTST proposes fixed-length patches and channel-independent attention. TimeMixer++ proposes **data-dependent FFT periods** and coarse channel mixing. This creates orthogonal experimental factors:

| Axis | Option A | Option B |
|---|---|---|
| Time segmentation | fixed PatchTST patches | FFT-selected time-image periods |
| Cross-feature handling | channel independent | coarsest-scale channel attention |
| Across-scale synthesis | single scale | multiscale pyramid |
| Scoring | single head | scale/resolution ensemble |

Design a 2×2 or factorial study rather than compare two unrelated full models and declare one component superior.

## 112. Connection to FactorVAE

FactorVAE's predictive uncertainty comes from a Gaussian latent factor distribution. TimeMixer++ does not provide the same uncertainty decomposition. It can potentially replace the **history encoder** or enrich factor exposures with multiscale patterns, but it does not automatically implement probabilistic factor inference or uncertainty calibration.

## 113. Connection to FactorVQVAE

FactorVQVAE learns discrete latent factor tokens and autoregressive dynamics. TimeMixer++ produces continuous time-image representations informed by frequency components. A proposed combination is to **quantize scale-specific temporal patterns** or to use selected frequency patterns to condition token prediction. This is an untested idea; do not call TimeMixer++ a VQ architecture.

## 114. Connection to PRISM-VQ

PRISM-VQ has a discrete cross-sectional codebook and a code-conditioned temporal Transformer/MoE. TimeMixer++ could be considered for the **Stage-2 temporal feature extractor**, but the original PRISM-VQ codebook is a cross-asset structure learned in Stage 1; Fourier patterns from TimeMixer++ serve a different role.

[EXTENSION] Keep codebook/experts/priors frozen as controlled baseline elements when testing a new temporal block. If re-training Stage 1, report that as a separate experimental condition.

## 115. Three distinct interaction axes in a hybrid financial model

[DERIVATION]

A stock machine learning example can have axes:

\[
\text{stock }i\ \times\ \text{lookback time }t\ \times\ \text{feature }c.
\]

TimeMixer++ primarily models **time and feature-channel** structure, with multiple resampling scales and frequency-dependent time images. MASTER/MATCC emphasize **cross-stock** interactions. PRISM-VQ additionally models **latent code/expert** structure. These axes should not be conflated merely because each model uses attention.

## 116. Recommended conservative adaptation ladder

[EXTENSION]

```text
Control 0: Current PRISM-VQ/MATCC/MASTER baseline (unchanged)
Control 1: Same model + simple multi-scale pooling
Control 2: Control 1 + directional scale mixing (TimeMixer 2024 style)
Control 3: Control 1 + FFT period views without TID
Control 4: Control 3 + TID, no MCM
Control 5: Control 4 + MCM
Control 6: Control 5 + MRM
Control 7: Full TimeMixer++ temporal encoder adaptation
```

A direct full-model insertion is a poor **first** scientific test because it cannot identify which expensive parts help.

## 117. Experiment TPP-01 — scale count under 8/20-day lookback

[EXTENSION]

- **Hypothesis**: adding one nondegenerate coarse scale improves short-horizon stock representation; additional scales become harmful once too few temporal observations remain.
- **Change**: vary \(M\in\{0,1,2\}\) with otherwise identical architecture, feature/label protocol, and training budget.
- **Market**: CSI300 and S&P500.
- **Metrics**: 5-seed IC/RankIC, ICIR/RankICIR, inference cost, months with negative RankIC.
- **Diagnostics**: actual FFT-bin count and coarsest length at each M.
- **Failure**: gains vanish or reverse after controlling for parameter count; M=2 destabilizes representation.
- **Do not** infer from the paper's longer-series \(M=3\) default.

## 118. Experiment TPP-02 — learned downsampling versus pooling

- **Hypothesis**: a learned stride-2 Conv preserves stock-specific predictive motifs better than fixed pooling.
- **Ablations**: average pooling, max pooling, anti-aliased low-pass pooling, strided Conv.
- **Control**: identical scale count, temporal encoder, classifier/rank head and GPU budget.
- **Measure**: test RankIC, stability, Fourier-peak sensitivity, runtime.
- **Failure**: convolution gains vanish after retraining baseline or higher-capacity Conv overfits one market.

## 119. Experiment TPP-03 — short-window FFT reliability

- **Hypothesis**: FFT-derived periods are unstable at 8/20 days; a confidence fallback can prevent noise amplification.
- **Variants**: no FFT; raw top-K; minimum peak-prominence threshold; spectral smoothing; a fallback fixed temporal partition.
- **Diagnostics**: top-K period Jaccard stability across adjacent days, period shifts per stock, RankIC conditioned on spectral confidence.
- **Failure**: peak stability bears no relation to predictive gains or confidence filtering discards useful rare patterns.

## 120. Experiment TPP-04 — learned TID vs MATCC AvgPool decomposition

- **Hypothesis**: learned latent time-image decomposition captures richer predictive trend/residual modes than fixed smoothing.
- **Variants**: MATCC AvgPool baseline, 1D learned decomposition, 2D TID-only module.
- **Controlled**: same RWKV/stock attention/prediction head, same label/preprocessing.
- **Metrics**: mean and tail RankIC, subperiod stability, drawdown and compute.
- **Failure**: no consistent 5-seed gain, worse OOS stability, or added model complexity dominates.

## 121. Experiment TPP-05 — bottom-up seasonal / top-down trend asymmetry

- **Hypothesis**: the TimeMixer family direction prior helps financial sequences when medium-horizon structure is learnable.
- **Variants**: both bottom-up; both top-down; reversed; original asymmetry; no scale interaction.
- **Diagnostics**: which scale contributes to scores and whether gains concentrate in specific market states.
- **Failure**: one generic symmetric mixer performs equally well or better under the same tuning budget.

## 122. Experiment TPP-06 — period-amplitude fusion versus predictive gating

- **Hypothesis**: amplitude is an imperfect proxy for financial predictive relevance.
- **Variants**: uniform mean, amplitude softmax (source mechanism), trainable weighted sum, validation-supervised gate, confidence-filtered gate.
- **Metrics**: RankIC, RankICIR, turnover, performance conditional on high/low FFT peak dominance.
- **Failure**: learned gates overfit validation; amplitude-based scheme is superior across seeds/markets.

## 123. Experiment TPP-07 — Channel Independence versus coarsened feature attention

- **Hypothesis**: explicit Alpha158 cross-feature relationships help after temporal compression, without the noise of dense raw-level mixing.
- **Variants**: PatchTST-style independence; late MLP fusion; coarse feature attention; full-resolution feature attention.
- **Control**: identical encoder latent width and parameter budget where possible.
- **Diagnostics**: feature-family interactions, effective rank, robustness to redundant-factor injection.
- **Failure**: attention mostly learns trivial feature duplicates or worsens generalization on S&P500.

## 124. Experiment TPP-08 — market-aware period selection

- **Hypothesis**: period selection from a market-wide reference sequence may be more stable than per-stock FFT on short histories.
- **Variants**: per-stock spectrum; aggregated stock-feature spectrum; explicit 63-dimensional market context; jointly conditioned periods.
- **Test**: periodic consistency, RankIC, sector patterns, major regime transitions.
- **Leakage rule**: market sequence ends by decision date; never use future market index returns.
- **Failure**: global period choices hide idiosyncratic alpha or gains do not replicate outside CSI300.

## 125. Experiment TPP-09 — market regimes and dual-axis representation

- **Hypothesis**: learned seasonal/trend streams respond differently during high-volatility or trend-reversal regimes.
- **Diagnostics**: cross-regime branch activation, branch-specific score IC, segment-wise RankIC and volatility of RankIC.
- **Ablation**: separate each branch, reverse direction, shuffle regime labels in an auxiliary probe.
- **Failure**: branch labels do not correspond to stable behavior or differences are explained by simple market beta.

## 126. Experiment TPP-10 — scale-specialist ranking heads

- **Hypothesis**: separate short/mid/long temporal heads have complementary cross-sectional signals.
- **Variants**: single-head, equal ensemble, MRM amplitude weighting, confidence or regime gated ensemble.
- **Control**: calibrate score scales, apply ranking loss only in an otherwise matched experiment.
- **Measure**: score correlations, incremental residual IC, monthly rank contributions, turnover.
- **Failure**: heads duplicate one another and ensemble gains disappear after proper validation calibration.

## 127. Experiment TPP-11 — residual predictive information beyond Alpha158 families

- **Hypothesis**: a new multiresolution temporal representation adds information beyond established feature families.
- **Procedure**: cross-sectionally neutralize model scores against size/volatility/momentum-style proxies from permitted data; compare residual IC.
- **Diagnostics**: return ranking inside size/value/industry bins; temporal stability; nonlinearity tests.
- **Failure**: all gains are explained by a simple preexisting factor tilt.

## 128. Experiment TPP-12 — longer lookback to unlock time imaging

- **Hypothesis**: longer context, e.g., 40 or 60 sessions, may supply enough periodic data for FFT-based views to be meaningful.
- **Variants**: T=8,20,40,60 with input dimensionality and number of scale steps adjusted deliberately.
- **Fairness**: compare a strong temporal baseline with **the same longer lookback** so that gain is attributable to architecture, not extra history.
- **Failure**: stronger long-context models fail to outperform simple pooling or validation sensitivity becomes excessive.

## 129. Experiment TPP-13 — leakage audit for multiresolution labels

- **Hypothesis**: a carefully causal implementation preserves the original predictor's information set.
- **Checks**: one-date feature cutoff; token masks; spectral computation only on time ≤t; no future label in period selection; rolling/expanding normalization training-only statistics.
- **Negative control**: deliberately shift input boundaries in a sandbox to ensure leakage detector changes results as expected, then remove the bad version.
- **Failure**: any proposed gain depends on information unavailable at decision time.

## 130. Experiment TPP-14 — robustness to market-feature corruption

- **Hypothesis**: coarse and frequency-aware representations reduce sensitivity to noisy market inputs.
- **Distortions**: Gaussian perturbation, random bursts, structured missing runs, delayed one-day market features, and stationary-vs-shifted noise.
- **Measure**: RankIC degradation curves, codebook/feature trajectory changes, cost-aware backtest stability.
- **Failure**: architecture appears robust only to artificial iid Gaussian noise, not temporally structured failures.

## 131. Experiment TPP-15 — anomaly/proxy regime detector

- **Hypothesis**: the same representation can support a secondary unsupervised anomaly or regime-transition head without sacrificing stock ranking.
- **Architecture**: shared temporal encoder + auxiliary historical reconstruction/residual head + return-ranking head.
- **Evaluation**: prediction RankIC, explanatory detection quality against independent historical regime labels, no test-derived optimization.
- **Failure**: auxiliary loss harms rank quality or anomaly scores merely mirror volatility.

## 132. Experiment TPP-16 — transfer across CSI300 and S&P500

- **Hypothesis**: shared time-image pattern primitives improve cross-market transfer compared with market-specific temporal encoders.
- **Protocol**: strict feature-unit alignment; train one market; test the other with clearly stated recalibration rules; add separate within-market controls.
- **Evaluate**: RankIC, calibration, number of fine-tuning samples, negative monthly RankIC incidence.
- **Failure**: source-market periodic patterns do not transfer or adaptation cost erases benefits.

## 133. Experiment TPP-17 — quantized multiscale patterns

- **Hypothesis**: compressing TimeMixer++ representations into a small VQ vocabulary can stabilize repetitive patterns.
- **Architecture**: temporal encoder → selective quantization at scale \(m\) or after MRM → downstream PRISM-like structural module.
- **Control**: continuous representation with equal bottleneck dimensionality; single-scale VQ; random code assignment.
- **Diagnostics**: code utilization, temporal code stability, residual IC, quantization error.
- **Failure**: 2D periodic representations become unstable codewords or quantization destroys rare but predictive events.

## 134. Experiment TPP-18 — scale/range conditional MoE

- **Hypothesis**: different experts should specialize in distinct temporal scales/periods or market regimes.
- **Variants**: fixed FMM sum; learned scale weights; sparse scale-expert routing; shared+specialized experts.
- **Diagnostics**: expert usage entropy and subperiod adaptation; compare with PRISM's original router without temporal imaging.
- **Failure**: expert collapse, no better OOS RankIC, or gains explained by increased parameter count.

## 135. Experiment TPP-19 — direct relation modeling after time images

- **Hypothesis**: TimeMixer++ embeddings capture stock-local temporal variation but need explicit cross-stock modeling for factor-aligned ranking.
- **Variants**: stock-independent temporal scoring; temporal features + MASTER-style attention; temporal features + sparse directed lead-lag graph.
- **Metric**: IC improvements within industry/size bins, compute scalability, attention stability.
- **Failure**: cross-stock relation adds no incremental signal or becomes unstable with changing constituents.

## 136. Experiment TPP-20 — cost-aware economic validation

- **Hypothesis**: a modest improvement in ranking stability may yield a larger net trading benefit if it reduces score churn.
- **Strategy**: same daily TopK-DropN policy as the existing research pipeline, same cost assumptions, same ensemble rules.
- **Measures**: AR, MDD, Sharpe, Calmar, Sortino, Omega, turnover, monthly/quarterly results, statistical uncertainty.
- **Failure**: positive RankIC but negative/inconsistent net return improvements; excessive churn erases all benefits.

## 137. Experiment TPP-21 — forecast/reconstruction multitask conflict

- **Hypothesis**: an encoder competent at forecasting and imputation may learn a more stable general representation for financial data.
- **Variants**: ranking alone; historical masked reconstruction; auxiliary short-horizon prediction; both.
- **Control**: identical input cutoff and no use of test future targets.
- **Failure**: multitask training causes negative transfer to stock ranking or gains rely on future-known masks.

## 138. Experiment TPP-22 — calendar-aware versus FFT-selected seasonal representations

- **Hypothesis**: observed calendar rhythms could outperform learned FFT peaks on very short market windows.
- **Variants**: Fourier top-K; lag-based cycle candidates; weekday/month-end/calendar encodings; combined learned gate.
- **Measure**: actual predictive value, OOS stability, market-specific differences, selection frequency.
- **Failure**: apparent seasonality is confined to one test slice or disappears after realistic costs.

## 139. Experiment TPP-23 — 2D imaging versus equal-capacity 1D temporal encoder

- **Hypothesis**: the 2D grid inductive bias, not merely more convolutions/attention, improves ranking.
- **Controls**: standard 1D Conv/MLP and temporal attention with matched parameters/FLOPs, identical data/time window, same selected periods as auxiliary features where possible.
- **Failure**: equal-capacity 1D baseline matches or beats 2D variants across markets.

## 140. Experiment TPP-24 — soft top-K period selection

- **Hypothesis**: discontinuities at Fourier peak boundaries harm adjacent-day signal stability.
- **Variants**: hard top-K (paper), smooth spectral mask, softmax over all eligible bins, sparse differentiable candidate selector.
- **Metrics**: adjacent-day score rank correlation, predictor variance, RankIC/turnover trade-off, latency.
- **Failure**: smoothing reduces informative adaptivity and harms OOS ranking.

## 141. Prioritization rubric for an autonomous research Agent

[EXTENSION]

For each candidate mechanism \(h\), score **before experimenting**:

| Criterion | Key question |
|---|---|
| Task fit | Does the original mechanism rely on long horizons or strong periodicity absent in stocks? |
| Novelty | Is the same mechanism already present in MASTER/MATCC/PRISM or related financial papers? |
| Minimal change | Can one module be replaced without revising the whole model? |
| Testability | Is there an isolation ablation and a cheap seed-0 smoke test? |
| Cost | Is memory/FLOPs reasonable at CSI300/SP500 cross-section size? |
| Robustness | Can five seeds and year/quarter/month/regime diagnostics reveal instability? |
| Economic validity | Can gains survive label consistency and cost-aware TopK evaluation? |
| Explanation | Can the mechanism be interpreted without inventing economic identities? |

A candidate should not be promoted solely because the original paper has a high venue or broad SOTA claims.

## 142. Recommended first experiments

[EXTENSION] **Low-risk mechanism exploration:**

1. TPP-01: short-window scale count.
2. TPP-02: downsampling variants.
3. TPP-04: learned decomposition vs MATCC pooling.
4. TPP-07: channel independence vs coarse mixing.
5. TPP-10: multi-scale ranking heads.

**Higher-risk / higher-novelty follow-up:**

- TPP-08: market-conditioned period selection.
- TPP-17: temporal VQ prototypes.
- TPP-18: scale/period-specific experts.
- TPP-19: multiscale temporal + cross-stock relations.

Defer exact FFT image modeling for very short windows until spectral reliability is quantified.

## 143. Cross-paper “who owns which mechanism?” index

| Mechanism | Source baseline/foundation | Notes |
|---|---|---|
| Discrete representation via codebook | VQ-VAE; FactorVQVAE; PRISM-VQ | Not native to TimeMixer++ |
| Continuous Gaussian posterior factors | VAE; FactorVAE | Not native to TimeMixer++ |
| Cross-stock momentary / cross-time attention | MASTER | Not identical to TimeMixer++'s channel axis |
| Explicit market trends + stock trend decomposition | MATCC | TimeMixer++'s TID is learned latent 2D version |
| Temporal patch tokenization | PatchTST | Fixed patches vs adaptive spectral periods |
| Multi-scale directional season/trend mixing | TimeMixer 2024 | Inherited, 2D convolutional adaptation in 2025 |
| FFT-based 2D time-series period imaging | TimesNet and successors | Established ingredient used in 2025 combination |
| Coarse channel attention + TID + MCM + MRM | TimeMixer++ 2025 | Main four-component pattern-extractor system |
| Structure-conditioned MoE factor loadings | PRISM-VQ | Distinct finance architecture, possible hybrid |

## 144. What is probably insufficient novelty for a new paper

[EXTENSION]

- Replace GRU by TimeMixer++ with no financial adaptation or mechanism diagnosis.
- Apply an FFT to Alpha158, concatenate frequencies and claim “novel spectral financial factors.”
- Use one extra channel attention layer without evidence of incremental information.
- Increase the number of scales or heads while keeping all other objectives unchanged.
- Compare a longer-history TimeMixer++ model to a shorter-history GRU and attribute all gains to architecture.
- Present visually pretty time-image attention as proof of causal financial factor identity.
- Measure only one test-market RankIC without 5-seed, subperiod, and portfolio diagnostics.

## 145. What could support stronger novelty

[EXTENSION]

A stronger future contribution would state a finance-specific unsolved issue and identify how a mechanism addresses it. Examples:

- **Unreliable short-window spectra:** confidence-calibrated scale/resolution selection conditioned on market regimes.
- **Overlapping temporal rhythms:** compositional time-image prototypes with uncertainty-aware factor routing.
- **Stock-dependent market response:** separate global market frequency states from stock-local scale representations.
- **Factor exposure instability:** multiscale representation as input to a shared/routed factor-loading model, with controlled financial-prior regularization.
- **Temporal and cross-stock interaction:** efficient sparse lagged stock-relation learning over temporal pattern prototypes.

Each needs discriminating ablations, fair competing models, and multi-market economic evaluation.


---

# Part V — Research writing, experimental discipline, and knowledge-base integration

## 146. Writing pattern: general TSPM rather than a single SOTA forecaster

[SOURCE/CRITIQUE]

The paper's story is constructed as:

1. Many real time series contain overlapping temporal scales and periodicities.
2. Different downstream tasks require different patterns and representation behavior.
3. Fixed, shallow, or one-axis architectures are insufficiently flexible for a general pattern machine.
4. Introduce a two-axis representational substrate: **sampling scale × spectral resolution**.
5. Within that substrate, introduce explicit mechanisms to extract, exchange, and fuse learned representations.
6. Establish relevance with broad multi-task benchmarking.
7. Analyze the four components through ablations and representation figures.

[EXTENSION] For a *single-task* financial KBS paper, copying the universal-eight-task marketing may dilute the stronger claim. Prefer a narrower economic question and a targeted evaluation rather than overextending to every time-series task.

## 147. Writing lesson: differentiate “representation” from “decision”

TimeMixer++ creates time-image patterns and then uses task-specific heads. That division is scientifically helpful:

- claim A: representation captures richer structure;
- claim B: task-specific head can exploit it;
- claim C: observed downstream performance improves.

An Agent should use **distinct evidence** for each. Visualization of image structure can support claim A qualitatively, but not prove claim C for a stock portfolio.

## 148. Writing lesson: show mechanism path before the final equation

The method builds a clear sequence:

\[
\text{multi-scale input}
\to \text{MRTI}
\to \text{TID}
\to \text{MCM}
\to \text{MRM}
\to \text{output}.
\]

For each step, the paper explains *what information is missing* and *why this operation is appropriate*. A new research paper should preserve this one-module/one-problem correspondence rather than invent multiple ad hoc blocks with redundant roles.

## 149. Writing lesson: pair 2D diagrams with precise tensor definitions

The original method is relatively easy to visualize (Fig.2, Fig.6–9), but period/frequency indexing and axis labels are sometimes hard to reconstruct from paper-only notation. An improved financial adaptation should provide:

- exact \([N,T,C]\) and \([N,p_k,q_k,D]\) shape transitions;
- which axis attention uses;
- zero-padding and masks;
- period selection and time cutoff;
- market/stock/factor semantics for every axis;
- end-to-end complexity and training-time cost.

This can become a reproducibility advantage in a new paper.

## 150. Writing lesson: report negative cases

[SOURCE/CRITIQUE]

Strong examples of what a careful reader should *not* omit include:

- Electricity imputation DLinear MSE beats TimeMixer++.
- PEMS03 RMSE favors original TimeMixer.
- Several M4 frequency subsets favor original TimeMixer on individual metrics.
- Several UEA classification datasets favor unrelated methods.
- Source has potentially erroneous text/table values.

Honest reporting of these exceptions helps a researcher identify *which inductive bias is actually useful and where*.

## 151. Writing lesson: avoid transferring source's percentage claims verbatim

If a paper states a relative improvement, calculate it against the correct *table row and metric*. Examples:

- Electricity long-term MSE: 0.182 → 0.165 vs predecessor = 0.017 absolute improvement, ~9.3% lower MSE.
- Solar-Energy: 0.216 → 0.203 vs predecessor = 0.013 lower, ~6.0% relative.
- Weather: 0.240 → 0.226 vs predecessor = 0.014 lower, ~5.8% relative.
- Anomaly mean F1 vs TimesNet: 86.34 → 87.47 = 1.13 pp, **not** 2.59% from Table 19.

[DERIVATION] These arithmetic comparisons come from printed table values; they are not separate experiments.

## 152. Claim / evidence / uncertainty ledger

| Claim | Paper evidence | Agent assessment |
|---|---|---|
| Multi-scale plus multi-resolution improves forecasting | Table 1 + component ablations | Strong benchmark-level support, not universal dominance |
| TID is useful | Table 7, Tables 11–12 | Strong within this architecture; no uniqueness theorem |
| MCM directional hierarchy helps | MCM removal ablation, inherited prior direction rationale | Supported, but financial applicability unknown |
| Coarsest feature mixing helps | channel ablations, especially multivariate PeMS | Supported on correlated-variable benchmarks |
| Top FFT amplitudes select useful periodic patterns | results for full mechanism; no isolated alternative spectral-selector table here | Plausible, partially identified; peak relevance unproven |
| Learned time images show trend/season separation | Figs.4,10,11,13 | Qualitative, not formal identification |
| Same backbone works across tasks | 8 task experiments | Broad evidence; task-specific training/heads still required |
| Zero-shot transfer is effective | Table 6, ETT cross-dataset | Supported within ETT family only |
| Scaling laws are established | Appendix K proposes them as future work | **Not** established |
| Stock-ranking performance improves | No CSI300/SP500 tests | **Not supported by source** |

## 153. How to design a “fair TimeMixer++ mechanism” financial paper

[EXTENSION]

**Base experiment protocol**:

- Fixed daily stock universe and *same* time-varying constituent policy for all models.
- Shared temporal train/validation/test split for CSI300 and S&P500.
- Same Alpha158 features, same extra market/JPK factor information if any.
- Exactly the same forward-return horizon and Qlib label definition.
- Same validation tuning budget, or explicitly reported tuning budget.
- 5 independent seeds for prediction metrics; clear distinction from ensemble backtests.
- Cost-aware TopK-DropN or separately presented cost-free baselines with no misleading comparisons.
- Yearly/quarterly/monthly/regime diagnostics, IC/RankIC and return/drawdown metrics.
- One-module isolation and at least one capacity-matched alternative.

**Core counterfactual**: could a 1D model with the same extra historical data/parameters explain the improvement?

## 154. Financial benchmark safety: target horizon incompatibility

[CRITIQUE/EXTENSION]

Earlier financial baselines often differ in return horizon and reference price convention. When adapting TimeMixer++, the original paper does not pick between one-day and five-day stock ranking labels. Your agent must read **local Qlib config**, not infer labels from `TimeMixer++` or merely copy paper performance numbers.

A standardized experiment record should include:

```yaml
market: CSI300_or_SP500
feature_set: Alpha158
lookback_days: 20
label_horizon_days: explicitly_from_local_config
label_price_reference: explicitly_from_local_config
train_range: local_protocol
validation_range: local_protocol
test_range: local_protocol
seed_set: [0, 1, 2, 3, 4]
normalization: local_protocol
portfolio_policy: cost_aware_TopK_DropN
external_market_features: true_or_false
```

This is a template **to fill**, not a claim about TimeMixer++ source data.

## 155. Stock rank loss implementation notes

[EXTENSION]

Possible research losses:

\[
\mathcal L_{\mathrm{hybrid}}
=\mathcal L_{\mathrm{point}}
+\lambda_{\mathrm{rank}}\mathcal L_{\mathrm{rank}}
+\lambda_{\mathrm{reg}}\mathcal L_{\mathrm{reg}}.
\]

- \(\mathcal L_{\mathrm{point}}\): MSE/Huber on returns, if appropriate.
- \(\mathcal L_{\mathrm{rank}}\): differentiable pairwise/listwise surrogate, predeclared from prior work or derived.
- \(\mathcal L_{\mathrm{reg}}\): temporal smoothness, scale diversity, and/or complexity control.

Do **not** attribute this objective to TimeMixer++: its original forecast benchmark uses L2-type losses and task-specific losses.

## 156. Factor-oriented adaptation: prediction ≠ priced risk

[EXTENSION]

If combining with PRISM-VQ, use TimeMixer++ output as a **representation** for dynamic exposures, not as an automatically interpretable financial factor.

For example:

\[
\beta_{i,t}=g(h_{i,t}^{\mathrm{TM++}},z_{q,i,t}),
\]

\[
\widehat y_{i,t}=\alpha_{i,t}+\beta_{p,i,t}^{\top}f_{p,t}+\beta_{l,i,t}^{\top}f_{l,i,t}.
\]

This is a **candidate hybrid** retaining a factor head, *not* an equation in TimeMixer++.

## 157. Economic interpretation criteria for learned periods

[EXTENSION]

Do not declare a spectral period to be a known market regime without at least:

- stable period assignment across neighboring dates and seeds;
- reproducible association to independently constructed macro/market variables;
- adjustment for leakage and multiple testing;
- incremental residual ranking utility beyond simple momentum/reversal/volatility features;
- plausible behavior in both CSI300 and S&P500.

Even then, correspondence ≠ causality.

## 158. The paper's explicit and implicit limitations

**Explicit/source-supported**:

- Appendix K: need for larger time-series datasets and scaling-law exploration; present model is a backbone toward TSPM scaling, not a demonstrated foundation-scale model.
- No cross-sectional stock-prediction experiments.
- A range of architecture decisions and implementation details are in repository rather than fully specified in the 37-page PDF.

**Agent critique/inferred**:

- short-sequence Fourier unreliability;
- input length and downsampling incompatibility;
- hard peak-selection discontinuity;
- aliasing and padding masks;
- distribution shift / per-market adaptation;
- attention complexity at high feature dimensions;
- lack of financial cost/regime evaluation;
- weak economic identifiability of “seasonal/trend” hidden branches.

## 159. Novelty checklist for future manuscripts

[EXTENSION]

A new finance paper that imports TimeMixer++ should explicitly answer:

1. Which distinct financial failure mode motivates the imported component?
2. Why does the original long-series mechanism need adaptation for short stock windows?
3. Which source mechanism has been modified, and which remains standard?
4. Which previous finance/ML papers already contain related multi-scale/decomposition/frequency ideas?
5. Are new input data available to all compared baselines?
6. What is the minimal counterfactual without the imported mechanism?
7. Are gains multi-market, multi-seed, and reasonably stable by subperiod?
8. What happens to turnover, execution costs, and downside risk?
9. Are the learned periods/codes stable and economically interpretable only to the degree evidence shows?
10. What result would falsify the method claim?

## 160. Suggested knowledge-base backlinks

[EXTENSION]

```yaml
related_papers:
  - path: ../foundations/TimeMixer_2024.md
    relation: predecessor
  - path: ../foundations/PatchTST.md
    relation: alternative_temporal_tokenization
  - path: ../baseline/MATCC_2024.md
    relation: finance_trend_decomposition
  - path: ../baseline/MASTER_2024.md
    relation: finance_cross_stock_attention
  - path: ../baseline/FactorVAE_2022.md
    relation: probabilistic_latent_factor_contrast
  - path: ../baseline/FactorVQVAE_2025.md
    relation: discrete_factor_contrast
  - path: ../baseline/PRISM-VQ_2026.md
    relation: discrete_structure_plus_dynamic_factor_loadings
```

Paths are **suggested cross-reference metadata**, not a claim that all these exact files already exist at those spellings; align filenames with the user's actual workspace.

## 161. Agent mechanism inventory (machine-oriented)

```yaml
mechanism_inventory:
  - id: tmpp.stride2_multiscale
    source: "Sec 3 Eq(1)"
    type: temporal_downsampling
    preconditions: [sufficient_history, compatible_padding]
    risk: [aliasing, short_window_collapse]
  - id: tmpp.coarse_channel_attention
    source: "Sec 3 Eq(2)"
    type: cross_variate_representation
    preconditions: [consistent_channel_semantics]
    risk: [collinearity, overfit]
  - id: tmpp.mrti
    source: "Sec 3 Eq(5)-(6)"
    type: spectral_period_tokenization
    preconditions: [informative_history, stable_fft_peaks]
    risk: [peak_discontinuity, padding]
  - id: tmpp.tid
    source: "Sec 3 Eq(7)"
    type: learned_axis_specific_decomposition
    preconditions: [valid_time_image_shape]
    risk: [unidentified_trend_season_semantics]
  - id: tmpp.season_mcm
    source: "Sec 3 Eq(8)"
    type: fine_to_coarse_2d_conv
    preconditions: [shape_aligned_scales]
    risk: [aliasing, inappropriate_direction]
  - id: tmpp.trend_mcm
    source: "Sec 3 Eq(9)"
    type: coarse_to_fine_2d_transconv
    preconditions: [shape_aligned_scales]
    risk: [upsampling_artifacts]
  - id: tmpp.mrm
    source: "Sec 3 Eq(11)"
    type: fft_amplitude_resolution_fusion
    preconditions: [valid_frequency_index_map]
    risk: [energy_not_predictivity]
  - id: tmpp.scale_heads
    source: "Sec 3 Eq(3)"
    type: multi_head_scale_ensemble
    preconditions: [task_specific_heads]
    risk: [head_redundancy, score_scale_mismatch]
```

## 162. Agent red flags before proposing new experiments

- The new candidate is actually just TimeMixer 2024 repeated with a different name.
- FFT uses timestamps after \(t\) while producing the score for date \(t\).
- The source's MSE results are compared numerically with a financial model's RankIC.
- An 8-step lookback is paired with \(M=3\) and frequency selection at a one-observation coarse level.
- Source internal numerical inconsistencies have been auto-corrected without audit trail.
- Frequency attention weights are claimed to be “economic factor loadings” without attribution tests.
- Test-set information influences selection of \(M\), \(K\), loss weights, or financial priors.
- Fewer seeds and a cheaper baseline make an apparent architecture improvement look larger.
- An improvement in forecast error is declared to be an improvement in risk-adjusted portfolio performance without a backtest.

## 163. Paper source location map

| PDF printed location | What to retrieve from source |
|---|---|
| pp.1–2 | Title/abstract, TSPM motivation, CKA reasoning, eight-task framing |
| p.3 | Related work; novelty positioning against decomposition and prior time-series backbones |
| p.4 | Figure 2, multi-scale downsampling Eq.(1), channel mix Eq.(2) |
| p.5 | Residual MixerBlock Eq.(4), MRTI FFT top-K Eq.(5) and reshape Eq.(6) |
| p.6 | TID Eq.(7), MCM Eq.(8–10), MRM Eq.(11), experiment overview |
| p.7 | Long-term forecasting Table 1, M4 Table 2 |
| p.8 | PeMS Table 3, imputation Table 4, few-shot Table 5 |
| p.9 | Zero-shot Table 6, classification/anomaly discussion, Figure 3 |
| p.10 | Long-term component-ablation Table 7, Fig.4, conclusion |
| pp.15–16 | Appendix A dataset counts, training settings, baseline sources and metrics |
| pp.17–19 | Appendix B mechanism diagrams Fig.5–9; Appendix C short/anomaly ablations Tables 11–12 |
| pp.20–22 | Appendix D time-image interpretation figures, CKA representation plots, efficiency figure |
| p.23 | Appendix F sensitivity for scales/layers/input length/top-K |
| p.24 | Appendix G three-run uncertainties Tables 13–15 |
| pp.25–27 | Appendix H full long-term, M4, PeMS, anomaly, and classification results; Appendix K future scaling work |
| pp.28–37 | Visualization gallery comparing model predictions across datasets |

## 164. Figure-by-figure interpretation guide

- **Figure 1:** Eight-task radar-style summary and CKA-vs-performance scatterplots. Use for high-level scope, *not* literal proof of one CKA criterion.
- **Figure 2:** Core TimeMixer++ block overview. Crucial for understanding \(m\times k\) indexing.
- **Figure 3:** Classification and anomaly performance across methods; supplementary raw tables are more precise.
- **Figure 4:** Period views and learned seasonal/trend images (Traffic). Visually suggests representational diversity across resolutions.
- **Figure 5:** Channel attention before embedding, at coarsest scale.
- **Figure 6:** FFT period selection and multiscale time-image layouts.
- **Figure 7:** Dual-axis attention decomposition; inspect axis layout carefully.
- **Figure 8:** Opposite-direction seasonal/trend 2D conv mixing.
- **Figure 9:** FFT amplitude weighted multi-resolution mixing.
- **Figures 10/11/13:** Traffic/ETTm1/Electricity time-image analyses under selected scales and periods.
- **Figure 12:** Task-dependent CKA plots; interpret associations cautiously.
- **Figure 14:** Efficiency bubble plots; use actual axes and hardware before calling a model cheaper.
- **Figure 15:** Hyperparameter sensitivity.
- **Figures 16–25:** Qualitative predictions on selected long/short datasets; examples illustrate fit but are not variance estimates.

## 165. Compact retrieval summary

[SOURCE + bounded critique] TimeMixer++ (Wang et al., ICLR 2025) is a general-purpose time-series representation architecture that extends TimeMixer 2024 from simple multiscale 1D decomposition/MLP mixing to joint **time-scale × frequency-resolution** modeling. Historical multivariate series are recursively downsampled by stride-2 convolution; the coarsest scale is used for cross-variable channel attention. Each residual MixerBlock selects top-K FFT periods from the coarsest hidden sequence; every scale is padded/reshaped into K 2D time images (**MRTI**). Dual-axis attention produces learned seasonal/trend image branches (**TID**). Seasonal images exchange information fine-to-coarse via 2D convolution, while trend images exchange information coarse-to-fine via transposed 2D convolution (**MCM**). Per-scale period views are fused using FFT-amplitude softmax weights (**MRM**), and task-specific scale heads are ensembled. The authors evaluate eight task families on many non-financial benchmarks (ETT/Weather/Electricity/Traffic/Solar/Exchange, M4, PeMS, imputation, UEA classification, anomaly detection, few/zero-shot), usually with three runs; key long-term Table 7 ablations show removing TID worsens average MSE from 0.300 to 0.329. The paper reports strong aggregate performance but is not best on every individual metric: e.g., DLinear has lower Electricity imputation MSE, TimeMixer has lower PEMS03 RMSE, and several M4 groups favor older models. Internal numerical inconsistencies include the ETT mean MAE in Table 1 versus Appendix Table 16 and classification accuracy 75.9% in prose versus 75.3% in Table 20. The paper does **not** establish any IC/RankIC or equity backtest performance. For financial research, useful mechanism hypotheses include adaptive frequency views, learned latent decomposition, cross-scale directional mixing, and scale-head fusion, with special care because original source sequences are far longer than 8/20-day stock lookbacks. Appendix K names larger time-series datasets and scaling-law research as explicit future directions.

## 166. One-paragraph experimental design memory

Agent recommendation: Before migrating this paper, keep a simple time-scale baseline and a non-FFT decomposition baseline, then add one new TimeMixer++ mechanism at a time. Track feature/label horizon and source-data cutoff, five seeds, CSI300 and S&P500, IC/RankIC/IR, monthly/quarterly performance, and net-of-cost TopK-DropN. Treat Fourier peak instability, code/representation identifiability, and cross-market protocol drift as failure risks. Review original TimeMixer and MATCC so that a trend decomposition idea is not erroneously framed as novel. Do not trust inconsistent source aggregate cells without consulting their corresponding full appendix tables.

## 167. Agent final takeaways

1. **Central innovation:** construct a two-dimensional grid of **temporal scales × period resolutions** and learn patterns across both axes.
2. **Key representation mechanism:** MRTI turns 1D histories into period-indexed time images; FFT selects the image geometry.
3. **Key decomposition mechanism:** TID uses separate latent axial attentions, rather than fixed moving-average trend splitting.
4. **Key hierarchical mechanism:** fine-to-coarse seasonal 2D Conv; coarse-to-fine trend 2D TransConv.
5. **Key fusion mechanism:** FFT-amplitude-weighted resolutions within each scale, then ensemble task heads across scales.
6. **Strongest long-term component ablation:** removing TID increases average MSE 0.300 → 0.329 in Table 7.
7. **Major empirical advantage:** breadth across forecasting, imputation, classification, anomaly, data-scarce and cross-ETT transfer tasks.
8. **Main scientific caution:** broad average superiority is not all-cell superiority; source data contain contradictions that should be tracked.
9. **Main applicability risk for Qlib:** 8–20 historical sessions may be too short for robust FFT + multi-level downsampling.
10. **Best possible reuse:** extract and test short-window **scale mixing, channel interaction, and learned decomposition** as mechanisms before committing to the full 2025 image pipeline.
11. **Explicit author future direction:** build larger time-series datasets and investigate scaling laws for general time-series pattern machines.
12. **Overall research-agent role:** `frontier/` mechanism source, **not** an already validated financial baseline.
