---
paper_id: PatchTST_2023
title: "A Time Series Is Worth 64 Words: Long-term Forecasting with Transformers"
short_name: "PatchTST"
authors:
  - Yuqi Nie
  - Nam H. Nguyen
  - Phanwadee Sinthong
  - Jayant Kalagnanam
year: 2023
venue: "International Conference on Learning Representations (ICLR 2023)"
arxiv_id: "2211.14730"
arxiv_version: "v2 (2023-03-05)"
paper_type: foundations
subtype:
  - long_term_time_series_forecasting
  - representation_learning
  - transformer
  - patching
  - channel_independence
  - masked_self_supervision
source_document: "Nie et al., A Time Series Is Worth 64 Words, uploaded 24-page ICLR 2023 PDF"
source_pages: 24
main_task: "Multivariate long-horizon time-series point forecasting"
secondary_tasks:
  - masked_patch_self_supervised_pretraining
  - linear_probing
  - finetuning
  - cross_dataset_transfer
original_datasets:
  - Weather
  - Traffic
  - Electricity
  - ILI
  - ETTh1
  - ETTh2
  - ETTm1
  - ETTm2
key_mechanisms:
  - time_series_patch_tokenization
  - shared_weight_channel_independence
  - patch_level_transformer
  - instance_normalization
  - masked_patch_reconstruction
  - transferable_time_series_representations
recommended_directory: "papers/foundations/PatchTST.md"
primary_finance_relevance:
  - local_temporal_pattern_encoding
  - parameter_sharing_across_series
  - masked_pretraining
  - late_cross_variable_or_cross_asset_fusion
source_confidence: "Original user-supplied PDF; source facts distinguished from derivations and proposed extensions"
---

# PatchTST — A Time Series Is Worth 64 Words: Long-term Forecasting with Transformers

> **Paper role:** foundation / reusable architecture, NOT a reproduced Qlib financial baseline.  
> **Original source:** Yuqi Nie, Nam H. Nguyen, Phanwadee Sinthong, Jayant Kalagnanam; ICLR 2023, 24-page supplied PDF; arXiv:2211.14730v2.  
> **Suggested path:** `papers/foundations/PatchTST.md`.  
> **Core idea:** local **patch-level tokens** + **channel-independent computation with shared parameters** + (optionally) **masked-patch self-supervised pretraining**.

## How to read this Agent document

We consistently separate four types of information:

- **`[SOURCE]`** Explicit information from the attached original paper, with original section / page / equation / table where possible.
- **`[DERIVATION]`** Mathematical or implementation consequence derived from the source; do not misattribute to the paper's original wording.
- **`[CRITIQUE]`** Independent evaluation of what the evidence does or does not establish.
- **`[EXTENSION]`** Hypotheses and experiments for cross-sectional stock ranking; these are **not** results reported by PatchTST.

**Key reliability rule:** The source paper is a long-horizon sequence-value forecasting paper measured mainly with MSE/MAE. It does **not** contain CSI300, S&P500, Alpha158, RankIC, TopK, Sharpe, or financial backtests. Any cross-sectional finance discussion below is an explicitly marked transfer proposal.

---

# Part I — Paper Identity, Problem, Motivation, and Contributions

## 1. Metadata and position in the literature `[SOURCE]`

- Full title: *A Time Series Is Worth 64 Words: Long-term Forecasting with Transformers*.
- Short name: **PatchTST** (Patch Time Series Transformer).
- Authors: Yuqi Nie, Nam H. Nguyen, Phanwadee Sinthong, Jayant Kalagnanam.
- ICLR **2023** conference paper, not a 2026 frontier publication.
- The uploaded PDF is arXiv v2 dated 5 March 2023.
- Paper body: 9 pages; references/appendix extend to 24 PDF pages.
- Subject: multivariate, long-term, regular-gridded time-series forecasting and self-supervised representation learning.
- Core network: **vanilla Transformer encoder**, with changes primarily in **input tokenization** and **channel handling**.
- Venue matters historically because the paper answers contemporary criticism that simple linear forecasters could outperform elaborate time-series Transformers.

## 2. The research problem `[SOURCE]`

Suppose a multivariate time series has `M` variables and a look-back of `L` time points. Given history, forecast the next `T` time points of **each variable**:

$$
X_{1:L}=\{x_t\}_{t=1}^{L},\qquad x_t\in\mathbb R^M,
$$

$$
\widehat X_{L+1:L+T}=f_\theta(X_{1:L})\in\mathbb R^{M\times T}.
$$

The original objective is predicting **future sequence values**, not ranking a changing universe of assets. The forecast horizon `T` here is **not** the historical look-back used in many financial-model descriptions.

**Original challenge:** Transformer models employing one token per timestamp often have high computational cost and disappointing long-horizon accuracy, even compared with a simple linear method such as DLinear.

## 3. Historical conflict motivating the paper `[SOURCE]`

The Introduction highlights a challenge raised by *Are Transformers Effective for Time Series Forecasting?* (Zeng et al., 2022): several elaborate Transformer forecasting designs do not consistently outperform elementary linear baselines. PatchTST does not respond by introducing yet another complicated attention kernel. It challenges the **input representation** and **variable-mixing choice**.

Argument chain:

1. Vanilla timestamp-token Transformer treats each timestamp as a token, despite local adjacent numeric observations often sharing semantic content.
2. Transformer attention scales quadratically in token count; point-level attention becomes expensive on long histories.
3. Many earlier multivariate forecasters combine all variables into one token, which can mix unrelated variable dynamics too early and increase data needs.
4. Representing a short subsequence as **one patch token** provides local context, shortens attention, and permits longer historical reach.
5. Processing each channel **independently but with shared weights** allows channel-specific attention patterns and parameter sharing without premature channel mixing.
6. The same backbone can learn representations through **masked-patch reconstruction**, then transfer to downstream tasks.

## 4. Original contributions `[SOURCE]`

### 4.1 Patching as a tokenization principle

Group consecutive time steps into local patches, use each patch as a Transformer token. The paper proposes this to gain three benefits:

- richer local semantic content than a single point;
- quadratic reduction in the number of attention-map entries at fixed history;
- the ability to use a longer look-back given compute/memory constraints.

### 4.2 Channel Independence (CI)

Split multivariate data into univariate channels; **each channel has its own forward-pass values and attention maps**, while **sharing** the embedding projection, Transformer backbone weights, and, in the canonical setup, prediction machinery.

This is not a separate set of parameters for every variable.

### 4.3 Self-supervised masked patch learning

Pretrain the same model by masking patches and reconstructing them; assess linear probing, end-to-end fine-tuning, and cross-dataset transfer.

### 4.4 Empirical characterization

The paper investigates supervised forecasting against strong Transformer and DLinear baselines, 2×2 patching/CI ablations, scaling with look-back, self-supervised training and transfer, varying patch length, instance normalization, seeds, and model dimensions.

## 5. What the paper does NOT claim `[SOURCE/CRITIQUE]`

- It is **not** the invention of Transformer self-attention or patch embeddings in all modalities; it explicitly cites prior NLP/CV patch or subword ideas.
- It does not propose GCN-style explicit inter-variable graphs as a central mechanism.
- It does not show that all forms of cross-channel information are useless; the authors identify modeling cross-channel relationships as a future direction.
- It does not prove patching improves every dataset/horizon equally; some individual original-table cells favor baselines.
- It does not test minute/day stock rankings, short 8-/20-point finance sequences, or transaction-cost portfolios.
- It does not compare every baseline using a fully identical look-back and tuning budget; it explicitly strengthens some baselines by rerunning their look-backs.

---

# Part II — Formal Architecture and Mathematics

## 6. Notation dictionary `[SOURCE, §3.1, pp.3–5]`

| Symbol | Meaning | Canonical example |
|---|---|---|
| `B` | minibatch sample count | configurable |
| `M` | number of variables/channels in each multivariate series | Traffic 862 |
| `L` | input historical look-back | 336 or 512 |
| `T` | output forecast horizon | 96, 192, 336, 720 (except ILI) |
| `P` | patch length in original time steps | 16 supervised |
| `S` | patch stride | 8 supervised |
| `N` | patch count per channel | 42 at `L=336`; 64 at `L=512` |
| `D` | Transformer embedding dimension | 128 in default large-data models |
| `H` | number of attention heads | 16 in default large-data models |
| `F` | Transformer FFN intermediate dimension | 256 in default large-data models |
| `x^{(i)}` | univariate history of channel `i` | length `L` |
| `X_p^{(i)}` | raw patch matrix for channel `i` | `P × N` |
| `X_d^{(i)}` | embedded patch tokens | `D × N` |
| `Z^{(i)}` | Transformer output patches | `D × N` |

In the source the letter `T` represents **forecast length**; many financial codes use `T` for look-back, so rename axes clearly in local implementations.

## 7. Overall supervised computation graph `[SOURCE, Fig.1a/b, p.4]`

```text
Original multivariate sequence [B, M, L]
       │
       ├── channel 1 [B, L] ─┐
       ├── channel 2 [B, L] ─┤    Same parameters, independent forward passes
       │         ...        ├───────────────────────────────────────┐
       └── channel M [B, L] ─┘                                       │
                         │                                          │
                         ▼                                          │
             per-instance normalization                             │
                         │                                          │
                         ▼                                          │
           patchify each univariate sequence                         │
             [B·M, N, P]                                            │
                         │                                          │
                         ▼                                          │
            shared linear patch projection                          │
                         │                                          │
                         ▼                                          │
            add learnable positional embedding                      │
                         │                                          │
                         ▼                                          │
             shared Transformer encoder                             │
                [B·M, N, D]                                         │
                         │                                          │
                         ▼                                          │
                    flatten                                        │
                         │                                          │
                         ▼                                          │
             shared linear prediction head                          │
                [B·M, T]                                           │
                         │                                          │
                         ▼                                          │
             reverse instance normalization                         │
                         │                                          │
                         ▼                                          │
                reshape to [B, M, T] ◄───────────────────────────────┘
```

**Figure interpretation:** Fig. 1a separates input channels, Fig. 1b shows supervised patch embedding + backbone + flatten head, and Fig. 1c shows a different masked-patch reconstruction head for self-supervision. The backbone idea is shared; the training objective and decoder head are not identical.

## 8. Channel independence: exact meaning `[SOURCE]`

Given:

$$
X\in\mathbb R^{B\times M\times L},
$$

channel `i` is:

$$
X^{(i)}\in\mathbb R^{B\times L}.
$$

**Same learned function**, applied independently:

$$
\hat x^{(i)}=f_\theta\big(x^{(i)}\big),\qquad i=1,\ldots,M.
$$

The **parameter vector `θ` is shared**, but channel outputs, Q/K/V activations and attention weights are computed **from that channel's own values**. Hence different channels can have different attention matrices without introducing distinct Transformer parameter sets.

Practical implementation from Appendix A.1.5:

$$
[B,M,P,N]\longrightarrow[B\cdot M,P,N]
$$

(or in conventional batch-first token layout, `[B·M, N, P]`). Then apply a standard Transformer and reshape the predictions back to `[B,M,T]`.

### 8.1 Why parameter sharing matters `[DERIVATION]`

If channels used separate networks, parameter count could grow approximately proportionally to `M`. With shared weights, the Transformer-parameter count is approximately independent of `M` even though **compute for a fixed batch still scales with M**. This is an architectural trade-off, not free computation.

### 8.2 What CI does NOT model directly `[CRITIQUE]`

A pure channel-independent Transformer does not perform explicit channel-to-channel attention. Any cross-channel generalization from sharing parameters is **not equivalent to learning time-varying pairwise cross-channel causal or attention edges**. That is why the original paper openly proposes later cross-channel modeling as future work.

## 9. Patch extraction `[SOURCE, §3.1, p.4]`

For channel `i`:

$$
 x^{(i)}_{1:L}=[x_1^{(i)},\dots,x_L^{(i)}].
$$

Extract `N` patches, each of length `P`, with stride `S`:

$$
X_p^{(i)}\in\mathbb R^{P\times N}.
$$

The original supervised model pads **`S` copies of the last observed value** to the right. With this specific padding rule the paper gives:

$$
\boxed{N=\left\lfloor\frac{L-P}{S}\right\rfloor+2}.\tag{patch-count}
$$

This is not the same as a no-padding unfold formula, which gives `floor((L-P)/S)+1`.

### 9.1 Actual named variants `[DERIVATION from SOURCE §4.1]`

| Variant | Look-back `L` | Patch length `P` | Stride `S` | Resulting patches `N` |
|---|---:|---:|---:|---:|
| `PatchTST/42` | 336 | 16 | 8 | `floor((336-16)/8)+2 = 42` |
| `PatchTST/64` | 512 | 16 | 8 | `floor((512-16)/8)+2 = 64` |

**Critical naming warning:** `/64` and `/42` denote **patch counts**, not patch lengths, model layers or forecast steps. The default supervised **patch length is 16**, stride **8**.

### 9.2 Why overlap is allowed `[INTERPRETATION]`

For `P=16, S=8`, adjacent patches overlap by 8 values. Overlap retains boundary information and lets the model represent local behavior with smoother transitions than strictly disjoint partitioning. The original paper allows overlapped **or** non-overlapped supervised patches; this explanation is interpretive, not a separately isolated causal finding.

### 9.3 Computation trade-off `[DERIVATION]`

Standard point-token attention for a univariate channel costs roughly:

$$
O(L^2D)
$$

for the attention-matrix multiplications, whereas PatchTST costs:

$$
O(N^2D),\qquad N\approx L/S.
$$

Attention-map entries shrink approximately by:

$$
\left(\frac{L}{N}\right)^2\approx S^2.
$$

But full runtime includes patch projection, attention parameters, normalization, outputs, and `M` independent channels; therefore **total measured speed is not guaranteed to improve by exactly `S²`**.

## 10. Patch embedding `[SOURCE, §3.1, p.4]`

A trainable linear projection maps a whole patch to `D` dimensions:

$$
W_p\in\mathbb R^{D\times P}.
$$

For channel `i`:

$$
X_d^{(i)}=W_pX_p^{(i)}+W_{pos},\qquad
W_{pos}\in\mathbb R^{D\times N},
$$

$$
X_d^{(i)}\in\mathbb R^{D\times N}.
$$

The added position embedding is **learnable** (not a required sinusoidal encoding). It encodes the sequence order of patches.

### 10.1 Interpretation `[INTERPRETATION]`

Patch projection is not merely a downsampling operator. Its trainable weights can detect local shapes (spikes, ramps, oscillations and plateaus) across `P` adjacent points. The Transformer then relates these **learned local-pattern tokens** instead of raw scalar timestamps.

## 11. Backbone: deliberately vanilla Transformer `[SOURCE, §3.1, pp.4–5]`

For each channel, let token sequence in row-first convention be:

$$
H=(X_d^{(i)})^\top\in\mathbb R^{N\times D}.
$$

For attention head `h`:

$$
Q_h=HW_h^Q,\qquad K_h=HW_h^K,\qquad V_h=HW_h^V.
$$

Scaled dot-product attention:

$$
A_h=\operatorname{softmax}\left(\frac{Q_hK_h^\top}{\sqrt{d_k}}\right),
$$

$$
O_h=A_hV_h.
$$

Head outputs are projected/concatenated as in standard multi-head self-attention, followed by residual connections and an FFN.

**Normalization detail:** The source specifically describes **BatchNorm** in the Transformer encoder block, with a footnote citing results favoring BatchNorm over LayerNorm for time-series Transformers. Reimplementations must not silently assume LayerNorm merely because it is common in NLP.

## 12. Supervised forecast head `[SOURCE, Fig.1b, p.5]`

The final channel representation is:

$$
Z^{(i)}\in\mathbb R^{D\times N}.
$$

Flatten and apply a linear head:

$$
\hat x^{(i)}_{L+1:L+T}
= W_{out}\operatorname{vec}(Z^{(i)})+b_{out},
$$

with shapes:

$$
W_{out}\in\mathbb R^{T\times DN},\qquad
\hat x^{(i)}_{L+1:L+T}\in\mathbb R^T.
$$

**Notation warning:** The explicit `W_out` symbol/shape here is an implementation consequence of the stated Flatten+Linear head, not an additional separately numbered equation in the source.

### 12.1 Non-autoregressive multi-step head `[DERIVATION]`

The supervised model predicts a horizon-length vector in one forward pass. It does not need to roll out one future value at a time as an autoregressive decoder would.

## 13. Supervised objective `[SOURCE, §3.1, p.5]`

The paper uses **mean squared error** summed/averaged over channels and forecast values. In the paper's compact notation:

$$
\mathcal L_{\text{sup}}=
\mathbb E_X\left[
\frac{1}{M}\sum_{i=1}^M
\left\|
\hat x^{(i)}_{L+1:L+T}-x^{(i)}_{L+1:L+T}
\right\|_2^2
\right].
$$

Exact scaling with `T` depends on MSE reduction in implementation; this does not alter the conceptual goal.

**Not used by the original supervised PatchTST:** RankIC, Sharpe, cross-sectional pairwise ranking loss, portfolio utility, financial factor reconstruction, VQ codebook or expert routing.

## 14. Instance normalization `[SOURCE, §3.1, p.5; Appendix A.4.4, pp.16–18]`

For each input channel instance, center and scale the historical values:

$$
\mu^{(i)}=\operatorname{mean}(x^{(i)}_{1:L}),\qquad
s^{(i)}=\operatorname{std}(x^{(i)}_{1:L}),
$$

$$
\bar x_t^{(i)}=\frac{x_t^{(i)}-\mu^{(i)}}{s^{(i)}}.
$$

The model predicts in normalized coordinates, then restores original level and scale:

$$
\hat x_{L+h}^{(i)}
=s^{(i)}\hat{\bar x}_{L+h}^{(i)}+\mu^{(i)}.
$$

**Implementation clarification:** These equations express the source's instance-normalization operation; an actual implementation needs an epsilon to avoid division by zero, whose exact value is not provided in this text.

The paper shows instance normalization improves some results, but the main gain remains even without it (Table 11); hence normalization alone does not explain the full PatchTST advantage.

## 15. End-to-end tensor-shape trace `[DERIVATION]`

Canonical supervised with `B=32, M=21, L=336, P=16, S=8, N=42, D=128, T=96`:

```text
Raw input                                  [32, 21, 336]
Instance normalize                        [32, 21, 336]
Patchify                                   [32, 21, 42, 16]
Fold variable/channel into batch          [672, 42, 16]
Shared projection into D=128              [672, 42, 128]
Positional embedding                       [672, 42, 128]
Shared Transformer encoder                [672, 42, 128]
Flatten                                   [672, 5376]
Linear head                               [672, 96]
Undo instance normalization               [672, 96]
Restore channel axis                      [32, 21, 96]
```

This is an **illustrative dimension trace**, not the actual minibatch used for every experiment. The model's core structural invariant is `[B·M,N,D]` inside the Transformer.

## 16. Minimal supervised implementation skeleton `[DERIVATION]`

```python
# Illustrative pseudocode, not copied from the repository.
# Input: [B, M, L]

def patchtst_forward(x, patch_len=16, stride=8):
    B, M, L = x.shape
    mean = x.mean(dim=-1, keepdim=True)
    std = x.std(dim=-1, keepdim=True).clamp_min(1e-5)
    normed = (x - mean) / std

    # Paper's supervised padding: repeat last value S times.
    tail = normed[..., -1:].repeat_interleave(stride, dim=-1)
    padded = concatenate([normed, tail], axis=-1)

    patches = unfold(padded, size=patch_len, step=stride)
    # [B, M, N, P] -> [B*M, N, P]
    patches = patches.reshape(B * M, -1, patch_len)

    tokens = patch_projection(patches) + learned_position_embedding
    states = shared_transformer_encoder(tokens)
    normalized_prediction = linear_head(states.flatten(start_dim=1))
    # [B*M, T] -> [B, M, T]
    prediction = normalized_prediction.reshape(B, M, -1)
    return prediction * std + mean
```

**Engineering cautions:** actual channel layout / interpolation, unbiased or biased standard deviation, padding API, BatchNorm behavior, learned position sizing and head parameter sharing must be verified in the real code path. Do not treat this skeleton as an exact official implementation.

---

# Part III — Masked Self-Supervised Learning and Transfer

## 17. Why self-supervised pretraining? `[SOURCE, §3.2, p.5]`

PatchTST is not only a supervised forecast architecture. The authors also ask whether the backbone can learn useful representations **without requiring future forecast labels**.

A common time-series masked-learning issue: if only isolated timestamps are masked, a network may reconstruct missing values using nearby interpolation, without learning high-level patterns. Masking **whole patches** increases the local information gap and encourages pattern-level representations.

## 18. Self-supervised input protocol `[SOURCE, §3.2 and §4.2]`

- Look-back input: `L=512`.
- **Patch length `P=12`** for this pretraining experiment.
- **Non-overlapping patches**, unlike the canonical supervised `P=16,S=8` overlap setting.
- **42 patches** per input sequence (as explicitly reported).
- Random patch masking ratio: **40%**.
- Masked patches are **set to zero**.
- Reconstruction target: **masked patches**.
- Loss: **MSE** on patch reconstruction.
- Pretraining: **100 epochs**.

**Important implementation difference:** Self-supervised `L=512,P=12` produces 42 usable non-overlapping patches; do **not** mechanically reuse the supervised end-padding `+2` patch-count rule. Exact handling of the final remainder should be verified in the official implementation if perfect reconstruction is needed.

## 19. Self-supervised computation graph `[SOURCE, Fig.1c, p.4]`

```text
One univariate channel history [L=512]
      ↓
Instance normalization
      ↓
Non-overlapping patch extraction [N=42,P=12]
      ↓
Sample patch-mask indices (about 40%)
      ↓
Set selected full patches to zero
      ↓
Shared patch projection + position embeddings
      ↓
Shared Transformer encoder
      ↓
Linear reconstruction head D→P per patch
      ↓
Reconstruct ONLY masked patches with MSE
```

The reconstruction head is **not the same** as the supervised flattened horizon head.

## 20. Masked objective `[DERIVATION reflecting SOURCE]`

Let `Ω_i` be the set of masked patch positions for channel `i`. An explicit loss consistent with the described task is:

$$
\mathcal L_{mask}
=\frac{1}{\sum_i|\Omega_i|}
\sum_{i=1}^{M}\sum_{n\in\Omega_i}
\bigl\|\hat X^{(i)}_{p,n}-X^{(i)}_{p,n}\bigr\|_2^2.
$$

This formula is **our explicit mathematical rendering of the textual description**, not a numbered equation in the source.

## 21. Linear probing `[SOURCE, §4.2, p.6]`

After pretraining:

1. **Freeze** the pretrained Transformer backbone.
2. Replace masked reconstruction head with a supervised forecasting head.
3. Train only the new head for **20 epochs**.

Purpose: test whether representation alone carries usable downstream information without updating its feature extractor.

## 22. Fine-tuning `[SOURCE, §4.2, p.6]`

The original protocol:

1. train the newly attached linear head alone for **10 epochs**;
2. unfreeze the whole model;
3. jointly fine-tune for **20 more epochs**.

This is deliberately a two-stage downstream adaptation; it is not simply full-model tuning from the first batch.

## 23. Cross-dataset transfer `[SOURCE, §4.2; Tables 5–6, pp.8–9]`

The paper conducts more than one transfer test:

- **Table 5:** pretrain on **Electricity**, fine-tune on other datasets including Weather and Traffic.
- **Table 6:** pretrain on **Traffic**, transfer representation to **ETTh1** and compare with self-supervised learning directly on ETTh1 and prior representation-learning methods.

Because every channel shares a common backbone, source and target datasets **can have different numbers of variables**. This is a meaningful transfer advantage relative to a model with a fixed multivariate input projection.

## 24. Critical pretraining-versus-supervision distinction `[CRITIQUE]`

The paper documents that masked pretraining improves several large-data settings. It does **not** establish universal dominance: some datasets/horizons and transfer cases show fine-tuning that is only comparable to or slightly worse than supervised-from-scratch. Read Table 4/5 cells, not just the abstract's aggregate narrative.

**For future Agent experiments:** evaluate masked pretraining only using data from the permitted training period and validate which unlabeled historical windows are legally available. Self-supervision does not automatically make future test-period inputs acceptable during model development.

---
# Part IV — Empirical Study: Datasets, Baselines, and Main Results

## 25. What exactly is experimentally evaluated? `[SOURCE, §4, pp.6–9]`

The original paper investigates **long-term multivariate forecasting**, not stock selection. Supervised predictors forecast continuous future values, evaluated with:

$$
\mathrm{MSE}=\frac1Q\sum_{q=1}^{Q}(\hat y_q-y_q)^2,
$$

$$
\mathrm{MAE}=\frac1Q\sum_{q=1}^{Q}|\hat y_q-y_q|.
$$

Here the detailed scalar formulas merely restate the conventional metric definitions; `Q` counts the evaluated forecast entries.

**Neither MSE nor MAE is cross-sectional RankIC.** A low forecast MSE for electricity or traffic cannot be taken as evidence of positive stock return ranking skill.

## 26. Datasets — complete Table 2 `[SOURCE, p.6]`

| Dataset | Channels / features `M` | Timesteps | Domain / cadence |
|---|---:|---:|---|
| Weather | 21 | 52,696 | Meteorological variables, Germany |
| Traffic | 862 | 17,544 | California freeway sensor occupancy |
| Electricity | 321 | 26,304 | Hourly customer electricity consumption |
| ILI | 7 | 966 | Weekly influenza-like illness statistics |
| ETTh1 | 7 | 17,420 | Electricity-transformer temperature, hourly |
| ETTh2 | 7 | 17,420 | Electricity-transformer temperature, hourly |
| ETTm1 | 7 | 69,680 | Electricity-transformer temperature, 15-minute |
| ETTm2 | 7 | 69,680 | Electricity-transformer temperature, 15-minute |

The same eight benchmark families are used for supervised forecasting and representation learning. Dataset properties vary widely, especially in `M` and number of training time steps. That variation matters when interpreting the benefit of CI.

## 27. Data split and chronology — exact information boundary `[SOURCE/CRITIQUE]`

The supplied PDF relies on standard time-series benchmark datasets and experimental setups, but **does not provide a unified explicit table of exact train/validation/test date boundaries for every dataset**.

- **Supervised learning:** time-series forecasting evaluated on standard predefined tasks; baseline results partly collected from prior work and partly re-run.
- **Self-supervised learning:** pretraining followed by linear probing / fine-tuning.
- **Cross-dataset transfer:** pretraining dataset and downstream dataset specified by experiment (e.g. Electricity→other datasets, Traffic→ETTh1).
- **Model selection:** Appendix A.7 says best trained models are chosen using validation data.

**For a reproducibility-oriented Agent:** do not invent train start/end dates, validation duration, or test ranges from unrelated benchmark repositories. To reproduce exactly, inspect the author's released configurations/dataset loader and write those boundaries into a local reproduction-specific protocol file.

## 28. A carefully considered omission: Exchange-rate `[SOURCE, Appendix A.1.1, p.13]`

The authors discuss an exchange-rate dataset often included in earlier time-series forecasting benchmarks but **choose not to include it**. Their reasoning is domain-aware:

1. Financial exchange rates differ substantially from many regular physical/electricity sequences.
2. Under efficient markets, persistence/random-walk baselines can be extremely difficult to outperform on next-value squared error.
3. A simple last-value baseline can equal or outperform sophisticated forecasters in exchange-rate MSE.
4. Comparing generic forecasting models on finance requires a baseline suited to financial unpredictability.

This omission is important for a financial research Agent: the paper does not claim it solves asset-price forecasting, and the authors themselves acknowledge limits to transferring standard MSE benchmarks into finance.

## 29. Forecast horizons `[SOURCE]`

For most datasets:

$$
T\in\{96,192,336,720\}.
$$

For weekly ILI:

$$
T\in\{24,36,48,60\}.
$$

Each task specifies an input look-back `L` and prediction horizon `T`. The paper reports different look-back choices across model classes and actively tests their influence.

## 30. Main baseline set `[SOURCE, §4.1]`

| Baseline | Family | Why included |
|---|---|---|
| DLinear | Linear decomposition-based forecaster | Strong simple challenge to Transformers |
| FEDformer | Fourier/frequency-enhanced Transformer | Powerful frequency-centric competitor |
| Autoformer | Auto-correlation / decomposition Transformer | Temporal-decomposition competitor |
| Informer | ProbSparse attention Transformer | Efficient long-sequence competitor |
| Pyraformer | Pyramidal attention | Hierarchical long-range mechanism |
| LogTrans | LogSparse/convolutional attention | Earlier efficient temporal Transformer |

The paper does not include LSTM/TCN/DeepAR in this main supervised table, explaining in the appendix that previous long-horizon benchmark studies favored Transformer variants for these tasks. This is a **baseline-selection choice**, not a proof that LSTM/TCN are categorically inferior in all domains.

## 31. Baseline fairness: look-back treatment `[SOURCE/CRITIQUE]`

Original defaults mentioned in the paper:

- earlier Transformer-based baselines often used `L=96`;
- DLinear commonly used `L=336`;
- ILI uses shorter special look-backs.

Authors acknowledge this can underestimate Transformer competitors, so they re-run **FEDformer, Autoformer, and Informer** over six look-backs and retain their best results for each forecasting task.

Ordinary datasets' grid:

$$
L\in\{24,48,96,192,336,720\}.
$$

ILI grid:

$$
L\in\{24,36,48,60,104,144\}.
$$

**Critique:** This strengthens selected Transformer baselines, but does not mean every method is trained with identical input histories, identical architecture widths, or identical tuning budgets. In particular, PatchTST/64 uses 512 history points whereas many earlier architectures do not. PatchTST/42 (336 points) is the more directly comparable input-window variant relative to DLinear.

## 32. Supervised PatchTST variants `[SOURCE]`

| Model | Look-back | Patch length | Stride | Tokens |
|---|---:|---:|---:|---:|
| PatchTST/42 | 336 | 16 | 8 | 42 |
| PatchTST/64 | 512 | 16 | 8 | 64 |

Both use **channel independence** and the same high-level encoder/head idea. More tokens here are derived from a **longer input history**, not from finer patch size or more layers.

## 33. Original overall main-result statement `[SOURCE, §4.1, p.6]`

Compared with the strongest reported Transformer-based baseline results across the specified benchmark suite, the authors report aggregate reductions:

| Comparison | MSE reduction | MAE reduction |
|---|---:|---:|
| PatchTST/64 vs. best Transformer-based results | **21.0%** | **16.7%** |
| PatchTST/42 vs. best Transformer-based results | **20.2%** | **16.4%** |

These are **paper-reported aggregate summaries**, not independently recomputed confidence intervals or significance tests. See the complete reproduced Table 3 in the numerical appendix of this MD for actual per-dataset, per-horizon cells.

## 34. Selected Table 3 examples `[SOURCE, Table 3, p.7]`

Each cell below is `MSE / MAE`; lower is better.

| Dataset | Horizon | PatchTST/64 | PatchTST/42 | DLinear | FEDformer | Autoformer |
|---|---:|---|---|---|---|---|
| Weather | 96 | **0.149 / 0.198** | 0.152 / 0.199 | 0.176 / 0.237 | 0.238 / 0.314 | 0.249 / 0.329 |
| Traffic | 96 | **0.360 / 0.249** | 0.367 / 0.251 | 0.410 / 0.282 | 0.576 / 0.359 | 0.597 / 0.371 |
| Electricity | 96 | **0.129 / 0.222** | 0.130 / 0.222 | 0.140 / 0.237 | 0.186 / 0.302 | 0.196 / 0.313 |
| ILI | 24 | **1.319 / 0.754** | 1.522 / 0.814 | 2.215 / 1.081 | 2.624 / 1.095 | 2.906 / 1.182 |
| ETTh1 | 192 | 0.413 / 0.429 | 0.414 / 0.421 | **0.405 / 0.416** | 0.423 / 0.446 | 0.456 / 0.457 |
| ETTm1 | 720 | **0.416 / 0.420** | 0.420 / 0.424 | 0.425 / 0.421 | 0.446 / 0.458 | 0.527 / 0.493 |
| ETTm2 | 720 | **0.362 / 0.385** | 0.367 / 0.385 | 0.397 / 0.421 | 0.410 / 0.420 | 0.414 / 0.419 |

**Nuance:** ETTh1 horizon 192 is a concrete case where DLinear's MSE is better than both PatchTST variants. Main paper performance is strong **on average**, not an everywhere-dominance theorem.

## 35. Traffic case study from Table 1 `[SOURCE, Table 1, p.2]`

The authors intentionally compare patching, longer histories, and token count on 862-channel Traffic (`T=96` forecast):

| Model / setup | Look-back `L` | Tokens `N` | MSE |
|---|---:|---:|---:|
| CI timestamp-token Transformer | 96 | 96 | 0.518 |
| CI downsampled longer history | 380 | 96 | 0.447 |
| CI timestamp-token Transformer | 336 | 336 | 0.397 |
| PatchTST (supervised) | 336 | 42 | 0.367 |
| PatchTST (self-supervised setting) | 336 | 42 | 0.349 |
| FEDformer (channel-mixing) | 336 | 336 | 0.597 |
| DLinear | 336 | 336 | 0.410 |

The self-supervised `.349` entry is specifically this case-study setup; it is not necessarily identical to the Table 4 Transfer/Fine-tuning row, which uses other explicit settings. **Keep those experiment contexts separate.**

## 36. Running-time evidence `[SOURCE, Table 1, p.2]`

At look-back `L=336`, patch vs. non-patch wall-clock training time in the Table 1 setup:

| Dataset | With patches (s) | Without patches (s) | Reported gain |
|---|---:|---:|---:|
| Traffic | 464 | 10,040 | ×22 |
| Electricity | 300 | 5,730 | ×19 |
| Weather | 156 | 680 | ×4 |

Interpretation:

- Benefit grows with dataset/channel and token interaction cost.
- A theoretical attention-map reduction does not force one universal speedup ratio.
- These measurements are **original experiment-dependent** and should not be quoted as expected speedups for short financial inputs.

## 37. Look-back scaling `[SOURCE, §4.3, Fig.2, p.9; Appendix A.4.2/Table 9]`

The authors vary:

$$
L\in\{24,48,96,192,336,720\}
$$

with main qualitative plots for Weather, Traffic, Electricity and horizons 96/720. PatchTST generally benefits from longer look-back, whereas some competing Transformer methods do not.

For example, Traffic horizon 96, PatchTST/42-style varying input:

| Look-back `L` | MSE | MAE |
|---:|---:|---:|
| 24 | 0.766 | 0.419 |
| 48 | 0.671 | 0.381 |
| 96 | 0.477 | 0.305 |
| 192 | 0.401 | 0.267 |
| 336 | 0.367 | 0.251 |
| 720 | 0.365 | 0.251 |

**Observation:** Improvement becomes smaller after reaching a sufficiently long context. Longer history is not always unconditionally better, especially across dataset and horizon changes.

## 38. Patch length sensitivity `[SOURCE, Appendix A.4.1, p.15]`

The source body studies patch-length choices including:

$$
P\in\{4,8,16,24,32,40\},
$$

with Figure 4 caption also displaying a grid containing `2` and `12`. The plot uses look-back `336`, forecast horizon `96`, and (for that study) stride equal to patch length, i.e. **non-overlapping** patches.

Takeaway stated by authors:

- Performance changes relatively modestly over multiple patch sizes.
- `P` around **8–16** is often a useful general range in those datasets.
- A suitable patch size can be dataset-specific.

**Document-level specification nuance:** The descriptive paragraph and figure caption enumerate slightly different candidate grids. Preserve this discrepancy instead of pretending the PDF reports one perfectly identical sweep list.

## 39. Supervised hyperparameters `[SOURCE, Appendix A.1.4–A.1.5, p.14]`

**Default large-data configuration**:

| Parameter | Value |
|---|---|
| Transformer encoder layers | 3 |
| Attention heads | 16 |
| Hidden/embedding dimension `D` | 128 |
| FFN dimension `F` | 256 |
| FFN activation | GELU |
| Encoder dropout | 0.2 |
| Canonical supervised patch `P` | 16 |
| Canonical supervised stride `S` | 8 |
| Look-back `L` | 336 (`/42`), 512 (`/64`) |
| Normalization | instance normalization |
| Attention-block norm | BatchNorm |
| Forecast head | Flatten + Linear |
| Forecast objective | MSE |

For the **smaller datasets ILI, ETTh1, ETTh2**, use a reduced model:

- heads `H=4`;
- embedding `D=16`;
- FFN `F=128`;
- still 3 encoder layers by default unless a code-specific option overrides it.

## 40. Training and optimizer provenance `[SOURCE/CRITIQUE]`

The exact supervised optimizer, learning rate, batch size, scheduler, patience, every seed and complete dataset-wise epoch protocol are **not all specified together** in the supplied PDF.

The appendix does specify:

- model architecture widths and default dropout;
- selected look-backs and patch configuration;
- 20-epoch reduction for certain large-data ablation comparisons because unpatched channel-independent variants are extremely expensive;
- fixed seed `2021` for most main results;
- separate five-seed robustness experiment (seeds 2019–2023).

**Do not import learning rates or epochs from another PatchTST implementation and label them as paper facts.** Use official configurations when building a faithful reimplementation.

---

# Part V — Self-Supervision, Transfer, and Robustness Results

## 41. Self-supervised comparison protocol `[SOURCE, §4.2, pp.6–8]`

The authors pretrain masked-patch PatchTST on each dataset for 100 epochs, then compare:

1. **Supervised from scratch:** no pretraining.
2. **Linear probing:** pretrained backbone frozen; head trained 20 epochs.
3. **Fine-tuning:** pretrained backbone; head-only 10 epochs then whole model 20 epochs.

The main claim is improved representation reuse and performance with less downstream supervised training.

## 42. Selected Table 4 results `[SOURCE, Table 4, p.7]`

Cell format `MSE / MAE`; all are long-term forecast evaluations, not trading outcomes.

| Dataset | Horizon | Fine-tuning | Linear probing | Supervised from scratch | DLinear |
|---|---:|---|---|---|---|
| Weather | 96 | **0.144 / 0.193** | 0.158 / 0.209 | 0.152 / 0.199 | 0.176 / 0.237 |
| Traffic | 96 | **0.352 / 0.244** | 0.399 / 0.294 | 0.367 / 0.251 | 0.410 / 0.282 |
| Electricity | 96 | **0.126 / 0.221** | 0.138 / 0.237 | 0.130 / 0.222 | 0.140 / 0.237 |
| ETTh1 | 192 | 0.431 / 0.443 | **0.411 / 0.428** | 0.414 / 0.421 | **0.405 / 0.416** |
| ETTh2 | 96 | 0.285 / 0.345 | 0.280 / 0.341 | **0.274 / 0.334** | 0.289 / 0.353 |

**Interpretation:** The fine-tuned model is often strongest on large datasets. On individual small-data cases, pretraining is not guaranteed to beat straightforward supervision or DLinear. This limits blanket claims.

## 43. Cross-dataset transfer results `[SOURCE, Table 5, p.8]`

Pretrain on **Electricity**; transfer/fine-tune to downstream Weather/Traffic. The Table 5 numbers are:

| Dataset | Horizon | Transferred FT | Linear probing | Same-dataset supervised |
|---|---:|---|---|---|
| Weather | 96 | 0.145 / 0.195 | 0.163 / 0.216 | 0.152 / 0.199 |
| Weather | 192 | 0.193 / 0.243 | 0.205 / 0.252 | 0.197 / 0.243 |
| Weather | 336 | 0.244 / 0.280 | 0.253 / 0.289 | 0.249 / 0.283 |
| Weather | 720 | 0.321 / 0.337 | 0.320 / 0.335 | 0.320 / 0.335 |
| Traffic | 96 | 0.388 / 0.273 | 0.400 / 0.288 | 0.367 / 0.251 |
| Traffic | 192 | 0.400 / 0.277 | 0.412 / 0.293 | 0.385 / 0.259 |
| Traffic | 336 | 0.408 / 0.280 | 0.425 / 0.307 | 0.398 / 0.265 |
| Traffic | 720 | 0.447 / 0.310 | 0.457 / 0.317 | 0.434 / 0.287 |

**Critical nuance:** Electricity→Traffic transfer underperforms supervised PatchTST on all four listed Traffic horizons. Nevertheless transferred performance often remains competitive with non-PatchTST baselines. Transferability is real, but not universally better than in-domain training.

## 44. Other representation-learning baselines `[SOURCE, Table 6, p.8]`

The authors compare transferred/self-supervised PatchTST with:

- BTSF;
- TS2Vec;
- TNC;
- TS-TCC.

Table 6 focuses on ETTh1 representation learning and **linear probing** for comparable downstream use.

| Horizon | PatchTST transferred MSE | PatchTST within-dataset SSL MSE | BTSF MSE | TS2Vec MSE | TNC MSE | TS-TCC MSE |
|---:|---:|---:|---:|---:|---:|---:|
| 24 | 0.312 | 0.322 | 0.541 | 0.599 | 0.632 | 0.653 |
| 48 | 0.339 | 0.354 | 0.613 | 0.629 | 0.705 | 0.720 |
| 168 | 0.424 | 0.419 | 0.640 | 0.755 | 1.097 | 1.129 |
| 336 | 0.472 | 0.445 | 0.864 | 0.907 | 1.454 | 1.492 |
| 720 | 0.508 | 0.487 | 0.993 | 1.048 | 1.604 | 1.603 |

The Table 6 source reports improvement ranges of **34.5%–48.8%** against the best listed competitors (metric/task dependent), not one universal uplift.

## 45. Multi-seed robustness `[SOURCE, Appendix A.6.1, pp.18–20; Table 14]`

**Unusually important scientific-reporting fact:**

- Most headline main-text and earlier appendix benchmark tables were produced with **fixed random seed 2021**.
- To assess robustness separately, the supervised model is rerun with **five seeds**:
  
  `2019, 2020, 2021, 2022, 2023`.
- For self-supervised robustness, authors **pretrain the model once** and then conduct **five fine-tuning runs** with different random batches.

Therefore the self-supervised 5-run variation does **not** integrate randomness from five independently retrained pretraining backbones. This matters whenever using Table 14 to argue total training-pipeline stability.

Selected Table 14 examples (mean ± standard deviation):

| Dataset | Horizon | Supervised MSE | Self-supervised / FT MSE |
|---|---:|---:|---:|
| Weather | 96 | 0.1525 ± 0.0024 | 0.1450 ± 0.0008 |
| Traffic | 96 | 0.3669 ± 0.0006 | 0.3528 ± 0.0022 |
| Electricity | 96 | 0.1304 ± 0.0006 | 0.1256 ± 0.0002 |
| ETTh1 | 96 | 0.3752 ± 0.0008 | 0.3700 ± 0.0035 |
| ETTh2 | 96 | 0.2749 ± 0.0005 | 0.2869 ± 0.0039 |

Notice the selected ETTh2 case where self-supervised MSE is worse, reinforcing the importance of dataset-conditional conclusions.

## 46. Model-width / layer sensitivity `[SOURCE, Appendix A.6.2, Fig.5, p.20]`

The authors vary:

$$
\text{layers}\in\{3,4,5\},\qquad D\in\{128,256\},\qquad F=2D.
$$

Six configurations are studied on forecast horizon 96. The figure suggests relative robustness on many datasets, with higher variation on small ILI.

**Implication:** Much of the model's value lies in patching/CI design rather than exquisite tuning of one layer count / width configuration, at least within the tested range.

## 47. Instance normalization ablation `[SOURCE, Appendix A.4.4, Table 11]`

With/without instance norm for PatchTST/64 and /42:

- Weather 96: `/64` MSE 0.149 **with** norm vs. 0.161 **without**.
- Traffic 96: `/64` 0.360 vs. 0.413.
- Electricity 96: `/64` 0.129 vs. 0.133.
- ILI 24: `/64` 1.319 vs. 3.563 (large benefit in this small dataset).

Authors conclude normalized inputs help but that patching and CI are the main general advantages, because unnormalized variants still outperform many competing Transformer baselines in several tasks.

## 48. Performance/dataset caveat `[CRITIQUE]`

Paper measurements use:

- point-wise MSE / MAE;
- standard multivariate forecasting datasets;
- mainly long histories;
- selected fixed/random-seed protocols;
- a subset of baseline result sources external to this paper.

They do not directly test:

- ranking precision on volatile returns;
- portfolio risk / turnover;
- cross-market robustness of a transferred financial model;
- short 8-/20-day lookbacks;
- macro-regime shifts using finance-specific evaluation.

---

# Part VI — Ablations and Why Channel Independence Helps

## 49. Full 2×2 intervention design `[SOURCE, §4.3; Appendix A.4.3]`

The four versions used in Tables 7 and 10 are:

| Variant | Patching | Channel independence | Implementation concept |
|---|---|---|---|
| **P+CI** | yes | yes | Full PatchTST |
| CI | no | yes | Set patch length=1, stride=1; independent timestamp-level streams |
| P | yes | no | Patch channel-mixing design |
| Original | no | no | Traditional multivariate TST-style point-token model |

This clean 2×2 design is a particularly reusable *experimental methodology* for Agent-generated model innovations: isolate two mechanisms individually and jointly rather than only comparing a full model with one reduced variant.

## 50. Exact change for P-only `[SOURCE, Appendix A.4.3, p.15]`

The paper clarifies:

- CI version reshapes `[B,M,P,N]` to `[B·M,P,N]`.
- Channel-mixing with patches can reshape to `[B,M·P,N]`, mixing channels within the patch token feature dimension.

So **patching and CI are orthogonal implementation choices**; neither inherently requires the other.

## 51. Table 7 selected ablations `[SOURCE, p.9]`

Each result is `MSE / MAE` (lower is better).

| Dataset | Horizon | P+CI | CI-only | Patch-only | Neither (Original) | FEDformer |
|---|---:|---|---|---|---|---|
| Weather | 96 | **0.152 / 0.199** | 0.164 / 0.213 | 0.168 / 0.223 | 0.177 / 0.236 | 0.238 / 0.314 |
| Traffic | 96 | **0.367 / 0.251** | 0.397 / 0.271 | 0.595 / 0.376 | OOM | 0.576 / 0.359 |
| Electricity | 96 | **0.130 / 0.222** | 0.136 / 0.231 | 0.196 / 0.307 | 0.205 / 0.318 | 0.186 / 0.302 |
| ILI | 36 | **1.430 / 0.834** | 2.000 / 1.002 | 2.564 / 1.058 | 2.126 / 0.935 | 2.516 / 1.021 |

The `ILI` row is from full Appendix Table 10, not main Table 7.

The `-` entries in original tables denote **out of GPU memory**, not poor-but-measured accuracy. Main Table 7 explicitly cites NVIDIA A40 48GB even when batch size is reduced to 1.

## 52. How to interpret the 2×2 experiment `[DERIVATION/CRITIQUE]`

On Weather horizon 96 (MSE):

- Full P+CI: `0.152`.
- CI only: `0.164`, so adding patching to a CI backbone improves by `0.012`.
- P only: `0.168`, so adding CI to a patching backbone improves by `0.016`.
- Neither: `0.177`.

On Traffic horizon 96, P-only has a large gap (`0.595`) against P+CI (`0.367`), showing CI can be particularly important in datasets with many heterogeneous channels.

But **do not over-generalize** to all contexts: some small-dataset cells in Table 10 are mixed and the effect depends on horizon/optimization/resource limitations.

## 53. Channel independence: adaptability `[SOURCE, Appendix A.7.1, pp.21–23]`

Each independent channel pass creates its own attention weights:

$$
A^{(i)}=\operatorname{softmax}\left(
\frac{Q(x^{(i)})K(x^{(i)})^\top}{\sqrt{d_k}}
\right).
$$

Although parameters in Q and K networks are shared, input changes imply different attention patterns:

$$
A^{(i)}\neq A^{(j)}\quad\text{in general}.
$$

Figure 6 (PDF p.23) shows channel-specific attention maps and forecast trajectories on Electricity. Similar channel behavior can produce similar attention maps, while unrelated channels can use different patterns.

### Why that matters `[INTERPRETATION]`

An early all-channel token forces one shared time-token interaction geometry for several variables. CI allows **adaptive attention by input series** with a common learned temporal grammar.

## 54. Channel independence: sample efficiency `[SOURCE, Appendix A.7.1, Fig.7]`

Paper hypothesis: learning all cross-channel dependencies jointly can demand much more data. CI isolates temporal structure in each series, allowing the shared temporal model to learn from pooled channels.

Figure 7 (PDF p.24, left) compares test error as the fraction of training data increases. CI improves faster under limited data.

**Evidence scope:** one controlled analysis on Weather with the authors' configurations; generalization to finance is a hypothesis, not a demonstrated result.

## 55. Channel independence: avoiding overfitting `[SOURCE, Appendix A.7.1, Fig.7]`

Figure 7 (right) compares test loss across epochs:

- channel-mixing test error starts deteriorating after early iterations;
- channel-independent test error remains lower and trains more smoothly.

Authors interpret CI as a form of architectural regularization against premature high-dimensional channel interactions.

## 56. Channel independence: noise isolation `[SOURCE/INTERPRETATION]`

The appendix explicitly notes that channel mixing can spread noise from some channels into the joint embedding of all variables. CI restricts each channel's noise primarily to its own processing path.

This does **not** guarantee that a shared-parameter model is insensitive to corrupted-channel training examples; bad channels can still affect common weight updates. It only removes *direct same-forward channel mixing* before the later fusion stage.

## 57. Cross-channel relationships remain an open direction `[SOURCE, §5 and Appendix A.7]`

The conclusion explicitly says cross-channel dependencies should be modeled properly in future work. Appendix A.7 suggests channel-independent encoders can be extended with graph-based mechanisms for spatial relationships.

This is the natural opening for a hybrid architecture:

```text
Per-variable CI temporal encoder
      ↓
Grouped or learned channel relation module
      ↓
Cross-asset / portfolio representation
```

**Important:** This hybrid is our architectural extension; PatchTST's original model does not implement the shown late relation stage.

## 58. CI transferred to other Transformer architectures `[SOURCE, Appendix A.7.2, Table 15]`

The authors apply the same CI idea to:

- Informer;
- Autoformer;
- FEDformer.

Table 15 shows improvements in many settings, especially for Informer on Weather/ETT, but **not every dataset/horizon is improved**. Some large channel-independent configurations exceed memory/time budgets and receive `-`.

This supports CI as a potentially reusable architecture-level tool, but **not** as a universal plug-in improvement with guaranteed positive results.

## 59. A critical scientific caveat in the ablation protocol `[SOURCE, Appendix A.4.3, p.16]`

For some large datasets (Traffic and Electricity), default maximum epochs for ablation experiments are reduced **from 100 to 20** because the unpatched variants have extreme time/memory demands.

Consequences:

- A 2×2 ablation may compare variants that differ in convergence dynamics.
- The paper documents the choice; it should be retained in the knowledge base.
- A finance follow-up should consider compute-matched or convergence-matched controls where possible.

---
# Part VII — Inductive Bias, Mechanism-Level Interpretation, and Critical Assessment

## 60. Core inductive biases `[INTERPRETATION]`

PatchTST is built around at least five assumptions:

1. **Local temporal structure:** adjacent time points jointly describe a more meaningful local pattern than isolated scalars.
2. **Temporal redundancy:** adjacent points carry overlap in information, so a patch can replace multiple tokens without discarding all relevant content.
3. **Shared temporal vocabulary:** many different channels can benefit from one common parameterized temporal processing rule.
4. **Heterogeneous per-channel dynamics:** each channel should be allowed different attention activations and predictions even when using shared parameters.
5. **Useful unlabeled patterns:** reconstructing masked temporal patches can learn abstractions transferable to supervised forecasting.

Each is an inductive bias, not a universal truth about all datasets or asset classes.

## 61. Mechanism decomposition — atomic primitives

### Primitive M1: local patch tokenization `[SOURCE]`

$$
\underbrace{x_{t:t+P-1}}_{\text{local values}}
\longrightarrow
\underbrace{z_{patch,t}}_{\text{one learned token}}.
$$

**Purpose:** local semantic encoding + fewer tokens.

### Primitive M2: learned patch projection `[SOURCE]`

$$
z_{patch}=W_px_{patch}+b_p.
$$

**Purpose:** learn useful local motifs instead of fixed handcrafted statistics.

### Primitive M3: end-value repetition padding `[SOURCE]`

Append `S` copies of the last observed value before supervised `unfold`.

**Purpose:** include the latest history boundary within the patch grid; creates predictable patch count.

### Primitive M4: shared-weight channel independence `[SOURCE]`

$$
\hat y_i=f_\theta(x_i),\qquad \theta\text{ shared over }i.
$$

**Purpose:** avoid early variable mixing, control parameter count, learn channel-specific attention from channel-specific inputs.

### Primitive M5: shared attention, variable-specific activation `[SOURCE/DERIVATION]`

One parameterized attention function is applied to different series, producing different attention maps.

**Purpose:** flexible per-series temporal patterns without separate per-series networks.

### Primitive M6: instance-level normalization `[SOURCE]`

Normalize each channel history by statistics of that input instance, reverse the normalization after forecasting.

**Purpose:** ease learning across shifts in level/scale.

### Primitive M7: direct multi-horizon readout `[SOURCE]`

Flatten all patch states and linearly predict the entire future window.

**Purpose:** simple supervised decoding; modeling capacity resides primarily in encoder.

### Primitive M8: masked full-patch corruption `[SOURCE]`

Mask entire local segments, rather than isolated timestamps.

**Purpose:** avoid trivial point interpolation; promote pattern-level representation learning.

### Primitive M9: shared pretraining with later linear probe `[SOURCE]`

Unsupervised pretraining → supervised head training → optional end-to-end fine-tuning.

**Purpose:** exploit histories even where labels are expensive or not yet available.

### Primitive M10: channel-number-flexible transfer `[SOURCE]`

Pretrain on a dataset with `M_source` channels and fine-tune on a dataset with `M_target != M_source`, because channels are processed by shared weights rather than a fixed `M`-dependent embedding.

**Purpose:** portability across datasets with different variable counts.

## 62. Why patching may work, beyond runtime `[INTERPRETATION]`

A useful decomposition is:

1. **Representation:** a patch token can encode local slopes, reversals, plateaus, impulses and oscillations.
2. **Optimization:** token-count reduction lowers the size of the attention search space and associated gradient interactions.
3. **Inductive bias:** neighboring observations are strongly coupled; forcing local aggregation discourages treating every high-frequency fluctuation as independent evidence.
4. **Capacity reallocation:** available compute is spent attending over longer history of local patterns instead of over every scalar observation.

Not all four are isolated by the original experiments. The 2×2 ablation supports the **net effect** of patching, while individual causal pathways are hypotheses.

## 63. Why CI may work, beyond preventing parameter growth `[INTERPRETATION]`

The common intuition that a more expressive channel-mixing model must always beat CI ignores data scarcity and inductive bias.

- CI narrows the input hypothesis class to per-channel temporal functions.
- The shared network benefits from many channel instances as training examples.
- Channel-specific attention captures heterogeneous temporal patterns without requiring all variables to share one temporal attention map.
- Early mixing can blur an informative channel with noisy/unrelated ones.
- Reduced cross-variable interactions may improve finite-sample estimation even though it limits expressivity.

**Trade-off:** CI throws away direct cross-channel information unless another module restores it.

## 64. Two orthogonal issues: token axis versus interaction axis `[INTERPRETATION]`

PatchTST demonstrates why an Agent should separate:

| Design question | Possible choices | PatchTST's original choice |
|---|---|---|
| What is one token? | timestamp, patch, frequency block, learned segment | time patch |
| Which data axes interact? | within variable, across variables, across stocks, across time | within one variable's patches only |
| How are parameters shared? | independent networks, shared across variables, group-specific | fully shared channel backbone |
| How are outcomes decoded? | AR rollout, direct horizon head, ranking head | flatten + linear horizon head |
| How is representation learned? | supervised, masked reconstruction, contrastive | both supervised and masked reconstruction |

This factorization is useful when inventing finance-oriented hybrids: preserve the valuable tokenization while selectively altering interaction scope.

## 65. Limits of the original theoretical efficiency claim `[CRITIQUE]`

The reduction `O(L²)` to `O(N²)` describes the **attention portion per channel**. It does not mean:

- parameter count shrinks by `S²`;
- inference latency on a short 8-point signal improves `S²`;
- adding cross-stock attention remains cheap;
- input/patch projections, FFNs, output heads or channel reshaping become negligible;
- reduced token count automatically improves forecasting.

Measured running times in the original paper should be read under their specific hardware/data regimes.

## 66. What the 2×2 ablation does and does not isolate `[CRITIQUE]`

The P+CI vs P-only vs CI-only vs Original comparison isolates **design combinations**, but also changes:

- token count and computational graph;
- available parameter/embedding interactions;
- memory feasibility (some cells OOM);
- potential convergence speed and effective epoch budgets.

So an Agent can conclude that the full design works better under those conditions, but not that *all improvement is causally attributable to local semantics alone* or *to perfect noise disentanglement*.

## 67. Data / evaluation fairness critique `[CRITIQUE]`

Strengths:

- broad set of eight benchmarks;
- comparisons with DLinear and leading Transformer alternatives;
- patching×CI ablations;
- explicit look-back sensitivity;
- self-supervised and transferred representations;
- multiple random seeds and model-size robustness in appendix;
- candid discussion of exchange-rate caution and omitted financial benchmark.

Caveats:

- main results commonly single seed 2021;
- pretraining variance not fully represented by five downstream fine-tuning runs;
- different model families may use different look-backs;
- per-dataset split dates are not explicitly enumerated in the supplied text;
- selected OOM cases preclude numerical comparison;
- key studies focus on long history and long forecast, not ranking returns;
- no compute-matched modern cross-domain test inside the original paper.

## 68. Known non-results `[CRITIQUE]`

The PDF does not report:

- a systematic **finance** application;
- stock-market cross-sectional interaction results;
- multi-asset transaction costs or portfolio tests;
- factor exposure / alpha / beta interpretations;
- a learned VQ codebook;
- expert routing / sparse MoE;
- a theorem guaranteeing CI improves OOD generalization;
- a quantitative noise-injection robustness experiment on financial signals.

The Agent must not infer absent components just because later papers adapted similar ideas.

## 69. Failure conditions and transfer risks `[CRITIQUE/EXTENSION]`

| Condition | Why PatchTST may struggle | Suggested diagnostic |
|---|---|---|
| Very short history | too few patches to learn useful patch-to-patch attention | token-count and performance by `L,P,S` |
| Nonstationary asset dynamics | local motifs may not transfer across regimes | yearly/regime RankIC |
| Strong cross-variable causal coupling | CI ignores relationships needed for forecasting | CI vs early/late fusion ablation |
| Rare abrupt shocks | smoothing/compression can mute important new information | event-window performance |
| Mismatched channels | shared weights may impose one unsuitable prior across heterogeneous input types | group-specific encoder experiment |
| Trend/level conveys signal | instance normalization can remove useful scale-level cues | with/without norm |
| Weakly informative local motifs | patch tokens add overhead without signal | patch vs timestamp baseline |
| Need precise localization | coarse patch representation can obscure exact event timing | varying stride and overlap |
| Masked reconstruction too easy | network reconstructs local interpolation rather than predictive structure | longer masks, blocked masking, downstream improvement |
| Look-ahead in transfer | pretraining on future evaluation dates leaks distributional information | strict train-only pretraining |

The last entry is a methodological warning for predictive finance, not a problem identified in the original PatchTST paper's protocols.

---

# Part VIII — Cross-Paper Connections and Evolution of Methods

## 70. PatchTST compared with Transformer `[SOURCE/INTERPRETATION]`

PatchTST **reuses** vanilla Transformer blocks; it changes the representation and axis of attention.

| Topic | Generic time-point Transformer | PatchTST |
|---|---|---|
| Token | one timestamp vector | one local univariate patch |
| Channel mixing | often immediate | none inside primary backbone |
| Sequence length for attention | `L` | `N≈L/S` |
| Parameter sharing | architecture-dependent | shared across channels |
| Local semantics | learned across individual timestamps | built into each patch embedding |
| Transfer to changed channel count | may require new first layer | naturally supported by shared channel backbone |

**Agent use:** a novel tokenization choice can be more consequential than a novel attention formula.

## 71. PatchTST and ViT `[SOURCE/INTERPRETATION]`

The introduction explicitly connects patching in time series with visual patches in the Vision Transformer (ViT), but differs in axes and semantics:

- ViT tokenizes spatial image patches.
- PatchTST tokenizes **temporal segments of an individual time series channel**.
- Both emphasize local context as the token unit and shorten global interaction sequences.

A financial Agent can consider analogous patch units over:

- short temporal feature windows;
- groups of Alpha158 features;
- multi-scale time-frequency slices.

Only the first of these is close to the original paper's implementation.

## 72. PatchTST and DLinear `[SOURCE/CRITIQUE]`

DLinear is not merely a weak baseline in the paper. Its challenge is part of the original reason for revisiting Transformer forecasting design. PatchTST demonstrates that an intelligently chosen input representation can let a vanilla Transformer compete with strong linear methods.

Do **not** conflate that observation with a claim that longer model depth or complexity always improves finance prediction.

## 73. PatchTST and TimeMixer `[EXTENSION]`

TimeMixer-type mechanisms focus on **multiple resolutions and trend/seasonal decomposition**, whereas PatchTST emphasizes **local segment tokenization** and **channel independence**.

Potential orthogonal combination:

$$
\text{multi-scale decomposition}
\rightarrow
\text{patch tokens per scale}
\rightarrow
\text{cross-scale mixing}.
$$

This is a **new conceptual combination** for Agent hypothesis generation; it is not implemented or tested in this paper. Check latest related work before claiming novelty.

## 74. PatchTST and MATCC `[EXTENSION grounded by prior Baseline notes]`

MATCC's time pipeline decomposes the signal into trend and fluctuation, then performs within-stock temporal processing and cross-stock attention. PatchTST provides a different route to local temporal inductive bias.

Possible architecture:

```text
MATCC stock trend + residual
              ↓
Patch tokenization inside each stream
              ↓
Shared or stream-specific temporal encoder
              ↓
MATCC cross-stock attention
              ↓
Temporal/stock ranking head
```

**Research question:** Is temporal patching complementary to an already powerful trend-decomposition module, or merely redundant?

## 75. PatchTST and MASTER `[EXTENSION]`

MASTER preserves time-specific stock embeddings, then performs intra-stock and inter-stock aggregation. Patching could be integrated into its intra-stock module, **but** doing so changes its two-hop time-specific relay interpretation: patches, not daily timestamps, would become relays.

Potential issue:

$$
(stock_u,\text{day }t)
\rightarrow
(stock_v,\text{patch }p)
$$

is not the same as the original direct or factorized day-level relation. A model must define alignment between stocks with differing patch boundaries and handle temporal lag interpretation carefully.

## 76. PatchTST and FactorVAE `[EXTENSION]`

FactorVAE uses a temporal stock encoder before learning latent market factors and factor loadings. Replacing its GRU stock encoder with PatchTST-style local segment processing is technically possible, but alone may be a weak novelty claim. Better hypotheses might involve:

- patch-level posterior/latent supervision;
- different short/mid horizon latent factors;
- uncertainty-aware temporal patch weighting.

Preserve FactorVAE's distinction between training-time privileged future information and inference-time historical input.

## 77. PatchTST and FactorVQVAE / PRISM-VQ `[EXTENSION]`

These later financial models focus on **discrete latent structure** and conditional dynamics. A clear multi-level interpretation is:

| Dimension | PatchTST-style mechanism | VQ/MoE financial mechanism |
|---|---|---|
| Primitive represented | local time pattern | stock structural regime / latent factor |
| Quantization | absent | present |
| Encoder conditioning | within-channel temporal patterns | cross-section/market structure |
| Conditional compute | not present | possible through MoE/router |
| Output | future time-series values | stock scores/factor loadings |

Two possible migration paths:

1. **Patch before structural encoding:** replace/local-enhance the historical stock encoder, then learn VQ structural codes from resulting stock features.
2. **Patch inside Stage 2:** preserve existing structural codebook and modify only temporal-loading computation.

A strong research design should test where patching adds incremental information, rather than just adding more layers to PRISM-VQ.

## 78. Novelty decomposition `[SOURCE/CRITIQUE]`

**Already known before PatchTST:**

- Transformer self-attention;
- local patching in other domains such as ViT;
- instance normalization;
- MSE forecasting;
- masked reconstruction as a general SSL paradigm;
- channel-independent modeling in some non-Transformer contexts.

**Paper's real methodological contribution:**

- channel-independent **patch-level Transformer forecasting** with shared weights;
- an empirical demonstration that this combination improves long-horizon forecasting, scalability and representation transfer;
- an extension to masked-patch self-supervised time-series learning using the same backbone.

**Not sufficient novelty by itself in 2026:** simply replacing a GRU with a stock PatchTST, or concatenating PatchTST and an existing factor model, without solving a new finance-specific challenge or showing controlled incremental benefit.

---

# Part IX — Cross-Sectional Stock Adaptation: Carefully Scoped Opportunities

## 79. Mapping the axes before designing a financial adaptation `[EXTENSION]`

A typical financial input (single date as training instance):

$$
X_t\in\mathbb R^{N_t\times L\times C}
$$

where:

- `N_t`: contemporaneous number of stocks;
- `L`: days in historical window (often **8** or **20** in the discussed baselines);
- `C`: indicators such as **158** Alpha158 features;
- target: per-stock future return/ranking score.

This has **three meaningful axes** instead of the original multivariate sequence's `M × L`:

1. **stock axis** `N_t`;
2. **time axis** `L`;
3. **feature channel axis** `C`.

An Agent MUST specify what a **channel** is before copying PatchTST's Channel Independence into finance.

## 80. Three distinct interpretations of 'channel' in finance `[EXTENSION]`

### Interpretation A: each technical indicator is a channel

Within **one stock**, process each Alpha158 feature's temporal stream independently with shared temporal parameters, then aggregate feature-channel outputs into a stock embedding.

```text
stock_i [L, 158]
    ↓ split into 158 feature streams
[158,L] → shared patch encoder → [158,D]
    ↓ feature gating / attention
stock embedding [D]
    ↓ cross-stock interaction
prediction score
```

**Benefit:** isolates potentially noisy indicators and shares parameters.  
**Danger:** explicit interactions across price, volume, moving averages etc. are lost until later aggregation.

### Interpretation B: each stock is treated as a channel

Represent each stock as a time series of **derived scalar** values (or an already aggregated feature representation) and share temporal model parameters across stocks.

**Benefit:** scalable stockwise encoding and potential transfer across changing constituent counts.  
**Danger:** stock-specific multivariate feature vectors must be encoded first, and cross-stock relationships are suppressed unless restored by a later layer.

### Interpretation C: each feature group is a channel group

Group features by type (price trend, volatility, volume, momentum, market context) and apply a group-specific or shared encoder, followed by late group fusion.

**Benefit:** intermediate regularization between total independence and full mixing.  
**Danger:** group definitions introduce priors and may need careful ablation.

**Research preference:** A or C usually gives a cleaner first experimental hypothesis than blindly treating all `N_t×158` values as one time-series channel.

## 81. Original 16/8 patch setting is inappropriate for very short histories `[DERIVATION]`

Under the original supervised padding rule:

$$
N=\left\lfloor\frac{L-P}{S}\right\rfloor+2.
$$

Illustrative patch-count calculations:

| History `L` | Patch `P` | Stride `S` | Tokens `N` | Practical implication |
|---:|---:|---:|---:|---|
| 8 | 4 | 2 | 4 | enough for minimal patch interaction |
| 8 | 6 | 2 | 3 | short patch sequence |
| 20 | 4 | 2 | 10 | more temporal tokens |
| 20 | 5 | 2 | 9 | overlapping local motifs |
| 20 | 8 | 4 | 5 | coarse midscale patterns |
| 20 | 16 | 8 | 2 | nearly no meaningful patch-to-patch attention |

**Hard rule:** `P≤L`. For `P>L`, this source formula does not define the intended setup without redesign.

## 82. Why just 'replace GRU with PatchTST' may fail `[EXTENSION]`

In a `20`-day stock history, original PatchTST's `P=16,S=8` yields only two patch tokens. The core Transformer then cannot express rich long-range patch relations because it has almost no tokens to attend over.

Additionally:

- Alpha158 features already contain multi-horizon engineered statistics, including overlapping rolling windows;
- historical financial return signals are noisier than many physical-system trajectories;
- finance output is often **cross-sectional rank**, not a future sequence of each indicator;
- feature-specific scales and information levels differ;
- too many channels or premature CI can discard alpha-relevant interactions.

Therefore the transfer hypothesis needs a redesigned patch vocabulary, shorter patch sizes, feature grouping, or a longer raw input history—not a renamed generic backbone.

## 83. Finance-specific architecture proposal A: CI patch → feature fusion → cross-stock `[EXTENSION]`

```text
Qlib Alpha158: [Nstock,L,C]
       ↓
Feature/channel split: [Nstock,C,L]
       ↓
Shared patch embedding + Transformer over time
       ↓
Feature outputs: [Nstock,C,D]
       ↓
Feature importance aggregation (learned gate)
       ↓
Stock representation: [Nstock,D]
       ↓
Cross-stock attention / factor model
       ↓
Ranking score for each stock
```

**Novelty must come from a well-motivated adaptation**, e.g. regime-conditioned feature aggregation or cross-sectional sharing, not from simply running an ordinary PatchTST encoder.

## 84. Finance-specific architecture proposal B: patch before VQ `[EXTENSION]`

```text
Stock feature sequence
       ↓
Patch local temporal patterns
       ↓
Stock temporal embedding
       ↓
Cross-sectional contextualization
       ↓
Vector Quantization
       ↓
Codebook / factor loading system
```

**Hypothesis:** local motifs can improve the **quality and stability** of stock structural prototypes if they filter isolated short-term noise before quantization.

**Potential conflict:** a patch may suppress abrupt informative regime changes, hurting code adaptability. Measure code turnover, active-code ratio, per-regime RankIC.

## 85. Finance-specific architecture proposal C: structure-conditioned PatchTST `[EXTENSION]`

A PRISM-VQ-compatible modification would be:

$$
\text{stock history patches}
\xrightarrow{\text{structure-conditioned temporal encoder}}
h_{i,t},
$$

conditioned by:

$$
z_{q,i,t}.
$$

Potential mechanisms:

- prepend a VQ structure token to temporal patch tokens;
- modulation of patch projection by structural code;
- code-routed patch-level experts;
- shared expert for universal temporal patterns and specialized experts for code-specific motifs.

**Question to test:** Does stock structural conditioning change which local temporal motifs are predictive in distinct market environments?

## 86. Finance-specific proposal D: multi-scale patches with short look-backs `[EXTENSION]`

Instead of one `P`, use a small bank:

$$
P\in\{2,4,8\}
$$

at `L=20`, possibly with suitable strides, and learn scale selection from a market/regime context.

Comparison to TimeMixer-style multiscale mechanisms should be explicit: scale mixing and patch tokenization are separate ideas, and a fair novelty claim must show why their combination is distinct and useful.

## 87. Finance-specific proposal E: masked financial patch pretraining `[EXTENSION]`

Pretrain local stock temporal representations using masked patches from **historical training-period only**, then fine-tune on return ranking.

Candidate downstream stages:

- stock encoder for MASTER/MATCC;
- stock encoder before VQ in a factor model;
- temporal encoder in a code-conditioned MoE.

**Need to test:** does masking produce features useful for *future returns* or merely reconstruct highly redundant engineered indicators?

Controlled comparisons:

- randomly initialized encoder;
- masked patch pretraining;
- future-return supervised pretraining;
- full fine-tuning vs. linear probing;
- same compute budget where feasible.

## 88. Finance-specific proposal F: late cross-channel/cross-stock relation learning `[EXTENSION]`

The original authors explicitly identify proper modeling of cross-channel dependencies as future work.

A candidate financial design:

$$
\underbrace{\text{CI temporal patch encoder}}_{\text{avoid early noise mixing}}
\rightarrow
\underbrace{\text{relation learner}}_{\text{recover necessary cross-asset interactions}}
\rightarrow
\text{factor / ranking head}.
$$

The key is **selective, delayed interactions**, not permanent independence.

## 89. Most useful original writing/method lesson for a finance paper `[INTERPRETATION]`

The authors show how a low-complexity conceptual change can challenge sophisticated architectures:

> identify the wrong information granularity, change the unit of representation, test both quality and complexity, and validate its portability through pretraining.

For financial ML, the analogous high-value question may be:

> What is the correct **unit of representation** for a changing stock market: day, patch, event, regime, stock, industry, or factor prototype?

An Agent should experiment with representation granularity as a first-class design variable.

---

# Part X — Falsifiable Experiment Hooks for the Research Agent

## 90. General experimental rules `[EXTENSION]`

All experiments below are **proposed research**, not results from the original paper.

Before any model comparison:

- use a consistent feature/label construction;
- hold CSI300 and S&P500 splits fixed;
- match prediction horizons exactly (5-day versus next-day labels are not interchangeable);
- separate train/validation/test information chronologically;
- use seed-0 smoke tests before full multi-seed runs;
- compare parameter count, compute, memory and wall-clock time;
- report IC, ICIR, RankIC, RankICIR and identical cost-aware portfolio metrics;
- report monthly/quarterly/regime variability, not only aggregate score;
- record negative results and failure conditions.

## 91. E0 — Patch feasibility audit, before model training `[EXTENSION]`

**Hypothesis:** short-lookback patching is useful only where token count stays sufficient.

- **Intervention:** tabulate `[L,P,S] → N` and final tensor shapes on `L=8` and `L=20`.
- **Success:** meaningful `N` and no shape/mask errors.
- **Failure:** only 1–2 tokens, extreme duplication from repeated padding, invalid history alignment.
- **Effort:** minimal, code-level static check.

## 92. E1 — 2×2 patching × CI test on one baseline `[EXTENSION]`

Use original-paper factorial design on a fixed local stock encoder:

1. neither patching nor CI;
2. patch only;
3. CI only;
4. P+CI.

**Hypothesis:** P+CI improves out-of-sample RankIC stability beyond patching alone.

**Null/failure criterion:** gains disappear under 5 seeds, performance is worse after controlling added capacity, or performance deteriorates on S&P500.

## 93. E2 — Patch-scale sweep on short financial histories `[EXTENSION]`

At `L=20`, compare `P∈{2,4,5,8}` with valid strides (e.g. 1/2/4 depending on P); at `L=8`, compare `P∈{2,4}`.

Keep fixed:

- train/valid/test;
- feature set;
- prediction horizon;
- downstream architecture;
- backtest.

**Hypothesis:** there is a nontrivial local scale capturing short-horizon predictive motifs.

**Failure:** little/no robust ranking gain or high sensitivity to tiny hyperparameter changes.

## 94. E3 — Channel-independent vs. grouped features `[EXTENSION]`

Compare:

- full feature mixing;
- per-feature channel independence;
- group-wise independence (domain-informed groups);
- per-feature CI followed by late learned cross-feature attention.

**Hypothesis:** selective late fusion outperforms both extremes (fully independent and fully mixed).

**Diagnostics:** feature-group attribution, correlation among outputs, seed stability, robustness to random feature dropout.

## 95. E4 — Masked patch pretraining for stock representation `[EXTENSION]`

Train encoder on train-period unlabeled features using masked patches, then use identical supervised fine-tuning for two markets.

Evaluate:

- pretrain once + five downstream seeds;
- **five independent pretraining seeds** as a more stringent follow-up;
- linear probe vs. full fine-tuning;
- fraction of training labels.

**Hypothesis:** masked pretraining helps especially when supervised labels are scarce, but may be less valuable when Alpha158 already encodes local temporal patterns.

## 96. E5 — Patch encoder insertion into PRISM-VQ Stage 2 `[EXTENSION]`

Keep Stage 1 fixed. Change **only the temporal feature encoder** used in Stage 2 to a patch-based variant.

**Hypothesis:** better local time-pattern summary improves dynamic factor loadings without disturbing cross-sectional code quality.

**Ablations:** baseline Stage2 Transformer vs. patched Transformer; with/without structure token; fixed total parameter budget if possible.

## 97. E6 — Patch encoder insertion before financial VQ `[EXTENSION]`

Keep Stage 2 unchanged. Replace/augment only the historical stock encoding preceding Stage 1 cross-asset VQ.

**Hypothesis:** VQ codebooks trained from patch-aware stock features are more stable and predictive.

**Diagnostics:** code perplexity, dead-code ratio, code transition entropy, regime performance, RankIC.

**Failure:** improved reconstruction but no ranking/portfolio gain, or reduced code diversity.

## 98. E7 — Regime-dependent patch scale `[EXTENSION]`

Use historical market state to select scales or combine patch lengths.

**Hypothesis:** volatility regimes change useful temporal granularity; a fixed `P` is restrictive.

**Ablate:** fixed small patch; fixed large patch; learned mixture; regime-conditioned mixture.

**Required controls:** no future market-state leakage; similar compute; 5 seeds; regime-specific results.

## 99. E8 — Patch × decomposition synergy `[EXTENSION]`

Combine patching with MATCC-like trend/residual decomposition in four configurations:

1. neither;
2. patch only;
3. decomposition only;
4. both.

**Hypothesis:** decomposition separates temporal frequencies while patches capture within-scale motifs.

**Failure:** redundant mechanisms worsen performance or inflate capacity without incremental predictive benefit.

## 100. E9 — Late stock relation model after CI `[EXTENSION]`

Use a CI patch encoder per stock/feature, then add MASTER-style stock attention.

**Hypothesis:** delayed cross-stock mixing reduces early noise propagation while preserving interactions necessary for ranking.

**Ablate:** early mixing, late mixing, both, no mixing.

**Diagnostics:** market-state sensitivity, compute scaling, monthwise RankIC stability.

## 101. E10 — Evaluate the claim on a second market `[EXTENSION]`

A model that improves CSI300 alone may simply exploit one market's characteristics.

**Minimum bar:** ensure positive robust results or a clearly understood failure mode on S&P500, not just a CSI300 gain.

**Report:** comparable labels and splits; seed mean ± std; full ranking metrics and identical trading-cost protocol; qualitative/quantitative regime diagnostics.

## 102. E11 — Match tokenization gains against simpler alternatives `[EXTENSION]`

To distinguish patching from general smoothing/downsampling:

- strided sampling without learned patch projection;
- average/max pooling;
- 1D convolution with equivalent receptive field;
- grouped temporal MLP;
- learned patch projection (PatchTST-style).

**Hypothesis:** learned local patch representations outperform simpler compression at matched compute.

**Failure:** simple pooling performs equally well, suggesting novel patch machinery is unnecessary in short-window finance.

---

# Part XI — Paper Writing, Presentation, and Evidence Design

## 103. Introduction narrative `[SOURCE/INTERPRETATION]`

The paper uses a strong rhetorical progression:

1. Transformers have succeeded broadly.
2. But recent time-series results (DLinear) challenge their usefulness in this domain.
3. The apparent problem may not be attention itself, but **representation granularity** and **channel mixing**.
4. Propose two intentionally simple design changes.
5. Explain computational and statistical reasons each may help.
6. Offer an early convincing Traffic case study before introducing architecture equations.
7. Show supervised forecasting and masked-pretraining transfer as two payoffs.

**Writing lesson:** a meaningful contribution can arise from revisiting task assumptions instead of increasing architectural complexity.

## 104. Contribution claims tied to evidence `[SOURCE/INTERPRETATION]`

| Claim | Most relevant evidence | Limit |
|---|---|---|
| Patch reduces attention costs | complexity derivation + Table 1 timing | total model cost is not `S²` exact |
| Patch improves forecasting | Table 3 + P/CI ablation | not every cell wins |
| CI improves generalization | Table 7/10 + Fig.7 | domain/dataset dependent |
| Longer look-back becomes usable | Figure 2 + Table 9 | saturation and dataset effects |
| Learned representation is reusable | Table 4–6, transfer | some transfer cases worse |
| Results are not one lucky seed | Table 14 | self-supervision pretraining seed not fully varied |
| Feature/architecture insensitive | Appendix sensitivity | tested configurations limited |

This claim→evidence mapping is an especially useful template for Agent-authored empirical papers.

## 105. Figure-by-figure understanding `[SOURCE]`

| Figure | PDF page | What is displayed | What it is meant to prove |
|---|---:|---|---|
| Fig. 1(a) | 4 | M channels processed independently by shared Transformer | Channel-independent overall architecture |
| Fig. 1(b) | 4 | Instance normalization → patches → embedding → encoder → flatten/head | Supervised model components |
| Fig. 1(c) | 4 | masked-patch reconstruction head | Same backbone supports SSL |
| Fig. 2 | 9 | errors vs. look-back on Weather/Traffic/Electricity | capacity to exploit longer context |
| Fig. 3 (appendix) | 14 | forecast trajectories | qualitative forecast fit |
| Fig. 4 (appendix) | 15 | patch-length sensitivity | robustness of patch-size choice |
| Fig. 5 (appendix) | 20 | depth/width sensitivity | parameter robustness |
| Fig. 6 (appendix) | 23 | different channels' attention maps and forecasts | input-dependent attention patterns under CI |
| Fig. 7 (appendix) | 24 | test loss vs training size and epochs | data efficiency / reduced overfitting for CI |

The main Fig. 1 is essential to understand because it resolves the misconception that CI requires a separate model for every channel.

## 106. Table-by-table argument structure `[SOURCE]`

| Table | PDF page | Main role |
|---|---:|---|
| Table 1 | 2 | Traffic case study: patch count, MSE, runtime |
| Table 2 | 6 | dataset scale and number of channels |
| Table 3 | 7 | supervised forecasting vs. strong baselines |
| Table 4 | 7 | pretraining, linear probing, fine-tuning |
| Table 5 | 8 | Electricity pretraining → downstream transfer |
| Table 6 | 8 | ETTh1 representation learning vs. contrastive methods |
| Table 7 | 9 | patch × CI ablation on selected datasets |
| Table 8 | 15 | univariate forecast auxiliary benchmark |
| Table 9 | 16 | extended look-back sweep |
| Table 10 | 17 | full 2×2 patch × CI results |
| Table 11 | 18 | instance normalization ablation |
| Tables 12–13 | 19 | detailed SSL and transfer tables |
| Table 14 | 20 | multi-seed results |
| Table 15 | 22 | CI applied to Informer, Autoformer, FEDformer |

## 107. Important method-writing techniques `[INTERPRETATION]`

- Use **one core visual** (Figure 1) showing both supervised and self-supervised variants without conflating their heads.
- Present **notation with dimensions**, especially patch count and channels.
- Show the memory/computation reason for patching before claiming accuracy gains.
- Use a **2×2 factorial ablation**, not only full vs. without patch.
- Explicitly compare to the unusually strong simple baseline (DLinear).
- Put extensive sweeps in the appendix and retain their experimental conditions.
- Give a genuine **future-work direction** that addresses the method's trade-off (missing cross-channel dependencies).

## 108. Author-stated limitations and future work `[SOURCE, §5, p.9]`

The conclusion explicitly emphasizes:

1. **Cross-channel relationships remain incompletely modeled** by the channel-independent design.
2. Further work should **incorporate cross-channel correlations properly**, while benefiting from CI.
3. PatchTST could serve as a building block for future Transformer-based forecasting and time-series foundation models.
4. Patching is a simple transferable operation potentially usable in other architectures.

These are the primary explicit future directions. Do not attribute specific VQ, MoE, finance or regret-bound proposals to the original authors.

## 109. Additional limitations identified by an Agent `[CRITIQUE]`

- Single-patch-scale supervised configuration; multiscale context not a principal mechanism.
- Late learned forecast head depends on the chosen patch count; adapting to arbitrary history lengths needs careful embedding/head handling.
- CI does not explicitly capture cross-variable dependence inside the backbone.
- More favorable performance can depend on dataset size, forecast horizon and normalized input scale.
- Pretraining gains vary across datasets and do not always beat supervised training.
- Original demonstrations emphasize long smooth time-series datasets, not noisy finance and rankings.
- Extreme channel counts increase runtime despite parameter sharing.

These are **analytical critiques**, not a literal “limitations” list from the source.

## 110. Writing a future finance paper inspired by PatchTST `[EXTENSION]`

A strong research narrative would avoid generic claims such as "Patching improves stocks because it worked on Weather." Instead:

1. **Observed domain failure:** short-window financial indicators are noisy, market-dependent, and heterogeneous.
2. **Method gap:** point-wise or fully mixed encoding may entangle noisy channels too early; raw PatchTST ignores inter-stock interactions.
3. **Specific hypothesis:** learn local motifs with shared CI temporal processing, then explicitly and sparsely fuse feature/stock context.
4. **Concrete novelty:** new regime/structure-aware patch selection, relation fusion or factor coupling designed for financial constraints.
5. **Evidence:** proper factorial ablation; full financial protocols; test on both CSI300/SP500 and multiple market regimes.
6. **Economic assessment:** not only RankIC but realistic turnover, cost-aware Sharpe/drawdown and model complexity.

---

# Part XII — Reproducibility, Source Gaps, and Agent Instructions

## 111. Faithful reproduction checklist `[SOURCE/CRITIQUE]`

**Data:**

- Eight original datasets and their actual time frequencies.
- Use original released dataset/time splits (not all spelled out in this PDF).
- Forecast horizons `{96,192,336,720}` except ILI `{24,36,48,60}`.

**Supervised encoder:**

- `L=336` or `512`.
- `P=16`, `S=8`.
- End padding repeats last observation `S` times.
- Correct token count: `42` or `64`.
- Channel independence with **shared** embedding/Transformer.
- Learnable positional embedding.
- Transformer encoder with **BatchNorm**.
- Flatten + linear direct horizon head.
- Instance normalization before patching and reversing at output.
- MSE target.

**Backbone defaults:**

- 3 layers, 16 heads, D=128, F=256, dropout 0.2 for larger datasets.
- Reduced 4 heads/D=16/F=128 for ILI, ETTh1, ETTh2.

**Self-supervision:**

- `L=512`, `P=12`, non-overlapping `42` patches.
- Mask ratio `40%`.
- Mask value zero; reconstruct masked patches with MSE.
- 100-epoch pretraining.
- Linear probing 20 epochs or 10 epochs head + 20 epochs whole-network FT.

**Baselines/statistics:**

- Original main benchmark largely fixed seed `2021`.
- Appendix robustness uses seeds `2019..2023`.
- Pretrained once / 5 downstream fine-tuning runs in the stated SSL robustness design.
- Acknowledge some ablation configurations OOM even at batch 1, 48GB.
- Note 20-epoch limit in heavy selected ablation cases.

## 112. Facts NOT specified exactly in the supplied PDF `[SOURCE-gap]`

The PDF does **not** provide a single complete, directly executable specification for:

- exact per-dataset chronological split start/end dates;
- full learning rate / optimizer / scheduler grid for all supervised tasks;
- every actual minibatch setting;
- exact epsilon and variance convention in instance normalization;
- some dataset-specific epoch/early-stopping details;
- every implementation choice for data-loader/window sampling;
- complete hardware-specific runtime settings for every benchmark variant.

For an exact numerical reproduction, inspect released code/configs and label those extra details **CODE-VERIFIED**, not **SOURCE-PDF**.

## 113. Negative / mixed results worth retaining `[SOURCE/CRITIQUE]`

- DLinear outperforms PatchTST on selected individual Table 3 cells (e.g., ETTh1 `T=192` MSE).
- Electricity→Traffic transferred PatchTST is worse than in-domain supervised PatchTST in Table 5.
- On ETTh2 horizon 96, self-supervised fine-tuning can be worse than supervised-from-scratch.
- Some channel-mixing / unpatched variants fail to fit even 48GB GPU memory, so no accuracy is available.
- Channel independence is not always an improvement when applied to every other architecture (Table 15).
- The selected main results are often seed 2021 rather than 5-seed averages.
- CI's virtues are dataset/architecture dependent, not universal guarantees.

**Agent rule:** Negative findings are reusable research knowledge; never drop them because they complicate a tidy SOTA story.

## 114. Source-grounding convention for future retrieval

The document's conceptual sections contain original section/page markers. The numerical appendices that follow preserve important full paper tables in machine-readable compact form.

Do not use this MD as proof of a source claim unless it is tagged `[SOURCE]` or specifically linked to a figure/table/section in the source map. `[INTERPRETATION]`, `[CRITIQUE]` and `[EXTENSION]` are separate layers and may require new experiments.

## 115. Agent Research Instructions

When reading or reusing PatchTST:

1. **Remember `/42` and `/64` are patch TOKEN COUNTS, not patch lengths.**
2. **Keep supervised and self-supervised patching configurations distinct.**
3. **Keep Channel Independence's shared weights separate from channel-specific activations.**
4. **Do not treat CI as proof cross-channel correlations are unimportant.**
5. **Do not import ICLR forecasting MSE into Qlib stock RankIC comparisons.**
6. **Use the original 2×2 patch × CI study as a model for your own factorial ablations.**
7. **Record the original dataset/forecast length before comparing results across papers.**
8. **Do not claim all original main tables are five-seed means; appendix seed protocol differs.**
9. **Handle 8-/20-day stock histories as a separate problem; adjust patch length and stride.**
10. **Make 'channel' explicit: feature, stock, or feature-group axis in finance.**
11. **If adding cross-stock attention, test whether CI/late interaction are truly complementary.**
12. **If adding VQ or MoE, show a novel mechanism beyond mere module concatenation.**
13. **Keep original author future work separate from Agent-suggested financial extensions.**
14. **Verify source PDF or official code for any critical ambiguous hyperparameter or experimental protocol.**
15. **Report real out-of-sample rank metrics and cost-aware portfolios for financial proposals.**

## 116. Compact Agent Takeaways

| Question | Best concise answer |
|---|---|
| Why this paper? | Revisits **what counts as a Transformer token** and how multivariate channels should interact |
| Most important idea | Patch temporal subsequences + per-channel forward processing with a shared Transformer |
| Core advantage | Better long-context representation at lower attention cost; CI regularizes cross-variable modeling |
| Essential supervised patch parameters | `P=16, S=8, L=336/512, N=42/64` |
| Essential SSL patch parameters | `L=512, P=12, nonoverlap, 42 patches, mask 40%` |
| Original experimental target | Multivariate long-horizon sequence-value MSE/MAE |
| Most important original ablation | P+CI / CI-only / patch-only / neither |
| Most important originally stated open issue | How to model useful **cross-channel** interactions without losing CI's gains |
| Largest risk for local finance transfer | 8–20-step history is far shorter than original 336/512 |
| Best migration philosophy | CI/local temporal extraction first, selective feature/stock relation later |

## 117. Compact Method Graph

```text
                 PATCHTST (ICLR 2023)

              Multivariate history [M,L]
                           │
                           ▼
                 Split into M channels
                           │
               SAME weights for all channels
                           │
           ┌───────────────┴────────────────┐
           ▼                                ▼
    SUPERVISED                         SELF-SUPERVISED
           │                                │
     Instance norm                      Instance norm
           │                                │
  Overlapping patches                   Non-overlap patches
   P=16, stride=8                      P=12, 42 patches
           │                                │
           │                           Mask 40% of patches
           │                                │
           └─────────────┬──────────────────┘
                         ▼
          Shared patch projection + position
                         ▼
                 Shared Transformer
                         │
            ┌────────────┴───────────────┐
            ▼                            ▼
    Flatten + linear head       Per-patch reconstruction head
            ▼                            ▼
    Forecast all T steps          Masked patch MSE
            │                            │
            │                         Pretrain
            │                            │
            │                   Linear probe / fine-tune
            ▼                            ▼
      MSE and MAE                 Forecast MSE and MAE

        No intrinsic stock relation / VQ / MoE / rank loss
```

---

# Part XIII — Original Source Locator and Original-vs-Extension Boundary

## 118. Source location map (PDF page numbers)

| Original PDF page | Sections / evidence |
|---:|---|
| 1 | Title, abstract, introduction, two core designs |
| 2 | Table 1 Traffic case study, efficiency rationale |
| 3 | Related work, first part of model specification |
| 4 | Figure 1; channel independence, patch count, patch embeddings |
| 5 | Attention, BatchNorm, direct head, supervised loss, instance norm, masked-patch pretraining |
| 6 | Eight datasets, Table 2, baselines, forecast settings, SSL protocols |
| 7 | Full supervised Table 3; self-supervised Table 4 |
| 8 | Transfer Table 5; representation-learning Table 6; ablation motivation |
| 9 | P×CI Table 7, look-back Figure 2, explicit future work |
| 13 | Dataset provenance and finance-specific Exchange-rate benchmark caution; baseline setup |
| 14 | Model defaults, tensor reshape implementation, Figure 3 |
| 15 | Univariate Table 8, patch-length Figure 4, detailed ablation implementation |
| 16 | Full look-back Table 9; ablation budgets and normalization discussion |
| 17 | Full ablation Table 10 |
| 18 | Full normalization Table 11; seed-method description |
| 19 | Full SSL Table 12 and transfer Table 13 |
| 20 | Multi-seed Table 14 and dimension sensitivity Figure 5 |
| 21 | Detailed explanations of CI adaptability, data efficiency, noise/overfitting |
| 22 | Full CI-on-other-models Table 15 |
| 23 | Channel-specific attention visualization Figure 6 |
| 24 | CI data-volume and epoch-overfitting Figure 7 |

## 119. Bibliographic relationships worth Agent retrieval

Original paper explicitly relates to:

- Transformer (*Attention Is All You Need*);
- ViT (patch-based vision Transformer);
- DLinear (*Are Transformers Effective for Time Series Forecasting?*);
- Informer, Autoformer, FEDformer, Pyraformer, LogTrans;
- masked-autoencoder learning in NLP/CV;
- time-series self-supervised representations such as TS2Vec;
- reversible / instance normalization and distribution shift.

Additional Agent-side finance links (not source citations):

- MASTER / MATCC: intra-stock temporal representations + inter-stock interactions;
- FactorVAE: latent-factor posterior/prior learning;
- FactorVQVAE / PRISM-VQ: discrete codebooks, factors, and conditional computation;
- TimeMixer: decomposition and multi-scale mixing.

## 120. Ultra-compact retrieval summary

**PatchTST** (Nie et al., ICLR 2023) is a Transformer-based long-horizon time-series forecast method with two central innovations: **patching**, i.e. converting contiguous univariate temporal segments into trainable patch tokens, and **channel independence**, i.e. applying a shared Transformer to each multivariate channel separately without early cross-channel mixing. The supervised version instance-normalizes each input channel, adds `S` repeated end-values, extracts overlapping patches `P=16,S=8`, uses `N=floor((L-P)/S)+2` tokens (`N=42` at `L=336` or `N=64` at `L=512`), maps each patch to `D` dimensions, applies a vanilla Transformer encoder with BatchNorm, and predicts the full horizon with a Flatten+Linear head using MSE. A separate masked-patch self-supervised variant uses `L=512`, nonoverlapping `P=12`, 42 patches, 40% patch masking and MSE reconstruction; the learned backbone can be linearly probed, fine-tuned or transferred between datasets with different channel counts. Experiments cover eight non-financial multivariate benchmark datasets; the original authors report aggregate MSE/MAE improvements over contemporary Transformer baselines, substantial reductions in runtime in long-context cases, a full 2×2 patching×CI ablation, history length/normalization/parameter sensitivity and additional seed analyses. Important caveats are mixed individual results, OOM entries for unpatched alternatives, main-table versus multi-seed appendix differences, and lack of explicit finance-domain ranking tests. Authors' explicit future direction is to combine channel independence with properly learned cross-channel correlations. For cross-sectional stock prediction, treat PatchTST as a **local temporal representation primitive**, choose much shorter patches for 8-/20-day windows, distinguish stock vs. feature channels, and independently validate late feature/stock fusion plus real rank/backtest improvements.

---

# Part XIV — Full Numerical Evidence (machine-readable transcriptions)

> Data appendix generated from the user-provided PDF's **layout-aware text layer** and separately reconciled with table structure. Each numeric entry is a compact `MSE/MAE` cell unless specifically noted. `-` is reproduced as an unavailable result (commonly OOM/time budget), **not** zero. All data are from original long-horizon forecasting benchmarks, NOT financial returns.

## 121. Full Table 3 — supervised forecasting (32 task rows)

**Original:** Table 3, PDF p.7; eight models × eight datasets × four horizons. Columns show `MSE/MAE` for each method unless otherwise specified.

| Dataset | Horizon | PatchTST/64 | PatchTST/42 | DLinear | FEDformer | Autoformer | Informer | Pyraformer | LogTrans |
|---|---:|---|---|---|---|---|---|---|---|
| Weather | 96 | 0.149/0.198 | 0.152/0.199 | 0.176/0.237 | 0.238/0.314 | 0.249/0.329 | 0.354/0.405 | 0.896/0.556 | 0.458/0.490 |
| Weather | 192 | 0.194/0.241 | 0.197/0.243 | 0.220/0.282 | 0.275/0.329 | 0.325/0.370 | 0.419/0.434 | 0.622/0.624 | 0.658/0.589 |
| Weather | 336 | 0.245/0.282 | 0.249/0.283 | 0.265/0.319 | 0.339/0.377 | 0.351/0.391 | 0.583/0.543 | 0.739/0.753 | 0.797/0.652 |
| Weather | 720 | 0.314/0.334 | 0.320/0.335 | 0.323/0.362 | 0.389/0.409 | 0.415/0.426 | 0.916/0.705 | 1.004/0.934 | 0.869/0.675 |
| Traffic | 96 | 0.360/0.249 | 0.367/0.251 | 0.410/0.282 | 0.576/0.359 | 0.597/0.371 | 0.733/0.410 | 2.085/0.468 | 0.684/0.384 |
| Traffic | 192 | 0.379/0.256 | 0.385/0.259 | 0.423/0.287 | 0.610/0.380 | 0.607/0.382 | 0.777/0.435 | 0.867/0.467 | 0.685/0.390 |
| Traffic | 336 | 0.392/0.264 | 0.398/0.265 | 0.436/0.296 | 0.608/0.375 | 0.623/0.387 | 0.776/0.434 | 0.869/0.469 | 0.734/0.408 |
| Traffic | 720 | 0.432/0.286 | 0.434/0.287 | 0.466/0.315 | 0.621/0.375 | 0.639/0.395 | 0.827/0.466 | 0.881/0.473 | 0.717/0.396 |
| Electricity | 96 | 0.129/0.222 | 0.130/0.222 | 0.140/0.237 | 0.186/0.302 | 0.196/0.313 | 0.304/0.393 | 0.386/0.449 | 0.258/0.357 |
| Electricity | 192 | 0.147/0.240 | 0.148/0.240 | 0.153/0.249 | 0.197/0.311 | 0.211/0.324 | 0.327/0.417 | 0.386/0.443 | 0.266/0.368 |
| Electricity | 336 | 0.163/0.259 | 0.167/0.261 | 0.169/0.267 | 0.213/0.328 | 0.214/0.327 | 0.333/0.422 | 0.378/0.443 | 0.280/0.380 |
| Electricity | 720 | 0.197/0.290 | 0.202/0.291 | 0.203/0.301 | 0.233/0.344 | 0.236/0.342 | 0.351/0.427 | 0.376/0.445 | 0.283/0.376 |
| ILI | 24 | 1.319/0.754 | 1.522/0.814 | 2.215/1.081 | 2.624/1.095 | 2.906/1.182 | 4.657/1.449 | 1.420/2.012 | 4.480/1.444 |
| ILI | 36 | 1.579/0.870 | 1.430/0.834 | 1.963/0.963 | 2.516/1.021 | 2.585/1.038 | 4.650/1.463 | 7.394/2.031 | 4.799/1.467 |
| ILI | 48 | 1.553/0.815 | 1.673/0.854 | 2.130/1.024 | 2.505/1.041 | 3.024/1.145 | 5.004/1.542 | 7.551/2.057 | 4.800/1.468 |
| ILI | 60 | 1.470/0.788 | 1.529/0.862 | 2.368/1.096 | 2.742/1.122 | 2.761/1.114 | 5.071/1.543 | 7.662/2.100 | 5.278/1.560 |
| ETTh1 | 96 | 0.370/0.400 | 0.375/0.399 | 0.375/0.399 | 0.376/0.415 | 0.435/0.446 | 0.941/0.769 | 0.664/0.612 | 0.878/0.740 |
| ETTh1 | 192 | 0.413/0.429 | 0.414/0.421 | 0.405/0.416 | 0.423/0.446 | 0.456/0.457 | 1.007/0.786 | 0.790/0.681 | 1.037/0.824 |
| ETTh1 | 336 | 0.422/0.440 | 0.431/0.436 | 0.439/0.443 | 0.444/0.462 | 0.486/0.487 | 1.038/0.784 | 0.891/0.738 | 1.238/0.932 |
| ETTh1 | 720 | 0.447/0.468 | 0.449/0.466 | 0.472/0.490 | 0.469/0.492 | 0.515/0.517 | 1.144/0.857 | 0.963/0.782 | 1.135/0.852 |
| ETTh2 | 96 | 0.274/0.337 | 0.274/0.336 | 0.289/0.353 | 0.332/0.374 | 0.332/0.368 | 1.549/0.952 | 0.645/0.597 | 2.116/1.197 |
| ETTh2 | 192 | 0.341/0.382 | 0.339/0.379 | 0.383/0.418 | 0.407/0.446 | 0.426/0.434 | 3.792/1.542 | 0.788/0.683 | 4.315/1.635 |
| ETTh2 | 336 | 0.329/0.384 | 0.331/0.380 | 0.448/0.465 | 0.400/0.447 | 0.477/0.479 | 4.215/1.642 | 0.907/0.747 | 1.124/1.604 |
| ETTh2 | 720 | 0.379/0.422 | 0.379/0.422 | 0.605/0.551 | 0.412/0.469 | 0.453/0.490 | 3.656/1.619 | 0.963/0.783 | 3.188/1.540 |
| ETTm1 | 96 | 0.293/0.346 | 0.290/0.342 | 0.299/0.343 | 0.326/0.390 | 0.510/0.492 | 0.626/0.560 | 0.543/0.510 | 0.600/0.546 |
| ETTm1 | 192 | 0.333/0.370 | 0.332/0.369 | 0.335/0.365 | 0.365/0.415 | 0.514/0.495 | 0.725/0.619 | 0.557/0.537 | 0.837/0.700 |
| ETTm1 | 336 | 0.369/0.392 | 0.366/0.392 | 0.369/0.386 | 0.392/0.425 | 0.510/0.492 | 1.005/0.741 | 0.754/0.655 | 1.124/0.832 |
| ETTm1 | 720 | 0.416/0.420 | 0.420/0.424 | 0.425/0.421 | 0.446/0.458 | 0.527/0.493 | 1.133/0.845 | 0.908/0.724 | 1.153/0.820 |
| ETTm2 | 96 | 0.166/0.256 | 0.165/0.255 | 0.167/0.260 | 0.180/0.271 | 0.205/0.293 | 0.355/0.462 | 0.435/0.507 | 0.768/0.642 |
| ETTm2 | 192 | 0.223/0.296 | 0.220/0.292 | 0.224/0.303 | 0.252/0.318 | 0.278/0.336 | 0.595/0.586 | 0.730/0.673 | 0.989/0.757 |
| ETTm2 | 336 | 0.274/0.329 | 0.278/0.329 | 0.281/0.342 | 0.324/0.364 | 0.343/0.379 | 1.270/0.871 | 1.201/0.845 | 1.334/0.872 |
| ETTm2 | 720 | 0.362/0.385 | 0.367/0.385 | 0.397/0.421 | 0.410/0.420 | 0.414/0.419 | 3.001/1.267 | 3.625/1.451 | 3.048/1.328 |

**Interpretation caution:** values retain their original long-term forecasting task/horizon; missing values do not imply zero error.


## 122. Full Table 4 — masked pretraining and downstream supervision (12 rows)

**Original:** Table 4, PDF p.7; SSL fine-tuning versus linear probing and supervised baselines. Columns show `MSE/MAE` for each method unless otherwise specified.

| Dataset | Horizon | PatchTST-FT | PatchTST-linear-probe | PatchTST-supervised | DLinear | FEDformer | Autoformer | Informer |
|---|---:|---|---|---|---|---|---|---|
| Weather | 96 | 0.144/0.193 | 0.158/0.209 | 0.152/0.199 | 0.176/0.237 | 0.238/0.314 | 0.249/0.329 | 0.354/0.405 |
| Weather | 192 | 0.190/0.236 | 0.203/0.249 | 0.197/0.243 | 0.220/0.282 | 0.275/0.329 | 0.325/0.370 | 0.419/0.434 |
| Weather | 336 | 0.244/0.280 | 0.251/0.285 | 0.249/0.283 | 0.265/0.319 | 0.339/0.377 | 0.351/0.391 | 0.583/0.543 |
| Weather | 720 | 0.320/0.335 | 0.321/0.336 | 0.320/0.335 | 0.323/0.362 | 0.389/0.409 | 0.415/0.426 | 0.916/0.705 |
| Traffic | 96 | 0.352/0.244 | 0.399/0.294 | 0.367/0.251 | 0.410/0.282 | 0.576/0.359 | 0.597/0.371 | 0.733/0.410 |
| Traffic | 192 | 0.371/0.253 | 0.412/0.298 | 0.385/0.259 | 0.423/0.287 | 0.610/0.380 | 0.607/0.382 | 0.777/0.435 |
| Traffic | 336 | 0.381/0.257 | 0.425/0.306 | 0.398/0.265 | 0.436/0.296 | 0.608/0.375 | 0.623/0.387 | 0.776/0.434 |
| Traffic | 720 | 0.425/0.282 | 0.460/0.323 | 0.434/0.287 | 0.466/0.315 | 0.621/0.375 | 0.639/0.395 | 0.827/0.466 |
| Electricity | 96 | 0.126/0.221 | 0.138/0.237 | 0.130/0.222 | 0.140/0.237 | 0.186/0.302 | 0.196/0.313 | 0.304/0.393 |
| Electricity | 192 | 0.145/0.238 | 0.156/0.252 | 0.148/0.240 | 0.153/0.249 | 0.197/0.311 | 0.211/0.324 | 0.327/0.417 |
| Electricity | 336 | 0.164/0.256 | 0.170/0.265 | 0.167/0.261 | 0.169/0.267 | 0.213/0.328 | 0.214/0.327 | 0.333/0.422 |
| Electricity | 720 | 0.193/0.291 | 0.208/0.297 | 0.202/0.291 | 0.203/0.301 | 0.233/0.344 | 0.236/0.342 | 0.351/0.427 |

**Interpretation caution:** values retain their original long-term forecasting task/horizon; missing values do not imply zero error.


## 123. Full Table 8 — univariate forecasting (16 rows)

**Original:** Table 8, PDF p.15; ETT univariate oil-temperature forecasting. Columns show `MSE/MAE` for each method unless otherwise specified.

| Dataset | Horizon | PatchTST/64 | PatchTST/42 | DLinear | FEDformer | Autoformer | Informer | LogTrans |
|---|---:|---|---|---|---|---|---|---|
| ETTh1 | 96 | 0.059/0.189 | 0.055/0.179 | 0.056/0.180 | 0.079/0.215 | 0.071/0.206 | 0.193/0.377 | 0.283/0.468 |
| ETTh1 | 192 | 0.074/0.215 | 0.071/0.205 | 0.071/0.204 | 0.104/0.245 | 0.114/0.262 | 0.217/0.395 | 0.234/0.409 |
| ETTh1 | 336 | 0.076/0.220 | 0.081/0.225 | 0.098/0.244 | 0.119/0.270 | 0.107/0.258 | 0.202/0.381 | 0.386/0.546 |
| ETTh1 | 720 | 0.087/0.236 | 0.087/0.232 | 0.189/0.359 | 0.142/0.299 | 0.126/0.283 | 0.183/0.355 | 0.475/0.629 |
| ETTh2 | 96 | 0.131/0.284 | 0.129/0.282 | 0.131/0.279 | 0.128/0.271 | 0.153/0.306 | 0.213/0.373 | 0.217/0.379 |
| ETTh2 | 192 | 0.171/0.329 | 0.168/0.328 | 0.176/0.329 | 0.185/0.330 | 0.204/0.351 | 0.227/0.387 | 0.281/0.429 |
| ETTh2 | 336 | 0.171/0.336 | 0.185/0.351 | 0.209/0.367 | 0.231/0.378 | 0.246/0.389 | 0.242/0.401 | 0.293/0.437 |
| ETTh2 | 720 | 0.223/0.380 | 0.224/0.383 | 0.276/0.426 | 0.278/0.420 | 0.268/0.409 | 0.291/0.439 | 0.218/0.387 |
| ETTm1 | 96 | 0.026/0.123 | 0.026/0.121 | 0.028/0.123 | 0.033/0.140 | 0.056/0.183 | 0.109/0.277 | 0.049/0.171 |
| ETTm1 | 192 | 0.040/0.151 | 0.039/0.150 | 0.045/0.156 | 0.058/0.186 | 0.081/0.216 | 0.151/0.310 | 0.157/0.317 |
| ETTm1 | 336 | 0.053/0.174 | 0.053/0.173 | 0.061/0.182 | 0.084/0.231 | 0.076/0.218 | 0.427/0.591 | 0.289/0.459 |
| ETTm1 | 720 | 0.073/0.206 | 0.074/0.207 | 0.080/0.210 | 0.102/0.250 | 0.110/0.267 | 0.438/0.586 | 0.430/0.579 |
| ETTm2 | 96 | 0.065/0.187 | 0.065/0.186 | 0.063/0.183 | 0.067/0.198 | 0.065/0.189 | 0.088/0.225 | 0.075/0.208 |
| ETTm2 | 192 | 0.093/0.231 | 0.094/0.231 | 0.092/0.227 | 0.102/0.245 | 0.118/0.256 | 0.132/0.283 | 0.129/0.275 |
| ETTm2 | 336 | 0.121/0.266 | 0.120/0.265 | 0.119/0.261 | 0.130/0.279 | 0.154/0.305 | 0.180/0.336 | 0.154/0.302 |
| ETTm2 | 720 | 0.172/0.322 | 0.171/0.322 | 0.175/0.320 | 0.178/0.325 | 0.182/0.335 | 0.300/0.435 | 0.160/0.321 |

**Interpretation caution:** values retain their original long-term forecasting task/horizon; missing values do not imply zero error.


## 124. Full Table 9 — look-back window sensitivity (32 rows)

**Original:** Table 9, PDF p.16; all six look-back windows for each horizon. For ILI the effective historical length is the parenthesized header value. Columns show `MSE/MAE` for each method unless otherwise specified.

| Dataset | Horizon | L=24 (ILI 24) | L=48 (ILI 36) | L=96 (ILI 48) | L=192 (ILI 60) | L=336 (ILI 104) | L=720 (ILI 144) |
|---|---:|---|---|---|---|---|---|
| Weather | 96 | 0.222/0.246 | 0.212/0.243 | 0.178/0.219 | 0.160/0.204 | 0.152/0.199 | 0.147/0.198 |
| Weather | 192 | 0.265/0.279 | 0.254/0.277 | 0.224/0.259 | 0.204/0.245 | 0.197/0.243 | 0.190/0.240 |
| Weather | 336 | 0.325/0.322 | 0.310/0.316 | 0.278/0.298 | 0.257/0.285 | 0.249/0.283 | 0.242/0.282 |
| Weather | 720 | 0.404/0.374 | 0.385/0.365 | 0.350/0.346 | 0.329/0.338 | 0.320/0.335 | 0.304/0.328 |
| Traffic | 96 | 0.766/0.419 | 0.671/0.381 | 0.477/0.305 | 0.401/0.267 | 0.367/0.251 | 0.365/0.251 |
| Traffic | 192 | 0.725/0.398 | 0.616/0.356 | 0.471/0.299 | 0.406/0.268 | 0.385/0.259 | 0.382/0.258 |
| Traffic | 336 | 0.752/0.410 | 0.635/0.364 | 0.485/0.305 | 0.421/0.277 | 0.398/0.265 | 0.398/0.267 |
| Traffic | 720 | 0.786/0.427 | 0.673/0.383 | 0.518/0.325 | 0.452/0.297 | 0.434/0.287 | 0.436/0.289 |
| Electricity | 96 | 0.268/0.316 | 0.225/0.293 | 0.174/0.259 | 0.138/0.230 | 0.130/0.222 | 0.130/0.224 |
| Electricity | 192 | 0.259/0.316 | 0.217/0.291 | 0.178/0.265 | 0.149/0.243 | 0.148/0.240 | 0.147/0.241 |
| Electricity | 336 | 0.283/0.335 | 0.238/0.309 | 0.196/0.282 | 0.169/0.262 | 0.167/0.261 | 0.163/0.259 |
| Electricity | 720 | 0.321/0.365 | 0.278/0.342 | 0.237/0.316 | 0.211/0.299 | 0.202/0.291 | 0.197/0.290 |
| ILI | 24 | 3.062/1.118 | 1.610/0.803 | 1.281/0.704 | 1.300/0.700 | 1.522/0.814 | 1.470/0.793 |
| ILI | 36 | 2.732/1.071 | 1.262/0.731 | 1.251/0.752 | 1.367/0.776 | 1.430/0.834 | 1.518/0.856 |
| ILI | 48 | 3.059/1.117 | 1.991/0.845 | 1.901/0.879 | 1.690/0.812 | 1.673/0.854 | 1.834/0.921 |
| ILI | 60 | 2.610/1.057 | 1.702/0.829 | 1.611/0.844 | 1.526/0.795 | 1.529/0.862 | 1.656/0.885 |
| ETTh1 | 96 | 0.464/0.445 | 0.410/0.417 | 0.393/0.408 | 0.382/0.401 | 0.375/0.399 | 0.376/0.408 |
| ETTh1 | 192 | 0.521/0.474 | 0.469/0.448 | 0.445/0.434 | 0.428/0.425 | 0.414/0.421 | 0.413/0.431 |
| ETTh1 | 336 | 0.570/0.498 | 0.516/0.469 | 0.484/0.451 | 0.451/0.436 | 0.431/0.436 | 0.445/0.454 |
| ETTh1 | 720 | 0.575/0.522 | 0.509/0.487 | 0.480/0.471 | 0.452/0.459 | 0.449/0.466 | 0.458/0.477 |
| ETTh2 | 96 | 0.333/0.362 | 0.307/0.348 | 0.294/0.343 | 0.285/0.340 | 0.274/0.336 | 0.279/0.341 |
| ETTh2 | 192 | 0.422/0.409 | 0.397/0.399 | 0.377/0.393 | 0.356/0.386 | 0.339/0.379 | 0.349/0.386 |
| ETTh2 | 336 | 0.442/0.432 | 0.412/0.420 | 0.381/0.409 | 0.350/0.395 | 0.331/0.380 | 0.375/0.409 |
| ETTh2 | 720 | 0.462/0.453 | 0.434/0.441 | 0.412/0.433 | 0.395/0.427 | 0.379/0.422 | 0.394/0.434 |
| ETTm1 | 96 | 0.624/0.481 | 0.424/0.403 | 0.321/0.360 | 0.291/0.340 | 0.290/0.342 | 0.295/0.348 |
| ETTm1 | 192 | 0.671/0.507 | 0.468/0.429 | 0.362/0.384 | 0.328/0.365 | 0.332/0.369 | 0.334/0.373 |
| ETTm1 | 336 | 0.714/0.533 | 0.501/0.453 | 0.392/0.402 | 0.365/0.389 | 0.366/0.392 | 0.361/0.393 |
| ETTm1 | 720 | 0.744/0.554 | 0.553/0.484 | 0.450/0.435 | 0.422/0.423 | 0.420/0.424 | 0.416/0.419 |
| ETTm2 | 96 | 0.212/0.290 | 0.189/0.272 | 0.178/0.260 | 0.169/0.254 | 0.165/0.255 | 0.162/0.254 |
| ETTm2 | 192 | 0.282/0.334 | 0.260/0.317 | 0.249/0.307 | 0.230/0.294 | 0.220/0.292 | 0.216/0.293 |
| ETTm2 | 336 | 0.354/0.376 | 0.328/0.359 | 0.313/0.346 | 0.280/0.329 | 0.278/0.329 | 0.269/0.329 |
| ETTm2 | 720 | 0.458/0.433 | 0.429/0.415 | 0.400/0.398 | 0.378/0.386 | 0.367/0.385 | 0.350/0.380 |

**Interpretation caution:** values retain their original long-term forecasting task/horizon; missing values do not imply zero error.


## 125. Full Table 10 — patching × channel-independence factorial ablation (32 rows)

**Original:** Table 10, PDF p.17; 2×2 architectural ablation plus FEDformer comparator. Columns show `MSE/MAE` for each method unless otherwise specified.

| Dataset | Horizon | P+CI (full) | CI only | Patch only | Neither (Original) | FEDformer |
|---|---:|---|---|---|---|---|
| Weather | 96 | 0.152/0.199 | 0.164/0.213 | 0.168/0.223 | 0.177/0.236 | 0.238/0.314 |
| Weather | 192 | 0.197/0.243 | 0.205/0.250 | 0.213/0.262 | 0.221/0.270 | 0.275/0.329 |
| Weather | 336 | 0.249/0.283 | 0.255/0.289 | 0.266/0.300 | 0.271/0.306 | 0.339/0.377 |
| Weather | 720 | 0.320/0.335 | 0.327/0.343 | 0.351/0.359 | 0.340/0.353 | 0.389/0.409 |
| Traffic | 96 | 0.367/0.251 | 0.397/0.271 | 0.595/0.376 | - | 0.576/0.359 |
| Traffic | 192 | 0.385/0.259 | 0.411/0.276 | 0.612/0.387 | - | 0.610/0.380 |
| Traffic | 336 | 0.398/0.265 | 0.423/0.282 | 0.651/0.391 | - | 0.608/0.375 |
| Traffic | 720 | 0.434/0.287 | 0.457/0.309 | - | - | 0.621/0.375 |
| Electricity | 96 | 0.130/0.222 | 0.136/0.231 | 0.196/0.307 | 0.205/0.318 | 0.186/0.302 |
| Electricity | 192 | 0.148/0.240 | 0.164/0.263 | 0.215/0.323 | - | 0.197/0.311 |
| Electricity | 336 | 0.167/0.261 | 0.168/0.262 | 0.228/0.338 | - | 0.213/0.328 |
| Electricity | 720 | 0.202/0.291 | 0.219/0.312 | 0.244/0.345 | - | 0.233/0.344 |
| ILI | 24 | 1.522/0.814 | 2.111/1.048 | 2.157/0.964 | 2.737/1.081 | 2.624/1.095 |
| ILI | 36 | 1.430/0.834 | 2.000/1.002 | 2.564/1.058 | 2.126/0.935 | 2.516/1.021 |
| ILI | 48 | 1.673/0.854 | 2.167/1.029 | 2.348/1.022 | 2.178/0.971 | 2.505/1.041 |
| ILI | 60 | 1.529/0.862 | 2.075/1.021 | 2.486/1.065 | 2.354/1.026 | 2.742/1.122 |
| ETTh1 | 96 | 0.375/0.399 | 0.365/0.395 | 0.416/0.438 | 0.455/0.459 | 0.376/0.415 |
| ETTh1 | 192 | 0.414/0.421 | 0.403/0.415 | 0.459/0.464 | 0.503/0.486 | 0.423/0.446 |
| ETTh1 | 336 | 0.431/0.436 | 0.430/0.433 | 0.484/0.480 | 0.514/0.503 | 0.444/0.462 |
| ETTh1 | 720 | 0.449/0.466 | 0.449/0.454 | 0.500/0.494 | 0.531/0.520 | 0.469/0.492 |
| ETTh2 | 96 | 0.274/0.336 | 0.277/0.337 | 0.334/0.388 | 0.348/0.394 | 0.332/0.374 |
| ETTh2 | 192 | 0.339/0.379 | 0.343/0.384 | 0.381/0.418 | 0.395/0.424 | 0.407/0.446 |
| ETTh2 | 336 | 0.331/0.380 | 0.333/0.383 | 0.361/0.414 | 0.369/0.419 | 0.400/0.447 |
| ETTh2 | 720 | 0.379/0.422 | 0.379/0.420 | 0.423/0.448 | 0.433/0.458 | 0.412/0.469 |
| ETTm1 | 96 | 0.290/0.342 | 0.300/0.354 | 0.326/0.368 | 0.324/0.370 | 0.326/0.390 |
| ETTm1 | 192 | 0.332/0.369 | 0.333/0.374 | 0.391/0.405 | 0.373/0.398 | 0.365/0.415 |
| ETTm1 | 336 | 0.366/0.392 | 0.369/0.397 | 0.427/0.425 | 0.415/0.421 | 0.392/0.425 |
| ETTm1 | 720 | 0.420/0.424 | 0.413/0.423 | 0.481/0.457 | 0.480/0.459 | 0.446/0.458 |
| ETTm2 | 96 | 0.165/0.255 | 0.166/0.259 | 0.195/0.274 | 0.208/0.289 | 0.180/0.271 |
| ETTm2 | 192 | 0.220/0.292 | 0.223/0.295 | 0.259/0.314 | 0.265/0.328 | 0.252/0.318 |
| ETTm2 | 336 | 0.278/0.329 | 0.279/0.330 | 0.297/0.345 | 0.323/0.365 | 0.324/0.364 |
| ETTm2 | 720 | 0.367/0.385 | 0.370/0.387 | 0.400/0.404 | 0.469/0.444 | 0.410/0.420 |

**Interpretation caution:** values retain their original long-term forecasting task/horizon; missing values do not imply zero error.


## 126. Full Table 11 — instance normalization ablation (32 rows)

**Original:** Table 11, PDF p.18; with/without per-instance normalization. Columns show `MSE/MAE` for each method unless otherwise specified.

| Dataset | Horizon | PatchTST/64 with norm | PatchTST/64 without norm | PatchTST/42 with norm | PatchTST/42 without norm | FEDformer | Autoformer | Informer |
|---|---:|---|---|---|---|---|---|---|
| Weather | 96 | 0.149/0.198 | 0.161/0.219 | 0.152/0.199 | 0.156/0.210 | 0.238/0.314 | 0.249/0.329 | 0.354/0.405 |
| Weather | 192 | 0.194/0.241 | 0.201/0.254 | 0.197/0.243 | 0.199/0.250 | 0.275/0.329 | 0.325/0.370 | 0.419/0.434 |
| Weather | 336 | 0.245/0.282 | 0.253/0.298 | 0.249/0.283 | 0.248/0.294 | 0.339/0.377 | 0.351/0.391 | 0.583/0.543 |
| Weather | 720 | 0.314/0.334 | 0.323/0.357 | 0.320/0.335 | 0.313/0.342 | 0.389/0.409 | 0.415/0.426 | 0.916/0.705 |
| Traffic | 96 | 0.360/0.249 | 0.413/0.295 | 0.367/0.251 | 0.425/0.299 | 0.576/0.359 | 0.597/0.371 | 0.733/0.410 |
| Traffic | 192 | 0.379/0.256 | 0.425/0.302 | 0.385/0.259 | 0.439/0.302 | 0.610/0.380 | 0.607/0.382 | 0.777/0.435 |
| Traffic | 336 | 0.392/0.264 | 0.435/0.307 | 0.398/0.265 | 0.456/0.316 | 0.608/0.375 | 0.623/0.387 | 0.776/0.434 |
| Traffic | 720 | 0.432/0.286 | 0.473/0.321 | 0.434/0.287 | 0.488/0.333 | 0.621/0.375 | 0.639/0.395 | 0.827/0.466 |
| Electricity | 96 | 0.129/0.222 | 0.133/0.230 | 0.130/0.222 | 0.131/0.226 | 0.186/0.302 | 0.196/0.313 | 0.304/0.393 |
| Electricity | 192 | 0.147/0.240 | 0.148/0.244 | 0.148/0.240 | 0.150/0.244 | 0.197/0.311 | 0.211/0.324 | 0.327/0.417 |
| Electricity | 336 | 0.163/0.259 | 0.164/0.262 | 0.167/0.261 | 0.168/0.267 | 0.213/0.328 | 0.214/0.327 | 0.333/0.422 |
| Electricity | 720 | 0.197/0.290 | 0.196/0.291 | 0.202/0.291 | 0.201/0.298 | 0.233/0.344 | 0.236/0.342 | 0.351/0.427 |
| ILI | 24 | 1.319/0.754 | 3.563/1.317 | 1.522/0.814 | 3.489/1.345 | 2.624/1.095 | 2.906/1.182 | 4.657/1.449 |
| ILI | 36 | 1.579/0.870 | 3.426/1.205 | 1.430/0.834 | 4.629/1.550 | 2.516/1.021 | 2.585/1.038 | 4.650/1.463 |
| ILI | 48 | 1.553/0.815 | 4.309/1.449 | 1.673/0.854 | 3.746/1.383 | 2.505/1.041 | 3.024/1.145 | 5.004/1.542 |
| ILI | 60 | 1.470/0.788 | 4.065/1.402 | 1.529/0.862 | 5.174/1.622 | 2.742/1.122 | 2.761/1.114 | 5.071/1.543 |
| ETTh1 | 96 | 0.370/0.400 | 0.385/0.410 | 0.375/0.399 | 0.388/0.412 | 0.376/0.415 | 0.435/0.446 | 0.941/0.769 |
| ETTh1 | 192 | 0.413/0.429 | 0.417/0.432 | 0.414/0.421 | 0.430/0.438 | 0.423/0.446 | 0.456/0.457 | 1.007/0.786 |
| ETTh1 | 336 | 0.422/0.440 | 0.439/0.449 | 0.431/0.436 | 0.454/0.458 | 0.444/0.462 | 0.486/0.487 | 1.038/0.784 |
| ETTh1 | 720 | 0.447/0.468 | 0.478/0.494 | 0.449/0.466 | 0.494/0.497 | 0.469/0.492 | 0.515/0.517 | 1.144/0.857 |
| ETTh2 | 96 | 0.274/0.337 | 0.299/0.359 | 0.274/0.336 | 0.313/0.374 | 0.332/0.374 | 0.332/0.368 | 1.549/0.952 |
| ETTh2 | 192 | 0.341/0.382 | 0.354/0.404 | 0.339/0.379 | 0.402/0.432 | 0.407/0.446 | 0.426/0.434 | 3.792/1.542 |
| ETTh2 | 336 | 0.329/0.384 | 0.374/0.420 | 0.331/0.380 | 0.448/0.465 | 0.400/0.447 | 0.477/0.479 | 4.215/1.642 |
| ETTh2 | 720 | 0.379/0.422 | 0.479/0.492 | 0.379/0.422 | 0.688/0.588 | 0.412/0.469 | 0.453/0.490 | 3.656/1.619 |
| ETTm1 | 96 | 0.293/0.346 | 0.308/0.358 | 0.290/0.342 | 0.308/0.358 | 0.326/0.390 | 0.510/0.492 | 0.626/0.560 |
| ETTm1 | 192 | 0.333/0.370 | 0.335/0.375 | 0.332/0.369 | 0.356/0.390 | 0.365/0.415 | 0.514/0.495 | 0.725/0.619 |
| ETTm1 | 336 | 0.369/0.392 | 0.362/0.392 | 0.366/0.392 | 0.389/0.411 | 0.392/0.425 | 0.510/0.492 | 1.005/0.741 |
| ETTm1 | 720 | 0.416/0.420 | 0.432/0.429 | 0.420/0.424 | 0.430/0.439 | 0.446/0.458 | 0.527/0.493 | 1.133/0.845 |
| ETTm2 | 96 | 0.166/0.256 | 0.172/0.258 | 0.165/0.255 | 0.167/0.257 | 0.180/0.271 | 0.205/0.293 | 0.355/0.462 |
| ETTm2 | 192 | 0.223/0.296 | 0.245/0.306 | 0.220/0.292 | 0.226/0.303 | 0.252/0.318 | 0.278/0.336 | 0.595/0.586 |
| ETTm2 | 336 | 0.274/0.329 | 0.306/0.346 | 0.278/0.329 | 0.301/0.348 | 0.324/0.364 | 0.343/0.379 | 1.270/0.871 |
| ETTm2 | 720 | 0.362/0.385 | 0.391/0.404 | 0.367/0.385 | 0.392/0.407 | 0.410/0.420 | 0.414/0.419 | 3.001/1.267 |

**Interpretation caution:** values retain their original long-term forecasting task/horizon; missing values do not imply zero error.


## 127. Full Table 14 — seed robustness (28 rows; ILI absent)

**Original:** Table 14, PDF p.20. Columns are `mean±standard_deviation`, not MSE/MAE pairs. Supervised: five full runs across seeds 2019–2023. Self-supervised: one pretraining run, five separate downstream fine-tuning runs.

| Dataset | Horizon | Sup. MSE | Sup. MAE | SSL/FT MSE | SSL/FT MAE |
|---|---:|---:|---:|---:|---:|
| Weather | 96 | 0.1525±0.0024 | 0.2002±0.0023 | 0.1450±0.0008 | 0.1937±0.0010 |
| Weather | 192 | 0.1975±0.0015 | 0.2434±0.0010 | 0.1893±0.0003 | 0.2364±0.0006 |
| Weather | 336 | 0.2494±0.0012 | 0.2841±0.0014 | 0.2413±0.0003 | 0.2774±0.0005 |
| Weather | 720 | 0.3194±0.0002 | 0.3352±0.0003 | 0.3156±0.0020 | 0.3316±0.0016 |
| Traffic | 96 | 0.3669±0.0006 | 0.2504±0.0007 | 0.3528±0.0022 | 0.2443±0.0016 |
| Traffic | 192 | 0.3858±0.0004 | 0.2586±0.0004 | 0.3729±0.0013 | 0.2531±0.0009 |
| Traffic | 336 | 0.3994±0.0010 | 0.2672±0.0016 | 0.3846±0.0020 | 0.2588±0.0011 |
| Traffic | 720 | 0.4383±0.0097 | 0.2913±0.0104 | 0.4241±0.0007 | 0.2816±0.0010 |
| Electricity | 96 | 0.1304±0.0006 | 0.2234±0.0006 | 0.1256±0.0002 | 0.2210±0.0003 |
| Electricity | 192 | 0.1482±0.0002 | 0.2403±0.0002 | 0.1451±0.0002 | 0.2397±0.0010 |
| Electricity | 336 | 0.1659±0.0006 | 0.2596±0.0006 | 0.1624±0.0010 | 0.2576±0.0009 |
| Electricity | 720 | 0.2019±0.0006 | 0.2917±0.0006 | 0.1990±0.0002 | 0.2916±0.0002 |
| ETTh1 | 96 | 0.3752±0.0008 | 0.3999±0.0004 | 0.3700±0.0035 | 0.4001±0.0023 |
| ETTh1 | 192 | 0.4127±0.0012 | 0.4207±0.0006 | 0.4146±0.0012 | 0.4287±0.0013 |
| ETTh1 | 336 | 0.4278±0.0033 | 0.4334±0.0028 | 0.4285±0.0018 | 0.4402±0.0017 |
| ETTh1 | 720 | 0.4462±0.0035 | 0.4637±0.0027 | 0.4670±0.0052 | 0.4768±0.0033 |
| ETTh2 | 96 | 0.2749±0.0005 | 0.3363±0.0006 | 0.2869±0.0039 | 0.3439±0.0016 |
| ETTh2 | 192 | 0.3385±0.0010 | 0.3789±0.0014 | 0.3523±0.0048 | 0.3855±0.0027 |
| ETTh2 | 336 | 0.3288±0.0010 | 0.3823±0.0027 | 0.3779±0.0057 | 0.4112±0.0030 |
| ETTh2 | 720 | 0.3784±0.0010 | 0.4212±0.0009 | 0.3993±0.0054 | 0.4385±0.0038 |
| ETTm1 | 96 | 0.2893±0.0009 | 0.3415±0.0007 | 0.2876±0.0012 | 0.3427±0.0011 |
| ETTm1 | 192 | 0.3316±0.0008 | 0.3695±0.0007 | 0.3296±0.0026 | 0.3688±0.0016 |
| ETTm1 | 336 | 0.3661±0.0022 | 0.3914±0.0012 | 0.3583±0.0015 | 0.3879±0.0016 |
| ETTm1 | 720 | 0.4200±0.0056 | 0.4243±0.0033 | 0.4094±0.0044 | 0.4193±0.0013 |
| ETTm2 | 96 | 0.1647±0.0011 | 0.2538±0.0010 | 0.1637±0.0020 | 0.2537±0.0024 |
| ETTm2 | 192 | 0.2223±0.0018 | 0.2936±0.0014 | 0.2175±0.0011 | 0.2908±0.0013 |
| ETTm2 | 336 | 0.2775±0.0020 | 0.3297±0.0010 | 0.2706±0.0016 | 0.3260±0.0016 |
| ETTm2 | 720 | 0.3648±0.0024 | 0.3833±0.0010 | 0.3539±0.0023 | 0.3799±0.0024 |

**Uncertainty warning:** the self-supervised standard deviation excludes independent pretraining-backbone variability.


## 128. Full Table 15 — CI transfers to other forecasters (32 rows)

**Original:** Table 15, PDF p.22; CI applied to Informer, Autoformer, and FEDformer. Columns show `MSE/MAE` for each method unless otherwise specified.

| Dataset | Horizon | PatchTST/42 | Informer | Informer-CI | Autoformer | Autoformer-CI | FEDformer | FEDformer-CI |
|---|---:|---|---|---|---|---|---|---|
| Weather | 96 | 0.152/0.199 | 0.300/0.384 | 0.174/0.232 | 0.266/0.336 | 0.227/0.289 | 0.217/0.296 | 0.214/0.278 |
| Weather | 192 | 0.197/0.243 | 0.598/0.544 | 0.214/0.270 | 0.307/0.367 | 0.269/0.318 | 0.276/0.336 | 0.258/0.322 |
| Weather | 336 | 0.249/0.283 | 0.578/0.523 | 0.266/0.310 | 0.359/0.395 | 0.315/0.344 | 0.339/0.380 | 0.302/0.336 |
| Weather | 720 | 0.320/0.335 | 1.059/0.741 | 0.327/0.356 | 0.419/0.428 | 0.384/0.389 | 0.403/0.428 | 0.374/0.369 |
| Traffic | 96 | 0.367/0.251 | 0.719/0.391 | 0.705/0.402 | 0.613/0.388 | - | 0.587/0.366 | - |
| Traffic | 192 | 0.385/0.259 | 0.696/0.379 | 0.720/0.407 | 0.616/0.382 | - | 0.604/0.373 | - |
| Traffic | 336 | 0.398/0.265 | 0.777/0.420 | 0.750/0.421 | 0.622/0.337 | - | 0.621/0.383 | - |
| Traffic | 720 | 0.434/0.287 | 0.864/0.472 | - | 0.660/0.408 | - | 0.626/0.382 | - |
| Electricity | 96 | 0.130/0.222 | 0.274/0.368 | 0.203/0.299 | 0.201/0.317 | - | 0.193/0.308 | - |
| Electricity | 192 | 0.148/0.240 | 0.296/0.386 | 0.221/0.316 | 0.222/0.334 | - | 0.201/0.315 | - |
| Electricity | 336 | 0.167/0.261 | 0.300/0.394 | 0.241/0.337 | 0.231/0.338 | - | 0.214/0.329 | - |
| Electricity | 720 | 0.202/0.291 | 0.373/0.439 | 0.314/0.391 | 0.254/0.361 | - | 0.246/0.355 | - |
| ILI | 24 | 1.522/0.814 | 5.764/1.677 | 5.514/1.629 | 3.483/1.287 | 4.210/1.500 | 3.228/1.260 | 3.280/1.264 |
| ILI | 36 | 1.430/0.834 | 4.755/1.467 | 5.515/1.628 | 3.103/1.148 | 2.809/1.162 | 2.679/1.080 | 2.862/1.126 |
| ILI | 48 | 1.673/0.854 | 4.763/1.469 | 5.263/1.574 | 2.669/1.085 | 3.218/1.267 | 2.622/1.078 | 2.834/1.150 |
| ILI | 60 | 1.529/0.862 | 5.264/1.564 | 5.330/1.602 | 2.770/1.125 | 3.627/1.396 | 2.857/1.157 | 3.115/1.240 |
| ETTh1 | 96 | 0.375/0.399 | 0.865/0.713 | 0.590/0.517 | 0.449/0.459 | 0.414/0.421 | 0.376/0.419 | 0.387/0.407 |
| ETTh1 | 192 | 0.414/0.421 | 1.008/0.792 | 0.677/0.566 | 0.500/0.482 | 0.453/0.448 | 0.420/0.448 | 0.439/0.438 |
| ETTh1 | 336 | 0.431/0.436 | 1.107/0.809 | 0.710/0.600 | 0.521/0.496 | 0.496/0.468 | 0.459/0.465 | 0.479/0.455 |
| ETTh1 | 720 | 0.449/0.466 | 1.181/0.865 | 0.777/0.660 | 0.514/0.512 | 0.662/0.568 | 0.506/0.507 | 0.485/0.478 |
| ETTh2 | 96 | 0.274/0.336 | 3.755/1.525 | 0.390/0.410 | 0.358/0.397 | 0.337/0.373 | 0.346/0.388 | 0.297/0.348 |
| ETTh2 | 192 | 0.339/0.379 | 5.602/1.931 | 0.456/0.463 | 0.456/0.452 | 0.409/0.419 | 0.429/0.439 | 0.382/0.399 |
| ETTh2 | 336 | 0.331/0.380 | 4.721/1.835 | 0.523/0.503 | 0.482/0.486 | 0.432/0.443 | 0.496/0.487 | 0.410/0.428 |
| ETTh2 | 720 | 0.379/0.422 | 3.647/1.625 | 0.843/0.661 | 0.515/0.511 | 0.443/0.463 | 0.463/0.474 | 0.422/0.444 |
| ETTm1 | 96 | 0.290/0.342 | 0.672/0.571 | 0.383/0.414 | 0.505/0.475 | 0.455/0.441 | 0.379/0.419 | 0.408/0.413 |
| ETTm1 | 192 | 0.332/0.369 | 0.795/0.669 | 0.420/0.434 | 0.553/0.496 | 0.598/0.512 | 0.426/0.441 | 0.445/0.432 |
| ETTm1 | 336 | 0.366/0.392 | 1.212/0.871 | 0.465/0.467 | 0.621/0.537 | 0.566/0.504 | 0.445/0.459 | 0.476/0.452 |
| ETTm1 | 720 | 0.420/0.424 | 1.166/0.823 | 0.529/0.502 | 0.671/0.561 | 0.680/0.557 | 0.543/0.490 | 0.533/0.481 |
| ETTm2 | 96 | 0.165/0.255 | 0.365/0.453 | 0.208/0.298 | 0.255/0.339 | 0.218/0.308 | 0.203/0.287 | 0.198/0.284 |
| ETTm2 | 192 | 0.220/0.292 | 0.533/0.563 | 0.274/0.345 | 0.281/0.340 | 0.281/0.339 | 0.269/0.328 | 0.259/0.320 |
| ETTm2 | 336 | 0.278/0.329 | 1.363/0.887 | 0.351/0.394 | 0.339/0.372 | 0.336/0.370 | 0.325/0.366 | 0.315/0.353 |
| ETTm2 | 720 | 0.367/0.385 | 3.379/1.338 | 0.482/0.474 | 0.433/0.432 | 0.428/0.418 | 0.421/0.415 | 0.412/0.406 |

**Interpretation caution:** values retain their original long-term forecasting task/horizon; missing values do not imply zero error.


## 129. Notes on numerical transcription and source interpretation

- All values above are transcribed from the attached original **PDF**, not sourced from later internet benchmarks or a repository commit.
- Data were parsed from the PDF's layout-preserving text and checked against expected dataset/horizon grouping.
- One line-wrapped Table 9 row (Traffic, horizon 720) was restored using the numeric values printed in the source, not guessed.
- A missing Table 10/15 entry (`-`) means the source did not provide a numeric result, often because of GPU memory limitations or excessive runtime.
- Original source table numbers and PDF page numbers are recorded in the headings, so a future Agent can verify a suspicious cell directly.
- Large tables are included for targeted lookup and methodological comparison; **do not load all of them into every planning context if a specific mechanism or one dataset is enough**.

---

# Part XV — Final Research-Agent Operating Summary

## 130. Recommended retrieval strategy

1. Read YAML, Sections 2–5 and 116/120 to learn the mechanism.
2. Read Sections 7–23 for equations, tensor shapes and training.
3. Read Sections 25–59 to understand original experiments, fairness, and uncertainty.
4. Read Sections 60–78 to extract reusable building blocks and their assumptions.
5. Read Sections 79–102 **only** when designing a financial transfer.
6. Read the full numeric tables only for verifying concrete experimental claims.
7. Before proposing any publication-level novelty, inspect `papers/frontier/`, `papers/related_work/`, local code and existing experimental summaries.

## 131. Final one-line memory

**PatchTST teaches that time-series prediction depends critically on the granularity of temporal tokens and the timing of cross-variable mixing; for short-window cross-sectional finance, transfer the principle, not the original long-horizon hyperparameters.**
