---
paper_id: Fedus_Zoph_Shazeer_2022_SwitchTransformer
title: "Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity"
short_name: Switch Transformer
authors:
  - William Fedus
  - Barret Zoph
  - Noam Shazeer
venue: Journal of Machine Learning Research
volume: 23
publication_year: 2022
publication_month: 2022-04
submission: "2021-08"
revision: "2022-03"
pages: 39
paper_type: foundation
research_family:
  - mixture_of_experts
  - sparse_conditional_computation
  - expert_routing
  - transformer
  - distributed_training
  - model_scaling
  - training_stability
source:
  uploaded_pdf: "Switch Transformers Scaling to Trillion Parameter Models(1).pdf"
  original_pdf_pages: 39
  journal_page: "https://jmlr.org/papers/v23/21-0998.html"
  implementation: "https://github.com/tensorflow/mesh/blob/master/mesh_tensorflow/transformer/moe.py"
  citation_format: "Fedus, Zoph, and Shazeer (2022), JMLR 23"
  version_note: "2022 JMLR version; distinguish from preprint versions"
key_mechanisms:
  - top_1_expert_routing
  - softmax_router_probability
  - scaled_expert_output
  - capacity_factor_and_overflow
  - differentiable_load_balance_surrogate
  - selective_float32_router
  - reduced_initialization_scale
  - expert_dropout
  - sparse_ffn_replacement
  - expert_data_model_parallelism
  - sparse_to_dense_distillation
core_equations:
  - "p_i(x) = softmax(W_r x)_i"
  - "i^*(x)=argmax_i p_i(x)"
  - "y(x)=p_{i^*}(x) E_{i^*}(x)"
  - "C=(tokens_per_batch/number_of_experts)*capacity_factor"
  - "L_balance=alpha*N*sum_i f_i P_i"
source_integrity_tags:
  SOURCE: "Explicitly stated or reported in the attached paper"
  DERIVATION: "Derived mathematically from reported definitions, not a direct quote"
  INTERPRETATION: "Reasoned model-level explanation grounded in the mechanism"
  CRITIQUE: "Limitation or challenge identified by the curator"
  EXTENSION: "New cross-sectional-finance research idea, not an author claim"
  VERIFY: "Ambiguous, inconsistent, implementation-dependent, or requiring code inspection"
source_issues:
  - "Table 9 apparent Num. Experts / Num Layers header-value transposition"
  - "Table 11 contradicts its own higher-is-better header with lower-is-better caption"
  - "Appendix F code illustration has implementation-specific details, not portable production pseudocode"
related_papers:
  foundations:
    - MoE_1991.md
    - Sparse_MoE_2017.md
  baselines:
    - PRISM-VQ_2026.md
    - FactorVQVAE_2025.md
    - MASTER_2024.md
    - MATCC_2024.md
relevance_to_agent:
  conceptual_foundation: very_high
  routing_implementation: very_high
  load_balance: very_high
  training_stability: very_high
  finance_transfer: high
  giant_scale_infrastructure: reference_only
priority: very_high
---

# Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity

> Fedus, Zoph, and Shazeer, *Journal of Machine Learning Research*, Volume 23 (2022), 39 pages.
>
> Research-Agent foundational knowledge file; source: attached 2022 JMLR journal version.
>
> **Source attribution convention:** `[SOURCE]` = original paper; `[DERIVATION]` = mathematical elaboration; `[INTERPRETATION]` = grounded mechanism analysis; `[CRITIQUE]` = independent criticism; `[EXTENSION]` = experimental proposal; `[VERIFY]` = not established from the paper or source inconsistency. Section, table, and PDF page numbers refer to the JMLR article.

## 000. How This Document Should Be Used

[SOURCE] This paper's central goal is to increase the number of model parameters independently of the FLOPs expended on each example, by sending each token to a single trainable feed-forward expert inside a Transformer.

[INTERPRETATION] This is not primarily a paper on learning economically meaningful specialized experts. Its key target is the *quality–compute–parameter trade-off* of large neural networks and the engineering required to make sparse expert routing stable and efficient.

Use this file to:

1. Recover the exact definition of Switch Top-1 routing, including the **non-unit selected gate multiplier**.
2. Understand why Top-1 can receive a gradient despite selecting just one expert.
3. Implement the original batch load-balancing auxiliary loss correctly.
4. Understand capacity factor, expert overflow, and what happens when a token is dropped by the expert layer.
5. Separate numerical stabilization of the router from ordinary model-wide mixed-precision training.
6. Distinguish *expert activation count*, *total parameter count*, *FLOPs/token*, *wall-clock speed*, and *generalization*.
7. Check which original experimental comparisons support the claims and which do not.
8. Learn from unsuccessful alternatives in the appendices.
9. Study links from Jacobs et al. (1991) and Shazeer et al. (2017) to later structure-conditioned MoE in stock prediction.
10. Propose finance experiments with explicit controls and falsification criteria.

---

# PART I — RESEARCH QUESTION AND HISTORICAL POSITION

## 001. Bibliographic Identity

[SOURCE; PDF p.1]

- Full title: *Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity*.
- Authors: William Fedus, Barret Zoph, Noam Shazeer (Google).
- Journal: *Journal of Machine Learning Research* 23 (2022), pp. 1–39.
- Submitted August 2021, revised March 2022, published April 2022.
- Editor: Alexander Clark.
- Official paper code link printed in the footer: `https://github.com/tensorflow/mesh/blob/master/mesh_tensorflow/transformer/moe.py`.
- Paper released under CC BY 4.0.

## 002. Central Research Question

[SOURCE; Abstract and §§1–2]

> Can a Transformer acquire much more parameter capacity, improving the quality achievable at fixed computational resources, while simplifying sparse expert routing enough to make training stable and efficient?

This is distinct from asking whether a sparse layer has fewer parameters or whether the number of experts always improves accuracy.

## 003. Why This Research Matters

[SOURCE] Dense Transformers reuse approximately the same learned FFN parameters for every token. Increasing model width or depth generally increases both parameter count and per-token FLOPs. Conditional computation decouples these quantities by computing with only a subset of the parameters on each token.

[INTERPRETATION] The paper tests a *fourth scaling dimension*:

\[
\text{total parameters} \uparrow
\quad\text{while}\quad
\text{FLOPs per token} \approx \text{constant}.
\]

A model can own 128 independently learned FFN experts but activate just one of them on a token.

## 004. Challenges Inherited from Earlier MoE

[SOURCE; §1]

- Complexity in gating and dispatch.
- Communication between devices hosting different expert parameters.
- Instability caused by discrete routing changes.
- Imbalanced expert utilization and resulting overflow.
- Dependence on low-precision training for throughput at large scale.
- Overfitting when very large sparse pre-trained models are fine-tuned on small tasks.

## 005. Main Hypothesis

[INTERPRETATION] For suitable data and training conditions, increasing the number of specialized expert parameters without increasing the number of active experts can create additional useful representational capacity.

Crucial qualification: the paper tests this primarily on language data, not on low-SNR small-sample equity returns. Whether the same trade-off helps stock ranking is a research question.

## 006. Published Contributions

[SOURCE; pp.2–3]

1. A simpler Top-1 MoE / Switch FFN layer.
2. Improved training stability from selective precision and initialization.
3. A differentiable auxiliary load-balancing loss.
4. Expert-specific dropout for downstream fine-tuning.
5. Speed–quality improvements over FLOP-matched dense and Top-2 MoE baselines.
6. Scaling to hundreds of billions and ~1.6 trillion parameters with data, model, and expert parallelism.
7. Positive results for many downstream tasks and 101-language pre-training.
8. Sparse-teacher to dense-student distillation.

## 007. The Historical Three-Paper Chain

[INTERPRETATION]

| Work | Core question | Routing principle | Main evidence |
|---|---|---|---|
| Jacobs et al. 1991 | How do experts divide different subtasks? | Dense stochastic/soft competition | Task decomposition and training speed |
| Shazeer et al. 2017 | Can conditional computation scale expert capacity? | Noisy Top-k; k commonly >1 | Language modeling and translation with many experts |
| Fedus et al. 2022 | Can sparse routing be simpler, faster, and stable? | Top-1 Switch routing | Dense / Top-2 compute-speed trade-offs and massive scaling |

Do not attribute Top-1 Switch routing, capacity factor, or selective routing precision to the 1991 paper.

## 008. What Switch Transformer Is Not

[SOURCE/INTERPRETATION]

- Not a new attention sparsification scheme: the sparsity acts on **feed-forward expert weights**.
- Not a reduction to one total expert: every routed token uses one expert, but different tokens choose different experts.
- Not proof that sparse routing is always faster on every accelerator.
- Not proof that more parameters always outperform carefully tuned small models.
- Not proof that each expert captures a clean human-labeled semantic category.
- Not a financial-factor model or a cross-sectional stock-rank predictor.

---

# PART II — MATHEMATICAL AND ARCHITECTURAL MECHANISM

## 009. Dense Transformer FFN Baseline

[INTERPRETATION]

A standard transformer block combines self-attention and position-wise feed-forward computation. The dense FFN may be written as:

\[
\mathrm{FFN}(x)=W_2\phi(W_1x+b_1)+b_2.
\]

The same matrices are applied to each token, giving the dense model a shared transformation.

## 010. What Switch Replaces

[SOURCE; Figure 2, PDF p.4]

Only selected FFN layers are replaced with a *Switch FFN* containing independent FFN experts and a router. The self-attention block remains the same in the default architecture.

The paper often places an expert FFN at **every other** FFN layer. Larger Switch-C is an exception (see Table 9).

## 011. Per-Token Inputs and Outputs

Let:

- \(x\in\mathbb R^{d_{model}}\): token representation;
- \(N\): number of experts;
- \(E_i(x)\in\mathbb R^{d_{model}}\): expert \(i\)'s FFN output;
- \(W_r\in\mathbb R^{N\times d_{model}}\): router projection;
- \(p_i(x)\): full-softmax probability assigned to expert \(i\).

Each expert has its own parameters; standard Switch experts share an architecture, not weights.

## 012. Router Probability

[SOURCE; Eq. (1), §2.1, PDF p.3]

\[
 h(x)=W_rx,
\]

\[
 p_i(x)=\frac{e^{h_i(x)}}{\sum_{j=1}^{N} e^{h_j(x)}}.
\]

The router produces a normalized probability distribution across all experts.

## 013. Baseline Top-k MoE Output

[SOURCE; Eq. (2), PDF p.4]

\[
 y=\sum_{i\in \mathcal T_k(x)}p_i(x)E_i(x).
\]

\(\mathcal T_k(x)\) is the set of selected expert indices.

The equation does not imply that the selected gates are renormalized to sum to one over selected experts.

## 014. Switch Top-1 Selection

[SOURCE; §2.1]

\[
 i^*(x)=\underset{i}{\arg\max}\;p_i(x).
\]

Only \(E_{i^*(x)}\) is computed for that token.

## 015. Switch Output Formula

[SOURCE; Eq. (2) with k=1]

\[
 \boxed{y=p_{i^*(x)}(x)\,E_{i^*(x)}(x)}.
\]

[INTERPRETATION] The full-softmax gate is a confidence-scaled multiplier, even though only one expert executes.

## 016. The Crucial Top-1 Gradient Detail

[SOURCE; §2.1] The authors argue that Top-1 routing remains trainable because the chosen expert output is multiplied by the differentiable gate probability.

[DERIVATION] For a fixed winner \(i^*\), ignore the nondifferentiability of the identity-selection boundary:

\[
\frac{\partial y}{\partial h_j}
=
E_{i^*}(x)\,\frac{\partial p_{i^*}}{\partial h_j}
=
E_{i^*}(x)\,p_{i^*}(\mathbf 1[j=i^*]-p_j).
\]

The selected gate has a gradient even with just one executed expert. The exact argmax identity remains piecewise constant and does not become smoothly differentiable.

## 017. Warning: Top-1 Renormalization Would Change This Mechanism

[DERIVATION/CRITIQUE]

If a programmer mistakenly normalizes the single surviving gate by its own value, then:

\[
 \frac{p_{i^*}}{p_{i^*}}=1
\]

and the gate multiplier becomes constant. This removes the mechanism highlighted in §2.1.

**Implementation invariant:** selecting Top-1 and retaining the selected full-softmax probability are two separate operations.

## 018. Why Top-1 Rather Than Top-2?

[SOURCE; PDF p.4]

The authors claim three benefits:

1. One FFN call rather than two reduces expert compute.
2. Reduced activations and per-expert assignment demand permit smaller expert capacity.
3. Dispatch and interdevice communication are simpler.

An important historical detail: the original Shazeer et al. 2017 discussion conjectured that \(k>1\) helped gating gradients, which Switch challenges empirically.

## 019. Top-1 Is Not Automatically Better in Every Setting

[CRITIQUE]

Top-1 reduces compute and simplifies routing, but also:

- loses the direct averaging of multiple expert predictions;
- may be brittle under ambiguous expert assignments;
- can be sensitive to a single poorly trained expert;
- magnifies overload when one expert dominates;
- can be harder to use for complementary shared/specialized factor mixtures.

These are research hypotheses, not established failures of the paper's reported models.

## 020. Router Pseudocode (Conceptual)

[DERIVATION from Eqs. (1)–(2); excludes distributed dispatch]

```python
logits = linear_router(x.float())           # [B, N]
probs = logits.softmax(dim=-1)              # [B, N]
selected_prob, selected_id = probs.max(-1)  # [B], [B]
# Dispatch each x to exactly one corresponding FFN.
selected_output = evaluate_selected_experts(x, selected_id)
y = selected_output * selected_prob[..., None]
```

This represents the mathematical idea, not a drop-in reproduction of Mesh TensorFlow code.

## 021. Expert Capacity: Definition

[SOURCE; Eq. (3), PDF p.5]

Let \(B\) denote the total number of tokens being dispatched in the relevant batch and \(N\) the expert count:

\[
\boxed{C=\frac{B}{N}\times\mathrm{capacity\_factor}}.
\]

In an implementation \(C\) must be made an integer, with the rounding and sharding convention checked in code.

## 022. Why Capacity Factor Exists

[SOURCE; Figure 3, PDF p.5]

Tokens are rarely distributed exactly uniformly. A factor above 1 allocates slack so busy experts can receive more than the average number of tokens.

- Lower factor: lower padded compute/memory, greater overflow risk.
- Higher factor: lower overflow risk, more idle slots/communication.

## 023. Overflow and Dropped Tokens

[SOURCE; §2.2, PDF p.6]

When more tokens choose an expert than fit in its capacity:

1. The overflowing token is **not processed by that expert**.
2. Its representation passes to the next layer through the residual connection.
3. The token is **not removed from the training example or from the data set**.

Do not conflate expert-layer token dropping with dropout regularization or dropped training samples.

## 024. Why Dropped-Token Rate Matters

[SOURCE] The authors state that keeping token drops low helps sparse scaling; in their experiments typical dropped-token fractions were below 1% when load balancing was adequate.

[CRITIQUE] The paper's empirical figure is not a universal guarantee for a new small-batch setting.

## 025. Overflow Can Distort Expert Training

[DERIVATION/INTERPRETATION]

A popular expert may receive more attempted assignments than its capacity and have some of those examples bypassed. Consequently, the effective per-expert training distribution differs from the router's nominal assignment distribution.

For finance, monitor both intended routing and actual executed routing.

## 026. Expert Capacity vs. Parameter Capacity

[INTERPRETATION]

- **Parameter capacity:** how many expert weights exist in the model.
- **Expert capacity:** how many tokens a single expert can process in a batch.
- **Active parameters:** parameters of experts actually used by a given token.

These are different quantities; keep separate metrics.

## 027. Differentiable Auxiliary Load Balancing

[SOURCE; Eqs. (4)–(6), PDF p.6]

\[
\boxed{\mathcal L_{\mathrm{balance}}
=\alpha N\sum_{i=1}^N f_iP_i.}
\]

The two vectors:

\[
 f_i=\frac{1}{T}\sum_{x\in\mathcal B}\mathbf1[\arg\max_j p_j(x)=i],
\]

\[
 P_i=\frac{1}{T}\sum_{x\in\mathcal B}p_i(x).
\]

\(T=|\mathcal B|\) is the number of tokens.

## 028. Meaning of f and P

[SOURCE]

- \(f_i\): fraction of tokens routed to expert \(i\), under Top-1 decision.
- \(P_i\): mean router *probability mass* assigned to expert \(i\), aggregated over the batch.

The former is hard and piecewise discrete. The latter is differentiable with respect to router logits.

## 029. Why Their Product Is Used

[INTERPRETATION] A popular expert typically receives both high selected count and high probability. A positive product \(f_iP_i\) penalizes high joint concentration, encouraging probability mass to move toward less selected experts.

The authors describe this as a differentiable surrogate despite \(f\) itself being nondifferentiable.

## 030. Balanced Reference Value

[DERIVATION]

For uniform routing:

\[
 f_i=P_i=1/N.
\]

Therefore:

\[
 N\sum_i f_iP_i=N\cdot N\cdot\frac{1}{N^2}=1
\]

and:

\[
\mathcal L_{\mathrm{balance}}=\alpha.
\]

Do not mistake \(\alpha\) for a zero-valued balance loss at the balanced solution.

## 031. Caveat on Global Minimization

[CRITIQUE] If \(f\) is held fixed as an arbitrary distribution while optimizing \(P\) without accounting for the dependence of hard assignments on logits, the dot product alone does not guarantee a uniform globally optimal solution. The paper presents a practical surrogate and reports empirical balancing; it does not prove universal optimization guarantees.

## 032. Paper's Default Balance Coefficient

[SOURCE; PDF p.6]

\[
\alpha=10^{-2}.
\]

The authors sweep powers of ten from \(10^{-1}\) to \(10^{-5}\) and report \(10^{-2}\) balances load without overwhelming primary cross-entropy on their experiments.

It is not a universal financial-model default.

## 033. Balance Objective and Total Loss

[DERIVATION]

\[
\mathcal L_{\mathrm{total}}
=\mathcal L_{\mathrm{task}}+
\mathcal L_{\mathrm{balance}}.
\]

In the original pre-training study, task loss comes from language-token prediction. In a finance adaptation, task loss could instead be a regression or rank objective, but such variants are not evaluated in the paper.

## 034. Original Shazeer 2017 vs. Switch Balance

[SOURCE; §2.2]

Shazeer 2017 uses distinct importance and load losses. Switch adopts a *single* auxiliary expression involving the vector of chosen-token fractions \(f\) and the mean probability vector \(P\).

Hence a later PRISM-VQ-style \(M\sum_j f_jP_j\) is mechanistically closer to Switch's loss than to the two-CV-loss formulation of Shazeer 2017.

## 035. Gradient Explanation for Load Balance

[DERIVATION] With hard count \(f_i\) treated as constant on a backward step,

\[
\frac{\partial\mathcal L_{balance}}{\partial h_{t,j}}
=\frac{\alpha N}{T}\sum_i f_i\frac{\partial p_{t,i}}{\partial h_{t,j}}
=\frac{\alpha N}{T}\,p_{t,j}\left(f_j-\sum_i f_ip_{t,i}\right).
\]

This illuminates how the differentiable probability vector \(P\) can move despite the hard dispatch decisions.

## 036. Correct Metric for Balance Diagnosis

[EXTENSION]

In addition to auxiliary loss values, track:

- pre-capacity Top-1 routing histogram;
- post-capacity executed histogram;
- per-expert mean probabilities;
- max-to-mean assignment ratio;
- coefficient of variation of assignments;
- fraction of tokens skipped due to overflow;
- utilization stability across steps/seeds.

The single scalar balance loss is not a complete substitute for those diagnostics.

## 037. Figures 2 and 3 Must Be Read Together

[SOURCE; PDF pp.4–5]

- Figure 2: what replaces a Transformer FFN and how two tokens can choose two different experts.
- Figure 3: why routing decisions interact with finite expert capacity and fixed-shaped distributed buffers.

[INTERPRETATION] Architecture and capacity policy cannot be separated when benchmarking wall-clock speed.

---

# PART III — STABILITY AND OPTIMIZATION

## 038. Instability Sources

[SOURCE; §2.4]

1. Hard switching changes which expert receives which gradient.
2. Router softmax in low precision can be numerically unstable.
3. Large expert weights and router logits can destabilize training.
4. Overfitting increases during downstream fine-tuning.

## 039. Selective Precision: Main Idea

[SOURCE; §2.4, PDF pp.8–9]

Use float32 **inside the router operations** while storing/communicating the rest of the model using bfloat16.

```
bfloat16 token state
      |
      v
float32 local router / softmax
      |
      v
expert selection + dispatch/combine
      |
      v
cast dispatch/combine to bfloat16
      |
      v
bfloat16 expert communication and computation
```

## 040. Why Not Float32 Everywhere?

[SOURCE] Communication of float32 tensors across devices increases costs. The authors constrain the float32 computation to a local subroutine and recast before expensive all-to-all exchange.

## 041. Selective Precision Table (Original Table 2)

[SOURCE; PDF p.8]

| Training mode | Quality: negative log perplexity (higher better) | Speed (examples/sec) |
|---|---:|---:|
| Switch-Base float32 | -1.718 | 1160 |
| Switch-Base bfloat16 | -3.780 (diverged) | 1390 |
| Switch-Base selective precision | -1.716 | 1390 |

[INTERPRETATION] The stabilizing intervention achieved near-float32 quality at bfloat16 speed in this particular 32-expert experiment.

## 042. Reduced Initialization Scale

[SOURCE; §2.4, PDF p.9]

The authors initialize weights from a truncated normal with mean zero and a scale depending on fan-in, and recommend multiplying the standard Transformer initialization scale by 0.1.

[VERIFY] Carefully follow the PDF/code for the precise fan-in square-root notation in an implementation; the PDF states a scaled, fan-in-dependent standard deviation and the factor-of-ten reduction is the robust reported recommendation.

## 043. Initialization Ablation (Original Table 3)

[SOURCE; PDF p.9; three seeds; 3.5k steps]

| Init scale | Mean Neg. Log Perp. (higher better) | Std. dev. across runs |
|---|---:|---:|
| 0.1× | -2.72 | 0.01 |
| 1.0× | -3.60 | 0.68 |

Interpretation: the reduced scale improves both early training quality and seed-to-seed stability.

## 044. Expert Dropout During Fine-Tuning

[SOURCE; §2.4, PDF pp.9–10]

Apply stronger dropout inside expert FFN layers than in the shared non-expert layers.

- Non-expert dropout: 0.1.
- Expert dropout: 0.4 in the highlighted variant.

This is targeted regularization of conditionally activated parameter blocks, not model-wide uniform dropout.

## 045. Why Expert Dropout?

[SOURCE] Sparse models may have much more total parameter capacity than a FLOP-matched dense model and can overfit small fine-tuning tasks. Stronger expert-specific regularization is proposed to alleviate this.

## 046. Expert-Dropout Ablation (Original Table 4)

[SOURCE; PDF p.10]

| Model / dropout | GLUE | CNNDM | SQuAD | SuperGLUE |
|---|---:|---:|---:|---:|
| T5-Base d=0.1 | 82.9 | 19.6 | 83.5 | 72.4 |
| Switch-Base d=0.1 | 84.7 | 19.1 | 83.7 | 73.0 |
| Switch-Base d=0.2 | 84.4 | 19.2 | 83.9 | 73.2 |
| Switch-Base d=0.3 | 83.9 | 19.6 | 83.4 | 70.7 |
| Switch-Base d=0.1, expert-d=0.4 | 85.2 | 19.6 | 83.7 | 73.0 |

[CRITIQUE] The phrase "best on four tasks" should be nuanced: the expert-dropout configuration ties some results and is not best on every individual metric in this small table.

## 047. Stability Does Not Equal Generalization

[INTERPRETATION] Router float32 and smaller initial weights address numerical/optimization instability. Expert dropout addresses overfitting. These are distinct failure modes and should not be collapsed into one generic "robustness" effect.

## 048. Small-Scale Applicability

[SOURCE; Appendix D] The paper reports improvements over T5-Base even with 2, 4, or 8 experts under compute-matched settings.

[CRITIQUE] This is evidence that Switch need not require thousands of devices, not evidence that Top-1 MoE always improves a stock predictor with only a few years of daily data.

---

# PART IV — EXPERIMENTS, COMPARATORS, AND NUMERICAL EVIDENCE

## 049. Research Questions Answered Empirically

[INTERPRETATION]

The paper's results test at least seven separable hypotheses:

1. Does Top-1 beat Top-2 at a favorable speed–quality operating point?
2. Can low expert capacity factors work once load balancing is applied?
3. Do additional experts improve quality under fixed FLOPs/token?
4. Do per-step benefits persist on a wall-clock basis?
5. Do pre-training advantages transfer to supervised downstream tasks?
6. Can huge sparse teachers be compressed into useful dense students?
7. Does sparsity scale in multilingual and trillion-parameter settings?

## 050. The Main Data Protocol

[SOURCE; §§2.3–5]

Most experiments use language-related datasets:

- C4: cleaned web-crawl text for pre-training;
- GLUE, SuperGLUE, SQuAD, Winogrande, XSum, ANLI, ARC, closed-book QA for downstream evaluations;
- mC4 with 101 languages (107 script/variant tasks) for multilingual experiments.

Do not invent one universal train/validation/test split across these heterogeneous datasets. The paper has different protocols for pre-training, fine-tuning, validation, and multilingual tasks.

## 051. Pre-Training Objective

[SOURCE; §2.3]

T5-style span denoising: mask 15% of text tokens and replace masked spans with special sentinel tokens; train to recover missing material.

Main pre-training quality metric is **negative log perplexity**, for which the main tables/figures mark **higher as better** (less negative is better).

## 052. Evaluation Units Matter

[SOURCE/INTERPRETATION]

- `examples/sec`: observed throughput on specified hardware.
- `quality after 100k steps`: learning quality at fixed optimizer steps.
- `time to quality`: wall-clock duration to reach a specified loss threshold.
- `FLOPs/token`: model arithmetic per token, excluding communication.
- `number of parameters`: weight storage/capacity, not active inference FLOPs.

A large sparse model can win on one and lose on another.

## 053. Original Table 1: Switch vs. Top-2 MoE vs. Dense

[SOURCE; PDF p.7]

| Model | Capacity factor | Quality at 100k steps (Neg. Log Perp. ↑) | Hours to threshold ↓ | Examples/sec ↑ |
|---|---:|---:|---:|---:|
| T5-Base | — | -1.731 | not achieved within 100k steps | 1600 |
| T5-Large | — | -1.550 | 131.1 | 470 |
| MoE-Base | 2.0 | -1.547 | 68.7 | 840 |
| Switch-Base | 2.0 | -1.554 | 72.8 | 860 |
| MoE-Base | 1.25 | -1.559 | 80.7 | 790 |
| Switch-Base | 1.25 | -1.553 | 65.0 | 910 |
| MoE-Base | 1.0 | -1.572 | 80.1 | 860 |
| Switch-Base | 1.0 | -1.561 | **62.8** | **1000** |
| Switch-Base+ | 1.0 | **-1.534** | 67.6 | 780 |

All MoE and Switch variants in this table use 128 experts, expert FFN at every other FFN layer, 32 TPUv3 cores.

## 054. The Quality Threshold in Table 1

[SOURCE] The threshold is negative log perplexity \(-1.50\). The paper calls it an arbitrarily chosen fixed-quality point. Numbers in the time column are the time to this threshold, not duration to finish exactly 100k steps.

## 055. Nuance: Top-2 Wins One Equal-Capacity Comparison

[SOURCE; Table 1]

At capacity factor 2.0:

- Top-2 MoE: -1.547 at 100k steps, 68.7 hours to threshold;
- Top-1 Switch: -1.554 at 100k steps, 72.8 hours.

The Top-2 MoE is slightly better by those measures in this row.

[CRITIQUE] Do **not** rewrite the result as "Switch Top-1 beats Top-2 at every capacity factor and every metric."

## 056. Why Top-1 Looks Stronger at Low Capacity

[SOURCE; Table 1]

At capacity factor 1.0:

- Top-1 is faster in examples/sec (1000 vs. 860);
- Top-1 has higher quality (-1.561 vs. -1.572);
- Top-1 reaches threshold sooner (62.8 vs. 80.1 hours).

Thus the speed–quality advantage depends on the operating point and capacity budget.

## 057. What Switch-Base+ Changes

[SOURCE; Table 1 note]

Switch-Base+ increases width so the model's observed examples/sec approximately matches a Top-2 MoE's speed. It uses a larger model hidden dimension and more heads compared with Switch-Base; it is not identical in architecture size.

[VERIFY] The source's note reports changing model hidden size from 768 to 896 and heads "14 to 16"; check paper-specific configs instead of interpreting this as a universal T5-Base head-count specification.

## 058. Head-to-Head Fairness Caveats

[INTERPRETATION]

The paper reports both FLOP matching and throughput matching in different comparisons. The Top-2 MoE generally performs more expert operations per token. A fixed-core comparison does not guarantee matched active computation unless the authors explicitly state FLOP matching.

For any new comparison, report both active FLOPs and real wall-clock time.

## 059. Original Figure 1: Scaling/Sample Efficiency Overview

[SOURCE; PDF p.2]

Figure 1 left shows lowering test loss as model sparse parameters grow, and right compares negative log perplexity versus training step between T5-Base and Switch models with varying expert counts.

The main message is that expert count is an additional dimension for scaling parameter capacity at approximately fixed active computation.

## 060. Original Figure 4: Step-Based Scaling

[SOURCE; PDF p.11]

Figure 4 left increases the number of experts from 2 through 256; the 256-expert Switch-Base has about 14.7B parameters in this setting. Right compares learning curves across expert counts.

The paper reports that 64-expert Switch-Base reaches the T5-Base reference performance in about 60k steps rather than 450k.

## 061. Step-Based Speedup Is Not a Wall-Clock Speedup

[DERIVATION]

\[
\frac{450{,}000}{60{,}000}=7.5.
\]

That is a step/sample-efficiency ratio. Actual wall-clock improvement must account for tokens/sec, distributed communication, and optimizer overhead.

## 062. Original Figure 5: Wall-Clock Comparison

[SOURCE; PDF p.12]

On 32 TPUv3 cores and equal FLOPs per example, 64-expert Switch-Base reaches the same selected quality in approximately **one-seventh the time** required by dense T5-Base.

This is an empirical wall-clock improvement for the reported setup, not a promise for a single-GPU finance experiment.

## 063. Original Figure 6: Against T5-Large

[SOURCE; PDF p.13]

T5-Large uses about 3.5× the FLOPs/token of T5-Base. The smaller-FLOP Switch-Base is still about 2.5× faster to the chosen quality than T5-Large in this setup.

## 064. Cost-Quality Pareto Claim

[INTERPRETATION] The key empirical object is a Pareto frontier over:

\[
(\text{model quality},\text{training time},\text{FLOPs/token},\text{memory}).
\]

Comparing only held-out perplexity without the cost axes is insufficient to reconstruct this paper's argument.

## 065. Pre-Training Has a Large Data Regime

[SOURCE; §3]

The C4 corpus has over 180 billion target tokens; the scaling experiments are designed to avoid being immediately constrained by the available data.

[CRITIQUE] This assumption is sharply different from 2009–2025 daily equity observations.

## 066. Pre-Training Architecture Baselines

[SOURCE] Principal comparisons use:

- T5-Base;
- T5-Large;
- T5-XXL;
- Top-2 MoE Transformer;
- Switch variants with different expert counts/capacity factors;
- Switch-Base+;
- sparse Switch-C / Switch-XXL at giant scale.

This is *not* a comparison against financial GRU, MASTER, or PRISM-VQ baselines.

## 067. Original Table 5: Main Fine-Tuning Results (Part 1)

[SOURCE; PDF p.14; validation sets]

| Model | GLUE | SQuAD | SuperGLUE | Winogrande XL |
|---|---:|---:|---:|---:|
| T5-Base | 84.3 | 85.5 | 75.1 | 66.6 |
| Switch-Base | 86.7 | 87.2 | 79.5 | 73.3 |
| T5-Large | 87.8 | 88.1 | 82.7 | 79.1 |
| Switch-Large | 88.5 | 88.6 | 84.7 | 83.0 |

## 068. Original Table 5: Main Fine-Tuning Results (Part 2)

[SOURCE; PDF p.14]

| Model | XSum | ANLI R3 | ARC Easy | ARC Challenge |
|---|---:|---:|---:|---:|
| T5-Base | 18.7 | 51.8 | 56.7 | 35.5 |
| Switch-Base | 20.3 | 54.0 | 61.3 | 32.8 |
| T5-Large | 20.9 | 56.6 | 68.8 | 35.5 |
| Switch-Large | 22.3 | 58.6 | 66.0 | 35.5 |

## 069. Original Table 5: Main Fine-Tuning Results (Part 3)

[SOURCE; PDF p.14]

| Model | Closed-book Web QA | Closed-book Natural QA | Closed-book Trivia QA |
|---|---:|---:|---:|
| T5-Base | 26.6 | 25.8 | 24.5 |
| Switch-Base | 27.4 | 26.8 | 30.7 |
| T5-Large | 27.7 | 27.6 | 29.5 |
| Switch-Large | 31.3 | 29.5 | 36.9 |

## 070. Fine-Tuning Metrics Are Not All the Same

[SOURCE; §4.1]

- GLUE / SuperGLUE: composite averages.
- SQuAD and closed-book QA: exact match.
- XSum / CNNDM: ROUGE-2.
- ARC, ANLI, Winogrande: accuracy.

Do not average these raw numbers into one composite unless a scientifically defined aggregation is provided.

## 071. Fine-Tuning Protocol

[SOURCE; PDF p.13]

- Baselines include 223M T5-Base and 739M T5-Large.
- Main comparable pre-training examples: approximately \(2^{20}\) tokens/batch, 550k steps, 576B total tokens.
- Downstream fine-tuning: batch approximately 1M tokens, up to 16k steps.
- Model quality is evaluated every 200 steps and peak validation value reported.
- Expert dropout configuration is applied during fine-tuning.

## 072. Negative Results in Downstream Fine-Tuning

[SOURCE; PDF pp.14–15]

Switch is **not** best on all tasks:

- ARC Challenge: T5-Base 35.5 vs Switch-Base 32.8.
- ARC Easy: T5-Large 68.8 vs Switch-Large 66.0.
- ARC Challenge: T5-Large and Switch-Large both 35.5.

[CRITIQUE] The broad statement "all downstream tasks improve" would be incorrect.

## 073. Distillation Research Question

[SOURCE; §4.2]

Can a sparse Transformer teacher with many parameters train a deployable dense student that preserves some of the sparse model's quality advantage?

## 074. Distillation Methods

[SOURCE]

The authors compare:

1. Basic teacher distillation.
2. Initializing student non-expert layers with the teacher's weights.
3. A mixed objective with 0.75 ground-truth labels and 0.25 teacher probabilities.
4. Combined initialization and mixed objective.

## 075. Original Table 6: Distillation

[SOURCE; PDF p.16]

| Technique | Student/teacher parameters | Neg. Log Perp. ↑ | Preserved gain |
|---|---:|---:|---:|
| T5-Base | 223M | -1.636 | baseline |
| Switch-Base | 3,800M | -1.444 | teacher |
| Basic distillation | 223M | -1.631 | 3% |
| + non-expert teacher initialization | 223M | -1.598 | 20% |
| + 0.75 hard / 0.25 soft loss | 223M | -1.580 | 29% |
| Teacher non-expert init without distillation | 223M | -1.639 | worse than baseline |

## 076. Distillation Percentage Definition

[SOURCE/DERIVATION]

The preserved-gain figure measures the fraction of teacher **quality improvement over dense baseline** retained by the student, not the fraction of teacher perplexity or total predictive accuracy.

Illustration using the best student:

\[
\frac{-1.580-(-1.636)}{-1.444-(-1.636)}
=\frac{0.056}{0.192}\approx29.2\%.
\]

## 077. Original Table 7: Compression Sweep

[SOURCE; PDF p.16]

| Metric | Dense T5-Base | Switch 1.1B | Switch 2.0B | Switch 3.8B | Switch 7.4B | Switch 14.7B |
|---|---:|---:|---:|---:|---:|---:|
| Pre-trained quality ↑ | -1.636 | -1.505 | -1.474 | -1.444 | -1.432 | -1.427 |
| Distilled quality ↑ | — | -1.587 | -1.585 | -1.579 | -1.582 | -1.578 |
| Fraction of teacher gain | — | 37% | 32% | 30% | 27% | 28% |
| Compression % | — | 82% | 90% | 95% | 97% | 99% |

[INTERPRETATION] Much of the extra sparse-model gain does not fit into a fixed small dense student, even as sparse-teacher quality keeps improving.

## 078. Distillation Is Not Full Recovery

[SOURCE] The paper highlights ~30% preservation of the teacher quality advantage, not 100% recovery.

[CRITIQUE] In finance, an apparently successful sparse teacher might need a broader student architecture if its signal genuinely depends on multiple specialized regimes.

## 079. Fine-Tuned Distillation (Original Table 8)

[SOURCE; PDF p.17]

| Model | Parameters | FLOPs | SuperGLUE |
|---|---:|---:|---:|
| T5-Base | 223M | 124B | 74.6 |
| Switch-Base | 7410M | 124B | 81.3 |
| Distilled T5-Base | 223M | 124B | 76.6 |

This is again approximately 30% retention of the teacher's gain over the dense student baseline.

## 080. Multilingual Research Question

[SOURCE; §4.3]

Does a common sparse model increase sample efficiency across 101 languages, including underrepresented languages, compared with a common dense mT5-Base?

## 081. Multilingual Data

[SOURCE]

mC4 spans 101 languages. Because of script variants, the paper refers to 107 tasks in the training mixture. The main comparison uses 1M pre-training steps.

## 082. Original Figures 7 and 8: Multilingual Results

[SOURCE; PDF pp.17–18]

- Every one of 101 languages improves in negative log perplexity relative to mT5-Base in this comparison.
- Mean step speedup to dense-target quality is approximately 5×.
- 91% of languages have speedup of at least 4×.

These are **language-pre-training** results and do not imply 101 downstream tasks are uniformly better.

## 083. Why Multilingual Results Matter for the Mechanism

[INTERPRETATION] Multiple languages provide related but heterogeneous sub-distributions. Sparse conditional computation can share common transformations while dedicating distinct expert parameters to different input types.

A finance analogy would be heterogeneous market states, sectors, or asset classes. But the existence of the analogy is not equivalent to empirical transfer.

## 084. Large-Scale Capacity vs. FLOPs

[SOURCE; §5.6]

Two large architectures:

- Switch-XXL: about 395B parameters, FLOP-matched with T5-XXL.
- Switch-C: about 1.571T parameters with much lower FLOPs/sequence than T5-XXL.

The exact parameter and computation comparison is central: total capacity can grow while active arithmetic remains controlled.

## 085. Original Table 9: Model Dimensions, Part 1

[SOURCE; PDF p.22]

| Model | Parameters | FLOPs/seq | d_model | GEGLU FFN? | d_ff | d_kv | Heads |
|---|---:|---:|---:|---|---:|---:|---:|
| T5-Base | 0.2B | 124B | 768 | yes | 2048 | 64 | 12 |
| T5-Large | 0.7B | 425B | 1024 | yes | 2816 | 64 | 16 |
| T5-XXL | 11B | 6.3T | 4096 | yes | 10240 | 64 | 64 |
| Switch-Base | 7B | 124B | 768 | yes | 2048 | 64 | 12 |
| Switch-Large | 26B | 425B | 1024 | yes | 2816 | 64 | 16 |
| Switch-XXL | 395B | 6.3T | 4096 | yes | 10240 | 64 | 64 |
| Switch-C | 1571B | 890B | 2080 | no / blank checkbox | 6144 | 64 | 32 |

Note: paper's precise output head/style in the printed table is preserved; do not extrapolate to non-T5 models.

## 086. Original Table 9: Model Dimensions, Part 2 — IMPORTANT PRINTING ISSUE

[SOURCE; PDF p.22]

The physical table prints headings in this order:

`Expert Freq. | Num. Experts | Num Layers | Neg.Log.Perp @250k | Neg.Log.Perp @500k`.

But the numbers in the second and third columns appear semantically swapped: e.g., T5-Base displays `— | 12 | —` and Switch-Base displays `1/2 | 12 | 128`. Those represent 12 **layers** and 128 **experts**, respectively, based on model descriptions and Table 1.

[VERIFY] Preserve the source table's ordering and add this warning; use the interpreted architecture fields only after confirming in released code.

| Model | Expert layer frequency | Layers (interpreted from printed 2nd column) | Experts (interpreted from printed 3rd column) | Quality @250k ↑ | @500k ↑ |
|---|---:|---:|---:|---:|---:|
| T5-Base | — | 12 | — | -1.599 | -1.556 |
| T5-Large | — | 24 | — | -1.402 | -1.350 |
| T5-XXL | — | 24 | — | -1.147 | -1.095 |
| Switch-Base | 1/2 | 12 | 128 | -1.370 | -1.306 |
| Switch-Large | 1/2 | 24 | 128 | -1.248 | -1.177 |
| Switch-XXL | 1/2 | 24 | 64 | **-1.086** | **-1.008** |
| Switch-C | 1 | 15 | 2048 | -1.096 | -1.043 |

## 087. Why the Table 9 Correction Is Necessary

[VERIFY]

Literal reading of the printed header would claim:

- a dense T5 has 12 "experts" and unspecified layers;
- Switch-C has 15 "experts" and 2048 layers.

That directly conflicts with the prose's "2048 experts" Switch-C description. The safest agent behavior is to store both the printed and logically interpreted readings with a source-error flag.

## 088. Switch-XXL vs. Switch-C Trade-Off

[SOURCE; §§5.6, 8]

Switch-XXL: fewer total parameters but far more FLOPs per sequence.

Switch-C: more total parameters and lower FLOPs per sequence.

Empirically, the more computationally intensive Switch-XXL achieves better pre-training quality at 250k/500k steps than the far larger Switch-C.

## 089. More Parameters Do Not Guarantee Better Upstream Quality

[SOURCE; Table 9]

At 500k steps:

\[
Q_{Switch\text{-}XXL}=-1.008
\]

versus:

\[
Q_{Switch\text{-}C}=-1.043.
\]

Because higher negative log perplexity is better, the 395B model is better here than the 1.571T model.

[INTERPRETATION] Active computation and architecture depth/width also matter.

## 090. Giant-Model Training Stability Nuance

[SOURCE; PDF p.22]

The authors report that Switch-C did not exhibit training instability in their runs, whereas Switch-XXL sometimes did. Thus the higher-FLOP model was not necessarily easier to stabilize.

## 091. Giant-Model Downstream Limitations

[SOURCE; pp.22–23]

Partially pre-trained Switch-XXL:

- SQuAD validation: 89.7 vs a reported prior SOTA 91.3;
- SuperGLUE test: 87.5 vs a cited T5 value 89.3 and other SOTA 90.0;
- ANLI: 65.7 vs prior result 49.4;
- closed-book QA often improves over cited T5-XXL results.

The paper explicitly says upstream gains had **not fully translated** to all reasoning downstream benchmarks.

## 092. Closed-Book QA Comparisons at Giant Scale

[SOURCE; PDF p.23]

| Task | Switch-XXL reported | Prior T5-XXL comparison |
|---|---:|---:|
| Natural Questions | 34.4 | 32.8 |
| Web Questions | 41.0 | 37.2 |
| TriviaQA | 47.5 | 42.9 |

[VERIFY] Cross-paper comparisons can differ in training data, methods, and checkpoints. Do not infer a completely matched head-to-head ablation from these numbers.

## 093. How to Read Table 9 Correctly

[INTERPRETATION]

At equal FLOPs/seq:

- T5-XXL 11B / 6.3T;
- Switch-XXL 395B / 6.3T.

This cleanly illustrates more total parameters at matched active compute.

At unequal FLOPs/seq:

- Switch-C 1571B / 890B, with a different dimension/architecture profile.

The latter demonstrates a different efficiency point and should not be represented as strictly FLOP-matched to Switch-XXL.

## 094. How Model Scale Was Distributed

[SOURCE; §5]

The paper describes combinations of:

- data parallelism;
- model parallelism;
- expert parallelism.

Each has different communication structure. Scaling the number of experts alone does not determine total throughput.

## 095. Disentangle Paper's Scaling Claims

[INTERPRETATION]

| Claim | Grounded observation | Limitation |
|---|---|---|
| Top-1 is efficient | Table 1 with lower capacity factors | not always best every row |
| Sparse expert parameters help | Figures 1, 4 | evaluated on large NLP corpora |
| 7× pre-training time reduction | Figure 5 | specific 32-core setup |
| Sparse capacity aids multi-task | Figures 7–8 | upstream pre-training measures |
| Sparse teachers can be compressed | Tables 6–8 | only part of gain retained |
| Trillion scale is trainable | Table 9 | costly expert-parallel hardware |
| More parameters alone suffice | Not established | Switch-C vs XXL counters a simplistic view |

---

# PART V — DISTRIBUTED IMPLEMENTATION AND RESOURCE ECONOMICS

## 096. Tensor Symbols

[SOURCE; §5]

- \(B\): total tokens in batch;
- \(d_{model}\): token hidden dimension;
- \(d_{ff}\): FFN intermediate dimension;
- \(N\): total accelerator cores (in parallelism analysis);
- \(n\): data-parallel partitions;
- \(m\): model-parallel partitions;
- \(E\): total experts;
- \(C\): per-expert token capacity.

The paper reuses \(N\) for different quantities in different sections. Do not combine symbols from separate equations without renaming.

## 097. Data Parallelism

[SOURCE; §5.1]

Replicate model parameters across accelerators. Each worker receives its own subset of tokens. After backward passes, gradients are synchronized.

## 098. Model Parallelism

[SOURCE; §5.2]

Partition weight matrices along dimensions such as \(d_{ff}\). Intermediate tensors and reductions require communication between devices within layers.

## 099. Combined Data + Model Parallelism

[SOURCE; §5.3]

With \(N=n\times m\), each model-parallel group performs work on its share of tokens, while FFN hidden matrices are sharded along model dimensions.

## 100. Expert Parallelism

[SOURCE; §5.4]

Independent expert FFN parameters are placed on separate devices. Tokens are dynamically dispatched to the device hosting their selected expert. This requires **all-to-all** communication.

## 101. Router Dispatch / Combine Tensor Shapes

[SOURCE; Appendix F]

Typical shapes in the Mesh TensorFlow description:

```text
inputs:
    [num_cores, tokens_per_core, d_model]
router probabilities:
    [num_cores, tokens_per_core, num_experts]
expert mask:
    [num_cores, tokens_per_core, num_experts]
dispatch tensor:
    [num_cores, tokens_per_core, num_experts, expert_capacity]
expert input after routing:
    [num_experts, num_cores, expert_capacity, d_model]
expert output after computation:
    [num_experts, num_cores, expert_capacity, d_model]
combined output:
    [batch, sequence_length, d_model]
```

## 102. Routing Is Communication, Not Free Arithmetic

[INTERPRETATION]

Total parameter count can be inflated while active expert FLOPs stay roughly constant, but system cost includes:

- router logits over all experts;
- token permutation and packing;
- all-to-all transfer;
- load imbalance;
- padding/capacity overhead;
- optimizer state storage;
- synchronization.

MoE FLOPs efficiency does not imply identical throughput on a single 16GB GPU.

## 103. Figure 9: Partitioning Strategies

[SOURCE; PDF p.20]

Figure 9 diagrams weight and batch partitioning for data, model, combined data/model, expert/data, and expert/model/data parallelism over a 4×4 grid of accelerator cores.

It is a **systems-level illustration**, not a new probabilistic routing architecture.

## 104. Single-GPU Conditional Compute

[EXTENSION]

On one GPU, most cross-device all-to-all does not apply. But expert tensor scattering and small matrix batches can still cause inefficiency. The main engineering concern becomes achieving reasonably sized grouped matrix multiplications and avoiding Python loops over per-token assignments.

## 105. Parallelism Cost Model

[DERIVATION]

For a simple batch of \(B\) tokens, \(N\) total experts, Top-1, and an expert containing two dense matrices roughly \(d_{model}\times d_{ff}\), a conceptual active expert forward cost per token is:

\[
O(d_{model}d_{ff}),
\]

while the router has cost:

\[
O(Nd_{model}).
\]

The capacity bottleneck is controlled by occupancy, not by how many model parameters exist in storage.

## 106. Full vs. Active Expert Parameters

[DERIVATION]

Approximately:

\[
P_{experts,full}\approx N(2d_{model}d_{ff}),
\]

but per-token active expert weights under Top-1 are approximately:

\[
P_{experts,active}\approx2d_{model}d_{ff}.
\]

A top-k variant scales the active expert term by \(k\), all else equal.

## 107. Practical Compute-Matching Protocol

[EXTENSION]

For fair comparison of a new finance MoE with a dense head, report at least:

- total trainable parameters;
- per-token active parameters;
- profiler-based forward FLOPs;
- training milliseconds/date-batch;
- peak VRAM;
- tokens/stocks dispatched per expert;
- data movement and expert padding overhead where relevant.

## 108. Important Scaling-Law Qualification

[SOURCE/CRITIQUE] The paper advocates scaling expert parameters on massive text corpora; it does not establish a universal law of monotonic benefit on small specialized datasets.

When stock data are limited, added experts can reduce the number of observations updating each expert and may worsen generalization.

---

# PART VI — APPENDICES, NEGATIVE RESULTS, AND REPRODUCIBILITY

## 109. Appendix A — Switch Inside Self-Attention

[SOURCE; Appendix A, PDF pp.26–27]

The authors consider replacing the projection matrices producing Q, K, V in self-attention with Switch experts, instead of only replacing the FFN.

## 110. Motivation for Switch-Attention

[SOURCE/INTERPRETATION] Different tokens might benefit from specialized attention projections as well as specialized FFN transforms. This adds conditional computation to the mechanism deciding **what tokens attend to**, not only to post-attention processing.

## 111. Original Table 10: Switch Attention

[SOURCE; PDF p.27]

| Model | Precision | Quality @100k ↑ | Quality @16h ↑ | Speed (ex/sec) ↑ |
|---|---|---:|---:|---:|
| Experts FFN | float32 | -1.548 | -1.614 | 1480 |
| Expert Attention | float32 | -1.524 | -1.606 | 1330 |
| Expert Attention | bfloat16 | diverged | diverged | — |
| Experts FFN + Attention | float32 | **-1.513** | -1.607 | 1240 |
| Experts FFN + Attention | bfloat16 | diverged | diverged | — |

## 112. What Table 10 Actually Shows

[SOURCE/CRITIQUE]

Expert attention can improve per-step quality in float32 but is slower in throughput and exhibits bfloat16 instability. The authors therefore do **not** adopt Switch-attention as the default model.

A finance Agent should not call this a successful, stable established feature without citing the precision and speed limitations.

## 113. Appendix B — No-Token-Left-Behind

[SOURCE; PDF p.28]

This alternative iteratively reroutes overflowed tokens to their next-highest-probability available expert, potentially avoiding most skipped expert computations.

It tries to solve the question:

> If a token's first choice is full but another expert has space, why not redirect it?

## 114. The Surprising Negative Result

[SOURCE] The authors found **no empirical benefit** from No-Token-Left-Behind in their tests.

Their hypothesis is that replacing a preferred expert with a lower-ranked expert can harm learned associations, offsetting the benefits of fewer skipped tokens.

## 115. Implication of No-Token-Left-Behind Failure

[INTERPRETATION] Routing according to learned expert affinity can matter more than simply minimizing the number of bypassed tokens. A capacity-fallback mechanism is not automatically beneficial.

## 116. Appendix C — Exploration Versus Exploitation

[SOURCE; PDF p.29]

The router sees what happens for the selected expert, but not the counterfactual loss had it chosen another expert. The authors compare this with a contextual-bandit-like exploration problem.

This motivates studying modest randomization during training.

## 117. Router Exploration Variants

[SOURCE]

1. Deterministic argmax.
2. Sampling from router softmax.
3. Input dropout.
4. Input jitter.

## 118. Original Table 11: Routing Exploration

[SOURCE; PDF p.29]

| Strategy | Quality (Neg. Log Perp., ↑ by header) |
|---|---:|
| Argmax | -1.471 |
| Sample softmax | -1.570 |
| Input dropout | -1.480 |
| **Input jitter** | **-1.468** |

The text says jitter performs best and uses it in their reported training setups.

## 119. Table 11 Internal Annotation Conflict

[VERIFY]

Table 11's *metric header* explicitly has an upward arrow next to negative log perplexity, yet the *caption* says "lower is better". For negative log perplexity as used in the paper, higher (less negative) is better, matching boldface on -1.468.

The correct source-preserving response is to quote both conventions and flag the caption as inconsistent. Do not silently replace the actual numbers.

## 120. Appendix C's Strong Negative Evidence Against Softmax Sampling

[SOURCE; Table 11]

Sampling directly from the full softmax is clearly worse than argmax in the reported experiment:

\[
-1.570\;<\;-1.471.
\]

The paper supports small input jitter as an exploration strategy, not generic unconstrained stochastic dispatch.

## 121. Jitter-Description Reproduction Caveat

[VERIFY; Appendix C versus Appendix F]

Appendix C calls its perturbation **multiplicative jitter** on incoming representations. Appendix F's illustrative router code includes uniform noise added to router logits. The descriptions need implementation inspection to recover the exact source of randomness, its shape, and where it is applied.

For the Agent: keep "input jitter selected" as a source fact; mark the exact injected-noise implementation `VERIFY_IN_CODE`.

## 122. Appendix D — Few-Expert Regime

[SOURCE; PDF p.30]

Figure 12 compares Switch-Base with 2, 4, and 8 experts against T5-Base under the paper's controlled compute setting.

The figure shows improved pre-training trajectories even for very small expert counts.

## 123. Few Experts ≠ No Specialization Question

[INTERPRETATION] With only two experts, useful conditional computation is still possible, but all observed gains must be distinguished from:

- trivial widening of a parameterized FFN;
- stochastic regularization;
- routing-induced ensemble effects;
- data-dependent specialization.

A stock-model experiment requires additional controls to isolate which effect is responsible.

## 124. Appendix E — Upstream Quality Versus Downstream Quality

[SOURCE; PDF p.31]

Figure 13 plots C4 pre-training negative log perplexity against SuperGLUE and TriviaQA quality.

- Better upstream quality broadly correlates with better downstream quality.
- For a fixed upstream score, the dense model can be better on some reasoning tasks at large scale.
- Sparse models can look favorable on knowledge-heavy tasks.

## 125. Why Appendix E Is Crucial

[INTERPRETATION] A pretrained representation can optimize its pretraining objective very well while failing to improve a downstream task proportionately.

Analogously, better VQ reconstruction, code assignment accuracy, or MoE load balance does not guarantee better CSI300/S&P500 RankIC or portfolios.

## 126. Appendix F — Load Balance Pseudocode

[SOURCE; PDF p.32, Figure 14]

The pseudocode averages over tokens to obtain:

- hard token-dispatch fraction per expert;
- soft router-probability fraction per expert;

and multiplies their dot product by a factor equivalent to the number of experts, consistent with Eq. (4).

## 127. Appendix F — Router Pseudocode

[SOURCE; PDF p.33, Figure 15]

The router:

1. applies a learned projection to produce expert logits;
2. optionally introduces training noise;
3. converts router computations to float32;
4. softmaxes over all experts;
5. selects a Top-1 index **and its softmax gate probability**;
6. constructs a one-hot expert-assignment mask;
7. obtains the auxiliary balance loss;
8. assigns each token a within-expert position using cumulative sums;
9. discards assignments exceeding capacity;
10. creates dispatch/combine tensors;
11. casts dispatch/combine tensors back to bfloat16.

This sequence matters for exact reproduction.

## 128. Appendix F — Full Switch Layer Pseudocode

[SOURCE; PDF p.34, Figure 16]

The Switch FFN layer:

1. reshapes token data for cores;
2. invokes router to get dispatch/combine;
3. packs per-expert token inputs;
4. performs all-to-all communication to expert owners;
5. applies each expert's FFN;
6. performs reverse all-to-all;
7. combines expert outputs with retained gate weights;
8. reshapes to the original batch/time geometry.

## 129. Source Code Versus Mathematical Specification

[VERIFY]

The provided Mesh TensorFlow pseudocode is specific to its named tensor dimensions, compilation and accelerator layout. It is not a modern PyTorch drop-in implementation. An Agent should preserve the original tensor semantics, while reimplementing API details with an appropriate current framework.

## 130. Modern Minimal PyTorch-Style Specification

[DERIVATION; explanatory only]

```python
class ConceptualSwitchFFN(nn.Module):
    def __init__(self, d_model, experts):
        super().__init__()
        self.router = nn.Linear(d_model, len(experts), bias=False)
        self.experts = nn.ModuleList(experts)

    def forward(self, x):
        # x: [B, d_model], conceptual non-distributed version
        p = self.router(x.float()).softmax(-1)
        g, chosen = p.max(-1)
        out = torch.zeros_like(x)
        for expert_id, expert in enumerate(self.experts):
            mask = chosen == expert_id
            if mask.any():
                out[mask] = expert(x[mask]).to(out.dtype) * g[mask, None]
        return out
```

**WARNING:** This sketch omits capacity, overflow/residual bypass, efficient grouped dispatch, load loss, initialization, and optimized precision casts. It describes forward semantics only and is not a reproducibility-complete implementation.

## 131. Simple Balance-Loss Pseudocode

[DERIVATION]

```python
# router_probs: [B, E] after full softmax
selected = router_probs.argmax(dim=-1)  # [B]
f = torch.nn.functional.one_hot(selected, E).float().mean(0)
P = router_probs.mean(0)
loss_balance = alpha * E * (f.detach() * P).sum()
```

Be careful whether the implementation measures intended selections before capacity clipping versus actually processed selections: the original Eq. (5) counts selected Top-1 choices.

## 132. Capacity Pseudocode With Residual-Path Awareness

[DERIVATION]

```text
for each incoming token:
    compute router softmax
    choose top-1 expert
    assign token an arrival position within that expert's batch
    if position < expert_capacity:
        expert_result = gate * expert(token)
    else:
        expert_result = 0  # expert branch bypassed
    output = residual(token, expert_result)
```

Exact residual and normalization arrangement depends on the surrounding Transformer implementation.

## 133. Stochastic Routing and Evaluation Determinism

[SOURCE] The implementation description treats exploration perturbations as training-time behavior, and routes deterministically by highest probability at inference.

[EXTENSION] For stock ranking, explicitly test score variance under deterministic versus stochastic inference only if stochastic inference is a purposeful research design.

## 134. Numerical Stability of Softmax

[DERIVATION]

A stable softmax uses logits shifted by their maximum:

\[
p_i = \frac{\exp(h_i-m)}{\sum_j\exp(h_j-m)}, \quad m=\max_jh_j.
\]

This generic numerical formula is not a distinct novelty claim of the Switch paper; it explains the numerical risks motivating selective precision.

## 135. Capacity Factor Is Hardware-Dependent

[SOURCE/INTERPRETATION] TPU implementation in the paper uses fixed-shaped per-expert token buffers. Capacity factor balances padding and overflow. A dynamic-scheduling PyTorch implementation can implement dispatch differently, so the precise capacity policy is a reproduction choice that must be recorded.

## 136. Original Negative/Non-SOTA Findings Ledger

[SOURCE]

| Observation | Location | What the paper actually found |
|---|---|---|
| Top-1 not best in every table row | Table 1 | Top-2 slightly better at capacity factor 2.0 |
| Pure bfloat16 router diverged | Table 2 | Selective float32 fixes this tested instability |
| Uniformly increasing dropout can hurt | Table 4 | Expert-specific regularization helps selectively |
| Some ARC tasks lag dense T5 | Table 5 | Not all fine-tune benchmarks improve |
| Sparse teacher not fully distilled | Tables 6–8 | ~30% of *gain*, not total teacher quality |
| More total params can lose | Table 9 | Switch-C weaker upstream than Switch-XXL |
| Expert attention numerically unstable | Table 10 | bfloat16 variants diverge |
| Top-1 fallback rerouting not helpful | Appendix B | No-Token-Left-Behind gave no reported benefit |
| Unconstrained softmax sampling weaker | Table 11 | Small input jitter strongest among tested |
| Upstream gain not always downstream gain | Appendix E | Large dense models may be stronger on reasoning at fixed perplexity |

## 137. Validation Checklist for Tables and Figures

Use the PDF's rendered pages as authoritative whenever linearized text extraction scrambles columns:

- PDF p.4 Figure 2;
- p.5 Figure 3;
- p.7 Table 1;
- p.8 Table 2;
- p.9 Table 3;
- p.10 Table 4;
- pp.11–13 Figures 4–6;
- p.14 Table 5;
- pp.16–17 Tables 6–8;
- p.22 Table 9;
- p.27 Table 10;
- p.28 Figure 11;
- p.29 Table 11;
- p.30 Figure 12;
- p.31 Figure 13;
- pp.32–34 Figures 14–16.

## 138. Source Discrepancies / Verification Queue

[VERIFY]

**V1. Table 9 heading/value alignment.** Recheck `Num. Experts` and `Num Layers` labels against code; values appear swapped under headings in the printed table.

**V2. Table 11 polarity conflict.** The metric header says Neg. Log Perp. ↑, the caption says lower better; bolded -1.468 supports higher better.

**V3. Jitter placement.** Prose calls input perturbation multiplicative, illustrative router pseudocode shows noise added to logits. Resolve against actual implementation.

**V4. Training precision.** Check precisely where float32 is applied and which router tensors are cast back; do not force a simplistic all-float32 router if it changes dispatch numerics.

**V5. Capacity rounding.** Formula yields a noninteger in general; code imposes integer capacity and ordering of token positions.

**V6. Expert frequency.** Different model scales change how often a Switch layer replaces FFN.

**V7. Paper vs. reproduction.** Official Mesh TensorFlow code may have evolved after this JMLR publication; record commit hash if exact historical reproduction is attempted.

**V8. Distillation improvement definition.** The "30%" refers to quality improvement over dense baseline.

**V9. Scaling speedup definition.** Per-step and wall-clock factors must be kept separate.

**V10. Footnotes/figures.** Parsed text can flatten mathematical superscripts in training-batch sizes, e.g. \(2^{20}\) tokens rather than "220".

## 139. Full-Scope Reproduction: What Must Be Matched

[SOURCE/INTERPRETATION]

### Data

- T5-style C4 training corpus version and cleaning procedure;
- T5 masked-span pretraining targets;
- T5 vocabulary / tokenization;
- downstream task-specific validation splits;
- mC4 composition for multilingual experiments.

### Architecture

- exact T5 variant;
- number of experts per Switch layer;
- location/frequency of expert layers;
- top-1 gate with unrenormalized selected softmax multiplier;
- FFN hidden dimension and activation;
- residual/LN arrangement;
- whether attention expert variant is used (not default).

### Optimization

- router selective precision;
- initialization scale 0.1×;
- auxiliary balance coefficient;
- expert dropout for fine-tuning;
- stochastic router exploration method;
- optimizer/schedule/steps;
- precision and gradient update details.

### Systems

- capacity factor;
- overflow skip/bypass semantics;
- device placement;
- data/model/expert parallel topology;
- all-to-all implementation;
- fixed-shape buffers and padding;
- hardware and throughput measurement method.

### Reporting

- raw held-out negative log perplexity;
- training steps;
- hours to fixed-quality threshold;
- examples/sec;
- parameter count;
- FLOPs/seq or FLOPs/token;
- fine-tune task metric;
- seeds where specified.

## 140. Published Experiments Are Not Financial Backtests

[SOURCE]

- No CSI300 / S&P500 evaluation.
- No train/valid/test market split.
- No trading strategy, commission, Sharpe, IC, RankIC, or MDD.
- No financial factor labels or market regime definitions.

Do not hallucinate any such numerical results under this source.

---

# PART VII — THEORETICAL INTERPRETATION AND LIMITATIONS

## 141. Conditional Computation as Parameter Sharing Control

[INTERPRETATION]

A dense FFN shares its function across all tokens. A Switch FFN partitions the input space into routed regions, each with its own trainable transformation.

Formally, for region:

\[
\mathcal R_i=\{x: i=\arg\max_j p_j(x)\},
\]

Switch implements a piecewise mixture of expert functions:

\[
F(x)=p_{i}(x)E_i(x),\quad x\in\mathcal R_i.
\]

This is a capacity-allocation mechanism.

## 142. Hard Partition Boundaries and Local Smoothness

[DERIVATION/CRITIQUE]

Inside a region with fixed winner, output changes smoothly with neural network parameters (assuming smooth activations). At boundaries where the winner changes, the selected expert jumps from one function to another; the output need not be continuous.

No source proof claims global smoothness. This is a possible concern for market regime transitions.

## 143. Effective Training Samples Per Expert

[DERIVATION]

For B tokens and near-balanced Top-1 routing over N experts:

\[
\mathbb E[B_i]\approx B/N.
\]

With many experts and small batches, each expert receives few tokens even if the load is perfectly balanced. This is an important data-efficiency bottleneck in small financial datasets.

## 144. Why Bigger Total Capacity Helps in Large-Data Regimes

[INTERPRETATION]

A sufficiently large corpus supports learning specialized transformations in different input regions. With sparse routing, different experts can absorb different regularities without forcing all token types into one shared FFN.

But finite data must support expert specialization; otherwise, high capacity can simply increase variance.

## 145. Why Balance Is Not Equivalent to Specialization

[CRITIQUE]

A router can evenly distribute tokens at random, satisfying balance, while all experts learn redundant functions. Conversely, highly specialized experts on rare yet important regimes may legitimately have unequal use.

**Balancing is a utilization objective, not a semantic diversity guarantee.**

## 146. Why Uniform Utilization Might Be Wrong in Finance

[EXTENSION]

If crisis regimes occupy fewer trading days than ordinary markets, perfectly uniform routing across regime experts may artificially overemphasize rare states. More appropriate criteria may balance *data sufficiency* with *economic specialization*.

## 147. Strongest Internal Evidence for Efficiency

[SOURCE] Matched-compute and observed-time comparisons with dense T5 and Top-2 MoE in Table 1 and Figures 4–6 support the paper's core compute–quality message.

## 148. Weaker Evidence for Semantic Expert Understanding

[CRITIQUE]

The paper largely evaluates downstream scores, perplexity, and throughput. It does not present a comprehensive semantic analysis of expert roles or a controlled test that specialist experts map to human-interpretable skills.

Hence do not cite Switch as proof that expert #3 corresponds to a distinct latent market regime.

## 149. Limitations Explicitly Discussed

[SOURCE; §8, PDF p.25]

The authors' future directions include:

1. Stabilize largest models further, especially Switch-XXL.
2. Understand why better pre-training may fail to improve downstream reasoning.
3. Build principled joint scaling relationships across data/model/expert parallelism.
4. Explore **heterogeneous experts** with different compute requirements.
5. Apply Switch expertise outside Transformer FFN, including attention.
6. Explore non-language and multimodal domains.

## 150. Explicit Future Work: Stability

[SOURCE] The Switch-XXL instability remained unresolved even though smaller and differently designed large variants were stable in tested conditions. Authors mention exploring stronger regularization and adapted clipping.

## 151. Explicit Future Work: Compute-Conditioned Experts

[SOURCE] The paper's experts are architecturally homogeneous. The authors suggest heterogenous experts where more difficult examples can receive larger transformations.

[EXTENSION] For financial tasks, this suggests variable-compute routing for turbulent/uncertain periods—but such adaptations are not tested in the original source.

## 152. Explicit Future Work: Attention Experts

[SOURCE] Appendix A shows promising float32 quality for MoE in Q/K/V projections but significant bfloat16 instability. The paper leaves this as further research rather than a resolved default.

## 153. Explicit Future Work: Other Modalities

[SOURCE] The original paper directly invites applications outside language, but reports no finance, vision, or time-series performance tables in this source.

## 154. Additional Independent Critique: Parameter Efficiency

[CRITIQUE]

Total model parameters can be very high while each expert receives few updates. On small datasets this creates sample fragmentation; simply matching FLOPs can conceal differences in statistically effective capacity.

## 155. Additional Independent Critique: Expert Collapse

[CRITIQUE]

Auxiliary load balance helps prevent most tokens choosing a small subset of experts, but does not by itself ensure experts learn unique functions. It also may penalize legitimate imbalanced specialization.

## 156. Additional Independent Critique: Routing Noise

[CRITIQUE]

The paper's exploration experiment demonstrates that sampling a router distribution without control can hurt. In volatile finance, injecting routing noise may worsen ranking stability; any use of jitter should have a risk/benefit test.

## 157. Additional Independent Critique: Capacity Overflow Bias

[CRITIQUE]

Which tokens overflow can depend on batch ordering, cross-sectional stock count, or unstable gates. This can introduce an unintended selection bias in the training signal. The paper does not study this effect for variable-size financial cross-sections.

## 158. Additional Independent Critique: Computational Costs

[CRITIQUE]

The paper emphasizes accelerator-level scaling. A small local implementation can be slower than a dense head due to routing overhead and tiny expert batches. A finance study should report actual GPU time/VRAM rather than importing the source's TPU speedup claim.

## 159. Additional Independent Critique: Uncertainty

[CRITIQUE]

Top-1 routing chooses a single expert without explicitly propagating uncertainty over plausible expert identities. Near a routing boundary, prediction can be brittle; multiple/soft expert assignments might improve smoothness at extra compute cost.

## 160. Additional Independent Critique: Evaluation Objective

[CRITIQUE]

Switch evaluates token-prediction and NLP downstream quality. Using a MoE in stock ranking requires checking whether better regression loss maps to RankIC and risk-adjusted portfolio performance, rather than presuming this transfer.

---

# PART VIII — CROSS-PAPER MECHANISM LINEAGE

## 161. Switch vs. Jacobs et al. 1991

[INTERPRETATION]

| Dimension | Adaptive Mixtures of Local Experts (1991) | Switch Transformer (2022) |
|---|---|---|
| Central goal | Divide a supervised task into learnable local subproblems | Increase parameter capacity at limited active FLOPs |
| Model position | Usually overall mixture of local predictors | Layer embedded in deep Transformer |
| Router semantics | Distribution / mixture over expert predictions | Top-1 token-to-FFN decision |
| Competition | Probabilistic likelihood encourages expert responsibility | Hard dispatch with differentiable chosen softmax gate |
| Balance | Not the modern capacity-based balance objective | Auxiliary hard-fraction × soft-probability loss |
| Sparse compute | Not the focal contribution | Central objective |
| Expert overload | No fixed TPU expert-buffer mechanism | Explicit capacity factor and residual bypass |
| Demonstrated tasks | Vowel discrimination | Text pre-training, downstream NLP, multilingual scaling |

## 162. Switch vs. Shazeer et al. 2017

[INTERPRETATION]

| Dimension | Sparse MoE 2017 | Switch Transformer 2022 |
|---|---|---|
| Expert count | Potentially thousands+ | From 2 to 2048 in key Transformer examples |
| Gating | Noisy Top-k (often k=4) | Top-1 argmax of softmax |
| Sparse combination | Up to k weighted experts | One confidence-weighted expert |
| Noise | Learned Gaussian perturbation of expert logits | Training exploration with jitter; check code |
| Balancing | Importance CV² + Load CV² | \(\alpha N\sum_i f_iP_i\) |
| Baseline architecture | MoE between LSTM layers | FFN replacement in Transformer |
| Router numerical precision | Not central headline | Selective float32 routing |
| Per-expert capacity | Distributed batching/communication core issue | Explicit capacity-factor / overflow policy |
| Small downstream fine-tuning | Less central | Expert dropout and distillation |
| Main scaling narrative | Thousands of experts and large language models | Simpler routing, stability, sparse-vs-dense Pareto |

## 163. Do Not Attribute All MoE Ideas to Switch

[SOURCE/INTERPRETATION]

MoE routing, sparse conditional computation, and load balancing predate 2022. Switch's distinct innovations center on simplifying to Top-1 and developing the training/implementation/evaluation system around it.

## 164. Switch vs. PRISM-VQ (Research-Role Comparison)

[INTERPRETATION; PRISM-VQ descriptions are contextual baseline knowledge, not claims sourced to the Switch paper]

| Dimension | Switch Transformer | PRISM-VQ |
|---|---|---|
| Domain | NLP | Stock ranking |
| Routing unit | Token representation | Stock structural code |
| Router input | \(x\), current token hidden vector | Discrete VQ representation |
| Expert operator | Transformer FFN replacement | Factor-loading specialization |
| Typical experts | Many; Top-1 per token | Market-dependent; 2/Top-1 and 8/Top-4 |
| Load balancing | Hard selection fraction + soft probability | Similar scaled usage/probability form |
| Output | Transformed token state | Conditional factor-loading representation |
| Priors | None economic | 13 JKP-style financial priors |
| VQ | None | Learned codebook provides structural signal |
| Core goal | Scale parameter count / efficiency | Model heterogeneous financial structure |

## 165. Why PRISM-VQ's MoE Is Not Merely a Switch Layer

[INTERPRETATION] PRISM-VQ routes using a learned discrete stock structure rather than the raw temporal token itself, and experts participate in dynamic finance loadings rather than replacing generic FFNs. Top-k and balancing are reusable primitives; the *conditioning and output semantics* differ.

## 166. Shared and Routed Experts Are Not Provided by Switch's Standard Block

[INTERPRETATION] The paper's default Switch FFN chooses exactly one routed expert; there is no explicit always-on shared expert branch of the kind a modern shared-plus-specialized model may use.

If such a branch is introduced, it is a new architecture variant, not "the original Switch formulation."

## 167. How Switch Relates to MASTER / MATCC

[INTERPRETATION]

MASTER and MATCC are primarily about temporal and cross-stock representation interactions. Switch's mechanism addresses conditional computation and parameter specialization.

Potential integration points include:

- temporal FFNs;
- market-conditioned prediction heads;
- relation-specific projections;
- rank-scoring experts.

But arbitrary MoE integration can inflate complexity and impair interpretability; test specific failure modes.

## 168. How Switch Relates to FactorVAE / FactorVQVAE

[INTERPRETATION]

FactorVAE learns continuous latent factors; FactorVQVAE learns discrete latent factors. Switch does not learn a VQ codebook or posterior latent distribution.

A combined model may use latent factors or code indices as route-conditioning inputs, which is a later adaptation rather than an original result.

## 169. Most Transferable Primitive: Sparse Expert Selection

\[
\text{context}\rightarrow p(e\mid context)\rightarrow i^*\rightarrow E_{i^*}(x).
\]

The exact source of the `context` is an innovation decision:

- latent stock structure;
- market regime;
- temporal representation;
- macro/prior factors;
- forecast horizon.

## 170. Most Transferable Primitive: Load Balancing

\[
\alpha N\sum_i f_iP_i.
\]

This is a regularization starting point, not a complete guarantee of factor specialization or risk robustness.

## 171. Most Transferable Primitive: Routing Stability

A reproducible program should explicitly evaluate:

- selected gate probability;
- temperature / logit scale;
- routing boundary sensitivity;
- load loss;
- overflow;
- sensitivity to fp16/bfloat16/fp32;
- initialization and seed stability.

## 172. Most Transferable Primitive: Alternative Capacity–Compute Designs

Question to the Agent:

> Should additional parameters be used as more experts, wider experts, a shared expert, a stronger temporal encoder, or better input representation?

These choices must be compared under common compute and data conditions rather than assessed by total parameter count alone.

---

# PART IX — FINANCIAL TASK TRANSFER: RESEARCH AGENT'S EXPERIMENT CATALOG

> `[EXTENSION]` All experiments in this part are research proposals, not experiments reported by Fedus et al. (2022).
>
> Treat them as *hypotheses*, not as automatically novel scientific contributions. Confirm the state of the art before claiming novelty, and maintain a clean validation/test protocol.

## 173. Shared Evaluation Contract for All Finance Experiments

[EXTENSION]

Use a consistent experimental environment:

- Markets: CSI300 and S&P500, optionally CSI800 as expansion.
- Stock features: same Qlib Alpha158 pipeline as relevant baseline.
- No look-ahead: identical data/label alignment and constituent policy.
- Prediction horizon: match the baseline under comparison; do not compare 1-day versus 5-day IC/RankIC as if matched.
- Seeds: first seed-0 smoke test, then at least five official seeds for candidates worth advancing.
- Metrics: IC, ICIR, RankIC, RankICIR, and cost-aware portfolio AR/MDD/Sharpe/Calmar/Sortino/Omega where available.
- Diagnostics: annual, quarterly, monthly and market-regime performance; prediction/score decay; turnover and realized cost.
- Compute: model total/active parameters, wall-clock training, peak memory, route count distribution.
- Statistical claim: out-of-sample paired comparison where possible, with appropriate time dependence handling.
- Leakage audit: no use of test-period realized labels for routing, hyperparameter tuning, normalization, or codebook updates.

## 174. E01 — Dense Head vs. Top-1 Switch Head

[EXTENSION]

**Hypothesis:** A small conditional head learns distinct return relationships that one fixed dense head cannot represent well.

**Minimal test:** Hold upstream feature extractor fixed. Compare equal-compute dense FFN and Top-1 MoE FFN with 2, 4, 8 experts.

**Measure:** RankIC, RankICIR, five-seed variance, time/batch, active parameters.

**Failure criterion:** No stable OOS gain or worse quality-per-compute than dense head.

**Interpretability:** Compare expert predictions and errors conditional on market subperiods.

## 175. E02 — Top-1 vs. Top-2 vs. Top-4

[EXTENSION]

**Hypothesis:** Top-1 provides more stable specialization / lower complexity, whereas Top-2/4 may be better around regime boundaries.

**Ablations:** Keep total experts constant; vary k and report the resulting active FLOPs separately. Then introduce a compute-matched comparison by resizing expert networks.

**Measure:** RankIC, drop rate, routing entropy, runtime, near-boundary robustness.

**Failure:** Apparent gain disappears when active FLOPs are matched.

## 176. E03 — Balance Strength Sweep

[EXTENSION]

**Hypothesis:** Moderate regularization mitigates collapse; aggressive balancing prevents useful specialization.

**Conditions:** \(\alpha\in\{0,10^{-4},10^{-3},10^{-2},10^{-1}\}\), subject to observed loss scale.

**Measure:** Route CV, active experts, RankIC, regime-specific expert preference, seed variance.

**Failure:** Balance improves utilization without predictive or stability benefit; aggressive balance reduces performance.

## 177. E04 — Regime-Conditional Balance

[EXTENSION]

**Hypothesis:** Uniform global usage is not the right constraint if experts specialize by rare market regimes.

**Compare:** global batch balance, per-regime conditional balance, and no balance. Regime tags must be derived from historical information and defined without test labels.

**Metrics:** expert entropy within regime, return/IC stability, predictive gain on rare regimes.

**Failure:** Regime-aware balance merely injects noise or overfits rare regimes.

## 178. E05 — Capacity Factor and Overflow

[EXTENSION]

**Hypothesis:** Small stock universes and temporal changes in universe membership create harmful expert overflow if capacity is too tight.

**Conditions:** capacity factor 1.0, 1.25, 1.5, 2.0, and capacity-free local dispatch as a control.

**Measure:** pre/post routing counts, overflow fraction, loss, peak VRAM, time/date-batch.

**Failure:** Sophisticated capacity management has negligible effect under single-GPU dynamic dispatch.

## 179. E06 — No-Token-Left-Behind Revisited Carefully

[EXTENSION]

**Hypothesis:** A fallback expert can improve predictions when the primary expert overflows, but can also damage specialization—as original Appendix B warned.

**Compare:** residual bypass, next-best fallback, unlimited dynamic dispatch.

**Measure:** actual drop rate, accuracy for formerly overflowed samples, training cost.

**Failure:** Redispatch raises coverage but worsens model quality or compute-adjusted metrics.

## 180. E07 — Expert Dropout for Financial Overfitting

[EXTENSION]

**Hypothesis:** More aggressive dropout inside routed experts than inside shared trunk mitigates overfitting on limited daily data.

**Compare:** standard dropout everywhere, high expert-only dropout, and weight decay under matched parameter count.

**Measure:** train–valid gap, test RankIC, seed variance, regime robustness.

**Failure:** Expert dropout simply undertrains already data-starved experts.

## 181. E08 — Initialization Sensitivity

[EXTENSION]

**Hypothesis:** Router/expert initialization scale drives unstable seeds in sparse financial models.

**Compare:** full 1.0×, 0.1×, and hybrid router-only scaling under identical seeds and optimizer settings.

**Measure:** gradient norms, NaN incidence, learning curves, test metrics.

**Failure:** No measurable stability gain and degraded convergence after scale reduction.

## 182. E09 — Router Precision

[EXTENSION]

**Hypothesis:** Full-softmax routing and balance gradients benefit from float32 even when expert FFNs run with lower precision.

**Compare:** regular mixed precision vs explicitly float32 local router.

**Measure:** divergence rate, output stability, throughput, peak VRAM, OOS RankIC.

**Failure:** No instability exists and float32 routing adds overhead without practical benefits.

## 183. E10 — Temporal Features vs. Discrete Codes for Routing

[EXTENSION]

**Hypothesis:** Structural VQ codes provide a more stable routing key than noisy short-horizon temporal embeddings.

**Compare:** \(g(h_t)\), \(g(z_q)\), \(g([h_t,z_q])\), uniform random routing.

**Measure:** RankIC, expert activation persistence, seed variance, route switching rates.

**Failure:** Code-only routing is less adaptive and loses cross-market predictive power.

## 184. E11 — Code + Regime Residual Router

[EXTENSION]

**Hypothesis:** A stable code-based gate should be modulated by an independently learned, low-amplitude market-regime signal.

\[
\ell_i = g_{code}(z_{q,i}) + \lambda\,g_{regime}(m_t).
\]

**Ablations:** code only, regime only, residual combination, full concatenation gate.

**Failure:** The extra branch adds variance or duplicates code-derived information.

## 185. E12 — Shared + Routed Experts

[EXTENSION]

**Hypothesis:** An always-active shared component learns reusable patterns while routed specialists capture conditional effects; this reduces sample fragmentation.

\[
y=E_{shared}(x)+\sum_{i\in\mathcal T_k}g_i(x)E_i(x).
\]

**Compare:** dense only, routed only, shared only, shared+routed with parameter/FLOP controls.

**Measure:** RankIC, expert usage, seed stability, redundancy and regime performance.

**Failure:** Shared path dominates and routed experts contribute negligible incremental signal.

## 186. E13 — Frozen Gate vs. Learned Gate

[EXTENSION]

**Hypothesis:** Improvements may come from simply partitioning stock samples into groups, not from a sophisticated trainable router.

**Compare:** learned gate, random fixed gate, fixed size/industry grouping, fixed VQ code grouping.

**Measure:** incremental RankIC, compute, route stability.

**Failure:** Learned gate fails to exceed simple fixed partitions.

## 187. E14 — Conditional Expert Identity and Economic Meaning

[EXTENSION]

**Hypothesis:** Experts learn economically distinct response functions, not just arbitrary parameter partitions.

**Tests:** compare outputs across volatility buckets, size/value quintiles, industry, returns of external risk factors, and market drawdown periods.

**Controls:** random expert labeling, within-seed label permutations, route-preserving random experts.

**Failure:** Economic associations do not replicate across seeds or collapse after controlling for known factors.

## 188. E15 — Expert Function Redundancy

[EXTENSION]

**Hypothesis:** Standard balance loss permits unused *functional capacity* because experts converge to equivalent mappings.

**Measure:** output cosine similarity on common probes, centered kernel alignment, gradient similarity, disagreement.

**Test:** introduce diversity regularization only if redundancy is real.

**Failure:** Diversity rises but prediction worsens, indicating forced unnecessary differentiation.

## 189. E16 — Rare Market Regime Specialist

[EXTENSION]

**Hypothesis:** A minority of experts should specialize in crisis/low-liquidity states despite lower frequency.

**Compare:** globally uniform expert balance vs controlled unbalanced specialist allocation.

**Metrics:** tail-event RankIC, drawdown, performance under volatility spikes, route entropy.

**Failure:** Rare specialist has insufficient samples and degrades tail performance.

## 190. E17 — Market-Specific Expert Count

[EXTENSION]

**Hypothesis:** The optimal expert count depends on market signal strength and heterogeneity.

**Sweep:** \(N\in\{1,2,4,8,16\}\) for CSI300/S&P500 with identical validation budget.

**Measure:** mean/variance RankIC, model size, per-expert sample count, runtime.

**Failure:** A common small configuration is at least as good in all markets.

## 191. E18 — Heterogeneous Experts

[EXTENSION]

**Hypothesis:** Different regimes require different processing scales, as explicitly suggested by the Switch authors.

**Design:** experts with different widths, receptive fields, or time-series backbones, chosen under a constrained compute budget.

**Compare:** homogeneous matched-FLOP MoE, heterogenous MoE, dense multi-branch fusion.

**Failure:** Gains vanish once active FLOPs and tuning budget are controlled.

## 192. E19 — Horizon-Specialized Experts

[EXTENSION]

**Hypothesis:** One-day reversal, five-day ranking, and longer momentum require distinct temporal processing.

**Design:** horizon-conditioned gate or horizon-specific experts with a shared trunk.

**Protocol:** evaluate each horizon separately on correct labels, and test whether multi-horizon training helps the primary horizon.

**Failure:** No primary-horizon RankIC gain or harmful conflict between horizons.

## 193. E20 — Prior-Factor-Conditioned Routing

[EXTENSION]

**Hypothesis:** Market-wide JKP factor returns help decide which expert transformations are currently reliable.

\[
g_i=g(h_i,z_{q,i},f_{prior,t}).
\]

**Compare:** code only, prior only, code+prior, and code+prior with train-only priors.

**Failure:** Prior inputs add no independent information beyond VQ code and temporal state.

## 194. E21 — Stock/Time/Factor Routing Granularity

[EXTENSION]

**Hypothesis:** The appropriate routing unit in finance may be a stock, date, sector, latent factor, or temporal patch rather than a generic Transformer token.

**Compare:** per-stock gate, per-date market gate, per-factor gate, per-stock–time token gate.

**Measure:** stability, compute, performance and interpretability.

**Failure:** Fine-grained routing increases complexity without measurable predictive gain.

## 195. E22 — Ranking-Aligned Objective

[EXTENSION]

**Hypothesis:** The value of MoE specialization is better expressed by ranking than pointwise MSE.

**Compare:** MSE, pairwise ranking, hybrid MSE+ranking and a differentiable correlation surrogate.

**Measure:** out-of-sample IC/RankIC; long-only and long-short cost-aware portfolios; model variance.

**Failure:** Improved rank loss leads to no durable ranking or portfolio gain.

## 196. E23 — Routing Stability Around Market Transitions

[EXTENSION]

**Hypothesis:** Hard Top-1 routing causes abrupt score changes around regime boundaries.

**Compare:** Top-1, Top-2, temperature-calibrated gates, hysteresis or temporal gate smoothing.

**Measure:** per-stock route-flip rate, score variance, turnover, rank persistence, drawdown.

**Failure:** Smoother routing blurs useful regime shifts and worsens OOS signal.

## 197. E24 — Sparse Teacher, Dense Student

[EXTENSION]

**Hypothesis:** A larger financial MoE can serve as a research-time teacher and transfer a fraction of the predictive advantage to a small deployment model.

**Compare:** dense baseline, MoE teacher, supervised student from scratch, teacher-distilled student.

**Measure:** RankIC, drawdown, training and inference cost, retained gain as defined against dense baseline.

**Failure:** Teacher scores contain unstable noise and distillation does not transfer out of sample.

## 198. E25 — Expert Complementarity to Financial Priors

[EXTENSION]

**Hypothesis:** Routed experts capture predictive information orthogonal to value/size/momentum characteristics and prior factor exposures.

**Tests:** regress expert outputs and final scores on known factor proxies, evaluate residual RankIC and characteristic-neutral portfolios.

**Failure:** Apparent MoE benefit disappears after controlling known factor tilts.

## 199. E26 — Load Balance Versus Specialist Utility

[EXTENSION]

**Hypothesis:** Forcing equal expert usage is potentially at odds with unequal market-state prevalence and business importance.

**Compare:** uniform load balance, relaxed entropy target, prior-informed frequency target, no balance.

**Measure:** monthly/quarterly RankIC, tail risk, route distributions and utilization.

**Failure:** Nonuniform balancing yields unstable, undertrained experts or spurious test gains.

## 200. E27 — Portfolio Cost as an Additional Routing Objective

[EXTENSION]

**Hypothesis:** Conditional experts may improve ranking but increase turnover due to rapid assignment changes.

**Test:** route regularization penalizing unnecessary changes, with backtests using common realistic costs.

**Metrics:** pre-cost and post-cost AR, turnover, Sharpe, MDD, RankIC decay.

**Failure:** Reduced turnover comes at excessive alpha loss.

## 201. E28 — Conditional Relation Experts in MASTER/MATCC Family

[EXTENSION]

**Hypothesis:** Cross-stock relations differ by regime and may benefit from separate attention/aggregation experts.

**Compare:** one dense relation block, per-stock routed relation experts, market-gated relation experts, sparse interaction graph.

**Metrics:** two markets, equal active computation, IC/RankIC and runtime.

**Failure:** Relation specialists show no incremental gain or explode O(stock²) cost.

## 202. E29 — Optimizer and Training-Phase Specialization

[EXTENSION]

**Hypothesis:** Sparse experts require different regularization and learning schedules from shared trunk because they receive fewer updates.

**Compare:** common optimizer schedule, per-expert adaptive schedule, warm-start dense expert cloning, staged gate activation.

**Monitor:** gradient norms, effective update count, expert similarity, seed variance.

**Failure:** Apparent gains stem only from additional training budget.

## 203. E30 — Forward-Only Router Ablation

[EXTENSION]

**Hypothesis:** Retaining a non-unit selected softmax weight, as the original Switch paper does, matters for training the router and score calibration.

**Compare:** selected probability multiplier, hard gate = 1, straight-through surrogate, re-normalized selected gate.

**Measure:** router gradient norms, routing entropy, test RankIC, training stability.

**Failure:** Differences are negligible under an additional balance loss and stable training.

## 204. E31 — Cross-Market Transfer of Specialists

[EXTENSION]

**Hypothesis:** Some expert functions are reusable between Chinese and U.S. markets even if routing distributions differ.

**Design:** shared core experts plus market-specific router/adapters; strict non-overlapping chronological partitions.

**Controls:** independent-market models and shared trunk without MoE.

**Failure:** Transfer worsens both markets due to market-specific target distributions.

## 205. E32 — Architecture Choice Guided by Learning Curve

[EXTENSION]

**Hypothesis:** Additional sparse experts are useful only when the train-set size and regime diversity support them.

**Measure:** validation performance versus amount of training history, expert count, and active parameter budget.

**Failure:** larger MoE improves in-sample loss but worsens held-out RankIC at longer horizons.

## 206. E33 — Token Drop Distribution Bias

[EXTENSION]

**Hypothesis:** Expert overflow disproportionately affects certain firm-size, volatility, sector or regime subsets.

**Test:** log overflow flags per stock/date and test for association with protected experimental strata such as size or sector.

**Controls:** dynamic dispatch and larger capacity factors.

**Failure:** Observed pattern is random and overflow rates are negligible.

## 207. E34 — Factor Loading MoE Without Increasing Total Experts

[EXTENSION]

**Hypothesis:** Specialization gains may be driven more by *where* MoE is inserted than by how many experts exist.

**Compare:** MoE in temporal backbone, loading head, factor encoder, router-only modulation, multiple placements with equal active FLOPs.

**Failure:** The apparent architecture benefit disappears after compute matching.

## 208. E35 — Expert Utility by Market-Phase Interventions

[EXTENSION]

**Hypothesis:** Different routed experts have falsifiably different causal roles in the model.

**Intervention:** force routing away from one expert, substitute another expert under matched input, or zero individual expert contribution at inference.

**Measure:** market-state-specific performance drop, stock-class-specific effect, correlations with realized financial returns.

**Failure:** Swapping experts changes nothing except noise or broad scale.

---

# PART X — INNOVATION FILTER AND AGENT WORKFLOW

## 209. Rank Experiment Ideas by Scientific Value

[EXTENSION]

A practical order for a small research team:

**Tier A — Diagnostic before new architecture**

- E01: dense vs. Top-1;
- E02: Top-1/2/4;
- E03: balance coefficient;
- E10: code versus temporal routing;
- E12: shared+routed;
- E14/15: real expert specialization and redundancy.

**Tier B — Mechanism refinement**

- E11: code+regime router;
- E16: rare-regime specialist;
- E23: boundary stability;
- E26: balance vs specialist utility;
- E25: independent financial alpha beyond priors.

**Tier C — Higher-risk architecture**

- E18 heterogeneous experts;
- E19 horizon experts;
- E21 routing-unit alternatives;
- E28 relation experts;
- E31 cross-market transfer.

**Engineering controls**

- E05/E06 capacity handling;
- E07/E08 dropout and initialization;
- E09 router precision;
- E24 distillation;
- E33 overflow selection bias.

## 210. Three Different Kinds of "Improvement"

[INTERPRETATION]

1. **Algorithmic:** new routing or expert-function mechanism that generalizes better.
2. **Statistical:** better regularization, sample sharing, prior anchoring, or reduced variance.
3. **Systems:** lower runtime, communication, memory, and better hardware utilization.

Do not label a systems optimization as new representation learning without corresponding evidence.

## 211. What Would NOT Be Enough Novelty Alone

[CRITIQUE/EXTENSION]

- Increasing experts from 4 to 8.
- Setting k=1 because Switch already introduced it.
- Adding a plain load-balance loss copied from Eq. (4).
- Swapping standard ReLU expert FFN for an ordinary GELU FFN.
- Simply inserting an unmodified MoE into a financial baseline.
- Reporting only better IC under a different label horizon or market split.

A stronger contribution should alter the statistical role of specialists, the conditioning signal, the representation, or robust transfer to a new financial setting.

## 212. A More Convincing Novelty Claim

[EXTENSION]

Instead of saying:

> "We use MoE to improve stock forecasts."

Aim to demonstrate:

> "We identify a failure of stock-state-only routing under market transitions; a shared component preserves global regularities while a regime-modulated sparse route specializes factor exposures. Controlled ablations show improvement beyond a parameter-matched dense model, code-only routing, market-only routing and standard Switch routing, across both CSI300 and S&P500 with consistent out-of-sample gains and controlled turnover."

This is a proposed target claim. Do not use it until results actually support it.

## 213. Mechanism → Expected Evidence Mapping

| Proposed innovation | Evidence needed |
|---|---|
| Conditional specialization | Expert utility differs under controlled interventions |
| Improved routing stability | Lower route flip / lower seed variance without signal loss |
| Better market adaptation | Per-regime out-of-sample gain, not only global mean |
| Reduced expert collapse | More meaningful utilization **and** predictive gain |
| Financial-prior complementarity | Residual prediction beyond known characteristic tilts |
| Sparse compute efficiency | Active FLOPs, actual wall-clock and VRAM improvements |
| Ranking alignment | RankIC and portfolio evidence under same label |
| Trading robustness | Cost-aware portfolios, turnover, drawdown |

## 214. Smallest Useful Experimental Matrix

[EXTENSION]

| ID | Experts | Routing | Shared expert | Balance | Purpose |
|---|---:|---|---|---|---|
| A | 1 | None | No | No | Dense reference |
| B | 2 | Top-1 temporal | No | Yes | Minimal Switch |
| C | 4 | Top-1 temporal | No | Yes | Expert-count sensitivity |
| D | 4 | Top-2 temporal | No | Yes | Multi-expert ambiguity |
| E | 4 | Top-1 code | No | Yes | PRISM-style structural routing |
| F | 4 | Top-1 code | Yes | Yes | Shared+routed candidate |
| G | 4 | Top-1 code+regime | Yes | Yes | Regime adaptation |

For a fair comparison, introduce parameter/FLOP-matched controls when expert count, activated experts or shared branches change.

## 215. Staged Agent Loop

[EXTENSION]

```text
Paper knowledge:
  MoE 1991 → Sparse MoE 2017 → Switch 2022
                  +
  PRISM-VQ architecture + local Qlib results
                  ↓
Identify limitation supported by local diagnostics
                  ↓
Choose ONE mechanism hypothesis and ONE primary metric
                  ↓
Implement narrow ablation (seed 0)
                  ↓
Inspect loss/gradient/routing/overflow logs
                  ↓
If useful: five seeds × two markets
                  ↓
Add monthly, quarterly, yearly, regime and portfolio metrics
                  ↓
Check compute and turnover; test novelty against related work
                  ↓
Retain, revise or reject the hypothesis
```

## 216. Mandatory Pre-Experiment Diagnostics

[EXTENSION]

Before proposing a new router, the Agent should extract from local runs:

1. Actual model parameters and FLOPs.
2. Distribution of code assignments and expert assignments.
3. Entropy and load of each expert.
4. Market regime and time variation of each expert's use.
5. Sample count and update count per expert.
6. Correlations between expert predictions.
7. Per-regime and per-stock-decile ranking quality.
8. Score decay and holding-period dependence.
9. The exact label and price-timestamp convention.
10. TopK strategy, transaction costs and ensemble mode.

A new MoE idea without this profile is a speculative candidate, not a diagnosed improvement.

## 217. Separate Original Claims from Our Transfer Proposals

[SOURCE/EXTENSION]

- Original: Top-1 Switch can be effective on NLP/T5 benchmarks.
- Transfer hypothesis: Top-1 might help financial ranking.
- Original: load balancing improves expert utilization.
- Transfer hypothesis: regime-balanced routing might outperform global balance.
- Original: expert dropout can improve downstream task fine-tuning.
- Transfer hypothesis: expert-specific dropout might improve small financial dataset generalization.
- Original: huge sparse models can improve throughput-adjusted language learning.
- Transfer hypothesis: sparse conditional computation might be useful in a parameter-limited stock model.

Do not turn the right-column hypotheses into retrospective descriptions of what the paper did.

---

# PART XI — SCIENTIFIC WRITING AND PRESENTATION LESSONS

## 218. Strong Problem Framing

[SOURCE/INTERPRETATION]

The paper does not announce "we add an MoE" without a reason. It introduces a **measurable objective**:

\[
\text{increase parameters while holding per-token FLOPs fixed}.
\]

This enables meaningful quality–cost comparisons.

## 219. One Simple Main Change, Many Supporting Engineering Details

[INTERPRETATION]

The architectural idea is easy to articulate: **Top-1 routing**. The rest of the work solves the blockers to making that one simplification useful at scale—capacity, balance, precision, initialization, dropout and sharding.

This is a template for explaining a research method coherently rather than listing disconnected innovations.

## 220. Compare at Multiple Operating Points

[SOURCE/INTERPRETATION]

The paper reports results against dense and Top-2 MoE for capacity factor 1.0, 1.25, 2.0, and widened Switch-Base+, making the efficiency story more credible than one arbitrary configuration.

## 221. Negative Results Strengthen the Story

[SOURCE]

Examples:

- softmax sampling not preferred over jitter;
- No-Token-Left-Behind not beneficial;
- attention experts unstable in lower precision;
- Switch-C inferior on some metrics despite more parameters;
- ARC tasks with no gain;
- huge upstream performance not fully transferring to reasoning.

A scientific paper should report such exceptions rather than suppress them.

## 222. Original Figure/Table Role Map

| Display | Role in argument |
|---|---|
| Figure 1 | capacity and sample-efficiency headline |
| Figure 2 | Switch FFN architecture and individual-token routing |
| Figure 3 | capacity factor and token overflow mechanism |
| Table 1 | Top-1 vs Top-2 vs dense at multiple operating points |
| Tables 2–3 | selective precision and initialization stability |
| Table 4 | expert-specific fine-tuning regularization |
| Figures 4–6 | sample, step and wall-clock scaling |
| Table 5 | downstream task transfer and exceptions |
| Tables 6–8 | sparse-to-dense distillation trade-off |
| Figures 7–8 | multilingual transfer and speedup distribution |
| Figure 9 | data/model/expert parallelism |
| Table 9 | T5, Switch-XXL, Switch-C architecture and scaling |
| Figure 10/Table 10 | attempt to add conditional compute to attention |
| Figure 11 | attempted overflow rerouting |
| Table 11 | router exploration variants |
| Figure 12 | few-expert regime |
| Figure 13 | upstream/downstream quality relationship |
| Figures 14–16 | code-level dispatcher, balancing and experts |

## 223. Writing Formula: Motivation → Bottleneck → Design → Evidence

[INTERPRETATION]

```text
Large dense models: better quality but growing compute
        ↓
Conditional computation: high capacity at sparse activation
        ↓
Historical obstacle: dispatch + balancing + numerical instability
        ↓
Top-1 Switch + capacity/balance + selective precision
        ↓
Compute-matched quality, real-speed results, downstream tests
        ↓
Failures + limitations + explicit research agenda
```

## 224. How This Applies to a New Stock-Prediction Paper

[EXTENSION]

A credible finance paper using expert routing should explicitly state:

- Which failure of current cross-sectional prediction is observed?
- Why would conditional computation help that failure?
- Why use a particular context for routing?
- What controls remove parameter/compute confounding?
- What nontrivial ablation isolates the router mechanism?
- Is the improvement robust through time and across markets?
- Does it improve a realistically executable portfolio?

---

# PART XII — REPRODUCTION ARTIFACTS AND SHORT AGENT RETRIEVAL

## 225. Mechanism Inventory

| Primitive | Source | Inputs | Outputs | Main failure risk |
|---|---|---|---|---|
| Softmax router | Eq. 1 | hidden vector | N probabilities | overconfident logits |
| Top-1 dispatch | §2.1 | N probabilities | expert index | hard boundaries |
| Gate scaling | Eq. 2 | probability + FFN result | conditional state | lost gradients if renormalized |
| Capacity factor | Eq. 3 | batch, N, factor | per-expert buffer | overflow or padding waste |
| Load balancing | Eqs. 4–6 | hard frequency + soft probs | aux loss | forced uniform redundancy |
| Router selective fp32 | §2.4 | low-precision input | stable routing | extra compute if overapplied |
| Reduced init | §2.4 | parameter setup | better optimization | underpowered gradients |
| Expert dropout | §2.4 | expert intermediate activations | regularization | expert undertraining |
| Expert parallelism | §5 | routed tokens, expert weights | scalable compute | communication overhead |
| Distillation | §4.2 | sparse teacher | dense student | partial benefit retention |

## 226. Primary Equations in One Place

[SOURCE, unless noted]

**Router**

\[
p_i(x)=\mathrm{softmax}(W_rx)_i.
\]

**Switch expert output**

\[
y=p_{\arg\max_i p_i(x)}(x)E_{\arg\max_i p_i(x)}(x).
\]

**Capacity**

\[
C=(B/N)\cdot \mathrm{capacity\ factor}.
\]

**Load balance**

\[
\mathcal L_{balance}=\alpha N\sum_i f_iP_i.
\]

\[
f_i=\frac{1}{T}\sum_t \mathbf1[\arg\max_jp_j(x_t)=i].
\]

\[
P_i=\frac{1}{T}\sum_t p_i(x_t).
\]

## 227. Agent Input–Output Contracts

[DERIVATION]

```text
Router:
  in:  x [T, d_model]
  out: probabilities [T, N], selected_ids [T], selected_probs [T]

Dispatch:
  in:  x, selected_ids, expert_capacity
  out: per-expert token lists; overflow flags

Experts:
  in:  token sets grouped by expert
  out: expert results for routed tokens

Combine:
  in:  expert results, selected_probs, positions
  out: branch_output [T, d_model]

Residual transformer block:
  in:  x, branch_output
  out: next-layer token states [T, d_model]

Auxiliary loss:
  in:  probabilities, selected_ids
  out: scalar balance loss
```

## 228. Literal vs. Derived Complexity Claims

[SOURCE] Sparse expert activation increases total parameters without activating all expert weights for each token.

[DERIVATION] At fixed expert geometry, active expert FFN arithmetic is roughly independent of the number of experts under Top-1, but router cost grows with N and actual dispatch overhead remains implementation-dependent.

**Avoid saying:** "compute is exactly constant" under any increase in experts.

## 229. Original Results: Five High-Value Anchors

[SOURCE]

1. Table 1: Switch Top-1 beats Top-2 at the favorable low-capacity operating point but not in every row.
2. Table 2: selective fp32 routing avoids divergent bfloat16 training without observed speed loss.
3. Figure 5: ~7× wall-clock speed-to-quality vs T5-Base in the reported 32 TPUv3-core setup.
4. Table 7: compressed dense students preserve only ~27–37% of sparse-teacher quality gains.
5. Table 9: 395B Switch-XXL beats 1571B Switch-C on some upstream measurements, demonstrating parameter count is not the whole story.

## 230. Key Source Boundaries

[SOURCE] The paper does not demonstrate financial alpha, new-factor discovery, market-regime routing, codebook-based routing, or actual trading advantages. Those topics should be cited to other papers or reported as future research questions.

## 231. Search Keywords for an Agent

`mixture of experts`, `Switch Transformer`, `top-1 routing`, `chosen gate probability`, `softmax`, `router differentiability`, `hard argmax`, `load balance`, `f_i P_i`, `expert capacity`, `capacity factor`, `token dropping`, `overflow`, `selective precision`, `bfloat16`, `float32 router`, `expert dropout`, `parameter scale`, `model parallel`, `expert parallel`, `all-to-all`, `distillation`, `heterogeneous experts`, `financial regime routing`, `shared expert`, `PRISM-VQ`.

## 232. Source Page Map

| PDF pages | Content |
|---|---|
| 1–3 | abstract, research framing, scaling philosophy, main contributions |
| 3–5 | Top-1 routing, equations, Figure 2 and capacity Figure 3 |
| 6 | auxiliary balance loss Eqs. 4–6 |
| 7–8 | Switch/Top-2 comparison Table 1; selective precision Table 2 |
| 9–10 | reduced initialization and expert dropout Tables 3–4 |
| 11–13 | scaling and speed Figures 4–6 |
| 14 | downstream fine-tuning Table 5 |
| 15–17 | sparse-to-dense distillation Tables 6–8, multilingual study starts |
| 17–18 | 101-language results Figures 7–8 |
| 18–21 | data/model/expert parallelism, Figure 9 |
| 21–23 | trillion-scale architecture and results, Table 9 |
| 24 | discussion of the scope and practical implications |
| 25 | six explicitly stated future directions |
| 26–27 | Appendix A, expert self-attention, Table 10 |
| 28 | Appendix B, No-Token-Left-Behind negative result |
| 29 | Appendix C, router jitter and Table 11 |
| 30 | Appendix D, few-expert regime Figure 12 |
| 31 | Appendix E, upstream/downstream relation Figure 13 |
| 32–34 | Appendix F, MoE balance/router/layer pseudocode, Figures 14–16 |
| 35–39 | references |

## 233. Reference Snippets an Agent Should Quote Carefully

The following are **paraphrase-ready ideas**, not verbatim author quotes:

- The authors use parameter capacity as a scaling dimension separate from active FLOPs.
- A Switch layer chooses one expert per token while retaining a differentiable selected probability multiplier.
- The auxiliary balance objective mixes hard assignment fractions and differentiable soft probabilities.
- Overflowed tokens bypass expert computation through residual paths rather than being removed from the dataset.
- Stability benefits from precise router operations, smaller initialization, and targeted expert dropout.
- Increased sparse capacity is useful on sufficiently large data but does not guarantee monotonic downstream gains.
- Larger sparse pretraining teachers transfer only part of their advantage when distilled into small dense models.

## 234. Canonical One-Paragraph Summary

[SOURCE] Fedus, Zoph, and Shazeer (JMLR 2022) introduce Switch Transformers, replacing selected Transformer feed-forward layers with sparsely activated expert layers that dispatch each token to exactly one learned FFN. A softmax router selects the highest-probability expert, and the chosen full-softmax gate value scales its output so the router can receive gradients. They address expert imbalance using an auxiliary loss \(\alpha N\sum_i f_iP_i\) based on hard assignment fractions and mean router probabilities, and control overflow using a capacity factor and residual bypass of over-capacity token assignments. Selective float32 routing, a tenfold reduction in weight initialization scale, and expert-specific dropout improve training stability and downstream regularization. On compute-matched T5 language models, sparse expert capacity improves sample efficiency and quality per wall-clock time, with a 64-expert Switch-Base reaching a selected dense baseline quality in roughly one-seventh the time on 32 TPUv3 cores. The paper reports 101-language pre-training improvements, sparse-to-dense distillation retaining only about 30% of a teacher's quality gain, and architectures as large as approximately 1.6 trillion parameters. Its appendices also document meaningful failures: softmax sampling underperforms jitter, re-routing overflowed tokens has no demonstrated benefit, and attention experts become unstable with low precision. The findings concern large-scale language models; finance-specific routing and portfolio benefits require separate validation.

## 235. Research-Agent Decision Rules

1. **Read the selected gate definition before altering Top-1 routing.**
2. **Use both total and active parameters in any size comparison.**
3. **Never claim fixed FLOPs implies fixed wall-clock time.**
4. **Track overflow and whether it depends on stock/date/regime.**
5. **Balance utilization but verify functional expert diversity separately.**
6. **Treat load-balance \(\alpha=10^{-2}\) as a paper hyperparameter, not a universal optimum.**
7. **Benchmark Top-1 against Top-2 under a clearly stated compute budget.**
8. **Differentiate Top-1 token routing from code-gated multi-expert loading generation.**
9. **Do not force deterministic Top-1 routing to mean no router gradient.**
10. **Do not accidentally normalize one surviving gate to 1.**
11. **Preserve the numerical-stability strategy when using mixed precision.**
12. **Acknowledge the Appendix B and Appendix A negative results when proposing rerouting or expert attention.**
13. **Record original paper errors and ambiguities as VERIFY, not invented corrections.**
14. **Avoid copying giant-scale TPU speed numbers into GPU stock forecasts.**
15. **For financial extensions, compare at the same market/label horizon/split.**
16. **Require multi-seed evidence and cost-aware portfolio corroboration.**
17. **Do not call balanced experts interpretable without interventions.**
18. **Reject improvements that depend only on added tuning or active FLOPs.**
19. **Use the full paper's negative findings to prune speculative innovation proposals.**
20. **Make a falsifiable statement for every claimed MoE advantage.**

## 236. Final Takeaways

**Most important historical advance:** a simpler Top-1 router that enables a practical large sparse Transformer FFN.

**Most important equation:**

\[
\boxed{y=p_{i^*}(x)E_{i^*}(x),\ i^*=\arg\max_i p_i(x)}.
\]

**Most important stabilization mechanism:** the combination of balancing, capacity handling, selective router float32 and reduced initialization.

**Most important engineering distinction:** total parameters can scale without proportionate active FLOPs, but communication, memory and data distribution remain limiting.

**Most useful finance research warning:** balanced Top-1 expert use does not prove economically meaningful specialization.

**Best application to PRISM-VQ-style research:** controlled comparisons of code-based vs. temporal/regime routing, with shared/routed experts and explicit diagnostics, under identical label, multi-seed and cost-aware portfolio protocols.

---

# END OF AGENT KNOWLEDGE FILE
