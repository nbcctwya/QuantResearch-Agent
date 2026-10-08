---
paper_id: Shazeer_2017_Sparsely_Gated_MoE
title: "Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer"
short_name: Sparse MoE
authors:
  - Noam Shazeer
  - Azalia Mirhoseini
  - Krzysztof Maziarz
  - Andy Davis
  - Quoc Le
  - Geoffrey Hinton
  - Jeff Dean
year: 2017
venue: ICLR 2017
source_version: "arXiv:1701.06538v1, 23 January 2017; supplied PDF header says Under review as a conference paper"
arxiv: "1701.06538"
paper_type: foundation
topics:
  - mixture_of_experts
  - conditional_computation
  - noisy_top_k_gating
  - expert_routing
  - expert_load_balancing
  - distributed_model_parallelism
  - hierarchical_moe
  - language_modeling
  - machine_translation
importance_for_agent: very_high
related_local_papers:
  - "foundations/MoE_1991.md"
  - "baseline/PRISM-VQ_2026.md"
  - "baseline/FactorVQVAE_2025.md"
source_filename: "OUTRAGEOUSLY LARGE NEURAL NETWORKS.pdf"
source_pages: 19
formal_equations: "1-22 (including appendices)"
primary_contribution: "Practical sparse conditional computation with Noisy Top-k gating and load-balanced distributed MoE layers"
reading_mode: "Mechanism-first; preserve historical implementation and isolate modern financial extensions"
evidence_tags:
  - SOURCE
  - DERIVATION
  - INTERPRETATION
  - CRITIQUE
  - EXTENSION
  - VERIFY
---

# Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer

> **Research-Agent edition | Foundations | Shazeer et al. (2017)**  
> File destination: `papers/foundations/Sparse_MoE_2017.md`  
> Primary source: uploaded 19-page arXiv v1 PDF, including Appendices A–G.  
> Citation convention: `[SOURCE, PDF p.N]` for the source text or a figure/table on PDF page N. Other tags distinguish algebraic derivations, inferences, critiques, new research proposals, and details requiring verification.

---

# Part I. Research Identity and the Original Problem

## 001. Document objective

[SOURCE] This is the original modern **Sparsely-Gated Mixture-of-Experts layer** paper, not the first paper to introduce mixtures of local experts. Jacobs et al. (1991) proposed early adaptive mixtures; this 2017 work makes *sparse conditional computation* useful in a very large deep network.

[INTERPRETATION] For a research Agent, its highest-value contributions are **conditional activation**, **routing under gradient training**, **expert load balancing**, and **capacity-versus-computation analysis**. These are not synonymous with economic expert specialization.

## 002. How this document should be used

- Understand the exact routing equations before reimplementing a financial MoE.
- Reconstruct both balancing losses and why they are distinct.
- Learn which scaling claims are backed by the original experiments, and which depend on huge data and distributed GPUs.
- Separate the original mechanisms from later variants (top-1, shared experts, code-conditioned routing, capacity factors, dropless routing).
- Derive testable modifications for stock ranking on CSI300 and S&P500 rather than copying the model's language modeling architecture.
- Check the true original PDF for equations, numbers, and implementation omissions before treating this Markdown as definitive.

## 003. Bibliographic identity

- Title: *Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer*.
- Authors: Noam Shazeer, Azalia Mirhoseini, Krzysztof Maziarz, Andy Davis, Quoc Le, Geoffrey Hinton, Jeff Dean.
- Source supplied: arXiv:1701.06538v1, dated 23 January 2017.
- Source header: “Under review as a conference paper at ICLR 2017.”
- Historical venue: ICLR 2017; this PDF itself is the preprint/review version, **not** a later camera-ready reconstruction.
- Institution: Google Brain, with a Jagiellonian University affiliation for Maziarz.
- PDF extent: 19 pages, 9 pages of main paper and 10 pages of references/appendices.
- Original tasks: language modeling and machine translation; **not** financial prediction.

[SOURCE, PDF pp.1–3]

## 004. One-sentence central insight

> Increase the number of *available* expert parameters dramatically, but activate only a tiny chosen subset of those experts for each input, while keeping the routing/trainability and distributed implementation efficient.

[SOURCE, PDF pp.1–2]

## 005. Capacity versus computation

[SOURCE] The introduction explicitly distinguishes **parameter capacity** from **per-example computation**. Standard dense networks activate essentially all layers/parameters for each example. Sparse conditional networks need not do so.

[INTERPRETATION] Increasing the number of available experts $n$ with selected experts $k$ held constant may increase total parameters approximately linearly in $n$ while the expert-side active FLOPs remain approximately proportional to $k$.

[CRITIQUE] This does **not** make memory, communication, optimization or training data requirements independent of $n$. The paper devotes substantial attention to those costs.

## 006. Why “just use more experts” had previously failed

The authors identify five obstacles:

1. Branching on GPUs is often inefficient even when nominal FLOPs are small.
2. Routing splits large input batches into very small per-expert batches.
3. Cross-device movement of expert inputs/outputs can dominate arithmetic.
4. Some gates collapse onto a few experts without explicit balancing constraints.
5. Small datasets may not contain enough supervisory signal to train enormous networks.

[SOURCE, PDF p.2]

## 007. Original solution in four components

1. **Sparse MoE layer:** a collection of independent feed-forward experts.
2. **Noisy Top-k gate:** sparse and trainable input-dependent expert selection.
3. **Importance and Load penalties:** keep experts from becoming unusably imbalanced.
4. **Distributed batching:** aggregate examples for each expert across model/data-parallel devices and across unrolled time positions.

[INTERPRETATION] The empirical result is a *systems-and-learning co-design* result, not merely a new neural block.

## 008. Relation to Jacobs et al. (1991)

[SOURCE] The related-work discussion explicitly credits Jacobs et al. (1991) and Jordan & Jacobs (1994) for earlier MoE research.

Key contrast:

| Dimension | Adaptive Mixtures of Local Experts (1991) | Sparse MoE (2017) |
|---|---|---|
| Original purpose | Decompose tasks into local experts | Scale deep-model capacity with conditional computation |
| Expert role | Complete local predictors in a mixture | Interchangeable sub-networks inserted as a layer |
| Gate | Mixture probabilities | Sparse selection plus gate weights |
| Training emphasis | Competitive learning, local responsibility | End-to-end backprop, sparse routing, balancing |
| Sparsity implementation | Stochastic expert-choice interpretation, not modern compute-sparse Top-k | Explicit Top-k zero gates enabling skipped expert compute |
| Evidence | Vowel discrimination and convergence | Language modeling, translation, scale, hardware efficiency |

## 009. Relation to Eigen et al. (2013)

[SOURCE] The authors credit Eigen et al. for inserting MoE components inside deep models rather than using the whole model as one mixture.

Their distinctive extension: MoE is applied **at every text position**, allowing different expert selections at different positions and thus potentially fine-grained specialization.

[SOURCE, PDF p.3]

## 010. How not to misunderstand its novelty

- Not “MoE was invented in 2017.”
- Not “Top-k was first ever used to choose a model.”
- Not “massive parameter count means all parameters are active.”
- Not “load balancing creates economically meaningful experts.”
- Not “all 137 billion parameters improve prediction monotonically.”
- Not “more experts are always better for small, noisy financial datasets.”

---

# Part II. Core Sparse Mixture Mathematics

## 011. Basic notation

| Symbol | Meaning | Typical shape |
|---|---|---|
| $x$ | Input representation of one token/example | $\mathbb R^d$ |
| $n$ | Total number of experts | integer |
| $k$ | Activated experts per input | $1\le k\le n$ |
| $E_i$ | Expert network $i$ | $\mathbb R^d\to\mathbb R^{d'}$ |
| $G(x)$ | Nonnegative sparse gate vector | $\mathbb R^n$ |
| $W_g$ | Gate linear weights | $\mathbb R^{d\times n}$ |
| $W_{\mathrm{noise}}$ | Trainable noise-scale weights | $\mathbb R^{d\times n}$ |
| $X$ | Input batch for balancing terms | $B\times d$ |
| $\Phi$ | Standard normal CDF in the load estimator | scalar function |

[DERIVATION] Shape annotations are modern notation consistent with Eq. (1)–(11); source equations are formulated without an explicit vectorized tensor API.

## 012. Mixture output: Equation (1)

\[
y=\sum_{i=1}^{n}G(x)_i E_i(x).
\tag{1}
\]

The output dimension of every expert is identical, so their outputs can be added.

[SOURCE, PDF p.3]

## 013. Why sparse $G(x)$ saves expert compute

If $G(x)_i=0$, expert $i$ has no contribution and does **not** need to execute.

\[
\mathcal A(x)=\{i:G(x)_i\ne 0\},\qquad
\lvert\mathcal A(x)\rvert\le k.
\]

Compute:

\[
y=\sum_{i\in\mathcal A(x)}G(x)_iE_i(x).
\]

[DERIVATION] This rewritten form is algebraically equivalent to Eq. (1) when the gate is sparse.

## 014. Gate nonnegativity and normalization

[SOURCE] The proposed gate is Softmax after masking, so:

\[
G(x)_i\ge 0,\quad \sum_iG(x)_i=1.
\]

[INTERPRETATION] The layer output is a weighted combination of active expert outputs. This is not automatically a probabilistic posterior over latent economic regimes.

## 015. Dense softmax baseline: Equation (2)

\[
G_{\sigma}(x)=\operatorname{Softmax}(xW_g).
\tag{2}
\]

All $n$ entries are generally nonzero, so all experts would normally run.

[SOURCE, PDF p.4]

## 016. Why softmax alone is inadequate for computation sparsity

Dense probabilities can be very small but nonzero. Unless they are explicitly masked, one still needs expert results to form the exact weighted sum.

[DERIVATION] The computational benefit comes from **actual zeros**, not just a low-entropy dense distribution.

## 017. Noisy Top-k gate: Equation (3)

\[
G(x)=\operatorname{Softmax}\big(\operatorname{KeepTopK}(H(x),k)\big).
\tag{3}
\]

The masking operation sets all logits except the selected $k$ to $-\infty$.

[SOURCE, PDF p.4]

## 018. Noisy router logits: Equation (4)

\[
H(x)_i=(xW_g)_i+\epsilon_i\operatorname{Softplus}\big((xW_{\mathrm{noise}})_i\big),
\qquad \epsilon_i\sim\mathcal N(0,1).
\tag{4}
\]

The first part is a learned score; the second is learned-scale Gaussian perturbation.

[SOURCE, PDF p.4]

## 019. Top-k mask: Equation (5)

\[
\operatorname{KeepTopK}(v,k)_i=
\begin{cases}
v_i,& v_i\text{ lies among the top }k\text{ elements},\\
-\infty,&\text{otherwise}.
\end{cases}
\tag{5}
\]

Then Softmax makes all masked entries zero.

[SOURCE, PDF p.4]

## 020. Step-by-step gate walkthrough

For an input $x$:

1. Compute raw scores $s=xW_g$.
2. Compute per-expert noise scales $a=\operatorname{Softplus}(xW_{\mathrm{noise}})$.
3. Draw independent standard Gaussian perturbations $\epsilon$.
4. Produce $h=s+\epsilon\odot a$.
5. Keep the $k$ largest entries of $h$.
6. Set the rest to $-\infty$.
7. Apply Softmax to the selected logits.
8. Dispatch the input to selected experts only.
9. Weight and aggregate their outputs.

[DERIVATION] This is implementation-form pseudocode reconstructed from Eqs. (3)–(5).

## 021. Concrete numerical routing example

[INTERPRETATION / illustrative, not paper data]

Take:

\[
H(x)=[1.2,0.3,2.1,-0.7],\quad k=2.
\]

Keep indices 3 and 1 (1-based indexing):

\[
\widetilde H=[1.2,-\infty,2.1,-\infty].
\]

The active weights are proportional to $e^{1.2}$ and $e^{2.1}$, respectively. Inactive experts contribute exactly zero and need not be evaluated.

## 022. Why Gaussian noise is introduced

[SOURCE] The authors explicitly motivate noise as a way to help **load balancing**, especially for the smooth load estimator described in Appendix A.

[INTERPRETATION] Noise makes borderline experts have nonzero selection probability and gives a differentiable proxy for discrete assignment counts.

[CRITIQUE] Noise is *not* automatically a calibrated uncertainty estimate, market-regime distribution, or diversity guarantee.

## 023. Softplus on the noise scale

\[
\sigma_i(x)=\operatorname{Softplus}((xW_{\mathrm{noise}})_i)>0.
\]

[DERIVATION] The positivity helps interpret $\sigma_i$ as a scale and permits the Gaussian CDF formula in the appendix. A naive raw unrestricted scale would not have this property.

## 024. Differentiability with a Top-k mask

[SOURCE] The paper acknowledges hard Top-k discontinuities, but reports they were not problematic in experiments.

Within a fixed selected set and with $k>1$, the Softmax outputs of active experts have derivatives with respect to their scores. Thus end-to-end backprop trains the active gate and experts without a REINFORCE-style binary estimator.

[SOURCE, PDF p.4]

## 025. Why $k=1$ deserves caution

[DERIVATION] With one surviving logit, its Softmax probability is identically one. Consequently, **the main prediction path alone** provides no useful Softmax-weight gradient to the selected logit's score inside a locally unchanged selected set.

[CRITIQUE] This is a mathematical implication of Eqs. (3)–(5), **not** an experiment performed or analyzed for top-1 in this paper. Modern top-1 MoE variants require their own routing/training treatment.

## 026. What Noisy Top-k does not guarantee

- It does not guarantee balanced expert occupancy without regularization.
- It does not force experts to be semantically distinct.
- It does not guarantee stable experts across seeds or time.
- It does not guarantee optimal selection under a cost/latency constraint.
- It does not prevent one expert from being chosen across most inputs early in training.
- It does not by itself remove all distributed dispatch overhead.

## 027. Sparse router versus sparse expert parameters

[INTERPRETATION] Sparse *activation* here does not mean each expert is a pruned/sparse matrix. Selected experts are ordinary dense feed-forward networks. The sparsity applies to **which expert networks execute**.

## 028. Difference from hard winner-take-all competition

Jacobs (1991) uses stochastic selection to motivate objective decomposition, whereas the 2017 implementation computes a **weighted sum of the selected $k$ experts**. For $k=4$ in its major experiments, multiple specialists contribute to one token.

## 029. Gate information flow

```text
input x
   │
   ├──────────► W_g ─────────► learned scores
   │
   └──────────► W_noise ─► Softplus ─► Gaussian perturbation
                                        │
                      add noisy logits ◄─┘
                              │
                          KeepTopK
                              │
                        sparse Softmax
                              │
                          select k
                              │
                dispatch to active experts
                              │
                      weighted sum
                              │
                        MoE output
```

## 030. Tensor contracts for a modern implementation

```text
Input:           x                 [B, D]
Gate logits:     scores             [B, n]
Noise scales:    scales             [B, n]
Noisy scores:    H                  [B, n]
Selected IDs:    topk_idx           [B, k]
Sparse gate:     G                  [B, n] (or sparse representation)
Expert e input:  dispatched_x[e]    [B_e, D]
Expert e output: expert_out[e]      [B_e, D_out]
Combined output: y                  [B, D_out]
```

[DERIVATION] A tensorized implementation outline, not an API from the 2017 source.

---

# Part III. Why MoE Training Collapses and How the Paper Fixes It

## 031. Expert collapse: the positive feedback loop

[SOURCE] Authors observe gates favoring a few experts. Those experts get more examples and train faster, which makes the gate favor them still more.

\[
\text{more routing}
\rightarrow
\text{more updates}
\rightarrow
\text{stronger expert}
\rightarrow
\text{more routing}.
\]

This is the motivation for auxiliary utilization losses, not merely an interpretability preference.

[SOURCE, PDF p.5]

## 032. Two different meanings of “balance”

**Importance balance:** experts should receive similar **total gate weight** across a batch.

**Load balance:** experts should receive similar **numbers of dispatched input examples** across a batch.

A few high-weight inputs and many low-weight inputs can have similar total importance but very different compute and memory loads.

[SOURCE, PDF pp.5–6, 13]

## 033. Importance vector: Equation (6)

For input batch $X$:

\[
\operatorname{Importance}(X)=\sum_{x\in X}G(x).
\tag{6}
\]

The $i$th entry is the sum of all gate weights assigned to expert $i$.

## 034. Coefficient of variation

\[
\operatorname{CV}(v)=\frac{\operatorname{Std}(v)}{\operatorname{Mean}(v)}.
\]

[DERIVATION] This standard statistic is how Eq. (7) and Eq. (11) penalize relative disparities rather than absolute magnitudes.

## 035. Importance balancing loss: Equation (7)

\[
\mathcal L_{\mathrm{importance}}(X)
=w_{\mathrm{importance}}\operatorname{CV}(\operatorname{Importance}(X))^2.
\tag{7}
\]

Source location: main paper §4.

[SOURCE, PDF p.5]

## 036. Intuition of CV-squared regularization

[DERIVATION] If all experts receive identical total weight, their importance vector is constant, its standard deviation is zero, and the penalty vanishes. If a few receive disproportionate weight, dispersion grows.

## 037. Why importance alone is insufficient

[SOURCE] Importance can be balanced even if one expert sees a few inputs with large gate weights and another sees many with small weights.

In a distributed system the latter expert performs more forward/backward work and stores more activations. This motivates an independent load penalty.

[SOURCE, PDF p.6]

## 038. Actual load is discrete

\[
\operatorname{Count}_i(X)=\sum_{x\in X}\mathbf1\{G(x)_i>0\}.
\]

[DERIVATION] The expression counts inputs for expert $i$. However, it is discontinuous in Top-k choices and not suitable as a direct differentiable training target.

## 039. Key appendix trick: expected inclusion probability

Appendix A introduces $P(x,i)$, the probability that expert $i$ is selected if its independent Gaussian noise is resampled while the already-sampled noise of all **other** experts is held fixed.

This is a smooth inclusion probability based on a Gaussian CDF.

[SOURCE, PDF p.13]

## 040. Probability condition: Equation (8)

Let $s_i=(xW_g)_i$, $\sigma_i=\operatorname{Softplus}((xW_{\mathrm{noise}})_i)$. Then:

\[
P(x,i)=\Pr\left[s_i+\epsilon_i\sigma_i>
\operatorname{kth\_excluding}(H(x),k,i)\right].
\tag{8}
\]

Here $\operatorname{kth\_excluding}$ is the $k$th largest score among the **other** experts.

## 041. Why exclude expert $i$ from the threshold calculation?

[DERIVATION] The event “expert $i$ belongs to the top $k$” is equivalent to “its score exceeds the $k$th highest competing score.” Excluding its own score allows the conditional threshold to be regarded as fixed with respect to its noise variable, producing a tractable CDF.

## 042. Gaussian-CDF form: Equation (9)

\[
P(x,i)=\Phi\left(
\frac{(xW_g)_i-\operatorname{kth\_excluding}(H(x),k,i)}
{\operatorname{Softplus}((xW_{\mathrm{noise}})_i)}
\right).
\tag{9}
\]

$\Phi$ denotes the standard normal CDF.

[SOURCE, PDF p.13]

## 043. Derivation of Equation (9)

[DERIVATION]

\[
P(x,i)=\Pr[s_i+\epsilon_i\sigma_i>\theta_i]
=\Pr[\epsilon_i>(\theta_i-s_i)/\sigma_i].
\]

Since $\epsilon_i\sim\mathcal N(0,1)$:

\[
P(x,i)=1-\Phi((\theta_i-s_i)/\sigma_i)
=\Phi((s_i-\theta_i)/\sigma_i).
\]

where $\theta_i=\operatorname{kth\_excluding}(H(x),k,i)$.

## 044. Expected load: Equation (10)

\[
\operatorname{Load}(X)_i=\sum_{x\in X}P(x,i).
\tag{10}
\]

This is a differentiable *surrogate* for a discrete expert assignment count.

## 045. Load loss: Equation (11)

\[
\mathcal L_{\mathrm{load}}(X)
=w_{\mathrm{load}}\operatorname{CV}(\operatorname{Load}(X))^2.
\tag{11}
\]

[SOURCE, PDF p.13]

## 046. Complete conceptual task-plus-balance objective

\[
\mathcal L_{\mathrm{total}}=
\mathcal L_{\mathrm{task}}
+\mathcal L_{\mathrm{importance}}
+\mathcal L_{\mathrm{load}}.
\]

[DERIVATION] This collects the terms that the paper says are added to training. The authors do not present this exact combined equation as a numbered equation.

## 047. Balancing protects both performance and hardware

Importance/load constraints have two overlapping goals:

- Avoid a learned *rich-get-richer* expert collapse.
- Avoid one device/expert becoming the critical path or exhausting device memory.

[SOURCE] Appendix A directly tests the first and measures the second via maximum/mean load ratio.

## 048. Initialization for approximately balanced load

[SOURCE] The paper initializes $W_g$ and $W_{\mathrm{noise}}$ to **all zeros**, so the gate initially has no preference signal and some noise. The balancing penalties then have time to take effect without initial memory overload.

[SOURCE, PDF p.13]

[CRITIQUE] Exact edge-case behavior should be checked when implementing ties, k=1, zero noise scales, and all-zero initialization with a particular framework's numerical Softplus.

## 049. Table 6: balancing ablation (complete original values)

| $w_{importance}$ | $w_{load}$ | Test PPL ↓ | CV importance ↓ | CV load ↓ | Max/mean load ↓ |
|---:|---:|---:|---:|---:|---:|
| 0.0 | 0.0 | 39.8 | 3.04 | 3.01 | 17.80 |
| 0.2 | 0.0 | 35.6 | 0.06 | 0.17 | 1.47 |
| 0.0 | 0.2 | 35.7 | 0.22 | 0.04 | 1.15 |
| 0.1 | 0.1 | 35.6 | 0.06 | 0.05 | 1.14 |
| 0.01 | 0.01 | 35.7 | 0.48 | 0.11 | 1.37 |
| 1.0 | 1.0 | 35.7 | 0.03 | 0.02 | 1.07 |

[SOURCE, PDF p.13, Table 6] Same MoE-256-style architecture with different balance coefficients, trained for 10 epochs.

## 050. What Table 6 actually shows

The no-regularization model has both worse test perplexity and extreme load concentration. Adding *at least one* auxiliary balancing penalty fixes much of that damage in these experiments.

Crucial qualification: **balanced gate statistics are not evidence of useful semantic expert specialization by themselves.**

## 051. No universal balance weight in this paper

Different experiments use different balancing coefficients:

- 1B-word language model (Appendix C): $w_{importance}=w_{load}=0.1$.
- Translation experiments (Appendix E): $w_{importance}=w_{load}=0.01$ in the reported training protocol.

[SOURCE, PDF pp.15,17]

[INTERPRETATION] Auxiliary loss strength is problem-, architecture-, and load-dependent; transplanting $0.1$ verbatim into an unrelated financial regime is not justified.

## 052. Why “uniform importance” is not “uniform expertise”

[CRITIQUE] Load penalties encourage comparable total usage, not different learned functions. Multiple experts can have different IDs but similar outputs. To demonstrate specialization, evaluate conditional output differences, complementary errors, and input/market/regime selectivity, not merely CV scores.

## 053. When load balancing might be counterproductive

[EXTENSION] Financial data can contain genuinely unbalanced regime frequencies. Enforcing perfectly uniform routing over all regime experts may conflict with the data distribution. This is not tested in Shazeer et al. (2017).

Test a spectrum from:

\[
\lambda_{balance}=0
\quad\text{to stronger balance constraints},
\]

and measure prediction metrics and conditional expert utility, not merely occupancy.

---

# Part IV. Systems and Distributed Training

## 054. The shrinking-expert-batch problem

If batch size is $b$, total experts $n$, and each example activates $k$ experts, the average examples per expert is roughly:

\[
B_{expert}\approx \frac{bk}{n}.
\]

[SOURCE, PDF p.4]

[DERIVATION] This assumes approximately uniform assignments. The actual count can be badly skewed without balancing.

## 055. Why tiny expert batches are expensive

- GPU kernels become underutilized.
- Parameter loading and launch overhead are poorly amortized.
- Communication becomes large relative to useful computation.
- The slowest expert/device can dominate synchronous step time.

[SOURCE, PDF pp.2,4–5]

## 056. Mixed data and model parallelism

The paper's scheme:

- Standard LSTM layers and gate follow data-parallel replication.
- Experts are sharded, with a shared copy of each expert across replicas.
- Relevant inputs from multiple devices are dispatched to expert owners.
- Each expert processes the union of its selected samples.

[SOURCE, PDF pp.4–5]

## 057. Per-expert batch after combining devices

With $d$ data-parallel devices, each with local batch $b$, under approximately uniform loads:

\[
B_{expert}\approx\frac{kbd}{n}.
\]

Thus expert batch size improves by roughly a factor of $d$ relative to one local batch.

[SOURCE, PDF p.4]

## 058. “Convolutional” application across time positions

[SOURCE] The authors call the MoE independently at each text time position using the same expert/gate parameters, after a preceding recurrent layer has produced representations. The MoE can batch these time positions together.

This increases the dispatched batch by the number of unrolled positions, improving hardware efficiency.

## 059. Why this is not a recurrent expert model

The recurrent dependencies are in the surrounding LSTM layers. Each MoE itself is a feed-forward layer applied across time positions. The authors explicitly distinguish a hypothetical *recurrent MoE* whose current input depends on the previous MoE result.

[SOURCE, PDF p.5]

## 060. Bandwidth and arithmetic intensity

Expert inputs and outputs may need to travel between devices. The authors stress that fast expert computation is not enough: sufficient arithmetic must be performed per byte communicated.

For an expert with one hidden layer, a larger hidden dimension raises computation relative to input/output volume.

[SOURCE, PDF p.5]

## 061. Why larger experts can be hardware-efficient

[INTERPRETATION] A small expert may have a poor ratio of matrix multiplication to dispatch/communication overhead. Making each expert moderately wider can increase useful FLOPs while relatively fixed input/output traffic changes less.

[CRITIQUE] This is specific to the tested hardware and communication topology; not a blanket recommendation to widen MoEs in small Qlib experiments.

## 062. Recompute activations rather than store them

[SOURCE] On the 100B-word experiments, the authors avoid keeping expert hidden activations for backward pass and recompute them to save memory.

This is a precursor in spirit to modern activation checkpointing, but the particular implementation here is described at a systems level.

[SOURCE, PDF p.16]

## 063. Memory-saving optimizer details

[SOURCE] For the giant experiments, the paper reduces Adam's auxiliary storage by:

1. setting $\beta_1=0$ to eliminate first-moment tracking;
2. maintaining factored approximations to the second-moment accumulator using row and column averages.

These are **engineering details of the source experiment**, not generic requirements of Sparse MoE.

## 064. Why total parameter count can still matter

Even if inactive experts cost no forward FLOPs for a given example, they must be stored somewhere and trained on data over time.

[INTERPRETATION] The paper's giant-parameter results rely on distributing weights over tens to hundreds of GPUs; they do not demonstrate such models fit on one 16 GB card.

## 065. Router step vs. expert step

Possible separate computational components include:

\[
C_{total}=C_{gate}+C_{dispatch}+C_{active\ experts}+C_{combine}+C_{communication}.
\]

[DERIVATION] Conceptual cost accounting, not an exact formula given in the paper.

## 066. Why parameter-efficient is not the same as latency-efficient

Sparse MoE can increase total capacity at similar nominal active arithmetic but still incur extra:

- gather/scatter operations;
- irregular expert batch sizes;
- memory reads;
- distributed communication;
- synchronization.

[SOURCE / INTERPRETATION, PDF pp.2,4–5]

## 067. GPU efficiency actually reported

The experiments report observed TFLOPS/GPU on Tesla K40 devices.

- LSTM no-MoE baselines: roughly 1.07–1.29 TFLOPS/GPU.
- Low-compute MoEs: roughly 0.74–0.90 for most tested variants, excluding the underutilized 4-expert case.
- Highest-compute MoE: 1.56 TFLOPS/GPU.

[SOURCE, PDF p.7]

## 068. Why MoE-4 is an important control

The 4-expert condition with $k=4$ activates all four experts, so **there is no expert sparsity**. It behaves similarly to compute-matched dense alternatives and underuses available distributed hardware.

[SOURCE, PDF pp.6,15]

## 069. What matters for stock-research infrastructure

[EXTENSION] In CSI300/S&P500, the cross-section itself is only a few hundred stocks. A large $n$ with small $k$ can leave few examples per expert **per date**, even on a 16GB GPU. Aggregation across training dates may improve statistical batch size, but if the model uses same-date cross-stock interaction and varying universes, naive mixing across dates can change semantics.

## 070. Hardware-side metrics to log in a finance MoE experiment

Beyond RankIC and portfolio statistics:

- Active expert count per stock/day.
- Actual tokens/samples dispatched per expert.
- Gate-weight importance per expert.
- Gate/load CV and max/mean load.
- Step time, peak VRAM, GPU utilization.
- Router time and dispatch time.
- Total vs active parameter count.
- Throughput on same hardware as the baseline.

[EXTENSION] These are recommended measurements, not source-reported stock metrics.

---
# Part V. Original Experiments: What Was Actually Demonstrated

## 071. Original empirical program

The paper investigates:

1. Capacity scaling at roughly fixed active computation on 1-Billion-Word language modeling.
2. Additional computational scaling at high parameter count.
3. Effect of data scale up to 100 billion words.
4. WMT'14 single-pair machine translation.
5. Google production English→French machine translation.
6. Multilingual translation across twelve directed language pairs.
7. Importance/Load balancing loss ablations.
8. Qualitative expert specialization.
9. Training/hardware efficiency and several engineering alternatives.

[SOURCE, PDF pp.6–9 and Appendices A–G]

## 072. No financial experiments in the source

The paper contains no:

- stock returns or factor data;
- CSI300, S&P500 or Qlib experiment;
- RankIC, ICIR, Sharpe or drawdown;
- market-regime label;
- trading-cost model;
- codebook/VQ experiment;
- comparison against PRISM-VQ.

Any finance results in a future research system must be clearly marked as **new experiments**.

## 073. 1-Billion-Word benchmark description

[SOURCE] Source benchmark comprises about **829 million words**, from shuffled unique news sentences, with a **793,471-word vocabulary**.

The paper refers to it as the **1 Billion Word Language Modeling Benchmark**. The approximate benchmark title and exact corpus token count are not a contradiction.

[SOURCE, PDF p.6]

## 074. Why 1B-word tests are well suited to the claim

[INTERPRETATION] Language modeling creates many supervised token-level events and benefits from large representational capacity, making it a favorable setting for training many specialists.

[CRITIQUE] The transfer assumption to small financial datasets is nontrivial; a huge MoE may simply allocate too few usable updates per expert in finance.

## 075. 1B architecture: LSTM–MoE–LSTM

[SOURCE] Model structure:

```text
Word embedding
      ↓
LSTM layer 1
      ↓
Sparse MoE layer
      ↓
LSTM layer 2
      ↓
Output softmax
```

The MoE executes at each text position but is batched across positions after the first LSTM is evaluated.

[SOURCE, PDF pp.6,14]

## 076. 1B dimensions

Appendix C gives:

- Embedding dimension: 512.
- LSTM units: 512 for the low-computation experimental series.
- MoE input/output: 512.
- Expert MLP hidden size: 1024.
- Expert output: 512.
- ReLU hidden activation.
- Roughly one million parameters per expert (excluding small bias terms).
- Sigmoid after the MoE output, followed by dropout/residual machinery.

[SOURCE, PDF p.14]

## 077. Expert parameter approximation

\[
P_{expert}\approx512\cdot1024+1024\cdot512=1{,}048{,}576.
\]

[DERIVATION] This matches the paper's rounded **1M parameters per expert**.

## 078. Low-computation experiment's active cost

[SOURCE] Four experts are used per input. Approximate forward operations per position:

- first LSTM: 2M;
- MoE: 4M;
- second LSTM: 2M;
- total: roughly 8M ops/timestep, excluding the output softmax.

The paper's numbers are coarse counts and are not a modern profiler's exact GPU FLOP audit.

## 079. Low-computation model variants

- Flat MoE with 4, 32, 256 total experts.
- Hierarchical MoE with 256, 1024, 4096 total experts.
- For flat variants $k=4$.
- For hierarchical variants $k=2$ at each level, yielding four active leaf experts per input.

[SOURCE, PDF pp.6,14]

## 080. Compute-matched non-sparse baselines

The source also trains:

- **MoE-1-Wide:** one MLP expert with hidden size 4096.
- **MoE-1-Deep:** one MLP expert with four hidden layers of width 1024.
- **4xLSTM-512:** two extra 512-unit LSTMs instead of MoE.
- **LSTM-2048-512:** wide LSTM projected to 512.
- **MoE-4:** four experts all activated ($k=4$), thus a non-sparse special case.

[SOURCE, PDF p.15]

## 081. Purpose of compute matching

[INTERPRETATION] A comparison of a 4096-expert model with a small dense model is meaningful only when one separates:

- active operations per example;
- number of trainable parameters;
- wall-clock time;
- device count;
- train examples processed.

The paper performs both approximate compute matching and some explicit hardware throughput reporting.

## 082. Table 7: Full 1B-word experimental data

Values transcribed from the source; blank entries mean the paper's table does not report them.

| Model | Test PPL @10 epochs ↓ | Final test PPL ↓ | Ops/timestep (M) | Params excl. emb./softmax (M) | Total params (B) | DropProb | TFLOPS/GPU |
|---|---:|---:|---:|---:|---:|---:|---:|
| Kneser-Ney 5-gram* | — | 67.6 | 0.00001 | — | 1.8 | — | — |
| LSTM-512-512* | — | 54.1 | 2.4 | 2.4 | 0.8 | 0.1 | — |
| LSTM-1024-512* | — | 48.2 | 4.7 | 4.7 | 0.8 | 0.1 | — |
| LSTM-2048-512* | 45.0 | 43.7 | 9.4 | 9.4 | 0.8 | 0.1 | 0.61 |
| LSTM-2048-512 (rerun) | 44.7 | — | 9.4 | 9.4 | 0.8 | 0.1 | 1.21 |
| 4xLSTM-512 | 46.0 | — | 8.4 | 8.4 | 0.8 | 0.1 | 1.07 |
| MoE-1-Wide | 46.1 | — | 8.4 | 8.4 | 0.8 | 0.1 | 1.29 |
| MoE-1-Deep | 45.7 | — | 8.4 | 8.4 | 0.8 | 0.1 | 1.29 |
| MoE-4 | 45.0 | — | 8.4 | 8.4 | 0.8 | 0.1 | 0.52 |
| MoE-32 | 39.7 | — | 8.4 | 37.8 | 0.9 | 0.1 | 0.87 |
| MoE-256 | 35.7 | — | 8.6 | 272.9 | 1.1 | 0.1 | 0.81 |
| MoE-256-h | 36.0 | — | 8.4 | 272.9 | 1.1 | 0.1 | 0.89 |
| MoE-1024-h | 34.6 | — | 8.5 | 1079.0 | 1.9 | 0.2 | 0.90 |
| MoE-4096-h | 34.1 | — | 8.9 | 4303.4 | 5.1 | 0.2 | 0.74 |
| 2xLSTM-8192-1024* | 34.7 | 30.6 | 151.0 | 151.0 | 1.8 | 0.25 | 1.09 |
| MoE-34M | 31.3 | — | 33.8 | 4313.9 | 6.0 | 0.3 | 1.22 |
| MoE-143M | 28.0 | — | 142.7 | 4371.1 | 6.0 | 0.4 | 1.56 |

[SOURCE, PDF p.15, Table 7; overview p.7 Table 1]

## 083. How to read Table 7 correctly

- “PPL @10 epochs” and “final PPL” are **different columns**.
- The best published baseline has **34.7 after 10 epochs** and **30.6 after 100 epochs**.
- The high-budget MoE records **28.0 after 10 epochs**.
- The low-compute MoE-4096-h records **34.1 after 10 epochs**, around 8.9M ops/timestep.
- The 0.8–6.0B total parameter figures include very large embedding/softmax components and are not identical to the MoE layer's unique parameter count.

[CRITIQUE] Comparing 28.0 with 30.6 without saying how many epochs and at what training cost would be misleading.

## 084. Core 1B result: model capacity improves quality at near-constant active work

With approximately 8–9M ops/timestep, test perplexity falls as number of experts rises:

- MoE-4: 45.0;
- MoE-32: 39.7;
- MoE-256: 35.7;
- MoE-1024-h: 34.6;
- MoE-4096-h: 34.1.

[SOURCE, PDF p.15, Table 7]

This is the central scaling demonstration, **not** a proof that more experts work in all low-data domains.

## 085. Does hierarchical MoE always beat flat MoE?

At 256 total experts:

- Flat MoE-256: 35.7 PPL.
- Hierarchical MoE-256-h: 36.0 PPL.

[SOURCE, PDF p.15]

[CRITIQUE] The two-level scheme improves the practicality of very high expert counts; its mere presence is not an accuracy guarantee at identical size.

## 086. High-budget 1B results

High-capacity variants with different active compute:

| Variant | Test PPL @10 epochs | Ops/timestep (M) | Params excluding embedding/softmax (M) |
|---|---:|---:|---:|
| Low budget | 34.1 | 8.9 | 4303.4 |
| Medium budget | 31.3 | 33.8 | 4313.9 |
| High budget | 28.0 | 142.7 | 4371.1 |

[SOURCE, PDF pp.7,15]

## 087. Active computation still helps

[SOURCE] Increasing computation from ~8.9M to ~142.7M while keeping a very large parameter budget yields better perplexity.

[INTERPRETATION] Sparse capacity is not a substitute for adequate active model computation.

## 088. 1B training and optimizer

Appendix C:

- Framework: TensorFlow.
- Hardware: cluster of 16 Tesla K40 GPUs for matched-budget models.
- Batch: sentences totaling roughly 300,000 words.
- Duration: maximum 10 epochs, about 27,000 steps.
- Optimizer: Adam.
- LR: linear warm-up 1,000 steps, then inverse-square-root decay.
- Dropout: tuned separately, search steps of 0.1.
- Importance/load coefficients: 0.1 / 0.1.
- Per-model time: roughly 12–16 hours except MoE-4 (~18h), depending on configuration.

[SOURCE, PDF pp.14–15]

## 089. Original Figure 2

[SOURCE, PDF p.6]

- **Left:** perplexity versus model parameter capacity for approximately equal active computations.
- **Right:** perplexity versus active compute budget, comparing high-capacity MoE designs with previously published LSTM points.

[INTERPRETATION] These two panels deliberately isolate *different axes* of scale: capacity and computation.

## 090. 100-Billion-Word corpus

[SOURCE] The paper additionally builds a much larger internal news corpus totaling about 100 billion words. It trains under roughly constant ~8–10M ops/timestep and dramatically increases total number of experts.

This dataset is not a public financial dataset or an open finance replication resource.

[SOURCE, PDF pp.7–8]

## 091. 100B total expert counts

Models include:

\[
n\in\{32,256,1024,4096,16384,65536,131072\}.
\]

The paper's largest model reaches roughly **137 billion parameters in the MoE component**.

## 092. Table 8: Complete 100B-word scalability data

| Model | Test PPL @0.1 epoch ↓ | Test PPL @1 epoch ↓ | Ops/timestep (M) | Params excl. emb./softmax (M) | Total params (B) | TFLOPS/GPU |
|---|---:|---:|---:|---:|---:|---:|
| Kneser-Ney 5-gram | 67.1 | 45.3 | 0.00001 | — | 76.0 | — |
| 4xLSTM-512 | 54.5 | 47.0 | 8.4 | 8.4 | 0.1 | 1.23 |
| MoE-32 | 48.5 | 40.4 | 8.4 | 37.8 | 0.1 | 0.83 |
| MoE-256-h | 42.8 | 35.3 | 8.4 | 272.9 | 0.4 | 1.11 |
| MoE-1024-h | 40.3 | 32.7 | 8.5 | 1079.0 | 1.2 | 1.14 |
| MoE-4096-h | 38.9 | 30.9 | 8.6 | 4303.4 | 4.4 | 1.07 |
| MoE-16384-h | 38.2 | 29.7 | 8.8 | 17201.0 | 17.3 | 0.96 |
| MoE-65536-h | 38.2 | **28.9** | 9.2 | 68791.0 | 68.9 | 0.72 |
| MoE-131072-h | 39.8 | 29.2 | 9.7 | 137577.6 | 137.7 | 0.30 |

[SOURCE, PDF p.16, Table 8]

## 093. 100B result: the best is not the largest

**Important negative result:**

\[
\text{65,536 experts: PPL }28.9
\qquad\text{vs.}\qquad
\text{131,072 experts: PPL }29.2.
\]

The largest model worsens while parameter count nearly doubles.

[SOURCE] Authors suggest excessive sparsity as a possible explanation, without experimentally establishing that it is the sole cause.

## 094. 100B quality improvement

Relative to the compute-matched 4xLSTM baseline (PPL 47.0), the 65,536-expert model reaches PPL 28.9—approximately 39% lower perplexity (the paper's stated interpretation).

This result supports sparse capacity scaling *on massive text corpora*, not universal benefit for cross-sectional returns.

## 095. Figure 3: data-size interaction

[SOURCE, PDF p.7] Test perplexity is plotted against model capacity at ~10B and ~100B observed training words. The extra data permits more benefit from extra expert capacity before saturation.

[INTERPRETATION] Data scale is a causal *candidate* for whether specialist capacity is worthwhile. Exactly how much data stock markets supply is much more limited, especially after regime conditioning and temporal dependence.

## 096. 100B training systems details

Appendix D:

- 32 Tesla K40 GPUs typically; up to 64 or 128 for largest variants.
- Total batch roughly 2.5 million words.
- Single pass over approximately 100 billion words.
- Expert activations recomputed rather than fully stored.
- Reduced-memory optimizer: $\beta_1=0$ and factored second-moment statistics.

[SOURCE, PDF p.16]

## 097. Communication efficiency can decline at extreme scale

The 131,072-expert model has measured throughput of **0.30 TFLOPS/GPU**, well below many smaller models. Appendix D attributes this partly to unchanged overall batch size despite more devices.

[SOURCE, PDF pp.16–17]

This illustrates that even effective conditional computation can become throughput-limited by expert batch/sharding choices.

## 098. Machine translation: main setup

[SOURCE] The paper modifies the GNMT architecture. In single-pair translation:

- Encoder LSTM layers reduced from 9 to 3.
- Decoder LSTM layers reduced from 8 to 2.
- MoE layers inserted in encoder and decoder.
- Up to 2,048 experts per MoE layer.
- About 2 million parameters per expert.
- Around 8.7 billion total model parameters.
- Approximately 85M operations/timestep (paper convention).

[SOURCE, PDF pp.8,17]

## 099. WMT'14 train/development/test protocol

- English→French: approximately 36M sentence pairs.
- English→German: approximately 5M pairs.
- Development: combination of `newstest2012` and `newstest2013`.
- Test: `newstest2014`.

[SOURCE, PDF p.8]

## 100. Table 2: English→French

| System | Test PPL ↓ | Test BLEU ↑ | Ops/timestep (M) | Total params | Training |
|---|---:|---:|---:|---:|---|
| MoE 2048 experts | 2.69 | 40.35 | 85 | 8.7B | 3 days / 64 K40 |
| MoE 2048 experts (longer) | 2.63 | **40.56** | 85 | 8.7B | 6 days / 64 K40 |
| GNMT | 2.79 | 39.22 | 214 | 278M | 6 days / 96 K80 |
| GNMT + RL | 2.96 | 39.92 | 214 | 278M | 6 days / 96 K80 |
| PBMT | — | 37.0 | — | — | — |
| LSTM (6-layer) | — | 31.5 | — | — | — |
| LSTM (6-layer + PosUnk) | — | 33.1 | — | — | — |
| DeepAtt | — | 37.7 | — | — | — |
| DeepAtt + PosUnk | — | 39.2 | — | — | — |

[SOURCE, PDF p.8, Table 2]

## 101. Table 3: English→German

| System | Test PPL ↓ | Test BLEU ↑ | Ops/timestep (M) | Total params | Training |
|---|---:|---:|---:|---:|---|
| MoE 2048 experts | **4.64** | **26.03** | 85 | 8.7B | 1 day / 64 K40 |
| GNMT | 5.25 | 24.91 | 214 | 278M | 1 day / 96 K80 |
| GNMT + RL | 8.08 | 24.66 | 214 | 278M | 1 day / 96 K80 |
| PBMT | — | 20.7 | — | — | — |
| DeepAtt | — | 20.6 | — | — | — |

[SOURCE, PDF p.8, Table 3]

## 102. Table 4: Google production English→French

| System | Eval PPL ↓ | Eval BLEU ↑ | Test PPL ↓ | Test BLEU ↑ | Ops/timestep (M) | Total params | Training |
|---|---:|---:|---:|---:|---:|---|---|
| MoE 2048 experts | 2.60 | 37.27 | 2.69 | **36.57** | 85 | 8.7B | 1 day / 64 K40 |
| GNMT | 2.78 | 35.80 | 2.87 | 35.56 | 214 | 278M | 6 days / 96 K80 |

[SOURCE, PDF p.8, Table 4]

## 103. Translation result interpretation

[SOURCE] Reported gains include 40.56 BLEU vs GNMT+RL's 39.92 on English→French and 26.03 vs GNMT's 24.91 on English→German.

[CRITIQUE] The headline “+1.34 BLEU” on English→French refers to comparison against GNMT (39.22), whereas comparison with GNMT+RL (39.92) is smaller. It matters which baseline is named.

## 104. Translation hardware caveat

Different systems are trained on **different GPU families/counts** (K40 vs K80, 64 vs 96 devices). Paper-reported training time and nominal operations support a compelling systems case but are not a perfect same-device latency comparison.

[CRITIQUE] For a new paper claiming “efficient MoE,” replicate on the same hardware and use both wall clock and active operation count.

## 105. Multilingual setup

[SOURCE] The multilingual model is trained on twelve directed translation tasks in one network and compared with:

- separate monolingual GNMT models;
- a single multilingual GNMT model.

The MoE model processes approximately **3 billion training sentence pairs** and uses ~8.7B parameters.

[SOURCE, PDF p.9]

## 106. Multilingual differences from single-pair translation

Appendix E says the multilingual model uses:

- nonhierarchical MoE layers;
- 512 experts;
- top-2 routing;
- each expert with larger hidden width (8192);
- Noisy Top-k gating rather than Strictly Balanced Gating;
- approximately 102M ops/timestep.

[SOURCE, PDF p.17]

## 107. Table 5: full twelve-pair translation scores

| Direction | GNMT-Mono BLEU | GNMT-Multi BLEU | MoE-Multi BLEU | MoE-Multi minus GNMT-Multi |
|---|---:|---:|---:|---:|
| French → English | 36.47 | 34.40 | 37.46 | +3.06 |
| German → English | 31.77 | 31.17 | 34.80 | +3.63 |
| Japanese → English | 23.41 | 21.62 | 25.91 | +4.29 |
| Korean → English | 25.42 | 22.87 | 28.71 | +5.84 |
| Portuguese → English | 44.40 | 42.53 | 46.13 | +3.60 |
| Spanish → English | 38.00 | 36.04 | 39.39 | +3.35 |
| English → French | 35.37 | 34.00 | 36.59 | +2.59 |
| English → German | 26.43 | 23.15 | 24.53 | +1.38 |
| English → Japanese | 23.66 | 21.10 | 22.78 | +1.68 |
| English → Korean | 19.75 | 18.41 | **16.62** | **-1.79** |
| English → Portuguese | 38.40 | 37.35 | 37.90 | +0.55 |
| English → Spanish | 34.50 | 34.25 | 36.21 | +1.96 |

[SOURCE, PDF p.9, Table 5]

## 108. Other multilingual metadata

Table 5 also reports:

- GNMT-Mono: 278M params per model.
- GNMT-Multi: 278M params, 212M ops/timestep, 21 days / 96 K20 GPUs.
- MoE-Multi: 8.7B params, 102M ops/timestep, 12 days / 64 K40 GPUs.
- Dev perplexity: GNMT-Multi 4.14 vs MoE-Multi 3.35.

[SOURCE, PDF p.9]

## 109. Multilingual positive result

The MoE multilingual model improves on GNMT-Multi in **11 of 12** language directions.

[SOURCE] It even exceeds several of the individually trained monolingual systems (paper states 8 of 12).

## 110. Important negative result: English→Korean

MoE-Multi scores **16.62** while GNMT-Multi scores **18.41** on English→Korean.

The authors suggest severe overtraining due to oversampling scarce real examples as a possible reason.

[SOURCE, PDF p.9]

[INTERPRETATION] A model can improve average performance through expert capacity and still harm a rare or nonstationary subgroup.

## 111. Why multilingual results matter for finance

[EXTENSION] Multilingual MoE is an analogy for multi-market or multi-regime modeling:

- A single model shares information across related tasks.
- Experts specialize where task patterns differ.
- Low-frequency groups may suffer from poor sample diversity and overfitting.

This is *not* evidence that US/China stock returns behave like different languages. It is a hypothesis-generation analogy only.

## 112. Qualitative expert specialization

[SOURCE, PDF p.18, Table 9] Authors sort inputs by gate weight and inspect contexts associated with different expert IDs. Examples:

- Expert 381 often appears in contexts involving research or innovation.
- Expert 752 often appears near phrases of importance, leadership, or centrality.
- Expert 2004 often appears near rapidity/speed expressions.

This is a qualitative example of lexical/syntactic specialization.

## 113. Limitations of semantic specialization evidence

[CRITIQUE] Table 9 selects illustrative contexts for a small number of experts. It does not establish:

- that all experts specialize;
- that expert semantics are stable across random seeds;
- a causal relation between one expert and a linguistic function;
- optimality or exclusivity of the discovered experts;
- direct economic interpretation in another domain.

## 114. One-line empirical summary

The core finding is **better language modeling and translation quality at comparable or lower active computation when much larger total capacity is distributed across sparsely selected experts**, subject to balancing, dispatch, memory, bandwidth and data-scale requirements.

---

# Part VI. Figures, Tables and Evidence Structure

## 115. Figure 1: Sparse MoE in a recurrent network

[SOURCE, PDF p.2]

What the diagram shows:

- the MoE is inserted *between* ordinary recurrent layers;
- every input position is routed separately;
- a gate selects two experts in the diagram's illustrative instance;
- expert outputs are weighted and summed;
- non-selected experts are not computed for that input.

Why it matters: **MoE is a modular layer**, not necessarily the entire predictor.

## 116. Figure 2: quality vs two scaling axes

[SOURCE, PDF p.6]

- Left: fixed-ish active computation, increasing total parameter capacity.
- Right: fixed-ish high total capacity, increasing active computation.

[INTERPRETATION] These are two distinct controlled questions. A research Agent should copy this *factorial logic* when analyzing whether improvements come from extra available experts or extra active FLOPs.

## 117. Figure 3: larger datasets support larger expert vocabulary

[SOURCE, PDF p.7]

Comparison of the 10B-observation and 100B-observation phases across expert counts. At the full data scale, improvement continues to far higher expert counts before the final 131072-expert reversal.

## 118. Figure 4: model perplexity vs number of source words processed

[SOURCE, PDF p.18]

The source explicitly notes early differences in curves partly reflect different batch sizes. It plots both WMT'14 and a Google production English→French experiment.

[CRITIQUE] Curves are valuable for learning dynamics but do not by themselves separate every optimization difference from model architecture.

## 119. Table map

| Source | Scientific role |
|---|---|
| Table 1, p.7 | Summary of 1B high-capacity/computation comparison |
| Table 2, p.8 | WMT En→Fr benchmark |
| Table 3, p.8 | WMT En→De benchmark |
| Table 4, p.8 | Google production En→Fr |
| Table 5, p.9 | 12 multilingual translation directions |
| Table 6, p.13 | Importance/load balance ablation |
| Table 7, p.15 | Full 1B language model architecture/results |
| Table 8, p.16 | Full 100B model scaling |
| Table 9, p.18 | Illustrative expert contexts |

## 120. “Strong results” are multi-objective here

The paper assesses:

- test perplexity or BLEU;
- active ops per timestep;
- total parameter count;
- GPU training duration;
- GPU throughput;
- load imbalance.

[INTERPRETATION] This is more informative than reporting only a task score. Modern financial MoE evaluations should likewise track both predictive outcomes and routing efficiency.

---

# Part VII. Appendix-Only Mechanisms That Matter

## 121. Appendix A: differentiable load estimator

[SOURCE, PDF p.13]

- Why load as an actual count is nondifferentiable.
- Why injecting independent Gaussian noise yields an analytic conditional probability.
- Eqs. (8)–(11) for inclusion probability, expected load and CV loss.
- Initialization to balanced traffic.
- Table 6 showing substantial gains from balancing regularizers.

For an Agent, this appendix is **core mechanism**, not incidental implementation trivia.

## 122. Appendix B: two-level hierarchical MoE

[SOURCE] If $n$ is extremely large, a one-shot $n$-way router can be unwieldy. The paper groups experts and uses:

- primary gate over groups;
- secondary gate within each group;
- weighted selection of leaf experts.

[SOURCE, PDF p.14]

## 123. Hierarchical MoE Equation (12)

For group $i$ and within-group expert $j$:

\[
y_H=\sum_{i=1}^{a}\sum_{j=1}^{b}
G_{\mathrm{primary}}(x)_i\;G_i(x)_j\;E_{i,j}(x).
\tag{12}
\]

[SOURCE, PDF p.14]

## 124. Active experts under two-level routing

[DERIVATION] If the primary selects $k_1$ groups and each selected group chooses $k_2$ leaves, the maximum number of leaf experts evaluated per input is:

\[
k_{active}=k_1k_2.
\]

In the paper's large hierarchical LM experiments, $k_1=k_2=2$, yielding four active leaf experts.

## 125. Hierarchical importance: Equation (13)

\[
\operatorname{Importance}_H(X)_{i,j}
=\sum_{x\in X}
G_{\mathrm{primary}}(x)_i\,G_i(x)_j.
\tag{13}
\]

The leaf's effective importance is weighted by **both** levels.

## 126. Hierarchical load: Equation (14)

\[
\operatorname{Load}_H(X)_{i,j}
=\frac{
\operatorname{Load}_{\mathrm{primary}}(X)_i
\operatorname{Load}_{i}(X^{(i)})_j
}{|X^{(i)}|}.
\tag{14}
\]

[SOURCE] The authors choose this form to ensure gradients can influence the **primary gate**; a simpler inner-only estimator would miss that dependency.

[SOURCE, PDF p.14]

## 127. Hierarchical route capacity vs interpretability

[CRITIQUE] A two-level gate need not represent meaningful financial hierarchy. Primary groups are trained based on predictive/task and load objectives; labeling them “bull/bear” or “sector” without tests is unwarranted.

[EXTENSION] If a finance model intentionally assigns market regimes to level 1 and structural stock codes to level 2, that is a new inductive bias, not the 2017 paper's established meaning.

## 128. Appendix C: one-billion-word reproduction details

Contains:

- model dimensions;
- residual/dropout arrangement;
- expert hidden/output sizes;
- flat/hierarchical k;
- compute-matched baseline alternatives;
- optimizer and schedule;
- full Table 7;
- larger compute variants.

[SOURCE, PDF pp.14–16]

## 129. Appendix D: very-large-data systems details

Contains:

- GPU/device scaling;
- expert counts up to 131072;
- activation recomputation;
- optimizer memory reduction;
- full Table 8;
- analysis of the largest model's low measured throughput.

[SOURCE, PDF pp.16–17]

## 130. Appendix E: GNMT architecture and training details

Single-pair:

- 512-dimensional MoE input/output;
- 2048-unit expert ReLU hidden layers;
- up to 2048 experts per MoE;
- $k=2$ at each hierarchy level for four active experts;
- Strictly Balanced Gating used for single-pair experiments;
- Adam, warm-up 2000 steps, constant for 8000, inverse-square-root decay;
- DropProb 0.4; synchronization across up to 64 GPUs.

Multilingual:

- flat 512-expert MoE;
- top-2;
- 8192-unit expert hidden layers;
- approximately 102M ops/timestep.

[SOURCE, PDF p.17]

## 131. Strictly Balanced Gating is an alternate branch

[SOURCE] Appendix F explains that some machine-translation experiments used a **different masking scheme** because of then-existing infrastructure behavior. It enforces exactly equal batch size *per expert*, as opposed to relying only on soft regularization.

**Do not equate this mechanism with the normal Noisy Top-k gate in Eq. (3).**

[SOURCE, PDF pp.18–19]

## 132. Strictly balanced gate: Equation (15)

The unmasked scores still start from:

\[
G_\sigma(x)=\operatorname{Softmax}(xW_g).
\tag{15}
\]

## 133. Masked normalized gate: Equation (16)

\[
G(x)_i=
\frac{G_\sigma(x)_i M(G_\sigma(x))_i}
{\sum_{j=1}^{n}G_\sigma(x)_jM(G_\sigma(x))_j}.
\tag{16}
\]

[VERIFY] The normalization assumes an input has at least one selected expert. Edge-case/fallback behavior for zero assignments is not specified in the compact source formula; check original implementation for production use.

## 134. Ordinary binary Top-k mask: Equation (17)

\[
\operatorname{TopK}(v,k)_i=
\begin{cases}1,&v_i\text{ selected};\\0,&\text{otherwise.}\end{cases}
\tag{17}
\]

## 135. Batchwise top-m per expert: Equation (18)

For a whole input batch $X$ and target per-expert capacity:

\[
m=\frac{k|X|}{n},
\]

then the mask is defined by selecting the **top $m$ examples for each expert** rather than the top $k$ experts for each example.

\[
M_{\mathrm{batchwise}}(X,m)_{j,i}=
\begin{cases}
1,&X_{j,i}\text{ ranks in top }m\text{ for expert }i,\\
0,&\text{otherwise.}
\end{cases}
\tag{18}
\]

[SOURCE, PDF p.19]

## 136. Important distinction in batchwise mask

[DERIVATION] Fixing exactly $m$ assigned samples **per expert** does **not** fix exactly $k$ experts **per sample**. Some samples may receive multiple experts and some may receive none depending on the assignment. Source Eq. (18) guarantees the per-expert count, not the per-example count.

[CRITIQUE] This distinction can matter when implementing Eq. (16), because an all-zero row is undefined after normalization. The PDF does not provide full runtime fallback behavior.

## 137. Threshold approximation at inference: Equation (19)

Since inference may not have a large batch to compute the same batchwise top-$m$ assignments, a learned expert-wise threshold vector $T$ approximates routing:

\[
M_{\mathrm{threshold}}(x,T)_i=
\begin{cases}
1,&x_i>T_i,\\0,&\text{otherwise.}
\end{cases}
\tag{19}
\]

[SOURCE, PDF p.19]

## 138. Threshold learning: Equation (20)

The paper supplies an additional loss encouraging agreement between batchwise assignments and threshold decisions, written schematically as:

\[
\mathcal L_{\mathrm{batchwise}}(X,T,m)=
\sum_{j=1}^{|X|}\sum_{i=1}^{n}
\big[M_{\mathrm{threshold}}(X_j,T)_i-
M_{\mathrm{batchwise}}(X,m)_{j,i}\big](X_{j,i}-T_i).
\tag{20}
\]

[VERIFY] Eq. (20) reuses $X$ for the matrix of gate values in this appendix notation, not necessarily the original raw input vectors. The exact masking/gradient implementation is only partially specified in this excerpt.

[SOURCE, PDF p.19]

## 139. Why strict balancing should not be copied blindly

- It introduces batch-dependent routing.
- Inference requires a learned surrogate threshold mechanism.
- It targets **exact equal expert workload**, not necessarily best predictive specialization.
- In sparse financial cross-sections, batchwise selection may couple unrelated stocks in ways that change if the universe changes.

[INTERPRETATION / EXTENSION] Consider it a separately testable routing family, not the default replacement for Noisy Top-k.

## 140. Appendix G: attention efficiency is not MoE routing

Appendix G describes an *independent* implementation detail to speed up encoder-decoder attention in the GNMT-derived translation models.

The authors compare additive tanh attention and a factorized, matrix-multiplication-friendly alternative.

[SOURCE, PDF p.19]

## 141. Original GNMT-like attention: Equation (21)

\[
A_{GNMT}(x_i,y_j)=\sum_{d=1}^{n}V_d\,
\tanh\big((x_iU)_d+(y_jW)_d\big).
\tag{21}
\]

## 142. Fast alternative: Equation (22)

\[
A(x_i,y_j)=\sum_{d=1}^{n}V_d\,
\tanh((x_iU)_d)\tanh((y_jW)_d).
\tag{22}
\]

[SOURCE] This permits more efficient matrix multiplications across source/target positions, and the authors report little difference in translation quality.

## 143. Why include Appendix G in a foundations MD?

[INTERPRETATION] It demonstrates an important research-engineering practice: change the *factorization of an interaction* to improve compute efficiency while preserving functional quality. It is not part of the Noisy Top-k MoE mechanism itself.

## 144. Source equation index

| Equations | Content | PDF page |
|---|---|---:|
| (1) | Sparse weighted expert sum | 3 |
| (2) | Dense softmax gate | 4 |
| (3)–(5) | Noisy Top-k gate and masking | 4 |
| (6)–(7) | Importance and importance loss | 5 |
| (8)–(11) | Conditional selection probability and load loss | 13 |
| (12)–(14) | Hierarchical gate/output/utilization | 14 |
| (15)–(20) | Strictly balanced batchwise gating | 18–19 |
| (21)–(22) | Translation attention alternatives | 19 |

---
# Part VIII. Claim–Evidence Map and Critical Reading

## 145. Claim A: sparse gating massively expands capacity

**[SOURCE CLAIM]** More total expert parameters can be added while per-position expert computation is held roughly fixed.

**Evidence:** Figure 2, Tables 7–8; capacity grows from tens of millions to tens of billions of expert parameters with only a few active experts.

**Assessment:** Strongly supported in the giant language-model configurations, with infrastructure-specific qualifications.

## 146. Claim B: conditional computation is practically efficient

**[SOURCE CLAIM]** The approach achieves major capacity gains with moderate loss of arithmetic efficiency.

**Evidence:** TFLOPS/GPU, wall-clock times and fixed-ops comparisons; device-parallel batching and time-position batching.

**Assessment:** Compelling for the measured GPU clusters, not a general proof of low-latency efficiency on a single commodity GPU.

## 147. Claim C: auxiliary losses prevent harmful expert imbalance

**Evidence:** Table 6, particularly 39.8 PPL / 17.80 max-to-mean load without regularizers vs 35.6 / 1.14 with both regularizers at 0.1.

**Assessment:** Strong in the reported LM ablation.

## 148. Claim D: expert specialization emerges

**Evidence:** Table 9, hand-inspected high-gate contexts for selected translation experts.

**Assessment:** Suggestive qualitative support; no universal semantic-identification guarantee.

## 149. Claim E: more data permits more expert capacity

**Evidence:** Figure 3 and the 100B-word scaling results compared with smaller data.

**Assessment:** Supported within the tested text domain; optimal scale depends on representation, training regime, regularization and data.

## 150. Claim F: hierarchical routing is necessary for very large $n$

**Evidence:** Hierarchical MoE used in the largest configurations; described as reducing routing branching factor.

**Assessment:** Shows one practical scaling method. It does **not** establish that hierarchy is generally more accurate than flat routing at the same number of experts, and Table 7 gives a 256-expert counterexample.

## 151. Limitation: no broad hyperparameter / architecture disentanglement

The paper varies expert counts, active compute and some layout options, but it does not present a modern exhaustive factorial study across:

- routing noise scale;
- $k$;
- balance coefficients;
- expert depth and width;
- backbone type;
- dataset sizes;
- independent seed robustness in every result.

[CRITIQUE] Thus not every improvement can be attributed to “sparsity” alone.

## 152. Limitation: expertise might depend on task locality

Token-level syntax and semantics create repeated subproblems well suited to conditional experts. Financial cross-sectional signals may be less stationary, more continuous and less cleanly partitioned.

[EXTENSION] A finance transfer experiment should **verify the existence of stable conditional subproblems** before using a large expert vocabulary.

## 153. Limitation: large dataset is both an assumption and an outcome driver

The authors themselves stress that large data is critical. Their 100B-word model improves as data scale grows, but shows degradation at excessive expert count.

[CRITIQUE] This sharply limits any inference that $n=2048$ or $n=65536$ is appropriate in CSI300.

## 154. Limitation: expert usage ≠ expert utility

Even a perfectly balanced gate can dispatch examples among functionally redundant experts.

[EXTENSION] Evaluate expert *counterfactual contribution* to stock rankings, not only routing percentages.

## 155. Limitation: task scores can mask subgroup degradation

English→Korean degrades significantly despite overall multilingual improvement.

[EXTENSION] In quant finance, compare regime, year, size, volatility, industry and liquidity subsets to detect harm concentrated in minority conditions.

## 156. Limitation: 2017-era distributed hardware

The paper's K40 GPU clusters and communication assumptions are historically specific. Today's GPU software stacks differ substantially. The same principles remain relevant but runtime speedups must be rebenchmarked.

## 157. Limitation: no modern extreme-load capacity management

[CRITIQUE] The paper describes soft importance/load penalties and a special strictly balanced scheme, but it does not introduce the full set of later Transformer MoE conventions such as today's capacity-factor token dropping, top-1 Switch routing or expert choice routing as a single unified implementation.

Do not retroactively attribute those later designs to this paper.

## 158. Limitations explicitly visible in source text

- Excess sparsity can damage quality (131072 expert case).
- Per-expert microbatch shrinkage harms efficiency.
- Network bandwidth can dominate.
- Unequal load can cause memory/throughput problems.
- Not all language directions improve.
- Extremely large models require large training corpora and distributed hardware.

## 159. Future work actually in the original conclusion

The authors envision conditional computation being applied in other domains **provided sufficient training data**. Earlier in the paper they also discuss recurrent MoE possibilities and scaling farther toward enormous parameter/data scales.

[SOURCE, PDF pp.5,9]

[VERIFY] There is no formal numbered list of financial-machine-learning future directions in the original paper.

## 160. What the paper does not establish

The source alone does **not** establish that:

- noisy Top-k is better than all later routing methods;
- an expert corresponds to one semantic or economic factor;
- experts must have different architectures;
- one router will generalize through financial crises;
- scaling to billions of parameters is always useful;
- a MoE architecture ensures better IC/RankIC;
- higher MoE parameter count guarantees better risk-adjusted investment performance.

## 161. Evidence quality ranking for an Agent

| Claim | Evidence strength within original experiments | Main qualification |
|---|---|---|
| Sparse capacity scaling in text | Strong | Very large corpora and distributed GPUs |
| Balance regularization improves utilization | Strong | Tested in one representative LM configuration |
| More experts always better | Contradicted | 131072 experts worse than 65536 |
| Specialist interpretation | Suggestive | Selected qualitative contexts only |
| Hardware efficiency | Moderate to strong | Historical devices and workloads |
| Cross-domain financial adaptation | Not tested | Requires independent experiments |

---

# Part IX. Inductive Bias and Mechanism Inventory

## 162. Inductive bias 1: the task decomposes into recurring subproblems

A sparse expert layer assumes an input-conditioned subset of functions can model data better than applying exactly one general mapping everywhere.

[INTERPRETATION] This is useful only if there is structure that experts can reuse across many samples.

## 163. Inductive bias 2: conditional complexity is valuable

Rather than spending all available parameters on every input:

\[
\text{available capacity}\gg\text{active capacity per example}.
\]

Large models can memorize/represent diverse substructures without proportionally increasing active arithmetic.

## 164. Inductive bias 3: each expert is learnable and reusable

The same expert parameters are shared by all inputs routed to that expert.

[INTERPRETATION] This creates cross-example learning within a selected subpopulation. Specialists require enough recurrent sample assignments to train well.

## 165. Inductive bias 4: one input may need multiple experts

With $k=4$, the source emphasizes **a mixture of a handful of experts**, rather than only a hard single expert. Their weighted outputs can cooperate on a token.

## 166. Inductive bias 5: load is an optimization variable

Training quality depends not only on the task objective but also on the expert distribution created by the router.

The paper's main conceptual advance includes optimization of **utilization structure**.

## 167. Mechanism primitive A: conditional activation

```text
input → sparse expert subset → weighted result
```

Transferable outside language modeling, but dependent on sufficient data and efficient implementation.

## 168. Mechanism primitive B: learned noisy routing

```text
linear scores + learnable Gaussian noise → Top-k → Softmax
```

Can be adapted to code-conditioned or regime-conditioned routing, while retaining the distinction between *router features* and *expert inputs*.

## 169. Mechanism primitive C: separation of expert choice and expert function

Router chooses which expert to activate. Expert computes the transform.

[INTERPRETATION] This design creates a modular intervention point: change routing while freezing experts, or change experts while keeping routing fixed, to locate improvement sources.

## 170. Mechanism primitive D: importance regularization

\[
\operatorname{CV}(\sum_xG(x))^2.
\]

Regularizes aggregate gate mass.

## 171. Mechanism primitive E: expected-load regularization

\[
\operatorname{CV}(\sum_xP(x,i))^2.
\]

Regularizes approximate number of dispatched samples via differentiable selection probabilities.

## 172. Mechanism primitive F: hierarchical expert selection

```text
input → group gate → within-group gate → expert leaf
```

Useful when one flat gate's branching factor is too large.

## 173. Mechanism primitive G: time-position batching

Aggregate independent applications of a shared feed-forward MoE across time positions to form larger expert batches.

[INTERPRETATION] Conceptually compatible with other tokenized representations, but must respect causality and sequence/task semantics.

## 174. Mechanism primitive H: dispatch/batch aggregation

Gather selected samples across data-parallel replicas into sufficiently large per-expert workloads.

## 175. Mechanism primitive I: empirical separation of capacity and compute

At fixed active compute, vary expert vocabulary size. At fixed-ish total capacity, vary active compute. Compare both axes.

## 176. Mechanism primitive J: expert interpretability by routed examples

Rank inputs by each expert's gate weight and inspect recurrent commonalities. This is a first-pass qualitative diagnostic, not a proof of identified causal mechanisms.

---

# Part X. Connection to the User's Finance Baselines

## 177. The correct bridge to PRISM-VQ

PRISM-VQ uses a discrete structural representation to decide which expert receives a stock. This is **not** the same gate design as the original linear noisy Top-k scores over an ordinary token representation.

The conceptual bridge is:

\[
\text{input-conditioned routing}
\rightarrow
\text{conditional expert computation}.
\]

In PRISM-VQ the conditioning input may be a VQ code; in the 2017 paper it is the MoE layer's continuous input vector.

## 178. PRISM-VQ's use of MoE is not primarily giant-model scaling

[INTERPRETATION] In a small cross-sectional dataset, its expert layer primarily targets *dynamics specialization* and representation flexibility, rather than 1000x capacity growth.

Consequently, the most relevant transferred components are:

- expert-choice mechanisms;
- balancing objectives;
- routing noise and uncertainty;
- shared/specialized expert design;
- expert utilization diagnostics;
- conditional structure tests.

## 179. Cross-paper comparison

| Feature | Jacobs 1991 | Shazeer 2017 | PRISM-VQ 2026 |
|---|---|---|---|
| Central goal | Automatic local task decomposition | Efficient sparse capacity scaling | Structure-dependent dynamic factor exposures |
| Gate input | Example features | Continuous input embedding | Discrete learned stock structure (conceptually) |
| Expert selection | Competitive soft mixture | Noisy sparse Top-k | Code-conditioned sparse selection |
| Loss role | Mixture likelihood/responsibilities | Downstream task + importance/load auxiliary | Stock prediction + expert balancing / other model terms |
| Expert architecture | Small local networks | Large feed-forward MLPs | Lightweight specialist subnetworks |
| Interpretability | Task-region specialization | Token-context specialization | Structural/routing and factor loading interpretation |
| Typical evidence | Vowel discrimination | PPL/BLEU/capacity/throughput | RankIC/portfolios/ablation/routing |

[INTERPRETATION] The PRISM-VQ column refers to the separately archived baseline description; the 2017 source does not mention PRISM-VQ.

## 180. Connection to FactorVQVAE

FactorVQVAE learns discrete factor tokens and models their temporal transitions via autoregressive Transformer, but does not define the original Shazeer sparse expert routing as its core mechanism.

Potential combination (new): condition a token-dynamics expert bank on latent factor-token history.

## 181. Connection to MASTER

MASTER dynamically models cross-stock relations using attention and market-conditioned feature gating. A new sparse-expert design could specialize cross-stock relation transforms rather than only stock-local time dynamics.

[EXTENSION] In this case MoE specialists represent distinct *relation operators*, not the 2017 paper's language-level token experts.

## 182. Connection to MATCC

MATCC combines trend/fluctuation decomposition, RWKV time modeling and cross-stock interaction. An MoE could specialize by:

- short/long trend mixture;
- volatility and market regime;
- stock type;
- cross-stock relationship structure.

[EXTENSION] These are candidate modifications; MATCC itself is not evidence that sparse expert routing improves those tasks.

## 183. Connection to TimeMixer and TimeMixer++

Different temporal scales or resolutions can generate different specialist subproblems. Rather than dense weighted fusion across all scales, an MoE could conditionally activate scale-specific experts.

[CRITIQUE] A sequence length of 8 or 20 may be too short for deep scale hierarchies; do not port large-T hyperparameters blindly.

## 184. Core caution: two meanings of “structure”

- **VQ code structure:** learned cluster/prototype assignment in stock representation space.
- **MoE routing structure:** which computational transformation is activated.

Aligning the two can be productive, but a discrete cluster does not necessarily imply one optimal expert. This must be demonstrated.

## 185. Core caution: three meanings of “expert”

1. **Neural expert:** one trainable subnetwork in an MoE.
2. **Financial prior expert:** an external hand-designed economic signal or factor.
3. **Domain expert:** a human financial analyst or theory source.

The 2017 paper concerns **neural expert subnetworks**, not the 13 JKP financial prior factors in PRISM-VQ.

## 186. Routing conditioned on code vs. market regime

Two different inductive biases:

\[
G_i=g(z_{q,i})\quad\text{stock structural routing},
\]

\[
G_{i,t}=g(m_t)\quad\text{common market-regime routing}.
\]

A mixed gate is possible:

\[
G_{i,t}=g(z_{q,i},h_{i,t},m_t).
\]

[EXTENSION] These are proposed finance formulations, **not** Shazeer 2017 equations.

## 187. Shared vs. routed experts

Original 2017 layer has a set of routed experts; it does not establish a particular permanent shared expert as a universal component.

A new hybrid can take the form:

\[
y_i=E_{shared}(x_i)+\sum_{j\in\mathcal A_i}G_{ij}E_j(x_i).
\]

[EXTENSION] This may stabilize frequently recurring common patterns while reserving routed capacity for conditional deviations.

## 188. Why expert count is especially sensitive on S&P500

A lower ranking signal environment can make individual expert assignment distributions harder to learn. Extra experts can help specialization, but can equally multiply overfitting opportunities.

[EXTENSION] The correct empirical hypothesis is **nonmonotonic** expert count versus generalization, not “S&P500 necessarily needs more experts.”

## 189. Why per-date load matters in stock selection

Stock ranking often processes one day's full universe jointly. If $N_t$ stocks are routed to $k$ of $n$ experts, the expected assignments per expert at that date are roughly:

\[
\frac{N_t k}{n}.
\]

[DERIVATION] Under uniform independent-like routing. For $N_t$ around a few hundred, increasing $n$ very quickly creates low effective per-day batches.

## 190. Why temporal correlation changes sample-size interpretation

[CRITIQUE] If data span thousands of days, the raw number of stock-date samples may be millions, but adjacent days, index membership, common market shocks and feature overlaps create substantial dependence. An expert trained on highly correlated samples does not have the same effective information as one trained on independent text tokens.

## 191. Beware a false interpretation of per-stock routing

A gate that routes stocks based on current cross-sectional embeddings may mainly sort by:

- market capitalization;
- sector;
- recent volatility;
- missing-value pattern;
- some preprocessing artifact.

[EXTENSION] Test whether routing provides incremental return information beyond known characteristics.

## 192. Factor loading attribution

PRISM-like architectures produce explicit dynamic factor exposures. To verify specialization, examine whether experts cause meaningfully different *loadings* and *forecast errors*, not only different activation rates.

## 193. Negative transfer risk

Adding Shazeer-style noise or capacity to PRISM-VQ can degrade:

- seed stability;
- rare-regime estimation;
- expert utilization;
- calibration of output scales;
- rank performance;
- backtest after costs;
- compute/VRAM budget.

Test each axis separately.

## 194. Appropriate hierarchy for a research Agent

Before implementing a huge MoE, ask:

1. What are the hypothesized conditional subproblems?
2. What observable evidence suggests that one dense expert cannot model them jointly?
3. Which feature(s) should determine routing?
4. How many examples can each expert train on?
5. What is the minimal comparison that could falsify the idea?
6. How will specialization be measured independently of just gate occupancy?
7. Will the added capacity beat a dense model of comparable active computation?

---

# Part XI. Implementable Algorithm Sketches

## 195. Minimal Noisy Top-k gate pseudocode

[DERIVATION / implementation guide, not official code]

```python
# x: [B, D]; Wg, Wnoise: [D, N]; k: active experts
scores = x @ Wg
scales = softplus(x @ Wnoise)
noise = standard_normal_like(scores)
H = scores + noise * scales
indices = topk(H, k=k, dim=-1).indices
masked = full_like(H, -inf)
masked.scatter_(dim=-1, index=indices, src=H.gather(-1, indices))
G = softmax(masked, dim=-1)

# Dispatch only selected input rows to each expert E[j]
# Recombine the outputs with their sparse G weights.
```

Note: normal training/inference noise policy, tie handling, empty expert batches and all-to-all communication require explicit implementation decisions.

## 196. Balancing importance calculation

[DERIVATION]

```python
importance = G.sum(dim=0)  # [N]
loss_importance = w_importance * (importance.std() / importance.mean())**2
```

Check standard-deviation convention and epsilon protection when reproducing.

## 197. Smooth expected-load calculation

[DERIVATION]

```python
# Pseudocode: the threshold must exclude expert i.
threshold_i = kth_largest_of_other_scores(H, exclude=i, k=k)
prob_i = standard_normal_cdf((scores_i - threshold_i) / scales_i)
load_i = prob_i.sum_over_batch()
loss_load = w_load * CV(load_vector)**2
```

**Do not** replace this with hard Top-k counts inside an autograd graph and expect the same gradients.

## 198. Why naive load counts are inappropriate

[DERIVATION] If:

```python
load = (G > 0).float().sum(0)
```

then backprop through the comparison itself is nonexistent almost everywhere. The load penalty cannot adjust gate scores by the path Eq. (9) establishes.

## 199. Dispatch pseudocode

```text
for each example b:
    selected = topk_ids[b]
    for e in selected:
        append (b, x[b], G[b,e]) to expert_batch[e]

for expert e with nonempty expert_batch[e]:
    outputs[e] = E_e(expert_batch[e].inputs)

for each example b:
    result[b] = sum(gate_weight * expert_result of routed experts)
```

[INTERPRETATION] Real high-performance implementations replace Python loops with vectorized dispatch, segmented operations or grouped kernels.

## 200. Model parameter and active-compute accounting

For $n$ equal-size expert MLPs:

\[
P_{total\ experts}=nP_E,
\]

but approximately:

\[
C_{active\ experts}=kC_E,
\]

ignoring router, communication, dispatch, and non-expert layers.

[DERIVATION] This is the elementary capacity/computation distinction to preserve in all comparisons.

## 201. Formal expert-occupancy metrics

For $B$ input observations and routing weights $G_{bi}$:

\[
I_i=\sum_bG_{bi},\qquad
C_i=\sum_b\mathbf1(G_{bi}>0).
\]

Normalized occupancy:

\[
q_i=\frac{C_i}{\sum_jC_j}.
\]

[EXTENSION] In addition to the paper's importance/load CV, one may monitor entropy $-\sum_iq_i\log q_i$ and effective expert count $\exp(H(q))$.

## 202. Why entropy is not a sufficient specialization metric

A completely uniform router has maximum entropy but may be incapable of conditional specialization. Conversely, a genuinely rare regime might need a rarely active expert.

[EXTENSION] Pair entropy with conditional generalization evidence.

## 203. Proposed conditional-specialization diagnostic

For label $r$ (e.g., market regime), estimate:

\[
q(j\mid r)=P(\text{route to expert }j\mid r).
\]

Then compare expert routing distributions across regimes and ask whether the differences improve predictive performance on independent test periods.

[EXTENSION] Regime labels must be computed using only information available at inference time if they drive routing.

## 204. Proposed expert counterfactual test

For each expert $j$:

1. Freeze a trained checkpoint.
2. Reallocate or zero expert $j$ while normalizing remaining expert weights consistently.
3. Re-evaluate return predictions, RankIC and backtest.
4. Compare under the same seeds/dates and check which regimes are affected.

[EXTENSION] Carefully control the intervention because naively deleting an expert alters normalization and may introduce distribution shift.

## 205. Proposed routing turnover metric

For stock $i$ across available dates:

\[
T_i=\frac{1}{|\mathcal T_i|-1}
\sum_{t}\mathbf1\{\arg\max_jG_{i,t,j}\ne \arg\max_jG_{i,t-1,j}\}.
\]

[EXTENSION] Compare expert-route turnover to stock-return signal turnover and codebook assignment changes. High turnover is not automatically bad; it may reflect correct regime adaptation.

---
# Part XII. Falsifiable Cross-Sectional Finance Experiments (New Proposals)

> **[EXTENSION] Every experiment E01–E24 below is proposed for the research Agent; none appears in Shazeer et al. (2017).**
>
> All comparisons require identical label horizon, test dates, universe constituents, feature preprocessing, market information, stock eligibility, seed set and execution/cost protocol. A new MoE is only accepted as useful if the intended advantage survives *matched* tests, not merely a selected seed or one market.

## 206. E01 — Expert count at fixed active compute

**Hypothesis:** At fixed $k$, increasing $n$ initially improves RankIC by offering specialist capacity, then saturates or degrades as training data per expert shrink.

**Design:** Compare $n\in\{2,4,8,16\}$ while keeping the active expert width, $k$, optimizer and shared backbone fixed. Contrast against a dense baseline with similar active operations.

**Readouts:** CSI300/S&P500 5-seed IC, RankIC, RankICIR; annual/quarterly stability; actual expert load and parameter count.

**Failure criterion:** No reproducible incremental performance over simpler $n$ or performance drops with rising expert count; fail to claim “capacity gain” if training latency is disproportionate.

## 207. E02 — Active expert count $k$

**Hypothesis:** Too-sparse top-1 routing underutilizes complementary signals; top-2/top-4 may improve generalization, but can also increase interference and cost.

**Design:** Keep $n$ fixed (where feasible); test $k\in\{1,2,4\}$ while logging active FLOPs and latent/expert error correlations.

**Controls:** Normalize parameter/operation differences with a dense small-MLP variant, and avoid attributing changes to the expert count itself.

**Failure criterion:** No RankICIR or portfolio improvement despite extra work.

## 208. E03 — Importance vs. load regularization

**Hypothesis:** Importance-only and Load-only balancing have different effects on routing quality and training stability.

**Design:** Four variants: no auxiliary balancing, importance only, load only, both.

**Readouts:** Two CV metrics, max/mean actual load, dead experts, seed variance, out-of-sample RankIC/Sharpe.

**Failure criterion:** Balance improves while prediction deteriorates or expert functions remain redundant.

## 209. E04 — Balance coefficient sensitivity

**Hypothesis:** Weak balance leads to collapse; excessive balance prevents useful minority specialists.

**Design:** Small validation-only sweep of $\lambda_{balance}$; retain fully held-out test periods.

**Analysis:** Plot ranking performance against CV/entropy and per-regime expert usefulness.

**Failure criterion:** Best gate effectively uniform with no conditional skill, or improvement exists only with test-tuned coefficients.

## 210. E05 — Noisy versus deterministic routing

**Hypothesis:** Training-time noise helps discover more robust partitions and reduces early expert monopoly.

**Design:** Identical router architecture with and without learned Gaussian score noise. Evaluate stochastic training but explicitly set deterministic/controlled inference policy.

**Readouts:** Dead-expert ratio, early gate concentration, convergence, 5-seed variance, test RankIC.

**Failure criterion:** Noise produces unstable assignments or weaker held-out performance with no diversity benefit.

## 211. E06 — Code-only router versus temporal router

**Hypothesis:** PRISM-like stock-code routing is stable, but temporal inputs may capture conditional dynamics the code alone misses.

**Design:**

1. $G=g(z_q)$.
2. $G=g(h_{temp})$.
3. $G=g([z_q;h_{temp}])$.

Match experts, $k$, parameter budget where possible.

**Failure criterion:** Added temporal router complexity fails to improve out-of-sample RankICIR or increases collapse.

## 212. E07 — Market-regime-conditioned router

**Hypothesis:** Shared market conditions should influence which temporal expert processes each stock.

**Design:** Add a past-only market-state embedding $m_t$ to router input. Compare code-only, market-only and code+market routing.

**Critical protocol:** Market regime information must be observable by prediction time; do not create router labels with realized future test returns.

**Failure criterion:** Apparent gains vanish after time-safe regime construction or do not survive multi-year evaluation.

## 213. E08 — Shared plus routed experts

**Hypothesis:** A permanently shared expert captures stable signal while routed experts model departures, improving performance and reducing rare-expert instability.

**Design:**

- one shared expert only;
- routed experts only;
- shared + routed;
- capacity-matched dense control.

**Readouts:** mean RankIC, RankICIR, 5-seed spread, performance in low-sample/high-volatility subgroups.

**Failure criterion:** No gain beyond simply making the dense model larger.

## 214. E09 — Hierarchical market→stock routing

**Hypothesis:** A hierarchical router with market-state group selection followed by code-conditioned stock experts can improve conditional specialization.

**Design:** Compare flat router with two-level router at comparable active expert count and parameter budget.

**Readouts:** regime-specific routing, expert output diversity, runtime, performance.

**Failure criterion:** Hierarchy adds overhead with no coherent conditional gain or collapses primary groups.

## 215. E10 — Regime-specialist ablation

**Hypothesis:** Some experts are particularly useful during identifiable regimes (up/down/high volatility or liquidity stress).

**Design:** Remove/reweight one expert at inference using a fixed trained checkpoint and measure rank performance by held-out market regime.

**Failure criterion:** All experts have interchangeable effects despite diverse routing statistics.

## 216. E11 — Stable expert identity across seeds

**Hypothesis:** Experts learn reproducible functional roles rather than arbitrary seed-dependent permutations.

**Design:** Align experts across five seeds using output behavior, routing co-assignment or matched pairwise correlations; then compare roles.

**Caution:** Expert IDs are permutation-invariant, so comparing expert 0 from seed A to expert 0 from seed B is not meaningful without alignment.

**Failure criterion:** No stable aligned specialization or only one seed produces interpretable experts.

## 217. E12 — Expert specialization beyond company size/industry

**Hypothesis:** Learned experts capture conditional return relationships not explained solely by known observable stock groups.

**Design:** Regress or classify routing using known metadata (size, industry, volatility). Test incremental rank performance within characteristic-matched groups.

**Failure criterion:** Learned “experts” act only as size or sector partitions without incremental predictive benefit.

## 218. E13 — Static router vs. rolling-adapted router

**Hypothesis:** A frozen routing map becomes stale as markets change; conservative online/periodic adaptation may help.

**Design:** Compare static router with strictly causal periodic retraining and/or small router-only updates.

**Controls:** Equal access to training history and nonoverlapping validation/test protocol; account for retraining cost.

**Failure criterion:** Adaptation improves training rank but degrades held-out future dates.

## 219. E14 — Prior-aware router

**Hypothesis:** Historical JKP factor-return priors can condition which experts specialize in the current economic environment.

**Design:** Compare router inputs: code; past JKP priors; code + priors. Do not leak current/future factor returns.

**Readouts:** conditional expert activation and incremental RankIC beyond common priors.

**Failure criterion:** Prior-conditioned router adds no gain beyond just feeding factors to the main prediction head.

## 220. E15 — Expert output diversity regularization

**Hypothesis:** Balance penalties prevent usage collapse but not functional redundancy; explicit output diversity could improve complementary prediction.

**Design:** Original balancing losses versus additional expert-output decorrelation/contrastive penalty. Match compute and avoid encouraging uncorrelated noise.

**Failure criterion:** Diversity rises but performance does not, or specialists predict contradictory spurious patterns.

## 221. E16 — Small cross-section and batch aggregation

**Hypothesis:** Enlarging the training batch of routed stocks across several dates improves expert efficiency and stability.

**Design:** Compare per-date execution, microbatching across chronological dates, and vectorized grouped expert dispatch. Preserve date identities for cross-sectional operations.

**Readouts:** throughput, peak VRAM, expert sample counts, RankIC, temporal contamination checks.

**Failure criterion:** Better hardware efficiency comes with unintended cross-date normalization or worse out-of-sample signal.

## 222. E17 — Per-date balancing vs. long-window balancing

**Hypothesis:** Balancing every cross-section may be too restrictive; balancing over a rolling batch of training dates permits legitimate regime-specific usage while avoiding long-term collapse.

**Design:** Keep task loss fixed. Apply importance/load regularizers to one date or a multi-date minibatch/epoch window.

**Failure criterion:** Larger-window balance causes expert monopoly or slow adaptation; per-date balance improves predictive stability despite restriction.

## 223. E18 — Strictly balanced vs. soft-balanced routing

**Hypothesis:** Exact per-expert batch quotas improve throughput, but may force assignments that are economically inappropriate for rare structures.

**Design:** Implement a safe batchwise top-$m$ variant with explicit fallback for inputs assigned to no experts; compare with noisy Top-k + soft losses.

**Readouts:** routing gaps, memory, throughput, regime performance.

**Failure criterion:** Exact quotas produce sample-dependent instability, harmful rerouting, or no meaningful system improvement.

## 224. E19 — Adaptive top-k by uncertainty

**Hypothesis:** Some stock states require multiple specialists while easy examples can use one.

**Design:** Route adaptively according to entropy/confidence with a controlled average expert budget; compare fixed top-1/top-2/top-4 at similar average compute.

**Failure criterion:** Additional routing complexity does not improve uncertainty-sliced ranking or causes unpredictable latency.

## 225. E20 — Relation-expert routing in MASTER/MATCC

**Hypothesis:** Different market states require different cross-stock relation operators.

**Design:** Replace one homogeneous stock-attention relation transform with a small expert bank; route from market state or stock-time representation.

**Readouts:** regime-sliced RankIC, relation sparsity/stability, memory/time.

**Failure criterion:** Extra relation experts overfit without stable cross-stock evidence or degrade S&P500.

## 226. E21 — Rank-aware expert supervision

**Hypothesis:** MSE-only experts may not specialize in the part of return behavior that matters for cross-sectional ranking.

**Design:** Compare MSE+balance against MSE+ranking surrogate+balance at fixed routing and expert capacity.

**Critical control:** Same label horizon/normalization, ranking loss weight tuned on validation only.

**Failure criterion:** RankIC gains do not survive cross-market tests or significantly harm cost-aware portfolio metrics.

## 227. E22 — Sparse expert routing vs. dense blending

**Hypothesis:** True sparse selection has a useful inductive bias and compute advantage beyond a soft dense ensemble.

**Design:** Dense $n$-expert blending, sparse Top-k and one dense equivalent-parameter MLP; compare under both equal-total-params and equal-active-compute conditions.

**Failure criterion:** Sparse model has no meaningful quality/efficiency advantage at fair compute.

## 228. E23 — Specialist exposure to rare market regimes

**Hypothesis:** Experts selected for rare high-volatility regimes are undertrained and unstable.

**Design:** Tabulate expert utilization by regime frequency and out-of-sample conditional error; test shared anchor, mild replay/resampling or regularization within *training only*.

**Failure criterion:** Any apparent rare-regime gain fails on an independent later regime, or creates severe ordinary-regime degradation.

## 229. E24 — Structural code perturbation vs. routing uncertainty

**Hypothesis:** A hard VQ code-conditioned router can abruptly change expert selection near code boundaries; noisy/soft routing may smooth prediction sensitivity.

**Design:** Perturb latent stock representation within economically plausible ranges, compare route switches, score stability, ICIR and portfolio turnover across hard, noisy and soft code-routing variants.

**Failure criterion:** Soft/noisy routing reduces score stability or loses predictive advantage despite smoother gates.

---

# Part XIII. Practical Experiment Priority for a Stock-Research Agent

## 230. The most economical initial test

[EXTENSION] Start with a small **parameter-controlled 2×2** comparison inside the already validated local baseline infrastructure:

| Variant | Shared expert | Sparse routed expert |
|---|---|---|
| A | No | No (single dense control) |
| B | No | Yes |
| C | Yes | No |
| D | Yes | Yes |

Keep the stock features, target, training data, codebook and backtest identical.

This first test answers whether shared+routed computation is useful **at all** before exploring elaborate regime hierarchies.

## 231. Follow-up gating comparison

If the expert bank matters, freeze or match its architecture while varying:

- code-only routing;
- temporal-only routing;
- code+temporal routing;
- code+temporal+market routing.

Separate any benefits of **extra router information** from benefits of a larger parameter count.

## 232. Follow-up balancing comparison

Test no/importance/load/both penalties. Record actual counts and predictive results. Do **not** optimize only toward visually uniform expert usage.

## 233. One-change-at-a-time ablation rule

[EXTENSION] The Agent should not simultaneously change:

- router inputs;
- expert number;
- codebook;
- prediction loss;
- training data protocol.

Otherwise a positive experiment cannot identify which mechanism helped.

## 234. Stock-prediction evaluation contract

At minimum:

- CSI300 and S&P500;
- five fixed seeds;
- IC / ICIR / RankIC / RankICIR;
- same-date aligned predictions and targets;
- annual/quarterly/monthly breakdown;
- code/expert utilization and stability;
- identical cost-aware backtest when comparing investments;
- run time and GPU memory.

[EXTENSION] These criteria are tailored for a modern Qlib research system, not reported in the original 2017 paper.

## 235. Prevent information leakage

- Router must not see future realized returns or future-informed regime labels.
- Distinguish training-only posterior supervision from inference-time inputs.
- Do not use test-period performance to choose expert count or gate-loss weights.
- Align all baselines to the same label formula and prediction horizon before comparing IC.
- Portfolio cost and execution conventions must be identical.
- Each seed's trained model must be evaluated on the same held-out dates, with actual seed variance reported.

## 236. Conditional regime analysis

For each checkpoint and market:

1. compute daily RankIC/IC;
2. group dates by **ex-ante** observable market conditions;
3. inspect group-level mean and variance;
4. compare with no-MoE control;
5. pair with expert utilization and output diagnostics;
6. report whether gains persist in an independent future subperiod.

## 237. How to reason about negative results

Examples:

- **Noisy routing helps load but not RankIC:** optimization effect without demonstrated predictive specialization.
- **More experts hurt both markets:** possible sample fragmentation or overfitting; do not simply increase capacity again.
- **More experts help CSI but not SP:** market-specific utility; examine sector/volatility distributions before inferring causal reasons.
- **Shared expert helps stability but not mean RankIC:** test whether reducing large-loss dates justifies it in risk-adjusted outcomes.
- **RankIC rises but backtest worsens:** investigate score concentration, turnover, tails, costs and label/portfolio mismatch.

## 238. Stopping rule for unsuccessful research branches

[EXTENSION] Before running a new experiment, explicitly state a failure condition. Stop a branch if:

- it repeatedly underperforms a simpler controlled alternative;
- apparent gains are seed-specific;
- it increases runtime/memory without corresponding predictive improvement;
- it worsens the more challenging market substantially;
- result depends on violating data-timing or test-selection protocol.

Do not continue adding complexity merely to preserve an attractive narrative.

## 239. Minimum viable paper-worthy claim

For an MoE innovation to carry scientific weight, seek evidence for **both**:

1. **Predictive improvement:** consistent held-out ranking and/or economically robust portfolio improvement.
2. **Mechanistic explanation:** an ablation and diagnostic connecting the gain to expert specialization, routing stability, common-vs-specific decomposition, or other stated mechanism.

A bigger gate network with better one-seed RankIC is insufficient.

---

# Part XIV. Paper-Writing Patterns Extracted from Shazeer et al. (2017)

## 240. Lead with an engineering contradiction

The paper starts with the tension:

\[
\text{high model capacity desired}
\quad\text{but}\quad
\text{high dense-compute cost prohibitive}.
\]

Then defines conditional computation as a route around the contradiction.

## 241. Identify why previous solutions failed

Before presenting its layer, it lists *specific practical roadblocks*:

- branching;
- small expert batches;
- communication;
- load imbalance;
- insufficient data.

This strengthens the necessity of the method beyond “we add experts.”

## 242. Match innovation to failure mode

| Failure mode | Proposed solution |
|---|---|
| Dense capacity tied to compute | Sparse expert activation |
| Gate cannot easily choose conditional routes | Noisy Top-k routing |
| Expert monopoly | Importance and load balancing |
| Tiny expert batches | Mixed data/model parallelism, time batching |
| Very large flat gate | Hierarchical MoE |
| Parameter memory pressure | Activation recomputation and optimizer-state compression |

## 243. Separate theoretical and systems claims

The paper does not stop after describing Top-k. It asks whether conditional computation can actually run efficiently on the target GPU clusters.

A modern finance paper should similarly distinguish:

- architectural hypothesis;
- prediction result;
- computational implementation;
- deployability and trading cost.

## 244. Decompose resource scaling in experiments

Source Figures 2 and 3 ask different questions:

- Does larger **available capacity** help at fixed computation?
- Does larger **active computation** help at fixed high capacity?
- Does larger **dataset** allow more capacity benefit?

This 3-axis design is worth copying, with much smaller models and controlled financial data.

## 245. Report engineering baselines as carefully as model scores

The paper logs ops per timestep, GPU throughput and training time as explicit primary evidence. A financial MoE paper claiming practical effectiveness should not omit these.

## 246. Make negative experiments visible

The 131,072-expert degradation and English→Korean performance drop are important because they identify failure conditions and delimit the original claims.

## 247. The “conditional specialization” writing template

```text
Observed limitation:
One shared mapping underfits heterogeneous subproblems.

Proposed mechanism:
Input-conditioned selection among specialist functions.

Optimization challenge:
Hard routing collapses or starves experts.

Solution:
Trainable sparse gate + utilization regularization.

Experiment:
Fixed compute, vary expert capacity and routing conditions.

Evidence:
Main task gains + routing diversity + subgroup diagnosis.

Caveat:
Conditional specialization requires enough data per expert.
```

## 248. Avoid overclaiming semantic interpretability

Use language such as:

> “Routing statistics and routed-example inspection suggest conditional specialization.”

rather than:

> “Expert 4 is definitely the momentum expert and Expert 5 is the volatility expert.”

The latter needs independent financial evidence.

---

# Part XV. Source Integrity and Reproducibility Checklist

## 249. Source identity checks

- [x] Uploaded PDF is 19 pages.
- [x] Title and authors match arXiv:1701.06538v1.
- [x] Source status reads “Under review as a conference paper at ICLR 2017.”
- [x] Main numbered Eqs. (1)–(7) were included.
- [x] Appendix Eqs. (8)–(22) were included or carefully summarized.
- [x] Original tables 1–9 have a source-location map; numeric tables 2–8 are transcribed where material.
- [x] Figure 1–4 roles are retained.
- [x] Original experiments are separated from financial research proposals.

## 250. Formulas that must be reproduced for full comprehension

\[
y=\sum_iG(x)_iE_i(x)
\]

\[
G(x)=\operatorname{Softmax}(\operatorname{KeepTopK}(xW_g+\epsilon\odot\operatorname{Softplus}(xW_{noise}),k))
\]

\[
I(X)=\sum_{x\in X}G(x),\qquad
\mathcal L_{importance}=w_I\operatorname{CV}(I(X))^2
\]

\[
P(x,i)=\Phi\left(\frac{s_i-\theta_i}{\sigma_i}\right),\quad
\operatorname{Load}_i(X)=\sum_xP(x,i),\quad
\mathcal L_{load}=w_L\operatorname{CV}(\operatorname{Load}(X))^2
\]

\[
y_H=\sum_{i,j}G_{primary}(x)_iG_i(x)_jE_{ij}(x).
\]

## 251. Source-grounded numerical anchors

If any of these are different in another summary, verify the original source:

| Anchor | Correct source value |
|---|---|
| 1B words actually in benchmark | approximately 829M |
| Low-compute expert selections | 4 active per input |
| 1B high-budget perplexity, 10 epochs | 28.0 |
| 1B best published baseline, 10 epochs / 100 epochs | 34.7 / 30.6 |
| 100B best expert count | 65,536 |
| 100B best PPL | 28.9 |
| 100B 131,072-expert PPL | 29.2 |
| Largest approximate parameter scale | 137.7B |
| No balance loss PPL / max-to-mean load | 39.8 / 17.80 |
| Two 0.1 balance losses PPL / max-to-mean load | 35.6 / 1.14 |
| WMT En→Fr BLEU longer MoE | 40.56 |
| WMT En→De MoE BLEU | 26.03 |
| Multilingual loss: En→Ko | MoE 16.62 vs GNMT multi 18.41 |

## 252. Reproduction prerequisites not fully supplied by the PDF

The PDF is unusually detailed for a systems paper, but exact reproduction of the historical distributed implementation still requires:

- original framework version and code;
- distributed device mapping and communication scheduling;
- exact per-device memory optimizations;
- random seeds and detailed run-to-run variation;
- precision modes and kernel implementations;
- all preprocessing, tokenization and evaluation scripts;
- exact numerical handling of Top-k ties, empty batches, unassigned examples in strict-balance mode;
- unspecified inference noise behavior for the standard gate;
- test harness and checkpoint selection policy.

[VERIFY] Do not invent these details merely because a modern standard implementation would choose one convention.

## 253. Common misreadings to reject

- “All 137B parameters run for each word.” **False** — most experts are inactive.
- “137B is the best model.” **False** on 100B test perplexity.
- “Load and Importance losses are the same.” **False**.
- “Top-k is smooth everywhere.” **False**.
- “The paper proves top-1 routing works.” **False**.
- “Uniform routing proves semantic specialization.” **False**.
- “The paper uses only noisy top-k in every translation experiment.” **False** (strictly balanced gate appears in some).
- “MoE started in 2017.” **False** (cites Jacobs 1991).
- “The experiments demonstrate finance results.” **False**.
- “Sparse experts eliminate deployment overhead.” **False**.

## 254. Differentiating source facts from new derivations

This document uses the following tags:

- `[SOURCE]`: stated or shown directly in supplied PDF.
- `[DERIVATION]`: mathematically inferred from source definitions.
- `[INTERPRETATION]`: model-level explanatory reading, not a new source fact.
- `[CRITIQUE]`: evidence limitations or alternative explanations.
- `[EXTENSION]`: new research proposal, especially for finance.
- `[VERIFY]`: source/protocol ambiguity requiring original code or further inspection.

## 255. Suggested retrieval chunks for the Agent

- **Need original MoE idea?** Sections 003–010.
- **Need Noisy Top-k implementation?** Sections 011–030 and 195–200.
- **Need balancing?** Sections 031–053 and 196–198.
- **Need hardware/scalability?** Sections 054–070, 075–097.
- **Need original empirical evidence?** Sections 071–120.
- **Need hierarchical MoE?** Sections 122–127.
- **Need strict batchwise routing?** Sections 131–139.
- **Need finance innovation ideas?** Sections 177–239.
- **Need exact caution on negative results?** Sections 093, 110, 145–161.

## 256. Source location map (PDF physical pages)

| Pages | Original material |
|---|---|
| 1 | Abstract, motivation, conditional computation introduction |
| 2 | Figure 1, practical challenges, MoE as a layer |
| 3 | Prior MoE lineage, expert weighted sum Eq. (1) |
| 4 | Dense gate and Noisy Top-k Eqs. (2)–(5), shrinking-batch motivation |
| 5 | Parallelism, communication, importance Eq. (6)–(7) |
| 6 | Load-balancing motivation; 1B language model and Figure 2 |
| 7 | High-capacity Table 1, efficiency, 100B Figure 3 |
| 8 | 100B result, translation Tables 2–4 |
| 9 | Multilingual Table 5, conclusion |
| 10–12 | References |
| 13 | Appendix A, load estimator Eqs. (8)–(11), Table 6 |
| 14 | Appendix B and beginning of C, hierarchy Eqs. (12)–(14) |
| 15 | Appendix C, 1B architecture/training/Table 7 |
| 16 | Appendix C.2, D and Table 8 |
| 17 | Appendix D/E, translation architecture/training details |
| 18 | Appendix E Figure 4, specialization Table 9, Appendix F intro |
| 19 | Strict balancing Eqs. (16)–(20), attention Eqs. (21)–(22) |

## 257. Agent comprehension questions

A research Agent has read this paper correctly if it can answer:

1. Why is a weighted dense mixture insufficient to reduce expert FLOPs?
2. What exactly does the Gaussian noise do in Noisy Top-k?
3. Why is $k=1$ not equivalent to $k>1$ for ordinary active-gate Softmax gradients?
4. Why do Importance and Load need different losses?
5. How does the Gaussian-CDF surrogate derive from a top-$k$ threshold excluding one expert?
6. Why is expert load an issue even when the total weight is balanced?
7. What does mixed data/model parallelism solve?
8. How does hierarchical routing combine primary and secondary gates?
9. Why is the 65,536-vs-131,072 expert result important?
10. Which machine-translation experiments used an alternative batchwise gate?
11. What evidence supports versus fails to prove semantic specialization?
12. What needs to change before the mechanism makes sense on a stock-date cross-section?

## 258. Most important mathematical lesson

**The routing computation must remain trainable despite sparse/discrete assignment.**

The 2017 solution couples:

\[
\text{Noisy Top-k}
+\text{active-gate backprop}
+\text{smooth expected load}
+\text{balancing regularization}.
\]

No single component fully explains why the overall training and systems approach works.

## 259. Most important empirical lesson

**Available capacity can rise much faster than active computation, but gains depend on data, balance and systems efficiency, and need not be monotonic in expert count.**

The 100B experiment directly documents a capacity saturation/reversal and a throughput penalty at its extreme scale.

## 260. Most important lesson for cross-sectional finance

**Expert specialization must be demonstrated rather than presumed.**

A useful financial MoE should show an independently repeatable relation among:

- inputs/market conditions;
- expert selection;
- distinct expert functions;
- improved out-of-sample ranking and investment behavior.

Expert count, routing entropy and visually distinct gate heatmaps are supporting diagnostics, not substitutes for predictive validation.

---

# Compact Retrieval Summary

Shazeer et al. (2017) introduce the modern Sparsely-Gated Mixture-of-Experts layer as a scalable conditional-computation component. The layer combines outputs of independent feed-forward experts via a sparse gate $y=\sum_iG(x)_iE_i(x)$. The proposed Noisy Top-k gate adds learnable Gaussian noise to input-dependent linear routing logits, retains the top $k$ logits, masks others to $-\infty$, and Softmax-normalizes the active weights. This allows only selected experts to execute, increasing total available parameters without proportional active expert FLOPs. To avoid routing collapse and distributed-load bottlenecks, the paper separately penalizes the coefficient of variation of total gate importance and smooth expected expert load. The expected-load estimator uses a Gaussian CDF applied to the logit's margin over the $k$th-highest competing logit, derived in Appendix A. Hardware efficiency further requires batching expert inputs across data-parallel devices and time positions; hierarchical MoE reduces branching factor at very high expert counts. In a 1B-word language model, a ~4B-expert-parameter model reaches 28.0 PPL after ten epochs at larger compute, and compute-matched MoEs improve as expert capacity grows. On 100B words, 65,536 experts (~68.9B total parameters) obtain 28.9 PPL, but 131,072 experts (~137.7B) worsen to 29.2, demonstrating nonmonotonic returns. Translation models obtain strong WMT results, though multilingual English→Korean degrades. Some single-pair translation experiments use a separate Strictly Balanced Gating scheme rather than the standard Noisy Top-k. This is a large-text/distributed-compute paper, not a financial backtest. For CSI300/S&P500 research, its most reusable mechanisms are noisy sparse routing, importance-versus-load balancing, conditional specialization diagnostics, compute-controlled expert-count experiments and scalable batch dispatch; stock market applicability must be established by new leakage-safe, five-seed and cross-market experiments.
