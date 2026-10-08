---
paper_id: Fama_French_1992_CrossSection
title: "The Cross-Section of Expected Stock Returns"
authors:
  - Eugene F. Fama
  - Kenneth R. French
venue: "The Journal of Finance"
year: 1992
volume: XLVII
issue: 2
paper_type: finance_foundation
subtype:
  - empirical_asset_pricing
  - cross_sectional_returns
domain:
  - asset_pricing
  - cross_sectional_expected_returns
  - anomalies
  - firm_characteristics
methodology:
  - portfolio_sorts
  - fama_macbeth_regression
  - robustness_analysis
sample_main: "NYSE, AMEX, NASDAQ; July 1963-December 1990"
sample_appendix: "NYSE; 1941-1990"
core_variables:
  - market_beta
  - size_ME
  - book_to_market_BE_ME
  - earnings_price_EP
  - market_leverage_A_ME
  - book_leverage_A_BE
empirical_study: true
core_findings:
  - market_beta_has_little_explanatory_power_once_independent_of_size
  - size_negatively_related_to_average_return
  - book_to_market_positively_related_to_average_return
  - size_and_book_to_market_absorb_much_of_EP_and_leverage_effects
relevance_to_agent:
  financial_theory: high
  cross_sectional_prediction: high
  empirical_design: very_high
  deep_learning_architecture: low
  evaluation_methodology: very_high
priority: foundational
---

# The Cross-Section of Expected Stock Returns

## 0. Document Purpose

This document is an Agent-oriented reconstruction of Eugene F. Fama and Kenneth R. French's 1992 paper, *The Cross-Section of Expected Stock Returns*.

The paper is fundamentally different from Sharpe (1964). Sharpe is a theoretical equilibrium paper; Fama and French (1992) is an empirical cross-sectional asset-pricing paper. Therefore, the most important information for a research Agent is not only the paper's conclusions, but also:

1. how the sample is constructed;
2. how information is time-aligned to avoid using accounting data before it is public;
3. how market beta is estimated;
4. how the authors deliberately create beta variation that is not merely a proxy for firm size;
5. how portfolio sorts and Fama-MacBeth regressions complement one another;
6. how competing explanatory variables are tested jointly;
7. how robustness tests are used to attack alternative explanations;
8. how the authors separate empirical regularities from rational and behavioral interpretations;
9. which limitations and open questions the authors explicitly acknowledge.

This paper should be read by an Agent as both:

- a foundational result about the cross-section of stock returns; and
- a methodological template for identifying whether an apparent predictor has incremental explanatory power after controlling for correlated characteristics.

---

# 1. Metadata

- **Title:** The Cross-Section of Expected Stock Returns
- **Authors:** Eugene F. Fama and Kenneth R. French
- **Journal:** The Journal of Finance
- **Volume:** XLVII
- **Issue:** No. 2
- **Date:** June 1992
- **Journal pages:** 427–465
- **Main return-test period:** July 1963 to December 1990
- **Appendix period:** 1941 to 1990 for NYSE stocks
- **Main data sources:** CRSP and merged COMPUSTAT annual industrial files
- **Paper type:** Empirical asset pricing / cross-sectional return study
- **Primary empirical tools:** portfolio sorts and month-by-month Fama-MacBeth cross-sectional regressions

---

# 2. Research Problem

## 2.1 Central Question

The Sharpe-Lintner-Black (SLB) asset-pricing model predicts that:

1. expected stock returns are positively and linearly related to market beta;
2. market beta is sufficient to describe the cross-section of expected returns.

However, prior empirical work had documented several variables that appeared to predict average stock returns beyond beta:

- firm size / market equity \(ME\);
- leverage;
- book-to-market equity \(BE/ME\);
- earnings-price ratio \(E/P\).

Fama and French ask:

> When market beta, size, earnings-price, leverage, and book-to-market equity are considered jointly, which variables actually retain independent explanatory power for the cross-section of average stock returns?

---

## 2.2 Why This Is a Harder Question Than a Univariate Anomaly Test

The candidate variables are correlated with one another.

In particular:

- size and beta are strongly related in simple size-sorted portfolios;
- \(E/P\), leverage, and \(BE/ME\) are all related to stock price and can contain overlapping information;
- a variable may look important by itself only because it proxies for another characteristic.

Therefore, the paper's real methodological problem is:

> **How do we create enough independent variation in correlated characteristics to separate their cross-sectional effects?**

This problem is central to the paper.

---

# 3. Relation to the Sharpe-Lintner-Black Model

The paper begins with the SLB model because its key empirical target is the model's central prediction.

Under the SLB framework:

\[
E[R_i]
\]

should be a positive linear function of the asset's market beta:

\[
\beta_i
=
\frac{\operatorname{Cov}(R_i,R_M)}
{\operatorname{Var}(R_M)}.
\]

In words:

> stocks with larger sensitivity to the market portfolio should have larger expected returns.

The stronger claim is that beta should be sufficient to explain the cross-section of expected returns.

Fama and French test this claim against several observed return regularities.

---

# 4. Prior Empirical Challenges Motivating the Paper

The paper organizes prior evidence into several empirical contradictions or extensions to the SLB model.

## 4.1 Size

Banz (1981) finds that market equity:

\[
ME = \text{stock price} \times \text{shares outstanding}
\]

adds explanatory power beyond beta.

Empirical pattern:

- small stocks tend to have average returns that are too high relative to their beta;
- large stocks tend to have average returns that are too low.

---

## 4.2 Leverage

Bhandari (1988) documents a positive relation between leverage and average return.

This is problematic for a strict single-beta model because, if leverage creates risk, that risk should already be captured by market beta.

---

## 4.3 Book-to-Market Equity

Earlier work finds that average return is positively associated with:

\[
\frac{BE}{ME},
\]

where:

- \(BE\): book value of common equity;
- \(ME\): market value of equity.

High \(BE/ME\) means book value is high relative to market valuation.

---

## 4.4 Earnings-Price Ratio

Basu and other prior work finds that average return is related to:

\[
\frac{E}{P},
\]

where \(E\) is earnings and \(P\) represents market price/equity value.

Ball's interpretation, emphasized in this paper, is that \(E/P\) may act as a catch-all proxy for omitted risk factors:

\[
\text{high expected return}
\rightarrow
\text{low price relative to earnings}
\rightarrow
\text{high } E/P.
\]

---

# 5. Core Hypothesis about Redundancy

Fama and French note that size, \(E/P\), leverage, and \(BE/ME\) can all be understood as variables that scale or otherwise use market price.

Thus several may be different ways of extracting related information from stock prices.

The paper therefore expects that some of these variables may be redundant once tested jointly.

This is a major conceptual contribution:

> The question is not whether each anomaly works individually, but which small set of variables spans the return variation associated with the larger set.

---

# 6. Data

## 6.1 Main Sample

The main sample uses nonfinancial firms in the intersection of:

1. CRSP NYSE, AMEX, and NASDAQ return files;
2. merged COMPUSTAT annual industrial files.

### Exchange coverage

- NYSE and AMEX are present through the earlier part of the sample.
- NASDAQ returns enter the CRSP coverage in 1973.

### Main accounting-data period

COMPUSTAT accounting data are used for:

\[
1962\text{–}1989.
\]

The resulting main return-test period is:

\[
\text{July 1963 to December 1990}.
\]

---

## 6.2 Why Financial Firms Are Excluded

Financial firms are excluded because high leverage is structurally normal for them.

The authors argue that leverage does not have the same interpretation for:

- a bank or financial institution;
- a nonfinancial firm in which high leverage may signal distress.

Thus the leverage variables would not be economically comparable across those groups.

---

## 6.3 Why the Accounting Sample Begins in 1962

Two reasons are given.

### Availability

Book equity is generally unavailable in COMPUSTAT before 1962.

### Selection bias

More importantly, pre-1962 COMPUSTAT data are biased toward:

- large;
- historically successful firms.

The authors therefore avoid using earlier accounting data.

This is an important research-design decision:

> a longer sample is not automatically better if the earlier data have severe selection bias.

---

# 7. Time Alignment and Information Availability

This is one of the most important methodological parts of the paper.

The authors explicitly avoid explaining returns using accounting information that may not yet have been public.

For fiscal yearends in calendar year \(t-1\):

- accounting information is matched to stock returns from July of year \(t\) through June of year \(t+1\).

Thus:

\[
\text{Accounting data in } t-1
\rightarrow
\text{Return tests beginning July } t.
\]

This creates a minimum gap of roughly six months.

---

## 7.1 Why Six Months?

Although firms are formally required to file 10-K reports within 90 days, the paper notes that many do not comply exactly and many December-yearend firms filing on March 31 do not have the information public until April.

The authors therefore choose a conservative timing convention.

### Agent lesson

The paper treats **information availability** as part of the empirical design, not as an implementation detail.

For modern predictive research:

> feature timestamping must reflect when information could realistically have been known, not merely the accounting period to which the observation refers.

---

# 8. Market Equity Timing

Different market-equity measurements are used for different purposes.

## 8.1 December Market Equity

Market equity at the end of December of year \(t-1\) is used to compute accounting-price ratios such as:

\[
BE/ME,
\]

\[
A/ME,
\]

and:

\[
E/P.
\]

---

## 8.2 June Market Equity

Market equity in June of year \(t\) is used as the firm's size measure.

Thus:

\[
\text{Size}_{i,t} = \ln(ME_{i,\text{June }t}).
\]

---

# 9. Fiscal-Year Alignment Caveat

Using December market equity in ratios creates some mismatch for firms whose fiscal year does not end in December.

The authors explicitly discuss this issue.

They report robustness checks using:

- fiscal-yearend market equity instead of December market equity;
- only firms with December fiscal yearends.

The main return-test inferences are similar.

This is a good example of:

> identify a measurement concern, test an alternative implementation, and report whether the conclusion changes.

---

# 10. Firm Inclusion Requirements

For a firm to enter the July \(t\) return tests, the paper requires relevant CRSP and COMPUSTAT information.

Important requirements include:

- a CRSP stock price for December of \(t-1\);
- a CRSP stock price for June \(t\);
- at least 24 monthly returns out of the prior 60 months for pre-ranking beta estimation;
- accounting data for:
  - total book assets \(A\);
  - book equity \(BE\);
  - earnings \(E\).

---

# 11. Variable Definitions

## 11.1 Market Equity / Size

\[
ME
=
\text{price}\times\text{shares outstanding}.
\]

The regression variable is typically:

\[
\ln(ME).
\]

A negative slope means smaller firms have larger average returns.

---

## 11.2 Book Equity

Book equity \(BE\) is defined using common equity plus balance-sheet deferred taxes according to the paper's accounting convention.

---

## 11.3 Book-to-Market Equity

\[
\frac{BE}{ME}.
\]

The regression variable is:

\[
\ln(BE/ME).
\]

High \(BE/ME\):

- low market price relative to book value;
- potentially weak or distressed firm prospects.

---

## 11.4 Total Book Assets

\[
A = \text{total book assets}.
\]

---

## 11.5 Market Leverage

The paper uses:

\[
\frac{A}{ME}
\]

and in the regressions:

\[
\ln(A/ME).
\]

The authors interpret this as a measure of **market leverage**.

---

## 11.6 Book Leverage

The paper uses:

\[
\frac{A}{BE}
\]

and:

\[
\ln(A/BE).
\]

The authors interpret this as **book leverage**.

---

## 11.7 Earnings-Price Ratio

For firms with positive earnings:

\[
E(+)/P
\]

captures positive earnings relative to market equity.

For firms with negative earnings, \(E(+)/P\) is set to zero and a separate dummy is used:

\[
D_{E/P}
=
\begin{cases}
1, & E<0\\
0, & E\ge 0
\end{cases}.
\]

This prevents negative current earnings from being mechanically interpreted in the same way as positive earnings-price ratios.

---

# 12. Why Negative Earnings Are Treated Separately

The paper argues that current positive earnings can plausibly proxy for expected future earnings embedded in stock price.

When earnings are negative, however, the same interpretation becomes problematic.

Therefore:

- positive \(E/P\) receives a continuous variable;
- negative earnings receive a dummy.

This is a useful feature-engineering lesson:

> A variable may need a different representation in a qualitatively different economic regime rather than being treated as one continuous scale.

---

# 13. Fama-MacBeth Regression Framework

The empirical asset-pricing tests use the Fama-MacBeth cross-sectional regression approach.

Each month \(t\), the cross-section of stock returns is regressed on candidate characteristics.

A generic specification is:

\[
R_{i,t}
=
a_t
+
b_{1,t}\beta_i
+
b_{2,t}\ln(ME_i)
+
b_{3,t}\ln(BE_i/ME_i)
+
b_{4,t}\ln(A_i/ME_i)
+
b_{5,t}\ln(A_i/BE_i)
+
b_{6,t}D_{E/P,i}
+
b_{7,t}E_i(+)/P_i
+
\varepsilon_{i,t}.
\]

Different regressions use different subsets of these explanatory variables.

The paper then computes the time-series average of each monthly slope:

\[
\bar b_k
=
\frac{1}{T}
\sum_{t=1}^{T} b_{k,t}.
\]

The reported t-statistic is the average slope divided by its time-series standard error.

Interpretation:

> A variable has an average premium if its monthly cross-sectional slope has a reliably nonzero time-series mean.

---

# 14. Why Individual Stocks Are Used in the FM Regressions

For firm characteristics such as:

- size;
- \(E/P\);
- leverage;
- \(BE/ME\);

the variables are directly measured for individual firms.

The authors do not want to lose this information by aggregating firms into portfolios before running the cross-sectional regressions.

However, beta is estimated more noisily for individual stocks.

Their solution is hybrid:

> estimate beta at the portfolio level for precision, then assign the portfolio beta to individual stocks.

This is an important methodological compromise between:

- measurement precision;
- cross-sectional information granularity.

---

# 15. The Central Identification Problem: Beta vs. Size

Prior studies often use size portfolios.

But in size portfolios, size and beta are extremely highly correlated.

The paper notes a correlation around:

\[
-0.988
\]

in related size-portfolio evidence.

Therefore, if small portfolios have:

- high beta;
- high average return;

one cannot determine whether the premium is associated with:

- beta;
- size;
- or both.

The paper's main empirical innovation is to deliberately create beta variation **within size groups**.

---

# 16. Construction of 100 Size-Beta Portfolios

## 16.1 First Sort: Size

In June each year:

1. all NYSE stocks on CRSP are ranked by \(ME\);
2. NYSE size decile breakpoints are determined;
3. eligible NYSE, AMEX, and NASDAQ stocks are allocated into 10 size portfolios using those NYSE breakpoints.

### Why NYSE breakpoints?

After NASDAQ is introduced, there are many small NASDAQ stocks.

If all exchanges set breakpoints, the size bins could become heavily concentrated among very small firms.

NYSE breakpoints provide a more stable reference.

---

## 16.2 Second Sort: Pre-Ranking Beta

Each size decile is subdivided into 10 beta portfolios.

Thus there are:

\[
10\times 10 = 100
\]

size-beta portfolios.

Pre-ranking beta is estimated using:

\[
24\text{ to }60
\]

monthly returns, depending on availability, from the five years before July of year \(t\).

The beta breakpoints are again based on eligible NYSE stocks.

---

## 16.3 Holding Period

After portfolios are formed in June:

- equal-weighted monthly returns are calculated from July \(t\) to June \(t+1\).

The process is repeated annually.

---

# 17. Post-Ranking Beta Estimation

For each of the 100 size-beta portfolios, the authors use post-ranking returns for the full main sample:

\[
\text{July 1963 to December 1990}
\]

or:

\[
330 \text{ months}.
\]

The market proxy is primarily the CRSP value-weighted portfolio of:

- NYSE;
- AMEX;
- NASDAQ after the latter enters the data.

---

# 18. Nonsynchronous-Trading Adjustment

The paper estimates beta as the **sum of the slopes** from a regression on:

- the current month's market return;
- the prior month's market return.

Schematically:

\[
R_{p,t}
=
\alpha_p
+
\beta_{p,0}R_{M,t}
+
\beta_{p,-1}R_{M,t-1}
+
u_{p,t},
\]

and:

\[
\beta_p^{\text{sum}}
=
\beta_{p,0}
+
\beta_{p,-1}.
\]

The purpose is to adjust for nonsynchronous trading.

The authors report that additional leads/lags and Fowler-Rorke corrections produce little change to their conclusions.

---

# 19. Time Variation in Beta

The use of full-period post-ranking beta is motivated by an assumption that variation in true portfolio betas is approximately proportional.

The paper states the condition:

\[
\beta_{jt}-\beta_j
=
k_t(\beta_j-\bar\beta),
\tag{1}
\]

where:

- \(\beta_{jt}\): true beta for portfolio \(j\) at time \(t\);
- \(\beta_j\): time-series mean beta of portfolio \(j\);
- \(\bar\beta\): mean across portfolio betas;
- \(k_t\): common proportional variation at time \(t\).

The appendix provides evidence that this approximation is reasonable for the relevant portfolios.

---

# 20. Important Modern-Agent Warning about Post-Ranking Beta

This is an interpretation for modern predictive research, not a criticism the authors frame in modern ML terms.

The full-period post-ranking beta uses the full return history of the portfolio, including observations occurring after earlier monthly return cross-sections.

Therefore:

> it is an econometric device for measuring persistent beta more precisely in an asset-pricing test, not an ex-ante feature that could be used unchanged in a real-time predictive system.

A modern research Agent should **not** directly copy this beta construction into a leakage-free forecasting pipeline.

Instead:

- use rolling or expanding-window beta estimates for prediction;
- reserve full-sample beta for descriptive or retrospective asset-pricing analysis.

---

# 21. Why the Two-Pass Size-Beta Sort Matters

The second beta sort achieves two things.

## 21.1 It creates a much wider range of beta

Across all 100 portfolios, post-ranking beta ranges approximately from:

\[
0.53
\]

to:

\[
1.79.
\]

This is much wider than the beta spread created by size portfolios alone.

---

## 21.2 It holds size approximately fixed within each size decile

Within each size decile, average \(\ln(ME)\) is similar across beta portfolios.

Therefore beta variation is not simply a refined size sort.

This is the key identification result:

> the paper obtains substantial beta variation that is relatively independent of size.

---

# 22. Informal Portfolio-Sort Evidence: Size

When stocks are sorted on size alone:

- average monthly return falls from about **1.64%** for the smallest portfolio to **0.90%** for the largest;
- beta also falls, roughly from **1.44** to **0.90**.

Thus a size sort appears, superficially, to support the SLB beta-return prediction.

But because size and beta move together, this evidence cannot identify which variable matters.

---

# 23. Informal Portfolio-Sort Evidence: Beta

When portfolios are formed on pre-ranking beta alone:

- post-ranking betas span roughly **0.81 to 1.73**;
- average returns show little spread;
- the extreme portfolios have average monthly returns around **1.20%** and **1.18%**.

So:

> large beta differences do not correspond to larger average-return differences.

---

# 24. Size-Then-Beta Sort: The Critical Result

Within a size decile:

- beta rises strongly across the beta-sorted portfolios;
- average return is flat or even declines slightly.

Across size deciles:

- average return declines with size.

Thus the paper's interpretation is:

> beta variation associated with size appears related to average return, but beta variation independent of size is not compensated.

The stronger conclusion is:

> controlling for size, there is no reliable positive relation between beta and average return.

---

# 25. Fama-MacBeth Results: Beta and Size

Table III provides the formal monthly cross-sectional tests.

## 25.1 Beta Alone

Average monthly beta slope:

\[
0.15\%
\]

with:

\[
t=0.46.
\]

This is statistically weak and economically unconvincing.

---

## 25.2 Size Alone

Average monthly slope on:

\[
\ln(ME)
\]

is:

\[
-0.15\%
\]

with:

\[
t=-2.58.
\]

Thus smaller stocks have higher average returns.

---

## 25.3 Beta and Size Together

When both are included:

\[
\beta:
-0.37
\quad
(t=-1.21)
\]

and:

\[
\ln(ME):
-0.17
\quad
(t=-3.41).
\]

The size effect remains strong; beta does not.

---

# 26. Why Measurement Error Does Not Easily Rescue Beta

The authors explicitly ask whether the weak beta result is caused by noisy beta estimates.

They argue against this explanation because:

1. post-ranking beta estimates have relatively small standard errors;
2. beta-sorted portfolios preserve the ordering of pre-ranking beta;
3. beta spreads are large relative to estimation uncertainty;
4. robustness checks using alternative pre- and post-ranking beta estimates produce similar conclusions;
5. the appendix finds similar results over a much longer sample.

This is a useful empirical-writing pattern:

> state the most obvious alternative explanation for a surprising result, then attack it directly with diagnostics.

---

# 27. Book-to-Market Evidence

The paper finds a very strong positive relation between:

\[
BE/ME
\]

and average return.

In the one-dimensional BE/ME portfolios:

- lowest-BE/ME portfolio average return: about **0.30% per month**;
- highest-BE/ME portfolio average return: about **1.83% per month**.

Spread:

\[
1.83\%-0.30\%
=
1.53\% \text{ per month}.
\]

The paper emphasizes that this spread is about twice the size spread observed in the one-dimensional size portfolios.

---

# 28. Book-to-Market Is Not a Disguised Beta Effect

Across BE/ME portfolios, post-ranking betas vary relatively little.

Thus the large return spread across BE/ME groups cannot readily be explained as a beta spread.

This is an important identification point.

---

# 29. Fama-MacBeth Result: Book-to-Market

With \(\ln(BE/ME)\) alone:

\[
\bar b_{BE/ME}
=
0.50\%
\]

per month, with:

\[
t=5.71.
\]

This is stronger than the univariate size result.

---

# 30. Size and Book-to-Market Together

When both are included:

\[
\ln(ME):
-0.11
\quad
(t=-1.99)
\]

and:

\[
\ln(BE/ME):
0.35
\quad
(t=4.44).
\]

Therefore:

- book-to-market does not eliminate size completely;
- size does not eliminate book-to-market;
- both retain incremental explanatory power.

---

# 31. Interaction between Size and Book-to-Market

The average monthly cross-sectional correlation between:

\[
\ln(ME)
\]

and:

\[
\ln(BE/ME)
\]

is approximately:

\[
-0.26.
\]

Thus small firms tend to have larger book-to-market ratios, but the correlation is far from perfect.

The paper cautions against treating size and book-to-market as the same variable.

---

# 32. 10 × 10 Size–Book-to-Market Matrix

Table V forms:

\[
10\times 10
\]

portfolios by:

1. size;
2. book-to-market within size.

This directly visualizes incremental effects.

The overall average monthly returns across the BE/ME groups are:

| BE/ME group | Low | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | High |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Return (%) | 0.64 | 0.98 | 1.06 | 1.17 | 1.24 | 1.26 | 1.39 | 1.40 | 1.50 | 1.63 |

The low-to-high book-to-market spread is therefore approximately:

\[
1.63-0.64=0.99\%
\]

per month.

The paper also reports an average size spread within BE/ME groups of roughly:

\[
0.58\%
\]

per month.

Thus:

- controlling for size, book-to-market still matters;
- controlling for book-to-market, size still matters.

---

# 33. Full Table V Return Matrix

Average monthly return percentages:

| Size \ BE/ME | All | Low | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | High |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| All | 1.23 | 0.64 | 0.98 | 1.06 | 1.17 | 1.24 | 1.26 | 1.39 | 1.40 | 1.50 | 1.63 |
| Small-ME | 1.47 | 0.70 | 1.14 | 1.20 | 1.43 | 1.56 | 1.51 | 1.70 | 1.71 | 1.82 | 1.92 |
| ME-2 | 1.22 | 0.43 | 1.05 | 0.96 | 1.19 | 1.33 | 1.19 | 1.58 | 1.28 | 1.43 | 1.79 |
| ME-3 | 1.22 | 0.56 | 0.88 | 1.23 | 0.95 | 1.36 | 1.30 | 1.30 | 1.40 | 1.54 | 1.60 |
| ME-4 | 1.19 | 0.39 | 0.72 | 1.06 | 1.36 | 1.13 | 1.21 | 1.34 | 1.59 | 1.51 | 1.47 |
| ME-5 | 1.24 | 0.88 | 0.65 | 1.08 | 1.47 | 1.13 | 1.43 | 1.44 | 1.26 | 1.52 | 1.49 |
| ME-6 | 1.15 | 0.70 | 0.98 | 1.14 | 1.23 | 0.94 | 1.27 | 1.19 | 1.19 | 1.24 | 1.50 |
| ME-7 | 1.07 | 0.95 | 1.00 | 0.99 | 0.83 | 0.99 | 1.13 | 0.99 | 1.16 | 1.10 | 1.47 |
| ME-8 | 1.08 | 0.66 | 1.13 | 0.91 | 0.95 | 0.99 | 1.01 | 1.15 | 1.05 | 1.29 | 1.55 |
| ME-9 | 0.95 | 0.44 | 0.89 | 0.92 | 1.00 | 1.05 | 0.93 | 0.82 | 1.11 | 1.04 | 1.22 |
| Large-ME | 0.89 | 0.93 | 0.88 | 0.84 | 0.71 | 0.79 | 0.83 | 0.81 | 0.96 | 0.97 | 1.18 |

Agent interpretation:

> The matrix is more informative than a single multivariate coefficient because it visually demonstrates that the book-to-market gradient exists inside size groups and the size gradient exists inside book-to-market groups.

---

# 34. Leverage Results

The paper uses two leverage variables:

\[
\ln(A/ME)
\]

and:

\[
\ln(A/BE).
\]

Their FM slopes have opposite signs.

## Market leverage

\[
\ln(A/ME):
0.50
\quad
(t=5.69).
\]

Higher market leverage is associated with higher average return.

## Book leverage

\[
\ln(A/BE):
-0.57
\quad
(t=-5.34).
\]

Higher book leverage is associated with lower average return in the joint leverage regression.

At first glance, this appears puzzling.

---

# 35. The Leverage Puzzle Resolves into Book-to-Market

The key algebraic identity is:

\[
\ln(BE/ME)
=
\ln(A/ME)
-
\ln(A/BE).
\]

The opposite-signed leverage slopes are similar in magnitude.

Thus the difference between:

- market leverage;
- book leverage;

is effectively book-to-market equity.

The authors conclude that the leverage result can largely be interpreted as a book-to-market result.

This is a particularly important Agent lesson:

> correlated variables may represent algebraically related transformations of the same underlying economic information.

Before calling multiple variables independent "factors," check whether their effects collapse into a simpler representation.

---

# 36. Economic Interpretation of the Leverage / BE-ME Link

High \(BE/ME\) means:

- the market values the firm's equity low relative to book value;
- market leverage is high relative to book leverage.

The authors interpret this as potentially reflecting:

- poor market-assessed prospects;
- relative distress;
- "market-imposed" leverage caused by a depressed equity price.

Thus book-to-market may summarize a distress-related state of the firm.

---

# 37. Earnings-Price Results

The one-dimensional E/P portfolios show a U-shaped pattern.

Approximate average monthly returns:

- negative earnings portfolio: **1.46%**;
- low positive E/P group: **0.93%**;
- highest E/P group: **1.72%**.

Thus:

- negative earners have high average returns;
- among firms with positive earnings, higher E/P is associated with higher return.

---

# 38. Fama-MacBeth E/P Results

Using E/P variables without size and book-to-market:

### Negative-earnings dummy

\[
0.57
\quad
(t=2.28).
\]

### Positive earnings-price ratio

\[
4.72
\quad
(t=4.57).
\]

Thus E/P appears important in isolation.

---

# 39. Size and Book-to-Market Absorb E/P

When size is added:

- the negative-earnings dummy loses much of its explanatory power.

When both size and book-to-market are added:

- the E/P dummy becomes weak;
- the positive E/P slope falls from:

\[
4.72
\]

to roughly:

\[
0.87
\quad
(t=1.23).
\]

Meanwhile size and book-to-market retain similar slopes to regressions without E/P.

Interpretation:

> much of the apparent E/P effect is shared with size and especially book-to-market.

---

# 40. Table III: Key Joint Regression Results

Selected monthly average FM slopes, with t-statistics in parentheses:

| Specification | \(\beta\) | \(\ln(ME)\) | \(\ln(BE/ME)\) | \(\ln(A/ME)\) | \(\ln(A/BE)\) | E/P Dummy | \(E(+)/P\) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Beta only | 0.15 (0.46) |  |  |  |  |  |  |
| Size only |  | -0.15 (-2.58) |  |  |  |  |  |
| Beta + size | -0.37 (-1.21) | -0.17 (-3.41) |  |  |  |  |  |
| BE/ME only |  |  | 0.50 (5.71) |  |  |  |  |
| Leverage pair |  |  |  | 0.50 (5.69) | -0.57 (-5.34) |  |  |
| E/P pair |  |  |  |  |  | 0.57 (2.28) | 4.72 (4.57) |
| Size + BE/ME |  | -0.11 (-1.99) | 0.35 (4.44) |  |  |  |  |
| Size + leverage pair |  | -0.11 (-2.06) |  | 0.35 (4.32) | -0.50 (-4.56) |  |  |
| Size + E/P pair |  | -0.16 (-3.06) |  |  |  | 0.06 (0.38) | 2.99 (3.04) |
| Size + BE/ME + E/P |  | -0.13 (-2.47) | 0.33 (4.46) |  |  | -0.14 (-0.90) | 0.87 (1.23) |
| Size + leverage + E/P |  | -0.13 (-2.47) |  | 0.32 (4.28) | -0.46 (-4.45) | -0.08 (-0.56) | 1.15 (1.57) |

The table supports the paper's central compression result:

> size and book-to-market retain robust explanatory power, while beta is weak and much of the apparent leverage and E/P information is redundant.

---

# 41. Main Parsimonious Result

The paper summarizes its evidence in three statements.

## 41.1 Beta

When beta variation unrelated to size is isolated:

> there is no reliable relation between beta and average return.

## 41.2 Leverage

The opposite effects of market and book leverage are well captured by book-to-market equity.

## 41.3 Earnings-Price

The E/P relation is largely absorbed by size and book-to-market.

Therefore the paper's main empirical characterization is:

\[
\boxed{
\text{Size} + \text{Book-to-Market}
}
\]

as a parsimonious description of the cross-section of average stock returns associated with the larger variable set.

---

# 42. Subperiod Robustness

The paper splits the main sample approximately into:

- July 1963–December 1976;
- January 1977–December 1990.

## 42.1 Beta

In the regression including beta, size, and book-to-market:

### 1963–1976

\[
\beta:
0.10
\quad
(t=0.25).
\]

### 1977–1990

\[
\beta:
-0.44
\quad
(t=-1.17).
\]

No economically reliable positive beta premium appears.

---

## 42.2 Size

The size effect is negative in both broad subperiods, but statistical power is weaker in the later subperiod.

---

## 42.3 Book-to-Market

The book-to-market slope is strikingly stable.

With size and BE/ME only:

- full period: \(0.35\);
- 1963–1976: \(0.36\);
- 1977–1990: \(0.35\).

The corresponding subperiod t-statistics remain strong.

This stability is a major reason the authors view BE/ME as particularly important.

---

# 43. January Seasonality

Prior research documents a stronger size effect in January.

The authors examine whether the book-to-market effect is similarly concentrated.

They find:

- January BE/ME slopes are roughly twice the February–December slopes;
- but February–December book-to-market slopes remain strongly positive.

Thus:

> book-to-market has a January seasonal, but its relation with average return is not merely a January effect.

---

# 44. Appendix: Why the 1941–1990 Tests Matter

The weak beta result for 1963–1990 conflicts with influential earlier studies.

To investigate whether the result is period-specific, the appendix studies NYSE stocks for:

\[
1941\text{–}1990.
\]

Accounting variables are excluded because pre-1962 COMPUSTAT accounting data are biased.

The appendix therefore focuses on:

- beta;
- size.

---

# 45. Appendix Result: Size Portfolios Can Make Beta Look Successful

For size portfolios over 1941–1990:

- small portfolios have high average returns;
- small portfolios also have high betas.

This produces a strong positive simple beta-return relation.

For example:

- smallest size portfolio average return: about **1.96% per month**;
- largest: about **0.93%**;
- corresponding sum betas roughly **1.60** and **0.95**.

At this level, the SLB model can look successful.

---

# 46. But Size and Beta Are Nearly Indistinguishable in Size Portfolios

For size portfolios:

\[
\operatorname{corr}(\ln ME,\beta)
\approx -0.98.
\]

This means a successful beta regression on size portfolios may simply be capturing the size effect.

This is the exact confounding problem the two-pass portfolio construction is designed to address.

---

# 47. Appendix Two-Pass Size-Beta Result

When each size decile is subdivided by beta:

- beta varies substantially within size deciles;
- average returns do not rise with beta;
- the simple beta slope falls sharply.

The univariate FM beta slope falls from roughly:

\[
1.4\% \text{ per month}
\]

with size-portfolio betas to roughly:

\[
0.23\% \text{ per month}
\]

when beta is measured from size-beta portfolios, and the latter is only about one standard error from zero.

---

# 48. Residual Size Effect

Once beta is allowed to vary independently of size, a strong residual size pattern emerges.

That is:

> beta no longer explains away the size-return relation.

This is further evidence that the beta premium in simpler tests was partly or largely a size effect in disguise.

---

# 49. Decade-Level Beta Instability

The appendix shows substantial instability in the simple beta premium across decades.

Examples:

- 1941–1950: positive and statistically notable beta slope;
- 1971–1980: positive but weak;
- 1981–1990: strongly negative beta slope.

However, once size is included, positive beta premia disappear in the relevant subperiod tests.

The authors conclude that evidence supporting a positive beta-return tradeoff is not robust to controlling for size.

---

# 50. Main Conclusion on the SLB Model

The paper does not claim that all market risk is irrelevant.

Its narrower empirical conclusion is:

> the simple one-dimensional SLB prediction that market beta alone explains the cross-section of average stock returns does not describe the data well in the tested periods.

The authors explicitly leave open the possibility that beta could have a role inside a richer multifactor model.

---

# 51. Rational Interpretation

The authors ask whether the size and book-to-market results can be consistent with rational asset pricing.

They note that the FM framework imposes a linear factor structure compatible with multifactor models.

Under a rational interpretation:

- size and \(BE/ME\) proxy for underlying dimensions of systematic risk;
- the observed average-return premia are compensation for that risk.

But they also state that this is not yet economically satisfying.

The central unresolved question is:

> What are the actual economic risks represented by size and book-to-market?

---

# 52. Relative Distress Interpretation

One proposed interpretation is a **relative distress** or relative-prospects effect.

High \(BE/ME\) firms tend to have:

- low market valuations relative to book value;
- weak earnings prospects;
- potentially greater sensitivity to adverse economic conditions.

If that economic risk is systematic and priced, high \(BE/ME\) firms may rationally have high expected returns.

---

# 53. Economic-Factor Interpretation

The authors suggest investigating whether size and book-to-market are proxies for exposure to more fundamental macroeconomic or credit-related risks.

Examples discussed include:

- differences between low- and high-grade corporate bond returns;
- business-condition variables;
- economic factors studied in prior multifactor work.

The key proposed test is:

> Do loadings on more fundamental economic risk factors absorb the size and book-to-market effects?

---

# 54. Irrational / Behavioral Interpretation

The authors also explicitly consider an alternative:

> the book-to-market relation may arise from market overreaction to firms' relative prospects.

Possible mechanism:

1. investors become excessively optimistic or pessimistic;
2. prices move too far relative to fundamentals;
3. book-to-market becomes extreme;
4. later correction creates predictable return differences.

The paper does not settle the rational-vs-irrational debate.

---

# 55. Simple Overreaction Test

The paper tests a simple DeBondt-Thaler-style implication using prior three-year returns.

If past losers subsequently outperform due to correction of overreaction, lagged three-year return should predict future average return.

In unreported FM regressions:

- the slope on three-year lagged return is small;
- it is statistically weak.

Thus this particular simple overreaction story is not supported by the paper's test.

This does **not** prove that all behavioral explanations are false.

It only weakens that specific implementation.

---

# 56. Practical Applications Discussed by the Authors

If the size and book-to-market effects are real and persistent, they matter for:

## Portfolio performance evaluation

A managed portfolio should be compared with benchmark portfolios having similar:

- size;
- \(BE/ME\).

## Expected-return estimation

Historical average returns of portfolios matched on size and book-to-market can be used to estimate the expected returns of strategies with similar characteristics.

The authors note that these applications do not require settling the rational-vs-irrational interpretation immediately, although persistence would be more questionable under some behavioral stories.

---

# 57. Explicit Caveat: Variable Definitions Matter

This is one of the paper's most important self-critiques.

The inferred premiums depend on how variables are defined.

For example, replacing:

\[
\ln(BE/ME)
\]

with:

\[
\ln(BE)
\]

while keeping size in the regression can leave:

- fitted values;
- intercept;
- \(R^2\);

unchanged, while reallocating slope magnitude between variables.

Therefore:

> individual coefficient interpretations can depend on parameterization even when explanatory content is unchanged.

The authors explicitly state that they do not have a definitive theoretical basis for choosing among all possible versions of the variables.

---

# 58. Explicit Caveat: The Asset Universe Is Stocks Only

The main tests are restricted to stocks.

The authors point out that adding:

- Treasury bills;
- bonds;
- other assets;

could change estimates of factor premia.

They note that large FM intercepts suggest the stock-only regressions may not fit low-return assets like T-bills well.

Thus the evidence is not presented as a universal all-asset equilibrium test.

---

# 59. Can Beta Be Saved?

The authors consider several possibilities.

## Better market proxy

A better market portfolio proxy could theoretically alter the beta result.

However, they regard a complete rescue of the one-factor SLB model as unlikely because it would need to:

1. restore a positive beta-average-return relation;
2. simultaneously eliminate the relevance of size and book-to-market.

## Multifactor role for beta

The authors view this as more plausible.

Beta might matter conditionally in a multifactor model even if its simple unconditional relation with average return is flat.

This is an important nuance:

> the evidence rejects a strong one-factor sufficiency claim more clearly than it rejects every possible role for market exposure.

---

# 60. Outlier Handling

In Table III's FM regressions, extreme values of several accounting ratios are limited.

The smallest and largest 0.5% of observations for:

- \(E(+)/P\);
- \(BE/ME\);
- \(A/ME\);
- \(A/BE\);

are set equal to the adjacent 0.5% and 99.5% fractile values.

The authors report that this treatment does not affect the main inferences.

This is effectively a winsorization-style robustness procedure.

---

# 61. Negative Book Equity

Firms with negative book equity are excluded from the main BE/ME tests.

The authors report that:

- they are relatively rare;
- they are concentrated in later sample years;
- their average returns are high, similar to high-BE/ME firms.

They interpret both:

- negative BE;
- high BE/ME;

as signals associated with poor earnings prospects.

---

# 62. What This Paper Does Not Do

A research Agent should not silently turn this paper into a different research design.

## No train / validation / test split

This is not a machine-learning forecasting experiment.

There is no conventional train/validation/test partition.

## No prediction-loss optimization

There is no neural network, regression model trained by minimizing a predictive loss, or hyperparameter search.

## No backtest of a trading strategy in the modern ML sense

Portfolio sorts are empirical diagnostic tools for average-return relations, not optimized trading strategies.

## No transaction-cost analysis

The paper's primary question is asset pricing and average returns, not implementable net trading performance.

## No formal size and value factor construction in this paper

The paper studies size and book-to-market characteristics and discusses mimicking portfolios conceptually, but the empirical core here is not a later-style factor-return model built from formal size/value factor portfolios.

---

# 63. Data Split vs. Modern ML: Correct Interpretation

For a modern Agent, the closest analogue to a "data split" is the information-timing protocol:

\[
\text{fiscal-year } t-1 \text{ accounting data}
\rightarrow
\text{July } t \text{ to June } t+1 \text{ returns}.
\]

This enforces a temporal gap for accounting variables.

However:

- post-ranking portfolio beta uses full-sample returns for measurement precision.

Therefore this paper should be treated as:

> a cross-sectional explanatory / asset-pricing design with careful information timing,

not as a pure out-of-sample prediction experiment.

---

# 64. Baselines and Comparator Variables

In modern terminology, the paper's "baselines" are not ML models.

The main comparator is the **Sharpe-Lintner-Black single-beta model**.

The competing explanatory variables are:

- beta;
- size;
- book-to-market;
- leverage;
- earnings-price.

The paper's empirical logic is essentially a feature-competition study:

> which variables retain incremental explanatory power after correlated alternatives are introduced?

---

# 65. Evidence Strength

## Strengths of the empirical evidence

- long main sample: 330 months;
- broad U.S. stock coverage;
- careful accounting-information lag;
- individual-stock FM regressions;
- portfolio-sort diagnostics;
- two-pass sorts to break beta-size collinearity;
- subperiod tests;
- alternative beta constructions;
- alternative market proxies;
- 50-year appendix for beta vs. size;
- explicit tests of plausible measurement-error explanations.

## Limits of the evidence

- stock universe only;
- variable definitions can alter coefficient attribution;
- economic source of size and BE/ME premia remains unresolved;
- beta measurement is designed for asset-pricing inference, not real-time forecasting;
- rational and behavioral interpretations are not decisively separated.

---

# 66. Authors' Explicit Future Research Directions

The paper contains unusually useful research suggestions.

## 66.1 Link mimicking portfolios to business conditions

The authors suggest constructing or interpreting portfolios that mimic the underlying common factors proxied by:

- size;
- book-to-market;

then studying how their returns relate to economic variables that measure changing business conditions.

Goal:

> identify the economic risks behind the statistical characteristics.

---

## 66.2 Test More Fundamental Economic Factors

The paper suggests testing whether loadings on economic factors such as credit/default-related variables can explain the size and book-to-market effects.

The question is:

\[
\text{Do characteristic premia disappear once true economic risk exposures are measured?}
\]

---

## 66.3 Test Distress-Factor Loadings

The authors specifically discuss distress-factor interpretations and ask whether loadings on distress-mimicking portfolios can absorb the size and BE/ME relations.

---

## 66.4 Study Economic Fundamentals Directly

The paper reports work in progress showing separation between high- and low-BE/ME firms on economic fundamentals.

A natural continuation is:

> map book-to-market groups to persistent operating performance, earnings strength, and distress.

---

## 66.5 Extend Beyond Stocks

The authors explicitly suggest extending tests to:

- Treasury bills;
- bonds;
- other assets.

This could change conclusions about average factor premia.

---

## 66.6 Explore Multifactor Models

If beta has a role, the authors suggest it is more likely to emerge in a multifactor framework that turns the flat simple beta-return relation into a conditional relation.

---

# 67. Implied Future Work

The following are reasonable research opportunities implied by the paper, but they should be distinguished from explicit author claims.

## 67.1 Stability Across Regimes

The decade-level beta results show strong instability.

A modern study can ask whether:

- size premium;
- book-to-market premium;
- beta premium;

are regime-dependent.

## 67.2 Time-Varying Characteristic Relevance

Instead of estimating one unconditional average FM slope:

\[
\bar b,
\]

model:

\[
b_t
\]

as a time-varying process related to market conditions.

## 67.3 Nonlinear Characteristic Interactions

The paper uses mostly linear monthly cross-sectional regressions and rank-based portfolio sorts.

Modern methods can ask whether expected return depends on nonlinear interactions such as:

\[
f(\text{size}, BE/ME, \text{market state}, \text{other characteristics}).
\]

## 67.4 Characteristic vs. Latent Risk Representation

A modern model can test whether learned latent features:

- reproduce size/value effects;
- subsume them;
- or add orthogonal predictive information.

---

# 68. Research-Agent Methodological Principles Extracted from the Paper

## Principle 1: Correlation is not incremental explanatory power

A variable that works alone may become redundant when correlated variables are added.

---

## Principle 2: Construct independent variation deliberately

The size-then-beta portfolio design is a general research pattern:

> if two candidate explanations are highly correlated, create matched or conditional groups in which one varies while the other is approximately fixed.

---

## Principle 3: Diagnose results with both sorting and regression

Portfolio sorts provide:

- transparency;
- monotonicity checks;
- nonlinear-pattern visibility.

Fama-MacBeth regressions provide:

- incremental-control tests;
- coefficient inference.

Using both creates a stronger empirical argument than either alone.

---

## Principle 4: Time alignment is part of causal credibility

Accounting values must be lagged until they could realistically have been known.

---

## Principle 5: Attack the strongest alternative explanation

The paper's surprising beta result is followed immediately by tests of:

- measurement error;
- market proxy choice;
- sample period;
- portfolio construction.

This is strong scientific practice.

---

## Principle 6: Separate empirical regularity from economic explanation

The paper distinguishes:

### Fact
Size and BE/ME describe average-return variation.

### Interpretation A
They proxy for rationally priced risk.

### Interpretation B
They capture mispricing / overreaction.

The data establish the regularity more strongly than the economic mechanism.

---

# 69. Novelty Decomposition

## Existing components

- CAPM / SLB beta tests;
- size anomaly;
- leverage anomaly;
- earnings-price anomaly;
- book-to-market anomaly;
- Fama-MacBeth methodology;
- portfolio sorting.

## Main empirical novelty

Jointly evaluate the roles of:

\[
\beta,\ ME,\ E/P,\ leverage,\ BE/ME
\]

in a common cross-sectional framework.

## Key identification novelty

Use two-pass size-beta portfolio sorts to generate beta variation that is not simply size variation.

## Main compression result

Show that:

\[
\text{size} + BE/ME
\]

capture much of the average-return variation associated with:

- size;
- book-to-market;
- leverage;
- E/P;

while beta contributes little in the tested specification.

---

# 70. Why the Paper's Argument Is Convincing Internally

The logical chain is:

### Step 1
Single-beta theory predicts a positive beta-return relation.

### Step 2
Simple size portfolios seem to show such a relation.

### Step 3
But size and beta are almost perfectly correlated in those portfolios.

### Step 4
Create beta variation inside fixed size groups.

### Step 5
The beta-return relation disappears while the size-return relation remains.

### Step 6
Test other anomalies individually.

### Step 7
Book-to-market is extremely strong.

### Step 8
Show leverage effects algebraically collapse toward BE/ME.

### Step 9
Show E/P is largely absorbed by size and BE/ME.

### Step 10
Replicate core beta-size conclusions in the 1941–1990 appendix.

This is a carefully layered empirical identification argument.

---

# 71. Writing Strategy

## 71.1 Start from a Famous Theory, Not from a New Variable

The paper begins with a clear theoretical benchmark:

> the SLB model.

Then it organizes anomalies as contradictions to that benchmark.

This gives the empirical study a strong narrative purpose.

---

## 71.2 State the Bottom Line Early

The introduction already tells the reader:

- beta is weak;
- size is robust;
- book-to-market is even stronger;
- size + BE/ME absorb leverage and E/P.

The body then earns that conclusion step by step.

---

## 71.3 Use Data Design to Resolve a Conceptual Dispute

The core argument is not "our regression coefficient is bigger."

It is:

> prior beta evidence is confounded because beta and size move together; our sort construction breaks that confounding.

This is stronger research storytelling because the method directly addresses the reason previous evidence was ambiguous.

---

## 71.4 Use Tables as Arguments

Each table has a specific logical role.

### Table I
Show beta variation independent of size.

### Table II
Show size sort appears to support beta, while beta sort does not.

### Table III
Formalize incremental explanatory power in FM regressions.

### Table IV
Show strong BE/ME and E/P patterns.

### Table V
Show size and BE/ME both work conditionally on the other.

### Table VI
Show subperiod robustness, especially for BE/ME.

### Appendix tables
Attack the claim that results are sample-specific or driven by beta construction.

Agent lesson:

> Every table should answer a specific objection or advance a specific link in the argument.

---

# 72. Limitations

## 72.1 Authors' Explicit Caveats

- coefficient attribution depends on variable definitions;
- there is no clear theoretical basis for preferring every specific characteristic parameterization;
- the tests use stocks only;
- other assets could change factor-premium inferences;
- a better market proxy or a multifactor model might give beta a conditional role;
- economic mechanisms behind size and BE/ME remain unresolved.

---

## 72.2 Additional Agent Critique

The following are modern analytical observations and should not be mistaken for claims made explicitly in the paper.

### Explanatory vs. predictive objective

The FM design tests average cross-sectional pricing relations, not real-time forecasting accuracy.

### Full-sample post-ranking beta

Useful for measurement precision, but not directly deployable as an ex-ante predictor.

### Characteristic interpretation

A characteristic can summarize a latent risk or mispricing mechanism without itself being the primitive economic cause.

### Linear-additive specification

Monthly FM regressions are primarily linear, while true return relations may be nonlinear or interactive.

---

# 73. Failure Modes for a Modern Agent Reusing This Paper

A research Agent should avoid the following mistakes.

## Mistake 1
Treating size and BE/ME as automatically causal risk factors.

The paper shows strong association, not a final causal mechanism.

## Mistake 2
Treating beta as universally useless.

The conclusion is about the tested simple relation and specifications; the authors explicitly allow a possible multifactor conditional role.

## Mistake 3
Using full-period post-ranking beta in a forecasting backtest.

That would not match a real-time information set.

## Mistake 4
Ignoring accounting publication lag.

That would create look-ahead contamination.

## Mistake 5
Calling every correlated anomaly an independent factor.

The paper itself shows how leverage and E/P can be largely redundant with simpler characteristics.

---

# 74. Reproducibility Checklist

A research Agent attempting to reproduce the main paper should specify:

## Universe
- nonfinancial NYSE / AMEX / NASDAQ firms;
- CRSP × COMPUSTAT intersection.

## Dates
- accounting data: 1962–1989;
- main returns: July 1963–December 1990.

## Accounting lag
- fiscal-year \(t-1\) data;
- returns July \(t\) to June \(t+1\).

## Size
- June \(ME\).

## Ratios
- December \(t-1\) market equity for denominators.

## Pre-ranking beta
- 24–60 monthly observations over preceding five years.

## Size-beta portfolios
- 10 NYSE size breakpoints;
- 10 beta groups inside each size decile;
- 100 portfolios.

## Holding returns
- equal-weighted;
- July to June.

## Post-ranking beta
- full-period portfolio returns;
- current + lagged market-return slope sum.

## FM regressions
- monthly individual-stock cross-sections;
- time-series mean of monthly slopes;
- t-statistic from time-series standard error.

## Outlier treatment
- 0.5% tails of selected accounting ratios capped as described.

---

# 75. Relationship to Sharpe (1964)

This paper is a natural empirical counterpart to Sharpe's theoretical result.

Sharpe's equilibrium logic implies that expected return should be related to systematic market risk.

Fama and French test a version of that implication through the SLB beta relation and find:

> beta does not adequately describe the observed cross-section once variation independent of size is isolated.

Thus the intellectual relation is:

\[
\text{Sharpe theoretical systematic-risk pricing}
\]

versus:

\[
\text{Fama-French empirical challenge to single-beta sufficiency}.
\]

A research Agent should distinguish:

- the theoretical idea that non-diversifiable risk matters;
- the empirical claim that one measured market beta is sufficient.

The paper challenges the second much more directly than the first.

---

# 76. Relationship to Cross-Sectional Machine Learning

This paper is highly relevant to modern cross-sectional prediction even though it predates modern ML.

It teaches that a predictor should be judged on:

1. incremental information after correlated features are controlled;
2. stability across time;
3. independence from obvious proxy variables;
4. correct time availability;
5. robustness to alternative construction choices;
6. economic interpretability where possible.

For ML research, this maps naturally to questions such as:

- does a learned representation add information beyond size/value characteristics?
- does a new model improve within characteristic-matched groups?
- are gains concentrated in one regime?
- are features available ex ante?
- does a module discover real orthogonal structure or merely repackage existing factors?

---

# 77. Potential Modern Experiment Hooks

These are research-Agent extensions, not experiments in the original paper.

## E1. Characteristic-neutral model comparison

Test whether a predictive model still adds rank information after neutralizing:

- size;
- book-to-market.

## E2. Conditional IC by size × value bins

Reproduce the paper's conditional-sort logic using prediction metrics:

\[
IC_{g,t}
\]

inside size-value groups.

Goal:

> verify that predictive power is not driven by a simple characteristic tilt.

## E3. Time-varying factor relevance

Estimate rolling cross-sectional slopes for:

- size;
- value;
- learned latent factors.

Compare stability and decay.

## E4. Learned representation vs. known characteristics

Regress model scores on known characteristics and measure residual predictive power.

---

# 78. Agent Research Instructions

When using this paper:

1. **Preserve the distinction between cross-sectional explanation and forecasting.**
2. **Do not describe the paper as a train/validation/test experiment.**
3. **Treat the accounting-data lag as a critical part of the empirical protocol.**
4. **Do not use full-sample post-ranking beta as a real-time predictor without modification.**
5. **When a new predictor is correlated with size or BE/ME, test incremental power conditionally.**
6. **Use sorts and regressions together where possible.**
7. **Distinguish statistical characteristics from primitive economic risk factors.**
8. **Do not claim the paper resolves rational-vs-behavioral explanations.**
9. **Do not claim market beta can never matter; the paper leaves open conditional multifactor roles.**
10. **When several variables are correlated, search for algebraic or economic redundancy before treating them as separate discoveries.**
11. **Check subperiod stability before declaring a universal cross-sectional premium.**
12. **If a surprising result contradicts a famous theory, explicitly test the most obvious measurement and sample-period objections.**

---

# 79. Agent Takeaways

## Most Important Research Problem

Which variables independently describe the cross-section of average U.S. stock returns once beta, size, leverage, earnings-price, and book-to-market are tested together?

## Most Important Methodological Idea

**Create variation in one candidate predictor that is independent of a confounding predictor.**

The size-then-beta sort is the canonical example.

## Most Important Empirical Result

For 1963–1990:

- size has a robust negative relation with average return;
- book-to-market has a robust positive relation;
- market beta has little explanatory power once its variation is separated from size.

## Strongest Characteristic in the Paper

Book-to-market equity is the most consistently powerful of the variables studied.

## Most Important Redundancy Result

Size and book-to-market absorb much of the apparent explanatory power of:

- leverage;
- E/P.

## Most Important Caveat

The statistical success of size and BE/ME does not by itself identify their true economic mechanism.

## Most Useful Research-Agent Lesson

A new variable or model is not genuinely informative merely because it predicts returns in isolation.

The stronger test is:

> **Does it add stable cross-sectional information after known correlated predictors are controlled, matched, or neutralized?**

---

# 80. Source Location Map

Useful journal-page locations for verification:

- **pp. 427–429:** motivation, SLB model, prior anomalies, main conclusions.
- **pp. 429–432:** sample construction, accounting lag, beta-estimation design.
- **pp. 432–440:** beta vs. size portfolio sorts, Table I–III, beta measurement-error discussion.
- **pp. 440–445:** book-to-market, leverage, E/P evidence.
- **pp. 445–449:** parsimonious size + BE/ME model, Table V–VI, subperiods, caveats.
- **pp. 449–452:** conclusions, rational and irrational interpretations, applications.
- **pp. 452–464:** appendix on size vs. beta for 1941–1990 and robustness diagnostics.

---

# 81. Compact Retrieval Summary

Fama and French (1992) test the joint roles of market beta, firm size, earnings-price, leverage, and book-to-market equity in the cross-section of average U.S. stock returns. Using nonfinancial CRSP-COMPUSTAT firms and returns from July 1963 to December 1990, they carefully lag accounting information, form portfolios designed to create beta variation independent of size, and run month-by-month Fama-MacBeth regressions. The central identification result is that the positive beta-return relation seen in simple size portfolios disappears once beta is allowed to vary independently of size, whereas the negative size-return relation remains. Book-to-market equity has an especially strong positive relation with average return. The opposite signs of market and book leverage largely collapse algebraically into book-to-market, while much of the E/P effect is absorbed by size and book-to-market. Subperiod and 1941–1990 appendix tests reinforce the weak role of beta and the robustness of size, while BE/ME is especially stable. The paper does not resolve whether size and BE/ME represent rationally priced risk or mispricing; it explicitly proposes research into underlying economic risk factors, distress, business conditions, other asset classes, and multifactor models.
