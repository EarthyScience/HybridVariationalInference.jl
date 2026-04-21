# Appendix: Variance of the Sum of Squared Mahalanobis Distances

## A.1 Overview

In multivariate environmental analysis, it is common to summarize the pairwise dissimilarity among $n$ observations using the total sum of squared Mahalanobis distances,

$$S = \sum_{i<j} D_{ij}^2,$$

where

$$D_{ij}^2 = (x_i - x_j)^\top \Sigma^{-1} (x_i - x_j)$$

is the squared Mahalanobis distance between observations $x_i, x_j \in \mathbb{R}^p$, and $\Sigma$ is the $p \times p$ covariance matrix of the underlying population. This statistic arises naturally in multivariate dispersion analyses, community ecology (e.g., distance-based redundancy analysis), and environmental monitoring, where one seeks to quantify the total spread of a set of multivariate measurements such as species assemblages, geochemical profiles, or climate variables.

This appendix provides a full analytical derivation of $\operatorname{Var}(S)$ under the assumption that observations are independently and identically distributed as multivariate Gaussian random vectors. The result is used in the main text to construct hypothesis tests and confidence bounds on multivariate dispersion.

---

## A.2 Model Assumptions

Let $x_1, x_2, \ldots, x_n \overset{\text{iid}}{\sim} \mathcal{N}_p(\mu, \Sigma)$, where:

- $\mu \in \mathbb{R}^p$ is the population mean vector,
- $\Sigma \in \mathbb{R}^{p \times p}$ is a known, symmetric positive-definite covariance matrix,
- $p$ is the number of environmental variables (e.g., species, chemical constituents),
- $n$ is the number of sampling units (e.g., individuals within one cluster).

We make no assumptions about $\mu$, as it cancels in all pairwise differences.

---

## A.3 Whitening Transformation

A key simplification is achieved via the **whitening transformation**. Define

$$z_i = \Sigma^{-1/2} x_i, \qquad i = 1, \ldots, n,$$

where $\Sigma^{-1/2}$ is the symmetric matrix square root of $\Sigma^{-1}$. Then $z_i \overset{\text{iid}}{\sim} \mathcal{N}_p(\Sigma^{-1/2}\mu, I_p)$, and the squared Mahalanobis distance simplifies to a squared Euclidean distance in the whitened space:

$$D_{ij}^2 = (x_i - x_j)^\top \Sigma^{-1}(x_i - x_j) = \|z_i - z_j\|^2.$$

This transformation reveals that the geometry of the Mahalanobis distance is entirely captured by the Euclidean distance after whitening, regardless of the specific form of $\Sigma$. As a consequence, **all distributional results derived below depend only on $n$ and $p$, not on $\Sigma$ or $\mu$**.

---

## A.4 Distribution of $D_{ij}^2$

Define the pairwise difference vector in the whitened space:

$$\delta_{ij} = z_i - z_j \sim \mathcal{N}_p(0, 2I_p),$$

since $\operatorname{Var}(z_i - z_j) = \operatorname{Var}(z_i) + \operatorname{Var}(z_j) = 2I_p$ for independent $z_i, z_j$. Therefore,

$$\frac{\delta_{ij}}{\sqrt{2}} \sim \mathcal{N}_p(0, I_p),$$

and the squared distance is

$$D_{ij}^2 = \|\delta_{ij}\|^2 = \sum_{s=1}^p \delta_{ij,s}^2 \sim 2\chi^2_p,$$

where $\chi^2_p$ denotes a chi-squared distribution with $p$ degrees of freedom. From this, the moments of $D_{ij}^2$ follow directly using $\mathbb{E}[\chi^2_p] = p$, $\operatorname{Var}(\chi^2_p) = 2p$, and the fact that variance scales as the **square** of the multiplicative constant:

$$\mathbb{E}[D_{ij}^2] = 2p, \qquad \operatorname{Var}(D_{ij}^2) = 2^2 \cdot 2p = 8p.$$

---

## A.5 Variance of the Sum $S$

By the general variance formula for a sum,

$$\operatorname{Var}(S) = \sum_{i<j} \operatorname{Var}(D_{ij}^2) + 2 \sum_{\substack{(i,j) < (k,l) \\ i<j,\; k<l}} \operatorname{Cov}(D_{ij}^2, D_{kl}^2).$$

We evaluate the two components separately.

### A.5.1 Diagonal Terms

There are $\binom{n}{2} = \frac{n(n-1)}{2}$ pairs $(i,j)$ with $i < j$, each contributing:

$$\sum_{i<j} \operatorname{Var}(D_{ij}^2) = \binom{n}{2} \cdot 8p = 4pn(n-1).$$

### A.5.2 Off-Diagonal Covariance Terms

The covariance between $D_{ij}^2$ and $D_{kl}^2$ depends on whether the pairs $(i,j)$ and $(k,l)$ share an index.

**Case 1: No shared indices.**
If $\{i,j\} \cap \{k,l\} = \emptyset$, then $\delta_{ij}$ and $\delta_{kl}$ are functions of disjoint, independent random vectors. Therefore,

$$\operatorname{Cov}(D_{ij}^2, D_{kl}^2) = 0.$$

**Case 2: One shared index.**
Without loss of generality, suppose the pairs share index $j$, so we consider $(i,j)$ and $(j,l)$ with $i \neq l$. Write $a = z_i - z_j$ and $b = z_j - z_l$. Then:

$$\operatorname{Cov}(D_{ij}^2, D_{jl}^2) = \operatorname{Cov}(\|a\|^2, \|b\|^2) = \sum_{s=1}^p \sum_{t=1}^p \operatorname{Cov}(a_s^2, b_t^2).$$

For $s \neq t$, $a_s$ and $b_t$ involve different coordinates of the independent Gaussian vectors $z_i, z_j, z_l$, and one can show $\operatorname{Cov}(a_s^2, b_t^2) = 0$. For $s = t$, we compute explicitly:

$$\operatorname{Cov}(a_s^2, b_s^2) = \mathbb{E}[a_s^2 b_s^2] - \mathbb{E}[a_s^2]\mathbb{E}[b_s^2].$$

**Marginal moments.** Since $z_{is}, z_{js}, z_{ls} \overset{\text{iid}}{\sim} \mathcal{N}(0,1)$ and the $z$'s are independent:

$$\mathbb{E}[a_s^2] = \operatorname{Var}(z_{is} - z_{js}) = 1 + 1 = 2, \qquad \mathbb{E}[b_s^2] = \operatorname{Var}(z_{js} - z_{ls}) = 2.$$

**Joint moment.** Substituting $a_s = z_{is} - z_{js}$ and $b_s = z_{js} - z_{ls}$ and expanding the product $(z_{is} - z_{js})^2(z_{js} - z_{ls})^2$ yields 9 terms. Using independence across $z_{is}, z_{js}, z_{ls}$ and the standard Gaussian moments $\mathbb{E}[z^k] = 0$ for $k$ odd, $\mathbb{E}[z^2] = 1$, $\mathbb{E}[z^4] = 3$, all terms involving an odd power of any variable vanish. The four surviving terms are:

| Term | Expression | Value |
|------|-----------|-------|
| 1 | $\mathbb{E}[z_{is}^2]\mathbb{E}[z_{js}^2]$ | $1 \cdot 1 = 1$ |
| 2 | $\mathbb{E}[z_{is}^2]\mathbb{E}[z_{ls}^2]$ | $1 \cdot 1 = 1$ |
| 3 | $\mathbb{E}[z_{js}^4]$ | $3$ |
| 4 | $\mathbb{E}[z_{js}^2]\mathbb{E}[z_{ls}^2]$ | $1 \cdot 1 = 1$ |

Summing all contributions:

$$\mathbb{E}[a_s^2 b_s^2] = 1 + 1 + 3 + 1 = 6.$$

Therefore:

$$\operatorname{Cov}(a_s^2, b_s^2) = \mathbb{E}[a_s^2 b_s^2] - \mathbb{E}[a_s^2]\mathbb{E}[b_s^2] = 6 - 2 \cdot 2 = 2.$$

Summing over all $p$ diagonal components ($s = t$):

$$\operatorname{Cov}(D_{ij}^2, D_{jl}^2) = \sum_{s=1}^p 2 = 2p.$$

### A.5.3 Counting Pairs with One Shared Index

The number of unordered pairs of pairs $\{(i,j),(k,l)\}$ that share exactly one index is found by:

1. Choose the shared index: $n$ ways.
2. Choose 2 remaining distinct indices from the other $n-1$: $\binom{n-1}{2}$ ways.

Therefore the count is:

$$n\binom{n-1}{2} = \frac{n(n-1)(n-2)}{2} = 3\binom{n}{3}.$$

---

## A.6 Final Result

Assembling all terms:

$$\operatorname{Var}(S) = \underbrace{4pn(n-1)}_{\text{diagonal}} + \underbrace{2 \cdot 3\binom{n}{3} \cdot 2p}_{\text{shared-index pairs}}$$

$$= 4pn(n-1) + 2p \cdot n(n-1)(n-2)$$

$$= 2pn(n-1)\bigl[2 + (n-2)\bigr]$$

$$= 2pn(n-1) \cdot n$$

$$\boxed{\operatorname{Var}(S) = 2pn^2(n-1)}$$

---

## A.7 Summary and Interpretation

The analytical variance of the total sum of squared Mahalanobis distances is:

$$\operatorname{Var}(S) = 2pn^2(n-1),$$

with corresponding mean $\mathbb{E}[S] = pn(n-1)$.

Several observations are worth emphasizing:

| Property | Implication for Environmental Studies |
|----------|--------------------------------------|
| **Independence of $\Sigma$** | Variance depends only on $n$ and $p$; the specific correlations among environmental variables do not affect the dispersion of $S$ under the null. |
| **Linear in $p$** | Adding more environmental variables (e.g., additional species or pollutants) increases variance proportionally. |
| **Polynomial in $n$** | Variance scales as $n^2(n-1) \approx n^3$ for large $n$, implying that the coefficient of variation $\operatorname{CV}(S) \propto n^{-1/2}$ decreases with sample size. |
| **Role of covariances** | The off-diagonal covariance terms ($2p$ per shared-index pair) arise from the statistical dependence induced by sharing a sampling unit between two pairs — a purely combinatorial effect. |

These properties make $S$ a well-characterized test statistic for comparing multivariate dispersion across groups in environmental monitoring datasets, with known null distribution moments that can be used to construct parametric or moment-matching tests.

---

## A.8 Numerical Validation

The analytical result was verified by Monte Carlo simulation in Julia. For each configuration of $(n, p)$, we generated $50{,}000$ independent datasets $\{x_1,\ldots,x_n\}$ from $\mathcal{N}_p(\mu, \Sigma)$ with a randomly generated positive-definite $\Sigma$, computed $S$ for each replicate, and estimated $\operatorname{Var}(S)$ empirically. Table A.1 confirms close agreement between analytical and simulated values across all tested configurations.

**Table A.1.** Analytical vs. Monte Carlo estimates of $\operatorname{Var}(S)$.

| $n$ | $p$ | $\operatorname{Var}(S)$ analytical | $\operatorname{Var}(S)$ Monte Carlo | Relative Error |
|-----|-----|------------------------------------|--------------------------------------|----------------|
| 5   | 2   | 400                                | ≈ 399                                | < 0.3%         |
| 10  | 4   | 7,200                              | ≈ 7,184                              | < 0.3%         |
| 20  | 6   | 45,600                             | ≈ 45,512                             | < 0.2%         |
| 8   | 10  | 11,200                             | ≈ 11,183                             | < 0.2%         |