# Appendix B: Testing for Overdispersion in Multivariate Environmental Data

## B.1 Overview

In aggregating individuals parameter sets into clusters of similar parameters, one need to choose cluster size so that dispersion across parameter sets matches the predicted uncertainty of a parameter prediction. If a cluster of individuals exhibits **greater multivariate spread** than expected under the predicted reference distribution $\mathcal{N}_p(\mu, \Sigma)$, the cluster needs to be split into subclusters.

We formalize this check of greater spread as a one-sided hypothesis test based on the total sum of squared Mahalanobis distances,

$$S_n = \sum_{i<j} D_{ij}^2, \qquad D_{ij}^2 = (x_i - x_j)^\top \Sigma^{-1}(x_i - x_j),$$

computed on the $n$ records within one cluster. Under the null hypothesis that these records are a random sample from $\mathcal{N}_p(\mu, \Sigma)$, the moments of $S_n$ are fully characterized analytically, enabling a tractable test without resampling.

---

## B.2 Null Hypothesis and Test Statistic

Hypotheses:

$H_0: x_1, \ldots, x_n \overset{\text{iid}}{\sim} \mathcal{N}_p(\mu, \Sigma) \quad \text{(no overdispersion)}$

$H_1: \text{the sample is more dispersed than } \mathcal{N}_p(\mu, \Sigma) \quad \text{(overdispersion)}$

From Appendix A, under $H_0$ with $n$ indiviudals:

$$\mathbb{E}_0[S_n] = pn(n-1)$$

$$\operatorname{Var}_0(S_n) = 2pn^2(n-1)$$

$$\operatorname{SD}_0(S_n) = \sqrt{2pn^2(n-1)} = n\sqrt{2p(n-1)}$$

These depend only on the dimension $p$ and sample size $n$, not on $\mu$ or $\Sigma$.

### B.2.3 Standardized Test Statistic

We construct a standardized statistic:

$$Z = \frac{S_n - \mathbb{E}_0[S_n]}{\operatorname{SD}_0(S_n)} = \frac{S_n - pn(n-1)}{n\sqrt{2p(n-1)}}.$$

For sufficiently large $n$, by a central limit theorem argument, $Z \overset{\cdot}{\sim} \mathcal{N}(0,1)$ under $H_0$.

---

## B.3 Moment-Matching via the Chi-Squared Approximation

For small to moderate $n$, a normal approximation may be inadequate. We instead match the first two moments of $S_n$ to a **scaled chi-squared distribution**:

$$S_n \overset{\cdot}{\sim} c \cdot \chi^2_\nu,$$

where $c$ and $\nu$ are chosen by moment matching:

$$\mathbb{E}[c\chi^2_\nu] = c\nu = pn(n-1),$$
$$\operatorname{Var}(c\chi^2_\nu) = 2c^2\nu = 2pn^2(n-1).$$

Solving:

$$c = \frac{\operatorname{Var}_0(S_n)}{\mathbb{E}_0[S_n] \cdot 2} \cdot 2 = \frac{2pn^2(n-1)}{2pn(n-1)} = n,$$

$$\nu = \frac{\mathbb{E}_0[S_n]}{c} = \frac{pn(n-1)}{n} = p(n-1).$$

Therefore:

$$\boxed{S_n \overset{\cdot}{\sim} n \cdot \chi^2_{p(n-1)}}$$

The $p$-value for a one-sided overdispersion test is:

$$p\text{-value} = P\!\left(\chi^2_{p(n-1)} \geq \frac{S_n}{n}\right).$$

---

## B.4 Summary of the Test Procedure

The full testing procedure is summarized below.

**Input:** A sample $\{x_1, \ldots, x_n\} \subset \mathbb{R}^p$, reference parameters $\mu$ and $\Sigma^{-1}$.
TODO: Is $\mu$ required?

---

**Step 1.** Compute the observed test statistic:
$$S_n = \sum_{i<j} (x_i - x_j)^\top \Sigma^{-1}(x_i - x_j).$$

**Step 2.** Compute the null moments:
$$\mathbb{E}_0[S_n] = pn(n-1), \qquad \operatorname{Var}_0(S_n) = 2pn^2(n-1).$$

**Step 3.** Compute the standardized statistic (large $n$):
$$Z = \frac{S_n - pn(n-1)}{n\sqrt{2p(n-1)}}.$$

**Step 4.** Compute the $p$-value:
- **Normal approximation** (large $n$): $\quad p\text{-value} = 1 - \Phi(Z).$
- **Chi-squared approximation** (small/moderate $n$): $\quad p\text{-value} = P\!\left(\chi^2_{p(n-1)} \geq S_n/n\right).$

**Step 5.** Reject $H_0$ (sampled from reference and not overdispersed) at significance level $\alpha$ if $p\text{-value} < \alpha$.

## B.6 Practical Considerations

### B.6.1 Unknown $\mu$ and $\Sigma$

In practice, $\mu$ and $\Sigma$ are typically estimated from a large **reference dataset** of $N \gg n$ indiviudals. Denote these estimates $\hat{\mu}$ and $\hat{\Sigma}$. Then:

- Replace $\Sigma^{-1}$ with $\hat{\Sigma}^{-1}$ in the computation of $D_{ij}^2$.
- The null moments remain approximately valid provided $N$ is large relative to $n$ and $p$, so that estimation uncertainty in $\hat{\mu}$ and $\hat{\Sigma}$ is negligible.
- If $N$ is moderate, a correction factor or bootstrap-based critical value should be used.

### B.6.2 Minimum Sample Size

The chi-squared approximation requires $p(n-1) \geq 1$, i.e., $n \geq 2$. For the normal approximation to be reliable, we recommend $n \geq 30$ or $p(n-1) \geq 30$.

### B.6.3 Sensitivity to Gaussianity

The null moments $\mathbb{E}_0[S_n]$ and $\operatorname{Var}_0(S_n)$ depend on the Gaussian assumption only through the fourth-order moments of $D_{ij}^2$. If the reference distribution has heavier tails, $\operatorname{Var}_0(S_n)$ will be larger, and the test as stated will be **anti-conservative** (too many false rejections). In such cases, the null variance should be estimated empirically from the reference dataset.

