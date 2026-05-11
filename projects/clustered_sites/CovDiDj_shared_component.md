Setup
Let $z_{is}, z_{js}, z_{ls} \overset{\text{iid}}{\sim} \mathcal{N}(0,1)$ (independent scalar Gaussians, one coordinate $s$ from each whitened observation). Define:
$$a_s = z_{is} - z_{js}, \qquad b_s = z_{js} - z_{ls}.$$
We need $\mathbb{E}[a_s^2]$, $\mathbb{E}[b_s^2]$, and $\mathbb{E}[a_s^2 b_s^2]$.
Step 1: $\mathbb{E}[a_s^2]$ and $\mathbb{E}[b_s^2]$
Since $z_{is}, z_{js} \overset{\text{iid}}{\sim} \mathcal{N}(0,1)$:
$$\mathbb{E}[a_s^2] = \operatorname{Var}(z_{is} - z_{js}) = \operatorname{Var}(z_{is}) + \operatorname{Var}(z_{js}) = 1 + 1 = 2,$$
and identically $\mathbb{E}[b_s^2] = 2$.
Step 2: $\mathbb{E}[a_s^2 b_s^2]$ via explicit expansion
Substitute the definitions:
$$\mathbb{E}[a_s^2 b_s^2] = \mathbb{E}!\left[(z_{is} - z_{js})^2(z_{js} - z_{ls})^2\right].$$
Expand each square:
$$= \mathbb{E}!\left[(z_{is}^2 - 2z_{is}z_{js} + z_{js}^2)(z_{js}^2 - 2z_{js}z_{ls} + z_{ls}^2)\right].$$
Expanding the full product gives 9 terms:
$$= \mathbb{E}\Big[ z_{is}^2 z_{js}^2 +

    2z_{is}^2 z_{js}z_{ls} +

    z_{is}^2 z_{ls}^2 +

    2z_{is}z_{js}^3 +\\

    4z_{is}z_{js}^2 z_{js}z_{ls} +

    2z_{is}z_{js}z_{ls}^2 +

    z_{js}^4 +

    2z_{js}^3 z_{ls} +

    z_{js}^2 z_{ls}^2 \Big].$$


Step 3: Evaluate each expectation
Using independence of $z_{is}, z_{js}, z_{ls}$ and the standard Gaussian moments:

$$\mathbb{E}[z^k] = \begin{cases} 0 & k \text{ odd} \\ 
1 & k=2 \\
3 & k=4 
\end{cases}$$

Term	Expectation	Value
- 1	$\mathbb{E}[z_{is}^2]\mathbb{E}[z_{js}^2]$	$1 \cdot 1 = 1$
- 2	$-2\mathbb{E}[z_{is}^2]\mathbb{E}[z_{js}]\mathbb{E}[z_{ls}]$	$-2 \cdot 1 \cdot 0 \cdot 0 = 0$
- 3	$\mathbb{E}[z_{is}^2]\mathbb{E}[z_{ls}^2]$	$1 \cdot 1 = 1$
- 4	$-2\mathbb{E}[z_{is}]\mathbb{E}[z_{js}^3]$	$-2 \cdot 0 \cdot 0 = 0$
- 5	$4\mathbb{E}[z_{is}]\mathbb{E}[z_{js}^2]\mathbb{E}[z_{ls}]$	$4 \cdot 0 \cdot 1 \cdot 0 = 0$
- 6	$-2\mathbb{E}[z_{is}]\mathbb{E}[z_{js}]\mathbb{E}[z_{ls}^2]$	$-2 \cdot 0 \cdot 0 \cdot 1 = 0$
- 7	$\mathbb{E}[z_{js}^4]$	$3$
- 8	$-2\mathbb{E}[z_{js}^3]\mathbb{E}[z_{ls}]$	$-2 \cdot 0 \cdot 0 = 0$
- 9	$\mathbb{E}[z_{js}^2]\mathbb{E}[z_{ls}^2]$	$1 \cdot 1 = 1$

Summing all non-zero contributions:
$$\mathbb{E}[a_s^2 b_s^2] = 1 + 0 + 1 + 0 + 0 + 0 + 3 + 0 + 1 = \mathbf{6}.$$
Step 4: The Covariance
$$\operatorname{Cov}(a_s^2, b_s^2) = \mathbb{E}[a_s^2 b_s^2] - \mathbb{E}[a_s^2]\mathbb{E}[b_s^2] = 6 - 2 \cdot 2 = \boxed{2}.$$
