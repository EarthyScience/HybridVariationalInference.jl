using LinearAlgebra, Distributions, Printf

"""
    overdispersion_test(Y, μ, Σ; α=0.05)

Test whether the q×p sample matrix Y (rows = indiviudals) is overdispersed
relative to the reference distribution N(μ, Σ).

Returns: S_n, E0, Var0, Z, p_value_normal, p_value_chisq
"""
function overdispersion_test(Y::Matrix{Float64}, μ::Vector{Float64},
                              Σ::Matrix{Float64}; α::Float64=0.05)

    n, p = size(Y)
    Σ_inv = inv(Σ)

    # Step 1: Compute S_n
    S_n = 0.0
    for i in 1:n
        for j in (i+1):n
            d = Y[i, :] - Y[j, :]
            S_n += dot(d, Σ_inv * d)
        end
    end

    # Step 2: Null moments
    E0   = p * n * (n - 1)
    Var0 = 2.0 * p * n^2 * (n - 1)
    SD0  = sqrt(Var0)

    # Step 3: Standardized statistic
    Z = (S_n - E0) / SD0

    # Step 4: p-values
    p_val_normal = 1.0 - cdf(Normal(0, 1), Z)

    ν  = p * (n - 1)          # degrees of freedom
    c  = Float64(n)           # scaling constant
    p_val_chisq = 1.0 - cdf(Chisq(ν), S_n / c)

    # Report
    println("="^55)
    println("  Overdispersion Test (n=$n, p=$p, α=$α)")
    println("="^55)
    @printf("  Observed  S_n         = %12.4f\n", S_n)
    @printf("  Expected  E₀[S_n]     = %12.4f\n", E0)
    @printf("  Std. Dev  SD₀[S_n]    = %12.4f\n", SD0)
    @printf("  Z statistic           = %12.4f\n", Z)
    @printf("  p-value (Normal)      = %12.6f\n", p_val_normal)
    @printf("  p-value (χ² approx)   = %12.6f\n", p_val_chisq)
    println("-"^55)
    decision = (p_val_chisq < α) ? "REJECT H₀ — overdispersed" :
                                    "FAIL TO REJECT H₀"
    println("  Decision (α=$α):  $decision")
    println("="^55)

    return (S_n=S_n, E0=E0, Var0=Var0, Z=Z,
            p_normal=p_val_normal, p_chisq=p_val_chisq)
end


# ── Simulation study ─────────────────────────────────────────────────────────

using Random
Random.seed!(123)

p, N = 4, 200       # dimension and reference sample size
n    = 20           # test sample size

# Reference distribution
A = randn(p, p);  Σ = A*A' + p*I;  μ = randn(p)

println("\n── Scenario 1: Sample drawn from H₀ (no overdispersion) ──\n")
Y_null = rand(MvNormal(μ, Σ), n)'   |> Matrix
overdispersion_test(Y_null, μ, Σ, α=0.05)

println("\n── Scenario 2: Sample drawn from inflated variance (2Σ) ──\n")
Y_over = rand(MvNormal(μ, 2*Σ), n)' |> Matrix
overdispersion_test(Y_over, μ, Σ, α=0.05)

println("\n── Scenario 3: Monte Carlo size and power study ──\n")
n_sim = 20_000
rejections_H0  = 0
rejections_H1  = 0

for _ in 1:n_sim
    Y0 = rand(MvNormal(μ, Σ),   n)' |> Matrix
    Y1 = rand(MvNormal(μ, 2*Σ), n)' |> Matrix

    r0 = overdispersion_test(Y0, μ, Σ, α=0.05)
    r1 = overdispersion_test(Y1, μ, Σ, α=0.05)

    rejections_H0 += (r0.p_chisq < 0.05)
    rejections_H1 += (r1.p_chisq < 0.05)
end

@printf("  Empirical size  (H₀ true, should ≈ 0.05): %.4f\n",
        rejections_H0 / n_sim)
@printf("  Empirical power (H₁ true, Σ→2Σ):          %.4f\n",
        rejections_H1 / n_sim)
