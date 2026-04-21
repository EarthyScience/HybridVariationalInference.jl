using LinearAlgebra, Statistics, Random, Distributions
using Printf

"""
Test the analytical variance Var(S) = 2p*n^2*(n-1)
where S = sum_{i<j} D^2_ij and D^2_ij is the squared Mahalanobis distance.
"""
function compute_S(X::Matrix{Float64}, Σ_inv::Matrix{Float64})
    n = size(X, 2)
    S = 0.0
    for i in 1:n
        for j in (i+1):n
            d = X[:, i] - X[:, j]
            S += dot(d, Σ_inv * d)
        end
    end
    return S
end

function run_test(;
    n::Int = 10,
    p::Int = 4,
    n_sim::Int = 10_000,
    seed::Int = 42
)
    Random.seed!(seed)

    # Generate a random positive definite covariance matrix Σ
    A = randn(p, p)
    Σ = A * A' + p * I   # ensure positive definiteness
    Σ_inv = inv(Σ)
    μ = randn(p)

    dist = MvNormal(μ, Σ)

    # ----- Analytical results -----
    analytical_mean  = 2.0 * p * binomial(n, 2)          # E[S] = 2p * C(n,2)
    analytical_var   = 2.0 * p * n^2 * (n - 1)           # Var(S)

    # ----- Monte Carlo estimates -----
    S_samples = Vector{Float64}(undef, n_sim)
    for sim in 1:n_sim
        X = rand(dist, n)          # p × n matrix
        S_samples[sim] = compute_S(X, Σ_inv)
    end

    mc_mean = mean(S_samples)
    mc_var  = var(S_samples)

    # ----- Report -----
    println("="^55)
    println("  Parameters: n=$n, p=$p, n_sim=$n_sim")
    println("="^55)
    println("  Quantity        Analytical       Monte Carlo")
    println("-"^55)
    @printf("  E[S]          %12.4f      %12.4f\n", analytical_mean, mc_mean)
    @printf("  Var(S)        %12.4f      %12.4f\n", analytical_var,  mc_var)
    println("-"^55)
    @printf("  Rel. err E[S]   %.6f\n", abs(mc_mean - analytical_mean) / analytical_mean)
    @printf("  Rel. err Var(S) %.6f\n", abs(mc_var  - analytical_var)  / analytical_var)
    println("="^55)

    return mc_mean, mc_var, analytical_mean, analytical_var
end

# ── Run for several (n, p) combinations ──────────────────────────────────────
println("\nTesting Var(S) = 2p·n²·(n-1) for Mahalanobis distance sum\n")

configs = [(5, 2), (10, 4), (20, 6), (8, 10)]

for (n, p) in configs
    run_test(n=n, p=p, n_sim=50_000)
    println()
end