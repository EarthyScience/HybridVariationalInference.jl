"""
Create `n_covar` correlated covariates 
from uncorrelated row-wise vector `x_pc`,
with correlations `rhos` to the linear combinations of `x_pc`.

By default correlations, `rhos = (1.0),0.88,0.78,0.69,0.61 ...`, 
decrease exponentially as `e^{-i/rhodec}`, with `rhodec = 8`.
"""
function compute_correlated_covars(rng::AbstractRNG, x_pc::AbstractMatrix{T};
    n_covar=size(x_pc, 1) + 3,
    rhodec=8,
    rhos=vcat(T(1.0), exp.(.-(1:(n_covar-1)) ./ T(rhodec)))) where {T}
    n_covar_pc, n_site = size(x_pc)
    A = rand(rng, T, n_covar, n_covar_pc)
    x_oc = (A * x_pc)
    # add noise to decorrelate
    rhoM = repeat(rhos, 1, n_site)
    noise = randn(rng, T, n_covar, n_site) .* T(0.2)
    x_o = rhoM .* x_oc .+ (1 .- rhoM) .* noise
    return x_o
end

"""
    scale_centered_at(x, m, σrel=1.0)
    scale_centered_at(x, m, σ)

Centers and rescales rows of matrix `x` around vector `m`. The scale can
either be given relative to `m` or specified as a vector of same size as `m`.
"""
function scale_centered_at(x::AbstractMatrix, m::AbstractVector, σrel::Real=1.0)
    σ = m .* σrel
    scale_centered_at(x, m, σ)
end
function scale_centered_at(x::AbstractMatrix, m::AbstractVector, σ::AbstractVector)
    dt = fit(ZScoreTransform, x, dims=2)
    x_unit_scaled = StatsBase.transform(dt, x)
    m .+  x_unit_scaled .* σ
end

