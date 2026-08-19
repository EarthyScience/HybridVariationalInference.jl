# Prompt: 
# Implement a WeightedDataLoader in Julia that wraps MLUtils.DataLoader and performs weighted resampling. The implementation should satisfy the following requirements:
#     Define a WeightedObsView{D} struct with fields data::D and weights::Vector{Float64}.
#     Implement MLUtils.numobs and MLUtils.getobs for WeightedObsView. In getobs, use StatsBase.sample with Weights to draw length(indices) weighted indices with replacement, then call getobs on the underlying data. Shield the sampling and data fetching from Zygote using ChainRulesCore.@ignore_derivatives. Return data as-is without any type conversion.
#     Define a WeightedDataLoader constructor that takes data, weights, and keyword arguments batchsize=32 and any extra kwargs forwarded to MLUtils.DataLoader. It should construct a WeightedObsView and wrap it in a MLUtils.DataLoader, returning a native MLUtils.DataLoader so it is directly recognized by Optimization.jl.
#     Write a @testset using SimpleChains, Optimisers, Optimization, OptimizationOptimisers, ChainRulesCore, MLUtils, and StatsBase that tests:
#         The loader is a native MLUtils.DataLoader
#         Correct batch shapes and types are preserved for Float32, Float64, Int arrays, NTuple, and NamedTuple datasets
#         Minority class oversampling works correctly
#         End-to-end training with Optimization.solve and OptimizationOptimisers.Adam reduces the loss, where the loss function receives a batch directly as p (not the loader) 
#         Add methods getindex and length WeightedObsView to work on the data field

"""
    WeightedObsView{D}

A view over a dataset that supports weighted resampling via `MLUtils.getobs`.
"""
struct WeightedObsView{D}
    data::D
    weights::Vector{Float64}
end

"""
    MLUtils.numobs(wov::WeightedObsView)

Return the number of observations in the underlying dataset.
"""
MLUtils.numobs(wov::WeightedObsView) = MLUtils.numobs(wov.data)

"""
    MLUtils.getobs(wov::WeightedObsView, indices)

Draw `length(indices)` weighted samples (with replacement) using `StatsBase.sample`
with `Weights`, then fetch those observations from the underlying dataset.
Both sampling and data fetching are shielded from Zygote via
`ChainRulesCore.@ignore_derivatives`.
"""
function MLUtils.getobs(wov::WeightedObsView, indices)
    n = length(indices)
    obs = ChainRulesCore.@ignore_derivatives begin
        w = Weights(wov.weights)
        sampled_indices = StatsBase.sample(1:MLUtils.numobs(wov.data), w, n; replace=true)
        MLUtils.getobs(wov.data, sampled_indices)
    end
    return obs
end

# ============================================================
# WeightedDataLoader constructor
# ============================================================

"""
    WeightedDataLoader(data, weights; batchsize=32, kwargs...)

Construct a weighted data loader that performs weighted resampling.

Wraps `data` in a `WeightedObsView` and then in a `MLUtils.DataLoader`,
returning a native `MLUtils.DataLoader` so it is directly recognised by
Optimization.jl.

# Arguments
- `data`: Any dataset compatible with `MLUtils.numobs` / `MLUtils.getobs`.
- `weights`: A `Vector{Float64}` of per-observation sampling weights.
- `batchsize`: Number of observations per batch (default: 32).
- `kwargs...`: Additional keyword arguments forwarded to `MLUtils.DataLoader`.
"""
function WeightedDataLoader(data, weights::AbstractVector; batchsize::Int=32, kwargs...)
    wov = WeightedObsView(data, Vector{Float64}(weights))
    return MLUtils.DataLoader(wov; batchsize=batchsize, kwargs...)
end


# ============================================================
# Base interface
# ============================================================

Base.length(wov::WeightedObsView)                        = length(wov.data)
Base.getindex(wov::WeightedObsView, i)                   = wov.data[i]
