# How to specify custom Penalties


``` @meta
CurrentModule = HybridVariationalInference  
```

This guide shows how the user can specify a customized penalties to help
the solver to converge to global minimum.

## Motivation

The basic cost in HVI is the negative log of the joint probability, i.e.
the likelihood of the observations given the parameters \* prior probability
of the parameters.

Sometimes there is additional knowledge not encoded in the prior, such as
one parameter must be larger than another, or entropy-weights of the
ML-parameters, and the solver accept a function to add additional loss terms.
The loglikelihood function assigns a cost to the mismatch between predictions and
observations. This often needs to be customized to the specific inversion.

This guide walks through the specification of such additional penalties.

First load necessary packages.

``` julia
using HybridVariationalInference
using SimpleChains
using ComponentArrays: ComponentArrays as CA
using JLD2
```

This tutorial reuses and modifies the fitted object saved at the end of the
[Basic workflow without GPU](@ref) tutorial, that used a log-Likelihood
function assuming observation error to be distributed independently normal.

``` julia
fname = "intermediate/basic_cpu_results.jld2"
print(abspath(fname))
prob = probo_normal = load(fname, "probo");
```

## Write function to compute the penalty loss

The function signature corresponds to the one described in [`apply_penalty_computer`](@ref).

In this example we want to avoid local minima when parameter, `r1`, is larger than
70% of the maximum observation.

``` julia
function compute_penalty_r1(y_pred::AbstractMatrix, addq_pred::AbstractMatrix, 
            θMs_tr::AbstractMatrix, θP::AbstractVector, 
            y_obs::AbstractMatrix, i_sites,
            ϕg, ϕq::AbstractVector)
    # compute the maximum of observed rates at each site
    y_obs_max = map(col -> maximum(x -> isfinite(x) ? x : zero(x), col), 
        eachcol(y_obs))
    # add a penalty if r1 is larger than 0.95 times the maximum
    sum(max.(zero(eltype(θMs_tr)), θMs_tr[:,:r1] .- 0.95 .* y_obs_max))
end
```

## Update the problem and redo the inversion

HybridProblem has keyword argument `penalty_computer` to specify the Callable
that computes the penalty. It defaults to `ZeroPenaltyComputer`, which
returns zero penalty cost.

Here we construct a [`CustomPenaltyComputer`](@ref) with the function specified
above and update the problem.

``` julia
penalty_computer = CustomPenaltyComputer(compute_penalty_r1)
prob_pen = HybridProblem(prob; penalty_computer)

using OptimizationOptimisers
import Zygote
# silence warning of no GPU backend found (because we did not import CUDA here)
ENV["MLDATADEVICES_SILENCE_WARN_NO_GPU"] = 1

# first run a few iterators with updating only optimizing the mean
solver_point = HybridPointSolver(; alg=Adam(0.02))
(; probo) = solve(prob_pen, solver_point; 
    callback = callback_loss(100), # output during fitting
    epochs = 5,
); probo_pen_point = probo;

# starting from this, also estimate the posterior uncertainty parameters
solver = HybridPosteriorSolver(; alg=Adam(0.02), n_MC=3)
(; probo) = solve(probo_pen_point, solver; 
    callback = callback_loss(100), # output during fitting
    epochs = 5,
);
```

## Writing a customized PenaltyComputer

In the above example, the maximum of the observations in the batch
are recomputed each time, the PenaltyComputer is called.

This can be avoided, because the function receives argument, `i_sites`,
which can be used to index precomputed observation maxima, stored
in a struct implementing type `AbstractPenaltyComputer`
and function `apply_penalty_computer`.

``` julia
struct R1PenaltyComputer{T} <: AbstractPenaltyComputer where T
    ys_max::Vector{T}
end
function R1PenaltyComputer(ys::AbstractMatrix)
  ys_max = vec(maximum(ys; dims = 1))
  R1PenaltyComputer(ys_max)
end
function HybridVariationalInference.apply_penalty_computer(
    pc::R1PenaltyComputer,
    y_pred::AbstractMatrix, addq_pred::AbstractMatrix, θMs_tr::AbstractMatrix, θP::AbstractVector, 
    y_obs::AbstractMatrix, i_sites, 
    ϕg, ϕq::AbstractVector
    )
    y_obs_max = pc.ys_max[i_sites]
    #@assert y_obs_max == map(col -> maximum(x -> isfinite(x) ? x : zero(x), col), eachcol(y_obs))
    # add a penalty if r1 is larger than 0.95 times the maximum
    sum(max.(zero(eltype(θMs_tr)), θMs_tr[:,:r1] .- 0.95 .* y_obs_max))
end

ys = get_hybridproblem_train_dataloader(probo).data[3]
penalty_computer = R1PenaltyComputer(ys)
```

Rerunning the inversion using with the update PenaltyComputer:

``` julia
prob_pen = HybridProblem(probo; penalty_computer)
(; probo) = solve(prob_pen, solver; 
    callback = callback_loss(100), # output during fitting
    epochs = 5,
);
```
