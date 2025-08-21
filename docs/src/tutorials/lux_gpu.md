# How to move computations to GPU


``` @meta
CurrentModule = HybridVariationalInference  
```

This guide shows how to configure the setup and inversion of a HybridProblem so
that computations of the ML model and maybe also the process-based model
are executed on GPU.

## Motivation

Machine learning is often accellerated by moving computations form CPU
to GPU. So does HVI.

First load necessary packages.

``` julia
using HybridVariationalInference
using ComponentArrays: ComponentArrays as CA
using Bijectors
using Lux
using SimpleChains # only loading save object
using StatsFuns
using StableRNGs
using MLUtils
using JLD2
using Random
# using CairoMakie
# using PairPlots   # scatterplot matrices
```

This tutorial reuses and modifies the fitted object saved at the end of the
[Basic workflow without GPU](@ref) tutorial.

``` julia
fname = "intermediate/basic_cpu_results.jld2"
print(abspath(fname))
prob = probo_chain = load(fname, "probo");
```

## Updating the ML model of the problem to use LUX

Because the SimpleChains ML model used in the basic tutorial does not support
GPU, we reconstruct the model using the LUX framework.
Note that all the setup is almost the same, as in the basic worklfow. The
only diffrence is that a `Lux.Chains` object is provided to `construct_ChainsApplicator`.

``` julia
n_out = length(prob.θM) # number of individuals to predict 
n_covar = 5 #size(xM,1)
n_input = n_covar 

g_lux = Lux.Chain(
    Lux.Dense(n_covar => n_covar * 4, tanh),
    Lux.Dense(n_covar * 4 => n_covar * 4, tanh),
    Lux.Dense(n_covar * 4 => n_out, logistic, use_bias = false)
)
# get a template of the parameter vector, ϕg0
rng = StableRNG(111)
g_chain_app, ϕg0 = construct_ChainsApplicator(rng, g_lux)
#
priorsM = [prob.priors[k] for k in keys(prob.θM)]
lowers, uppers = get_quantile_transformed(priorsM, prob.transM)
FT = eltype(prob.θM)
g_chain_scaled = NormalScalingModelApplicator(g_chain_app, lowers, uppers, FT)
```

Update the `HybridProblem` to use this ML model.

``` julia
prob_lux = HybridProblem(probo_chain; g=g_chain_scaled, ϕg=ϕg0)
```

## Specifying GPU devices during solve

The [`solve`](@ref) method for the HybridPosteriorSolver accepts argument `gdevs`,
Its a `NamedTuple` with entries`gdev_M` and `gdev_P`, for the
ML model on and the process-basee model (PBM) respectively.
They specify functions that are applied to move callables and data to GPU.

They default to `identity`, meaning that nothing is moved from CPU to GPU.
Function `gpu_device()` from package `MLDataDevices` can be used instead
for teh standard GPU device.

Hence specify
- `gdevs = (; gdev_M=gpu_device(), gdev_P=gpu_device())`: to move both ML model and PBM to GPU
- `gdevs = (; gdev_M=gpu_device(), gdev_P=identity)`: to move both ML model to GPU but execute the PBM (and parameter transformation) on CPU

In addition, the libraries of the GPU device need to be activated by
importing respective Julia packages.
Currently, only CUDA is tested with this `HybridVariationalInference` package.

``` julia
import CUDA, cuDNN # so that gpu_device() returns a CUDADevice
#CUDA.device!(4)
gdevs = (; gdev_M=gpu_device(), gdev_P=gpu_device())
#gdevs = (; gdev_M=gpu_device(), gdev_P=identity)

using OptimizationOptimisers
import Zygote
solver = HybridPosteriorSolver(; alg=Adam(0.02), n_MC=3)

(; probo) = solve(prob_lux, solver; 
    callback = callback_loss(100), 
    epochs = 10,
    gdevs,
); probo_lux = probo;
```

## Moving results from GPU to CPU

The sampling and prediction methods, also take this `gdevs` keyword argument.

``` julia
n_sample_pred = 400
(y_dev, θsP_dev, θsMs_dev) = (; y, θsP, θsMs) = predict_hvi(
  rng, probo_lux; n_sample_pred, gdevs);
```

If `gdev_P` is not an `AbstractGPUDevice` then all the results are on CPU.
If `gdev_P` is an `AbstractGPUDevice` then the results are GPUArrays
and need to be transferred to CPU.

``` julia
typeof(θsMs_dev)
```

    ComponentArrays.ComponentArray{Float32, 3, CUDA.CuArray{Float32, 3, CUDA.DeviceMemory}, Tuple{ComponentArrays.Axis{(i = 1:800,)}, ComponentArrays.Axis{(r1 = 1, K1 = 2)}, ComponentArrays.Axis{(i = 1:400,)}}}

Handling of a `ComponentArrays` backed by GPUArrays can result
in errors of scalar indexing. Therefore, use a semicolon
to suppress printing.
Also for moving the `ComponentArrays` to CPU, use function
[`apply_preserve_axes`](@ref) to circumvent this error.

``` julia
cdev = cpu_device()
y = cdev(y_dev)
θsP = apply_preserve_axes(cdev, θsP_dev)
θsMs = apply_preserve_axes(cdev, θsMs_dev)
```
