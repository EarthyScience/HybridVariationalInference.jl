module HybridVariationalInference

using ComponentArrays: ComponentArrays as CA
using Random
using StatsBase # fit ZScoreTransform
using Combinatorics # gen_hybridcase_synthetic/combinations
using GPUArraysCore
using LinearAlgebra
using CUDA
using ChainRulesCore
using TransformVariables
using Zygote  # Zygote.@ignore CUDA.randn
using BlockDiagonals

export inverse_ca
include("util._transformvariablesjl")

export ComponentArrayInterpreter, flatten1, get_concrete
include("ComponentArrayInterpreter.jl")

export AbstractModelApplicator, construct_SimpleChainsApplicator, construct_FluxApplicator,
       construct_LuxApplicator
include("ModelApplicator.jl")

export AbstractHybridCase, gen_hybridcase_MLapplicator, gen_hybridcase_PBmodel, get_hybridcase_sizes, get_hybridcase_FloatType, gen_hybridcase_synthetic,
       get_hybridcase_par_templates, gen_cov_pred
include("hybrid_case.jl")

export applyf, gf, get_loss_gf
include("gf.jl")

export compute_correlated_covars, scale_centered_at
include("gencovar.jl")

export callback_loss
include("util_opt.jl")

#export - all internal
include("cholesky.jl")

include("elbo.jl")

export DoubleMM
include("DoubleMM/DoubleMM.jl")

end
