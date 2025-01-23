module HybridVariationalInference

using ComponentArrays: ComponentArrays as CA
using Random
using StatsBase # fit ZScoreTransform
using Combinatorics # gen_hybridcase_synthetic/combinations
using GPUArraysCore
using LinearAlgebra
using CUDA
using ChainRulesCore
using Bijectors
using Zygote  # Zygote.@ignore CUDA.randn
using BlockDiagonals
using MLUtils  # dataloader

export ComponentArrayInterpreter, flatten1, get_concrete
include("ComponentArrayInterpreter.jl")

export AbstractModelApplicator, construct_SimpleChainsApplicator, construct_FluxApplicator,
       construct_LuxApplicator
include("ModelApplicator.jl")

export AbstractGPUDataHandler, NullGPUDataHandler, get_default_GPUHandler
include("GPUDataHandler.jl")

export AbstractHybridCase, get_hybridcase_MLapplicator, get_hybridcase_PBmodel, get_hybridcase_sizes, get_hybridcase_FloatType, gen_hybridcase_synthetic,
       get_hybridcase_par_templates, get_hybridcase_transforms, get_hybridcase_train_dataloader,
       gen_cov_pred
include("hybrid_case.jl")

export HybridProblem
include("HybridProblem.jl")

export applyf, gf, get_loss_gf
include("gf.jl")

export compute_correlated_covars, scale_centered_at
include("gencovar.jl")

export callback_loss
include("util_opt.jl")

export neg_logden_indep_normal, entropy_MvNormal
include("logden_normal.jl")

export get_ca_starts
include("cholesky.jl")

export neg_elbo_transnorm_gf, predict_gf
include("elbo.jl")

export init_hybrid_params
include("init_hybrid_params.jl")

export DoubleMM
include("DoubleMM/DoubleMM.jl")

end
