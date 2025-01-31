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
using CommonSolve
#using OptimizationOptimisers # default alg=Adam(0.02)
using Optimization

export ComponentArrayInterpreter, flatten1, get_concrete
include("ComponentArrayInterpreter.jl")

export AbstractModelApplicator, construct_ChainsApplicator
export construct_3layer_MLApplicator, select_ml_engine
include("ModelApplicator.jl")

export AbstractGPUDataHandler, NullGPUDataHandler, get_default_GPUHandler
include("GPUDataHandler.jl")

export AbstractHybridProblem, get_hybridproblem_MLapplicator, get_hybridproblem_PBmodel, 
        get_hybridproblem_float_type, gen_hybridcase_synthetic,
       get_hybridproblem_par_templates, get_hybridproblem_transforms, get_hybridproblem_train_dataloader,
       get_hybridproblem_neg_logden_obs, 
       get_hybridproblem_n_covar, 
       #update,
       gen_cov_pred
include("AbstractHybridProblem.jl")

export HybridProblem
include("HybridProblem.jl")

export applyf, gf, get_loss_gf
include("gf.jl")

export compute_correlated_covars, scale_centered_at
include("gencovar.jl")

export callback_loss
include("util_opt.jl")

export cpu_ca
include("util_ca.jl")

export neg_logden_indep_normal, entropy_MvNormal
include("logden_normal.jl")

export get_ca_starts
include("cholesky.jl")

export neg_elbo_transnorm_gf, predict_gf
include("elbo.jl")

export init_hybrid_params
include("init_hybrid_params.jl")

export AbstractHybridSolver, HybridPointSolver, HybridPosteriorSolver
include("HybridSolver.jl")


export DoubleMM
include("DoubleMM/DoubleMM.jl")

end
