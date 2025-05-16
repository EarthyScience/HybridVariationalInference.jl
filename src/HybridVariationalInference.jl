module HybridVariationalInference

using ComponentArrays: ComponentArrays as CA
using Random
using StatsBase # fit ZScoreTransform
using StatsFuns # norminvcdf
using Combinatorics # gen_hybridproblem_synthetic/combinations
using GPUArraysCore
using LinearAlgebra
using MLDataDevices
#import CUDA #, cuDNN  # moved to HybridVariationalInferenceCUDAExt
using ChainRulesCore
using Bijectors
using BlockDiagonals
using MLUtils  # dataloader
using CommonSolve
#using OptimizationOptimisers # default alg=Adam(0.02)
using Optimization
using Distributions, DistributionFits
using StaticArrays: StaticArrays as SA
using Functors
using Test: Test # @inferred

export extend_stacked_nrow, StackedArray
#export Exp
include("bijectors_utils.jl")

export AbstractComponentArrayInterpreter, ComponentArrayInterpreter,
       StaticComponentArrayInterpreter
export flatten1, get_concrete, get_positions, stack_ca_int, compose_interpreters
export construct_partric
include("ComponentArrayInterpreter.jl")

export AbstractModelApplicator, construct_ChainsApplicator
export construct_3layer_MLApplicator, select_ml_engine
export NullModelApplicator, MagnitudeModelApplicator, NormalScalingModelApplicator
include("ModelApplicator.jl")

# export AbstractGPUDataHandler, NullGPUDataHandler, get_default_GPUHandler
# include("GPUDataHandler.jl")

export AbstractHybridProblem, get_hybridproblem_MLapplicator, get_hybridproblem_PBmodel,
       get_hybridproblem_ϕunc,
       get_hybridproblem_float_type, gen_hybridproblem_synthetic,
       get_hybridproblem_par_templates, get_hybridproblem_transforms,
       get_hybridproblem_train_dataloader,
       get_hybridproblem_neg_logden_obs,
       get_hybridproblem_n_covar,
       get_hybridproblem_n_site_and_batch,
       get_hybridproblem_cor_ends,
       get_hybridproblem_priors,
       get_hybridproblem_pbmpar_covars,
#update,
       gen_cov_pred,
       construct_dataloader_from_synthetic,
       gdev_hybridproblem_dataloader,
       setup_PBMpar_interpreter
include("AbstractHybridProblem.jl")

export AbstractHybridProblemInterpreters, HybridProblemInterpreters,
       get_int_P, get_int_M,
       get_int_Ms_batch, get_int_Ms_site, get_int_Mst_batch, get_int_Mst_site,
       get_int_PMs_batch, get_int_PMs_site, get_int_PMst_batch, get_int_PMst_site
include("hybridprobleminterpreters.jl")

export HybridProblem
export get_quantile_transformed
include("HybridProblem.jl")

export map_f_each_site, gf, get_loss_gf
include("gf.jl")

export compute_correlated_covars, scale_centered_at
include("gencovar.jl")

export callback_loss
include("util_opt.jl")

export cpu_ca
include("util_ca.jl")

export neg_logden_indep_normal, entropy_MvNormal
include("logden_normal.jl")

export get_ca_starts, get_ca_ends, get_cor_count
include("cholesky.jl")

export neg_elbo_gtf, predict_hvi
include("elbo.jl")

export init_hybrid_params, init_hybrid_ϕunc
include("init_hybrid_params.jl")

export AbstractHybridSolver, HybridPointSolver, HybridPosteriorSolver
include("HybridSolver.jl")

export DoubleMM
include("DoubleMM/DoubleMM.jl")

end
