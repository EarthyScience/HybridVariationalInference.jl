module HybridVariationalInference

using ComponentArrays: ComponentArrays as CA
using Random
using StatsBase # fit ZScoreTransform
using Combinatorics # gen_cov_pred/combinations

export ComponentArrayInterpreter, flatten1
include("ComponentArrayInterpreter.jl")

export AbstractModelApplicator, construct_SimpleChainsApplicator, construct_FluxApplicator,
       construct_LuxApplicator
include("ModelApplicator.jl")

export AbstractHybridCase, gen_g, gen_f, get_case_sizes, get_case_FloatType, gen_cov_pred
export applyf, gf, get_loss_gf
include("gf.jl")

export compute_correlated_covars, scale_centered_at
include("gencovar.jl")

export callback_loss
include("util_opt.jl")

export DoubleMM
include("DoubleMM/DoubleMM.jl")

end
