module HybridVariationalInference

using ComponentArrays: ComponentArrays as CA
using Random
using StatsBase # fit ZScoreTransform

export ComponentArrayInterpreter, flatten1
include("ComponentArrayInterpreter.jl")

export AbstractModelApplicator, construct_SimpleChainsApplicator, construct_FluxApplicator,
       construct_LuxApplicator
include("ModelApplicator.jl")

export applyf, gf, get_loss_gf
include("gf.jl")

export compute_correlated_covars, scale_centered_at
include("gencovar.jl")

export callback_loss
include("util_opt.jl")

include("DoubleMM/DoubleMM.jl")

end
