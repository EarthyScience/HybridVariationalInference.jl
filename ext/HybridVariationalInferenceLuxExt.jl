module HybridVariationalInferenceLuxExt 

using HybridVariationalInference, Lux
using HybridVariationalInference: HybridVariationalInference as HVI
using ComponentArrays: ComponentArrays as CA
using Random
using StatsFuns: logistic
import Functors
import GPUArraysCore

# AbstractModelApplicator that stores a Lux.StatefulLuxLayer, so that
# it can be applied with given inputs and parameters
# The `int_ϕ` ComponentArrayInterpreter, attaches the correct axes to the
# supplied parameters, that do not need to keep the Axis information
struct LuxApplicator{MT1, MT2, IT} <: AbstractModelApplicator 
    stateful_layer_test::MT1
    stateful_layer_train::MT2
    int_ϕ::IT
    is_testmode::Bool
end

Functors.@functor LuxApplicator ()  # prevent warning for putting stateful_layers to GPU

function HVI.construct_ChainsApplicator(rng::AbstractRNG, m::Chain, float_type=Float32; device = gpu_device()) 
    ps, st = Lux.setup(rng, m)
    ps_ca = float_type.(CA.ComponentArray(ps)) 
    st = st |> device
    stateful_layer_train = StatefulLuxLayer{true}(m, nothing, st)
    stateful_layer_test = Lux.testmode(stateful_layer_train)
    int_ϕ = get_concrete(ComponentArrayInterpreter(ps_ca))
    LuxApplicator(stateful_layer_test, stateful_layer_train, int_ϕ, false), ps_ca
end

function HVI.apply_model(app::LuxApplicator, x, ϕ; is_testmode=false) 
    ϕd = CA.getdata(ϕ)
    if (ϕ isa SubArray) && (ϕ.parent isa GPUArraysCore.AbstractGPUArray)
        # Lux has problems with SubArrays of GPUArrays, need to convert to plain Array
        ϕc = app.int_ϕ(ϕd[:])
    else
        ϕc = app.int_ϕ(ϕd)
    end
    # tmp(x, ϕc)
    if is_testmode
        app.stateful_layer_test(x, ϕc)
    else
        app.stateful_layer_train(x, ϕc)
    end
end



function HVI.construct_3layer_MLApplicator(
        rng::AbstractRNG, prob::HVI.AbstractHybridProblem, ::Val{:Lux};
        scenario::Val{scen},
        p_dropout = 0.2) where scen
    (;θM) = get_hybridproblem_par_templates(prob; scenario)
    n_out = length(θM)
    n_covar = get_hybridproblem_n_covar(prob; scenario)
    n_pbm_covars = length(get_hybridproblem_pbmpar_covars(prob; scenario))
    n_input = n_covar + n_pbm_covars
    #(; n_covar, n_θM) = get_hybridproblem_sizes(prob; scenario)
    float_type = get_hybridproblem_float_type(prob; scenario)
    is_using_dropout = :use_dropout ∈ scen
    g_chain = if is_using_dropout 
        Lux.Chain(
            # dense layer with bias that maps to 8 outputs and applies `tanh` activation
            Lux.Dense(n_input => n_input * 4, tanh),
            Lux.Dropout(p_dropout),
            Lux.Dense(n_input * 4 => n_input * 4, tanh),
            Lux.Dropout(p_dropout),
            # dense layer without bias that maps to n outputs and `logistic` activation
            Lux.Dense(n_input * 4 => n_out, logistic, use_bias = false)
        )
    else
        Lux.Chain(
            # dense layer with bias that maps to 8 outputs and applies `tanh` activation
            Lux.Dense(n_input => n_input * 4, tanh),
            Lux.Dense(n_input * 4 => n_input * 4, tanh),
            # dense layer without bias that maps to n outputs and `logistic` activation
            Lux.Dense(n_input * 4 => n_out, logistic, use_bias = false)
        )
    end
    construct_ChainsApplicator(rng, g_chain, float_type)
end


# function HVI.HybridProblem(rng::AbstractRNG, 
#     θP::CA.ComponentVector, θM::CA.ComponentVector, g_chain::Chain, 
#     args...; device = gpu_device(), kwargs...)
#     # constructor with SimpleChain
#     g, ϕg = construct_ChainsApplicator(rng, g_chain, eltype(θM); device)
#     HybridProblem(θP, θM, g, ϕg, args...; kwargs...)
# end

end # module
