module HybridVariationalInferenceFluxExt

using HybridVariationalInference, Flux
using HybridVariationalInference: HybridVariationalInference as HVI
using ComponentArrays: ComponentArrays as CA
using Random
using StatsFuns: logistic

struct FluxApplicator{RT} <: AbstractModelApplicator
    rebuild::RT
end

struct PartricFluxApplicator{RT, MT, YT} <: AbstractModelApplicator
    rebuild::RT
end

const FluxApplicatorU{RT} = Union{FluxApplicator{RT},PartricFluxApplicator{RT}} where RT


function HVI.construct_ChainsApplicator(rng::AbstractRNG, m::Chain, float_type::DataType)
    # TODO: care fore rng and float_type
    ϕ, rebuild = Flux.destructure(m)
    FluxApplicator(rebuild), ϕ
end

function HVI.apply_model(app::FluxApplicator, x::T, ϕ) where T
    # assume no size informmation in x -> can hint the type of the result
    # to be the same as the type of the input
    m = app.rebuild(ϕ)
    res = m(x)::T
    res
end


function HVI.construct_partric(app::FluxApplicator{RT}, x, ϕ) where RT
    m = app.rebuild(ϕ)
    y = m(x)
    PartricFluxApplicator{RT, typeof(m), typeof(y)}(app.rebuild)
end

function HVI.apply_model(app::PartricFluxApplicator{RT, MT, YT}, x, ϕ) where {RT, MT, YT}
    m = app.rebuild(ϕ)::MT
    res = m(x)::YT
    res
end

# struct FluxGPUDataHandler <: AbstractGPUDataHandler end
# HVI.handle_GPU_data(::FluxGPUDataHandler, x::AbstractArray) = cpu(x)

# function __init__()
#     #@info "HybridVariationalInference: setting FluxGPUDataHandler"
#     HVI.set_default_GPUHandler(FluxGPUDataHandler())
# end

# function HVI.HybridProblem(θP::CA.ComponentVector, θM::CA.ComponentVector, g_chain::Flux.Chain, 
#     args...; kwargs...)
#     # constructor with Flux.Chain
#     g, ϕg = construct_FluxApplicator(g_chain)
#     HybridProblem(θP, θM, g, ϕg, args...; kwargs...)
# end

function HVI.construct_3layer_MLApplicator(
        rng::AbstractRNG, prob::HVI.AbstractHybridProblem, ::Val{:Flux};
        scenario::Val{scen}) where scen
    (;θM) = get_hybridproblem_par_templates(prob; scenario)
    n_out = length(θM)
    n_covar = get_hybridproblem_n_covar(prob; scenario)
    n_pbm_covars = length(get_hybridproblem_pbmpar_covars(prob; scenario))
    n_input = n_covar + n_pbm_covars
    #(; n_covar, n_θM) = get_hybridproblem_sizes(prob; scenario)
    float_type = get_hybridproblem_float_type(prob; scenario)
    is_using_dropout = :use_dropout ∈ scen
    is_using_dropout && error("dropout scenario not supported with Flux yet.")
    g_chain = Flux.Chain(
        # dense layer with bias that maps to 8 outputs and applies `tanh` activation
        Flux.Dense(n_input => n_input * 4, tanh),
        Flux.Dense(n_input * 4 => n_input * 4, tanh),
        # dense layer without bias that maps to n outputs and `logistic` activation
        Flux.Dense(n_input * 4 => n_out, logistic, bias = false)
    )
    construct_ChainsApplicator(rng, g_chain, float_type)
end

function HVI.cpu_ca(ca::CA.ComponentArray)
    CA.ComponentArray(cpu(CA.getdata(ca)), CA.getaxes(ca))
end





end # module
