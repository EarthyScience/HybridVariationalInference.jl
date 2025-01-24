module HybridVariationalInferenceFluxExt

using HybridVariationalInference, Flux
using HybridVariationalInference: HybridVariationalInference as HVI
using ComponentArrays: ComponentArrays as CA
using Random

struct FluxApplicator{RT} <: AbstractModelApplicator
    rebuild::RT
end

function HVI.construct_ChainsApplicator(rng::AbstractRNG, m::Chain, float_type::DataType)
    # TODO: care fore rng and float_type
    ϕ, rebuild = destructure(m)
    FluxApplicator(rebuild), ϕ
end

function HVI.apply_model(app::FluxApplicator, x, ϕ)
    m = app.rebuild(ϕ)
    m(x)
end

struct FluxGPUDataHandler <: AbstractGPUDataHandler end
HVI.handle_GPU_data(::FluxGPUDataHandler, x::AbstractArray) = cpu(x)

function __init__()
    #@info "HybridVariationalInference: setting FluxGPUDataHandler"
    HVI.set_default_GPUHandler(FluxGPUDataHandler())
end

# function HVI.HybridProblem(θP::CA.ComponentVector, θM::CA.ComponentVector, g_chain::Flux.Chain, 
#     args...; kwargs...)
#     # constructor with Flux.Chain
#     g, ϕg = construct_FluxApplicator(g_chain)
#     HybridProblem(θP, θM, g, ϕg, args...; kwargs...)
# end

function HVI.get_hybridcase_MLapplicator(rng::AbstractRNG, case::HVI.DoubleMM.DoubleMMCase, ::Val{:Flux};
        scenario::NTuple = ())
    (; n_covar, n_θM) = get_hybridcase_sizes(case; scenario)
    float_type = get_hybridcase_float_type(case; scenario)
    n_out = n_θM
    is_using_dropout = :use_dropout ∈ scenario
    is_using_dropout && error("dropout scenario not supported with Flux yet.")
    g_chain = Flux.Chain(
        # dense layer with bias that maps to 8 outputs and applies `tanh` activation
        Flux.Dense(n_covar => n_covar * 4, tanh),
        Flux.Dense(n_covar * 4 => n_covar * 4, tanh),
        # dense layer without bias that maps to n outputs and `identity` activation
        Flux.Dense(n_covar * 4 => n_out, identity, bias = false)
    )
    construct_ChainsApplicator(rng, g_chain, float_type)
end



end # module
