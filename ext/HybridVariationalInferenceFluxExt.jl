module HybridVariationalInferenceFluxExt

using HybridVariationalInference, Flux
using HybridVariationalInference: HybridVariationalInference as HVI

struct FluxApplicator{RT} <: AbstractModelApplicator
    rebuild::RT
end

function HVI.construct_FluxApplicator(m::Chain)
    _, rebuild = destructure(m)
    FluxApplicator(rebuild)
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

function HVI.get_hybridcase_MLapplicator(case::HVI.DoubleMM.DoubleMMCase, ::Val{:Flux};
        scenario::NTuple = ())
    (; n_covar, n_θM) = get_hybridcase_sizes(case; scenario)
    FloatType = get_hybridcase_FloatType(case; scenario)
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
    ϕ, _ = destructure(g_chain)
    construct_FluxApplicator(g_chain), ϕ
end

end # module
