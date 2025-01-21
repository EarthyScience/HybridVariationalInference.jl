module HybridVariationalInferenceLuxExt 

using HybridVariationalInference, Lux
using HybridVariationalInference: HybridVariationalInference as HVI
using ComponentArrays: ComponentArrays as CA
using Random

struct LuxApplicator{MT, IT} <: AbstractModelApplicator 
    stateful_layer::MT
    int_ϕ::IT
end

function HVI.construct_LuxApplicator(m::Chain; device = gpu_device()) 
    ps, st = Lux.setup(Random.default_rng(), m)
    ps_ca = CA.ComponentArray(ps) 
    st = st |> device
    stateful_layer = StatefulLuxLayer{true}(m, nothing, st)
    #stateful_layer(x_o_gpu[:, 1:n_site_batch], ps_ca)
    int_ϕ = get_concrete(ComponentArrayInterpreter(ps_ca))
    LuxApplicator(stateful_layer, int_ϕ)
end

function HVI.apply_model(app::LuxApplicator, x, ϕ) 
    ϕc = app.int_ϕ(ϕ)
    app.stateful_layer(x, ϕc)
end

function HVI.HybridProblem(θP::CA.ComponentVector, θM::CA.ComponentVector, g_chain::Chain, 
    args...; kwargs...)
    # constructor with SimpleChain
    g = construct_LuxApplicator(g_chain)
    FT = eltype(θM)
    ϕg = randn(FT, length(g.int_ϕ))
    HybridProblem(θP, θM, g, ϕg, args...; kwargs...)
end

end # module
