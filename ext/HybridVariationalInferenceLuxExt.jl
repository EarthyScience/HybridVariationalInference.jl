module HybridVariationalInferenceLuxExt 

using HybridVariationalInference, Lux
using HybridVariationalInference: HybridVariationalInference as HVI
using ComponentArrays: ComponentArrays as CA
using Random
using StatsFuns: logistic



struct LuxApplicator{MT, IT} <: AbstractModelApplicator 
    stateful_layer::MT
    int_ϕ::IT
end

function HVI.construct_ChainsApplicator(rng::AbstractRNG, m::Chain, float_type=Float32; device = gpu_device()) 
    ps, st = Lux.setup(rng, m)
    ps_ca = float_type.(CA.ComponentArray(ps)) 
    st = st |> device
    stateful_layer = StatefulLuxLayer{true}(m, nothing, st)
    #stateful_layer(x_o_gpu[:, 1:n_site_batch], ps_ca)
    int_ϕ = get_concrete(ComponentArrayInterpreter(ps_ca))
    LuxApplicator(stateful_layer, int_ϕ), ps_ca
end

function HVI.apply_model(app::LuxApplicator, x, ϕ) 
    ϕc = app.int_ϕ(ϕ)
    app.stateful_layer(x, ϕc)
end

# function HVI.HybridProblem(rng::AbstractRNG, 
#     θP::CA.ComponentVector, θM::CA.ComponentVector, g_chain::Chain, 
#     args...; device = gpu_device(), kwargs...)
#     # constructor with SimpleChain
#     g, ϕg = construct_ChainsApplicator(rng, g_chain, eltype(θM); device)
#     HybridProblem(θP, θM, g, ϕg, args...; kwargs...)
# end

end # module
