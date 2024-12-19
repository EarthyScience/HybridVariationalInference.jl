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

end # module
