module HybridVariationalInferenceSimpleChainsExt 

using HybridVariationalInference, SimpleChains
using HybridVariationalInference: HybridVariationalInference as HVI

struct SimpleChainsApplicator{MT} <: AbstractModelApplicator 
    m::MT
end

HVI.construct_SimpleChainsApplicator(m::SimpleChain) = SimpleChainsApplicator(m)

HVI.apply_model(app::SimpleChainsApplicator, x, ϕ) = app.m(x, ϕ)

end # module
