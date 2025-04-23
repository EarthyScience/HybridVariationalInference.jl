module HybridVariationalInferenceSimpleChainsExt

using HybridVariationalInference, SimpleChains
using HybridVariationalInference: HybridVariationalInference as HVI
using StatsFuns: logistic
using ComponentArrays: ComponentArrays as CA
using Random

struct SimpleChainsApplicator{MT} <: AbstractModelApplicator
    m::MT
end

function HVI.construct_ChainsApplicator(rng::AbstractRNG, m::SimpleChain, FloatType=Float32) 
    ϕ = SimpleChains.init_params(m, FloatType; rng);
    SimpleChainsApplicator(m), ϕ
end

HVI.apply_model(app::SimpleChainsApplicator, x, ϕ) = app.m(x, ϕ)

function HVI.construct_3layer_MLApplicator(
    rng::AbstractRNG, prob::HVI.AbstractHybridProblem, ::Val{:SimpleChains};
    scenario::NTuple = ())
    n_covar = get_hybridproblem_n_covar(prob; scenario)
    n_pbm_covars = length(get_hybridproblem_pbmpar_covars(prob; scenario))
    n_input = n_covar + n_pbm_covars
    FloatType = get_hybridproblem_float_type(prob; scenario)
    (;θM) = get_hybridproblem_par_templates(prob; scenario)
    n_out = length(θM)
    is_using_dropout = :use_dropout ∈ scenario
    g_chain = if is_using_dropout
        SimpleChain(
            static(n_input), # input dimension (optional)
            # dense layer with bias that maps to 8 outputs and applies `tanh` activation
            TurboDense{true}(tanh, n_input * 4),
            SimpleChains.Dropout(0.2), # dropout layer
            TurboDense{true}(tanh, n_input * 4),
            SimpleChains.Dropout(0.2),
            # dense layer without bias that maps to n outputs and `logistic` activation
            TurboDense{false}(logistic, n_out)
        )
    else
        SimpleChain(
            static(n_input), # input dimension (optional)
            # dense layer with bias that maps to 8 outputs and applies `tanh` activation
            TurboDense{true}(tanh, n_input * 4),
            TurboDense{true}(tanh, n_input * 4),
            # dense layer without bias that maps to n outputs and `logistic` activation
            TurboDense{false}(logistic, n_out)
        )
    end
    construct_ChainsApplicator(rng, g_chain, FloatType)
end

end # module
