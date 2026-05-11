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

function HVI.apply_model(app::SimpleChainsApplicator, x, ϕ; is_testmode=false) 
    app.m(x, ϕ)
end

function HVI.construct_3layer_MLApplicator(
    rng::AbstractRNG, prob::HVI.AbstractHybridProblem, ::Val{:SimpleChains};
    scenario::Val{scen}) where scen
    (; n_input, n_output) = get_numberof_inputs_outputs(prob; scenario)
    float_type = get_hybridproblem_float_type(prob; scenario)
    is_using_dropout = :use_dropout ∈ scen
    g_chain = if is_using_dropout
        SimpleChain(
            static(n_input), # input dimension (optional)
            # dense layer with bias that maps to 8 outputs and applies `tanh` activation
            TurboDense{true}(tanh, n_input * 4),
            SimpleChains.Dropout(0.2), # dropout layer
            TurboDense{true}(tanh, n_input * 4),
            SimpleChains.Dropout(0.2),
            # dense layer without bias that maps to n outputs and `logistic` activation
            TurboDense{false}(logistic, n_output)
        )
    else
        SimpleChain(
            static(n_input), # input dimension (optional)
            # dense layer with bias that maps to 8 outputs and applies `tanh` activation
            TurboDense{true}(tanh, n_input * 4),
            TurboDense{true}(tanh, n_input * 4),
            # dense layer without bias that maps to n outputs and `logistic` activation
            TurboDense{false}(logistic, n_output)
        )
    end
    construct_ChainsApplicator(rng, g_chain, float_type)
end

end # module
