module HybridVariationalInferenceSimpleChainsExt

using HybridVariationalInference, SimpleChains
using HybridVariationalInference: HybridVariationalInference as HVI
using StatsFuns: logistic

struct SimpleChainsApplicator{MT} <: AbstractModelApplicator
    m::MT
end

HVI.construct_SimpleChainsApplicator(m::SimpleChain) = SimpleChainsApplicator(m)

HVI.apply_model(app::SimpleChainsApplicator, x, ϕ) = app.m(x, ϕ)

function HVI.gen_g(case::HVI.DoubleMM.DoubleMMCase, ::Val{:SimpleChains};
        scenario::NTuple=())
    (;n_covar, n_θM) = get_case_sizes(case; scenario)
    FloatType = get_case_FloatType(case; scenario)
    n_out = n_θM
    is_using_dropout = :use_dropout ∈ scenario
    g_chain = if is_using_dropout
        SimpleChain(
            static(n_covar), # input dimension (optional)
            # dense layer with bias that maps to 8 outputs and applies `tanh` activation
            TurboDense{true}(tanh, n_covar * 4),
            SimpleChains.Dropout(0.2), # dropout layer
            TurboDense{true}(tanh, n_covar * 4),
            SimpleChains.Dropout(0.2),
            # dense layer without bias that maps to n outputs and `identity` activation
            TurboDense{false}(identity, n_out)
        )
    else
        SimpleChain(
            static(n_covar), # input dimension (optional)
            # dense layer with bias that maps to 8 outputs and applies `tanh` activation
            TurboDense{true}(tanh, n_covar * 4),
            TurboDense{true}(tanh, n_covar * 4),
            # dense layer without bias that maps to n outputs and `identity` activation
            TurboDense{false}(identity, n_out)
        )
    end
    ϕ = SimpleChains.init_params(g_chain, FloatType);
    SimpleChainsApplicator(g_chain), ϕ
end

end # module
