using HybridVariationalInference
using Test
using SimpleChains
using StatsFuns: logistic

@testset "SimpleChainsModelApplicator" begin
    n_covar = 5
    n_out = 2
    g_chain = SimpleChain(
        static(n_covar), # input dimension (optional)
        TurboDense{true}(tanh, n_covar * 4),
        TurboDense{true}(tanh, n_covar * 4),
        TurboDense{false}(logistic, n_out)
    )
    g = construct_SimpleChainsApplicator(g_chain)
    n_site = 3
    x = rand(n_covar, n_site)
    ϕ = SimpleChains.init_params(g_chain);
    y = g(x, ϕ)
    @test size(y) == (n_out, n_site)
end;
