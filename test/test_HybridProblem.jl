using Test
using HybridVariationalInference
using StableRNGs
using Random
using Statistics
using ComponentArrays: ComponentArrays as CA
using Bijectors

using SimpleChains
using MLUtils
import Zygote

using OptimizationOptimisers

const MLengine = Val(nameof(SimpleChains))

construct_problem = () -> begin
    θP = CA.ComponentVector{Float32}(r0 = 0.3, K2 = 2.0)
    θM = CA.ComponentVector{Float32}(r1 = 0.5, K1 = 0.2)
    transP = elementwise(exp)
    transM = Stacked(elementwise(identity), elementwise(exp))
    n_covar = 5
    n_batch = 10
    int_θdoubleMM = get_concrete(ComponentArrayInterpreter(
        flatten1(CA.ComponentVector(; θP, θM))))
    function f_doubleMM(θ::AbstractVector, x)
        # extract parameters not depending on order, i.e whether they are in θP or θM
        θc = int_θdoubleMM(θ)
        r0, r1, K1, K2 = θc[(:r0, :r1, :K1, :K2)]
        y = r0 .+ r1 .* x.S1 ./ (K1 .+ x.S1) .* x.S2 ./ (K2 .+ x.S2)
        return (y)
    end
    function f_doubleMM_with_global(θP::AbstractVector, θMs::AbstractMatrix, x)
        pred_sites = applyf(f_doubleMM, θMs, θP, x)
        pred_global = eltype(pred_sites)[]
        return pred_global, pred_sites
    end
    n_out = length(θM)
    g_chain = SimpleChain(
        static(n_covar), # input dimension (optional)
        # dense layer with bias that maps to 8 outputs and applies `tanh` activation
        TurboDense{true}(tanh, n_covar * 4),
        TurboDense{true}(tanh, n_covar * 4),
        # dense layer without bias that maps to n outputs and `identity` activation
        TurboDense{false}(identity, n_out)
    )
    # g, ϕg = construct_SimpleChainsApplicator(g_chain)
    #
    rng = StableRNG(111)
    (; xM, n_site, θP_true, θMs_true, xP, y_global_true, y_true, y_global_o, y_o
) = gen_hybridcase_synthetic(DoubleMM.DoubleMMCase(), rng;)
    train_loader = MLUtils.DataLoader((xM, xP, y_o), batchsize = n_batch)
    # HybridProblem(θP, θM, transM, transP, n_covar, n_batch, f_doubleMM_with_global, 
    #     g, ϕg, train_loader)
    HybridProblem(θP, θM, g_chain, f_doubleMM_with_global, 
        transM, transP, n_covar, n_batch, train_loader)
end
prob = construct_problem();
scenario = (:default,)

#(; n_covar, n_batch, n_θM, n_θP) = get_hybridcase_sizes(prob; scenario)

@testset "loss_gf" begin
    #----------- fit g and θP to y_o
    g, ϕg0 = get_hybridcase_MLapplicator(prob, MLengine; scenario)
    train_loader = get_hybridcase_train_dataloader(prob; scenario)
    (xM, xP, y_o) = first(train_loader)
    f = get_hybridcase_PBmodel(prob; scenario)
    par_templates = get_hybridcase_par_templates(prob; scenario)

    int_ϕθP = ComponentArrayInterpreter(CA.ComponentVector(
        ϕg = 1:length(ϕg0), θP = par_templates.θP))
    p = p0 = vcat(ϕg0, par_templates.θP .* 0.8)  # slightly disturb θP_true

    # Pass the site-data for the batches as separate vectors wrapped in a tuple

    y_global_o = Float64[]
    loss_gf = get_loss_gf(g, f, y_global_o, int_ϕθP)
    l1 = loss_gf(p0, first(train_loader)...)
    gr = Zygote.gradient(p -> loss_gf(p, train_loader.data...)[1], p0)
    @test gr[1] isa Vector

    () -> begin
        optf = Optimization.OptimizationFunction((ϕ, data) -> loss_gf(ϕ, data...)[1],
            Optimization.AutoZygote())
        optprob = OptimizationProblem(optf, p0, train_loader)

        res = Optimization.solve(
            #        optprob, Adam(0.02), callback = callback_loss(100), maxiters = 1000);
            optprob, Adam(0.02), maxiters = 1000)

        l1, y_pred_global, y_pred, θMs_pred = loss_gf(res.u, train_loader.data...)
        @test isapprox(par_templates.θP, int_ϕθP(res.u).θP, rtol = 0.11)
    end
end
