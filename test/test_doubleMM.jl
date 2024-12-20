using Test
using HybridVariationalInference
using StableRNGs
using Random
using Statistics
using ComponentArrays: ComponentArrays as CA

using SimpleChains
using MLUtils
import Zygote

using OptimizationOptimisers

const case = DoubleMM.DoubleMMCase()
const MLengine = Val(nameof(SimpleChains))
scenario = (:default,)

par_templates = get_hybridcase_par_templates(case; scenario)

(; n_covar, n_site, n_batch, n_θM, n_θP) = get_hybridcase_sizes(case; scenario)

rng = StableRNG(111)
(; xM, θP_true, θMs_true, xP, y_global_true, y_true, y_global_o, y_o
) = gen_hybridcase_synthetic(case, rng; scenario);

@testset "gen_hybridcase_synthetic" begin
    @test isapprox(
        vec(mean(CA.getdata(θMs_true); dims = 2)), CA.getdata(par_templates.θM), rtol = 0.02)
    @test isapprox(vec(std(CA.getdata(θMs_true); dims = 2)),
        CA.getdata(par_templates.θM) .* 0.1, rtol = 0.02)

    # test same results for same rng
    rng2 = StableRNG(111)
    gen2 = gen_hybridcase_synthetic(case, rng2; scenario);
    @test gen2.y_o == y_o
end

@testset "loss_g" begin
    g, ϕg0 = gen_hybridcase_MLapplicator(case, MLengine; scenario);

    function loss_g(ϕg, x, g)
        ζMs = g(x, ϕg) # predict the log of the parameters
        θMs = exp.(ζMs)
        loss = sum(abs2, θMs .- θMs_true)
        return loss, θMs
    end
    loss_g(ϕg0, xM, g)
    Zygote.gradient(x -> loss_g(x, xM, g)[1], ϕg0);

    optf = Optimization.OptimizationFunction((ϕg, p) -> loss_g(ϕg, xM, g)[1],
        Optimization.AutoZygote())
    optprob = Optimization.OptimizationProblem(optf, ϕg0);
    #res = Optimization.solve(optprob, Adam(0.02), callback = callback_loss(100), maxiters = 600);
    res = Optimization.solve(optprob, Adam(0.02), maxiters = 600);

    ϕg_opt1 = res.u;
    pred = loss_g(ϕg_opt1, xM, g)
    θMs_pred = pred[2]
    #scatterplot(vec(θMs_true), vec(θMs_pred))
    @test cor(vec(θMs_true), vec(θMs_pred)) > 0.9
end

@testset "loss_gf" begin
    #----------- fit g and θP to y_o
    g, ϕg0 = gen_hybridcase_MLapplicator(case, MLengine; scenario);
    f = gen_hybridcase_PBmodel(case; scenario)

    int_ϕθP = ComponentArrayInterpreter(CA.ComponentVector(
        ϕg = 1:length(ϕg0), θP = par_templates.θP))
    p = p0 = vcat(ϕg0, par_templates.θP .* 0.8);  # slightly disturb θP_true

    # Pass the site-data for the batches as separate vectors wrapped in a tuple
    train_loader = MLUtils.DataLoader((xM, xP, y_o), batchsize = n_batch)

    loss_gf = get_loss_gf(g, f, y_global_o, int_ϕθP)
    l1 = loss_gf(p0, train_loader.data...)[1]

    optf = Optimization.OptimizationFunction((ϕ, data) -> loss_gf(ϕ, data...)[1],
        Optimization.AutoZygote())
    optprob = OptimizationProblem(optf, p0, train_loader)

    res = Optimization.solve(
#        optprob, Adam(0.02), callback = callback_loss(100), maxiters = 1000);
        optprob, Adam(0.02), maxiters = 1000);

    l1, y_pred_global, y_pred, θMs_pred = loss_gf(res.u, train_loader.data...)
    @test isapprox(par_templates.θP, int_ϕθP(res.u).θP, rtol = 0.11)
    @test cor(vec(θMs_true), vec(θMs_pred)) > 0.9

    () -> begin
        scatterplot(vec(θMs_true), vec(θMs_pred))
        scatterplot(log.(vec(θMs_true)), log.(vec(θMs_pred)))
        scatterplot(vec(y_pred), vec(y_o))
        hcat(par_templates.θP, int_ϕθP(p0).θP, int_ϕθP(res.u).θP)
    end
end
