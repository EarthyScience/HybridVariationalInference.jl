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
scenario = (:default,)

par_templates = get_hybridcase_par_templates(case; scenario)

rng = StableRNG(111)
(; xM, n_site, θP_true, θMs_true, xP, y_global_true, y_true, y_global_o, y_o, y_unc
) = gen_hybridcase_synthetic(rng, case; scenario);

@testset "gen_hybridcase_synthetic" begin
    @test isapprox(
        vec(mean(CA.getdata(θMs_true); dims = 2)), CA.getdata(par_templates.θM), rtol = 0.02)
    @test isapprox(vec(std(CA.getdata(θMs_true); dims = 2)),
        CA.getdata(par_templates.θM) .* 0.1, rtol = 0.02)

    # test same results for same rng
    rng2 = StableRNG(111)
    gen2 = gen_hybridcase_synthetic(rng2, case; scenario);
    @test gen2.y_o == y_o
end

@testset "loss_g" begin
    g, ϕg0 = get_hybridcase_MLapplicator(rng, case; scenario);
    (;transP, transM) = get_hybridcase_transforms(case; scenario)

    function loss_g(ϕg, x, g, transM)
        # @show first(x,5)
        # @show first(ϕg,5)
        ζMs = g(x, ϕg) # predict the parameters on unconstrained space
        # @show first(ζMs,5)
        θMs = reduce(hcat, map(transM, eachcol(ζMs))) # transform each column
        loss = sum(abs2, θMs .- θMs_true)
        return loss, θMs
    end
    loss_g(ϕg0, xM, g, transM)
    Zygote.gradient(ϕg -> loss_g(ϕg, xM, g, transM)[1], ϕg0);
    #
    optf = Optimization.OptimizationFunction((ϕg, p) -> loss_g(ϕg, xM, g, transM)[1],
        Optimization.AutoZygote())
    optprob = Optimization.OptimizationProblem(optf, ϕg0);
    #res = Optimization.solve(optprob, Adam(0.02), callback = callback_loss(100), maxiters = 600);
    res = Optimization.solve(optprob, Adam(0.02), maxiters = 600);
    #
    ϕg_opt1 = res.u;
    #first(ϕg_opt1,5)
    pred = loss_g(ϕg_opt1, xM, g, transM);
    θMs_pred = θMs_pred_1 = pred[2]
    #scatterplot(vec(θMs_true), vec(θMs_pred))
    #@test cor(vec(θMs_true), vec(θMs_pred)) > 0.9
    @test cor(θMs_true[:,1], θMs_pred[:,1]) > 0.9
    @test cor(θMs_true[:,2], θMs_pred[:,2]) > 0.9
end

@testset "loss_gf" begin
    #----------- fit g and θP to y_o  (without uncertainty, without transforming θP)
    g, ϕg0 = get_hybridcase_MLapplicator(case; scenario);
    (;transP, transM) = get_hybridcase_transforms(case; scenario)
    f = get_hybridcase_PBmodel(case; scenario)

    int_ϕθP = ComponentArrayInterpreter(CA.ComponentVector(
        ϕg = 1:length(ϕg0), θP = par_templates.θP))
    p = p0 = vcat(ϕg0, par_templates.θP .* 0.8);  # slightly disturb θP_true
    #p = p0 = vcat(ϕg_opt1, par_templates.θP);  # almost true

    # Pass the site-data for the batches as separate vectors wrapped in a tuple
    n_batch = 10
    train_loader = MLUtils.DataLoader((xM, xP, y_o, y_unc), batchsize = n_batch)
    # get_hybridcase_train_dataloader recreates synthetic data different θ_true
    #train_loader = get_hybridcase_train_dataloader(case, rng; scenario)

    loss_gf = get_loss_gf(g, transM, f, y_global_o, int_ϕθP)
    l1 = loss_gf(p0, first(train_loader)...)[1]
    (xM_batch, xP_batch, y_o_batch, y_unc_batch) = first(train_loader)
    Zygote.gradient(p0 -> loss_gf(p0, xM_batch, xP_batch, y_o_batch, y_unc_batch)[1], p0)

    optf = Optimization.OptimizationFunction((ϕ, data) -> loss_gf(ϕ, data...)[1],
        Optimization.AutoZygote())
    optprob = OptimizationProblem(optf, p0, train_loader)

    res = Optimization.solve(
        #optprob, Adam(0.02), callback = callback_loss(100), maxiters = 5000);
        optprob, Adam(0.02), maxiters = 1000);

    l1, y_pred_global, y_pred, θMs_pred = loss_gf(res.u, train_loader.data...)
    #l1, y_pred_global, y_pred, θMs_pred = loss_gf(p0, xM, xP, y_o, y_unc);
    θMs_pred = CA.ComponentArray(θMs_pred, CA.getaxes(θMs_true))
    #TODO @test isapprox(par_templates.θP, int_ϕθP(res.u).θP, rtol = 0.15)
    @test cor(vec(θMs_true), vec(θMs_pred)) > 0.9
    @test cor(θMs_true[:,1], θMs_pred[:,1]) > 0.9
    @test cor(θMs_true[:,2], θMs_pred[:,2]) > 0.9

    () -> begin
        scatterplot(vec(θMs_true), vec(θMs_pred))
        scatterplot(θMs_true[1,:], θMs_pred[1,:])
        scatterplot(θMs_true[2,:], θMs_pred[2,:])
        scatterplot(log.(vec(θMs_true)), log.(vec(θMs_pred)))
        scatterplot(vec(y_pred), vec(y_o))
        hcat(par_templates.θP, int_ϕθP(p0).θP, int_ϕθP(res.u).θP)
    end
end
