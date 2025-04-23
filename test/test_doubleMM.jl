using Test
using HybridVariationalInference
using HybridVariationalInference: HybridVariationalInference as HVI
using StableRNGs
using Random
using Statistics
using ComponentArrays: ComponentArrays as CA
using Bijectors

using SimpleChains
using MLUtils
import Zygote

using OptimizationOptimisers
using MLDataDevices

const prob = DoubleMM.DoubleMMCase()
scenario = (:default,)
#using Flux
#scenario = (:use_Flux,)

par_templates = get_hybridproblem_par_templates(prob; scenario)

@testset "get_hybridproblem_priors" begin
    θall = vcat(par_templates...)
    priors = get_hybridproblem_priors(prob; scenario)
    @test mean(priors[:K2]) == θall.K2
    @test quantile(priors[:K2], 0.95) ≈ θall.K2 * 3 # fitted in f_doubleMM
end

rng = StableRNG(111)
(; xM, n_site, θP_true, θMs_true, xP, y_global_true, y_true, y_global_o, y_o, y_unc
) = gen_hybridproblem_synthetic(rng, prob; scenario);
i_sites = 1:n_site

@testset "gen_hybridproblem_synthetic" begin
    @test isapprox(
        vec(mean(CA.getdata(θMs_true); dims = 2)), CA.getdata(par_templates.θM), rtol = 0.02)
    @test isapprox(vec(std(CA.getdata(θMs_true); dims = 2)),
        CA.getdata(par_templates.θM) .* 0.1, rtol = 0.02)

    # test same results for same rng
    rng2 = StableRNG(111)
    gen2 = gen_hybridproblem_synthetic(rng2, prob; scenario)
    @test gen2.y_o == y_o
end

@testset "loss_g" begin
    g, ϕg0 = get_hybridproblem_MLapplicator(rng, prob; scenario)
    (; transP, transM) = get_hybridproblem_transforms(prob; scenario)

    function loss_g(ϕg, x, g, transM)
        # @show first(x,5)
        # @show first(ϕg,5)
        ζMs = g(x, ϕg) # predict the parameters on unconstrained space
        # @show first(ζMs,5)
        θMs = reduce(hcat, map(transM, eachcol(ζMs))) # transform each column
        loss = sum(abs2, θMs .- θMs_true)
        return loss, θMs
    end
    l = loss_g(ϕg0, xM, g, transM)
    @test isfinite(l[1])
    Zygote.gradient(ϕg -> loss_g(ϕg, xM, g, transM)[1], ϕg0)
    #
    # actual optimization (do not need to test each time)
    () -> begin
        #histogram(ϕg0)
        optf = Optimization.OptimizationFunction((ϕg, p) -> loss_g(ϕg, xM, g, transM)[1],
            Optimization.AutoZygote())
        optprob = Optimization.OptimizationProblem(optf, ϕg0)
        #res = Optimization.solve(optprob, Adam(0.02), callback = callback_loss(100), maxiters = 600);
        res = Optimization.solve(optprob, Adam(0.02), maxiters = 600)
        #
        ϕg_opt1 = res.u
        #histogram(ϕg_opt1) # all similar magnitude around zero
        #first(ϕg_opt1,5)
        pred = loss_g(ϕg_opt1, xM, g, transM)
        θMs_pred = θMs_pred_1 = pred[2]
        #scatterplot(vec(θMs_true), vec(θMs_pred))
        #@test cor(vec(θMs_true), vec(θMs_pred)) > 0.9
        @test cor(θMs_true[:, 1], θMs_pred[:, 1]) > 0.9
        @test cor(θMs_true[:, 2], θMs_pred[:, 2]) > 0.9
    end
end

#redirect_stderr(open(touch(tempname()), "r"))

@testset "loss_gf" begin
    #----------- fit g and θP to y_o  (without uncertainty, without transforming θP)
    g, ϕg0 = get_hybridproblem_MLapplicator(prob; scenario)
    (; transP, transM) = get_hybridproblem_transforms(prob; scenario)
    f = get_hybridproblem_PBmodel(prob; scenario)

    intϕ = ComponentArrayInterpreter(CA.ComponentVector(
        ϕg = 1:length(ϕg0), ϕP = par_templates.θP))
    p = p0 = vcat(ϕg0, HVI.apply_preserve_axes(inverse(transP), par_templates.θP) .- 
        convert(eltype(ϕg0), 0.1))  # slightly disturb θP_true
    #p = p0 = vcat(ϕg_opt1, par_templates.θP);  # almost true

    # Pass the site-data for the batches as separate vectors wrapped in a tuple
    n_batch = 10
    train_loader = MLUtils.DataLoader((xM, xP, y_o, y_unc, i_sites), batchsize = n_batch)
    # get_hybridproblem_train_dataloader recreates synthetic data different θ_true
    train_loader2 = get_hybridproblem_train_dataloader(prob; scenario, n_batch = n_site)
    pbm_covars =  get_hybridproblem_pbmpar_covars(prob; scenario)

    #loss_gf = get_loss_gf(g, transM, f, y_global_o, intϕ; gdev = identity)
    loss_gf = get_loss_gf(g, transM, transP, f, y_global_o, intϕ; pbm_covars)
    l1 = loss_gf(p0, first(train_loader)...)[1]
    (xM_batch, xP_batch, y_o_batch, y_unc_batch, i_sites_batch) = first(train_loader)
    Zygote.gradient(
        p0 -> loss_gf(
            p0, xM_batch, xP_batch, y_o_batch, y_unc_batch, i_sites_batch)[1], CA.getdata(p0))

    optf = Optimization.OptimizationFunction((ϕ, data) -> loss_gf(ϕ, data...)[1],
        Optimization.AutoZygote())
    optprob = OptimizationProblem(optf, CA.getdata(p0), train_loader)

    res = Optimization.solve(
        #optprob, Adam(0.02), callback = callback_loss(100), maxiters = 5000);
        optprob, Adam(0.02), maxiters = 1000)

    l1, y_pred_global, y_pred, θMs_pred, θP_pred = loss_gf(res.u, train_loader.data...)
    #l1, y_pred_global, y_pred, θMs_pred = loss_gf(p0, xM, xP, y_o, y_unc);
    θMs_pred = CA.ComponentArray(θMs_pred, CA.getaxes(θMs_true))
    #TODO @test isapprox(par_templates.θP, intϕ(res.u).ϕP, rtol = 0.15)
    #@test cor(vec(θMs_true), vec(θMs_pred)) > 0.8
    @test cor(θMs_true[:, 1], θMs_pred[:, 1]) > 0.8
    @test cor(θMs_true[:, 2], θMs_pred[:, 2]) > 0.8
    # started from low values -> increased but not too much above true values
    @test all(transP(intϕ(p0).ϕP) .< θP_pred .< (1.2 .* par_templates.θP))

    () -> begin
        #@usingany UnicodePlots
        scatterplot(vec(θMs_true), vec(θMs_pred))
        scatterplot(θMs_true[1, :], θMs_pred[1, :])
        scatterplot(θMs_true[2, :], θMs_pred[2, :])
        scatterplot(log.(vec(θMs_true)), log.(vec(θMs_pred)))
        scatterplot(vec(y_pred), vec(y_o))
        hcat(par_templates.θP, intϕ(p0).ϕP, intϕ(res.u).ϕP, transP(intϕ(p0).ϕP), θP_pred)
    end
end

using CUDA: CUDA
using Flux
using GPUArraysCore

gdev = gpu_device()
cdev = cpu_device()
if gdev isa MLDataDevices.AbstractGPUDevice
    @testset "transfer NormalScalingModelApplicator to gpu" begin
        scenario = (:use_Flux,)
        g, ϕg0 = get_hybridproblem_MLapplicator(rng, prob; scenario)
        ϕg = gdev(ϕg0)
        xM_gpu = gdev(xM)
        g_gpu = gdev(g)
        @test g_gpu.μ isa GPUArraysCore.AbstractGPUArray
        y_gpu =  g_gpu(xM_gpu, ϕg)
        y = g(xM, ϕg0)
        @test cdev(y_gpu) ≈ y
    end;
end
