using Test
using HybridVariationalInference
using StableRNGs
using Random
using Statistics
using ComponentArrays: ComponentArrays as CA
using Bijectors
using DistributionFits
using StatsFuns: logistic

using SimpleChains
using MLUtils
import Zygote

using OptimizationOptimisers
using MLDataDevices

cdev = cpu_device()

construct_problem = () -> begin
    FT = Float32
    θP = CA.ComponentVector{FT}(r0=0.3, K2=2.0)
    θM = CA.ComponentVector{FT}(r1=0.5, K1=0.2)
    transP = elementwise(exp)
    transM = Stacked(elementwise(identity), elementwise(exp))
    cor_ends = (P=1:length(θP), M=[length(θM)]) # assume r0 independent of K2
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
        pred_sites = applyf(f_doubleMM, θMs, θP, CA.ComponentVector{FT}(), x)
        pred_global = eltype(pred_sites)[]
        return pred_global, pred_sites
    end
    n_out = length(θM)
    rng = StableRNG(111)
    # dependency on DeoubleMMCase -> take care of changes in covariates
    (; xM, n_site, θP_true, θMs_true, xP, y_global_true, y_true, y_global_o, y_o, y_unc
    ) = gen_hybridcase_synthetic(rng, DoubleMM.DoubleMMCase())
    n_covar = size(xM,1)
    g_chain = SimpleChain(
        static(n_covar), # input dimension (optional)
        # dense layer with bias that maps to 8 outputs and applies `tanh` activation
        TurboDense{true}(tanh, n_covar * 4),
        TurboDense{true}(tanh, n_covar * 4),
        # dense layer without bias that maps to n outputs and `identity` activation
        TurboDense{false}(logistic, n_out)
    )
    # g, ϕg = construct_SimpleChainsApplicator(g_chain)
    #
    py = neg_logden_indep_normal
    n_batch = 10
    get_train_loader = let xM = xM, xP = xP, y_o = y_o, y_unc = y_unc
        function inner_get_train_loader(rng; n_batch, kwargs...)
            MLUtils.DataLoader((xM, xP, y_o, y_unc), batchsize=n_batch, partial=false)
        end
    end
    θall = vcat(θP, θM)
    priors_dict = Dict{Symbol, Distribution}(keys(θall) .=> fit.(LogNormal, θall, QuantilePoint.(θall .* 3, 0.95)))
    priors_dict[:r1] = fit(Normal, θall.r1, qp_uu(3 * θall.r1)) # not transformed to log-scale
    # scale (0,1) outputs MLmodel to normal distribution fitted to priors translated to ζ
    priorsM = [priors_dict[k] for k in keys(θM)]
    app, ϕg0 = construct_ChainsApplicator(rng, g_chain)
    g_chain_scaled = NormalScalingModelApplicator(app, priorsM, transM, FT)
    #g_chain_scaled = app
    HybridProblem(θP, θM, g_chain_scaled, ϕg0, f_doubleMM_with_global, priors_dict, py,
        transM, transP, get_train_loader, cor_ends)
end
prob = construct_problem();
scenario = (:default,)

@testset "loss_gf" begin
    #----------- fit g and θP to y_o
    rng = StableRNG(111)
    g, ϕg0 = get_hybridproblem_MLapplicator(prob; scenario)
    train_loader = get_hybridproblem_train_dataloader(rng, prob; n_batch=10, scenario)
    (xM, xP, y_o, y_unc) = first(train_loader)
    f = get_hybridproblem_PBmodel(prob; scenario)
    par_templates = get_hybridproblem_par_templates(prob; scenario)
    #f(par_templates.θP, hcat(par_templates.θM, par_templates.θM), xP[1:2])
    (; transM, transP) = get_hybridproblem_transforms(prob; scenario)

    int_ϕθP = ComponentArrayInterpreter(CA.ComponentVector(
        ϕg=1:length(ϕg0), θP=par_templates.θP))
    # slightly disturb θP_true
    p = p0 = vcat(ϕg0, par_templates.θP .* convert(eltype(ϕg0), 0.8))  

    # Pass the site-data for the batches as separate vectors wrapped in a tuple

    y_global_o = Float64[]
    loss_gf = get_loss_gf(g, transM, f, y_global_o, int_ϕθP)
    l1 = loss_gf(p0, first(train_loader)...)
    gr = Zygote.gradient(p -> loss_gf(p, train_loader.data...)[1], CA.getdata(p0))
    @test gr[1] isa Vector

    () -> begin
        optf = Optimization.OptimizationFunction((ϕ, data) -> loss_gf(ϕ, data...)[1],
            Optimization.AutoZygote())
        optprob = OptimizationProblem(optf, p0, train_loader)

        res = Optimization.solve(
            #        optprob, Adam(0.02), callback = callback_loss(100), maxiters = 1000);
            optprob, Adam(0.02), maxiters=1000)

        l1, y_pred_global, y_pred, θMs_pred = loss_gf(res.u, train_loader.data...)
        @test isapprox(par_templates.θP, int_ϕθP(res.u).θP, rtol=0.11)
    end
end

using CUDA: CUDA
using cuDNN: cuDNN
using MLDataDevices, GPUArraysCore
import Flux

@testset "neg_elbo_transnorm_gf" begin
    rng = StableRNG(111)
    g, ϕg0 = get_hybridproblem_MLapplicator(prob)
    train_loader = get_hybridproblem_train_dataloader(rng, prob; n_batch=10)
    (xM, xP, y_o, y_unc) = first(train_loader)
    n_batch = size(y_o, 2)
    f = get_hybridproblem_PBmodel(prob)
    (θP0, θM0) = get_hybridproblem_par_templates(prob)
    (; transP, transM) = get_hybridproblem_transforms(prob)
    py = get_hybridproblem_neg_logden_obs(prob)
    cor_ends = get_hybridproblem_cor_ends(prob)

    (; ϕ, transPMs_batch, interpreters, get_transPMs, get_ca_int_PMs) = init_hybrid_params(
        θP0, θM0, cor_ends, ϕg0, n_batch; transP, transM)
    ϕ_ini = ϕ
    ϕ.unc

    py = get_hybridproblem_neg_logden_obs(prob)

    cost = neg_elbo_transnorm_gf(rng, ϕ_ini, g, transPMs_batch, f, py,
        xM, xP, y_o, y_unc, map(get_concrete, interpreters);
        n_MC=8, cor_ends)
    @test cost isa Float64
    gr = Zygote.gradient(
        ϕ -> neg_elbo_transnorm_gf(rng, ϕ, g, transPMs_batch, f, py,
            xM, xP, y_o, y_unc, map(get_concrete, interpreters);
            n_MC=8, cor_ends),
        CA.getdata(ϕ_ini))
    @test gr[1] isa Vector

    gdev = gpu_device()
    if gdev isa MLDataDevices.AbstractGPUDevice 
        @testset "neg_elbo_transnorm_gf gpu" begin
            g, ϕg0 = begin
                n_covar = size(xM, 1)
                n_out = length(θM0)
                g_chain = Flux.Chain(
                    # dense layer with bias that maps to 8 outputs and applies `tanh` activation
                    Flux.Dense(n_covar => n_covar * 4, tanh),
                    Flux.Dense(n_covar * 4 => n_covar * 4, tanh),
                    # dense layer without bias that maps to n outputs and `identity` activation
                    Flux.Dense(n_covar * 4 => n_out, logistic, bias=false)
                )
                construct_ChainsApplicator(g_chain, eltype(θM0))
            end
            ϕ_ini.ϕg = ϕg0
            ϕ = gdev(CA.getdata(ϕ_ini))
            xMg = gdev(xM)
            g_dev = gdev(g)
            cost = neg_elbo_transnorm_gf(rng, ϕ, g_dev, transPMs_batch, f, py,
                xMg, xP, y_o, y_unc, map(get_concrete, interpreters);
                n_MC=8, cor_ends)
            @test cost isa Float64
            gr = Zygote.gradient(
                ϕ -> neg_elbo_transnorm_gf(rng, ϕ, g_dev, transPMs_batch, f, py,
                    xMg, xP, y_o, y_unc, map(get_concrete, interpreters);
                    n_MC=8, cor_ends),
                ϕ)
            @test gr[1] isa GPUArraysCore.AbstractGPUArray
            @test eltype(gr[1]) == get_hybridproblem_float_type(prob)
        end
    end
end

@testset "HybridPointSolver" begin
    rng = StableRNG(111)
    solver = HybridPointSolver(; alg=Adam(0.02), n_batch=11)
    (; ϕ, resopt) = solve(prob, solver; scenario, rng,
        #callback = callback_loss(100), maxiters = 1200
        #maxiters = 1200
        #maxiters = 20
        maxiters=200,
        dev = cdev,
        #gpu_handler = NullGPUDataHandler
    )
    (; θP) = get_hybridproblem_par_templates(prob; scenario)
    @test ϕ.θP.r0 < 1.5 * θP.r0
end;

@testset "HybridPosteriorSolver" begin
    rng = StableRNG(111)
    solver = HybridPosteriorSolver(; alg=Adam(0.02), n_batch=11, n_MC=3)
    (; ϕ, θP, resopt) = solve(prob, solver; scenario, rng,
        #callback = callback_loss(100), maxiters = 1200
        #maxiters = 20 # yields error
        maxiters=200,
        dev = cdev
    )
    θPt = get_hybridproblem_par_templates(prob; scenario).θP
    @test θP.r0 < 1.5 * θPt.r0
end;
