using Test
using HybridVariationalInference
using HybridVariationalInference: HybridVariationalInference as HVI
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
using Suppressor

cdev = cpu_device()

#scenario = (:default,)
#scenario = (:covarK2,)


construct_problem = (;scenario=(:default,)) -> begin
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
    ) = gen_hybridproblem_synthetic(rng, DoubleMM.DoubleMMCase())
    n_covar = size(xM,1)
    n_input = (:covarK2 ∈ scenario) ? n_covar +1 : n_covar
    g_chain = SimpleChain(
        static(n_input), # input dimension (optional)
        # dense layer with bias that maps to 8 outputs and applies `tanh` activation
        TurboDense{true}(tanh, n_input * 4),
        TurboDense{true}(tanh, n_input * 4),
        # dense layer without bias that maps to n outputs and `identity` activation
        TurboDense{false}(logistic, n_out)
    )
    # g, ϕg = construct_SimpleChainsApplicator(g_chain)
    #
    py = neg_logden_indep_normal
    n_batch = 10
    i_sites = 1:n_site
    get_train_loader = let xM = xM, xP = xP, y_o = y_o, y_unc = y_unc, i_sites = i_sites
        function inner_get_train_loader(; n_batch, kwargs...)
            MLUtils.DataLoader((xM, xP, y_o, y_unc, i_sites), batchsize=n_batch, partial=false)
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
    ϕunc0 = init_hybrid_ϕunc(cor_ends, zero(FT)) 
    pbm_covars = (:covarK2 ∈ scenario) ? (:K2,) : ()
    HybridProblem(θP, θM, g_chain_scaled, ϕg0, ϕunc0, f_doubleMM_with_global, priors_dict, py,
        transM, transP, get_train_loader, n_covar, n_site, cor_ends, pbm_covars)
end

test_without_flux = (scenario) -> begin
    gdev = @suppress gpu_device()

    prob = probc = construct_problem(;scenario);
    #@descend construct_problem(;scenario)

    @testset "n_input and pbm_covars" begin
        g, ϕ_g = get_hybridproblem_MLapplicator(prob; scenario);
        if :covarK2 ∈ scenario
            @test g.app.m.inputdim == (static(6),) # 5 + 1 (ncovar + n_pbm)
            @test get_hybridproblem_pbmpar_covars(prob; scenario) == (:K2,)
        else
            @test g.app.m.inputdim == (static(5),) 
            @test get_hybridproblem_pbmpar_covars(prob; scenario) == ()
        end
    end

    @testset "loss_gf" begin
        #----------- fit g and θP to y_o
        rng = StableRNG(111)
        g, ϕg0 = get_hybridproblem_MLapplicator(prob; scenario)
        train_loader = get_hybridproblem_train_dataloader(prob; n_batch=10, scenario)
        (xM, xP, y_o, y_unc, i_sites) = first(train_loader)
        f = get_hybridproblem_PBmodel(prob; scenario)
        par_templates = get_hybridproblem_par_templates(prob; scenario)
        #f(par_templates.θP, hcat(par_templates.θM, par_templates.θM), xP[1:2])
        (; transM, transP) = get_hybridproblem_transforms(prob; scenario)
        pbm_covars = get_hybridproblem_pbmpar_covars(prob; scenario)
        intϕ = ComponentArrayInterpreter(CA.ComponentVector(
            ϕg=1:length(ϕg0), ϕP=par_templates.θP))
        # slightly disturb θP_true
        p = p0 = vcat(ϕg0, par_templates.θP .* convert(eltype(ϕg0), 0.8))  

        # Pass the site-data for the batches as separate vectors wrapped in a tuple
        y_global_o = Float64[]
        loss_gf = get_loss_gf(g, transM, transP, f, y_global_o, intϕ; pbm_covars)
        l1 = loss_gf(p0, first(train_loader)...)
        tld = train_loader.data
        gr = Zygote.gradient(p -> loss_gf(p, tld...)[1], CA.getdata(p0))
        @test gr[1] isa Vector

        () -> begin
            optf = Optimization.OptimizationFunction((ϕ, data) -> loss_gf(ϕ, data...)[1],
                Optimization.AutoZygote())
            optprob = OptimizationProblem(optf, p0, train_loader)

            res = Optimization.solve(
                #        optprob, Adam(0.02), callback = callback_loss(100), maxiters = 1000);
                optprob, Adam(0.02), maxiters=1000)

            l1, y_pred_global, y_pred, θMs_pred = loss_gf(res.u, train_loader.data...)
            @test isapprox(par_templates.θP, intϕ(res.u).θP, rtol=0.11)
        end
    end
end

test_without_flux((:default,))
test_without_flux((:covarK2,))

import CUDA, cuDNN
using GPUArraysCore
import Flux

gdev = gpu_device()
#methods(HVI.vec2uutri)

test_with_flux = (scenario) -> begin
    prob = probc = construct_problem(;scenario);

    @testset "HybridPointSolver" begin
        rng = StableRNG(111)
        solver = HybridPointSolver(; alg=Adam(0.02), n_batch=11)
        (; ϕ, resopt) = solve(prob, solver; scenario, rng,
            #callback = callback_loss(100), maxiters = 1200
            #maxiters = 1200
            #maxiters = 20
            maxiters=200,
            gdev = identity,
            #gpu_handler = NullGPUDataHandler
        )
        (; θP) = get_hybridproblem_par_templates(prob; scenario)
        @test ϕ.ϕP.r0 < 1.5 * θP.r0
        @test ϕ.ϕP.K2 < 1.5 * log(θP.K2)
    end;

    @testset "HybridPosteriorSolver" begin
        rng = StableRNG(111)
        solver = HybridPosteriorSolver(; alg=Adam(0.02), n_batch=11, n_MC=3)
        (; ϕ, θP, resopt) = solve(prob, solver; scenario, rng,
            callback = callback_loss(100), maxiters = 1200,
            #maxiters = 20 # too small so that it yields error
            #maxiters=200,
            θmean_quant = 0.01,   # test constraining mean to initial prediction     
            gdev = identity
        )
        θPt = get_hybridproblem_par_templates(prob; scenario).θP
        @test θP.r0 < 1.5 * θPt.r0
        @test exp(ϕ.μP.K2) == θP.K2 < 1.5 * θP.K2
        θP
        prob.θP
    end;

    if gdev isa MLDataDevices.AbstractGPUDevice 
        @testset "HybridPosteriorSolver gpu" begin
            scenf = (scenario..., :use_Flux, :use_gpu, :omit_r0)
            rng = StableRNG(111)
            prob = probg = HybridProblem(DoubleMM.DoubleMMCase(); scenario = scenf)
            solver = HybridPosteriorSolver(; alg=Adam(0.02), n_batch=11, n_MC=3)
            n_batches_in_epoch = get_hybridproblem_n_site(prob; scenario) ÷ solver.n_batch
            (; ϕ, θP, resopt) = solve(prob, solver; scenario = scenf, rng,
                maxiters = 37, # smallest value by trial and error
                #maxiters = 20 # too small so that it yields error
                θmean_quant = 0.01,   # test constraining mean to initial prediction     
            );
            @test CA.getdata(ϕ) isa GPUArraysCore.AbstractGPUVector
            @test cdev(ϕ.unc.ρsM)[1] > 0
        end;
        # does not work with general Bijector:
        # @testset "HybridPosteriorSolver also f on gpu" begin
        #     scenario = (:use_Flux, :use_gpu, :omit_r0, :f_on_gpu)
        #     rng = StableRNG(111)
        #     probg = HybridProblem(DoubleMM.DoubleMMCase(); scenario)
        #     prob = HVI.update(probg)
        #     #prob = HVI.update(probg, transM = identity, transP = identity)
        #     solver = HybridPosteriorSolver(; alg=Adam(0.02), n_batch=11, n_MC=3)
        #     n_batches_in_epoch = get_hybridproblem_n_site(prob; scenario) ÷ solver.n_batch
        #     (; ϕ, θP, resopt) = solve(prob, solver; scenario, rng,
        #         maxiters = 37, # smallest value by trial and error
        #         #maxiters = 20 # too small so that it yields error
        #         #θmean_quant = 0.01,   # TODO make possible on gpu
        #         cdev = identity # do not move ζ to cpu # TODO infer in solve from scenario
        #     )
        #     @test CA.getdata(ϕ) isa GPUArraysCore.AbstractGPUVector
        #     @test cdev(ϕ.unc.ρsM)[1] > 0
        # end;    
    end # if gdev isa MLDataDevices.AbstractGPUDevice 
end # test_with flux

test_with_flux((:default,))
test_with_flux((:covarK2,))
