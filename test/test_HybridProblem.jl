using Test
using HybridVariationalInference
using HybridVariationalInference: HybridVariationalInference as CP
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

#scenario = Val((:default,))
#scenario = Val((:covarK2,))
#scen = CP._val_value(scenario)

function construct_problem(; scenario::Val{scen}) where scen
    FT = Float32
    θP = CA.ComponentVector{FT}(r0=0.3, K2=2.0)
    θM = CA.ComponentVector{FT}(r1=0.5, K1=0.2)
    transP = Stacked((CP.Exp(),),(1:2,))  # elementwise(exp)
    transM = Stacked(identity, CP.Exp()) # test different par transforms
    cor_ends = (P=1:length(θP), M=[length(θM)]) # assume r0 independent of K2
    int_θdoubleMM = get_concrete(ComponentArrayInterpreter(
        flatten1(CA.ComponentVector(; θP, θM))))
    function f_doubleMM(θ::AbstractVector{ET}, x; intθ1) where ET
        # extract parameters not depending on order, i.e whether they are in θP or θM
        local θc = intθ1(θ)
        (r0, r1, K1, K2) = map((:r0, :r1, :K1, :K2)) do par
            CA.getdata(θc[par])::ET
        end
        local y = r0 .+ r1 .* x.S1 ./ (K1 .+ x.S1) .* x.S2 ./ (K2 .+ x.S2)
        return (y)
    end    
    f_doubleMM_sites = let intθ1 = int_θdoubleMM, f_doubleMM=f_doubleMM,
        θFix = CA.ComponentVector{FT}()
        function f_doubleMM_with_global_inner(
            θP::AbstractVector{ET}, θMs::AbstractMatrix, xP
        ) where ET
            #first(eachcol(xP))
            local θMst = θMs'  # map_f_each:site requires sites-last format
            local pred_sites = map_f_each_site(f_doubleMM, θMst, θP, θFix, xP; intθ1)
            local pred_global = eltype(pred_sites)[]
            return pred_global, pred_sites
        end
    end
    n_out = length(θM)
    rng = StableRNG(111)
    # n_batch = 10
    n_site, n_batch = get_hybridproblem_n_site_and_batch(CP.DoubleMM.DoubleMMCase(); scenario)
    # dependency on DeoubleMMCase -> take care of changes in covariates
    (; xM, θP_true, θMs_true, xP, y_global_true, y_true, y_global_o, y_o, y_unc
    ) = gen_hybridproblem_synthetic(rng, DoubleMM.DoubleMMCase(); scenario)
    n_covar = size(xM,1)
    n_input = (:covarK2 ∈ scen) ? n_covar +1 : n_covar
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
    i_sites = 1:n_site
    # get_train_loader = let xM = xM, xP = xP, y_o = y_o, y_unc = y_unc, i_sites = i_sites
    #     function inner_get_train_loader(; n_batch, kwargs...)
    #         MLUtils.DataLoader((xM, xP, y_o, y_unc, i_sites), batchsize=n_batch, partial=false)
    #     end
    # end
    train_dataloader = MLUtils.DataLoader(
        (xM, xP, y_o, y_unc, i_sites), batchsize=n_batch, partial=false)
    θall = vcat(θP, θM)
    priors_dict = Dict{Symbol, Distribution}(
        keys(θall) .=> fit.(LogNormal, θall, QuantilePoint.(θall .* 3, 0.95)))
    priors_dict[:r1] = fit(Normal, θall.r1, qp_uu(3 * θall.r1)) # not transformed to log-scale
    # scale (0,1) outputs MLmodel to normal distribution fitted to priors translated to ζ
    priorsM = [priors_dict[k] for k in keys(θM)]
    lowers, uppers = get_quantile_transformed(priorsM, transM)
    app, ϕg0 = construct_ChainsApplicator(rng, g_chain)
    g_chain_scaled = NormalScalingModelApplicator(app, lowers, uppers, FT)
    #g_chain_scaled = app
    ϕunc0 = init_hybrid_ϕunc(cor_ends, zero(FT)) 
    pbm_covars = (:covarK2 ∈ scen) ? (:K2,) : ()
    HybridProblem(θP, θM, g_chain_scaled, ϕg0, ϕunc0, 
        f_doubleMM_sites, f_doubleMM_sites, priors_dict, py,
        transM, transP, train_dataloader, n_covar, n_site, n_batch, 
        cor_ends, pbm_covars)
end 

@testset "f_doubleMM from ProbSpec" begin
    θ1 = CA.ComponentVector(r0=1.1, r1=2.1, K1=3.1, K2=4.1)
    θ2 = reverse(θ1)
    int_θdoubleMM = get_concrete(ComponentArrayInterpreter(θ2))
    xP1 = CA.ComponentVector(S1=[1,1,0.3,0.1], S2=[0.1,0.3,1,1,])
    n_site = 4
    xPint = ComponentArrayInterpreter((n_site,), ComponentArrayInterpreter(xP1))
    xP = xPint(repeat(xP1, outer=(1,4)))
    function test_f_doubleMM(θ::AbstractVector{ET}, x; intθ1) where ET
        # extract parameters not depending on order, i.e whether they are in θP or θM
        θc = intθ1(θ)
        (r0, r1, K1, K2) = map((:r0, :r1, :K1, :K2)) do par
            CA.getdata(θc[par])::ET
        end
        r0 .+ r1 .* x.S1 ./ (K1 .+ x.S1) .* x.S2 ./ (K2 .+ x.S2)
    end    
    y = @inferred test_f_doubleMM(CA.getdata(θ2), xP1; intθ1 = int_θdoubleMM)
    # using ShareAdd; @usingany Cthulhu
    # @descend_code_warntype test_f_doubleMM(CA.getdata(θ2), xP1)
end

test_without_flux = (scenario) -> begin
    #scen = CP._val_value(scenario)
    gdev = @suppress gpu_device()

    prob = probc = construct_problem(;scenario);
    #@descend construct_problem(;scenario)

    @testset "n_input and pbm_covars  $(last(CP._val_value(scenario)))" begin
        g, ϕ_g = get_hybridproblem_MLapplicator(prob; scenario);
        if :covarK2 ∈ CP._val_value(scenario)
            @test g.app.m.inputdim == (static(6),) # 5 + 1 (ncovar + n_pbm)
            @test get_hybridproblem_pbmpar_covars(prob; scenario) == (:K2,)
        else
            @test g.app.m.inputdim == (static(5),) 
            @test get_hybridproblem_pbmpar_covars(prob; scenario) == ()
        end
    end

    @testset "loss_gf $(last(CP._val_value(scenario)))" begin
        #----------- fit g and θP to y_o
        rng = StableRNG(111)
        g, ϕg0 = get_hybridproblem_MLapplicator(prob; scenario)
        n_site, n_batch = get_hybridproblem_n_site_and_batch(prob; scenario)
        train_loader = get_hybridproblem_train_dataloader(prob; scenario)
        (xM, xP, y_o, y_unc, i_sites) = first(train_loader)
        f = get_hybridproblem_PBmodel(prob; scenario, use_all_sites = false)
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
        loss_gf = get_loss_gf(g, transM, transP, f, y_global_o, intϕ; 
            pbm_covars, n_site_batch = n_batch)
        (_xM, _xP, _y_o, _y_unc, _i_sites) = first(train_loader)
        l1 = @inferred (
            # @descend_code_warntype (
            loss_gf(p0, _xM, _xP, _y_o, _y_unc, _i_sites))
        tld = first(train_loader)
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

test_without_flux(Val((:default,)))
test_without_flux(Val((:covarK2,)))

import CUDA, cuDNN
using GPUArraysCore
import Flux

gdev = gpu_device()
#methods(CP.vec2uutri)

test_with_flux = (scenario) -> begin
    prob = probc = construct_problem(;scenario);

    @testset "HybridPointSolver $(last(CP._val_value(scenario)))" begin
        rng = StableRNG(111)
        solver = HybridPointSolver(; alg=Adam(0.02))
        (; ϕ, resopt, probo) = solve(prob, solver; scenario, rng,
            #callback = callback_loss(100), maxiters = 1200
            #maxiters = 1200
            #maxiters = 20
            maxiters=200,
            gdev = identity,
            #gpu_handler = NullGPUDataHandler
            is_inferred = Val(true),
        )
        (; θP) = get_hybridproblem_par_templates(prob; scenario)
        θPo = (() -> begin
            (; θP) = get_hybridproblem_par_templates(probo; scenario); 
            θP
        end)()
        @test θPo.r0 < 1.5 * θP.r0
        @test ϕ.ϕP.K2 < 1.5 * log(θP.K2)
    end;

    @testset "HybridPosteriorSolver  $(last(CP._val_value(scenario)))" begin
        rng = StableRNG(111)
        solver = HybridPosteriorSolver(; alg=Adam(0.02), n_MC=3)
        (; ϕ, θP, resopt) = solve(prob, solver; scenario, rng,
            #callback = callback_loss(100), maxiters = 1200,
            #maxiters = 20 # too small so that it yields error
            maxiters=37,
            θmean_quant = 0.01,   # test constraining mean to initial prediction     
            gdev = identity,
            is_inferred = Val(true),
        )
        θPt = get_hybridproblem_par_templates(prob; scenario).θP
        @test θP.r0 < 1.5 * θPt.r0
        @test exp(ϕ.μP.K2) == θP.K2 < 1.5 * θP.K2
        θP
        prob.θP
    end;



    if gdev isa MLDataDevices.AbstractGPUDevice 
        @testset "HybridPosteriorSolver gpu  $(last(CP._val_value(scenario)))" begin
            scenf = Val((CP._val_value(scenario)..., :use_Flux, :use_gpu, :omit_r0))
            rng = StableRNG(111)
            # here using DoubleMMCase() directly rather than construct_problem
            #(;transP, transM) = get_hybridproblem_transforms(DoubleMM.DoubleMMCase(); scenario = scenf)
            prob = probg = HybridProblem(DoubleMM.DoubleMMCase(); scenario = scenf);
            solver = HybridPosteriorSolver(; alg=Adam(0.02),  n_MC=3)
            n_site, n_batch = get_hybridproblem_n_site_and_batch(prob; scenario = scenf)
            n_batches_in_epoch =  n_site ÷ n_batch
            (; ϕ, θP, resopt) = solve(prob, solver; scenario = scenf, rng,
                maxiters = 37, # smallest value by trial and error
                #maxiters = 20 # too small so that it yields error
                θmean_quant = 0.01,   # test constraining mean to initial prediction     
                is_inferred = Val(true),
            );
            @test CA.getdata(ϕ) isa GPUArraysCore.AbstractGPUVector
            #@test cdev(ϕ.unc.ρsM)[1] > 0 # too few iterations in test -> may fail
            #
            solver = HybridPosteriorSolver(; alg=Adam(0.02), n_MC=3)
            (; ϕ, θP, resopt, probo) = solve(prob, solver; scenario = scenf,
                maxiters = 37, 
                is_inferred = Val(true),
            );
            @test cdev(ϕ.unc.ρsM)[1] > 0 
            @test probo.ϕunc == cdev(ϕ.unc)
            n_sample_pred = 22
            (; y, θsP, θsMs) = predict_hvi(
                rng, probo; scenario = scenf, n_sample_pred, is_inferred=Val(true));            
            (_xM, _xP, _y_o, _y_unc, _i_sites) = get_hybridproblem_train_dataloader(prob; scenario).data
            @test size(y) == (size(_y_o)..., n_sample_pred)
            @test size(θsP) == (size(probo.θP,1), n_sample_pred)
            
            test_correlation = () -> begin
                n_epoch = 20 # requires 
                (; ϕ, θP, resopt, probo) = solve(prob, solver; scenario = scenf,
                    maxiters = n_batches_in_epoch * n_epoch, 
                    callback = callback_loss(n_batches_in_epoch*5)
                );
                @test cdev(ϕ.unc.ρsM)[1] > 0 
                @test probo.ϕunc == cdev(ϕ.unc)
                # predict using problem and its associated dataloader
                n_sample_pred = 201
                (; y, θsP, θsMs) = predict_hvi(rng, probo; scenario = scenf, n_sample_pred);            
                # to inspect correlations among θP and θMs construct ComponentVector
                hpints = HybridProblemInterpreters(prob; scenario)
                int_mPMs = stack_ca_int(Val((n_sample_pred,)), get_int_PMst_site(hpints))
                θs =  int_mPMs(CP.flatten_hybrid_pars(θsP, θsMs))
                mean_θ = CA.ComponentVector(vec(mean(CA.getdata(θs), dims=1)), last(CA.getaxes(θs)))
                mean_θ.Ms
                sd_θ = CA.ComponentVector(vec(std(CA.getdata(θs), dims=1)), last(CA.getaxes(θs)))
                sd_θ.Ms
                pos = get_positions(ComponentArrayInterpreter(mean_θ))
                residθs = θs .- mean_θ

                cr = cor(CA.getdata(residθs'))
                pos_P = get_positions(ComponentArrayInterpreter(θs[:P,1]))
                i_sites = [1,2,3]
                #ax = map(x -> axes(x,1), get_hybridproblem_par_templates(probo; scenario = scenf))
                is = vcat(pos.P, vec(pos.Ms[i_sites,:]))
                cr[is,is]
            end

        end;
        @testset "HybridPosteriorSolver also f on gpu  $(last(CP._val_value(scenario)))" begin
            scenf = Val((CP._val_value(scenario)..., :use_Flux, :use_gpu, :omit_r0, :f_on_gpu))
            rng = StableRNG(111)
            probg = HybridProblem(DoubleMM.DoubleMMCase(); scenario = scenf);
            #prob = CP.update(probg, transM = identity, transP = identity);
            solver = HybridPosteriorSolver(; alg=Adam(0.02), n_MC=3)
            n_site, n_batch = get_hybridproblem_n_site_and_batch(prob; scenario = scenf)
            n_batches_in_epoch = n_site ÷ n_batch
            (; ϕ, θP, resopt, probo) = solve(prob, solver; scenario = scenf, rng,
                maxiters = 37, # smallest value by trial and error
                #maxiters = 20 # too small so that it yields error
                #θmean_quant = 0.01,   # TODO make possible on gpu
                cdev = identity, # do not move ζ to cpu # TODO infer in solve from scenario
                is_inferred = Val(true),
            );
            @test CA.getdata(ϕ) isa GPUArraysCore.AbstractGPUVector
            n_sample_pred = 11
            (; y, θsP, θsMs) = predict_hvi(
                rng, probo; scenario = scenf, n_sample_pred,is_inferred = Val(true));
            # @test cdev(ϕ.unc.ρsM)[1] > 0 # too few iterations
        end;    
    end # if gdev isa MLDataDevices.AbstractGPUDevice 
end # test_with flux

test_with_flux(Val((:default,)))
test_with_flux(Val((:covarK2,)))
