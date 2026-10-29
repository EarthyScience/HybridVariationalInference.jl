ENV["MLDATADEVICES_SILENCE_WARN_NO_GPU"]="1" # suppress warning on missing CUDA
#using LinearAlgebra, BlockDiagonals
using LinearAlgebra
using StatsFuns: logistic

using Test
using HybridVariationalInference
using HybridVariationalInference: HybridVariationalInference as CP
using HybridVariationalInference: HybridVariationalInference as HVI
using StableRNGs
using Random
using ComponentArrays: ComponentArrays as CA
#using TransformVariables
using Bijectors
#import PreallocationTools as PAT
import JLD2
import PDMats
import StaticArrays as SA

import Lux
# import Enzyme
import ForwardDiff
import Optimisers
using CommonSolve

import MLUtils
import PreallocationTools as PAT



prob = DoubleMM.DoubleMMCase()
#scenario = Val((:use_Lux,:usePopPBM,:covarK2,:uniscaling,))
#scenario = Val((:use_Lux,:usePopPBM,:uniscaling,)) # omit covars -> vector site ϕm
scenario = Val((:use_Lux,:usePopPBM,:uniscaling,:use_rangescaling)) # omit covars, rangescaling


diffchunk = ForwardDiff.Chunk(8)
rng = StableRNG(1234)
(; xM, θP_true, θMs_true, xP,  y_true,  y_o, y_unc,) = gen_hybridproblem_synthetic(rng, prob; scenario);
i_indiv_batch = 10 .+ (1:20)
n_site = length(i_indiv_batch)
par_templates = get_hybridproblem_par_templates(prob; scenario)
n_θP = length(par_templates.θP)
n_θM = length(par_templates.θM)
n_cov = get_hybridproblem_n_covar(prob; scenario)
pbm_covars = get_hybridproblem_pbmpar_covars(prob; scenario)
n_covP = length(pbm_covars)
pbm_covar_indices = SA.SVector{length(pbm_covars)}([findfirst(==(k), keys(par_templates.θP)) for k in pbm_covars])


    g, ϕg = get_hybridproblem_MLapplicator(prob; scenario)
    ϕgv = collect(ϕg)
    #
    n_MC = 3; n_MC_val = Val(3)
    n_obs, n_site_all = size(y_o)
    transP, transM  = get_hybridproblem_transforms(prob; scenario)
    #
    cor_ends = get_hybridproblem_cor_ends(prob; scenario)
    approx1 = get_hybridproblem_HVIApproximation(prob; scenario)

    # TODO update init_hybrid_ϕunc
    ϕq, approx = init_hybrid_ϕunc(approx1, cor_ends; transM, par_templates.θM, n_site)
    μ0_bias = 0.9
    μ0_bias = 1.0 # true μP
    ϕqPc1 = CA.ComponentVector(
        μζP = log.(CA.getdata(par_templates.θP .* eltype(par_templates.θP)(μ0_bias))),
        #logσ_ζP = ones(n_θP) .* log(0.01),
        logσ_ζP = ϕq.logσ_ζP,
        ρsP = ϕq.ρsP,
        )
    ϕqIc1 = CA.ComponentVector(
        logσ_ζM_offsets = ϕq.logσ_ζM_offsets,
        ρsM = ϕq.ρsM,
        )
    ϕqP = ϕqP2 = CA.getdata(ϕqPc1)
    ϕqI = CA.getdata(ϕqIc1)
    intϕqP = get_concrete(ComponentArrayInterpreter(ϕqPc1))
    intϕqI = get_concrete(ComponentArrayInterpreter(ϕqIc1))
    #
    indiv_args1 = (;xM = CA.getdata(xM)[:,i_indiv_batch], xP = CA.getdata(xP)[:,i_indiv_batch], 
        y_o = CA.getdata(y_o)[:,i_indiv_batch], y_unc = CA.getdata(y_unc)[:,i_indiv_batch], 
        i_indiv_train = i_indiv_batch) 
    #
    pbm = create_nsite_applicator(get_hybridproblem_PBmodel(prob; scenario), n_MC; diffchunk)      
    nljoint_args_fix =(;
         f! = pbm, 
         fLogLik = get_hybridproblem_neg_logden_obs(prob; scenario)
         )

    # ϕqIcS = CA.ComponentVector(;logσ_ζM_offsets = ϕqIc1.logσ_ζM[2:end] .- ϕqIc1.logσ_ζM[1], ϕqIc1.ρsM)
    # intϕqIS = get_concrete(ComponentArrayInterpreter(ϕqIcS))
    # approxS_ = CP.MeanUniScalingHVIApproximation(ϕqIc1.logσ_ζM[1])
    # approxS = get_hybridproblem_HVIApproximation(prob; scenario)
    n_M = get_numberof_MLinputs(approx, par_templates.θM)
    h0 = CP.prepare_elbo_helpers(approx, ϕg, ϕqP; n_θP, n_θM, n_site, n_MC, n_cov, n_covP, n_M, cor_ends, n_obs);
    h0_point = CP.prepare_elbo_helpers(approx, ϕg, ϕqP; n_θP, n_θM, n_site, n_MC=1, n_cov, n_covP, n_M, cor_ends, n_obs);
    randn!(h0.θsP) # for testing should initialized to finite values
    randn!(h0_point.θsP) # for testing should initialized to finite values
    rnormPM = CP.prepare_rnorm(ϕqP; n_θP, n_θM, n_MC, n_site)
    rnormPM_point = CP.prepare_rnorm(ϕqP; n_θP, n_θM, n_MC=1, n_site)

#-------------------------------------------------------------
@testset "grad_neg_elbo_sites" begin
    local ϕqIc, ϕqPc, rng1, indiv_args, ϕ, sample_args, primal
    ϕqIc = intϕqI(ϕqI)
    ϕqPc = intϕqP(ϕqP)
    #
    rng1 = StableRNG(1234)
    CP.randnPM!(rng1, rnormPM)
    randn!(rng1, ϕgv)
    randn!(rng1, xM)
    indiv_args = indiv_args1
    ϕ = (;ϕg=ϕgv, ϕqP, ϕqI)
    sample_args = (;
        approx, g, is_testmode = true, pbm_covar_indices, intϕqP, intϕqI, cor_ends, 
        transP, transM)
    primal = CP.neg_elbo_sites!(h0, CP.NegElbo(), ϕ, rnormPM, sample_args, indiv_args, nljoint_args_fix)
    function loop_neg_elbo_sites!(n, h0, ϕ, rnormPM, sample_args, indiv_args, nljoint_args_fix)
        for _ in 1:n
            @noinline CP.neg_elbo_sites!(h0, CP.NegElbo(), ϕ, rnormPM, sample_args, indiv_args, nljoint_args_fix)
        end
        nothing
    end
    #@profview_allocs loop_neg_elbo_sites!(100_000, h0, ϕ, rnormPM, sample_args, indiv_args, nljoint_args_fix)
    function alloc_neg_elbo_sites!(h0, ϕ, rnormPM, sample_args, indiv_args, nljoint_args_fix)
        @allocated loop_neg_elbo_sites!(10, h0, ϕ, rnormPM, sample_args, indiv_args, nljoint_args_fix)
    end
    # allocations come from Lux StatefulLuxLayer
    alloc_neg_elbo_sites!(h0, ϕ, rnormPM, sample_args, indiv_args, nljoint_args_fix)
    @test alloc_neg_elbo_sites!(h0, ϕ, rnormPM, sample_args, indiv_args, nljoint_args_fix) <= 128_000
    #
    local sample_args_rev, res0, gradh0
    sample_args_rev = (; sample_args..., is_testmode=false)
    res0, gradh0 = @inferred CP.grad_neg_elbo_sites(
        h0, (;), CP.NegElbo(), ϕ, rnormPM, sample_args, indiv_args, nljoint_args_fix; n_MC_val,
        #n_workers = 3,
        )
    res0_point, gradh0_point = @inferred CP.grad_neg_elbo_sites(
        h0_point, (;), CP.NegLogDensity(), ϕ, rnormPM, sample_args, indiv_args, nljoint_args_fix; n_MC_val,
        #n_workers = 3,
        )
    function loop_grad_neg_elbo_sites(
        n, h0, gradh0, loss_mode, ϕ, rnormPM, sample_args, indiv_args, nljoint_args_fix; n_MC_val)
        for _ in 1:n
            @noinline CP.grad_neg_elbo_sites(
                h0, gradh0, loss_mode, ϕ, rnormPM, sample_args, indiv_args, nljoint_args_fix; n_MC_val)
        end
        nothing
    end
    function alloc_grad_neg_elbo_sites(h0, gradh0, loss_mode, ϕ, rnormPM, sample_args, indiv_args, nljoint_args_fix; n_MC_val)
        # avoid global variables -> pass them through function
        loop_grad_neg_elbo_sites(1, h0, gradh0, loss_mode, ϕ, rnormPM, sample_args, indiv_args, nljoint_args_fix; n_MC_val)
        GC.gc()        
        #@profview_allocs loop_grad_neg_elbo_sites(10_000,h0, gradh0, loss_mode, ϕ, rnormPM, sample_args, indiv_args, nljoint_args_fix; n_MC_val)
        #@profview loop_grad_neg_elbo_sites(10_000,h0, gradh0, loss_mode, ϕ, rnormPM, sample_args, indiv_args, nljoint_args_fix; n_MC_val)
        #@test (@allocated loop_grad_neg_elbo_sites(100,h0, gradh0, loss_mode, ϕ, rnormPM, sample_args, indiv_args, nljoint_args_fix; n_MC_val)) <= 6_748_800
        # TODO inspect why more allocations when calling from runtests or command line
        local alloc
        alloc = @allocated loop_grad_neg_elbo_sites(100,h0, gradh0, loss_mode, ϕ, rnormPM, sample_args, indiv_args, nljoint_args_fix; n_MC_val)
        @test alloc <= 6_753_536  #6_748_800 #
        alloc
    end
    alloc_grad_neg_elbo_sites(h0, gradh0, CP.NegElbo(), ϕ, rnormPM, sample_args, indiv_args, nljoint_args_fix; n_MC_val)
    alloc_grad_neg_elbo_sites(h0_point, gradh0_point, CP.NegLogDensity(), ϕ, rnormPM, sample_args, indiv_args, nljoint_args_fix; n_MC_val)
end

# @testset "grad_neg_elbo_sites_enzyme" begin
#     local ϕqIc, ϕqPc, rng1, indiv_args, ϕ, sample_args, primal
#     ϕqIc = intϕqI(ϕqI)
#     ϕqPc = intϕqP(ϕqP)
#     #
#     rng1 = StableRNG(1234)
#     CP.randnPM!(rng1, rnormPM)
#     randn!(rng1, ϕgv)
#     randn!(rng1, xM)
#     indiv_args = indiv_args1
#     ϕ = (;ϕg=ϕgv, ϕqP, ϕqI)
#     sample_args = (;
#         approx, g, is_testmode = true, pbm_covar_indices, intϕqP, intϕqI, cor_ends, 
#         transP, transM)
#     primal = CP.neg_elbo_sites!(h0_point, CP.NegLogDensity(), ϕ, rnormPM_point, sample_args, indiv_args, nljoint_args_fix)
#     dϕ, _ = CP.grad_neg_elbo_sites_enzyme(
#         h0_point, (;), CP.NegLogDensity(), ϕ, rnormPM_point, sample_args, indiv_args, nljoint_args_fix,
#         )
#     dϕ.ϕqP
# end

@testset "HybridPointSolverSite" begin
    # in order to test inversion, set the dataloader to the synthetic data
    rng1 = rng = StableRNG(111)
    local xM, xP, y_o, y_unc, θP_true, θMs_true, y_true
    (; xM, xP, y_o, y_unc, θP_true, θMs_true, y_true) = HVI.gen_hybridproblem_synthetic(rng1, prob; scenario)
    # constrain to training dataset
    n_indiv = size(y_o,2)
    i_indiv_test = HVI.get_i_sites_test(prob; scenario)
    i_indiv = setdiff(1:n_indiv, i_indiv_test)
    (; xM, xP, y_o, y_unc, θMs_true, y_true) = map(
        (; xM, xP, y_o, y_unc, θMs_true, y_true)) do x
        x[:,i_indiv]
    end
    prob_syn = prob
    y_o2 = get_hybridproblem_train_dataloader(prob; scenario).data[3]
    @test y_o == y_o2 # make sure to use the same synthetic data for good y_true
    solver_point = HybridPointSolverSite(alg=Optimisers.Adam(0.0001))
    #ϕ0 = (;ϕg = copy(CA.getdata(ϕg)), ϕqP=copy(ϕqP), ϕqI=copy(ϕqI))
    res = CommonSolve.solve(prob_syn, solver_point, CP.NegLogDensity(), (;); scenario, 
        epochs=0,  );
    ϕ0 = res.ϕ
    ϕ0.ϕqP[1:length(θP_true)] .+= log(0.9)  # start from 90%
    exp.(ϕ0.ϕqP[1:2]) # updated during solve
    res = CommonSolve.solve(prob_syn, solver_point, CP.NegLogDensity(), ϕ0; scenario, 
        epochs=2_000,
        show_train_loss = Val(true)
        );
    hp, hi_agg = HVI.predict_point!(res.h0, prob_syn, res.ϕ; scenario)
    @test hp.θsP[:,1] ≈ θP_true atol=0.01
    @test mean(abs.(hi_agg.θsM .- θMs_true)) < 0.01
    () -> begin
        exp.(res.ϕ.ϕqP[1:2])
        #@test exp.(intϕqP(res.ϕ.ϕqP).μζP) ≈ θP_true atol = 0.1
        # run forward
        ϕ_tmp = (;res.ϕ.ϕg, ϕ0.ϕqP, ϕ0.ϕqI)
        hp0, hi_agg0 = HVI.predict_point!(res.h0, prob_syn, ϕ0; scenario)
        hpt, hi_aggt = HVI.predict_point!(res.h0, prob_syn, ϕ_tmp; scenario)
        # @usingany UnicodePlots
        i_obs = 3
        #scatterplot(hi_agg.θsM[i_obs,:], θMs_true[i_obs,:])
        scatterplot(y_true[i_obs,:], hi_agg.y[i_obs,:])
        scatterplot(y_true[i_obs,:], hi_agg0.y[i_obs,:])
        scatterplot(y_true[i_obs,:], hi_aggt.y[i_obs,:])
        scatterplot(hi_aggt.y[i_obs,:], hi_agg.y[i_obs,:])
        sum(abs2, hi_agg0.y .- y_true)
        sum(abs2, hi_agg.y .- y_true)
        sum(abs2, hi_aggt.y .- y_true) # with true ζP much worse?
        scatterplot(θMs_true[1,:], hi_agg.θsM[1,:], )
        scatterplot(θMs_true[2,:], hi_agg.θsM[2,:])
    end
 end






