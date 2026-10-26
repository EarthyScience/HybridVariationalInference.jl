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


rng = StableRNG(1234)

prob = DoubleMM.DoubleMMCase()
scenario = Val((:use_Lux,:usePopPBM,:covarK2,:uniscaling,))
#scenario = Val((:use_Lux,:usePopPBM,:uniscaling,)) # omit covars -> vector site ϕm
diffchunk = ForwardDiff.Chunk(8)

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
    ϕqPc1 = CA.ComponentVector(
        μζP = log.(CA.getdata(par_templates.θP)),
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

# @testset "HybridPointSolverSite" begin
#     hp = CP.prepare_pointloss_helpers(approx, ϕg, ϕqP; n_θP, n_θM, n_site, n_MC, n_cov, n_covP, n_M, cor_ends, n_obs);
#     indiv_args = indiv_args1
#     CP.check_pointloss_helpers(hp, indiv_args.xM, pbm_covar_indices)
#     ϕ = (;ϕg=ϕgv, ϕqP, ϕqI)
#     sample_args = (;
#         approx, g, is_testmode=true, pbm_covar_indices, intϕqP, intϕqI, cor_ends, 
#         transP, transM)
#     nL = CP.pointloss_sites!(hp, ϕ, sample_args, indiv_args, nljoint_args_fix)
#     @test isfinite(nL)
#     function loop_pointloss_sites!(n, hp, ϕ, sample_args, indiv_args, nljoint_args_fix)
#         for _ in 1:n
#             @noinline CP.pointloss_sites!(hp, ϕ, sample_args, indiv_args, nljoint_args_fix)
#         end
#         nothing
#     end
#     # allocations in StatefulLuxLayer
#     #@profview_allocs loop_pointloss_sites!(10_000, hp, ϕ, sample_args, indiv_args, nljoint_args_fix)
#     function alloc_pointloss_sites!(hp, ϕ, sample_args, indiv_args, nljoint_args_fix)
#         loop_pointloss_sites!(1, hp, ϕ, sample_args, indiv_args, nljoint_args_fix)
#         @allocated loop_pointloss_sites!(10, hp, ϕ, sample_args, indiv_args, nljoint_args_fix)
#     end
#     alloc_pointloss_sites!(hp, ϕ, sample_args, indiv_args, nljoint_args_fix)
#     alloc_pointloss_sites!(hp, ϕ, sample_args, indiv_args, nljoint_args_fix) <= 47_200
#     #
#     res0, gradh0 = CP.grad_pointloss_sites(
#         #@descend_code_warntype CP.pointloss_sites!(
#         hp, (;), ϕ, sample_args, indiv_args, nljoint_args_fix)
#     @test res0.dϕqPc.μζP[1] != 0
#     @test res0.dϕqPc.logσ_ζP[1] == 0
#     @test res0.dϕqIc.logσ_ζM_offsets[1] == 0
#     function loop_grad_pointloss_sites(
#         n, hp, gradh0, ϕ, sample_args, indiv_args, nljoint_args_fix)
#         for _ in 1:n
#             @noinline CP.grad_pointloss_sites(
#                 hp, gradh0, ϕ, sample_args, indiv_args, nljoint_args_fix)
#         end
#         nothing
#     end
#     function alloc_grad_pointloss_sites(hp, gradh0, ϕ, sample_args, indiv_args, nljoint_args_fix)
#         # avoid global variables -> pass them through function
#         loop_grad_pointloss_sites(1, hp, gradh0, ϕ, sample_args, indiv_args, nljoint_args_fix)
#         # fewer allocations than calls
#         #@profview_allocs loop_grad_pointloss_sites(1_000,hp, gradh0, ϕ, sample_args, indiv_args, nljoint_args_fix)
#         alloc = @allocated loop_grad_pointloss_sites(100,hp, gradh0, ϕ, sample_args, indiv_args, nljoint_args_fix) 
#         @test alloc <= 3_011_200 
#         alloc
#     end
#     alloc_grad_pointloss_sites(hp, gradh0, ϕ, sample_args, indiv_args, nljoint_args_fix)
#  end






