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
    randn!(h0.θsP) # for testing should initialized to finite values
    rnormPM = CP.prepare_rnorm(ϕqP; n_θP, n_θM, n_MC, n_site)

#-------------------------------------------------------------

# @testset "compute_nelboi_z!" begin
#     ϕqIc = intϕqI(ϕqI)
#     ϕqPc = intϕqP(ϕqP)
#     @assert keys(indiv_args1)[1] == :xM
#     xM = indiv_args1.xM; r2end = 2:length(indiv_args1)
#     nljoint_args_inds = NamedTuple{keys(indiv_args1)[r2end]}(values(indiv_args1)[r2end])
#     CP.sample_ζsP!(h0.ζsP, h0.logσ_ζP, approx, rnormPM.P, ϕqPc, cor_ends.P, h0.sample_buffers) # n_P * n_MC
#     CP.g_apply!(h0.ϕms, ϕg, xM, h0.ζsP, nothing, g, h0.xMP, false)     
#     hi1 = h0.helpers_sites[1]
#     nljoint_args_ind1 = NamedTuple{keys(nljoint_args_inds)}(
#         first(zip_eachlastdims(nljoint_args_inds)))
#     rnormM1 = rnormPM.M[1]
#     ϕms1 = h0.ϕms[:,1]
#     θsP1 = h0.θsP
#     CP.compute_nelboi_z!(hi1, approx, rnormM1, nljoint_args_ind1, ϕms1, ϕqIc, θsP1, 
#         cor_ends.M, transM, nljoint_args_fix)     
#     #@code_warntype CP.compute_nelboi_z!(hi1, rnormM1, nljoint_args_ind1, ϕms1, ϕqIc, θsP1)
#     # need two wrap in tmpf to avoid allocations due to boxing
#     function tmpf(hi, approx, rnormM1, nljoint_args_ind1, ϕms1, ϕqIc, θsP1, cor_endsM, transM, nljoint_args_fix)
#         @test (@allocated CP.compute_nelboi_z!(hi, approx, rnormM1, nljoint_args_ind1, ϕms1, 
#             ϕqIc, θsP1, cor_endsM, transM, nljoint_args_fix)) == 0
#     end
#     tmpf(hi1, approx, rnormM1, nljoint_args_ind1, ϕms1, ϕqIc, θsP1, cor_ends.M, transM, nljoint_args_fix)
#         #@profview CP.compute_nelboi_z!(hi1, approx, rnormM1, nljoint_args_ind1, ϕms1, ϕqIc, θsP1, cor_ends.M, transM, nljoint_args_fix)
#         #using BenchmarkTools
#         #@btime CP.compute_nelboi_z!($hi1, $approx, $rnormM1, $nljoint_args_ind1, $ϕms1, $ϕqIc, $θsP1)    
#     #
#     # test with views as input to compute_nelboi_z! and differnt ϕm per MC
#     h21 = h2.helpers_sites[1]
#     CP.g_apply!(h2.ϕms_mcs, ϕg2, xM, h2.ζsP, pbm_covar_indices2, g2, h2.xMP, false)     
#     ϕm = randn!(similar(h2.ϕms_mcs[:,:,1]))
#     inputs = CA.ComponentVector(ϕqIc=ϕqIc, ϕm = ϕm, θsP= θsP1)
#     CP.compute_nelboi_z!(h21, approx, rnormM1, nljoint_args_ind1, inputs.ϕm, inputs.ϕqIc, 
#         inputs.θsP, cor_ends.M, transM, nljoint_args_fix)
#     # with 
#     ϕms1_ = view(inputs, Val(:ϕm))
#     ϕqIc_ = view(inputs, Val(:ϕqIc))
#     θsP1_ = view(inputs, Val(:θsP))
#     function loop_compute_nelboi_z(n, h21, approx, rnormM1, nljoint_args_ind1, ϕms1_, ϕqIc_, θsP1_, cor_endsM, transM, nljoint_args_fix)
#         for _ in 1:n
#             @noinline CP.compute_nelboi_z!(h21, approx, rnormM1, nljoint_args_ind1, ϕms1_, ϕqIc_, θsP1_, cor_endsM, transM, nljoint_args_fix)
#         end
#         nothing
#     end
#     function alloc_compute_nelboi_z(h21, approx, rnormM1, nljoint_args_ind1, ϕms1_, ϕqIc_, θsP1_, cor_endsM, transM, nljoint_args_fix)
#         # avoid global variables -> pass them through function
#         loop_compute_nelboi_z(1, h21, approx, rnormM1, nljoint_args_ind1, ϕms1_, ϕqIc_, θsP1_, cor_endsM, transM, nljoint_args_fix) # thorough warmup
#         #@profview_allocs loop_compute_nelboi_z(10_000,h21, approx, rnormM1, nljoint_args_ind1, ϕms1_, ϕqIc_, θsP1_, cor_endsM, transM, nljoint_args_fix)
#         @test (@allocated loop_compute_nelboi_z(100,h21, approx, rnormM1, nljoint_args_ind1, ϕms1_, ϕqIc_, θsP1_, cor_endsM, transM, nljoint_args_fix)) == 0
#     end
#     alloc_compute_nelboi_z(h21, approx, rnormM1, nljoint_args_ind1, ϕms1_, ϕqIc_, θsP1_, cor_ends.M, transM, nljoint_args_fix)
#     # 
#     gradh2 = @inferred CP.prepare_gradelbo_helpers(inputs.ϕqIc, inputs.ϕm, inputs.θsP, ϕg, ϕqPc, approx; 
#         pbm_covar_indices=pbm_covar_indices2, n_workers=1,
#         h=h2, rnormMi = rnormPM.M[1], nljoint_args_indi=nljoint_args_ind1,
#         n_site, n_cov, cor_ends, transP, transM, nljoint_args_fix, n_MC_val)
#     hw_channel = gradh2.hw_channel
#     # alternative: invoke forwarddiff_grad_nelboi_z! with ϕqIc_ and θsP_ as views,
#     # passing dϕmvecs as the plain array. The views are created once *outside* the
#     # measured region so that their construction is not counted by @allocated.
#     function loop_forwarddiff_grad_nelboi_z(
#         n, hi, approx, rnormM, nljoint_args_indi, i, ϕm_, ϕqIc_, θsP_, dϕmvecs_, dθsP_, hw_channel::Channel, omit_gradient, cor_endsM, transM::Stacked, nljoint_args_fix, n_MC_val)
#         for _ in 1:n
#             @noinline CP.forwarddiff_grad_nelboi_z!(hi, approx, rnormM, nljoint_args_indi, ϕm_, i, 
#                 ϕqIc_, θsP_, dϕmvecs_, dθsP_, hw_channel, cor_endsM, transM, nljoint_args_fix, n_MC_val, omit_gradient)
#         end
#         nothing
#     end
#     ϕm_   = view(inputs, Val(:ϕm))
#     ϕqIc_ = view(inputs, Val(:ϕqIc))
#     θsP_  = view(inputs, Val(:θsP))
#     dϕmvecs = gradh2.dϕmvecs
#     dθsP = gradh2.dθsP
#    function alloc_forwarddiff_grad_nelboi_z(h21, approx, rnormM1, nljoint_args_indi, ϕm_, ϕqIc_, θsP_, dϕmvecs, dθsP_, hw_channel::Channel, cor_endsM, transM::Stacked, nljoint_args_fix, n_MC_val)
#         loop_forwarddiff_grad_nelboi_z(1, h21, approx, rnormM1, nljoint_args_indi, 1, ϕm_, ϕqIc_, θsP_, dϕmvecs, dθsP_, hw_channel, true, cor_endsM, transM, nljoint_args_fix, n_MC_val)
#         @test (@allocated loop_forwarddiff_grad_nelboi_z(100, h21, approx, rnormM1, nljoint_args_indi, 1, ϕm_, ϕqIc_, θsP_, dϕmvecs, dθsP_, hw_channel, true, cor_endsM, transM, nljoint_args_fix, n_MC_val)) == 0
#         loop_forwarddiff_grad_nelboi_z(1, h21, approx, rnormM1, nljoint_args_indi, 1, ϕm_, ϕqIc_, θsP_, dϕmvecs, dθsP_, hw_channel, nothing, cor_endsM, transM, nljoint_args_fix, n_MC_val)
#         @test (@allocated loop_forwarddiff_grad_nelboi_z(100, h21, approx, rnormM1, nljoint_args_indi, 1, ϕm_, ϕqIc_, θsP_, dϕmvecs, dθsP_, hw_channel, nothing, cor_endsM, transM, nljoint_args_fix, n_MC_val)) == 0
#     end
#     alloc_forwarddiff_grad_nelboi_z(h21, approx, rnormM1, nljoint_args_ind1, ϕm_, ϕqIc_, θsP_, dϕmvecs, dθsP, hw_channel, cor_ends.M, transM, nljoint_args_fix, n_MC_val)

#     #@usingany Cthulhu
#     #@descend_code_warntype tmp_g(h21, rnormM1, nljoint_args_ind1, inputs, 1, dϕmvecs, nothing, hw_channel)
#     #@descend_code_warntype tmp_g2(h21, rnormM1, nljoint_args_ind1, inputs, 1, dϕmvecs, nothing, hw_channel)
#     #@usingany BenchmarkTools
#     #@benchmark tmp_g($h21, $rnormM1, $nljoint_args_ind1, $inputs, 1, $dϕmvecs, true, $hw_channel)
#     #@profview_allocs loop_forwarddiff_grad_nelboi_z(10_000, h21, approx, rnormM1, nljoint_args_ind1, 1,  ϕm_, ϕqIc_, θsP_, dϕmvecs, dθsP,   hw_channel,          nothing,       cor_ends.M, transM,         nljoint_args_fix, n_MC_val)

#     approx2 = MeanHVIApproximation()
#     hiM1 = h0M.helpers_sites[1]
#     @test isfinite(CP.compute_nelboi_z!(hiM1, approx2, rnormM1, nljoint_args_ind1, 
#         ϕms1, ϕqIc, θsP1, cor_ends.M, transM, nljoint_args_fix))
#     function alloc_compute_nelboi_zM!(hiM1, approx2, rnormM1, nljoint_args_ind1, 
#         ϕms1, ϕqIc, θsP1, cor_endsM, transM, nljoint_args_fix)
#         @test (@allocated CP.compute_nelboi_z!(hiM1, approx2, rnormM1, nljoint_args_ind1, 
#         ϕms1, ϕqIc, θsP1, cor_endsM, transM, nljoint_args_fix)) == 0 
#     end
#     alloc_compute_nelboi_zM!(hiM1, approx2, rnormM1, nljoint_args_ind1, 
#         ϕms1, ϕqIc, θsP1, cor_ends.M, transM, nljoint_args_fix)
# end

@testset "grad_neg_elbo_sites" begin
    ϕqIc = intϕqI(ϕqI)
    ϕqPc = intϕqP(ϕqP)
    is_testmode = true
    #
    rng1 = StableRNG(1234)
    CP.randnPM!(rng1, rnormPM)
    randn!(rng1, ϕgv)
    randn!(rng1, xM)
    indiv_args = indiv_args1
    ϕ = (;ϕg=ϕgv, ϕqP, ϕqI)
    sample_args = (;
        approx, g, is_testmode, pbm_covar_indices, intϕqP, intϕqI, cor_ends, 
        transP, transM)
    primal = CP.neg_elbo_sites!(h0, ϕ, rnormPM, sample_args, indiv_args, nljoint_args_fix)
    function loop_neg_elbo_sites!(n, h0, ϕ, rnormPM, sample_args, indiv_args, nljoint_args_fix)
        for _ in 1:n
            @noinline CP.neg_elbo_sites!(h0, ϕ, rnormPM, sample_args, indiv_args, nljoint_args_fix)
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
    sample_args_rev = (; sample_args..., is_testmode=false)
    res0, gradh0 = CP.grad_neg_elbo_sites(
        #@descend_code_warntype CP.neg_elbo_sites!(
        h0, (;), ϕ, rnormPM, sample_args, indiv_args, nljoint_args_fix; n_MC_val)
    function loop_grad_neg_elbo_sites(
        n, h0, gradh0, ϕ, rnormPM, sample_args, indiv_args, nljoint_args_fix; n_MC_val)
        for _ in 1:n
            @noinline CP.grad_neg_elbo_sites(
                h0, gradh0, ϕ, rnormPM, sample_args, indiv_args, nljoint_args_fix; n_MC_val)
        end
        nothing
    end
    function alloc_grad_neg_elbo_sites(h0, gradh0, ϕ, rnormPM, sample_args, indiv_args, nljoint_args_fix; n_MC_val)
        # avoid global variables -> pass them through function
        loop_grad_neg_elbo_sites(1, h0, gradh0, ϕ, rnormPM, sample_args, indiv_args, nljoint_args_fix; n_MC_val)
        #@profview_allocs loop_grad_neg_elbo_sites(1_000,h0, gradh0, ϕ, rnormPM, sample_args, indiv_args, nljoint_args_fix; n_MC_val)
        @test (@allocated loop_grad_neg_elbo_sites(100,h0, gradh0, ϕ, rnormPM, sample_args, indiv_args, nljoint_args_fix; n_MC_val)) <= 6_748_800
    end
    alloc_grad_neg_elbo_sites(h0, gradh0, ϕ, rnormPM, sample_args, indiv_args, nljoint_args_fix; n_MC_val)
    #_i_sites_train = 1:n_site    
    #@usingany BenchmarkTools
    #@benchmark CP.grad_neg_elbo_sites($h2, $gradh2, $rnormPM, $ϕg2v, $ϕqP, $ϕqI, $g2, $pbm_covar_indices2; i_sites_train = _i_sites_train, intϕqP=$intϕqP, intϕqI=$intϕqI, xM=$xM, is_testmode = false)
end

@testset "HybridPointSolverSite" begin
    
end






