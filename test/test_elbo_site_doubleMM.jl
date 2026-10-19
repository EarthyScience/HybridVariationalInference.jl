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



    g, ϕg = get_hybridproblem_MLapplicator(prob; scenario)
    ϕgv = collect(ϕg)
    #
    n_MC = 3; n_MC_val = Val(3)
    n_obs, n_site_all = size(y_o)
    transP, transM  = get_hybridproblem_transforms(prob; scenario)
    #
    cor_ends = get_hybridproblem_cor_ends(prob; scenario)
    # TODO update init_hybrid_ϕunc
    ϕqPc1 = CA.ComponentVector(
        μζP = log.(CA.getdata(par_templates.θP)),
        logσ_ζP = ones(n_θP) .* log(0.01),
        ρsP = [0.1],
        )
    ϕqIc1 = CA.ComponentVector(
        logσ_ζM = ones(n_θM) .* log(0.01),
        ρsM = [0.1],
        )
    ϕqP = ϕqP2 = CA.getdata(ϕqPc1)
    ϕqI = CA.getdata(ϕqIc1)
    intϕqP = get_concrete(ComponentArrayInterpreter(ϕqPc1))
    intϕqI = get_concrete(ComponentArrayInterpreter(ϕqIc1))
    #
    indiv_args1 = (;xM = CA.getdata(xM)[:,i_indiv_batch], xP = CA.getdata(xP)[:,i_indiv_batch], 
        y_o = CA.getdata(y_o)[:,i_indiv_batch], y_unc = CA.getdata(y_unc)[:,i_indiv_batch], 
        indiv_train = i_indiv_batch) 
    #
    pbm = create_nsite_applicator(get_hybridproblem_PBmodel(prob; scenario), n_MC; diffchunk)      
    nljoint_args_fix =(;
         f! = pbm, 
         fLogLik = get_hybridproblem_neg_logden_obs(prob; scenario)
         )

    ϕqIcS = CA.ComponentVector(;logσ_ζM_offsets = ϕqIc1.logσ_ζM[2:end] .- ϕqIc1.logσ_ζM[1], ϕqIc1.ρsM)
    intϕqIS = get_concrete(ComponentArrayInterpreter(ϕqIcS))
    approxS_ = CP.MeanUniScalingHVIApproximation(ϕqIc1.logσ_ζM[1])
    approxS = get_hybridproblem_HVIApproximation(prob; scenario)
    n_M = get_numberof_MLinputs(approxS, par_templates.θM)
    h0S = CP.prepare_elbo_helpers(approxS, ϕg, ϕqP; n_θP, n_θM, n_site, n_MC, n_cov, n_covP, n_M, cor_ends, n_obs);
    randn!(h0S.θsP) # for testing should initialized to finite values

#-------------------------------------------------------------

@testset "compute_nelboi_z!" begin
    ϕqIc = intϕqI(ϕqI)
    ϕqPc = intϕqP(ϕqP)
    @assert keys(indiv_args1)[1] == :xM
    xM = indiv_args1.xM; r2end = 2:length(indiv_args1)
    nljoint_args_inds = NamedTuple{keys(indiv_args1)[r2end]}(values(indiv_args1)[r2end])
    CP.sample_ζsP!(h0.ζsP, h0.logσ_ζP, approx, rnormPM.P, ϕqPc, cor_ends.P, h0.sample_buffers) # n_P * n_MC
    CP.g_apply!(h0.ϕms, ϕg, xM, h0.ζsP, nothing, g, h0.xMP, false)     
    hi1 = h0.helpers_sites[1]
    nljoint_args_ind1 = NamedTuple{keys(nljoint_args_inds)}(
        first(zip_eachlastdims(nljoint_args_inds)))
    rnormM1 = rnormPM.M[1]
    ϕms1 = h0.ϕms[:,1]
    θsP1 = h0.θsP
    CP.compute_nelboi_z!(hi1, approx, rnormM1, nljoint_args_ind1, ϕms1, ϕqIc, θsP1, 
        cor_ends.M, transM, nljoint_args_fix)     
    #@code_warntype CP.compute_nelboi_z!(hi1, rnormM1, nljoint_args_ind1, ϕms1, ϕqIc, θsP1)
    # need two wrap in tmpf to avoid allocations due to boxing
    function tmpf(hi, approx, rnormM1, nljoint_args_ind1, ϕms1, ϕqIc, θsP1, cor_endsM, transM, nljoint_args_fix)
        @test (@allocated CP.compute_nelboi_z!(hi, approx, rnormM1, nljoint_args_ind1, ϕms1, 
            ϕqIc, θsP1, cor_endsM, transM, nljoint_args_fix)) == 0
    end
    tmpf(hi1, approx, rnormM1, nljoint_args_ind1, ϕms1, ϕqIc, θsP1, cor_ends.M, transM, nljoint_args_fix)
        #@profview CP.compute_nelboi_z!(hi1, approx, rnormM1, nljoint_args_ind1, ϕms1, ϕqIc, θsP1, cor_ends.M, transM, nljoint_args_fix)
        #using BenchmarkTools
        #@btime CP.compute_nelboi_z!($hi1, $approx, $rnormM1, $nljoint_args_ind1, $ϕms1, $ϕqIc, $θsP1)    
    #
    # test with views as input to compute_nelboi_z! and differnt ϕm per MC
    h21 = h2.helpers_sites[1]
    CP.g_apply!(h2.ϕms_mcs, ϕg2, xM, h2.ζsP, pbm_covar_indices2, g2, h2.xMP, false)     
    ϕm = randn!(similar(h2.ϕms_mcs[:,:,1]))
    inputs = CA.ComponentVector(ϕqIc=ϕqIc, ϕm = ϕm, θsP= θsP1)
    CP.compute_nelboi_z!(h21, approx, rnormM1, nljoint_args_ind1, inputs.ϕm, inputs.ϕqIc, 
        inputs.θsP, cor_ends.M, transM, nljoint_args_fix)
    # with 
    ϕms1_ = view(inputs, Val(:ϕm))
    ϕqIc_ = view(inputs, Val(:ϕqIc))
    θsP1_ = view(inputs, Val(:θsP))
    function loop_compute_nelboi_z(n, h21, approx, rnormM1, nljoint_args_ind1, ϕms1_, ϕqIc_, θsP1_, cor_endsM, transM, nljoint_args_fix)
        for _ in 1:n
            @noinline CP.compute_nelboi_z!(h21, approx, rnormM1, nljoint_args_ind1, ϕms1_, ϕqIc_, θsP1_, cor_endsM, transM, nljoint_args_fix)
        end
        nothing
    end
    function alloc_compute_nelboi_z(h21, approx, rnormM1, nljoint_args_ind1, ϕms1_, ϕqIc_, θsP1_, cor_endsM, transM, nljoint_args_fix)
        # avoid global variables -> pass them through function
        loop_compute_nelboi_z(1, h21, approx, rnormM1, nljoint_args_ind1, ϕms1_, ϕqIc_, θsP1_, cor_endsM, transM, nljoint_args_fix) # thorough warmup
        #@profview_allocs loop_compute_nelboi_z(10_000,h21, approx, rnormM1, nljoint_args_ind1, ϕms1_, ϕqIc_, θsP1_, cor_endsM, transM, nljoint_args_fix)
        @test (@allocated loop_compute_nelboi_z(100,h21, approx, rnormM1, nljoint_args_ind1, ϕms1_, ϕqIc_, θsP1_, cor_endsM, transM, nljoint_args_fix)) == 0
    end
    alloc_compute_nelboi_z(h21, approx, rnormM1, nljoint_args_ind1, ϕms1_, ϕqIc_, θsP1_, cor_ends.M, transM, nljoint_args_fix)
    # 
    gradh2 = @inferred CP.prepare_gradelbo_helpers(inputs.ϕqIc, inputs.ϕm, inputs.θsP, ϕg, ϕqPc, approx; 
        pbm_covar_indices=pbm_covar_indices2, n_workers=1,
        h=h2, rnormMi = rnormPM.M[1], nljoint_args_indi=nljoint_args_ind1,
        n_site, n_cov, cor_ends, transP, transM, nljoint_args_fix, n_MC_val)
    hw_channel = gradh2.hw_channel
    # alternative: invoke forwarddiff_grad_nelboi_z! with ϕqIc_ and θsP_ as views,
    # passing dϕmvecs as the plain array. The views are created once *outside* the
    # measured region so that their construction is not counted by @allocated.
    function loop_forwarddiff_grad_nelboi_z(
        n, hi, approx, rnormM, nljoint_args_indi, i, ϕm_, ϕqIc_, θsP_, dϕmvecs_, dθsP_, hw_channel::Channel, omit_gradient, cor_endsM, transM::Stacked, nljoint_args_fix, n_MC_val)
        for _ in 1:n
            @noinline CP.forwarddiff_grad_nelboi_z!(hi, approx, rnormM, nljoint_args_indi, ϕm_, i, 
                ϕqIc_, θsP_, dϕmvecs_, dθsP_, hw_channel, cor_endsM, transM, nljoint_args_fix, n_MC_val, omit_gradient)
        end
        nothing
    end
    ϕm_   = view(inputs, Val(:ϕm))
    ϕqIc_ = view(inputs, Val(:ϕqIc))
    θsP_  = view(inputs, Val(:θsP))
    dϕmvecs = gradh2.dϕmvecs
    dθsP = gradh2.dθsP
   function alloc_forwarddiff_grad_nelboi_z(h21, approx, rnormM1, nljoint_args_indi, ϕm_, ϕqIc_, θsP_, dϕmvecs, dθsP_, hw_channel::Channel, cor_endsM, transM::Stacked, nljoint_args_fix, n_MC_val)
        loop_forwarddiff_grad_nelboi_z(1, h21, approx, rnormM1, nljoint_args_indi, 1, ϕm_, ϕqIc_, θsP_, dϕmvecs, dθsP_, hw_channel, true, cor_endsM, transM, nljoint_args_fix, n_MC_val)
        @test (@allocated loop_forwarddiff_grad_nelboi_z(100, h21, approx, rnormM1, nljoint_args_indi, 1, ϕm_, ϕqIc_, θsP_, dϕmvecs, dθsP_, hw_channel, true, cor_endsM, transM, nljoint_args_fix, n_MC_val)) == 0
        loop_forwarddiff_grad_nelboi_z(1, h21, approx, rnormM1, nljoint_args_indi, 1, ϕm_, ϕqIc_, θsP_, dϕmvecs, dθsP_, hw_channel, nothing, cor_endsM, transM, nljoint_args_fix, n_MC_val)
        @test (@allocated loop_forwarddiff_grad_nelboi_z(100, h21, approx, rnormM1, nljoint_args_indi, 1, ϕm_, ϕqIc_, θsP_, dϕmvecs, dθsP_, hw_channel, nothing, cor_endsM, transM, nljoint_args_fix, n_MC_val)) == 0
    end
    alloc_forwarddiff_grad_nelboi_z(h21, approx, rnormM1, nljoint_args_ind1, ϕm_, ϕqIc_, θsP_, dϕmvecs, dθsP, hw_channel, cor_ends.M, transM, nljoint_args_fix, n_MC_val)

    #@usingany Cthulhu
    #@descend_code_warntype tmp_g(h21, rnormM1, nljoint_args_ind1, inputs, 1, dϕmvecs, nothing, hw_channel)
    #@descend_code_warntype tmp_g2(h21, rnormM1, nljoint_args_ind1, inputs, 1, dϕmvecs, nothing, hw_channel)
    #@usingany BenchmarkTools
    #@benchmark tmp_g($h21, $rnormM1, $nljoint_args_ind1, $inputs, 1, $dϕmvecs, true, $hw_channel)
    #@profview_allocs loop_forwarddiff_grad_nelboi_z(10_000, h21, approx, rnormM1, nljoint_args_ind1, 1,  ϕm_, ϕqIc_, θsP_, dϕmvecs, dθsP,   hw_channel,          nothing,       cor_ends.M, transM,         nljoint_args_fix, n_MC_val)

    approx2 = MeanHVIApproximation()
    hiM1 = h0M.helpers_sites[1]
    @test isfinite(CP.compute_nelboi_z!(hiM1, approx2, rnormM1, nljoint_args_ind1, 
        ϕms1, ϕqIc, θsP1, cor_ends.M, transM, nljoint_args_fix))
    function alloc_compute_nelboi_zM!(hiM1, approx2, rnormM1, nljoint_args_ind1, 
        ϕms1, ϕqIc, θsP1, cor_endsM, transM, nljoint_args_fix)
        @test (@allocated CP.compute_nelboi_z!(hiM1, approx2, rnormM1, nljoint_args_ind1, 
        ϕms1, ϕqIc, θsP1, cor_endsM, transM, nljoint_args_fix)) == 0 
    end
    alloc_compute_nelboi_zM!(hiM1, approx2, rnormM1, nljoint_args_ind1, 
        ϕms1, ϕqIc, θsP1, cor_ends.M, transM, nljoint_args_fix)
end

@testset "pullback_g_apply!" begin
    #     @test ϕms0 == ϕms0z
    #     @test ϕms2 == ϕms2z
    #     @test size(ϕms0) == (n_M, n_site) 
    #     @test size(ϕms2) == (n_M, n_MC, n_site) 
    #     () -> begin # gradient(sum)
    #         gr_zygote = Zygote.gradient((ϕg2) -> sum(CP.g_apply_oop(ϕg2, xM, ζP, pbm_covar_indices2, g2, xMP)), ϕg2v )
    #         s, pullback_s_zygote = Zygote.pullback((ϕg2) -> sum(CP.g_apply_oop(ϕg2, xM, ζP, pbm_covar_indices2, g2, xMP)), ϕg2v )
    #         #gr_zygote2 = pullback_s_zygote(ones(eltype(ϕg2v), size(ϕg2v)...))
    #         gr_zygote2 = pullback_s_zygote(one(eltype(ϕg2v)))
    #         @test gr_zygote2[1] ≈ gr_zygote[1]
    #         y, pullback_zygote = Zygote.pullback((ϕg2) -> CP.g_apply_oop(ϕg2, xM, ζP, pbm_covar_indices2, g2, xMP), ϕg2v )
    #         gr_zygote3 = pullback_zygote(ones(eltype(y), size(y)...))
    #         gr_zygote3[1] ≈ gr_zygote[1]
    #     end
    #     # 
    #     # concatenate function f3(g(ϕ_g))
    #     f3 = (x) -> sum(3.0 .* x)
    #     CP.g_apply_oop(ϕg2, xM, ζsP, pbm_covar_indices2, g2, xMP)
    #     # one pass of composed function
    #     gr_zygote = Zygote.gradient((ϕg2, ζsP) -> f3(CP.g_apply_oop(ϕg2, xM, ζsP, pbm_covar_indices2, g2, xMP)), ϕg2v, ζsP )
    #     s, pullback_s_zygote = Zygote.pullback((ϕg2) -> f3(CP.g_apply_oop(ϕg2, xM, ζsP, pbm_covar_indices2, g2, xMP)), ϕg2v )
    #     gr_zygote2 = pullback_s_zygote(one(eltype(s)))
    #     @test gr_zygote2[1] ≈ gr_zygote[1]
    #     # mixed AD, differentiate f3 by FowardDiff and pull back through g
    #     y_oop = CP.g_apply_oop(ϕg2, xM, ζsP, pbm_covar_indices2, g2, xMP)
    #     #gr_h = Zygote.gradient(y -> sum(f3(y)), y1)[1]
    #     gr_h = ForwardDiff.gradient(y -> sum(f3(y)), y_oop)
    #     y, pullback_zygote = Zygote.pullback((ϕg2) -> CP.g_apply_oop(ϕg2, xM, ζsP, pbm_covar_indices2, g2, xMP), ϕg2v )
    #     @test y == y_oop
    #     gr_zygote3 = pullback_zygote(gr_h)
    #     gr_zygote3[1] ≈ gr_zygote[1]
    #     #
    #     dϕg = zero(ϕg2v)
    #     dζsP = zero(ζsP)
    #     #Enzyme.make_one!(y)
    #     y .= rand()
    #     dϕg .= rand() # check that initial values do not effect result
    #     dζsP .= rand() # check that initial values do not effect result
    #     dy = convert.(eltype(y), gr_h)
    #     CP.pullback_g_apply!(y, dϕg, dζsP, dy, ϕg2v, xM, ζsP, pbm_covar_indices2, g2, h)
    #     @test y == y_oop
    #     @test dϕg ≈ gr_zygote[1]
    #     @test dζsP ≈ gr_zygote[2] rtol=1e-3
    #     @test dy ≈ gr_h # not modified
    #     #@benchmark CP.pullback_g_apply!(y, dϕg, dy, ϕg2v, xM, ζsP, pbm_covar_indices2, g2, h)
    #     #
        # () -> begin # explicitly splitting the forward and backward pass
        #     # they get cached anymay and require allocating the Duplicated Wrappers twice
        #     #   hence there is no performance benefit
        #     # Compile once outside the hot loop
        #     fwd, rev = Enzyme.autodiff_thunk(
        #         Enzyme.ReverseSplitNoPrimal,
        #         Enzyme.Const{typeof(g_apply!)},
        #         Enzyme.Const,
        #         Enzyme.Duplicated{typeof(y)},
        #         Enzyme.Duplicated{typeof(ϕg2v)},
        #         Enzyme.Const{typeof(xM)},
        #         Enzyme.Const{typeof(ζP)},
        #         Enzyme.Const{typeof(pbm_covar_indices2)},
        #         Enzyme.Const{typeof(g2)},
        #         Enzyme.Duplicated{typeof(h.xMP)}
        #     )
        #     # take care, dy is also modified
        #     function grad2_g_apply!(y, dϕg, dy, ϕg, xM, ζP, pbm_covar_indices, g, h, fwd, rev)
        #         fill!(dϕg, zero(eltype(dϕg)))
        #         fill!(h.dxMP,  zero(eltype(h.dxMP)))
        #         copyto!(h.dy, dy) # copy to avoid modifying dy
        #         tape, _, _ = fwd(
        #             Enzyme.Const(g_apply!),
        #             Enzyme.Duplicated(y, h.dy),
        #             Enzyme.Duplicated(ϕg, dϕg),
        #             Enzyme.Const(xM),
        #             Enzyme.Const(ζP),
        #             Enzyme.Const(pbm_covar_indices),
        #             Enzyme.Const(g),
        #             Enzyme.Duplicated(h.xMP, h.dxMP)
        #         )
        #         rev(
        #             Enzyme.Const(g_apply!),
        #             Enzyme.Duplicated(y, h.dy),
        #             Enzyme.Duplicated(ϕg, dϕg),
        #             Enzyme.Const(xM),
        #             Enzyme.Const(ζP),
        #             Enzyme.Const(pbm_covar_indices),
        #             Enzyme.Const(g),
        #             Enzyme.Duplicated(h.xMP, h.dxMP),
        #             tape
        #         )
        #         return nothing
        #     end

    #         dy = convert.(eltype(y), gr_h) 
    #         grad2_g_apply!(y, dϕg, dy, ϕg2v, xM, ζP, pbm_covar_indices2, g2, h, fwd, rev)
    #         @test y == y_oop
    #         @test dϕg ≈ gr_zygote[1]
    #         @test dy ≈ gr_h # not modified
    #         #@usingany BenchmarkTools
    #         #@benchmark grad2_g_apply!(y, dϕg, dy, ϕg2v, xM, ζP, pbm_covar_indices2, g2, h, fwd, rev)
    #     end
end

@testset "pullback_sample_ζsP!" begin
    #     ϕqPc = intϕqP(ϕqP)
    #     rnormP = zero(ζsP)
    #     randn!(rnormP)  # before input gaussian noise
    #     ζsP .= 0
    #     #logσ_ζP = zero(ϕqPc.logσ_ζP) # cretes a view rather than copy
    #     logσ_ζP = zero(ϕqPc.logσ_ζP)
    #     CP.sample_ζsP!(ζsP, logσ_ζP, rnormP, ϕqPc)
    #     mean(ζsP; dims=2)
    #     ζsP1 = copy(ζsP)

    #     # Enzyme result via the mutating routine (2-D: n_θP * n_MC × n_in)
    #     dζsP = zero(ζsP) .+ one(eltype(ζsP))
    #     dlogσ_ζP = zero(logσ_ζP) .+ one(eltype(ζsP))
    #     dϕqc = zero(ϕqPc) 

    #     randn!(dϕqc) # test that is zerod inside pullback
    #     dζsP_ = copy(dζsP)
    #     rnormP_ = copy(rnormP)
    #     logσ_ζP_ = copy(logσ_ζP)
    #     dlogσ_ζP_ = copy(dlogσ_ζP)
    #     #CP.pullback_sample_ζsP!(dϕqc, dζsP, dlogσ_ζP, rnormP, logσ_ζP, ϕqPc) # needs rnormP to be noise
    #     CP.pullback_sample_ζsP!(dϕqc, dζsP, dlogσ_ζP, ζsP, logσ_ζP, rnormP, ϕqPc)
    #     @test ζsP == ζsP1 # same forward result
    #     @test rnormP == rnormP_
    #     @test dζsP == dζsP_
    #     @test dlogσ_ζP == dlogσ_ζP_
    #     @test logσ_ζP == logσ_ζP_
    #     # without correlation
    #     #@test all(dϕqc[Val(:μζP)] .== n_MC)
    #     # #@test dϕqc[Val(:logσ_ζP)] ≈ vec(sum(rnormP; dims=2)) # 
    #     dϕqc_comb = copy(dϕqc)

    #     randn!(ζsP)  # test initial not relevant
    #     pb_sample_ζsP = CP.primal_pullback_sample_ζsP!(ζsP, logσ_ζP, rnormP, ϕqPc)
    #     @test ζsP == ζsP1 # same forward result
    #     @test rnormP ≈ rnormP_# computed the forward pass
    #     @test logσ_ζP == logσ_ζP_
    #     dϕqc .= 0.1 # test initial value not relevant
    #     #dϕqc .= 0.01 # should not influence results
    #     pb_sample_ζsP(dϕqc, dζsP, dlogσ_ζP)    
    #     #pb_sample_ζsP(rnormP, logσ_ζP)
    #     @test rnormP ≈ rnormP_  # did not modify
    #     @test logσ_ζP == logσ_ζP_ # not modified
    #     @test dlogσ_ζP == dlogσ_ζP_ # not modified
    #     @test dζsP == dζsP_
    #     #hcat(dϕqc, dϕqc_comb)
    #     @test CA.getdata(dϕqc) ≈ CA.getdata(dϕqc_comb)
    #     #
    #     # test another pullback
    #     #dζsP .= dζsP * eltype(dζsP)(2)
    #     pb_sample_ζsP(dϕqc, dζsP, dlogσ_ζP)    
    #     @test CA.getdata(dϕqc) ≈ CA.getdata(dϕqc_comb)
    #     #
    #     # @usingany BenchmarkTools
    #     # @benchmark pb_sample_ζsP(dϕqc, dζsP, dlogσ_ζP)    
end

# to be differentiated, return only first entry of result
function _ftmp2(h0, ϕ, rnormPM, sample_args, indiv_args, nljoint_args_fix) 
    CP.neg_elbo_sites!(h0, ϕ, rnormPM, sample_args, indiv_args, nljoint_args_fix)[1]    
end

function grad_neg_elbo_sites_enzyme() # differentiate entire neg_elbo_sites by enzyme
    # do not use DiffCache here for helpers_sites
    h0p = CP.prepare_elbo_helpers(approx, ϕg, ϕqP; 
        n_θP, n_θM, n_site, n_MC, n_cov, n_covP = n_covP0, n_M, cor_ends, n_obs,
        use_diff_cache = Val(false))
    # and store results to compare to hand-crafted mixed AD
    dh0p = Enzyme.make_zero(h0p)
    @test dh0p !== h0p # real copy rather than reference
    #    
    indiv_args = indiv_args1
    ϕ = (;ϕg=ϕgv, ϕqP, ϕqI)
    dϕ = map(zero, ϕ)
    sample_args = (;
        approx, g, is_testmode, pbm_covar_indices = nothing, intϕqP, intϕqI, cor_ends, 
        transP, transM)
    #
    rng1 = StableRNG(1234)
    CP.randnPM!(rng1, rnormPM)
    randn!(rng1, ϕ.ϕg)
    randn!(rng1, indiv_args.xM)
    primal_enz = tmp = _ftmp2(h0p, ϕ, rnormPM, sample_args, indiv_args, nljoint_args_fix)
    Enzyme.make_zero!(dϕ)
    Enzyme.make_zero!(dh0p)
    Enzyme.autodiff(
            Enzyme.set_runtime_activity(Enzyme.Reverse) ,
            _ftmp2,
            Enzyme.Active,
            Enzyme.Duplicated(h0p, dh0p),
            Enzyme.Duplicated(ϕ, dϕ),
            Enzyme.DuplicatedNoNeed(rnormPM, Enzyme.make_zero(rnormPM)),
            Enzyme.Const(sample_args),
            Enzyme.Const(indiv_args),
            Enzyme.Const(nljoint_args_fix),
        )   
    () -> begin
        dϕg0_enz ≈ dϕ.ϕg
        dϕqP0_enz ≈ dϕ.ϕqP
        dϕqI0_enz ≈ dϕ.ϕqI
        hcat(dϕg0_enz, dϕ.ϕg)
    end
    dϕg0_enz = copy(dϕ.ϕg)
    dϕqP0_enz = copy(dϕ.ϕqP)
    dϕqI0_enz = copy(dϕ.ϕqI)
    () -> begin
        #@usingany JLD2
        #fname = "intermediate/test_enzyme_dphi0.jld2"
        fname = "intermediate/test_enzymeT_dphi0.jld2"
        mkpath("intermediate")
        JLD2.jldsave(fname, false, IOStream; primal_enz, dϕg0_enz, dϕqP0_enz, dϕqI0_enz)
        primal_enz, dϕg0_enz, dϕqP0_enz, dϕqI0_enz = JLD2.load(fname, 
            "primal_enz", "dϕg0_enz", "dϕqP0_enz", "dϕqI0_enz");
    end

    h2p = CP.prepare_elbo_helpers(approx, ϕg2, ϕqP; 
        n_θP, n_θM, n_site, n_MC, n_cov, n_covP = n_covP2, n_M, cor_ends, n_obs,
        use_diff_cache = Val(false))
    dh2p = Enzyme.make_zero(h2p)
    indiv_args = indiv_args1
    ϕ2 = (;ϕg=ϕg2v, ϕqP, ϕqI)
    dϕ2 = map(zero, ϕ2)
    sample_args2 = (;
        approx, g=g2, is_testmode, pbm_covar_indices = pbm_covar_indices2, intϕqP, intϕqI, cor_ends, 
        transP, transM)
    #
    rng1 = StableRNG(1234)
    CP.randnPM!(rng1, rnormPM)
    randn!(rng1, ϕ2.ϕg)
    randn!(rng1, indiv_args.xM)
    primal_enz = _ftmp2(h2p, ϕ2, rnormPM, sample_args2, indiv_args, nljoint_args_fix)
    Enzyme.make_zero!(dϕ2)
    Enzyme.make_zero!(dh2p)
    Enzyme.autodiff(
            Enzyme.set_runtime_activity(Enzyme.Reverse) ,
            _ftmp2,
            Enzyme.Active,
            Enzyme.Duplicated(h2p, dh2p),
            Enzyme.Duplicated(ϕ2, dϕ2),
            Enzyme.DuplicatedNoNeed(rnormPM, Enzyme.make_zero(rnormPM)),
            Enzyme.Const(sample_args2),
            Enzyme.Const(indiv_args),
            Enzyme.Const(nljoint_args_fix),
        )   
    () -> begin
        primal2_enz ≈ primal_enz
        dϕg2_enz ≈ dϕ2.ϕg
        @test dϕqP2_enz ≈ dϕ2.ϕqP  atol=0.001
        dϕqI2_enz ≈ dϕ2.ϕqI
        hcat(dϕg2_enz, dϕ2.ϕg)
        hcat(dϕqP2_enz, dϕ2.ϕqP)
    end
    dϕg2_enz = copy(dϕ2.ϕg)
    dϕqP2_enz = copy(dϕ2.ϕqP)
    dϕqI2_enz = copy(dϕ2.ϕqI)
    () -> begin
        #fname = "intermediate/test_enzyme_dphi2.jld2"
        fname = "intermediate/test_enzymeT_dphi2.jld2"
        mkpath("intermediate")
        JLD2.jldsave(fname, false, IOStream; primal2_enz, dϕg2_enz, dϕqP2_enz, dϕqI2_enz)
        primal2_enz, dϕg2_enz, dϕqP2_enz, dϕqI2_enz = JLD2.load(fname, 
            "primal2_enz", "dϕg2_enz", "dϕqP2_enz", "dϕqI2_enz");
    end
end

function grad_neg_elbo_sites_enzyme_scaled() 
    # now with approxS and respective arguments
    is_testmode = false
    indiv_args = indiv_args1
    ϕ2S = (;ϕg=ϕg2v, ϕqP, ϕqI=CA.getdata(ϕqIcS))
    dϕ2S = map(zero, ϕ2S)
    sample_args2S = (; 
        approx=approxS, g=g2, is_testmode, pbm_covar_indices = pbm_covar_indices2, 
        intϕqP, intϕqI=intϕqIS, cor_ends, 
        transP, transM)
    h2Sp = CP.prepare_elbo_helpers(sample_args2S.approx, ϕg2, ϕqP; 
        n_θP, n_θM, n_site, n_MC, n_cov, n_covP = n_covP2, n_M, cor_ends, use_diff_cache = Val(false))
    dh2Sp = Enzyme.make_zero(h2Sp)
    #
    rng1 = StableRNG(1234)
    CP.randnPM!(rng1, rnormPM)
    randn!(rng1, ϕ2S.ϕg)
    randn!(rng1, indiv_args.xM)
    primal_enz2S = tmp = _ftmp2(h2Sp, ϕ2S, rnormPM, sample_args2S, indiv_args)
    Enzyme.make_zero!(dϕ2S)
    Enzyme.make_zero!(dh2Sp)
    Enzyme.autodiff(
            Enzyme.set_runtime_activity(Enzyme.Reverse) ,
            _ftmp2,
            Enzyme.Active,
            Enzyme.Duplicated(h2Sp, dh2Sp),
            Enzyme.Duplicated(ϕ2S, dϕ2S),
            Enzyme.DuplicatedNoNeed(rnormPM, Enzyme.make_zero(rnormPM)),
            Enzyme.Const(sample_args2S),
            Enzyme.Const(indiv_args),
        )   
    dϕg2S_enz = copy(dϕ2S.ϕg)
    dϕqP2S_enz = copy(dϕ2S.ϕqP)
    dϕqI2S_enz = copy(dϕ2S.ϕqI)
    () -> begin
        #fname = "intermediate/test_enzyme_dphi2.jld2"
        fname = "intermediate/test_enzymeS_dphi2.jld2"
        mkpath("intermediate")
        JLD2.jldsave(fname, false, IOStream; primal_enz2S, dϕg2S_enz, dϕqP2S_enz, dϕqI2S_enz)
        primal_enz2S, dϕg2S_enz, dϕqP2S_enz, dϕqI2S_enz = JLD2.load(fname, 
            "primal_enz2S", "dϕg2S_enz", "dϕqP2S_enz", "dϕqI2S_enz");
    end
end

function tmp_reference_revision()
    #@usingany ReferenceRevision
    ref = open_process(rev="origin/sitepar", path="tmp/ref", use=[:Revise, :ShareAdd])
    ref.eval(:(Pkg.activate("tmp/ref/test"))) 
    #ref.eval(:(Pkg.instantiate()))
    #ref.Pkg.project().path
    ref.eval(:(using HybridVariationalInference))
    ref.eval(:(import HybridVariationalInference as HVI))
    ref.eval(:(using Lux)) # to deserialize objects 
    ref.eval(:(using SharedArrays)) # to deserialize objects 
    #ref.eval(:(@usingany Infiltrator)) # not output during infiltrat in revision
    tmp = ref.HVI.test3(3)
    ref.Revise.retry() # after every change of source file
    res0v, gradh0v = ref.HVI.grad_neg_elbo_sites(
        h0, 
        (;), 
        approx,
        rnormPM,
        ϕgv, ϕqP, ϕqI, g, nothing; #pbm_covar_indices2;
        i_sites_train = 1:n_site,     
        intϕqP, intϕqI,
        xM,
        cor_ends,
        transP, transM, 
        is_testmode = false,
        )    
    close(ref)

    map(k -> k => res0v[k] ≈ res0[k], keys(res0))
    hcat(res0v.dϕqIc, res0.dϕqIc)
end

@testset "grad_neg_elbo_sites" begin
    ϕqIc = intϕqI(ϕqI)
    ϕqPc = intϕqP(ϕqP)
    is_testmode = false
    n_threads_proc = min(Threads.nthreads(), 4)
    n_workers = Distributed.nworkers() * n_threads_proc
    basesize = n_site ÷ Distributed.nworkers()
    distributedEx = Transducers.DistributedEx(;basesize, 
        threads_basesize = Int(ceil(basesize / n_threads_proc))) 
    #
    rng1 = StableRNG(1234)
    CP.randnPM!(rng1, rnormPM)
    randn!(rng1, ϕgv)
    randn!(rng1, xM)
    indiv_args = indiv_args1
    ϕ = (;ϕg=ϕgv, ϕqP, ϕqI)
    sample_args = (;
        approx, g, is_testmode, pbm_covar_indices = nothing, intϕqP, intϕqI, cor_ends, 
        transP, transM)
    primal = CP.neg_elbo_sites!(h0, ϕ, rnormPM, sample_args, indiv_args, nljoint_args_fix)
    res0, gradh0 = CP.grad_neg_elbo_sites(
        #@descend_code_warntype CP.neg_elbo_sites!(
        h0, (;), ϕ, rnormPM, sample_args, indiv_args, nljoint_args_fix; n_MC_val)
    res0_, gradh0_ = CP.grad_neg_elbo_sites( # test deterministic result and distributed
        h0, (;), ϕ, rnormPM, sample_args, indiv_args, nljoint_args_fix; n_MC_val,
        executor = distributedEx
    )    
    @test all(map(≈, res0_,  res0))
    # # if we saved Enzyme results earlier, compare to them
    if isfile("intermediate/test_enzymeT_dphi0.jld2")
        primal_enz, dϕg0_enz, dϕqP0_enz, dϕqI0_enz = JLD2.load(
            "intermediate/test_enzymeT_dphi0.jld2", 
            "primal_enz", "dϕg0_enz", "dϕqP0_enz", "dϕqI0_enz");
        @test primal_enz ≈ primal[1]
        @test dϕg0_enz ≈ res0.dϕg
        @test dϕqI0_enz ≈ res0.dϕqIc
        @test dϕqP0_enz ≈ res0.dϕqPc
        #hcat(dϕqI0_enz, CA.getdata(res0.dϕqIc))
        #hcat(dϕqP0_enz, CA.getdata(res0.dϕqPc))
        #hcat(dϕg0_enz, CA.getdata(res0.dϕg))
        #dϕqP0_enz - CA.getdata(res0.dϕqP)
    end
    #
    #---------------- matrix mode with population covariates
    ϕ2 = (;ϕ..., ϕg=ϕg2v)
    sample_args2 = (;sample_args..., g=g2, pbm_covar_indices = pbm_covar_indices2)
    rng1 = StableRNG(1234)
    CP.randnPM!(rng1, rnormPM)
    randn!(rng1, ϕ2.ϕg)
    randn!(rng1, indiv_args.xM)
    primal2 = CP.neg_elbo_sites!(h2, ϕ2, rnormPM, sample_args2, indiv_args, nljoint_args_fix)[1]
    res0, gradh2 = CP.grad_neg_elbo_sites(h2, (;), ϕ2, rnormPM, sample_args2, indiv_args, nljoint_args_fix; n_MC_val)
    res0_, gradh2_ = CP.grad_neg_elbo_sites(h2, (;), ϕ2, rnormPM, sample_args2, indiv_args, nljoint_args_fix; n_MC_val,
        executor = distributedEx,)    
    @test all(map(≈, res0_,  res0))
    # if we saved Enzyme results earlier, compare to them
    if isfile("intermediate/test_enzymeT_dphi2.jld2")
        primal2_enz, dϕg2_enz, dϕqP2_enz, dϕqI2_enz = JLD2.load(
            "intermediate/test_enzymeT_dphi2.jld2", 
            "primal2_enz", "dϕg2_enz", "dϕqP2_enz", "dϕqI2_enz");
        @test primal2_enz ≈ primal2[1]
        @test dϕg2_enz ≈ res0.dϕg
        @test dϕqI2_enz ≈ res0.dϕqIc
        @test dϕqP2_enz ≈ res0.dϕqPc atol = 0.001
        #hcat(dϕqP2_enz, res0.dϕqP)
    end
    #---------------- approxM with non-empty h.sample_buffers and scaling
    # and using eachcol in PBM function
    ϕ2S = (;ϕ2..., ϕqI=CA.getdata(ϕqIcS))
    sample_args2S = (;sample_args2..., approx=approxS, intϕqI=intϕqIS)
    #nljoint_args_fix2S = nljoint_args_fix =(; f! = pbm_col)
    nljoint_args_fix2S = (; nljoint_args_fix..., f! = pbm_col)
    # h2Sp = CP.prepare_elbo_helpers(sample_args2S.approx, ϕg2, ϕqP; 
    #     n_θP, n_θM, n_site, n_MC, n_cov, n_covP = n_covP2, n_M, cor_ends, use_diff_cache = Val(false))
    rng1 = StableRNG(1234)
    CP.randnPM!(rng1, rnormPM)
    randn!(rng1, ϕ2S.ϕg)
    randn!(rng1, indiv_args.xM)
    primal2S = CP.neg_elbo_sites!(h2S, ϕ2S, rnormPM, sample_args2S, indiv_args, nljoint_args_fix2S)
    res0, gradh2S = CP.grad_neg_elbo_sites(
        h2S, (;), ϕ2S, rnormPM, sample_args2S, indiv_args, nljoint_args_fix2S; n_MC_val, 
        #executor = distributedEx,
    )   
    if isfile("intermediate/test_enzymeS_dphi2.jld2")
        primal_enz2S, dϕg2S_enz, dϕqP2S_enz, dϕqI2S_enz = JLD2.load(
            "intermediate/test_enzymeS_dphi2.jld2", 
            "primal_enz2S", "dϕg2S_enz", "dϕqP2S_enz", "dϕqI2S_enz");
        @test primal_enz2S ≈ primal2S[1]
        @test dϕg2S_enz ≈ res0.dϕg
        @test dϕqI2S_enz ≈ res0.dϕqIc atol=0.01
        @test dϕqP2S_enz ≈ res0.dϕqPc atol=0.01
        #hcat(dϕqP2_enz, res0.dϕqP)
        #hcat(dϕqI2S_enz, res0.dϕqI)
    end

    function loop_grad_neg_elbo_sites(
        n, h2S, gradh2S, ϕ2S, rnormPM, sample_args2S, indiv_args, nljoint_args_fix)
        for _ in 1:n
            @noinline CP.grad_neg_elbo_sites(
                h2S, gradh2S, ϕ2S, rnormPM, sample_args2S, indiv_args, nljoint_args_fix; n_MC_val)
        end
        nothing
    end
    function alloc_grad_neg_elbo_sites(h2S, gradh2S, ϕ2S, rnormPM, sample_args2S, indiv_args, nljoint_args_fix)
        # avoid global variables -> pass them through function
        loop_grad_neg_elbo_sites(1, h2S, gradh2S, ϕ2S, rnormPM, sample_args2S, indiv_args, nljoint_args_fix)
        #@profview_allocs loop_grad_neg_elbo_sites(1_000,h2S, gradh2S, ϕ2S, rnormPM, sample_args2S, indiv_args, nljoint_args_fix)
        @test (@allocated loop_grad_neg_elbo_sites(100,h2S, gradh2S, ϕ2S, rnormPM, sample_args2S, indiv_args, nljoint_args_fix)) <= 3_264_000 #3_260_800
    end
    alloc_grad_neg_elbo_sites(h2S, gradh2S, ϕ2S, rnormPM, sample_args2S, indiv_args, nljoint_args_fix2S)
    #_i_sites_train = 1:n_site    
    #@usingany BenchmarkTools
    #@benchmark CP.grad_neg_elbo_sites($h2, $gradh2, $rnormPM, $ϕg2v, $ϕqP, $ϕqI, $g2, $pbm_covar_indices2; i_sites_train = _i_sites_train, intϕqP=$intϕqP, intϕqI=$intϕqI, xM=$xM, is_testmode = false)

end






