#using LinearAlgebra, BlockDiagonals
using LinearAlgebra
using StatsFuns: logistic

using Test
using HybridVariationalInference
using HybridVariationalInference: HybridVariationalInference as CP
using StableRNGs
using Random
using ComponentArrays: ComponentArrays as CA
#using TransformVariables
using Bijectors
import PreallocationTools as PAT

rng = StableRNG(1234)

n_covP0 = 0
n_covP2 = 2
n_cov = 3
n_θP = 3 
n_θM = 3 
n_M = n_θM + 1 # additional uncertainty scaling factor

import Lux
import Zygote
import Enzyme
import ForwardDiff

    n_input = n_cov + n_covP0
    chain0 = Lux.Chain(
        # dense layer with bias that maps to 8 outputs and applies `tanh` activation
        Lux.Dense(n_input => n_input * 4, tanh),
        Lux.Dense(n_input * 4 => n_input * 4, tanh),
        # dense layer without bias that maps to n outputs and `logistic` activation
        Lux.Dense(n_input * 4 => n_M, logistic, use_bias = false)
    )
    n_input = n_cov + n_covP2
    chain2 = Lux.Chain(
        # dense layer with bias that maps to 8 outputs and applies `tanh` activation
        Lux.Dense(n_input => n_input * 4, tanh),
        Lux.Dense(n_input * 4 => n_input * 4, tanh),
        # dense layer without bias that maps to n outputs and `logistic` activation
        Lux.Dense(n_input * 4 => n_M, logistic, use_bias = false)
    )
    g, ϕg = construct_ChainsApplicator(rng, chain0, Float32)
    ϕgv = collect(ϕg)
    #
    n_site = 8
    n_MC = 3
    #
    ϕqPc1 = CA.ComponentVector(
        μζP = [-1, 0, 1.0], 
        logσ2_ζP = ones(n_MC) .* log(0.01),
        )
    ϕqIc1 = CA.ComponentVector(
        logσ2_ζM = ones(n_MC) .* log(0.02),
        )

    ϕqP = ϕqP2 = CA.getdata(ϕqPc1)
    ϕqI = CA.getdata(ϕqIc1)
    intϕqP = get_concrete(ComponentArrayInterpreter(ϕqPc1))
    intϕqI = get_concrete(ComponentArrayInterpreter(ϕqIc1))
    #
    xM = randn(eltype(ϕg),n_cov, n_site)
    y = CP.apply_model(g, xM, ϕg)
    #
    ζP = randn(n_θP)
    ζsP = ζP .+ 0.1 * randn(n_θP, n_MC)
    #
    # without population covariates
    pbm_covar_indices0 = Int[]   
    xMP0=zeros(eltype(xM), size(xM,1) + n_covP0, size(xM,2))
    ϕms0vz = CP.g_apply_oop(ϕg, xM, ζsP, pbm_covar_indices0, g, xMP0)
    #ϕms0v = convert.(eltype(ϕqP), zero(ϕms0vz))
    ϕms0v = similar(ϕms0vz, eltype(ϕqP))
    CP.g_apply!(ϕms0v, ϕgv, xM, ζsP, pbm_covar_indices0, g, xMP0)
    ϕms0v == ϕms0vz
    # @usingany BenchmarkTools
    # @benchmark CP.g_apply!(ϕms0v, ϕgv, xM, ζsP, pbm_covar_indices0, g, xMP0)
    # tmpf = (ϕms0v, ϕgv, xM, ζsP, pbm_covar_indices0, g, xMP0) -> @allocated CP.g_apply!(ϕms0v, ϕgv, xM, ζsP, pbm_covar_indices0, g, xMP0)
    # tmpf(ϕms0v, ϕgv, xM, ζsP, pbm_covar_indices0, g, xMP0)
    # with providing nothing instead of an empty list, omit the n_mc dimension
    ϕms0z = CP.g_apply_oop(ϕg, xM, ζsP, nothing, g, xMP0)
    ϕms0 = similar(ϕms0z, eltype(ϕqP))
    CP.g_apply!(ϕms0, ϕg, xM, ζsP, nothing, g, xMP0)
    ϕms0 == ϕms0z

    #
    # with popuolation covariates
    pbm_covar_indices2 = Int[2,3]   
    g2, ϕg2 = construct_ChainsApplicator(rng, chain2, Float32)
    ϕg2v = collect(ϕg2)
    xMP=zeros(eltype(xM), (size(xM,1)+ n_covP2), size(xM,2) * n_MC)
    ϕms2z = CP.g_apply_oop(ϕg2, xM, ζsP, pbm_covar_indices2, g2, xMP)
    ϕms2 = similar(ϕms2z, eltype(ϕqP))
    CP.g_apply!(ϕms2, ϕg2, xM, ζsP, pbm_covar_indices2, g2, xMP)
    ϕms2 == ϕms2z

    # preallocate helpers and shadows
    rnormPM = CP.prepare_rnorm(ϕqP; n_θP, n_θM, n_MC, n_site)
    # size(rnormPM.P)
    CP.randnPM!(rng, rnormPM)
    h0 = CP.prepare_elbo_helpers(ϕg, ϕqP; n_θP, n_θM, n_site, n_MC, n_cov, n_covP = n_covP0, n_M)
    h2 = CP.prepare_elbo_helpers(ϕg2, ϕqP; n_θP, n_θM, n_site, n_MC, n_cov, n_covP = n_covP2, n_M)
    CP.check_elbo_helpers(h0, xM, pbm_covar_indices0; n_ϕg = length(ϕg))
    CP.check_elbo_helpers(h2, xM, pbm_covar_indices2; n_ϕg = length(ϕg2))

# @testset "sample_ζsM!" begin
#     h0_1 = h0.helpers_sites[1]
#     # test allocation
#     CP.randnPM!(rng, h2)
#     ϕm = rand(n_θM+1, n_MC)
#     j = 3
#     # wrap inside function to aovid allocation due to boxing type unstable globals
#     ((ϕm, n_θM,j) -> @allocated ϕm[:,j][1:n_θM])(ϕm,n_θM,j)
#     ((ϕm,n_θM,j) -> @allocated view(ϕm,1:n_θM,j))(ϕm, n_θM,j)  
#     ζsM = similar(h0_1.rnormM)
#     logσ2_ζM = zeros(n_θM)
#     ϕqIc = intϕqI(ϕqI)
#     buffer_nθM = zeros(n_θM)
#     CP.sample_ζsM!(ζsM, logσ2_ζM, h0_1.rnormM, ϕqIc, ϕm, buffer_nθM)
#     @test ((h1) -> @allocated CP.sample_ζsM!(ζsM, logσ2_ζM, h1.rnormM, ϕqIc, ϕm, buffer_nθM))(h0_1) == 0
#     #
#     # vector version
#     CP.randnPM!(rng, h0)
#     ϕm1 = ϕm[:,1] 
#     CP.sample_ζsM!(ζsM, logσ2_ζM, h0_1.rnormM, ϕqIc, ϕm1, buffer_nθM)
#     #allocations because h1 is global
#     #  @allocated CP.sample_ζsM!(ζsM, logσ2_ζM, h1.rnormM, ϕqIc, ϕm1, buffer_nθM)
#     tmpf1 = (h1) -> @allocated CP.sample_ζsM!(ζsM, logσ2_ζM, h1.rnormM, ϕqIc, ϕm1, buffer_nθM)
#     @test (@allocated tmpf1(h0_1))  == 0

#     # capture global variables in closure to avoid allocations
#     get_f_fd1 = (h1, intϕqI) -> (ϕqP, ϕm1, template) -> begin
#         local ϕqIc = intϕqI(ϕqP) # without local allocations by @safetestset, shadows global
#         ζsMb = PAT.get_tmp(h1.ζsM_dc, template)
#         logσ2_ζMb = PAT.get_tmp(h1.logσ2_ζM_dc, template)
#         buffer_nθMb = PAT.get_tmp(h1.buffer_nθM_dc, template)
#         CP.sample_ζsM!(ζsMb, logσ2_ζMb, h1.rnormM, ϕqIc, ϕm1, buffer_nθMb)
#         sum(ζsMb) + sum(logσ2_ζMb)
#     end
#     f_fd1 = get_f_fd1(h0_1, intϕqI)
#     f_fd1(ϕqP, ϕm1, ϕqP)
#         # grad_ϕq = ForwardDiff.gradient(f_fd1, ϕqP)
#         # ϕqd = convert.(typeof(ForwardDiff.Dual(ϕqP[1])), ϕqP)
#         # @allocated f_fd1(ϕqd)
#     # vector version
#     @test (@allocated f_fd1(ϕqP, ϕm1, ϕqP)) == 0
#     ϕqd = convert.(typeof(ForwardDiff.Dual(ϕqP[1])), ϕqP)
#     @test (@allocated f_fd1(ϕqd, ϕm1, ϕqd)) == 0
#     # matrix version
#     @test (@allocated f_fd1(ϕqP, ϕm, ϕqP)) == 0
#     @test (@allocated f_fd1(ϕqd, ϕm, ϕqd)) == 0
#     # 
#     # combine vectors so that a single gradient call is enough
#     # first call creates dual storage, call wiht larger vector as template, here combined
#     ϕqm = CA.ComponentArray(; ϕqI, ϕm = ϕm1)
#     grads_ϕ = ForwardDiff.gradient(ϕqm -> f_fd1(ϕqm[Val(:ϕqI)], ϕqm[Val(:ϕm)], CA.getdata(ϕqm)), ϕqm)
#     ftmp_ = (ϕ) -> begin
#         ϕqI_ = view(ϕ, 1:length(ϕqI))
#         ϕm1_ = view(ϕ, length(ϕqI)+1:length(ϕ))
#         f_fd1(ϕqI_, ϕm1_, ϕ)
#     end
#     grads_ϕ = ForwardDiff.gradient(ftmp_, vcat(ϕqI, ϕm1))
#     #
#     # need to calls to gradients for the two vectors
#     grad_ϕm1 = ForwardDiff.gradient(ϕm1 -> f_fd1(ϕqI, ϕm1, ϕm1), ϕm1)
#     @test grad_ϕm1 == vcat(fill(n_MC, n_θM), 0)
#     grad_ϕqI = ForwardDiff.gradient(ϕqI -> f_fd1(ϕqI, ϕm1, ϕqI), ϕqI)
#     #@test all(intϕqI(grad_ϕq).logσ2_ζP .== 0)
#     #
#     @test grads_ϕ[1:length(ϕqI)] == grad_ϕqI
#     @test grads_ϕ[length(ϕqI)+1:end] == grad_ϕm1
# end

# @testset "pullback_g_apply!" begin
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
# end

# @testset "neg_elbo_sites!" begin
#     # @usingany Cthulhu
#     CP.randnPM!(rng, h0)
#     res0 = CP.neg_elbo_sites!(
#     #@descend_code_warntype CP.neg_elbo_sites!(
#         h0,
#         ϕgv, ϕqP, ϕqI, g, nothing;
#         n_MC, 
#         i_sites_train = 1:n_site,     
#         intϕqP, intϕqI,
#         xM,
#         is_testmode = false,
#     )    

#     # matrix version with population covariates
#     CP.randnPM!(rng, h2)
#     res = CP.neg_elbo_sites!(
#         h2, 
#         ϕg2v, ϕqP, ϕqI, g2, pbm_covar_indices2;
#         n_MC, 
#         i_sites_train = 1:n_site,     
#         intϕqP, intϕqI,
#         xM,
#         is_testmode = false,
#     )    
# end

# @testset "pullback_sample_ζsP!" begin
#     ϕqPc = intϕqP(ϕqP)
#     rnormP = zero(ζsP)
#     randn!(rnormP)  # before input gaussian noise
#     ζsP .= 0
#     #logσ2_ζP = zero(ϕqPc.logσ2_ζP) # cretes a view rather than copy
#     logσ2_ζP = zero(ϕqPc.logσ2_ζP)
#     CP.sample_ζsP!(ζsP, logσ2_ζP, rnormP, ϕqPc)
#     mean(ζsP; dims=2)
#     ζsP1 = copy(ζsP)

#     # Enzyme result via the mutating routine (2-D: n_θP * n_MC × n_in)
#     dζsP = zero(ζsP) .+ one(eltype(ζsP))
#     dlogσ2_ζP = zero(logσ2_ζP) .+ one(eltype(ζsP))
#     dϕqc = zero(ϕqPc) 

#     randn!(dϕqc) # test that is zerod inside pullback
#     dζsP_ = copy(dζsP)
#     rnormP_ = copy(rnormP)
#     logσ2_ζP_ = copy(logσ2_ζP)
#     dlogσ2_ζP_ = copy(dlogσ2_ζP)
#     #CP.pullback_sample_ζsP!(dϕqc, dζsP, dlogσ2_ζP, rnormP, logσ2_ζP, ϕqPc) # needs rnormP to be noise
#     CP.pullback_sample_ζsP!(dϕqc, dζsP, dlogσ2_ζP, ζsP, logσ2_ζP, rnormP, ϕqPc)
#     @test ζsP == ζsP1 # same forward result
#     @test rnormP == rnormP_
#     @test dζsP == dζsP_
#     @test dlogσ2_ζP == dlogσ2_ζP_
#     @test logσ2_ζP == logσ2_ζP_
#     # without correlation
#     #@test all(dϕqc[Val(:μζP)] .== n_MC)
#     # #@test dϕqc[Val(:logσ2_ζP)] ≈ vec(sum(rnormP; dims=2)) # 
#     dϕqc_comb = copy(dϕqc)

#     randn!(ζsP)  # test initial not relevant
#     pb_sample_ζsP = CP.primal_pullback_sample_ζsP!(ζsP, logσ2_ζP, rnormP, ϕqPc)
#     @test ζsP == ζsP1 # same forward result
#     @test rnormP ≈ rnormP_# computed the forward pass
#     @test logσ2_ζP == logσ2_ζP_
#     dϕqc .= 0.1 # test initial value not relevant
#     #dϕqc .= 0.01 # should not influence results
#     pb_sample_ζsP(dϕqc, dζsP, dlogσ2_ζP)    
#     #pb_sample_ζsP(rnormP, logσ2_ζP)
#     @test rnormP ≈ rnormP_  # did not modify
#     @test logσ2_ζP == logσ2_ζP_ # not modified
#     @test dlogσ2_ζP == dlogσ2_ζP_ # not modified
#     @test dζsP == dζsP_
#     #hcat(dϕqc, dϕqc_comb)
#     @test CA.getdata(dϕqc) ≈ CA.getdata(dϕqc_comb)
#     #
#     # test another pullback
#     #dζsP .= dζsP * eltype(dζsP)(2)
#     pb_sample_ζsP(dϕqc, dζsP, dlogσ2_ζP)    
#     @test CA.getdata(dϕqc) ≈ CA.getdata(dϕqc_comb)
#     #
#     # @usingany BenchmarkTools
#     # @benchmark pb_sample_ζsP(dϕqc, dζsP, dlogσ2_ζP)    
# end

import JLD2
function grad_neg_elbo_sites_enzyme() # differentiate entire neg_elbo_sites by enzyme
    # and store results to compare to hand-crafted mixed AD
    dh0 = Enzyme.make_zero(h0)
    @test dh0 !== h0 # real copy rather than reference
    dϕg = zero(ϕgv)
    dϕqP = zero(ϕqP)
    dϕqI = zero(ϕqI)
    _ftmp2 = (ϕgv, h, rnormPM, ϕqP, ϕqI, g, pbm_covar_indices, intϕqP, intϕqI, xM, i_sites_train) -> 
        CP.neg_elbo_sites!(
        h, rnormPM, ϕgv, ϕqP, ϕqI, g, pbm_covar_indices;
        i_sites_train,     
        intϕqP, intϕqI,
        xM,
        is_testmode = false,
        )[1]    
    pbm_covar_indices_nothing = nothing
    #_f(ϕg2v, h, g2, pbm_covar_indices2)
    
    Enzyme.make_zero!(dϕg)
    Enzyme.make_zero!(dϕqP)
    Enzyme.make_zero!(dϕqI)
    Enzyme.make_zero!(dh0)
    rng1 = StableRNG(1234)
    CP.randnPM!(rng1, rnormPM)   
    randn!(rng1, ϕgv)
    randn!(rng1, xM)
    primal_enz = _ftmp2(ϕgv, h0, rnormPM, ϕqP, ϕqI, g, pbm_covar_indices_nothing, intϕqP, intϕqI, xM, 1:n_site)
    Enzyme.autodiff(
            Enzyme.set_runtime_activity(Enzyme.Reverse) ,
            _ftmp2,
            Enzyme.Active,
            Enzyme.Duplicated(ϕgv, dϕg),
            Enzyme.Duplicated(h0, dh0),
            Enzyme.DuplicatedNoNeed(rnormPM, Enzyme.make_zero(rnormPM)),
            Enzyme.Duplicated(ϕqP, dϕqP),
            Enzyme.Duplicated(ϕqI, dϕqI),
            Enzyme.Const(g),
            Enzyme.Const(pbm_covar_indices_nothing),
            Enzyme.Const(intϕqP),
            Enzyme.Const(intϕqI),
            Enzyme.Const(xM),
            Enzyme.Const(1:n_site),
        )   
    dϕg0_enz = copy(dϕg)
    dϕqP0_enz = copy(dϕqP)
    dϕqI0_enz = copy(dϕqI)
    () -> begin
        #@usingany JLD2
        fname = "intermediate/test_enzyme_dphi0.jld2"
        mkpath("intermediate")
        JLD2.jldsave(fname, false, IOStream; dϕg0_enz, dϕqP0_enz, dϕqI0_enz)
        dϕg0_enz, dϕqP0_enz, dϕqI0_enz = JLD2.load(fname, "dϕg0_enz", "dϕqP0_enz", "dϕqI0_enz");
    end

    dh2 = Enzyme.make_zero(h2)
    @test dh2 !== h2 # real copy rather than reference
    dϕg2 = zero(ϕg2v)
    _ftmp2(ϕg2v, h2, ϕqP, ϕqI, g2, pbm_covar_indices2, intϕqP, intϕqI, xM, n_MC, 1:n_site)

    Enzyme.make_zero!(dϕg2)
    Enzyme.make_zero!(dϕqP)
    Enzyme.make_zero!(dϕqI)
    Enzyme.make_zero!(dh2)
    Enzyme.autodiff(
            Enzyme.set_runtime_activity(Enzyme.Reverse) ,
            _ftmp2,
            Enzyme.Active,
            Enzyme.Duplicated(ϕg2v, dϕg2),
            Enzyme.Duplicated(h2, dh2),
            Enzyme.Duplicated(ϕqP, dϕqP),
            Enzyme.Duplicated(ϕqI, dϕqI),
            Enzyme.Const(g2),
            Enzyme.Const(pbm_covar_indices2),
            Enzyme.Const(intϕqP),
            Enzyme.Const(intϕqI),
            Enzyme.Const(xM),
            Enzyme.Const(n_MC),
            Enzyme.Const(1:n_site),
        )   
    dϕg2_enz = copy(dϕg2)
    dϕqP2_enz = copy(dϕqP)
    dϕqI2_enz = copy(dϕqI)
    () -> begin
        #@usingany JLD2
        fname = "intermediate/test_enzyme_dphi2.jld2"
        mkpath("intermediate")
        jldsave(fname, false, IOStream; h, dϕg2_enz, dϕqP2_enz, dϕqI2_enz)
    end
end

@testset "grad_neg_elbo_sites" begin
    CP.check_elbo_helpers(h0, xM, nothing; n_ϕg = length(ϕgv))
    rng1 = StableRNG(1234)
    CP.randnPM!(rng1, rnormPM)
    randn!(rng1, ϕgv)
    randn!(rng1, xM)
    primal = CP.neg_elbo_sites!(
        h0, rnormPM,
        ϕgv, ϕqP, ϕqI, g, nothing;
        i_sites_train = 1:n_site,     
        intϕqP, intϕqI,
        xM,
        is_testmode = false,
    )
    res0 = CP.grad_neg_elbo_sites(
    #@descend_code_warntype CP.neg_elbo_sites!(
        h0, rnormPM,
        ϕgv, ϕqP, ϕqI, g, nothing;
        i_sites_train = 1:n_site,     
        intϕqP, intϕqI,
        xM,
        is_testmode = false,
    )    
    res0_ = CP.grad_neg_elbo_sites( # test deterministic result
    #@descend_code_warntype CP.neg_elbo_sites!(
        h0, rnormPM,
        ϕgv, ϕqP, ϕqI, g, nothing;
        i_sites_train = 1:n_site,     
        intϕqP, intϕqI,
        xM,
        is_testmode = false,
    )    
    @test res0_ == res0

    # if we saved Enzyme results earlier, compare to them
    if file.exists("intermediate/test_enzyme_dphi0.jld2")
        dϕg0_enz, dϕqP0_enz, dϕqI0_enz = JLD2.load("intermediate/test_enzyme_dphi0.jld2", "dϕg0_enz", "dϕqP0_enz", "dϕqI0_enz");
        @test dϕg0_enz ≈ res0.dϕg
        @test dϕqI0_enz ≈ res0.dϕqI
        @test dϕqP0_enz ≈ res0.dϕqP
    end

end






