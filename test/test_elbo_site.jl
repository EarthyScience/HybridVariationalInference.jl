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
    ϕqc1 = CA.ComponentVector(
        μζP = [-1, 0, 1.0], 
        logσ2_ζP = ones(n_MC) .* log(0.01),
        logσ2_ζM = ones(n_MC) .* log(0.02),
        )
    ϕq = ϕq2 = CA.getdata(ϕqc1)
    intϕq = get_concrete(ComponentArrayInterpreter(ϕqc1))
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
    ϕms0v = zero(ϕms0vz)
    CP.g_apply!(ϕms0v, ϕg, xM, ζsP, pbm_covar_indices0, g, xMP0)
    ϕms0v == ϕms0vz
    # @usingany BenchmarkTools
    # @benchmark CP.g_apply!(ϕms0, ϕg, xM, ζsP, pbm_covar_indices0, g, xMP0)
    # with providing nothing instead of an empty list, omit the n_mc dimension
    ϕms0z = CP.g_apply_oop(ϕg, xM, ζsP, nothing, g, xMP0)
    ϕms0 = zero(ϕms0z)
    CP.g_apply!(ϕms0, ϕg, xM, ζsP, nothing, g, xMP0)
    ϕms0 == ϕms0z

    #
    # with popuolation covariates
    pbm_covar_indices2 = Int[2,3]   
    g2, ϕg2 = construct_ChainsApplicator(rng, chain2, Float32)
    ϕg2v = collect(ϕg2)
    xMP=zeros(eltype(xM), (size(xM,1)+ n_covP2), size(xM,2) * n_MC)
    ϕms2z = CP.g_apply_oop(ϕg2, xM, ζsP, pbm_covar_indices2, g2, xMP)
    ϕms2 = zero(ϕms2z)
    CP.g_apply!(ϕms2, ϕg2, xM, ζsP, pbm_covar_indices2, g2, xMP)
    ϕms2 == ϕms2z

    # preallocate helpers and shadows
    h = (;
        rnormP = Matrix{eltype(ϕq)}(undef, n_θP, n_MC),
        ζsP = Matrix{eltype(ϕq)}(undef, n_θP, n_MC),
        logσ2_ζP = Vector{eltype(ϕq)}(undef, n_θP),
        ϕms = Matrix{eltype(ϕg)}(undef, n_M, n_site),
        ϕms_mcs = Array{eltype(ϕg),3}(undef, n_M, n_MC, n_site),
        xMP = Matrix{eltype(ϕg)}(undef, (n_cov + n_covP2), n_MC * n_site),
    )
    helpers_sites = Tuple((;
        rnormM = Matrix{eltype(ϕq)}(undef, n_θM, n_MC),
        ζsM_dc = PAT.DiffCache(Matrix{eltype(ϕq)}(undef, n_θM, n_MC)),
        logσ2_ζM_dc = PAT.DiffCache(Vector{eltype(ϕq)}(undef, n_θM)),
        buffer_nθM_dc = PAT.DiffCache(Vector{eltype(ϕq)}(undef, n_θM)),
    ) for i in 1:n_site)
    h = (;h... , helpers_sites, 
        dxMP  = zero(h.xMP),      # shadow for xMP
        dϕms = zeros(eltype(ϕg2), size(h.ϕms)),
        dϕms_mcs = zeros(eltype(ϕg2), size(h.ϕms_mcs)),
    )   
    h0 = h2 = h 

@testset "sample_ζsM!" begin
    h1 = h.helpers_sites[1]
    # test allocation
    CP.randnζ!(rng, h2)
    ϕm = rand(n_θM+1, n_MC)
    j = 3
    # wrap inside function to aovid allocation due to boxing type unstable globals
    ((ϕm, n_θM,j) -> @allocated ϕm[:,j][1:n_θM])(ϕm,n_θM,j)
    ((ϕm,n_θM,j) -> @allocated view(ϕm,1:n_θM,j))(ϕm, n_θM,j)  
    ζsM = similar(h1.rnormM)
    logσ2_ζM = zeros(n_θM)
    ϕqc = intϕq(ϕq)
    buffer_nθM = zeros(n_θM)
    CP.sample_ζsM!(ζsM, logσ2_ζM, h1.rnormM, ϕqc, ϕm, buffer_nθM)
    @test ((h1) -> @allocated CP.sample_ζsM!(ζsM, logσ2_ζM, h1.rnormM, ϕqc, ϕm, buffer_nθM))(h1) == 0
    #
    # vector version
    CP.randnζ!(rng, h0)
    ϕm1 = ϕm[:,1] 
    CP.sample_ζsM!(ζsM, logσ2_ζM, h1.rnormM, ϕqc, ϕm1, buffer_nθM)
    #allocations because h1 is global
    #  @allocated CP.sample_ζsM!(ζsM, logσ2_ζM, h1.rnormM, ϕqc, ϕm1, buffer_nθM)
    tmpf1 = (h1) -> @allocated CP.sample_ζsM!(ζsM, logσ2_ζM, h1.rnormM, ϕqc, ϕm1, buffer_nθM)
    @test (@allocated tmpf1(h1))  == 0

    # capture global variables in closure
    get_f_fd1 = (h1, intϕq) -> (ϕq, ϕm1, template) -> begin
        local ϕqc = intϕq(ϕq) # without local allocations by @safetestset, shadows global
        ζsMb = PAT.get_tmp(h1.ζsM_dc, template)
        logσ2_ζMb = PAT.get_tmp(h1.logσ2_ζM_dc, template)
        buffer_nθMb = PAT.get_tmp(h1.buffer_nθM_dc, template)
        CP.sample_ζsM!(ζsMb, logσ2_ζMb, h1.rnormM, ϕqc, ϕm1, buffer_nθMb)
        sum(ζsMb) + sum(logσ2_ζMb)
    end
    f_fd1 = get_f_fd1(h1, intϕq)
    f_fd1(ϕq, ϕm1, ϕq)
        # grad_ϕq = ForwardDiff.gradient(f_fd1, ϕq)
        # ϕqd = convert.(typeof(ForwardDiff.Dual(ϕq[1])), ϕq)
        # @allocated f_fd1(ϕqd)
    # vector version
    @test (@allocated f_fd1(ϕq, ϕm1, ϕq)) == 0
    ϕqd = convert.(typeof(ForwardDiff.Dual(ϕq[1])), ϕq)
    @test (@allocated f_fd1(ϕqd, ϕm1, ϕqd)) == 0
    # matrix version
    @test (@allocated f_fd1(ϕq, ϕm, ϕq)) == 0
    @test (@allocated f_fd1(ϕqd, ϕm, ϕqd)) == 0
    #
    grad_ϕq = ForwardDiff.gradient(ϕq -> f_fd1(ϕq, ϕm1, ϕq), ϕq)
    @test all(intϕq(grad_ϕq).logσ2_ζP .== 0)
    grad_ϕm1 = ForwardDiff.gradient(ϕm1 -> f_fd1(ϕq, ϕm1, ϕm1), ϕm1)
    @test grad_ϕm1 == vcat(fill(n_MC, n_θM), 0)
end

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
#     CP.randnζ!(rng, h)
#     res0 = CP.neg_elbo_sites!(
#     #@descend_code_warntype CP.neg_elbo_sites!(
#         h,
#         ϕgv, ϕq, g, nothing;
#         n_MC, 
#         i_sites_train = 1:n_site,     
#         intϕq,
#         xM,
#         is_testmode = false,
#     )    

#     CP.randnζ!(rng, h)
#     res = CP.neg_elbo_sites!(h, ϕg2v, ϕq, g2, pbm_covar_indices2;
#         n_MC, 
#         i_sites_train = 1:n_site,     
#         intϕq,
#         xM,
#         is_testmode = false,
#     )    
# end

# @testset "grad neg_elbo_sites!" begin
#     dh0 = zero(h0)
#     dϕg = zero(ϕgv)
#     dϕq = zero(ϕq)
#     _ftmp2 = (ϕgv, h, g, pbm_covar_indices, ϕq, intϕq, xM, n_MC, i_sites_train) -> 
#         CP.neg_elbo_sites!(
#         h, ϕgv, ϕq, g, pbm_covar_indices;
#         n_MC, 
#         i_sites_train,     
#         intϕq,
#         xM,
#         is_testmode = false,
#         )[1]    
#     pbm_covar_indices_nothing = nothing
#     CP.randnζ!(rng, h)    
#     _ftmp2(ϕgv, h, g, pbm_covar_indices_nothing, ϕq, intϕq, xM, n_MC, 1:n_site)
#     #_f(ϕg2v, h, g2, pbm_covar_indices2)
    
#     Enzyme.make_zero!(dϕg)
#     Enzyme.make_zero!(dϕq)
#     Enzyme.make_zero!(dh0)
#     rng1 = StableRNG(1234)
#     CP.randnζ!(rng1, h)    
#     Enzyme.autodiff(
#             Enzyme.set_runtime_activity(Enzyme.Reverse) ,
#             _ftmp2,
#             Enzyme.Active,
#             Enzyme.Duplicated(ϕgv, dϕg),
#             Enzyme.Duplicated(h, dh0),
#             Enzyme.Const(g),
#             Enzyme.Const(pbm_covar_indices_nothing),
#             Enzyme.Duplicated(ϕq, dϕq),
#             Enzyme.Const(intϕq),
#             Enzyme.Const(xM),
#             Enzyme.Const(n_MC),
#             Enzyme.Const(1:n_site),
#         )   
#     dϕg0_enz = copy(dϕg)
#     dϕq0_enz = copy(dϕq)
#     hcat(dϕg0_enz, dϕg)

#     dh2 = zero(h2)
#     dϕg2 = zero(ϕg2v)
#     dϕq2 = zero(ϕq2)
#     CP.randnζ!(rng1, h)    
#     _ftmp2(ϕg2v, h2, g2, pbm_covar_indices2, ϕq2, intϕq, xM, n_MC, 1:n_site)

#     Enzyme.make_zero!(dϕg2)
#     Enzyme.make_zero!(dϕq2)
#     Enzyme.make_zero!(dh2)
#     rng1 = StableRNG(1234)
#     CP.randnζ!(rng1, h)    
#     Enzyme.autodiff(
#             Enzyme.set_runtime_activity(Enzyme.Reverse) ,
#             _ftmp2,
#             Enzyme.Active,
#             Enzyme.Duplicated(ϕg2v, dϕg2),
#             Enzyme.Duplicated(h2, dh2),
#             Enzyme.Const(g2),
#             Enzyme.Const(pbm_covar_indices2),
#             Enzyme.Duplicated(ϕq2, dϕq2),
#             Enzyme.Const(intϕq),
#             Enzyme.Const(xM),
#             Enzyme.Const(n_MC),
#             Enzyme.Const(1:n_site),
#         )   
#     dϕg2_enz = copy(dϕg2)
#     dϕq2_enz = copy(dϕq2)
#     hcat(dϕg2_enz, dϕg2)
# end

@testset "pullback_sample_ζsP!" begin
    ϕqc = intϕq(ϕq)
    rnormP = zero(ζsP)
    randn!(rnormP)  # before input gaussian noise
    ζsP .= 0
    #logσ2_ζP = zero(ϕqc.logσ2_ζP) # cretes a view rather than copy
    logσ2_ζP = zero(ϕqc.logσ2_ζP)
    CP.sample_ζsP!(ζsP, logσ2_ζP, rnormP, ϕqc)
    mean(ζsP; dims=2)
    ζsP1 = copy(ζsP)

    # Enzyme result via the mutating routine (2-D: n_θP * n_MC × n_in)
    dζsP = zero(ζsP) .+ one(eltype(ζsP))
    dlogσ2_ζP = zero(logσ2_ζP) .+ one(eltype(ζsP))
    dϕqc = zero(ϕqc) 

    randn!(dϕqc) # test that is zerod inside pullback
    dζsP_ = copy(dζsP)
    rnormP_ = copy(rnormP)
    logσ2_ζP_ = copy(logσ2_ζP)
    dlogσ2_ζP_ = copy(dlogσ2_ζP)
    #CP.pullback_sample_ζsP!(dϕqc, dζsP, dlogσ2_ζP, rnormP, logσ2_ζP, ϕqc) # needs rnormP to be noise
    CP.pullback_sample_ζsP!(dϕqc, dζsP, dlogσ2_ζP, ζsP, logσ2_ζP, rnormP, ϕqc)
    @test ζsP == ζsP1 # same forward result
    @test rnormP == rnormP_
    @test dζsP == dζsP_
    @test dlogσ2_ζP == dlogσ2_ζP_
    @test logσ2_ζP == logσ2_ζP_
    # without correlation
    #@test all(dϕqc[Val(:μζP)] .== n_MC)
    # #@test dϕqc[Val(:logσ2_ζP)] ≈ vec(sum(rnormP; dims=2)) # 
    dϕqc_comb = copy(dϕqc)

    randn!(ζsP)  # test initial not relevant
    pb_sample_ζsP = CP.primal_pullback_sample_ζsP!(ζsP, logσ2_ζP, rnormP, ϕqc)
    @test ζsP == ζsP1 # same forward result
    @test rnormP ≈ rnormP_# computed the forward pass
    @test logσ2_ζP == logσ2_ζP_
    dϕqc .= 0.1 # test initial value not relevant
    #dϕqc .= 0.01 # should not influence results
    pb_sample_ζsP(dϕqc, dζsP, dlogσ2_ζP)    
    #pb_sample_ζsP(rnormP, logσ2_ζP)
    @test rnormP ≈ rnormP_  # did not modify
    @test logσ2_ζP == logσ2_ζP_ # not modified
    @test dlogσ2_ζP == dlogσ2_ζP_ # not modified
    @test dζsP == dζsP_
    #hcat(dϕqc, dϕqc_comb)
    @test CA.getdata(dϕqc) ≈ CA.getdata(dϕqc_comb)
    #
    # test another pullback
    #dζsP .= dζsP * eltype(dζsP)(2)
    pb_sample_ζsP(dϕqc, dζsP, dlogσ2_ζP)    
    @test CA.getdata(dϕqc) ≈ CA.getdata(dϕqc_comb)
    #
    # @usingany BenchmarkTools
    # @benchmark pb_sample_ζsP(dϕqc, dζsP, dlogσ2_ζP)    
end







