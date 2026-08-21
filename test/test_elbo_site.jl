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

rng = StableRNG(1234)

n_covP0 = 0
n_covP2 = 2
n_cov = 3
n_M = 4

import Lux
import Zygote
import Enzyme

@testset "g_apply!" begin
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
    #
    n_site = 8
    n_MC = 3
    n_P = 3
    #
    xM = randn(eltype(ϕg),n_cov, n_site)
    y = CP.apply_model(g, xM, ϕg)
    #
    ζP = randn(n_P)
    #
    # without population covariates
    pbm_covar_indices0 = Int[]   
    xMP0=zeros(eltype(xM), size(xM,1) + n_covP0, size(xM,2))
    ϕms = CP.g_apply(ϕg, xM, ζP, pbm_covar_indices0, g, xMP0)
    @test size(ϕms) == (n_M, n_site) 
    #
    # with popuolation covariates
    pbm_covar_indices2 = Int[2,3]   
    g2, ϕg2 = construct_ChainsApplicator(rng, chain2, Float32)
    ϕg2v = collect(ϕg2)
    xMP=zeros(eltype(xM), size(xM,1)+ n_covP2, size(xM,2) )
    ϕms = CP.g_apply(ϕg2, xM, ζP, pbm_covar_indices2, g2, xMP)
    @test size(ϕms) == (n_M, n_site) 
    #
    gr_zygote = Zygote.gradient((ϕg2) -> sum(CP.g_apply_zygote(ϕg2, xM, ζP, pbm_covar_indices2, g2, xMP)), ϕg2v )
    #
    () -> begin # enzyme gradient wihtout providing shadows
        _f = (ϕg2) ->sum(CP.g_apply(ϕg2, xM, ζP, pbm_covar_indices2, g2, xMP))
        _f(ϕg2)
        gr_enzyme = Enzyme.gradient(Reverse, _f, ϕg2)
        #@benchmark  Enzyme.gradient(set_runtime_activity(Reverse), _f, collect(ϕg2))
        gr_enzyme[1] ≈ gr_zygote[1]
    end
    #
    # need to explicitly pass the buffers for y and xMP to avoid allocation
    #  below also their shadows
    _f_ip = (ϕg2, y, xMP) -> begin
        CP.g_apply!(y, ϕg2, xM, ζP, pbm_covar_indices2, g2, xMP)
        sum(y)
    end
    # preallocate helpers and shadows
    h = (;
        xMP = Matrix{eltype(ϕg2)}(undef, n_cov + n_covP2, n_site),
        y = zeros(eltype(ϕg2), n_M, n_site),
    )
    h = (;h... , 
        dxMP  = zero(h.xMP),      # shadow for xMP
        dy    = zero(h.y),        # shadow for y  
    )
    dϕg = zero(ϕg2v)
    L1 = _f_ip(ϕg2, h.y, h.xMP)
    #
    () -> begin  # when primal value is not required, simpler
        # need to set shadows to zero before each gradient computation
        function grad_no_alloc1!(dϕg, _f_ip, ϕg, h)
            fill!(dϕg, zero(eltype(dϕg)))
            fill!(h.dy,    zero(eltype(h.dy)))
            fill!(h.dxMP,  zero(eltype(h.dxMP)))
            # enzymes already accumulates to dϕg
            Enzyme.autodiff(
                Reverse,
                _f_ip,
                Active,
                Duplicated(ϕg, dϕg),
                DuplicatedNoNeed(h.y, h.dy),
                DuplicatedNoNeed(h.xMP, h.dxMP)
            )
        end
        #
        grad_no_alloc1!(dϕg, _f_ip, ϕg2v, h)
        dϕg ≈ gr_zygote[1]
        # still few allocation in Lux
        #@benchmark grad_no_alloc!($dϕg, $_f_ip, $ϕg2v, $h)
        # @profview_allocs for i in 1:10000
        #     Enzyme.gradient(Reverse, _f_ip, ϕg2v, y, xMP)
        # end
    end
    #
    # Compile once outside the hot loop
    fwd, rev = Enzyme.autodiff_thunk(
        Enzyme.ReverseSplitWithPrimal,
        Enzyme.Const{typeof(_f_ip)},          
        Enzyme.Active,                        
        Enzyme.Duplicated{typeof(ϕg2v)},      
        Enzyme.Duplicated{typeof(h.y)},       
        Enzyme.Duplicated{typeof(h.xMP)}      
    )
    function grad_no_alloc!(dϕg, _f_ip, ϕg, h, fwd, rev)
        fill!(dϕg, zero(eltype(dϕg)))
        fill!(h.dy,    zero(eltype(h.dy)))
        fill!(h.dxMP,  zero(eltype(h.dxMP)))
        tape, primal, _ = fwd(
            Enzyme.Const(_f_ip),
            Enzyme.Duplicated(ϕg, dϕg),
            Enzyme.Duplicated(h.y, h.dy),
            Enzyme.Duplicated(h.xMP, h.dxMP)
        )
        rev(
            Enzyme.Const(_f_ip),
            Enzyme.Duplicated(ϕg, dϕg),
            Enzyme.Duplicated(h.y, h.dy),
            Enzyme.Duplicated(h.xMP, h.dxMP),
            one(eltype(ϕg)),
            tape
        )
        return dϕg, primal
    end

    grad_no_alloc!(dϕg, _f_ip, ϕg2v, h, fwd, rev)
    dϕg ≈ gr_zygote[1]
    # @benchmark grad_no_alloc!(dϕg, _f_ip, ϕg2v, h, fwd, rev)    
end





() -> begin
    #using SimpleChains
    chain0 = SimpleChain(
        static(n_cov + n_covP0),          # input size
        TurboDense(tanh, n_M*2),
        TurboDense(identity, n_M)
    )
    chain2 = SimpleChain(
        static(n_cov + n_covP2),          # input size
        TurboDense(tanh, n_M*2),
        TurboDense(identity, n_M)
    )
    ϕg = SimpleChains.init_params(chain0)
    g, ϕg = construct_ChainsApplicator(rng, chain0)
end

() -> begin
    #using Flux
    n_input = n_cov + n_covP0
    chain0 = Flux.Chain(
            # dense layer with bias that maps to 8 outputs and applies `tanh` activation
            Flux.Dense(n_input => n_input * 4, tanh),
            Flux.Dense(n_input * 4 => n_input * 4, tanh),
            # dense layer without bias that maps to n outputs and `logistic` activation
            Flux.Dense(n_input * 4 => n_M, logistic, bias = false)
        )
    n_input = n_cov + n_covP2
    chain2 = Flux.Chain(
            # dense layer with bias that maps to 8 outputs and applies `tanh` activation
            Flux.Dense(n_input => n_input * 4, tanh),
            Flux.Dense(n_input * 4 => n_input * 4, tanh),
            # dense layer without bias that maps to n outputs and `logistic` activation
            Flux.Dense(n_input * 4 => n_M, logistic, bias = false)
        )
end