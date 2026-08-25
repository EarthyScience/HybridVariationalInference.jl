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
    #
    n_site = 8
    n_MC = 3
    #
    ϕqc1 = CA.ComponentVector(μζP = [-0.1, 0.0, 0.1])
    ϕq = CA.getdata(ϕqc1)
    intϕq = ComponentArrayInterpreter(ϕqc1)
    #
    xM = randn(eltype(ϕg),n_cov, n_site)
    y = CP.apply_model(g, xM, ϕg)
    #
    ζP = randn(n_θP)
    #
    # without population covariates
    pbm_covar_indices0 = Int[]   
    xMP0=zeros(eltype(xM), size(xM,1) + n_covP0, size(xM,2))
    ϕms0 = CP.g_apply(ϕg, xM, ζP, pbm_covar_indices0, g, xMP0)
    #
    # with popuolation covariates
    pbm_covar_indices2 = Int[2,3]   
    g2, ϕg2 = construct_ChainsApplicator(rng, chain2, Float32)
    ϕg2v = collect(ϕg2)
    xMP=zeros(eltype(xM), size(xM,1)+ n_covP2, size(xM,2) )
    ϕms2 = CP.g_apply(ϕg2, xM, ζP, pbm_covar_indices2, g2, xMP)

    # preallocate helpers and shadows
    h = (;
        xMP = Matrix{eltype(ϕg)}(undef, n_cov + n_covP2, n_site),
        ζsP = Matrix{eltype(ϕq)}(undef, n_θP, n_MC),
        ϕms = Matrix{eltype(ϕg)}(undef, n_M, n_site),
    )
    helpers_sites = Tuple((;
        ζsM = Matrix{eltype(ϕq)}(undef, n_θM, n_MC),
    ) for i in 1:n_site)
    h = (;h... , helpers_sites, 
        dxMP  = Enzyme.make_zero(h.xMP),      # shadow for xMP
        dy = zeros(eltype(ϕg2), n_M, n_site),
    )    

@testset "g_apply!" begin
    @test size(ϕms0) == (n_M, n_site) 
    @test size(ϕms2) == (n_M, n_site) 
    () -> begin # gradient(sum)
        gr_zygote = Zygote.gradient((ϕg2) -> sum(CP.g_apply_zygote(ϕg2, xM, ζP, pbm_covar_indices2, g2, xMP)), ϕg2v )
        s, pullback_s_zygote = Zygote.pullback((ϕg2) -> sum(CP.g_apply_zygote(ϕg2, xM, ζP, pbm_covar_indices2, g2, xMP)), ϕg2v )
        #gr_zygote2 = pullback_s_zygote(ones(eltype(ϕg2v), size(ϕg2v)...))
        gr_zygote2 = pullback_s_zygote(one(eltype(ϕg2v)))
        @test gr_zygote2[1] ≈ gr_zygote[1]
        y, pullback_zygote = Zygote.pullback((ϕg2) -> CP.g_apply_zygote(ϕg2, xM, ζP, pbm_covar_indices2, g2, xMP), ϕg2v )
        gr_zygote3 = pullback_zygote(ones(eltype(y), size(y)...))
        gr_zygote3[1] ≈ gr_zygote[1]
    end
    # 
    # concatenate function f3(g(ϕ_g))
    f3 = (x) -> sum(3.0 .* x)
    # one pass of composed function
    gr_zygote = Zygote.gradient((ϕg2) -> f3(CP.g_apply_zygote(ϕg2, xM, ζP, pbm_covar_indices2, g2, xMP)), ϕg2v )
    s, pullback_s_zygote = Zygote.pullback((ϕg2) -> f3(CP.g_apply_zygote(ϕg2, xM, ζP, pbm_covar_indices2, g2, xMP)), ϕg2v )
    gr_zygote2 = pullback_s_zygote(one(eltype(s)))
    @test gr_zygote2[1] ≈ gr_zygote[1]
    # mixed AD, differentiate f3 by FowardDiff and pull back through g
    y1 = CP.g_apply_zygote(ϕg2, xM, ζP, pbm_covar_indices2, g2, xMP)
    #gr_h = Zygote.gradient(y -> sum(f3(y)), y1)[1]
    gr_h = ForwardDiff.gradient(y -> sum(f3(y)), y1)
    y, pullback_zygote = Zygote.pullback((ϕg2) -> CP.g_apply_zygote(ϕg2, xM, ζP, pbm_covar_indices2, g2, xMP), ϕg2v )
    @test y == y1
    gr_zygote3 = pullback_zygote(gr_h)
    gr_zygote3[1] ≈ gr_zygote[1]
    #
    _fj = (ϕg2) -> CP.g_apply(ϕg2, xM, ζP, pbm_covar_indices2, g2, xMP)
    y_oop = _fj(ϕg2v)
    #
    () -> begin # enzyme gradient wihtout providing shadows
        _f = (ϕg2) ->sum(CP.g_apply(ϕg2, xM, ζP, pbm_covar_indices2, g2, xMP))
        _f(ϕg2)
        gr_enzyme = Enzyme.gradient(Enzyme.Reverse, _f, ϕg2v)
        #@benchmark  Enzyme.gradient(set_runtime_activity(Reverse), _f, collect(ϕg2))
        gr_enzyme[1] ≈ gr_zygote[1]
    end
    () -> begin # enzyme gradient wihtout providing shadows
        J_enzyme = Enzyme.jacobian(Enzyme.Reverse, _fj, ϕg2v)
        #@benchmark  Enzyme.gradient(set_runtime_activity(Reverse), _f, collect(ϕg2))
        gr_enzyme[1] ≈ gr_zygote[1]
    end
    #
    # need to explicitly pass the buffers for y and xMP to avoid allocation
    #  below also their shadows
    #
    y = Enzyme.make_zero(y_oop)
    CP.g_apply!(y, ϕg2v, xM, ζP, pbm_covar_indices2, g2, xMP)
    @test y ≈ y_oop


    dϕg = Enzyme.make_zero(ϕg2v)
    #Enzyme.make_one!(y)
    y .= rand()
    dϕg .= rand() # check that initial values do not effect result
    dy = convert.(eltype(y), gr_h)
    CP.grad_g_apply!(y, dϕg, dy, ϕg2v, xM, ζP, pbm_covar_indices2, g2, h)
    @test y == y_oop
    @test dϕg ≈ gr_zygote[1]
    @test dy ≈ gr_h # not modified
    #@benchmark grad_g_apply!(y, dϕg, dy, ϕg2v, xM, ζP, pbm_covar_indices2, g2, h)
    #
    () -> begin # explicitly splitting the forward and backward pass
        # they get cached anymay and require allocating the Duplicated Wrappers twice
        #   hence there is no performance benefit
        # Compile once outside the hot loop
        fwd, rev = Enzyme.autodiff_thunk(
            Enzyme.ReverseSplitNoPrimal,
            Enzyme.Const{typeof(g_apply!)},
            Enzyme.Const,
            Enzyme.Duplicated{typeof(y)},
            Enzyme.Duplicated{typeof(ϕg2v)},
            Enzyme.Const{typeof(xM)},
            Enzyme.Const{typeof(ζP)},
            Enzyme.Const{typeof(pbm_covar_indices2)},
            Enzyme.Const{typeof(g2)},
            Enzyme.Duplicated{typeof(h.xMP)}
        )
        # take care, dy is also modified
        function grad2_g_apply!(y, dϕg, dy, ϕg, xM, ζP, pbm_covar_indices, g, h, fwd, rev)
            fill!(dϕg, zero(eltype(dϕg)))
            fill!(h.dxMP,  zero(eltype(h.dxMP)))
            copyto!(h.dy, dy) # copy to avoid modifying dy
            tape, _, _ = fwd(
                Enzyme.Const(g_apply!),
                Enzyme.Duplicated(y, h.dy),
                Enzyme.Duplicated(ϕg, dϕg),
                Enzyme.Const(xM),
                Enzyme.Const(ζP),
                Enzyme.Const(pbm_covar_indices),
                Enzyme.Const(g),
                Enzyme.Duplicated(h.xMP, h.dxMP)
            )
            rev(
                Enzyme.Const(g_apply!),
                Enzyme.Duplicated(y, h.dy),
                Enzyme.Duplicated(ϕg, dϕg),
                Enzyme.Const(xM),
                Enzyme.Const(ζP),
                Enzyme.Const(pbm_covar_indices),
                Enzyme.Const(g),
                Enzyme.Duplicated(h.xMP, h.dxMP),
                tape
            )
            return nothing
        end
        dy = convert.(eltype(y), gr_h) 
        grad2_g_apply!(y, dϕg, dy, ϕg2v, xM, ζP, pbm_covar_indices2, g2, h, fwd, rev)
        @test y == y_oop
        @test dϕg ≈ gr_zygote[1]
        @test dy ≈ gr_h # not modified
        #@usingany BenchmarkTools
        #@benchmark grad2_g_apply!(y, dϕg, dy, ϕg2v, xM, ζP, pbm_covar_indices2, g2, h, fwd, rev)
    end

end

@testset "neg_elbo_sites" begin
    res = CP.neg_elbo_sites(rng, ϕg2v, ϕq, g2;
        n_MC, 
        pbm_covar_indices = pbm_covar_indices2,
        i_sites_train = 1:n_site,     
        intϕq,
        elbo_helpers = h,      # tuple of preallocated arrays
        xM,
    )    
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