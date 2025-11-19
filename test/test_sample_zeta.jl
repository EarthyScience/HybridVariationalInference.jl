#using LinearAlgebra, BlockDiagonals

using Test
using Zygote
using HybridVariationalInference
using HybridVariationalInference: HybridVariationalInference as CP
using StableRNGs
import CUDA, cuDNN
using GPUArraysCore: GPUArraysCore
using MLDataDevices, Suppressor
using Random
#using SimpleChains
using ComponentArrays: ComponentArrays as CA
using Bijectors
using StableRNGs

#CUDA.device!(4)
rng = StableRNG(111)
ggdev = Suppressor.@suppress gpu_device()
cdev = cpu_device()

prob = DoubleMM.DoubleMMCase()
scenario = Val((:default,))

n_θM, n_θP = length.(values(get_hybridproblem_par_templates(prob; scenario)))

(; xM, θP_true, θMs_true, xP,  y_true,  y_o
) = gen_hybridproblem_synthetic(rng, prob; scenario)
n_site, n_batch = get_hybridproblem_n_site_and_batch(prob; scenario)

FT = get_hybridproblem_float_type(prob; scenario)

# set to 0.02 rather than zero for debugging non-zero correlations
cor_ends = (P=1:n_θP, M=[n_θM])
ρsP = zeros(FT, get_cor_count(cor_ends.P)) .+ FT(0.02)
ρsM = zeros(FT, get_cor_count(cor_ends.M)) .+ FT(0.02)

ϕq = CA.ComponentVector(;
    logσ2_ζP=fill(FT(-10.0), n_θP),
    coef_logσ2_ζMs=reduce(hcat, (FT[-10.0, 0.0] for _ in 1:n_θM)),
    ρsP,
    ρsM)

θ_true = θ = CA.ComponentVector(;
    P=θP_true,
    Ms=θMs_true)
transPMs = elementwise(exp) # all parameters on LogNormal scale
ζ_true = inverse(transPMs)(θ_true)
ϕ_true = vcat(ζ_true, CA.ComponentVector(ϕq=ϕq))
ϕ_cpu = vcat(ζ_true .+ FT(0.01), CA.ComponentVector(ϕq=ϕq))

interpreters = (; pmu=ComponentArrayInterpreter(ϕ_true),
    ϕq=ComponentArrayInterpreter(ϕ_true.ϕq)
) #, M=int_θM, PMs=int_θPMs)

n_MC = 3

@testset "transpose_Ms_sitefirst" begin
    x_true = collect(1:8)
    tmp = Iterators.take(enumerate(Iterators.repeated(x_true)), n_MC)
    collect(tmp)
    Xt = permutedims(stack(map(tmp) do (i, x)
        10 .* i .+ x
    end))
    _nP = 2; _nM = 3; _nsite = 2
    intm_PMs_parfirst = ComponentArrayInterpreter(
        P = (n_MC, _nP), Ms = (n_MC, _nM, _nsite))
    Xtc = intm_PMs_parfirst(Xt)
    #
    X = @inferred CP.transpose_mPMs_sitefirst(Xt, _nP, _nM, _nsite, n_MC)
    # using Cthulhu
    # @descend_code_warntype CP.transpose_mPMs_sitefirst(Xt, _nP, _nM, _nsite, n_MC)
    intm_PMs_sitefirst = ComponentArrayInterpreter(
        P = (n_MC, _nP), Ms = (n_MC, _nsite, _nM))
    Xc = intm_PMs_sitefirst(X)
    #@test Xc.P == @inferred Xtc[:P]  # need to use Val for type stability
    @test Xc.P == @inferred Xtc[Val(:P)]
    @test Xc.Ms[:,1,:] == Xtc.Ms[:,:,1] # first site
    @test Xc.Ms[:,2,:] == Xtc.Ms[:,:,2]
    @test Xc.Ms[:,:,2] == Xtc.Ms[:,2,:] # second parameter
end;

@testset "sample_ζresid_norm" begin
    ϕ = CA.getdata(ϕ_cpu)
    ϕc = interpreters.pmu(ϕ)
    ϕc.ϕq.coef_logσ2_ζMs[1,:] .= (log ∘ abs2).((0.1, 100.0))
    ϕc.ϕq.ρsM .= 0.0
    int_ϕq = get_concrete(ComponentArrayInterpreter(ϕc.ϕq))
    n_MC_pred = 300 # larger n_MC to test σ2
    n_site_batch = size(ϕc.Ms,2)
    app = MeanHVIApproximationMat()
    ζP_resids, ζMs_parfirst_resids, σ = @inferred CP.sample_ζresid_norm(app, rng, ϕc.P, ϕc.Ms, ϕc.ϕq;
        n_MC=n_MC_pred, cor_ends, int_ϕq) 
    #@usingany Cthulhu
    #@descend_code_warntype CP.sample_ζresid_norm(rng, ϕc.P, ϕc.Ms, ϕc.ϕq; n_MC, cor_ends, int_ϕq)
    #@test size(ζ_resid) == (length(ϕc.P) + n_site * n_θM, n_MC)
    n_θM = size(ϕc.Ms,1)
    @test size(ζP_resids) == (n_θP, n_MC_pred)
    @test size(ζMs_parfirst_resids) == (n_θM, n_site_batch, n_MC_pred)
    gr = Zygote.gradient(ϕc -> begin
        ζP_resids, ζMs_parfirst_resids, σ = CP.sample_ζresid_norm(
            app, rng, ϕc.P, ϕc.Ms, ϕc.ϕq;
            n_MC, cor_ends, int_ϕq)
        sum(ζP_resids) + sum(ζMs_parfirst_resids)
    end, ϕc)[1]
    @test length(gr) == length(ϕ)
    #
    n_θM, n_site_batch = size(ϕc.Ms)
    # intm_PMs = ComponentArrayInterpreter(
    #     P = (n_MC_pred, n_θP), Ms = (n_MC_pred, n_site_batch, n_θM))    
    # xc = intm_PMs(ζ_resid)
    # isapprox(std(xc.Ms[:,1,1]), 0.1, rtol = 0.1) # site 1 parameter 1 
    # isapprox(std(xc.Ms[:,:,1]), 0.1, rtol = 0.1) # parameter 1
    # isapprox(std(xc.Ms[:,:,2]), 100.1, rtol = 0.1) # parameter 2
    isapprox(std(ζMs_parfirst_resids[1,1,:]), 0.1, rtol = 0.1) # site 1 parameter 1 
    isapprox(std(ζMs_parfirst_resids[1,:,:]), 0.1, rtol = 0.1) # parameter 1
    isapprox(std(ζMs_parfirst_resids[2,:,:]), 100.1, rtol = 0.1) # parameter 2

    #
    if ggdev isa MLDataDevices.AbstractGPUDevice
        @testset "sample_ζresid_norm gpu" begin
            ϕcd = CP.apply_preserve_axes(ggdev, ϕc); # semicolon necessary
            @test CA.getdata(ϕcd) isa GPUArraysCore.AbstractGPUArray
            #ζP, ζMs, ϕq = ϕc.P, ϕc.Ms, ϕc.ϕq
            #urandn = CUDA.randn(length(ϕc.P) + length(ϕc.Ms), n_MC) |> gpu
            #include(joinpath(@__DIR__, "uncNN", "elbo.jl")) # callback_loss
            #ζ_resid, σ = sample_ζresid_norm(urandn, ϕc.P, ϕc.Ms, ϕc.ϕq; n_MC)
            #Zygote.gradient(ϕc -> sum(sample_ζresid_norm(urandn, ϕc.P, ϕc.Ms, ϕc.ϕq; n_MC)[1]), ϕc)[1]; 
            app = MeanHVIApproximationMat()
            ζP_resids, ζMs_parfirst_resids, σ = @inferred CP.sample_ζresid_norm(
                app, rng, CA.getdata(ϕcd.P), CA.getdata(ϕcd.Ms), CA.getdata(ϕcd.ϕq);
                n_MC = n_MC_pred, cor_ends, int_ϕq)
            #@descend_code_warntype CP.sample_ζresid_norm(rng, CA.getdata(ϕcd.P), CA.getdata(ϕcd.Ms), CA.getdata(ϕcd.ϕq); n_MC = n_MC_pred, cor_ends, int_ϕq)
            @test ζP_resids isa GPUArraysCore.AbstractGPUArray
            @test ζMs_parfirst_resids isa GPUArraysCore.AbstractGPUArray
            @test size(ζP_resids) == (n_θP, n_MC_pred)
            @test size(ζMs_parfirst_resids) == (n_θM, n_site_batch, n_MC_pred)
                    # Zygote gradient for many sites, use fewer sites here
            n_site_few = 20
            ϕcd_few = CA.ComponentVector(; P = ϕcd.P, Ms = ϕcd.Ms[:,1:n_site_few], ϕq = ϕcd.ϕq);
            gr = Zygote.gradient(ϕc -> begin
                ζP_resids, ζMs_parfirst_resids, σ = CP.sample_ζresid_norm(
                    app, rng, CA.getdata(ϕc.P), CA.getdata(ϕc.Ms), CA.getdata(ϕc.ϕq);
                    n_MC, cor_ends, int_ϕq)
                sum(ζP_resids) + sum(ζMs_parfirst_resids)
            end, ϕcd_few)[1];  # semicolon required
            # gr = Zygote.gradient(
            #     ϕc -> sum(CP.sample_ζresid_norm(
            #         rng, CA.getdata(ϕc.P), CA.getdata(ϕc.Ms), CA.getdata(ϕc.ϕq);
            #         n_MC, cor_ends, int_ϕq)[1]), ϕcd_few)[1]; # need semicolon
            # @test CA.getdata(gr) isa GPUArraysCore.AbstractGPUArray
            # CP.apply_preserve_axes(cdev, gr)
            #
            isapprox(std(ζMs_parfirst_resids[1,1,:]), 0.1, rtol = 0.1) # site 1 parameter 1 
            isapprox(std(ζMs_parfirst_resids[1,:,:]), 0.1, rtol = 0.1) # parameter 1
            isapprox(std(ζMs_parfirst_resids[2,:,:]), 100.1, rtol = 0.1) # parameter 2
        end
    end
end
