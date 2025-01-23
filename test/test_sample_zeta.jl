#using LinearAlgebra, BlockDiagonals

using Test
using Zygote
using HybridVariationalInference
using HybridVariationalInference: HybridVariationalInference as CP
using StableRNGs
using CUDA
using GPUArraysCore: GPUArraysCore
using Random
#using SimpleChains
using ComponentArrays: ComponentArrays as CA
using Bijectors

#CUDA.device!(4)
rng = StableRNG(111)

const case = DoubleMM.DoubleMMCase()
#const MLengine = Val(nameof(SimpleChains))
scenario = (:default,)

(; n_covar, n_batch, n_θM, n_θP) = get_hybridcase_sizes(case; scenario)

(; xM, n_site, θP_true, θMs_true, xP, y_global_true, y_true, y_global_o, y_o
) = gen_hybridcase_synthetic(case, rng; scenario)

# n_site = 2
# n_θP, n_θM = length(θ_true.θP), length(θ_true.θM)
# σ_θM = θ_true.θM .* 0.1  # 10% around expected
# θMs_true = θ_true.θM .+ randn(n_θM, n_site) .* σ_θM 

# set to 0.02 rather than zero for debugging non-zero correlations
ρsP = zeros(sum(1:(n_θP-1))) .+ 0.02
ρsM = zeros(sum(1:(n_θM-1))) .+ 0.02

ϕunc = CA.ComponentVector(;
    logσ2_logP=fill(-10.0, n_θP),
    coef_logσ2_logMs=reduce(hcat, ([-10.0, 0.0] for _ in 1:n_θM)),
    ρsP,
    ρsM)

θ_true = θ = CA.ComponentVector(;
    P=θP_true,
    Ms=θMs_true)
transPMs = elementwise(exp) # all parameters on LogNormal scale
ζ_true = inverse(transPMs)(θ_true)
ϕ_true = vcat(ζ_true, CA.ComponentVector(unc=ϕunc))
ϕ_cpu = vcat(ζ_true .+ 0.01, CA.ComponentVector(unc=ϕunc))

interpreters = (; pmu=ComponentArrayInterpreter(ϕ_true)) #, M=int_θM, PMs=int_θPMs)

n_MC = 3
@testset "sample_ζ_norm0 cpu" begin
    ϕ = CA.getdata(ϕ_cpu)
    ϕc = interpreters.pmu(ϕ)
    cor_starts=(P=(1,),M=(1,))
    ζ_resid, σ = CP.sample_ζ_norm0(rng, ϕc.P, ϕc.Ms, ϕc.unc; n_MC, cor_starts)
    @test size(ζ_resid) == (length(ϕc.P) + n_θM * n_site, n_MC)
    gr = Zygote.gradient(ϕc -> sum(CP.sample_ζ_norm0(
        rng, ϕc.P, ϕc.Ms, ϕc.unc; n_MC, cor_starts)[1]), ϕc)[1]
    @test length(gr) == length(ϕ)
end
#

if CUDA.functional()
    @testset "sample_ζ_norm0 gpu" begin
        ϕ = CuArray(CA.getdata(ϕ_cpu))
        cor_starts=(P=(1,),M=(1,))
        #tmp = ϕ[1:6]
        #vec2uutri(tmp)
        ϕc = interpreters.pmu(ϕ);
        @test CA.getdata(ϕc) isa GPUArraysCore.AbstractGPUArray
        #ζP, ζMs, ϕunc = ϕc.P, ϕc.Ms, ϕc.unc
        #urand = CUDA.randn(length(ϕc.P) + length(ϕc.Ms), n_MC) |> gpu
        #include(joinpath(@__DIR__, "uncNN", "elbo.jl")) # callback_loss
        #ζ_resid, σ = sample_ζ_norm0(urand, ϕc.P, ϕc.Ms, ϕc.unc; n_MC)
        #Zygote.gradient(ϕc -> sum(sample_ζ_norm0(urand, ϕc.P, ϕc.Ms, ϕc.unc; n_MC)[1]), ϕc)[1]; 
        ζ_resid, σ = CP.sample_ζ_norm0(rng, ϕc.P, ϕc.Ms, ϕc.unc; n_MC, cor_starts)
        @test ζ_resid isa GPUArraysCore.AbstractGPUArray
        @test size(ζ_resid) == (length(ϕc.P) + n_θM * n_site, n_MC)
        gr = Zygote.gradient(
            ϕc -> sum(CP.sample_ζ_norm0(rng, ϕc.P, ϕc.Ms, ϕc.unc; n_MC, cor_starts)[1]), ϕc)[1];
        @test length(gr) == length(ϕ)
        int_unc = ComponentArrayInterpreter(ϕc.unc)
        gr2 = Zygote.gradient(
            ϕc -> sum(CP.sample_ζ_norm0(rng, CA.getdata(ϕc.P), CA.getdata(ϕc.Ms),
                CA.getdata(ϕc.unc), int_unc; n_MC, cor_starts)[1]),
            ϕc)[1];
    end
end

# @testset "generate_ζ" begin
#     ϕ = CA.getdata(ϕ_cpu)
#     n_sample_pred = 200
#     intm_PMs_gen = ComponentArrayInterpreter(CA.ComponentVector(; θP_true,
#         θMs=CA.ComponentMatrix(
#             zeros(n_θM, n_site), first(CA.getaxes(θMs_true)), CA.Axis(i=1:n_sample_pred))))
#     int_μP_ϕg_unc=ComponentArrayInterpreter(ϕ_true)
#     interpreters = (; PMs = intm_PMs_gen, μP_ϕg_unc = int_μP_ϕg_unc  )
#     ζs, _ = CP.generate_ζ(rng, g, ϕ, xM, interpreters; n_MC=n_sample_pred)

# end;

