#using LinearAlgebra, BlockDiagonals

using Test
using Zygote
using HybridVariationalInference
using HybridVariationalInference: HybridVariationalInference as CP
using StableRNGs
import CUDA, cuDNN
using GPUArraysCore: GPUArraysCore
using MLDataDevices
using Random
#using SimpleChains
using ComponentArrays: ComponentArrays as CA
using Bijectors
using StableRNGs

#CUDA.device!(4)
rng = StableRNG(111)
ggdev = gpu_device()

const prob = DoubleMM.DoubleMMCase()
scenario = (:default,)

n_θM, n_θP = length.(values(get_hybridproblem_par_templates(prob; scenario)))

(; xM, n_site, θP_true, θMs_true, xP, y_global_true, y_true, y_global_o, y_o
) = gen_hybridproblem_synthetic(rng, prob; scenario)

FT = get_hybridproblem_float_type(prob; scenario)

# set to 0.02 rather than zero for debugging non-zero correlations
cor_ends = (P = 1:n_θP, M = [n_θM])
ρsP = zeros(FT, get_cor_count(cor_ends.P)) .+ FT(0.02)
ρsM = zeros(FT, get_cor_count(cor_ends.M)) .+ FT(0.02)

ϕunc = CA.ComponentVector(;
    logσ2_logP = fill(FT(-10.0), n_θP),
    coef_logσ2_logMs = reduce(hcat, (FT[-10.0, 0.0] for _ in 1:n_θM)),
    ρsP,
    ρsM)

θ_true = θ = CA.ComponentVector(;
    P = θP_true,
    Ms = θMs_true)
transPMs = elementwise(exp) # all parameters on LogNormal scale
ζ_true = inverse(transPMs)(θ_true)
ϕ_true = vcat(ζ_true, CA.ComponentVector(unc = ϕunc))
ϕ_cpu = vcat(ζ_true .+ FT(0.01), CA.ComponentVector(unc = ϕunc))

interpreters = (; pmu = ComponentArrayInterpreter(ϕ_true)) #, M=int_θM, PMs=int_θPMs)

n_MC = 3
@testset "sample_ζ_norm0 cpu" begin
    ϕ = CA.getdata(ϕ_cpu)
    ϕc = interpreters.pmu(ϕ)
    ζ_resid, σ = CP.sample_ζ_norm0(rng, ϕc.P, ϕc.Ms, ϕc.unc; n_MC, cor_ends)
    @test size(ζ_resid) == (length(ϕc.P) + n_θM * n_site, n_MC)
    gr = Zygote.gradient(
        ϕc -> sum(CP.sample_ζ_norm0(
            rng, ϕc.P, ϕc.Ms, ϕc.unc; n_MC, cor_ends)[1]), ϕc)[1]
    @test length(gr) == length(ϕ)
end
#

if ggdev isa MLDataDevices.AbstractGPUDevice
    @testset "sample_ζ_norm0 gpu" begin
        # sample only n_batch of 50
        n_site, n_batch = get_hybridproblem_n_site_and_batch(prob; scenario)
        ϕb = CA.ComponentVector(P = ϕ_cpu.P, Ms = ϕ_cpu.Ms[:,1:n_batch], unc = ϕ_cpu.unc)
        intb = ComponentArrayInterpreter(ϕb)
        ϕ = ggdev(CA.getdata(ϕb))
        #tmp = ϕ[1:6]
        #vec2uutri(tmp)
        ϕc = intb(ϕ)
        @test CA.getdata(ϕc) isa GPUArraysCore.AbstractGPUArray
        #ζP, ζMs, ϕunc = ϕc.P, ϕc.Ms, ϕc.unc
        #urand = CUDA.randn(length(ϕc.P) + length(ϕc.Ms), n_MC) |> gpu
        #include(joinpath(@__DIR__, "uncNN", "elbo.jl")) # callback_loss
        #ζ_resid, σ = sample_ζ_norm0(urand, ϕc.P, ϕc.Ms, ϕc.unc; n_MC)
        #Zygote.gradient(ϕc -> sum(sample_ζ_norm0(urand, ϕc.P, ϕc.Ms, ϕc.unc; n_MC)[1]), ϕc)[1]; 
        int_unc = ComponentArrayInterpreter(ϕc.unc)
        ζ_resid, σ = CP.sample_ζ_norm0(
            rng, CA.getdata(ϕc.P), CA.getdata(ϕc.Ms), CA.getdata(ϕc.unc), int_unc;
            n_MC, cor_ends)
        @test ζ_resid isa GPUArraysCore.AbstractGPUArray
        @test size(ζ_resid) == (length(ϕc.P) + n_θM * n_batch, n_MC)
        gr = Zygote.gradient(
            ϕc -> sum(CP.sample_ζ_norm0(
                rng, CA.getdata(ϕc.P), CA.getdata(ϕc.Ms), CA.getdata(ϕc.unc), int_unc;
                n_MC, cor_ends)[1]), ϕc)[1];
        @test length(gr) == length(ϕ)
        @test CA.getdata(gr) isa GPUArraysCore.AbstractGPUArray
        Array(gr)
        int_unc = ComponentArrayInterpreter(ϕc.unc)
        gr2 = Zygote.gradient(
            ϕc -> sum(CP.sample_ζ_norm0(rng, CA.getdata(ϕc.P), CA.getdata(ϕc.Ms),
                CA.getdata(ϕc.unc), int_unc; n_MC, cor_ends)[1]),
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
