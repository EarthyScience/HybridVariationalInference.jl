#using LinearAlgebra, BlockDiagonals

using Test
using HybridVariationalInference
using HybridVariationalInference: HybridVariationalInference as CP
using StableRNGs
using Random
using SimpleChains
using ComponentArrays: ComponentArrays as CA
#using TransformVariables
using Bijectors
using Zygote
using CUDA
using GPUArraysCore: GPUArraysCore

#CUDA.device!(4)
rng = StableRNG(111)

const case = DoubleMM.DoubleMMCase()
const MLengine = Val(nameof(SimpleChains))
scenario = (:default,)
FT = get_hybridcase_FloatType(case; scenario)

#θsite_true = get_hybridcase_par_templates(case; scenario)
g, ϕg0 = get_hybridcase_MLapplicator(case, MLengine; scenario);
f = get_hybridcase_PBmodel(case; scenario)

(; n_covar, n_batch, n_θM, n_θP) = get_hybridcase_sizes(case; scenario)

(; xM, n_site, θP_true, θMs_true, xP, y_global_true, y_true, y_global_o, y_o, σ_o
) = gen_hybridcase_synthetic(case, rng; scenario);

logσ2y = FT(2) .* log.(σ_o)
n_MC = 3
(; transP, transM) = get_hybridcase_transforms(case; scenario)
# transP = elementwise(exp)
# transM = Stacked(elementwise(identity), elementwise(exp))
#transM = Stacked(elementwise(identity), elementwise(exp), elementwise(exp)) # test mismatch
(; ϕ, transPMs_batch, interpreters, get_transPMs, get_ca_int_PMs) = init_hybrid_params(
    θP_true, θMs_true[:, 1], ϕg0, n_batch; transP, transM);
ϕ_ini = ϕ

() -> begin
    # correlation matrices
    ρsP = zeros(sum(1:(n_θP - 1)))
    ρsM = zeros(sum(1:(n_θM - 1)))

    () -> begin
        coef_logσ2_logMs = [-5.769 -3.501; -0.01791 0.007951]
        logσ2_logP = CA.ComponentVector(r0 = -8.997, K2 = -5.893)
        #mean_σ_o_MC = 0.006042

        ϕunc = CA.ComponentVector(;
            logσ2_logP = logσ2_logP,
            coef_logσ2_logMs = coef_logσ2_logMs,
            ρsP,
            ρsM)
    end

    # for a conservative uncertainty assume σ2=1e-10 and no relationship with magnitude
    logσ2y = 2 .* log.(σ_o)
    ϕunc0 = CA.ComponentVector(;
        logσ2_logP = fill(-10.0, n_θP),
        coef_logσ2_logMs = reduce(hcat, ([-10.0, 0.0] for _ in 1:n_θM)),
        ρsP,
        ρsM)
    #int_unc = ComponentArrayInterpreter(ϕunc0)

    transPMs_batch = as(
        (P = as(Array, asℝ₊, n_θP),
        Ms = as(Array, asℝ₊, n_θM, n_batch)))
    transPMs_allsites = as(
        (P = as(Array, asℝ₊, n_θP),
        Ms = as(Array, asℝ₊, n_θM, n_site)))

    ϕ_true = θ = CA.ComponentVector(;
        μP = θP_true,
        ϕg = ϕg0, #ϕg_opt,  # here start from randomized
        unc = ϕunc0)
    trans_gu = as(
        (μP = as(Array, asℝ₊, n_θP),
        ϕg = as(Array, length(ϕg0)),
        unc = as(Array, length(ϕunc0))))
    trans_g = as(
        (μP = as(Array, asℝ₊, n_θP),
        ϕg = as(Array, length(ϕg0))))

    int_PMs_batch = ComponentArrayInterpreter(CA.ComponentVector(; θP = θP_true,
        θMs = CA.ComponentMatrix(
            zeros(n_θM, n_batch), first(CA.getaxes(θMs_true)), CA.Axis(i = 1:n_batch))))

    interpreters = map(get_concrete,
        (;
            μP_ϕg_unc = ComponentArrayInterpreter(ϕ_true),
            PMs = int_PMs_batch,
            unc = ComponentArrayInterpreter(ϕunc0)
        ))

    ϕg_true_vec = CA.ComponentVector(
        TransformVariables.inverse(trans_gu, CP.cv2NamedTuple(ϕ_true)))
    ϕcg_true = interpreters.μP_ϕg_unc(ϕg_true_vec)
    ϕ_ini = ζ = vcat(ϕcg_true[[:μP, :ϕg]] .* 1.2, ϕcg_true[[:unc]])
    ϕ_ini0 = ζ = vcat(ϕcg_true[:μP] .* 0.0, ϕg0, ϕunc0)
end

@testset "generate_ζ" begin
    ζ, σ = CP.generate_ζ(
        rng, g, f, ϕ_ini, xM[:, 1:n_batch], map(get_concrete, interpreters);
        n_MC = 8)
    @test ζ isa Matrix
    gr = Zygote.gradient(
        ϕ -> sum(CP.generate_ζ(
            rng, g, f, ϕ, xM[:, 1:n_batch], map(get_concrete, interpreters);
            n_MC = 8)[1]),
        CA.getdata(ϕ_ini))
    @test gr[1] isa Vector
end;

# setup g as FluxNN on gpu
using Flux
FluxMLengine = Val(nameof(Flux))
g_flux, ϕg0_flux_cpu = get_hybridcase_MLapplicator(case, FluxMLengine; scenario)

if CUDA.functional()
    @testset "generate_ζ gpu" begin
        ϕ = CuArray(CA.getdata(ϕ_ini))
        xMg_batch = CuArray(xM[:, 1:n_batch])
        ζ, σ = CP.generate_ζ(
            rng, g_flux, f, ϕ, xMg_batch, map(get_concrete, interpreters);
            n_MC = 8)
        @test ζ isa CuMatrix
        @test eltype(ζ) == FT
        gr = Zygote.gradient(
            ϕ -> sum(CP.generate_ζ(
                rng, g_flux, f, ϕ, xMg_batch, map(get_concrete, interpreters);
                n_MC = 8)[1]),
            ϕ)
        @test gr[1] isa CuVector
    end
end

@testset "neg_elbo_transnorm_gf cpu" begin
    cost = neg_elbo_transnorm_gf(rng, g, f, ϕ_ini, y_o[:, 1:n_batch],
        xM[:, 1:n_batch], xP[1:n_batch], transPMs_batch, map(get_concrete, interpreters);
        n_MC = 8, logσ2y)
    @test cost isa Float64
    gr = Zygote.gradient(
        ϕ -> neg_elbo_transnorm_gf(
            rng, g, f, ϕ, y_o[:, 1:n_batch],
            xM[:, 1:n_batch], xP[1:n_batch],
            transPMs_batch, interpreters; n_MC = 8, logσ2y),
        CA.getdata(ϕ_ini))
    @test gr[1] isa Vector
end;

if CUDA.functional()
    @testset "neg_elbo_transnorm_gf gpu" begin
        ϕ = CuArray(CA.getdata(ϕ_ini))
        xMg_batch = CuArray(xM[:, 1:n_batch])
        xP_batch = xP[1:n_batch] # used in f which runs on CPU
        cost = neg_elbo_transnorm_gf(rng, g_flux, f, ϕ, y_o[:, 1:n_batch], 
            xMg_batch, xP_batch,
            transPMs_batch, map(get_concrete, interpreters);
            n_MC = 8, logσ2y)
        @test cost isa Float64
        gr = Zygote.gradient(
            ϕ -> neg_elbo_transnorm_gf(
                rng, g_flux, f, ϕ, y_o[:, 1:n_batch], 
                xMg_batch, xP_batch,
                transPMs_batch, interpreters; n_MC = 8, logσ2y),
            ϕ)
        @test gr[1] isa CuVector
        @test eltype(gr[1]) == FT
    end
end

@testset "predict_gf cpu" begin
    n_sample_pred = n_site = 200
    intm_PMs_gen = get_ca_int_PMs(n_site)
    trans_PMs_gen = get_transPMs(n_site)
    @test length(intm_PMs_gen) == 402
    @test trans_PMs_gen.length_in == 402
    y_pred = predict_gf(rng, g, f, ϕ_ini, xM, xP, map(get_concrete, interpreters);
        get_transPMs, get_ca_int_PMs, n_sample_pred)
    @test y_pred isa Array
    @test size(y_pred) == (size(y_o)..., n_sample_pred)
end

if CUDA.functional()
    @testset "predict_gf gpu" begin
        n_sample_pred = 200
        ϕ = CuArray(CA.getdata(ϕ_ini))
        xMg = CuArray(xM)
        y_pred = predict_gf(rng, g_flux, f, ϕ, xMg, xP, map(get_concrete, interpreters);
            get_transPMs, get_ca_int_PMs, n_sample_pred)
        @test y_pred isa Array
        @test size(y_pred) == (size(y_o)..., n_sample_pred)
    end
end
