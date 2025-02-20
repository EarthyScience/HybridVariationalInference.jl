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

const prob = DoubleMM.DoubleMMCase()
scenario = (:default,)
FT = get_hybridproblem_float_type(prob; scenario)

#θsite_true = get_hybridproblem_par_templates(prob; scenario)
g, ϕg0 = get_hybridproblem_MLapplicator(prob; scenario);
f = get_hybridproblem_PBmodel(prob; scenario)

n_covar = 5 
n_batch = 10
n_θM, n_θP = values(map(length, get_hybridproblem_par_templates(prob; scenario)))

(; xM, n_site, θP_true, θMs_true, xP, y_global_true, y_true, y_global_o, y_o, y_unc
) = gen_hybridcase_synthetic(rng, prob; scenario);

py = neg_logden_indep_normal

n_MC = 3
(; transP, transM) = get_hybridproblem_transforms(prob; scenario)
cor_ends = get_hybridproblem_cor_ends(prob; scenario)
# transP = elementwise(exp)
# transM = Stacked(elementwise(identity), elementwise(exp))
#transM = Stacked(elementwise(identity), elementwise(exp), elementwise(exp)) # test mismatch
(; ϕ, transPMs_batch, interpreters, get_transPMs, get_ca_int_PMs) = init_hybrid_params(
    θP_true, θMs_true[:, 1], cor_ends, ϕg0, n_batch; transP, transM);
ϕ_ini = ϕ

@testset "generate_ζ" begin
    ζ, σ = CP.generate_ζ(
        rng, g, ϕ_ini, xM[:, 1:n_batch], map(get_concrete, interpreters);
        n_MC = 8, cor_ends)
    @test ζ isa Matrix
    gr = Zygote.gradient(
        ϕ -> sum(CP.generate_ζ(
            rng, g, ϕ, xM[:, 1:n_batch], map(get_concrete, interpreters);
            n_MC = 8, cor_ends)[1]),
        CA.getdata(ϕ_ini))
    @test gr[1] isa Vector
end;

# setup g as FluxNN on gpu
using Flux

if CUDA.functional()
    scenario_flux = (scenario..., :use_Flux)
    g_flux, ϕg0_flux_cpu = get_hybridproblem_MLapplicator(prob; scenario = scenario_flux)
end

if CUDA.functional()
    @testset "generate_ζ gpu" begin
        ϕ = CuArray(CA.getdata(ϕ_ini))
        xMg_batch = CuArray(xM[:, 1:n_batch])
        ζ, σ = CP.generate_ζ(
            rng, g_flux, ϕ, xMg_batch, map(get_concrete, interpreters);
            n_MC = 8, cor_ends)
        @test ζ isa CuMatrix
        @test eltype(ζ) == FT
        gr = Zygote.gradient(
            ϕ -> sum(CP.generate_ζ(
                rng, g_flux, ϕ, xMg_batch, map(get_concrete, interpreters);
                n_MC = 8, cor_ends)[1]),
            ϕ)
        @test gr[1] isa CuVector
    end
end

@testset "neg_elbo_transnorm_gf cpu" begin
    cost = neg_elbo_transnorm_gf(rng, ϕ_ini, g, transPMs_batch, f, py,
        xM[:, 1:n_batch], xP[1:n_batch], y_o[:, 1:n_batch], y_unc[:, 1:n_batch],
        map(get_concrete, interpreters);
        n_MC = 8, cor_ends)
    @test cost isa Float64
    gr = Zygote.gradient(
        ϕ -> neg_elbo_transnorm_gf(rng, ϕ, g, transPMs_batch, f, py,
        xM[:, 1:n_batch], xP[1:n_batch], y_o[:, 1:n_batch], y_unc[:, 1:n_batch],
        map(get_concrete, interpreters);
        n_MC = 8, cor_ends),
        CA.getdata(ϕ_ini))
    @test gr[1] isa Vector
end;

if CUDA.functional()
    @testset "neg_elbo_transnorm_gf gpu" begin
        ϕ = CuArray(CA.getdata(ϕ_ini))
        xMg_batch = CuArray(xM[:, 1:n_batch])
        xP_batch = xP[1:n_batch] # used in f which runs on CPU
        cost = neg_elbo_transnorm_gf(rng, ϕ, g_flux, transPMs_batch, f, py, 
            xMg_batch, xP_batch, y_o[:, 1:n_batch], y_unc[:, 1:n_batch],
             map(get_concrete, interpreters);
            n_MC = 8, cor_ends)
        @test cost isa Float64
        gr = Zygote.gradient(
            ϕ -> neg_elbo_transnorm_gf(rng, ϕ, g_flux, transPMs_batch, f, py, 
            xMg_batch, xP_batch, y_o[:, 1:n_batch], y_unc[:, 1:n_batch],
             map(get_concrete, interpreters);
            n_MC = 8, cor_ends),
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
        get_transPMs, get_ca_int_PMs, n_sample_pred, cor_ends)
    @test y_pred isa Array
    @test size(y_pred) == (size(y_o)..., n_sample_pred)
end

if CUDA.functional()
    @testset "predict_gf gpu" begin
        n_sample_pred = 200
        ϕ = CuArray(CA.getdata(ϕ_ini))
        xMg = CuArray(xM)
        y_pred = predict_gf(rng, g_flux, f, ϕ, xMg, xP, map(get_concrete, interpreters);
            get_transPMs, get_ca_int_PMs, n_sample_pred, cor_ends)
        @test y_pred isa Array
        @test size(y_pred) == (size(y_o)..., n_sample_pred)
    end
end
