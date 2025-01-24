using HybridVariationalInference
using Test
using CUDA, GPUArraysCore
using Lux
using StatsFuns: logistic


@testset "LuxModelApplicator" begin
    n_covar = 5
    n_out = 2
    g_chain = Chain(
        Dense(n_covar => n_covar * 4, tanh),
        Dense(n_covar * 4 => n_covar * 4, tanh),
        Dense(n_covar * 4 => n_out, logistic, use_bias=false),
    );
    g, ϕ = construct_ChainsApplicator(g_chain, Float64; device = cpu_device());
    @test eltype(ϕ) == Float64
    g, ϕ = construct_ChainsApplicator(g_chain, Float32; device = cpu_device());
    @test eltype(ϕ) == Float32
    n_site = 3
    x = rand(Float32, n_covar, n_site)
    #ϕ = randn(Float32, Lux.parameterlength(g_chain))
    y = g(x, ϕ)
    @test size(y) == (n_out, n_site)
    #
    x = rand(Float32, n_covar, n_site) |> gpu_device()
    ϕ_gpu = ϕ |> gpu_device()
    #ϕ = randn(Float32, Lux.parameterlength(g_chain)) |> gpu_device()
    y = g(x, ϕ_gpu)
    #@test ϕ isa GPUArraysCore.AbstractGPUArray
    @test size(y) == (n_out, n_site)
end;
