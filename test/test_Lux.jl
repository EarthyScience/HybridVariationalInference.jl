using HybridVariationalInference
using Test
using Lux
using StatsFuns: logistic
using CUDA, GPUArraysCore


@testset "LuxModelApplicator" begin
    n_covar = 5
    n_out = 2
    g_chain = Chain(
        Dense(n_covar => n_covar * 4, tanh),
        Dense(n_covar * 4 => n_covar * 4, tanh),
        Dense(n_covar * 4 => n_out, logistic, use_bias=false),
    );
    g = construct_LuxApplicator(g_chain; device = cpu_device());
    n_site = 3
    x = rand(Float32, n_covar, n_site)
    ϕ = randn(Float32, Lux.parameterlength(g_chain))
    y = g(x, ϕ)
    @test size(y) == (n_out, n_site)
    #
    g = construct_LuxApplicator(g_chain; device = gpu_device());
    n_site = 3
    x = rand(Float32, n_covar, n_site) |> gpu_device()
    ϕ = randn(Float32, Lux.parameterlength(g_chain)) |> gpu_device()
    y = g(x, ϕ)
    #@test ϕ isa GPUArraysCore.AbstractGPUArray
    @test size(y) == (n_out, n_site)
end;
