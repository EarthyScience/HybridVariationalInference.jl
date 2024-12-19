using HybridVariationalInference
using Test
using Flux
using StatsFuns: logistic
using CUDA, GPUArraysCore

@testset "FluxModelApplicator" begin
    n_covar = 5
    n_out = 2
    g_chain = Chain(
        Dense(n_covar => n_covar * 4, tanh),
        Dense(n_covar * 4 => n_covar * 4, tanh),
        Dense(n_covar * 4 => n_out, logistic, bias=false),
    );
    g = construct_FluxApplicator(g_chain)
    n_site = 3
    x = rand(Float32, n_covar, n_site)
    ϕ, _rebuild = destructure(g_chain)
    y = g(x, ϕ)
    @test size(y) == (n_out, n_site)
    #
    n_site = 3
    x = rand(Float32, n_covar, n_site) |> gpu
    ϕ = ϕ |> gpu
    y = g(x, ϕ)
    #@test ϕ isa GPUArraysCore.AbstractGPUArray
    @test size(y) == (n_out, n_site)
end;
