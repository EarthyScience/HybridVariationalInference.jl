using Test
using StatsFuns: logistic
using CUDA, GPUArraysCore

using HybridVariationalInference
# @testset "get_default_GPUHandler before loading Flux" begin
#     # during overall package testing Flux could be loaded beforehand
#     h = get_default_GPUHandler()
#     @test h isa NullGPUDataHandler
#     x = CuArray(1:5)
#     xh = h(x)
#     @test xh === x
# end;

using Flux
@testset "get_default_GPUHandler after loading Flux" begin
    # difficult to  access type in ext
    # HybridVariationalInferenceFluxExt.FluxGPUDataHandler
    #typeof(HybridVariationalInference.default_GPU_DataHandler)
    h = get_default_GPUHandler()
    @test !(h isa NullGPUDataHandler)
    if CUDA.functional()
        x = CuArray(1:5)
        xh = h(x)
        @test xh isa Vector
    end
end;


@testset "FluxModelApplicator" begin
    n_covar = 5
    n_out = 2
    g_chain = Chain(
        Dense(n_covar => n_covar * 4, tanh),
        Dense(n_covar * 4 => n_covar * 4, tanh),
        Dense(n_covar * 4 => n_out, identity, bias=false),
    )
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

