using Test
using HybridVariationalInference
using ComponentArrays: ComponentArrays as CA
using StatsFuns
using Distributions
using MLDataDevices, CUDA, cuDNN, GPUArraysCore

@testset "NullModelApplicator" begin
    g = NullModelApplicator()
    c1 = CA.ComponentVector(a = (a1 = 1, a2 = 2:3), b = 3:4)
    y = g(c1, nothing)
    @test y == c1
end;

@testset "MagnitudeModelApplicator" begin
    app = NullModelApplicator()
    c1 = CA.ComponentVector(a = (a1 = 1, a2 = 2:3), b = 3:4)
    m = c1 * 2
    g = MagnitudeModelApplicator(app, m)
    y = g(c1, eltype(m)[])
    @test y == c1 .* m
end;

@testset "NormalScalingModelApplicator" begin
    app = NullModelApplicator()
    r = logistic.(randn(5)) # 0..1
    σ = fill(2.0, 5)
    μ = collect(exp.(1.0:5.0)) # different magnitudes
    g = NormalScalingModelApplicator(app, μ, σ)
    y = g(r, eltype(μ)[])
    p = normcdf.(μ, σ, y)
    #hcat(r, p)
    @test p ≈ r
    gdev = gpu_device()
    #cdev = cpu_device()
    if MLDataDevices.functional(gdev)
        g_gpu = g |> gdev
        @test g_gpu.μ isa GPUArraysCore.AbstractGPUArray
        r_gpu = r |> gdev
        y = g_gpu(r_gpu, eltype(g_gpu.μ)[])
        @test y isa GPUArraysCore.AbstractGPUArray
    end
end;
