using Test
using HybridVariationalInference
using HybridVariationalInference: HybridVariationalInference as HVI
using ComponentArrays: ComponentArrays as CA
using StatsFuns
using Distributions
using MLDataDevices, CUDA, cuDNN, GPUArraysCore

gdev = gpu_device()
cdev = cpu_device()

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
    #cdev = cpu_device()
    if gdev isa MLDataDevices.AbstractGPUDevice 
        g_gpu = g |> gdev
        @test g_gpu.μ isa GPUArraysCore.AbstractGPUArray
        r_gpu = r |> gdev
        y = g_gpu(r_gpu, eltype(g_gpu.μ)[])
        @test y isa GPUArraysCore.AbstractGPUArray
    end
end;

@testset "RangeScalingModelApplicator" begin
    app = NullModelApplicator()
    r = logistic.(randn(Float32, 5)) # 0..1
    lowers = collect(exp.(1.0:5.0)) # different magnitudes
    uppers = lowers .* 2
    g = RangeScalingModelApplicator(app, lowers, uppers, eltype(r))
    y = g(r, [])
    width = uppers .- lowers
    @test y ≈(r .* width .+ lowers)
    @test eltype(y) == eltype(r)
    #cdev = cpu_device()
    if gdev isa MLDataDevices.AbstractGPUDevice 
        g_gpu = g |> gdev
        @test g_gpu.offset isa GPUArraysCore.AbstractGPUArray
        @test g_gpu.width isa GPUArraysCore.AbstractGPUArray
        r_gpu = r |> gdev
        y_dev = g_gpu(r_gpu, [])
        @test y_dev isa GPUArraysCore.AbstractGPUArray
        @test cdev(y_dev) ≈ y
    end
end;
