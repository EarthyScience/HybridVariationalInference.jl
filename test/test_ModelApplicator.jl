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

@testset "MagnitudeModelApplicator subset" begin
    app = NullModelApplicator()
    c1 = CA.ComponentVector(a = (a1 = 1, a2 = 2:3), b = 3:4)
    range_scaled = 2:3
    m = 2
    g = MagnitudeModelApplicator(app, m; range_scaled)
    y = g(c1, eltype(m)[])
    @test y[range_scaled] == c1[range_scaled] .* m
    @test y[1:end .∉ Ref(range_scaled)] == c1[1:end .∉ Ref(range_scaled)] 
    ym = g(hcat(c1,c1 .* 2),eltype(m)[]) # transforming matrix
    @test ym[:,1] == y
    @test ym[:,2] == y .* 2
end;

@testset "NormalScalingModelApplicator" begin
    app = NullModelApplicator()
    r = logistic.(randn(5)) # 0..1
    σ = fill(2.0, 5)
    μ = collect(exp.(1.0:5.0)) # different magnitudes
    g = NormalScalingModelApplicator(app, μ, σ, 1:0)
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

@testset "NormalScalingModelApplicator subset" begin
    app = NullModelApplicator()
    r = logistic.(randn(10)) # 0..1
    r2 = logistic.(randn(10)) # 0..1
    σ = fill(2.0, 5)
    μ = collect(exp.(1.0:5.0)) # different magnitudes
    range_scaled = 2 .+ (1:length(σ))
    g = NormalScalingModelApplicator(app, μ, σ, range_scaled)
    y = g(r, eltype(μ)[])
    p = normcdf.(μ, σ, y[range_scaled])
    #hcat(r, p)
    @test p ≈ r[range_scaled]
    @test y[1:end .∉ Ref(range_scaled)] == r[1:end .∉ Ref(range_scaled)]
    rm = hcat(r, r2)
    ym = g(rm, eltype(μ)[])
    @test ym[:,1] == y
    p2 = normcdf.(μ, σ, ym[range_scaled,2])
    @test p2 ≈ r2[range_scaled]
    @test ym[1:end .∉ Ref(range_scaled),2] == r2[1:end .∉ Ref(range_scaled)]
    #cdev = cpu_device()
    if gdev isa MLDataDevices.AbstractGPUDevice 
        g_gpu = g |> gdev
        @test g_gpu.μ isa GPUArraysCore.AbstractGPUArray
        r_gpu = r |> gdev
        rm_gpu = rm |> gdev
        y = g_gpu(r_gpu, eltype(g_gpu.μ)[])
        @test y isa GPUArraysCore.AbstractGPUArray
        ym = g_gpu(rm_gpu, eltype(g_gpu.μ)[])
        @test ym isa GPUArraysCore.AbstractGPUArray
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
    rm = hcat(r, r .* 2)
    ym = g(rm, [])
    @test ym[:,1] == y
    @test ym[:,2] ≈(rm[:,2] .* width .+ lowers)
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

@testset "RangeScalingModelApplicator subset" begin
    app = NullModelApplicator()
    r = logistic.(randn(Float32, 10)) # 0..1
    lowers = collect(exp.(1.0:5.0)) # different magnitudes
    uppers = lowers .* 2
    range_scaled = 2 .+ (1:length(lowers))
    g = RangeScalingModelApplicator(app, lowers, uppers, eltype(r); range_scaled)
    y = @inferred g(r, [])
    width = uppers .- lowers
    @test y[range_scaled] ≈ (r[range_scaled] .* width .+ lowers)
    @test eltype(y) == eltype(r)
    @test y[1:end .∉ Ref(range_scaled)] == r[1:end .∉ Ref(range_scaled)]
    #cdev = cpu_device()
    rm = hcat(r, r .* 2)
    ym = g(rm, [])
    @test ym[:,1] == y
    @test ym[range_scaled,2] ≈(rm[range_scaled,2] .* width .+ lowers)
    @test ym[1:end .∉ Ref(range_scaled),2] == rm[1:end .∉ Ref(range_scaled),2]
    if gdev isa MLDataDevices.AbstractGPUDevice 
        g_gpu = g |> gdev
        @test g_gpu.offset isa GPUArraysCore.AbstractGPUArray
        @test g_gpu.width isa GPUArraysCore.AbstractGPUArray
        r_gpu = r |> gdev
        y_dev = g_gpu(r_gpu, [])
        @test y_dev isa GPUArraysCore.AbstractGPUArray
        @test cdev(y_dev) ≈ y
        rm_gpu = rm |> gdev
        ym_dev = g_gpu(rm_gpu, [])
        @test y_dev isa GPUArraysCore.AbstractGPUArray
        @test cdev(ym_dev) ≈ ym
    end
end;

