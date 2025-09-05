using Test
using HybridVariationalInference: HybridVariationalInference as HVI
using ComponentArrays
using MLDataDevices
import CUDA, cuDNN
using FillArrays

@testset "ones_similar_x" begin
    A = rand(Float64, 3, 4); 
    @test @inferred HVI.ones_similar_x(A, 3) isa FillArrays.AbstractFill #Vector
    @test @inferred HVI.ones_similar_x(A, size(A,1)) isa FillArrays.AbstractFill #Vector#Vector
end

gdev = gpu_device()
if gdev isa MLDataDevices.CUDADevice
    @testset "ones_similar_x" begin
        B = CUDA.rand(Float32, 5, 2);    # GPU matrix
        @test @inferred HVI.ones_similar_x(B, size(B,1)) isa CUDA.CuArray
        @test @inferred HVI.ones_similar_x(ComponentVector(b=B), size(B,1)) isa CUDA.CuArray
        @test @inferred HVI.ones_similar_x(B', size(B,1)) isa CUDA.CuArray
        @test @inferred HVI.ones_similar_x(@view(B[:,2]), size(B,1)) isa CUDA.CuArray
        @test @inferred HVI.ones_similar_x(ComponentVector(b=B)[:,1], size(B,1)) isa CUDA.CuArray
    end
end

