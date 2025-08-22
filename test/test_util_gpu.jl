using Test
using HybridVariationalInference: HybridVariationalInference as HVI
using ComponentArrays
using CUDA

@testset "ones_similar_x" begin
    A = rand(Float64, 3, 4); 
    B = CUDA.rand(Float32, 5, 2);    # GPU matrix
    @test HVI.ones_similar_x(A, 3) isa Vector
    @test HVI.ones_similar_x(A, size(A,1)) isa Vector
    @test HVI.ones_similar_x(B, size(B,1)) isa CuArray
    @test HVI.ones_similar_x(ComponentVector(b=B), size(B,1)) isa CuArray
    @test HVI.ones_similar_x(B', size(B,1)) isa CuArray
    @test HVI.ones_similar_x(@view(B[:,2]), size(B,1)) isa CuArray
    @test HVI.ones_similar_x(ComponentVector(b=B)[:,1], size(B,1)) isa CuArray
end

