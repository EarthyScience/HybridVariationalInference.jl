#using LinearAlgebra, BlockDiagonals
using LinearAlgebra

using Test
using HybridVariationalInference
using HybridVariationalInference: HybridVariationalInference as CP
using HybridVariationalInference: HybridVariationalInference as HVI
using StableRNGs
using Random
using ComponentArrays: ComponentArrays as CA
using Bijectors

# using Zygote
# import GPUArraysCore: GPUArraysCore
# #import CUDA, cuDNN
# using MLDataDevices

@testset "compute_σ_unconstrained" begin
    FT = Float32
    ζM = rand(FT, 5)
    rel_err = 0.1
    transM = Stacked((HVI.Exp(),identity),(1:2,3:5))
    θM = transM(ζM)
    σ = @inferred CP.compute_σ_unconstrained(transM, θM, rel_err)
    @test σ[1] == σ[2] ≈ FT(sqrt(log(rel_err^2 + 1.0))) # exp only depends on rel_err
    @test all(σ[3:5] .≈ FT(rel_err) .* θM[3:5])
    @test eltype(σ) == eltype(θM)
end

