using HybridVariationalInference
using Test
using Aqua

@testset "HybridVariationalInference.jl" begin
    @testset "Code quality (Aqua.jl)" begin
        Aqua.test_all(HybridVariationalInference)
    end
    # Write your tests here.
end
