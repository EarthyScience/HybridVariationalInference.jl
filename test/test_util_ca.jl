using Test
using HybridVariationalInference
using HybridVariationalInference: HybridVariationalInference as CP
using ComponentArrays: ComponentArrays as CA

@testset "compose_axes" begin
    @test (@inferred CP._add_interval(;ranges=(Val(1:3),), length = Val(2))) == (Val(1:3), Val(4:5))
    ls = Val.((3,1,2))
    @test (@inferred CP._construct_intervals(;lengths=ls)) == Val.((1:3, 4:4, 5:6))
    v1 = CA.ComponentVector(A=1:3)
    v2 = CA.ComponentVector(B=1:2)
    v3 = CA.ComponentVector(P=(x=1, y=2), Ms=zeros(3,2))
    nt = (;C1=v1, C2=v2, C3=v3)
    vt = CA.ComponentVector(; nt...)
    axs = map(CA.getaxes, nt)
    axc = @inferred CP.compose_axes(axs)
    @test axc == CA.getaxes(vt)[1] 
end

