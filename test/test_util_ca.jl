using Test
using HybridVariationalInference
using HybridVariationalInference: HybridVariationalInference as CP
using ComponentArrays: ComponentArrays as CA
using DataFrames

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

@testset "as_data_frame" begin
    v1 = CA.ComponentVector(a=1.1, b=2.1)
    v2 = CA.ComponentVector(a=1.2, b=2.2)
    cm = vcat(v1',v2')
    df = as_data_frame(cm)
    @test names(df) == ["a", "b"]
    @test collect(df[1,:]) == CA.getdata(v1)
    @test collect(df[2,:]) == CA.getdata(v2)
    v2c = copy(v2)
    # copy: 
    v2[1] = 1.3
    @test collect(df[2,:]) == CA.getdata(v2c)
    #
    # names in first dimension
    cmt = hcat(v1,v2)
    df = as_data_frame(cmt)
    @test collect(df[1,:]) == CA.getdata(v1)
    @test collect(df[2,:]) == CA.getdata(v2)
    #
    cm = cmt'
    df = as_data_frame(cm)
    @test collect(df[1,:]) == CA.getdata(v1)
    @test collect(df[2,:]) == CA.getdata(v2)
    #
    cma = stack([cm,cm .* 10])
    df = as_data_frame(cma)
    @test Array(df[1:2,1:2]) == CA.getdata(cm)
    @test Array(df[3:4,1:2]) == CA.getdata(cm .* 10)
    @test all(vec(df[1:2,:dim3]) .== 1)
    @test all(vec(df[3:4,:dim3]) .== 2)
    #
    cma1 = stack([cm',cm' .* 10])
    df1 = as_data_frame(cma1)
    @test df1 == df
    #
    cma4 = stack([cma, cma .* 10])
    df = as_data_frame(cma4)
    #
    cma41 = stack([cma1, cma1 .* 10])
    df2 = as_data_frame(cma41)
    @test df2 == df
end

