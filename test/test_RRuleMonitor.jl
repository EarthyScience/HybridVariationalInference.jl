using Test
using HybridVariationalInference
using HybridVariationalInference: HybridVariationalInference as HVI
using ComponentArrays: ComponentArrays as CA
# using MLDataDevices
# import CUDA, cuDNN
using DifferentiationInterface: DifferentiationInterface as DI
import Zygote
import ForwardDiff


function ftest(x)
    3 .* abs2.(x)
end

function ftest2(x1,x2)
    x1 .* abs2.(x2)
end


@testset "RRuleMonitor one argument" begin 
    x = collect(1.0:3.0)
    y = ftest(x)
    m = RRuleMonitor("ftest", ftest)
    y2 = m(x)
    @test y2 == y
    gr = Zygote.gradient(x -> sum(ftest(x)), x)[1]
    gr2 = Zygote.gradient(x -> sum(m(x)), x)[1]
    @test gr2 == gr
end

@testset "RRuleMonitor two argument" begin 
    x1 = collect(3.1:0.1:3.3)
    x2 = collect(1.0:3.0)
    y = ftest2(x1, x2)
    m = RRuleMonitor("ftest2", ftest2)
    y2 = m(x1, x2)
    @test y2 == y
    gr = Zygote.gradient((x1,x2) -> sum(ftest2(x1,x2)), x1, x2)
    gr2 = Zygote.gradient((x1,x2) -> sum(m(x1,x2)), x1, x2)
    @test gr2 == gr
    md = RRuleMonitor("ftest2_Forward", ftest2, DI.AutoForwardDiff())
    gr3 = Zygote.gradient((x1,x2) -> sum(md(x1,x2)), x1, x2)
    @test all(gr3 .≈ gr)
end

function ftestsqrt(x)
    is_pos = x .>= zero(x)
    sqrt.(is_pos .* x) .+ .!is_pos .* convert(eltype(x), NaN)
end
@testset "RRuleMonitor non-finite" begin 
    x = collect(2.0:-1.0:-1.0)
    y = ftestsqrt(x)
    m = RRuleMonitor("ftestsqrt", ftestsqrt)
    y2 = m(x)
    @test isequal(y2,y)
    gr = Zygote.gradient(x -> sum(ftestsqrt(x)), x)[1]
    @test_throws ErrorException Zygote.gradient(x -> sum(m(x)), x)
    #@test isequal(gr2,gr)
    #
    md = RRuleMonitor("ftestsqrt", ftestsqrt, DI.AutoForwardDiff())
    @test_throws ErrorException Zygote.gradient(x -> sum(md(x)), x)
end



