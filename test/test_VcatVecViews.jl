using Test
using LazyArrays
using HybridVariationalInference
import HybridVariationalInference as HVI
using ForwardDiff
using StableRNGs

@testset "vecview" begin
    rng = StableRNG(1234)
    v = collect(1:4)
    A = rand(1:10, 3,3)
    vecA = HVI.vecview(A)  
    function loop_vecview(n,A)
        for _ in 1:n
            sum(HVI.vecview(A))
        end
    end
    @test (@allocated loop_vecview(10_000,A)) == 0
    #@profview_allocs loop_vecview(10_000,A)
end

@testset "VcatVecViews" begin
    rng = StableRNG(1234)
    v = collect(1:4)
    A = rand(1:10, 3,3)
    Avv = VcatVecViews(A,v)
    @test Avv[1:9] == vec(A)
    @test Avv[10:end] == v

    @test (@inferred subarray(Avv,Val(1))) === A
    @test (@inferred subarray(Avv,Val(2))) === v

    #@code_llvm subarray(Avv,Val(2))

    A[1] = 40
    @test Avv[1] == 40 
    Avv[1] = 4
    @test A[1] == 4

    function loop_sumAvv(n,Avv)
        for _ in 1:n
            sum(Avv)
        end
    end
    @test (@allocated loop_sumAvv(10_000,Avv)) == 0

    function loop_subarray(n,Avv)
        for _ in 1:n
            sum(subarray(Avv,Val(1)))
        end
    end
    @test (@allocated loop_subarray(10_000,Avv)) == 0
    #@profview_allocs loop_subarray(10_000,Avv)

    @test similar(Avv) isa Matrix

    B = similar_views(Avv)
    @test B isa VcatVecViews
    @test subarray(B, Val(1)) isa Matrix

    B32 = similar_views(Avv, Float32)
    @test subarray(B, Val(1)) isa Matrix
    @test eltype(B32) == Float32

    Cv = copy(Avv)
    @test Cv == Avv
    #@test_broken size(subarray(Cv, Val(1))) == size(A)
    @test size(subarray(Cv, Val(1))) == (9,1) # !! reshaped to 1 column matrix

    C = copy_views(Avv)
    @test C == Avv
    @test size(subarray(C, Val(1))) == size(A)    
end

@testset "ForwardDiff" begin
    rng = StableRNG(1234)
    v = collect(1:4)
    A = rand(1:10, 3,3)
    Avv = VcatVecViews(A,v)
    compAv(A,v) = 2*sum(A) + 3*sum(v) # the hereogeneous computation
    compAvv(Avv) = compAv(subarray(Avv,Val(1)), subarray(Avv,Val(2))) # wrapper
    compAvv(Avv)
    cfg = ForwardDiff.GradientConfig(compAvv, Avv, ForwardDiff.Chunk(8))
    # replace the duals-Component of the GradientConfig by a proper VcatVecViews
    cfgv = HVI.update_duals(cfg, Avv)
    # can also supply a VcatVecViews mutated destination for the gradient
    #   note that similar or copy of a VcatVecViews only return a Vcat of Vector
    gr = VcatVecViews(A,v) 
    ForwardDiff.gradient!(gr, compAvv, Avv, cfgv)
    @test subarray(gr,Val(1)) isa Matrix
    @test all(subarray(gr,Val(1)) .== 2)
    @test all(subarray(gr,Val(2)) .== 3)

    function loop_gradient(n,gr,Avv,cfgv)
        for _ in 1:n
            ForwardDiff.gradient!(gr, compAvv, Avv, cfgv)
        end
    end
    @test (@allocated loop_gradient(10_000,gr,Avv,cfgv)) == 0
    #@profview loop_gradient(10_000,gr,Avv,cfgv)
end


