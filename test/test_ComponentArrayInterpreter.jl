using Test
using HybridVariationalInference
using HybridVariationalInference: HybridVariationalInference as CM
using ComponentArrays: ComponentArrays as CA

@testset "ComponentArrayInterpreter vector" begin
    component_counts = comp_cnts = (; P=2, M=3, Unc=5)
    m = ComponentArrayInterpreter(; comp_cnts...)
    testm = (m) -> begin
        #type of axes may differ
        #@test CM._get_ComponentArrayInterpreter_axes(m) == (CA.Axis(P=1:2, M=3:5, Unc=6:10),)
        @test length(m) == 10
        v = 1:length(m)
        cv = m(v)
        @test cv.Unc == 6:10
    end
    testm(m)
    m = get_concrete(m)
    testm(get_concrete(m))
    Base.isconcretetype(typeof(m))
end;

# () -> begin
#     # test generate code for length
#     @code_llvm length(m)
#     mc = get_concrete(m)
#     @code_llvm length(mc)
#     v = 1:length(m)
#     @code_llvm as_ca(v,m)
#     @code_llvm as_ca(v,mc)
# end

@testset "ComponentArrayInterpreter matrix in vector" begin
    component_shapes = (; P=2, M=(2, 3), Unc=5)
    m = ComponentArrayInterpreter(; component_shapes...)
    testm = (m) -> begin
        @test length(m) == 13
        a = 1:length(m)
        cv = m(a)
        @test cv.M == 2 .+ [1 3 5; 2 4 6]
    end
    testm(m)
    testm(get_concrete(m))
end;

@testset "ComponentArrayInterpreter matrix and array" begin
    mv = ComponentArrayInterpreter(; c1=2, c2=3)
    cv = mv(1:length(mv))
    n_col = 4
    mm = ComponentArrayInterpreter(cv, (n_col,)) # 1-tuple
    testm = (m) -> begin
        @test length(mm) == length(cv) * n_col
        cm = mm(1:length(mm))
        #cm[:c1,:]
        @test cm[:c1, 2] == 6:7
    end
    testm(mm)
    mmc = get_concrete(mm)
    testm(mmc)
    #
    n_z = 3
    mm = ComponentArrayInterpreter(cv, (n_col, n_z))
    testm = (m) -> begin
        @test length(mm) == length(cv) * n_col * n_z
        cm = mm(1:length(mm))
        @test cm[:c1, 2, 2] == 26:27
    end
    testm(mm)
    testm(get_concrete(mm))
    #
    n_row = 3
    mm = ComponentArrayInterpreter((n_row,), cv)
    testm = (m) -> begin
        @test length(mm) == n_row * length(cv)
        cm = mm(1:length(mm))
        @test cm[2, :c1] == [2, 5]
    end
    testm(mm)
    testm(get_concrete(mm))
end;

@testset "empty ComponentVector" begin
    x = CA.ComponentVector{Float32}()
    int1 = ComponentArrayInterpreter(x) 
    @test int1(CA.getdata(x)) == x
    int2 = ComponentArrayInterpreter(x, ())
    @test int2 == int1
    int3 = ComponentArrayInterpreter((), x)
    @test int3 == int1
end;



