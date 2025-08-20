using Test
using HybridVariationalInference
using HybridVariationalInference: HybridVariationalInference as CP
using ComponentArrays: ComponentArrays as CA

using MLDataDevices, GPUArraysCore
import Zygote

# import CUDA, cuDNN
using Suppressor

gdev = Suppressor.@suppress gpu_device() # not loaded CUDA
cdev = cpu_device()

@testset "construct StaticComponentArrayInterepreter" begin
    intv = @inferred CP.StaticComponentArrayInterpreter(CA.ComponentVector(a=1:3, b=reshape(4:9,3,2)))
    ints = @inferred CP.StaticComponentArrayInterpreter((;a=Val(3), b = Val((3,2))))
    # @descend_code_warntype CP.StaticComponentArrayInterpreter((;a=Val(3), b = Val((3,2))))
    @test ints == intv
end

@testset "ComponentArrayInterpreter cv-vector" begin
    component_counts = comp_cnts = (; P=2, M=3, Unc=5)  
    comp_cnts_val = (; P=Val(2), M=Val(3), Unc=Val(5))
    #component_counts = comp_cnts = CA.ComponentVector(P=1:2, M=1:3, Unc=1:5)
    
    m = @inferred ComponentArrayInterpreter(comp_cnts)
    m2 = @inferred CP.StaticComponentArrayInterpreter(comp_cnts_val)
    get_positions(m)
    testm = (m) -> begin
        #type of axes may differ
        #@test CP._get_ComponentArrayInterpreter_axes(m) == (CA.Axis(P=1:2, M=3:5, Unc=6:10),)
        @test length(m) == 10
        v = 1:length(m)
        cv = m(v)
        @test cv.Unc == 6:10
    end
    testm(m)
    #m = @inferred get_concrete(m)
    m = get_concrete(m)
    testm(get_concrete(m))
    Base.isconcretetype(typeof(m))

    cc0 = CA.ComponentVector(comp_cnts)
    sum(get_positions(ComponentArrayInterpreter(cc0)))
    Zygote.gradient(cc -> sum(cc), cc0)
    Zygote.gradient(cc -> sum(get_positions(ComponentArrayInterpreter(cc))), cc0)
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
    #component_shapes = CA.ComponentVector(P=1:2, M=reshape(1:6,2, 3), Unc=1:5)
    m = @inferred ComponentArrayInterpreter(component_shapes)
    testm = (m) -> begin
        @test length(m) == 13
        a = 1:length(m)
        cv = m isa CP.StaticComponentArrayInterpreter ? @inferred(m(a)) : m(a)
        @test cv.M == 2 .+ [1 3 5; 2 4 6]
        cv
    end
    testm(m)
    @inferred testm(get_concrete(m))
    # test creating ComponentArrayInterpreter insite differentiated function
    tmpf = (a) -> begin
        m = ComponentArrayInterpreter(component_shapes)
        cv = m(a)
        sum(cv.M)
    end
    Zygote.gradient(tmpf, 1:length(m))
end;

@testset "ComponentArrayInterpreter matrix and array" begin
    mvi = ComponentArrayInterpreter(; c1=2, c2=3)
    #mvi = ComponentArrayInterpreter(CA.ComponentVector(c1=1:2, c2=1:3))
    cv = mvi(1:length(mvi))
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
    mmi = ComponentArrayInterpreter(mvi, (n_col,)) # construct on interpreter itself
    testm(mmi)
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
    mmi = ComponentArrayInterpreter(mvi, (n_col, n_z)) # construct on interpreter itself
    testm(mmi)
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
    mm = ComponentArrayInterpreter((n_row,), mvi) # construct on interpreter itself
    testm(mmi)
end;

@testset "stack_ca_int" begin
    mvi = get_concrete(ComponentArrayInterpreter(CA.ComponentVector(c1=1:2, c2=1:3)))
    #mvi = ComponentArrayInterpreter(CA.ComponentVector(c1=1:2, c2=1:3))
    cv = mvi(1:length(mvi))
    n_col = 4
    n_dims = (n_col,)
    mm = @inferred CP.stack_ca_int(mvi, Val((n_col,))) # 1-tuple
    @inferred get_positions(mm) # sizes are inferred here
    testm = (m) -> begin
        @test length(mm) == length(cv) * n_col
        cm = mm(1:length(mm))
        #cm[:c1,:]
        @test cm[:c1, 2] == 6:7
    end
    testm(mm)
    #
    n_z = 3
    mm = @inferred stack_ca_int(mvi, Val((n_col, n_z)))
    testm = (m) -> begin
        @test mm isa AbstractComponentArrayInterpreter
        @test length(mm) == length(cv) * n_col * n_z
        cm = mm(1:length(mm))
        @test cm[:c1, 2, 2] == 26:27
    end
    testm(mm)
    #
    n_row = 3
    mm = @inferred stack_ca_int(Val((n_row,)), mvi)
    testm = (m) -> begin
        @test mm isa AbstractComponentArrayInterpreter
        @test length(mm) == n_row * length(mvi)
        cm = mm(1:length(mm))
        @test cm[2, :c1] == [2, 5]
    end
    testm(mm)
    #
    f_n_within = (n) -> begin
        mm = @inferred stack_ca_int(Val((n,)), mvi)
    end
    @test_broken @inferred f_n_within(3) # inferred is only 
    f_outer = () -> begin
        f_n_within_cols = (n) -> begin
            mm = @inferred stack_ca_int(mvi, Val((n,)))
            mm = get_concrete(ComponentArrayInterpreter(mvi, (3,))) # same effects
        end
        # @inferred f_n_within_cols(3) # inferred is only Any
        res = f_n_within_cols(3) # inferred is only 
        pos = @inferred get_positions(res) # but within this context size is known
        @inferred res(pos)
    end
    #pos_outer = @inferred f_outer() # but inferred return type is Any
    pos_outer = f_outer()
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

@testset "compose_interpreters" begin
    int1 = get_concrete(ComponentArrayInterpreter(CA.ComponentVector(x1=1:3, x2=4:5)))
    int2 = get_concrete(ComponentArrayInterpreter(CA.ComponentVector(y1=1:2, y2=3:5)))
    intm2 = get_concrete(ComponentArrayInterpreter(int2, (3,)))
    #intc = ComponentArrayInterpreter((a=int1, b=int2))
    ints = (a=int1, b=intm2)
    intc = @inferred compose_interpreters(;ints...)
    # @usingany Cthulhu
    # @descend_code_warntype CP.StaticComponentArrayInterpreter(a=int1, b=int2)
    # @descend_code_warntype CP.compose_axes(map(x -> CA.getaxes(x), ints))
    () -> begin
        nt = (a=int1, b=int2)
        nt isa NamedTuple{keys, <:NTuple{N, <:AbstractComponentArrayInterpreter}} where {keys, N}
        nt isa NamedTuple{keys, <:NTuple{N}} where {keys, N}
        nt isa NamedTuple{keys} where {keys}
    end
    #
    v3 = CA.ComponentVector(a = get_positions(int1), b = get_positions(intm2))
    intc2 = ComponentArrayInterpreter(v3)
    @test intc == intc2
    v3r = @inferred get_concrete(intc)(CA.getdata(v3))
    @test v3r == v3
    #@usingany BenchmarkTools
    #@benchmark ComponentArrayInterpreter(a=int1, b=int2) # 6 allocations?
    #@benchmark CP.StaticComponentArrayInterpreter(a=int1, b=int2) # still 5 allocations?
    #@benchmark CP.compose_axes((a=int1, b=int2)) # still 5 allocations?
    #@usingany Cthulhu
    # Cthulhu.@descend_code_typed ComponentArrayInterpreter(a=int1, b=int2)
    # @code_typed get_concrete(ComponentArrayInterpreter(a=int1, b=int2))
    if gdev isa MLDataDevices.AbstractGPUDevice 
        vd = gdev(CA.getdata(v3))
        f1 = (v) -> begin
            #intc = @inferred compose_interpreters(a=int1, b=intm2) # fails on Zygote
            intc = compose_interpreters(a=int1, b=intm2) 
            vc = intc(v)
            sum(vc.a.x1)::eltype(vc) # eltype necessary
            #sum(vc.a.x1)
        end
        @test @inferred f1(vd) == sum(v3.a.x1)
        df1 = Zygote.gradient(v -> f1(v), vd)[1];
        @test df1 isa AbstractGPUArray
    end

end;

@testset "type inference concrete Array interpreter" begin
    cai0 = ComponentArrayInterpreter(x=(3,2))
    cai = get_concrete(cai0)
    v = collect(1:length(cai))
    cv = cai(v)

    cv2 = @inferred CP.tmpf(v; cv) # cai by keyword argument
    #cv2 = @inferred CP.tmpf(v; cv=nothing, cai = cai0) # not inferred
    cv2 = CP.tmpf(v; cv=nothing, cai = cai0) # not inferred
    cv2 = @inferred CP.tmpf1(v; cai = get_concrete(cai0)) # cai by keyword argument
    #cv2 = @inferred CP.tmpf1(v; cai = cai0) # inside function does not infer
    cv2 = CP.tmpf1(v; cai = cai0) # get_concrete inside function does not infer outside
    cv2 = @inferred CP.tmpf2(v; cai=cai0) # only when specifying return type
    # () -> begin
    #     #cv2 = @code_warntype CP.tmpf(cai0) # Any
    #     #cv2 = @code_warntype CP.tmpf(cai)  # ok
    #     cv2 = @code_warntype CP.tmpf(v;cv, cai) # ok, keywords work
    #     cv2 = @code_warntype CP.tmpf(v;cv, cai=cai0) # Any
    #     cv2 = @code_warntype CP.tmpf(v; cv) # ok !!
    #     cv2 = CP.tmpf(v; cv)
    #     typeof(cv2)

    #     cv2 = CP.tmpf2(v; cai=cai)
    #     cv2 = @code_warntype CP.tmpf2(v; cai=cai) #ok
    #     cv2 = @code_warntype CP.tmpf2(v; cai=cai0) #
    #     cv2 = @code_warntype sum(CP.tmpf2(v; cai=cai0)) #
    #     cv2 = @code_warntype sum(CP.tmpf2(v; cai=cai0).x) #
    #     # @usingany Cthulhu
    #     # @descend_code_warntype CP.tmpf2(v; cai=cai0) 
    #     # @code_warntype CP.tmpf2(v; cai=cai0) 
    #     cv2 = CP.tmpf2(v; cai=cai0) #
    # end
end


