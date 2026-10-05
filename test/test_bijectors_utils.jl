using Test
using HybridVariationalInference
using HybridVariationalInference: HybridVariationalInference as CP
using StatsFuns

using Bijectors

using MLDataDevices
import CUDA, cuDNN
using Zygote

gdev = gpu_device()
cdev = cpu_device()



x = [0.1, 0.2, 0.3, 0.4]

function trans(x, b) 
       y, logjac = Bijectors.with_logabsdet_jacobian(b, x)
       sum(y .+ logjac)
end

b_elexp = elementwise(exp)
bs_elexp = Stacked((b_elexp,b_elexp),(1:3,4:4))
b_Exp = HybridVariationalInference.Exp()
bs_Exp = Stacked((b_Exp,b_Exp), (1:3,4:4))
#b3s = Stacked((b3,),(1:4,))

with_logabsdet_jacobian(bs_Exp, x )

@testset "with_logabsdet_jacobian_stacked!" begin
    bs = bs_elexp # is allocating ? 
    bs = bs_Exp 
    bs = Stacked((b_Exp,identity), (1:3,4:4))
    y = similar(x)
    lad = zero(eltype(x))
    y_true, logjac_true = with_logabsdet_jacobian(bs, x)

    y, logjac = CP.with_logabsdet_jacobian_stacked!(bs, x, y)
    @test y == y_true
    @test logjac == logjac_true
    function alloc_bs(bs, x, y) 
        @allocated CP.with_logabsdet_jacobian_stacked!(bs, x, y)
    end
    @test alloc_bs(bs, x, y) == 0
    function loop_bsExp(n, bs, x, y) 
        for i in 1:n
            CP.with_logabsdet_jacobian_stacked!(bs, x, y)
        end
    end
    #@profview_allocs loop_bsExp(10_000, bs, x, y)

    xs = repeat(x, 1, 5)
    ys = similar(xs)
    (ys, logjac) = CP.with_logabsdet_jacobian_stacked!(bs, ys, xs)
    @test ys[:,end] == y_true
    @test logjac == size(xs,2) * logjac_true
    @test ((bs, ys, xs) -> @allocated with_logabsdet_jacobian_stacked!(bs, ys, xs))(bs, ys, xs) == 0

    bsn = @inferred extend_stacked_nrow(bs_Exp, size(xs,2))
    (ys_, logjac) = CP.with_logabsdet_jacobian_stacked!(bsn, vec(xs'), vec(ys'))
    @test ys[:,end] == y_true
    @test logjac ≈ size(xs,2) * logjac_true
    @test alloc_bs(bsn, vec(xs'), vec(ys')) == 0

    # the loop variant is slightly faster
    #@usingany BenchmarkTools
    # @benchmark with_logabsdet_jacobian_stacked!($bs, $ys, $xs)
    # @benchmark CP.with_logabsdet_jacobian_stacked!($bsn, $(vec(xs')), $(vec(ys')))
end



y = trans(x, b_elexp)
dy = Zygote.gradient(x -> trans(x,b_elexp), x)


@testset "elementwise exp" begin
    ys = @inferred trans(x,bs_elexp)
    @test ys == y
    Zygote.gradient(x -> trans(x,bs_elexp), x)
end;

@testset "Exp" begin
    y1 = @inferred b_Exp(x)
    y2 = @inferred bs_Exp(x)
    @test all(inverse(b_Exp)(y2) .≈ x)
    @test all(inverse(bs_Exp)(y2) .≈ x)
    ye = @inferred trans(x, b_Exp)
    dye = Zygote.gradient(x -> trans(x,b_Exp), x)
    @test ye == y
    @test dye == dy
    ys = @inferred trans(x,bs_Exp)
    dys = Zygote.gradient(x -> trans(x,bs_elexp), x)
    @test dys == dy
end;

@testset "Logistic" begin
    c3 = HybridVariationalInference.Logistic()
    c3s = Stacked((c3,c3), (1:3,4:4))
    y1 = @inferred c3(x)
    y2 = @inferred c3s(x)
    @test all(inverse(c3)(y2) .≈ x)
    @test all(inverse(c3s)(y2) .≈ x)
    # test logabsdetjac
    gr = Zygote.gradient(x -> sum(logistic.(x)), x)[1]
    logjac = Bijectors.logabsdetjac(c3s, x) 
    @test logjac ≈ sum(log.(gr))
    y2b, logjac2= Bijectors.with_logabsdet_jacobian(c3s, x)
    @test y2b == y2
    @test logjac2== logjac
end;

if gdev isa MLDataDevices.AbstractGPUDevice
    xd = gdev(x)
    @testset "elementwise exp gpu" begin
        ys = @inferred trans(xd,b_elexp)
        @test ys ≈ y
        @test_broken Zygote.gradient(x -> trans(x,b_elexp), xd)
        @test_broken Zygote.gradient(x -> trans(x,bs_elexp), xd)
    end;
    
    @testset "Exp" begin
        ye = @inferred trans(xd, b_Exp)
        dye = Zygote.gradient(x -> trans(x,b_Exp), xd)
        @test ye ≈ y
        @test all(cdev(dye) .≈ dy)
        ys = @inferred trans(xd,bs_Exp)
        dys = Zygote.gradient(x -> trans(x,bs_Exp), xd)
        @test ys ≈ y
        @test all(cdev(dys) .≈ dy)
    end;
end

@testset "extend_stacked_nrow" begin
    nrow = 50    # faster on CPU by factor of 20
    #nrow = 20000 # faster on GPU
    X = reduce(hcat, ([x + y for x in 0:nrow] for y in 0:10:30))
    b1 = @inferred CP.Exp()
    b2 = identity
    b = @inferred Stacked((b1,b2), (1:1,2:size(X,2)))
    bs = @inferred extend_stacked_nrow(b, size(X,1))
    Xt = @inferred reshape(bs(vec(X)), size(X))
    @test Xt[:,1] == b1(X[:,1])
    @test Xt[:,2] == b2(X[:,2])
    if gdev isa MLDataDevices.AbstractGPUDevice
        Xd = gdev(X)
        Xtd = @inferred reshape(bs(vec(Xd)), size(Xd))
        #Xtd2, logjac = with_logabsdet_jacobian(bs, Xd)
        #@test Xtd2 == Xtd
        # test transpose in gradient function
        dys = Zygote.gradient(x -> sum(bs(vec(x'))), Xd')[1]
    #     () -> begin
    #         #@usingany BenchmarkTools
    #         @benchmark reshape(bs(vec(Xd)), size(Xd)) # macro not definedmetho
    #         vecXd = vec(Xd)
    #         @benchmark bs(vecXd)
    #         vecX = vec(X)
    #         @benchmark bs(vecX)
    #         Xdtrans = Xd'
    #         Xtrans = X'
    #         @benchmark Zygote.gradient(x -> sum(bs(vec(x'))), Xdtrans)[1]
    #         @benchmark Zygote.gradient(x -> sum(bs(vec(x'))), Xtrans)[1]
    #    end
    end
end

@testset "StackedArray" begin
    nrow = 5    # faster on CPU by factor of 20
    #nrow = 20000 # faster on GPU
    X = reduce(hcat, ([x + y for x in 0:nrow] for y in 0.0:10:30))
    b1 = @inferred CP.Exp()
    b2 = identity
    b = @inferred Stacked((b1,b2), (1:1,2:size(X,2)))
    bs = @inferred StackedArray(b, size(X,1))
    Xt = @inferred bs(X)
    @test Xt[:,1] == b1(X[:,1])
    @test Xt[:,2] == b2(X[:,2])
    X2 = @inferred inverse(bs)(Xt)
    @test X2 == X
    # test with Exp only
    be1 = Stacked((CP.Exp(),),(1:size(X,2),))
    bse = StackedArray(be1, size(X,1))
    Xt = @inferred bse(X) # works also for adjoint
    Xt2 = @inferred bse(copy(X')') # works also for adjoint
    @test Xt2 == Xt
    @inferred bse(X)
    #
    # with_logabsdet_jacobians
    Xt3, logjac = with_logabsdet_jacobian(bse, X) # single value
    Xt3, logjacs = CP.with_logabsdet_jacobians(bse, X)
    @test Xt3 == Xt
    @test sum(logjacs) == logjac
    @test size(logjacs) == size(Xt3) # logjac for all components
    #
    if gdev isa MLDataDevices.AbstractGPUDevice
        Xd = gdev(X)
        bse(Xd)
        Xtd = @inferred bs(Xd)
        Xtd2 = @inferred bs(copy(Xd')') # works also for adjoint
        Xtd2 = @inferred bse(copy(Xd')') # needs copy workaround 
        #bse.stacked(vec(Xd'))  # TODO write issue
        tmpf = (X, bs) -> begin
            Xt, logjac = with_logabsdet_jacobian(bs, X)
            sum(Xt) .+ logjac
        end
        tmpf(Xd, bs)
        # test transpose in gradient function
        dys = Zygote.gradient(X -> tmpf(X', bs), Xd')[1]
        @test all(dys[2:end,:] .== 1.0)
    end
end

@testset "with_logabsdet_jacobians" begin

end


    
