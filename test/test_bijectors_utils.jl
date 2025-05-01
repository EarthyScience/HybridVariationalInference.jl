using Test
using HybridVariationalInference
using HybridVariationalInference: HybridVariationalInference as CP

using Bijectors

using MLDataDevices
import CUDA, cuDNN
using Zygote




x = [0.1, 0.2, 0.3, 0.4]
gdev = gpu_device()
cdev = cpu_device()

function trans(x, b) 
       y, logjac = Bijectors.with_logabsdet_jacobian(b, x)
       sum(y .+ logjac)
end

b2 = elementwise(exp)
b2s = Stacked((b2,b2),(1:3,4:4))
b3 = HybridVariationalInference.Exp()
b3s = Stacked((b3,b3), (1:3,4:4))
#b3s = Stacked((b3,),(1:4,))


y = trans(x, b2)
dy = Zygote.gradient(x -> trans(x,b2), x)


@testset "elementwise exp" begin
    ys = trans(x,b2s)
    @test ys == y
    Zygote.gradient(x -> trans(x,b2s), x)
end;

@testset "Exp" begin
    y1 = b3(x)
    y2 = b3s(x)
    @test all(inverse(b3)(y2) .≈ x)
    @test all(inverse(b3s)(y2) .≈ x)
    ye = trans(x, b3)
    dye = Zygote.gradient(x -> trans(x,b3), x)
    @test ye == y
    @test dye == dy
    ys = trans(x,b3s)
    dys = Zygote.gradient(x -> trans(x,b2s), x)
    @test dys == dy
end;


if gdev isa MLDataDevices.AbstractGPUDevice
    xd = gdev(x)
    @testset "elementwise exp gpu" begin
        ys = trans(xd,b2)
        @test ys ≈ y
        @test_broken Zygote.gradient(x -> trans(x,b2), xd)
        @test_broken Zygote.gradient(x -> trans(x,b2s), xd)
    end;
    
    @testset "Exp" begin
        ye = trans(xd, b3)
        dye = Zygote.gradient(x -> trans(x,b3), xd)
        @test ye ≈ y
        @test all(cdev(dye) .≈ dy)
        ys = trans(xd,b3s)
        dys = Zygote.gradient(x -> trans(x,b3s), xd)
        @test ys ≈ y
        @test all(cdev(dys) .≈ dy)
    end;
end

@testset "extend_stacked_nrow" begin
    nrow = 50    # faster on CPU by factor of 20
    #nrow = 20000 # faster on GPU
    X = reduce(hcat, ([x + y for x in 0:nrow] for y in 0:10:30))
    b1 = CP.Exp()
    b2 = identity
    b = Stacked((b1,b2), (1:1,2:size(X,2)))
    bs = extend_stacked_nrow(b, size(X,1))
    Xt = reshape(bs(vec(X)), size(X))
    @test Xt[:,1] == b1(X[:,1])
    @test Xt[:,2] == b2(X[:,2])
    if gdev isa MLDataDevices.AbstractGPUDevice
        Xd = gdev(X)
        Xtd = reshape(bs(vec(Xd)), size(Xd))
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

    
