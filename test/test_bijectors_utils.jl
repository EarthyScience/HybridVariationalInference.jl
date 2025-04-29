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


y = trans(x, b2)
dy = Zygote.gradient(x -> trans(x,b2), x)


@testset "elementwise exp" begin
    ys = trans(x,b2s)
    @test ys == y
    Zygote.gradient(x -> trans(x,b2s), x)
end;

@testset "Exp" begin
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
    
