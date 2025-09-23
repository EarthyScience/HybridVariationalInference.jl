using Test
using HybridVariationalInference
using HybridVariationalInference: HybridVariationalInference as CP
using SimpleChains
using StableRNGs

import Zygote

import CUDA, cuDNN
using MLDataDevices

gdev = gpu_device()
cdev = cpu_device()

using OptimizationOptimisers
using Lux  # in order to load extension


@testset "repeat_rowvector_dummy" begin
    x = collect(1.0:5.0) 
    is_dummy = [false, false, true, true]
    xm = CP.repeat_rowvector_dummy(x', is_dummy)
    @test xm[1,:] == xm[2,:] == x
    @test all(isnan.(xm[3:4,:]))
    #
    gr = Zygote.gradient(x -> sum(CP.repeat_rowvector_dummy(x', is_dummy)), x)[1]
    @test gr == fill(2.0, size(x))
    gr = Zygote.gradient(x -> sum(CP.repeat_rowvector_dummy(x', is_dummy).^2), x)[1]
    @test gr == 2 .*2 .*x
    gr = Zygote.gradient(x -> sum(CP.repeat_rowvector_dummy(x', is_dummy) .* 2.0), x)[1]
    @test gr == fill(2*2.0,5)
    gr = Zygote.gradient(x -> sum(abs2, CP.repeat_rowvector_dummy(x', is_dummy)), x)[1]
    @test gr == 2 .*2 .*x
    gr = Zygote.gradient(x -> sum(sin, CP.repeat_rowvector_dummy(x', is_dummy)), x)[1]
    @test gr == 2 .* cos.(x)
    gr = (Zygote.gradient(x) do x 
        y = CP.repeat_rowvector_dummy(x', is_dummy) 
        z = sum(abs2, y[1,:]) + 3*sum(y[2,:])
    end)[1]
    @test gr == 2 .*x .+ 3
    #
    gdev = gpu_device()
    cdev = cpu_device()
    x_dev = gdev(x)
    is_dummy_dev = gdev(is_dummy)
    #is_dummy_dev = x_dev .>= 4.0
    xm_dev = CP.repeat_rowvector_dummy(x_dev', is_dummy_dev)
    gr = Zygote.gradient(x -> sum(sin, CP.repeat_rowvector_dummy(x', is_dummy_dev)), x_dev)[1];
    @test cdev(gr) ≈ 2 .* cos.(x)
end

@testset "repeat_rowvector_dummymatrix" begin
    x = collect(1.0:5.0) 
    is_dummy = fill(false, (4, length(x)))
    is_dummy[3:4, 2] .= true
    xm = CP.repeat_rowvector_dummy(x', is_dummy)
    @test xm[:,[1,3,4,5]] == xm[:,[1,3,4,5]] == repeat(x[[1,3,4,5]]', 4)
    @test xm[1:2,2] ==  fill(x[2], 2)
    @test all(isnan.(xm[3:4,2]))
    #
    #tmp = Zygote.gradient(is_dummy -> sum(repeat_rowvector_dummy(x', is_dummy)), is_dummy) 
    gr = Zygote.gradient(x -> sum(CP.repeat_rowvector_dummy(x', is_dummy)), x)[1]
    @test gr == [4,2,4,4,4.0]
    gr = Zygote.gradient(x -> sum(CP.repeat_rowvector_dummy(x', is_dummy).^2), x)[1]
    @test gr == [4,2,4,4,4.0] .* 2.0 .*x
    gr = Zygote.gradient(x -> sum(sin, CP.repeat_rowvector_dummy(x', is_dummy)), x)[1]
    @test gr == [4,2,4,4,4.0] .* cos.(x)
    gr = (Zygote.gradient(x) do x 
        y = CP.repeat_rowvector_dummy(x', is_dummy) 
        z = sum(abs2, y[1,:]) + 3*sum(y[2:end,:])
    end)[1]
    @test gr == 2 .* x .+ [3,1,3,3,3.0] .* 3
    #
    x_dev = gdev(x)
    is_dummy_dev = gdev(is_dummy)
    #is_dummy_dev = x_dev .>= 4.0
    xm_dev = CP.repeat_rowvector_dummy(x_dev', is_dummy_dev)
    #tmp = Zygote.gradient(is_dummy -> sum(repeat_rowvector_dummy(x', is_dummy)), is_dummy) 
    gr = Zygote.gradient(x -> sum(sin, CP.repeat_rowvector_dummy(x', is_dummy_dev)), x_dev)[1];
    @test cdev(gr) ≈ [4,2,4,4,4.0] .* cos.(x)
end


function test_driverNaN(scenario::Val{scen})  where scen
    prob = HybridProblem(DoubleMM.DoubleMMCase(); scenario);
    if (:use_rangescaling ∈ scen)
        @test prob.g isa RangeScalingModelApplicator
    end
    solver_point = HybridPointSolver(; alg=Adam(0.02))
    rng = StableRNG(111)
    (;ϕ, resopt, probo) = solve(prob, solver_point; rng,
        #callback = callback_loss(100), # output during fitting
        #callback = callback_loss(10), # output during fitting
        epochs = 2,
        scenario,
    );
    @test all(isfinite.(ϕ))
    (;y_pred, θMs, θP) = predict_point_hvi(rng, probo; scenario);
    _,_,y_obs,_ = get_hybridproblem_train_dataloader(prob; scenario).data
    @test size(y_pred) == size(y_obs)
    y_predc = cdev(y_pred)
    @test all(isfinite.(y_predc[isfinite.(y_obs)]))    
    #
    # takes long, only activate on suspicious
    () -> begin
        solver = HybridPosteriorSolver(; alg=Adam(0.02), n_MC=3)
        (; probo, interpreters) = solve(prob, solver; rng,
            callback = callback_loss(10), # output during fitting
            epochs = 2,
            scenario,
        );    
        @test all(isfinite.(probo.θP))
        n_sample_pred = 12
        (; y, θsP, θsMs, entropy_ζ) = predict_hvi(rng, probo; scenario, n_sample_pred);
        @test size(y) == (size(y_pred)..., n_sample_pred)
        yc = cdev(y)
        _ = map(eachslice(yc; dims = 3)) do ycs
            @test all(isfinite.(ycs[isfinite.(y_obs)]))    
        end
    end
end

@testset "HybridPointSolver driverNaN cpu" begin
    scenario = Val((:driverNAN,))
    test_driverNaN(scenario)
end

# @testset "HybridPointSolver driverNaN ML on gpu" begin
#     scenario = Val((:driverNAN, :use_Lux, :use_gpu))
#     test_driverNaN(scenario)
# end

@testset "HybridPointSolver driverNaN also PBM on gpu" begin
    scenario = Val((:driverNAN, :use_rangescaling, :use_Lux, :use_dropout, :use_gpu, :f_on_gpu))
    test_driverNaN(scenario)
end

