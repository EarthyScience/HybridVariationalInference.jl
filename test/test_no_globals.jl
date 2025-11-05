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

# scenario = Val(()); scen=()
function test_no_globals(scenario::Val{scen})  where scen
    scenario = Val((scen..., :no_globals))
    prob = HybridProblem(DoubleMM.DoubleMMCase(); scenario);
    @test isempty(prob.θP)
    solver_point = HybridPointSolver(; alg=Adam(0.02))
    rng = StableRNG(111)
    (;ϕ, resopt, probo) = solve(prob, solver_point; rng,
        #callback = callback_loss(100), # output during fitting
        #callback = callback_loss(10), # output during fitting
        epochs = 2,
        is_omit_priors = (:f_on_gpu ∈ scen), # prior computation does not work on gpu
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
    #() -> begin
        solver = HybridPosteriorSolver(; alg=Adam(0.02), n_MC=3)
        (; probo, interpreters) = solve(prob, solver; rng,
            #callback = callback_loss(10), # output during fitting
            is_omit_priors = (:f_on_gpu ∈ scen), # prior computation does not work on gpu
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
    #end
end

@testset "noglobals cpu" begin
    scenario = Val(())
    test_no_globals(scenario)
end

@testset "noglobals gpu" begin
    scenario = Val((:use_Lux, :use_gpu))
    test_no_globals(scenario)
end

# currently empty transformation of empty CuArrays do not work
# @testset "noglobals also PBM on gpu" begin
#     #scen = (:use_Lux, :use_gpu, :f_on_gpu)
#     scenario = Val((:use_Lux, :use_gpu, :f_on_gpu))
#     test_no_globals(scenario)
# end

