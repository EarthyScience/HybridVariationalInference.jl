using Test
using HybridVariationalInference
import HybridVariationalInference as CP
using ComponentArrays: ComponentArrays as CA


function f_pop!(pred, θsc, xPc)
    a1 = view(θsc, Val(:a1),:)
    a2 = view(θsc, Val(:a2),:)
    b = view(θsc, Val(:b),:)
    c = view(θsc, Val(:c),:)
    pred .= c' .* a1' .+  log.(a2') .* abs2.(cos.(b' .- 0.2)) .* abs2.(xPc.s1)
    pred
end

() -> begin
    include("test/test_scratch.jl")
end

@testset "PBMPopulationApplicator" begin
    n_obs = 3
    n_site = 5
    n_MC = 4
    xPvec = CA.ComponentVector(s1 = 1.0:n_obs)
    xPc = xPvec .* ones(n_site)' .+ abs2.(randn(n_obs, n_site) .* 0.1)
    θP = CA.ComponentVector(b=3.0)
    θM = CA.ComponentVector(a1=2.0,a2=1.0)
    θFix  = CA.ComponentVector(c=1.5)
    #
    θsP = (θP .* ones(n_MC)') 
    θsM = (θM .* ones(n_MC)') .+ abs2.(randn(length(θM), n_MC) .* 0.1)
    θsFix = (θFix .* ones(n_MC)') 
    θs = vcat(vcat(θsP, θsM), θsFix)
    y_obs = f_pop!(zeros(n_obs, n_MC), θs, xPvec)
    pred = similar(y_obs)
    g = PBMPopulationApplicator(f_pop!, n_MC; θP, θM, θFix, xPvec)
    ret = CP.apply_model!(similar(y_obs), g,θsP, θsM, xPvec)
    @test ret ≈ y_obs
    @test ((pred, g,θsP, θsM, xPvec) -> @allocated CP.apply_model!(pred, g,θsP, θsM, xPvec))(pred, g,θsP, θsM, xPvec) == 0
    # function loop_apply_model(n, g,θsP, θsM, xPvec)
    #     for i in 1:n
    #         CP.apply_model!(pred, g,θsP, θsM, xPvec)
    #     end
    # end
    # loop_apply_model(1, g,θsP, θsM, xPvec)
    # @profview_allocs loop_apply_model(10_000, g,θsP, θsM, xPvec)
end;

