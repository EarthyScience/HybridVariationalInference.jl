using Test
using HybridVariationalInference
using ComponentArrays: ComponentArrays as CA

import Zygote

using MLDataDevices, CUDA, cuDNN, GPUArraysCore

f_pop = function(θsc, xPc)
    local n_obs = size(xPc, 1)
    is_valid = isfinite.(CA.getdata(xPc))
    a1 = is_valid .* CA.getdata(θsc[:,:a1])' 
    a2 = is_valid .* CA.getdata(θsc[:,:a2])' 
    # b in θP has been expanded in PopulationApplicator
    b = is_valid .* CA.getdata(θsc[:,:b])' 
    y = a1 .+  log.(a2) .* abs2.(cos.(b .- 0.2)) .* xPc.^2
end

() -> begin
    include("test/test_scratch.jl")
end

@testset "PBMPopulationApplicator" begin
    n_obs = 3
    n_site = 5
    xPvec = CA.ComponentVector(s1 = 1.0:n_obs)
    xPc = xPvec .* ones(n_site)' .+ abs2.(randn(n_obs, n_site) .* 0.1)
    θP = CA.ComponentVector(b=3.0)
    θM = CA.ComponentVector(a1=2.0,a2=1.0)
    θFix  = CA.ComponentVector(c=1.5)
    #
    θMs = (ones(n_site) .* θM') .+ abs2.(randn(n_site, length(θM)) .* 0.1)
    θs = hcat(ones(n_site) .* θP', θMs)
    y_obs = f_pop(θs, xPc)
    g = PBMPopulationApplicator(f_pop, n_site; θP, θM, θFix, xPvec)
    ret = g(θP, θMs, xPc)
    @test ret ≈ y_obs

    gr = Zygote.gradient((θP, θMs) -> sum(g(θP, θMs, xPc)), CA.getdata(θP), CA.getdata(θMs))

    xPc_NaN = copy(xPc); xPc_NaN[2:3,1] .= NaN
    ret2 = g(θP, θMs, xPc_NaN)
    gr = Zygote.gradient((θP, θMs) -> sum(g(θP, θMs, xPc_NaN)), CA.getdata(θP), CA.getdata(θMs))
    @test all(isfinite.(gr[1])) 

    n_site_m = 4
    gm = create_nsite_applicator(g, n_site_m)
    retm = gm(θP, θMs[1:n_site_m,:], xPc[:,1:n_site_m])
    @test retm ≈ y_obs[:,1:n_site_m]
end;

f_global = function(θsc, θgc, xPc)
    local n_obs = size(xPc, 1)
    #is_dummy = isnan.(CA.getdata(xPc))
    is_valid = isfinite.(CA.getdata(xPc))
    # rep_fac = ones_similar_x(θsc, n_obs)
    # a1 = rep_fac .* CA.getdata(θsc[:,:a1])'
    # a2 = rep_fac .* CA.getdata(θsc[:,:a2])'
    #a1 = repeat_rowvector_dummy(CA.getdata(θsc[:,:a1])', is_dummy)
    #a2 = repeat_rowvector_dummy(CA.getdata(θsc[:,:a2])', is_dummy)
    a1 = is_valid .* CA.getdata(θsc[:,:a1])' 
    a2 = is_valid .* CA.getdata(θsc[:,:a2])' 
    b = CA.getdata(θgc.b) .* (is_valid)
    y = a1 .+  log.(a2) .* abs2.(cos.(b .- 0.2)) .* xPc.^2
end

@testset "PBMPopulationGlobalApplicator" begin
    n_obs = 3
    n_site = 5
    xPvec = CA.ComponentVector(s1 = 1.0:n_obs)
    xPc = xPvec .* ones(n_site)' .+ abs2.(randn(n_obs, n_site) .* 0.1)
    θP = CA.ComponentVector(b=3.0)
    θM = CA.ComponentVector(a1=2.0,a2=1.0)
    θFix  = CA.ComponentVector(c=1.5)
    #
    θMs = (ones(n_site) .* θM') .+ abs2.(randn(n_site, length(θM)) .* 0.1)
    y_obs = f_global(θMs, θP, xPc)
    g = PBMPopulationGlobalApplicator(f_global, n_site; θP, θM, θFix, xPvec)
    ret = g(θP, θMs, xPc)
    @test ret ≈ y_obs

    gr = Zygote.gradient((θP, θMs) -> sum(g(θP, θMs, xPc)), CA.getdata(θP), CA.getdata(θMs))

    xPc_NaN = copy(xPc); xPc_NaN[2:3,1] .= NaN
    ret2 = g(θP, θMs, xPc_NaN)
    gr = Zygote.gradient((θP, θMs) -> sum(g(θP, θMs, xPc_NaN)), CA.getdata(θP), CA.getdata(θMs))
    @test all(isfinite.(gr[1])) # \thetaP
    @test all(isfinite.(gr[2])) # solves finite gradient for a2 for first site

    n_site_m = 4
    gm = create_nsite_applicator(g, n_site_m)
    retm = gm(θP, θMs[1:n_site_m,:], xPc[:,1:n_site_m])
    @test retm ≈ y_obs[:,1:n_site_m]
end;

