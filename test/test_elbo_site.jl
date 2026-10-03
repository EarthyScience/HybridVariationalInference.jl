ENV["MLDATADEVICES_SILENCE_WARN_NO_GPU"]="1" # suppress warning on missing CUDA
import Distributed
if Distributed.nprocs() == 1
    Distributed.addprocs(2)
else
    if Distributed.nworkers() < 2; Distributed.addprocs(2-Distributed.nworkers()); end
end

#Distributed.pmap((i) -> Threads.nthreads(), 1:Distributed.nworkers())

#using LinearAlgebra, BlockDiagonals
using LinearAlgebra
using StatsFuns: logistic

using Test
Distributed.@everywhere using HybridVariationalInference
using HybridVariationalInference: HybridVariationalInference as CP
using StableRNGs
using Random
using ComponentArrays: ComponentArrays as CA
#using TransformVariables
using Bijectors
import PreallocationTools as PAT
import JLD2
import Transducers
import Profile
import PDMats

rng = StableRNG(1234)


n_covP0 = 0
n_covP2 = 2
n_cov = 3
n_θP = 3 
n_θM = 3 
n_M = n_θM + 1 # additional uncertainty scaling factor

import Lux
import Zygote
import Enzyme
import ForwardDiff

import SimpleChains
isUsingSimpleChains = false
#isUsingSimpleChains = true  # currently does not work with enyzem nor zygote
#   and provides no general way to compute the pullback wrt. both covariates xM and ϕg

    if isUsingSimpleChains    
        n_input = n_cov + n_covP0
        chain0 = SimpleChains.SimpleChain(
                SimpleChains.static(n_input), # input dimension (optional)
                # dense layer with bias that maps to 8 outputs and applies `tanh` activation
                SimpleChains.TurboDense{true}(tanh, n_input * 4),
                SimpleChains.TurboDense{true}(tanh, n_input * 4),
                # dense layer without bias that maps to n outputs and `logistic` activation
                SimpleChains.TurboDense{false}(logistic, n_M)
            )
        n_input = n_cov + n_covP2
        chain2 = SimpleChains.SimpleChain(
                SimpleChains.static(n_input), # input dimension (optional)
                # dense layer with bias that maps to 8 outputs and applies `tanh` activation
                SimpleChains.TurboDense{true}(tanh, n_input * 4),
                SimpleChains.TurboDense{true}(tanh, n_input * 4),
                # dense layer without bias that maps to n outputs and `logistic` activation
                SimpleChains.TurboDense{false}(logistic, n_M)
            )
    else
        n_input = n_cov + n_covP0
        chain0 = Lux.Chain(
            # dense layer with bias that maps to 8 outputs and applies `tanh` activation
            Lux.Dense(n_input => n_input * 4, tanh),
            Lux.Dense(n_input * 4 => n_input * 4, tanh),
            # dense layer without bias that maps to n outputs and `logistic` activation
            Lux.Dense(n_input * 4 => n_M, logistic, use_bias = false)
        )
        n_input = n_cov + n_covP2
        chain2 = Lux.Chain(
            # dense layer with bias that maps to 8 outputs and applies `tanh` activation
            Lux.Dense(n_input => n_input * 4, tanh),
            Lux.Dense(n_input * 4 => n_input * 4, tanh),
            # dense layer without bias that maps to n outputs and `logistic` activation
            Lux.Dense(n_input * 4 => n_M, logistic, use_bias = false)
        )
    end
    g, ϕg = construct_ChainsApplicator(rng, chain0, Float32)
    ϕgv = collect(ϕg)
    #
    n_site = 8
    n_MC = 3
    #
    #cor_ends = (P = [n_θP], M = [2, n_θM])
    cor_ends = (P = [2, n_θP], M = [n_θM])
    ϕqPc1 = CA.ComponentVector(
        μζP = [-1, 0, 1.0], 
        logσ_ζP = ones(n_MC) .* log(0.01),
        ρsP = [0.1],
        )
    ϕqIc1 = CA.ComponentVector(
        #logσ_ζM = ones(n_MC) .* log(0.02),
        logσ_ζM = log.([0.02, 0.06, 0.04]),
        ρsM = [0.1, 0.2, 0.3],
        )
    ϕqP = ϕqP2 = CA.getdata(ϕqPc1)
    ϕqI = CA.getdata(ϕqIc1)
    intϕqP = get_concrete(ComponentArrayInterpreter(ϕqPc1))
    intϕqI = get_concrete(ComponentArrayInterpreter(ϕqIc1))
    #
    xM = randn(eltype(ϕg),n_cov, n_site)
    y = CP.apply_model(g, xM, ϕg)
    #
    ζP = randn(n_θP)
    ζsP = ζP .+ 0.1 * randn(n_θP, n_MC)
    #
    # without population covariates
    pbm_covar_indices0 = Int[]   # but better use nothing for efficient dispatch
    xMP0=zeros(eltype(xM), (size(xM,1)+ n_covP0), size(xM,2) * n_MC)
    xMP0_old=zeros(eltype(xM), size(xM,1) + n_covP0, size(xM,2))
    #ϕms0vz = CP.g_apply_oop(ϕg, xM, ζsP, pbm_covar_indices0, g, xMP0)[:,1,:] # sites equal
    ϕms0vz = CP.g_apply_oop(ϕgv, xM, ζsP, pbm_covar_indices0, g, xMP0)
    #ϕms0v = convert.(eltype(ϕqP), zero(ϕms0vz))
    ϕms0v = similar(ϕms0vz, eltype(ϕqP))
    CP.g_apply!(ϕms0v, ϕgv, xM, ζsP, pbm_covar_indices0, g, xMP0, false)
    ϕms0v == ϕms0vz
    # @usingany BenchmarkTools
    # @benchmark CP.g_apply!(ϕms0v, ϕgv, xM, ζsP, pbm_covar_indices0, g, xMP0)
    # tmpf = (ϕms0v, ϕgv, xM, ζsP, pbm_covar_indices0, g, xMP0) -> @allocated CP.g_apply!(ϕms0v, ϕgv, xM, ζsP, pbm_covar_indices0, g, xMP0)
    # tmpf(ϕms0v, ϕgv, xM, ζsP, pbm_covar_indices0, g, xMP0)
    # with providing nothing instead of an empty list, omit the n_mc dimension
    ϕms0z = CP.g_apply_oop(ϕgv, xM, ζsP, nothing, g, xMP0)
    ϕms0 = similar(ϕms0z, eltype(ϕqP))
    CP.g_apply!(ϕms0, ϕgv, xM, ζsP, nothing, g, xMP0, false)
    ϕms0 == ϕms0z

    #
    # with popuolation covariates
    pbm_covar_indices2 = Int[2,3]   
    g2, ϕg2 = construct_ChainsApplicator(rng, chain2, Float32)
    ϕg2v = collect(ϕg2)
    xMP=zeros(eltype(xM), (size(xM,1)+ n_covP2), size(xM,2) * n_MC)
    ϕms2z = CP.g_apply_oop(ϕg2v, xM, ζsP, pbm_covar_indices2, g2, xMP)
    ϕms2 = similar(ϕms2z, eltype(ϕqP))
    CP.g_apply!(ϕms2, ϕg2v, xM, ζsP, pbm_covar_indices2, g2, xMP, false)
    ϕms2 == ϕms2z

    # preallocate helpers and shadows
    rnormPM = CP.prepare_rnorm(ϕqP; n_θP, n_θM, n_MC, n_site)
    # size(rnormPM.P)
    approx = CP.DiagonalHVIApproximation()
    CP.randnPM!(rng, rnormPM)
    h0 = CP.prepare_elbo_helpers(approx, ϕg, ϕqP; n_θP, n_θM, n_site, n_MC, n_cov, n_covP = n_covP0, n_M, cor_ends)
    h2 = CP.prepare_elbo_helpers(approx, ϕg2, ϕqP; n_θP, n_θM, n_site, n_MC, n_cov, n_covP = n_covP2, n_M, cor_ends)
    CP.check_elbo_helpers(h0, xM, pbm_covar_indices0; n_ϕg = length(ϕg))
    CP.check_elbo_helpers(h2, xM, pbm_covar_indices2; n_ϕg = length(ϕg2))
    randn!(h0.θsP) # for testing should initialized to finite values
    randn!(h2.θsP) # for testing should initialized to finite values
    #
    approxM = CP.MeanHVIApproximation()
    h0M = CP.prepare_elbo_helpers(approxM, ϕg, ϕqP; n_θP, n_θM, n_site, n_MC, n_cov, n_covP = n_covP0, n_M, cor_ends)
    h2M = CP.prepare_elbo_helpers(approxM, ϕg2, ϕqP; n_θP, n_θM, n_site, n_MC, n_cov, n_covP = n_covP2, n_M, cor_ends)
    randn!(h0M.θsP) # for testing should initialized to finite values
    randn!(h2M.θsP) # for testing should initialized to finite values
    #
    ϕqIcS = CA.ComponentVector(;logσ_ζM_offsets = ϕqIc1.logσ_ζM[2:end] .- ϕqIc1.logσ_ζM[1], ϕqIc1.ρsM)
    intϕqIS = get_concrete(ComponentArrayInterpreter(ϕqIcS))
    approxS = CP.MeanUniScalingHVIApproximation(ϕqIc1.logσ_ζM[1])
    h0S = CP.prepare_elbo_helpers(approxS, ϕg, ϕqP; n_θP, n_θM, n_site, n_MC, n_cov, n_covP = n_covP0, n_M, cor_ends)
    h2S = CP.prepare_elbo_helpers(approxS, ϕg2, ϕqP; n_θP, n_θM, n_site, n_MC, n_cov, n_covP = n_covP2, n_M, cor_ends)
    randn!(h0S.θsP) # for testing should initialized to finite values
    randn!(h2S.θsP) # for testing should initialized to finite values


@testset "_setU_scaled!" begin
    ϕqIc = intϕqI(ϕqI)
    U = zeros(n_θM, n_θM)
    CP._setU_scaled!(U, ϕqIc.ρsM)
    @test diag(U' * U) ≈ ones(eltype(U), n_θM)
    v = zeros(n_θM)
    CP._setρ_unscaled!(v, U)
    @test v ≈ ϕqIc.ρsM
end

@testset "sample_ζsP!" begin
    σt = [0.06, 0.08, 0.01]
    approxS = MeanUniScalingHVIApproximation(σt[1])
    n_MCt = 10_000
    rnormPt = randn(n_θP, n_MCt)
    ζsPt = similar(rnormPt)
    logσ_ζPt = zeros(n_θP)
    Σct = PDMats.PDMat([1.0 0.8 0.0; 0.8 1.0 0.0; 0.0 0.0 1.0])
    Ut = cholesky(Σct).U
    ρst = zeros(1)  #zeros(CP.sumn(n_θP-1))
    CP._setρ_unscaled!(ρst, Ut[1:2,1:2])
    ϕqPct = CA.ComponentVector(μζP = [-1.0, 0.0, 1.0], logσ_ζP=log.(σt), ρsP=ρst)
    h2M = CP.prepare_elbo_helpers(approxS, ϕg, ϕqP; n_θM, n_θP, n_site, n_MC = n_MCt, 
        n_cov, n_covP = n_covP0, n_M, cor_ends, use_diff_cache=Val(false))
    CP.sample_ζsP!(ζsPt, logσ_ζPt, approxS, rnormPt, ϕqPct, cor_ends.P, h2M.sample_buffers)
    @test vec(mean(ζsPt, dims=2)) ≈ ϕqPct.μζP atol=0.01
    @test cor(ζsPt[1,:], ζsPt[2,:]) ≈ Σct[1,2] atol=0.02
    @test cor(ζsPt[1,:], ζsPt[3,:]) ≈ Σct[1,3] atol=0.02
    @test cor(ζsPt[2,:], ζsPt[3,:]) ≈ Σct[2,3] atol=0.02
    @test std(ζsPt[1,:]) ≈ σt[1] atol=0.01
    @test std(ζsPt[2,:]) ≈ σt[2] atol=0.01
    @test std(ζsPt[3,:]) ≈ σt[3] atol=0.01  
    @test ((ζsP, logσ_ζP, approx, rnormP, ϕqPc, cor_endsP, sample_buffers) -> 
        @allocated CP.sample_ζsP!(ζsP, logσ_ζP, approx, rnormP, ϕqPc, cor_endsP, sample_buffers))(
        ζsPt, logσ_ζPt, approxS, rnormPt, ϕqPct, cor_ends.P, h2M.sample_buffers) == 0
    function loop_samplesample_ζsP(n, ζsP, logσ_ζP, approx, rnormP, ϕqPc, cor_endsP, sample_buffers) 
        for i in 1:n
            CP.sample_ζsP!(ζsP, logσ_ζP, approx, rnormP, ϕqPc, cor_endsP, sample_buffers)
        end
    end
    #@profview_allocs loop_samplesample_ζsP(1000, ζsPt, logσ_ζPt, approxS, rnormPt, ϕqPct, cor_ends.P, h2M.sample_buffers)
end

@testset "sample_ζsM!" begin
    ϕqIc = intϕqI(ϕqI)
    h0_1 = h0.helpers_sites[1]
    # test allocation
    ϕm = rand(n_θM+1, n_MC)
    j = 3
    # wrap inside function to aovid allocation due to boxing type unstable globals
    ((ϕm, n_θM,j) -> @allocated ϕm[:,j][1:n_θM])(ϕm,n_θM,j)
    ((ϕm,n_θM,j) -> @allocated view(ϕm,1:n_θM,j))(ϕm, n_θM,j)  
    rnormM1 = rnormPM.M[1]
    ζsM = similar(rnormM1)
    logσ_ζM = zeros(n_θM)
    sample_buffers = h0_1.sample_buffers
    CP.sample_ζsM!(ζsM, logσ_ζM, approx, rnormM1, ϕqIc, ϕm, cor_ends.M, sample_buffers)
    @test ((ζsM, logσ_ζM, approx, rnormM1, ϕqIc, ϕm, cor_endsM, sample_buffers) -> @allocated CP.sample_ζsM!(ζsM, logσ_ζM, approx, rnormM1, ϕqIc, ϕm, cor_endsM, sample_buffers))(
        ζsM, logσ_ζM, approx, rnormM1, ϕqIc, ϕm, cor_ends.M, sample_buffers) == 0
    #
    # vector version
    ϕm1 = ϕm[:,1] 
    CP.sample_ζsM!(ζsM, logσ_ζM, approx, rnormM1, ϕqIc, ϕm1, cor_ends.M, sample_buffers)
    #allocations because h1 is global
    #  @allocated CP.sample_ζsM!(ζsM, logσ_ζM, rnormM1, ϕqIc, ϕm1, buffer_nθM)
    tmpf1 = (ζsM, logσ_ζM, approx, rnormM1, ϕqIc, ϕm1, cor_endsM, sample_buffers) -> @allocated CP.sample_ζsM!(ζsM, logσ_ζM, approx, rnormM1, ϕqIc, ϕm1, cor_endsM, sample_buffers)
    @test tmpf1(ζsM, logσ_ζM, approx, rnormM1, ϕqIc, ϕm1, cor_ends.M, sample_buffers)  == 0

    # capture global variables in closure to avoid allocations
    get_f_fd1 = (h1, intϕqI, approx::AbstractHVIApproximation, cor_endsM) -> (ϕqP, ϕm1, rnormM1, template) -> begin
        local ϕqIc = intϕqI(ϕqP) # without local allocations by @safetestset, shadows global
        local ζsMb = PAT.get_tmp(h1.ζsM, template)
        local logσ_ζMb = PAT.get_tmp(h1.logσ_ζM, template)
        local sample_buffers = h1.sample_buffers
        CP.sample_ζsM!(ζsMb, logσ_ζMb, approx, rnormM1, ϕqIc, ϕm1, cor_endsM, sample_buffers)
        sum(ζsMb) + sum(logσ_ζMb)
    end
    f_fd1 = get_f_fd1(h0_1, intϕqI, approx, cor_ends.M)
        # grad_ϕq = ForwardDiff.gradient(f_fd1, ϕqP)
        # ϕqd = convert.(typeof(ForwardDiff.Dual(ϕqP[1])), ϕqP)
        # @allocated f_fd1(ϕqd)
    # vector version
    ϕqd = convert.(typeof(ForwardDiff.Dual(ϕqP[1])), ϕqP)
    f_fd1(ϕqP, ϕm1, rnormM1, ϕqP)
    f_fd1(ϕqd, ϕm1, rnormM1, ϕqd)
    @test (@allocated f_fd1(ϕqP, ϕm1, rnormM1, ϕqP)) == 0
    @test (@allocated f_fd1(ϕqd, ϕm1, rnormM1, ϕqd)) == 0
    # matrix version
    f_fd1(ϕqP, ϕm1, rnormM1, ϕqP)
    f_fd1(ϕqd, ϕm, rnormM1, ϕqd)
    @test (@allocated f_fd1(ϕqP, ϕm, rnormM1, ϕqP)) == 0
    @test (@allocated f_fd1(ϕqd, ϕm, rnormM1, ϕqd)) == 0

    approx2 = MeanHVIApproximation()
    n_MCt = 10_000
    rnormM1t = randn(n_θM, n_MCt)
    ζsMt = similar(rnormM1t)
    logσ_ζMt = zeros(n_θM)
    μt = randn(n_θM) 
    ϕmt = repeat(μt, 1, n_MCt)
    Σct = PDMats.PDMat([1.0 0.8 0.6; 0.8 1.0 0.8; 0.6 0.8 1.0])
    Ut = cholesky(Σct).U
    ρsMt = zeros(CP.sumn(n_θM-1))
    CP._setρ_unscaled!(ρsMt, Ut)
    σt = [0.06, 0.08, 0.01]
    ϕqIct = CA.ComponentVector(logσ_ζM=log.(σt), ρsM=ρsMt)
    h2M = CP.prepare_elbo_helpers(approx2, ϕg, ϕqP; n_θP, n_θM, n_site, n_MC = n_MCt, 
        n_cov, n_covP = n_covP0, n_M, cor_ends, use_diff_cache=Val(false))
    CP.sample_ζsM!(ζsMt, logσ_ζMt, approx2, rnormM1t, ϕqIct, ϕmt, cor_ends.M, h2M.helpers_sites[1].sample_buffers)
    @test vec(mean(ζsMt, dims=2)) ≈ μt atol=0.01
    @test cor(ζsMt[1,:], ζsMt[2,:]) ≈ Σct[1,2] atol=0.02
    @test cor(ζsMt[1,:], ζsMt[3,:]) ≈ Σct[1,3] atol=0.02
    @test cor(ζsMt[2,:], ζsMt[3,:]) ≈ Σct[2,3] atol=0.02
    @test std(ζsMt[1,:]) ≈ σt[1] atol=0.01
    @test std(ζsMt[2,:]) ≈ σt[2] atol=0.01
    @test std(ζsMt[3,:]) ≈ σt[3] atol=0.01  
    @test ((ζsM, logσ_ζM, approx, rnormM1, ϕqIc, ϕm, cor_endsM, sample_buffers) -> 
        @allocated CP.sample_ζsM!(ζsM, logσ_ζM, approx, rnormM1, ϕqIc, ϕm, cor_endsM, sample_buffers))(
        ζsMt, logσ_ζMt, approx2, rnormM1t, ϕqIct, ϕmt, cor_ends.M, h2M.helpers_sites[1].sample_buffers) == 0
    function loop_samplesample_ζsM(n, ζsM, logσ_ζM, approx, rnormM1, ϕqIc, ϕm, cor_endsM, sample_buffers) 
        for i in 1:n
            CP.sample_ζsM!(ζsM, logσ_ζM, approx, rnormM1, ϕqIc, ϕm, cor_endsM, sample_buffers)
        end
    end
    #@profview_allocs loop_samplesample_ζsM(1000, ζsMt, logσ_ζMt, approx2, rnormM1t, ϕqIct, ϕmt, cor_ends.M, h2M.helpers_sites[1].sample_buffers)
    @allocated loop_samplesample_ζsM(1000, ζsMt, logσ_ζMt, approx2, rnormM1t, ϕqIct, ϕmt, cor_ends.M, h2M.helpers_sites[1].sample_buffers)

    σt = [0.06, 0.08, 0.01]
    approxS = MeanUniScalingHVIApproximation(log(σt[1]))
    n_MCt = 10_000
    rnormM1t = randn(n_θM, n_MCt)
    ζsMt = similar(rnormM1t)
    logσ_ζMt = zeros(n_θM)
    μt = randn(n_θM) 
    scale_fac = 1.2
    ϕm_scaling = logistic(log(scale_fac))
    ϕmt = repeat(vcat(μt, ϕm_scaling), 1, n_MCt)
    Σct = PDMats.PDMat([1.0 0.8 0.6; 0.8 1.0 0.8; 0.6 0.8 1.0])
    Ut = cholesky(Σct).U
    ρsMt = zeros(CP.sumn(n_θM-1))
    CP._setρ_unscaled!(ρsMt, Ut)
    σt_scaled = σt .* scale_fac
    ϕqIct = CA.ComponentVector(logσ_ζM_offsets=(log.(σt[2:end]) .- log(σt[1])), ρsM=ρsMt)
    h2S = CP.prepare_elbo_helpers(approxS, ϕg, ϕqP; n_θP, n_θM, n_site, n_MC = n_MCt, 
        n_cov, n_covP = n_covP0, n_M, cor_ends, use_diff_cache=Val(false))
    CP.sample_ζsM!(ζsMt, logσ_ζMt, approxS, rnormM1t, ϕqIct, ϕmt, cor_ends.M, h2S.helpers_sites[1].sample_buffers)
    @test vec(mean(ζsMt, dims=2)) ≈ μt atol=0.01
    @test cor(ζsMt[1,:], ζsMt[2,:]) ≈ Σct[1,2] atol=0.02
    @test cor(ζsMt[1,:], ζsMt[3,:]) ≈ Σct[1,3] atol=0.02
    @test cor(ζsMt[2,:], ζsMt[3,:]) ≈ Σct[2,3] atol=0.02
    @test std(ζsMt; dims=2) ≈ σt_scaled atol=0.01
    @test ((ζsM, logσ_ζM, approx, rnormM1, ϕqIc, ϕm, cor_endsM, sample_buffers) -> 
        @allocated CP.sample_ζsM!(ζsM, logσ_ζM, approx, rnormM1, ϕqIc, ϕm, cor_endsM, sample_buffers))(
        ζsMt, logσ_ζMt, approxS, rnormM1t, ϕqIct, ϕmt, cor_ends.M, h2S.helpers_sites[1].sample_buffers) == 0
    #@profview_allocs loop_samplesample_ζsM(1000, ζsMt, logσ_ζMt, approxS, rnormM1t, ϕqIct, ϕmt, cor_ends.M, h2S.helpers_sites[1].sample_buffers)
    @allocated loop_samplesample_ζsM(1000, ζsMt, logσ_ζMt, approxS, rnormM1t, ϕqIct, ϕmt, cor_ends.M, h2S.helpers_sites[1].sample_buffers)

    # # Regression: sample_ζsM! must not allocate when reached through the
    # # ForwardDiff-reconstructed ComponentArray-view path used by
    # # grad_neg_elbo_sites (make_nelboiz_cl in elbo_site_grad.jl). The tests above
    # # pass concrete Float64 arrays, which miss this allocation: under ForwardDiff
    # # the arguments become Dual-element views into a flat ComponentArray, and the
    # # sampling allocates. Reconstruct cv_ = ComponentArray(cv, ax_inputs) and take
    # # the Val views, exactly as nelboiz_cl does. The output buffers are obtained
    # # from DiffCache get_tmp (preallocated, so their construction is not measured)
    # # and passed into nelboiz_alloc, mirroring compute_nelboi_z!.
    # cv_grad = CA.ComponentArray(; ϕqIc = ϕqIc, ϕm = ϕm, θsP = h0.θsP)
    # ax_inputs = CA.getaxes(similar(cv_grad))
    # x0 = CA.getdata(cv_grad)
    # chunk = ForwardDiff.Chunk(8)   # must match the DiffCache chunk below
    # gcv = zeros(length(x0))        # preallocated gradient buffer (mirrors hwi.grads_v)
    # hi_alloc = (;   # only the reusable buffers are hoisted out of the measured region
    #     ζsM_dc = PAT.DiffCache(zeros(n_θM, n_MC), 8),
    #     logσ_ζM_dc = PAT.DiffCache(zeros(n_θM), 8),
    #     buffer_nθM_dc = PAT.DiffCache(zeros(n_θM), 8),
    # )
    # function nelboiz_alloc(cv, hi, rnormM1, ax_inputs, template)
    #     cv_ = CA.ComponentArray(cv, ax_inputs)
    #     ζsM = PAT.get_tmp(hi.ζsM_dc, template)
    #     logσ_ζM = PAT.get_tmp(hi.logσ_ζM_dc, template)
    #     buffer_nθM = PAT.get_tmp(hi.buffer_nθM_dc, template)
    #     CP.sample_ζsM!(ζsM, logσ_ζM, rnormM1,
    #         view(cv_, Val(:ϕqIc)), view(cv_, Val(:ϕm)), buffer_nθM)
    #     sum(ζsM)
    # end
    # nelboi_f = x -> nelboiz_alloc(x, hi_alloc, rnormM1, ax_inputs, x)
    # nelboi_f(x0)
    # cfg = ForwardDiff.GradientConfig(nelboi_f, x0, chunk)
    # gfun = x -> begin
    #     ForwardDiff.gradient!(gcv, nelboi_f, x, cfg)
    #     nothing
    # end
    # gfun(x0)
    # @test (@allocated gfun(x0)) == 0
end

@testset "compute_nelboi_z!" begin
    ϕqIc = intϕqI(ϕqI)
    ϕqPc = intϕqP(ϕqP)
    CP.sample_ζsP!(h0.ζsP, h0.logσ_ζP, approx, rnormPM.P, ϕqPc, cor_ends.P, h0.sample_buffers) # n_P * n_MC
    CP.g_apply!(h0.ϕms, ϕg, xM, h0.ζsP, nothing, g, h0.xMP, false)     
    hi1 = h0.helpers_sites[1]
    i_site_train1 = 1:n_site
    rnormM1 = rnormPM.M[1]
    ϕms1 = h0.ϕms[:,1]
    θsP1 = h0.θsP
    CP.compute_nelboi_z!(hi1, approx, rnormM1, i_site_train1, ϕms1, ϕqIc, θsP1, cor_ends.M)     
    #@code_warntype CP.compute_nelboi_z!(hi1, rnormM1, i_site_train1, ϕms1, ϕqIc, θsP1)
    # need two wrap in tmpf to avoid allocations due to boxing
    function tmpf(hi, approx, rnormM1, i_site_train1, ϕms1, ϕqIc, θsP1, cor_endsM)
        @test (@allocated CP.compute_nelboi_z!(hi, approx, rnormM1, i_site_train1, ϕms1, ϕqIc, θsP1, cor_endsM)) == 0
    end
    tmpf(hi1, approx, rnormM1, i_site_train1, ϕms1, ϕqIc, θsP1, cor_ends.M)
        #@profview tmpf()
        #using BenchmarkTools
        #@btime CP.compute_nelboi_z!($hi1, $approx, $rnormM1, $i_site_train1, $ϕms1, $ϕqIc, $θsP1)    
    #
    # test with views as input to compute_nelboi_z! and differnt ϕm per MC
    h21 = h2.helpers_sites[1]
    CP.g_apply!(h2.ϕms_mcs, ϕg2, xM, h2.ζsP, pbm_covar_indices2, g2, h2.xMP, false)     
    ϕm = randn!(similar(h2.ϕms_mcs[:,:,1]))
    inputs = CA.ComponentVector(ϕqIc=ϕqIc, ϕm = ϕm, θsP= θsP1)
    CP.compute_nelboi_z!(h21, approx, rnormM1, i_site_train1, inputs.ϕm, inputs.ϕqIc, inputs.θsP, cor_ends.M)
    # test calling with views
    ϕms1_ = view(inputs, Val(:ϕm))
    ϕqIc_ = view(inputs, Val(:ϕqIc))
    θsP1_ = view(inputs, Val(:θsP))
    function loop_compute_nelboi_z(n, h21, approx, rnormM1, i_site_train1, ϕms1_, ϕqIc_, θsP1_, cor_endsM)
        for _ in 1:n
            @noinline CP.compute_nelboi_z!(h21, approx, rnormM1, i_site_train1, ϕms1_, ϕqIc_, θsP1_, cor_endsM)
        end
        nothing
    end
    function alloc_compute_nelboi_z(h21, approx, rnormM1, i_site_train1, ϕms1_, ϕqIc_, θsP1_, cor_endsM)
        # avoid global variables -> pass them through function
        loop_compute_nelboi_z(1, h21, approx, rnormM1, i_site_train1, ϕms1_, ϕqIc_, θsP1_, cor_endsM) # thorough warmup
        @test (@allocated loop_compute_nelboi_z(100,h21, approx, rnormM1, i_site_train1, ϕms1_, ϕqIc_, θsP1_, cor_endsM)) == 0
    end
    alloc_compute_nelboi_z(h21, approx, rnormM1, i_site_train1, ϕms1_, ϕqIc_, θsP1_, cor_ends.M)
    # 
    gradh2 = CP.prepare_gradelbo_helpers(inputs.ϕqIc, inputs.ϕm, inputs.θsP, ϕg, ϕqPc, approx; 
        pbm_covar_indices=pbm_covar_indices2, n_workers=1,
        h=h2, rnormM1 = rnormPM.M[1], i_site_train1 = 1,
        h2.diffchunk, n_site, n_cov, cor_ends)
    hw_channel = gradh2.hw_channel
    tmp = with_channel_element(x -> x.inputs_cv, hw_channel)
    tmp.ϕm

    # alternative: invoke forwarddiff_grad_nelboi_z! with ϕqIc_ and θsP_ as views,
    # passing dϕmvecs as the plain array. The views are created once *outside* the
    # measured region so that their construction is not counted by @allocated.
    function loop_forwarddiff_grad_nelboi_z(
        n, hi, approx, rnormM, i_site_train, i, ϕm_, ϕqIc_, θsP_, dϕmvecs_, hw_channel, omit_gradient, cor_endsM)
        for _ in 1:n
            @noinline CP.forwarddiff_grad_nelboi_z!(hi, approx, rnormM, i_site_train, ϕm_, i, 
                ϕqIc_, θsP_, dϕmvecs_, hw_channel, omit_gradient, cor_endsM)
        end
        nothing
    end
    ϕm_   = view(inputs, Val(:ϕm))
    ϕqIc_ = view(inputs, Val(:ϕqIc))
    θsP_  = view(inputs, Val(:θsP))
    dϕmvecs = gradh2.dϕmvecs
    i_site_train_1 = i_site_train1[1]
    function alloc_forwarddiff_grad_nelboi_z(h21, approx, rnormM1, i_site_train_1, ϕm_, ϕqIc_, θsP_, dϕmvecs, hw_channel, cor_endsM)
        loop_forwarddiff_grad_nelboi_z(1, h21, approx, rnormM1, i_site_train_1, 1, ϕm_, ϕqIc_, θsP_, dϕmvecs, hw_channel, true, cor_endsM)
        @test (@allocated loop_forwarddiff_grad_nelboi_z(100, h21, approx, rnormM1, i_site_train_1, 1, ϕm_, ϕqIc_, θsP_, dϕmvecs, hw_channel, true, cor_endsM)) == 0
        loop_forwarddiff_grad_nelboi_z(1, h21, approx, rnormM1, i_site_train_1, 1, ϕm_, ϕqIc_, θsP_, dϕmvecs, hw_channel, nothing, cor_endsM)
        @test (@allocated loop_forwarddiff_grad_nelboi_z(100, h21, approx, rnormM1, i_site_train_1, 1, ϕm_, ϕqIc_, θsP_, dϕmvecs, hw_channel, nothing, cor_endsM)) == 0
    end
    alloc_forwarddiff_grad_nelboi_z(h21, approx, rnormM1, i_site_train1[1], ϕm_, ϕqIc_, θsP_, dϕmvecs, hw_channel, cor_ends.M)

    #@usingany Cthulhu
    #@descend_code_warntype tmp_g(h21, rnormM1, i_site_train1, inputs, 1, dϕmvecs, nothing, hw_channel)
    #@descend_code_warntype tmp_g2(h21, rnormM1, i_site_train1, inputs, 1, dϕmvecs, nothing, hw_channel)
    #@usingany BenchmarkTools
    #@benchmark tmp_g($h21, $rnormM1, $i_site_train1, $inputs, 1, $dϕmvecs, true, $hw_channel)
    #@profview_allocs tmpgn(h21, rnormM1, i_site_train1, inputs, 1, dϕmvecs, true, hw_channel)
    #@profview_allocs tmpgn(h21, rnormM1, i_site_train1, inputs, 1, dϕmvecs, nothing, hw_channel)

    approx2 = MeanHVIApproximation()
    hiM1 = h0M.helpers_sites[1]
    @test isfinite(CP.compute_nelboi_z!(hiM1, approx2, rnormM1, i_site_train1, 
        ϕms1, ϕqIc, θsP1, cor_ends.M))
    function alloc_compute_nelboi_zM!(hiM1, approx2, rnormM1, i_site_train1, 
        ϕms1, ϕqIc, θsP1, cor_endsM)
        @test (@allocated CP.compute_nelboi_z!(hiM1, approx2, rnormM1, i_site_train1, 
        ϕms1, ϕqIc, θsP1, cor_endsM)) == 0 
    end
    alloc_compute_nelboi_zM!(hiM1, approx2, rnormM1, i_site_train1, 
        ϕms1, ϕqIc, θsP1, cor_ends.M)
end

@testset "pullback_g_apply!" begin
#     @test ϕms0 == ϕms0z
#     @test ϕms2 == ϕms2z
#     @test size(ϕms0) == (n_M, n_site) 
#     @test size(ϕms2) == (n_M, n_MC, n_site) 
#     () -> begin # gradient(sum)
#         gr_zygote = Zygote.gradient((ϕg2) -> sum(CP.g_apply_oop(ϕg2, xM, ζP, pbm_covar_indices2, g2, xMP)), ϕg2v )
#         s, pullback_s_zygote = Zygote.pullback((ϕg2) -> sum(CP.g_apply_oop(ϕg2, xM, ζP, pbm_covar_indices2, g2, xMP)), ϕg2v )
#         #gr_zygote2 = pullback_s_zygote(ones(eltype(ϕg2v), size(ϕg2v)...))
#         gr_zygote2 = pullback_s_zygote(one(eltype(ϕg2v)))
#         @test gr_zygote2[1] ≈ gr_zygote[1]
#         y, pullback_zygote = Zygote.pullback((ϕg2) -> CP.g_apply_oop(ϕg2, xM, ζP, pbm_covar_indices2, g2, xMP), ϕg2v )
#         gr_zygote3 = pullback_zygote(ones(eltype(y), size(y)...))
#         gr_zygote3[1] ≈ gr_zygote[1]
#     end
#     # 
#     # concatenate function f3(g(ϕ_g))
#     f3 = (x) -> sum(3.0 .* x)
#     CP.g_apply_oop(ϕg2, xM, ζsP, pbm_covar_indices2, g2, xMP)
#     # one pass of composed function
#     gr_zygote = Zygote.gradient((ϕg2, ζsP) -> f3(CP.g_apply_oop(ϕg2, xM, ζsP, pbm_covar_indices2, g2, xMP)), ϕg2v, ζsP )
#     s, pullback_s_zygote = Zygote.pullback((ϕg2) -> f3(CP.g_apply_oop(ϕg2, xM, ζsP, pbm_covar_indices2, g2, xMP)), ϕg2v )
#     gr_zygote2 = pullback_s_zygote(one(eltype(s)))
#     @test gr_zygote2[1] ≈ gr_zygote[1]
#     # mixed AD, differentiate f3 by FowardDiff and pull back through g
#     y_oop = CP.g_apply_oop(ϕg2, xM, ζsP, pbm_covar_indices2, g2, xMP)
#     #gr_h = Zygote.gradient(y -> sum(f3(y)), y1)[1]
#     gr_h = ForwardDiff.gradient(y -> sum(f3(y)), y_oop)
#     y, pullback_zygote = Zygote.pullback((ϕg2) -> CP.g_apply_oop(ϕg2, xM, ζsP, pbm_covar_indices2, g2, xMP), ϕg2v )
#     @test y == y_oop
#     gr_zygote3 = pullback_zygote(gr_h)
#     gr_zygote3[1] ≈ gr_zygote[1]
#     #
#     dϕg = zero(ϕg2v)
#     dζsP = zero(ζsP)
#     #Enzyme.make_one!(y)
#     y .= rand()
#     dϕg .= rand() # check that initial values do not effect result
#     dζsP .= rand() # check that initial values do not effect result
#     dy = convert.(eltype(y), gr_h)
#     CP.pullback_g_apply!(y, dϕg, dζsP, dy, ϕg2v, xM, ζsP, pbm_covar_indices2, g2, h)
#     @test y == y_oop
#     @test dϕg ≈ gr_zygote[1]
#     @test dζsP ≈ gr_zygote[2] rtol=1e-3
#     @test dy ≈ gr_h # not modified
#     #@benchmark CP.pullback_g_apply!(y, dϕg, dy, ϕg2v, xM, ζsP, pbm_covar_indices2, g2, h)
#     #
    # () -> begin # explicitly splitting the forward and backward pass
    #     # they get cached anymay and require allocating the Duplicated Wrappers twice
    #     #   hence there is no performance benefit
    #     # Compile once outside the hot loop
    #     fwd, rev = Enzyme.autodiff_thunk(
    #         Enzyme.ReverseSplitNoPrimal,
    #         Enzyme.Const{typeof(g_apply!)},
    #         Enzyme.Const,
    #         Enzyme.Duplicated{typeof(y)},
    #         Enzyme.Duplicated{typeof(ϕg2v)},
    #         Enzyme.Const{typeof(xM)},
    #         Enzyme.Const{typeof(ζP)},
    #         Enzyme.Const{typeof(pbm_covar_indices2)},
    #         Enzyme.Const{typeof(g2)},
    #         Enzyme.Duplicated{typeof(h.xMP)}
    #     )
    #     # take care, dy is also modified
    #     function grad2_g_apply!(y, dϕg, dy, ϕg, xM, ζP, pbm_covar_indices, g, h, fwd, rev)
    #         fill!(dϕg, zero(eltype(dϕg)))
    #         fill!(h.dxMP,  zero(eltype(h.dxMP)))
    #         copyto!(h.dy, dy) # copy to avoid modifying dy
    #         tape, _, _ = fwd(
    #             Enzyme.Const(g_apply!),
    #             Enzyme.Duplicated(y, h.dy),
    #             Enzyme.Duplicated(ϕg, dϕg),
    #             Enzyme.Const(xM),
    #             Enzyme.Const(ζP),
    #             Enzyme.Const(pbm_covar_indices),
    #             Enzyme.Const(g),
    #             Enzyme.Duplicated(h.xMP, h.dxMP)
    #         )
    #         rev(
    #             Enzyme.Const(g_apply!),
    #             Enzyme.Duplicated(y, h.dy),
    #             Enzyme.Duplicated(ϕg, dϕg),
    #             Enzyme.Const(xM),
    #             Enzyme.Const(ζP),
    #             Enzyme.Const(pbm_covar_indices),
    #             Enzyme.Const(g),
    #             Enzyme.Duplicated(h.xMP, h.dxMP),
    #             tape
    #         )
    #         return nothing
    #     end

#         dy = convert.(eltype(y), gr_h) 
#         grad2_g_apply!(y, dϕg, dy, ϕg2v, xM, ζP, pbm_covar_indices2, g2, h, fwd, rev)
#         @test y == y_oop
#         @test dϕg ≈ gr_zygote[1]
#         @test dy ≈ gr_h # not modified
#         #@usingany BenchmarkTools
#         #@benchmark grad2_g_apply!(y, dϕg, dy, ϕg2v, xM, ζP, pbm_covar_indices2, g2, h, fwd, rev)
#     end
end

@testset "pullback_sample_ζsP!" begin
#     ϕqPc = intϕqP(ϕqP)
#     rnormP = zero(ζsP)
#     randn!(rnormP)  # before input gaussian noise
#     ζsP .= 0
#     #logσ_ζP = zero(ϕqPc.logσ_ζP) # cretes a view rather than copy
#     logσ_ζP = zero(ϕqPc.logσ_ζP)
#     CP.sample_ζsP!(ζsP, logσ_ζP, rnormP, ϕqPc)
#     mean(ζsP; dims=2)
#     ζsP1 = copy(ζsP)

#     # Enzyme result via the mutating routine (2-D: n_θP * n_MC × n_in)
#     dζsP = zero(ζsP) .+ one(eltype(ζsP))
#     dlogσ_ζP = zero(logσ_ζP) .+ one(eltype(ζsP))
#     dϕqc = zero(ϕqPc) 

#     randn!(dϕqc) # test that is zerod inside pullback
#     dζsP_ = copy(dζsP)
#     rnormP_ = copy(rnormP)
#     logσ_ζP_ = copy(logσ_ζP)
#     dlogσ_ζP_ = copy(dlogσ_ζP)
#     #CP.pullback_sample_ζsP!(dϕqc, dζsP, dlogσ_ζP, rnormP, logσ_ζP, ϕqPc) # needs rnormP to be noise
#     CP.pullback_sample_ζsP!(dϕqc, dζsP, dlogσ_ζP, ζsP, logσ_ζP, rnormP, ϕqPc)
#     @test ζsP == ζsP1 # same forward result
#     @test rnormP == rnormP_
#     @test dζsP == dζsP_
#     @test dlogσ_ζP == dlogσ_ζP_
#     @test logσ_ζP == logσ_ζP_
#     # without correlation
#     #@test all(dϕqc[Val(:μζP)] .== n_MC)
#     # #@test dϕqc[Val(:logσ_ζP)] ≈ vec(sum(rnormP; dims=2)) # 
#     dϕqc_comb = copy(dϕqc)

#     randn!(ζsP)  # test initial not relevant
#     pb_sample_ζsP = CP.primal_pullback_sample_ζsP!(ζsP, logσ_ζP, rnormP, ϕqPc)
#     @test ζsP == ζsP1 # same forward result
#     @test rnormP ≈ rnormP_# computed the forward pass
#     @test logσ_ζP == logσ_ζP_
#     dϕqc .= 0.1 # test initial value not relevant
#     #dϕqc .= 0.01 # should not influence results
#     pb_sample_ζsP(dϕqc, dζsP, dlogσ_ζP)    
#     #pb_sample_ζsP(rnormP, logσ_ζP)
#     @test rnormP ≈ rnormP_  # did not modify
#     @test logσ_ζP == logσ_ζP_ # not modified
#     @test dlogσ_ζP == dlogσ_ζP_ # not modified
#     @test dζsP == dζsP_
#     #hcat(dϕqc, dϕqc_comb)
#     @test CA.getdata(dϕqc) ≈ CA.getdata(dϕqc_comb)
#     #
#     # test another pullback
#     #dζsP .= dζsP * eltype(dζsP)(2)
#     pb_sample_ζsP(dϕqc, dζsP, dlogσ_ζP)    
#     @test CA.getdata(dϕqc) ≈ CA.getdata(dϕqc_comb)
#     #
#     # @usingany BenchmarkTools
#     # @benchmark pb_sample_ζsP(dϕqc, dζsP, dlogσ_ζP)    
end

function grad_neg_elbo_sites_enzyme() # differentiate entire neg_elbo_sites by enzyme
    # do not use DiffCache here for helpers_sites
    h0p = CP.prepare_elbo_helpers(approx, ϕg, ϕqP; 
        n_θP, n_θM, n_site, n_MC, n_cov, n_covP = n_covP0, n_M, cor_ends, use_diff_cache = Val(false))
    # and store results to compare to hand-crafted mixed AD
    dh0p = Enzyme.make_zero(h0p)
    @test dh0p !== h0p # real copy rather than reference
    dϕg = zero(ϕgv)
    dϕqP = zero(ϕqP)
    dϕqI = zero(ϕqI)
    _ftmp2 = (ϕgv, h, approx, rnormPM, ϕqP, ϕqI, g, pbm_covar_indices, intϕqP, intϕqI, xM, cor_ends, i_sites_train) -> 
        CP.neg_elbo_sites!(
        h, approx, rnormPM, ϕgv, ϕqP, ϕqI, g, pbm_covar_indices;
        i_sites_train,     
        intϕqP, intϕqI,
        xM,
        cor_ends,
        is_testmode = false,
        )[1]    
    pbm_covar_indices_nothing = nothing
    #_f(ϕg2v, h, g2, pbm_covar_indices2)
    
    Enzyme.make_zero!(dϕg)
    Enzyme.make_zero!(dϕqP)
    Enzyme.make_zero!(dϕqI)
    Enzyme.make_zero!(dh0p)
    rng1 = StableRNG(1234)
    CP.randnPM!(rng1, rnormPM)   
    randn!(rng1, ϕgv)
    randn!(rng1, xM)
    primal_enz = _ftmp2(ϕgv, h0p, approx, rnormPM, ϕqP, ϕqI, g, pbm_covar_indices_nothing, intϕqP, intϕqI, xM, cor_ends,1:n_site)
    Enzyme.autodiff(
            Enzyme.set_runtime_activity(Enzyme.Reverse) ,
            _ftmp2,
            Enzyme.Active,
            Enzyme.Duplicated(ϕgv, dϕg),
            Enzyme.Duplicated(h0p, dh0p),
            Enzyme.Const(approx),
            Enzyme.DuplicatedNoNeed(rnormPM, Enzyme.make_zero(rnormPM)),
            Enzyme.Duplicated(ϕqP, dϕqP),
            Enzyme.Duplicated(ϕqI, dϕqI),
            Enzyme.Const(g),
            Enzyme.Const(pbm_covar_indices_nothing),
            Enzyme.Const(intϕqP),
            Enzyme.Const(intϕqI),
            Enzyme.Const(xM),
            Enzyme.Const(cor_ends),
            Enzyme.Const(1:n_site),
        )   
    dϕg0_enz = copy(dϕg)
    dϕqP0_enz = copy(dϕqP)
    dϕqI0_enz = copy(dϕqI)
    () -> begin
        #@usingany JLD2
        #fname = "intermediate/test_enzyme_dphi0.jld2"
        fname = "intermediate/test_enzymeT_dphi0.jld2"
        mkpath("intermediate")
        JLD2.jldsave(fname, false, IOStream; primal_enz, dϕg0_enz, dϕqP0_enz, dϕqI0_enz)
        primal_enz, dϕg0_enz, dϕqP0_enz, dϕqI0_enz = JLD2.load(fname, 
            "primal_enz", "dϕg0_enz", "dϕqP0_enz", "dϕqI0_enz");
    end

    h2p = CP.prepare_elbo_helpers(approx, ϕg2, ϕqP; 
        n_θP, n_θM, n_site, n_MC, n_cov, n_covP = n_covP2, n_M, cor_ends, use_diff_cache = Val(false))
    dϕg2 = zero(ϕg2v)
    dh2p = Enzyme.make_zero(h2p)
    @test dh2p !== h2p # real copy rather than reference

    Enzyme.make_zero!(dϕg2)
    Enzyme.make_zero!(dϕqP)
    Enzyme.make_zero!(dϕqI)
    Enzyme.make_zero!(dh2p)
    rng1 = StableRNG(1234)
    CP.randnPM!(rng1, rnormPM)   
    randn!(rng1, ϕg2v)
    randn!(rng1, xM)
    primal2_enz = _ftmp2(ϕg2v, h2p, approx, rnormPM, ϕqP, ϕqI, g2, pbm_covar_indices2, intϕqP, intϕqI, xM, cor_ends, 1:n_site)
    Enzyme.autodiff(
            Enzyme.set_runtime_activity(Enzyme.Reverse) ,
            _ftmp2,
            Enzyme.Active,
            Enzyme.Duplicated(ϕg2v, dϕg2),
            Enzyme.Duplicated(h2p, dh2p),
            Enzyme.Const(approx),
            Enzyme.DuplicatedNoNeed(rnormPM, Enzyme.make_zero(rnormPM)),
            Enzyme.Duplicated(ϕqP, dϕqP),
            Enzyme.Duplicated(ϕqI, dϕqI),
            Enzyme.Const(g2),
            Enzyme.Const(pbm_covar_indices2),
            Enzyme.Const(intϕqP),
            Enzyme.Const(intϕqI),
            Enzyme.Const(xM),
            Enzyme.Const(cor_ends),
            Enzyme.Const(1:n_site),
        )   
    dϕg2_enz = copy(dϕg2)
    dϕqP2_enz = copy(dϕqP)
    dϕqI2_enz = copy(dϕqI)
    () -> begin
        #fname = "intermediate/test_enzyme_dphi2.jld2"
        fname = "intermediate/test_enzymeT_dphi2.jld2"
        mkpath("intermediate")
        JLD2.jldsave(fname, false, IOStream; primal2_enz, dϕg2_enz, dϕqP2_enz, dϕqI2_enz)
        primal2_enz, dϕg2_enz, dϕqP2_enz, dϕqI2_enz = JLD2.load(fname, 
            "primal2_enz", "dϕg2_enz", "dϕqP2_enz", "dϕqI2_enz");
    end
end

function grad_neg_elbo_sites_enzyme_Cor() # differentiate entire neg_elbo_sites by enzyme
    # now with more complicated Correlation approximation
    # do not use DiffCache here for helpers_sites
    h0p = CP.prepare_elbo_helpers(approxM, ϕg, ϕqP; 
        n_θP, n_θM, n_site, n_MC, n_cov, n_covP = n_covP0, n_M, cor_ends, use_diff_cache = Val(false))
    # and store results to compare to hand-crafted mixed AD
    dh0p = Enzyme.make_zero(h0p)
    @test dh0p !== h0p # real copy rather than reference
    dϕg = zero(ϕgv)
    dϕqP = zero(ϕqP)
    dϕqI = zero(ϕqI)
    _ftmp2 = (ϕgv, h, approx, rnormPM, ϕqP, ϕqI, g, pbm_covar_indices, intϕqP, intϕqI, xM, cor_ends, i_sites_train) -> 
        CP.neg_elbo_sites!(
        h, approx, rnormPM, ϕgv, ϕqP, ϕqI, g, pbm_covar_indices;
        i_sites_train,     
        intϕqP, intϕqI,
        xM,
        cor_ends,
        is_testmode = false,
        )[1]    
    pbm_covar_indices_nothing = nothing
    #_f(ϕg2v, h, g2, pbm_covar_indices2)
    
    Enzyme.make_zero!(dϕg)
    Enzyme.make_zero!(dϕqP)
    Enzyme.make_zero!(dϕqI)
    Enzyme.make_zero!(dh0p)
    rng1 = StableRNG(1234)
    CP.randnPM!(rng1, rnormPM)   
    randn!(rng1, ϕgv)
    randn!(rng1, xM)
    primal_enz = _ftmp2(ϕgv, h0p, approxM, rnormPM, ϕqP, ϕqI, g, pbm_covar_indices_nothing, intϕqP, intϕqI, xM, cor_ends,1:n_site)
    Enzyme.autodiff(
            Enzyme.set_runtime_activity(Enzyme.Reverse) ,
            _ftmp2,
            Enzyme.Active,
            Enzyme.Duplicated(ϕgv, dϕg),
            Enzyme.Duplicated(h0p, dh0p),
            Enzyme.Const(approxM),
            Enzyme.DuplicatedNoNeed(rnormPM, Enzyme.make_zero(rnormPM)),
            Enzyme.Duplicated(ϕqP, dϕqP),
            Enzyme.Duplicated(ϕqI, dϕqI),
            Enzyme.Const(g),
            Enzyme.Const(pbm_covar_indices_nothing),
            Enzyme.Const(intϕqP),
            Enzyme.Const(intϕqI),
            Enzyme.Const(xM),
            Enzyme.Const(cor_ends),
            Enzyme.Const(1:n_site),
        )   
    dϕg0_enz = copy(dϕg)
    dϕqP0_enz = copy(dϕqP)
    dϕqI0_enz = copy(dϕqI)
    () -> begin
        #@usingany JLD2
        #fname = "intermediate/test_enzyme_dphi0.jld2"
        fname = "intermediate/test_enzymeM_dphi0.jld2"
        mkpath("intermediate")
        JLD2.jldsave(fname, false, IOStream; primal_enz, dϕg0_enz, dϕqP0_enz, dϕqI0_enz)
        primal_enz, dϕg0_enz, dϕqP0_enz, dϕqI0_enz = JLD2.load(fname, 
            "primal_enz", "dϕg0_enz", "dϕqP0_enz", "dϕqI0_enz");
    end

    h2p = CP.prepare_elbo_helpers(approxM, ϕg2, ϕqP; 
        n_θP, n_θM, n_site, n_MC, n_cov, n_covP = n_covP2, n_M, cor_ends, use_diff_cache = Val(false))
    dϕg2 = zero(ϕg2v)
    dh2p = Enzyme.make_zero(h2p)
    @test dh2p !== h2p # real copy rather than reference

    Enzyme.make_zero!(dϕg2)
    Enzyme.make_zero!(dϕqP)
    Enzyme.make_zero!(dϕqI)
    Enzyme.make_zero!(dh2p)
    rng1 = StableRNG(1234)
    CP.randnPM!(rng1, rnormPM)   
    randn!(rng1, ϕg2v)
    randn!(rng1, xM)
    primal2_enz = _ftmp2(ϕg2v, h2p, approxM, rnormPM, ϕqP, ϕqI, g2, pbm_covar_indices2, intϕqP, intϕqI, xM, cor_ends, 1:n_site)
    Enzyme.autodiff(
            Enzyme.set_runtime_activity(Enzyme.Reverse) ,
            _ftmp2,
            Enzyme.Active,
            Enzyme.Duplicated(ϕg2v, dϕg2),
            Enzyme.Duplicated(h2p, dh2p),
            Enzyme.Const(approxM),
            Enzyme.DuplicatedNoNeed(rnormPM, Enzyme.make_zero(rnormPM)),
            Enzyme.Duplicated(ϕqP, dϕqP),
            Enzyme.Duplicated(ϕqI, dϕqI),
            Enzyme.Const(g2),
            Enzyme.Const(pbm_covar_indices2),
            Enzyme.Const(intϕqP),
            Enzyme.Const(intϕqI),
            Enzyme.Const(xM),
            Enzyme.Const(cor_ends),
            Enzyme.Const(1:n_site),
        )   
    dϕg2_enz = copy(dϕg2)
    dϕqP2_enz = copy(dϕqP)
    dϕqI2_enz = copy(dϕqI)
    () -> begin
        #fname = "intermediate/test_enzyme_dphi2.jld2"
        fname = "intermediate/test_enzymeM_dphi2.jld2"
        mkpath("intermediate")
        JLD2.jldsave(fname, false, IOStream; primal2_enz, dϕg2_enz, dϕqP2_enz, dϕqI2_enz)
        primal2_enz, dϕg2_enz, dϕqP2_enz, dϕqI2_enz = JLD2.load(fname, 
            "primal2_enz", "dϕg2_enz", "dϕqP2_enz", "dϕqI2_enz");
    end
end

@testset "grad_neg_elbo_sites" begin
    ϕqIc = intϕqI(ϕqI)
    ϕqPc = intϕqP(ϕqP)
    n_threads_proc = min(Threads.nthreads(), 4)
    n_workers = Distributed.nworkers() * n_threads_proc
    basesize = n_site ÷ Distributed.nworkers()
    distributedEx = Transducers.DistributedEx(;basesize, threads_basesize = Int(ceil(basesize / n_threads_proc))) 

    #
    rng1 = StableRNG(1234)
    CP.randnPM!(rng1, rnormPM)
    randn!(rng1, ϕgv)
    randn!(rng1, xM)
    primal = CP.neg_elbo_sites!(
        h0, approx, rnormPM,
        ϕgv, ϕqP, ϕqI, g, nothing;
        i_sites_train = 1:n_site,     
        intϕqP, intϕqI,
        xM,
        cor_ends,
        is_testmode = false,
    )
    res0, gradh0 = CP.grad_neg_elbo_sites(
    #@descend_code_warntype CP.neg_elbo_sites!(
        h0, (;), approx,
        rnormPM,
        ϕgv, ϕqP, ϕqI, g, nothing;
        i_sites_train = 1:n_site,     
        intϕqP, intϕqI,
        xM,
        cor_ends,
        is_testmode = false,
    )    
    res0_, gradh0_ = CP.grad_neg_elbo_sites( # test deterministic result and distributed
        h0, gradh0, approx,
        rnormPM,
        ϕgv, ϕqP, ϕqI, g, nothing;
        i_sites_train = 1:n_site,     
        intϕqP, intϕqI,
        xM,
        cor_ends,
        is_testmode = false,
        executor = distributedEx
    )    
    @test all(map(≈, res0_,  res0))
    # if we saved Enzyme results earlier, compare to them
    if isfile("intermediate/test_enzymeT_dphi0.jld2")
        primal_enz, dϕg0_enz, dϕqP0_enz, dϕqI0_enz = JLD2.load(
            "intermediate/test_enzymeT_dphi0.jld2", 
            "primal_enz", "dϕg0_enz", "dϕqP0_enz", "dϕqI0_enz");
        @test primal_enz ≈ primal[1]
        @test dϕg0_enz ≈ res0.dϕg
        @test dϕqI0_enz ≈ res0.dϕqI
        @test dϕqP0_enz ≈ res0.dϕqP
        #hcat(dϕqP0_enz, CA.getdata(res0.dϕqP))
        #dϕqP0_enz - CA.getdata(res0.dϕqP)
    end
    #
    #---------------- matrix mode with population covariates
    rng1 = StableRNG(1234)
    CP.randnPM!(rng1, rnormPM)
    randn!(rng1, ϕg2v)
    randn!(rng1, xM)
    primal2 = CP.neg_elbo_sites!(
        h2, approx, rnormPM,
        ϕg2v, ϕqP, ϕqI, g2, pbm_covar_indices2;
        i_sites_train = 1:n_site,     
        intϕqP, intϕqI,
        xM,
        cor_ends,
        is_testmode = false,
    )
    res0, gradh2 = CP.grad_neg_elbo_sites(
    #@descend_code_warntype CP.neg_elbo_sites!(
        h2, (;), approx,
        rnormPM,
        ϕg2v, ϕqP, ϕqI, g2, pbm_covar_indices2;
        i_sites_train = 1:n_site,     
        intϕqP, intϕqI,
        xM,
        cor_ends,
        is_testmode = false,
    )    
    res0_, gradh2_ = CP.grad_neg_elbo_sites( # test deterministic result
        h2, gradh2, approx,
        rnormPM,
        ϕg2v, ϕqP, ϕqI, g2, pbm_covar_indices2;
        i_sites_train = 1:n_site,     
        intϕqP, intϕqI,
        xM,
        cor_ends,
        is_testmode = false,
        executor = distributedEx,
    )    
    @test all(map(≈, res0_,  res0))
    # if we saved Enzyme results earlier, compare to them
    if isfile("intermediate/test_enzymeT_dphi2.jld2")
        primal2_enz, dϕg2_enz, dϕqP2_enz, dϕqI2_enz = JLD2.load(
            "intermediate/test_enzymeT_dphi2.jld2", 
            "primal2_enz", "dϕg2_enz", "dϕqP2_enz", "dϕqI2_enz");
        @test primal2_enz ≈ primal2[1]
        @test dϕg2_enz ≈ res0.dϕg
        @test dϕqI2_enz ≈ res0.dϕqI
        @test dϕqP2_enz ≈ res0.dϕqP
        #hcat(dϕqP2_enz, res0.dϕqP)
    end
    #---------------- approxM with non-empty h.sample_buffers and scaling
    rng1 = StableRNG(1234)
    CP.randnPM!(rng1, rnormPM)
    randn!(rng1, ϕg2v)
    randn!(rng1, xM)
    primal2 = CP.neg_elbo_sites!(
        h2S, approxS, rnormPM,
        ϕg2v, ϕqP, CA.getdata(ϕqIcS), g2, pbm_covar_indices2;
        i_sites_train = 1:n_site,     
        intϕqP, intϕqI = intϕqIS,
        xM,
        cor_ends,
        is_testmode = false,
    )
    res0, gradh2S = CP.grad_neg_elbo_sites(
        h2S, (;), approxS,
        rnormPM,
        ϕg2v, ϕqP, CA.getdata(ϕqIcS), g2, pbm_covar_indices2;
        i_sites_train = 1:n_site,     
        intϕqP, intϕqI = intϕqIS,
        xM,
        cor_ends,
        is_testmode = false,
        executor = distributedEx,
    )    


    function loop_grad_neg_elbo_sites(n, h2, gradh2, approx, rnormPM, ϕg2v, ϕqP, ϕqI, g2, pbm_covar_indices2; 
        i_sites_train, intϕqP, intϕqI, xM, cor_ends, is_testmode)
        for _ in 1:n
            @noinline CP.grad_neg_elbo_sites(h2, gradh2, approx, rnormPM, ϕg2v, ϕqP, ϕqI, g2, pbm_covar_indices2; 
        i_sites_train, intϕqP, intϕqI, xM, cor_ends, is_testmode)
        end
        nothing
    end
    function alloc_grad_neg_elbo_sites(h2, gradh2, approx, rnormPM, ϕg2v, ϕqP, ϕqI, g2, 
            pbm_covar_indices2; i_sites_train, intϕqP, intϕqI, xM, cor_ends, is_testmode)
        # avoid global variables -> pass them through function
        loop_grad_neg_elbo_sites(1, h2, gradh2, approx, rnormPM, ϕg2v, ϕqP, ϕqI, g2, 
            pbm_covar_indices2; i_sites_train, intϕqP, intϕqI, xM, cor_ends, is_testmode)
        # @profview_allocs loop_grad_neg_elbo_sites(10_000,h2, gradh2, rnormPM, ϕg2v, 
        #      ϕqP, ϕqI, g2, pbm_covar_indices2; i_sites_train, intϕqP, intϕqI, xM, cor_ends, is_testmode)
        # a_ = @allocated loop_grad_neg_elbo_sites(100,h2, gradh2, rnormPM, ϕg2v, ϕqP, ϕqI, g2, 
        #     pbm_covar_indices2; i_sites_train, intϕqP, intϕqI, xM, cor_ends, is_testmode)        
        # @show a_
        @test (@allocated loop_grad_neg_elbo_sites(100,h2, gradh2, approx, rnormPM, ϕg2v, ϕqP, ϕqI, g2, 
            pbm_covar_indices2; i_sites_train, intϕqP, intϕqI, xM, cor_ends, is_testmode)) <= 2_546_672
    end
    alloc_grad_neg_elbo_sites(h2, gradh2, approx, rnormPM, ϕg2v, ϕqP, ϕqI, g2, pbm_covar_indices2; 
        i_sites_train = 1:n_site, intϕqP, intϕqI, xM, cor_ends, is_testmode = false)
    #_i_sites_train = 1:n_site    
    #@usingany BenchmarkTools
    #@benchmark CP.grad_neg_elbo_sites($h2, $gradh2, $rnormPM, $ϕg2v, $ϕqP, $ϕqI, $g2, $pbm_covar_indices2; i_sites_train = _i_sites_train, intϕqP=$intϕqP, intϕqI=$intϕqI, xM=$xM, is_testmode = false)

end






