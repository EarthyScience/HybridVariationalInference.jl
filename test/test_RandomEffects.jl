using Test
using HybridVariationalInference
using HybridVariationalInference: HybridVariationalInference as HVI
using HybridVariationalInference: HybridVariationalInference as CP
using LinearAlgebra
using StaticArrays: StaticArrays as SA
using ComponentArrays: ComponentArrays as CA
using Distributions

@testset "CVPrior_LKJ_Cauchy" begin
    prior = CVPrior_LKJ_Cauchy(3)
    cm = Hermitian([1.0 0.5 0.3; 0.5 1.0 0.4; 0.3 0.4 1.0] .* 2.0)
    logpdf_prior = logpdf(prior, cm)
    @test isfinite(logpdf_prior)
end

@testset "CVPrior_LKJ_Cauchy one column" begin
    prior = CVPrior_LKJ_Cauchy(1)
    cm = Hermitian(reshape([3.0],1,1))
    logpdf_prior = logpdf(prior, cm)
    @test isfinite(logpdf_prior)
end

@testset "CVPrior_LKJ_Cauchy Float32" begin
    FT = Float32
    prior = CVPrior_LKJ_Cauchy(3, one(FT))
    @test partype(prior.dLKJ) == FT
    cm = Hermitian(FT[1.0 0.5 0.3; 0.5 1.0 0.4; 0.3 0.4 1.0] .* FT(2.0))
    logpdf_prior = logpdf(prior, cm)
    @test isfinite(logpdf_prior)
    @test logpdf_prior isa FT
end

@testset "CVPrior_LKJ_Cauchy convert_prior" begin
    prior64 = CVPrior_LKJ_Cauchy(1)
    prior32 = HVI.convert_prior(prior64, one(Float32))
    @test logpdf(prior32, diagm([1.0f0])) isa Float32
end

() -> begin
    d = Cauchy(0.0, 0.5)
    #using StatsPlots
    plot(d, xlim=(0,12))
end

@testset "NullRandomEffects" begin
    nre = NullRandomEffects()
    n_site = 200
    ranef = HVI.get_ranef_computer(nre, (:_,), n_site)
    ϕq_ranef = setup_ϕq_ranef(ranef)
    @test size(ϕq_ranef.β) == (0, n_site)
    @test eltype(ϕq_ranef) == Float64
    μ = randn(0,n_site)
    μ2 = add_ranef(ranef, μ, ϕq_ranef, 1:n_site)
    @test μ2 == μ
    #
    ranef = HVI.get_ranef_computer(nre, (:_,), n_site, one(Float32))
    ϕq_ranef = setup_ϕq_ranef(ranef)
    @test ϕq_ranef.β isa AbstractMatrix{Float32}
    μ = randn(Float32, 0,n_site)
    μ2 = add_ranef(ranef, μ, ϕq_ranef, 1:n_site)
    @test μ2 == μ
    #
    β = HVI.sample_ranef(ranef, ϕq_ranef, 4, 3)
    @test eltype(β) == Float32
    @test β == zeros(1,4,3)  # one parameter in construction
end


@testset "RandomEffects" begin
    η = 3.0
    d = LKJCholesky(2, η)
    rL = rand(d) 
    L = rL.L
    corrm = L * L'
    # Array(rL) already gives corrm
    res = randn(40000,2)
    tau = [1.2,2.2]  # sqrt of main diagonal
    cm = diagm(tau) * corrm * diagm(tau)  
    ranef = (diagm(tau) * L * res')
    #ranef = res * Array(L)' * diagm(tau)
    #cov(ranef') # should roughly match cm 
    #cor(ranef')
    #
    θM = CA.ComponentVector(a=1.0, b=2.0, c=3.0)
    par_ranef = (:c, :a) # positions (3,1)
    prior_Σ = CVPrior_LKJ_Cauchy(length(par_ranef), η=η)
    re0 = RandomEffects(par_ranef; η=η)
    @test re0.prior_Σ == prior_Σ
    n_site = 200
    re = HVI.get_ranef_computer(re0, keys(θM), n_site)
    ϕq_ranef = setup_ϕq_ranef(re)
    @test size(ϕq_ranef[Val(:β)]) == (n_site, length(par_ranef))
    U = HVI.transformU_cholesky1(ϕq_ranef[Val(:coef_U)]); U' * U
    l1 = compute_nLranef(re, ϕq_ranef)
    ϕq_ranef2 = CA.ComponentVector(ϕq_ranef; σ = ϕq_ranef.σ .+ eltype(ϕq_ranef)(0.02))
    l2 = compute_nLranef(re, ϕq_ranef2)
    @test l1 < l2
    () -> begin
        # using Zygote
        # @usingany FiniteDiff
        Zygote.gradient(x -> compute_nLranef(re, x), ϕq_ranef2)
    end
    # 
    re32 = HVI.get_ranef_computer(re0, keys(θM), n_site, one(Float32))
    ϕq_ranef = setup_ϕq_ranef(re32)
    ϕq_ranef.β .= randn(Float32, size(ϕq_ranef.β)...)
    @test eltype(ϕq_ranef) == Float32
    l1 = compute_nLranef(re32, ϕq_ranef)
    @test l1 isa Float32
    i_sites = 1:20
    μ = randn(Float32, length(θM), length(i_sites))
    μ_updated = add_ranef(re32, μ, ϕq_ranef, i_sites)
    @test eltype(μ_updated) == Float32
    @test all((μ_updated .- μ)[2,:] .== 0)
    @test all((μ_updated .- μ)[[3,1],:] .≈ ϕq_ranef.β[i_sites,:]')
    #
    # test sampling
    n_site_pred = 5
    n_MC = 4
    res = HVI.sample_ranef(re32, ϕq_ranef, n_site_pred, n_MC)
    @test size(res) == (length(θM), n_site_pred, n_MC)
end

