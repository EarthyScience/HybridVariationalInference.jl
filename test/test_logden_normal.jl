using Test
using Distributions
using Test
using HybridVariationalInference
using HybridVariationalInference: HybridVariationalInference as CP
using LinearAlgebra


@testset "neg_logden_indep_normal vector" begin
    loglik_norm(y, μ, σ) = -1 / 2 .* (Distributions.log2π .+ 2 .* log.(σ) .+ abs2.(y .- μ) ./ abs2.(σ))
    loglik_norm_l(y, μ, logσ) = -1 / 2 .* (Distributions.log2π .+ 2 .* logσ .+ abs2.(y .- μ) ./ abs2.(exp.(logσ)))
    logden_norm_l(y, μ, logσ) = -1 / 2 .* (2 .* logσ .+ abs2.(y .- μ) ./ abs2.(exp.(logσ)))
    neg_logden_norm_l2(y, μ, logσ2) = (logσ2 .+ abs2.(y .- μ) .* exp.(.-logσ2)) ./ 2

    # first test that neg_logden_norm_l2 returns values of logpdf(Normal) up to an additive 
    μ = [1.0, 1.0]
    σ = [1.1, 2.0]
    logσ2 = log.(abs2.(σ))
    y = [1.2, 1.1]
    logpdf_true = logpdf.(Normal.(μ, σ), y)
    dlogpdf = logpdf_true .- logpdf_true[1]
    #loglik_norm(y, μ, σ)
    #loglik_norm_l(y, μ, log.(σ))
    #tmp = logden_norm_l(y, μ, log.(σ))
    #tmp .- tmp[1]
    tmp = neg_logden_norm_l2(y, μ, logσ2)
    @test isapprox(tmp .- tmp[1], -dlogpdf)

    # next test that the sum of neg_logden_norm_l2 corresponds to tested
    res_vec = neg_logden_indep_normal(y, μ, logσ2)
    @test res_vec ≈ sum(tmp)

    # test several columns
    μ3 = hcat(μ,μ,μ)
    y3 = hcat(y,y,y)
    resm = neg_logden_indep_normal(y3, μ3, logσ2)
    @test resm == [res_vec, res_vec, res_vec]
end;

@testset "entropy_MvNormal" begin
    S = Diagonal([4,5]) .+ rand(2,2)
    S2 = Symmetric(S*S)
    @test entropy_MvNormal(S2) ≈ entropy(MvNormal(S2))
end;





