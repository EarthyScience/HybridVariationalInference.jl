using HybridVariationalInference
using Test
using StableRNGs
using Statistics

"""
Compute the correlation between a prection of a linear regression y ~ X, and y
"""
function corX(y,X)
    b = X \ y
    pred = X * b
    cor(pred, y)
end

rng = StableRNG(111)
@testset "compute_correlated_covars" begin
        n_covar_pc = 2
        n_covar = n_covar_pc + 3 
        n_site = 10^n_covar_pc
        rhodec=8
        x_pc = rand(rng, Float32, n_covar_pc, n_site)
        x_o = compute_correlated_covars(rng, x_pc; n_covar, rhodec)    
        # first covariate is a linear combination of underlying
        @test corX(x_o[1, :], x_pc') ≈ 1.0f0
        # the others are decreasingly correlated
        @test corX(x_o[2, :], x_pc') > corX(x_o[n_covar, :], x_pc')
        () -> begin
            #using UnicodePlots
            scatterplot(x_pc[1, :], x_pc[2, :])
            scatterplot(x_o[1, :], x_o[2, :])
            scatterplot(x_o[1, :], x_o[4, :])
        end
end;

@testset "scale_centered_at" begin
    x = randn(5,100) .* 10
    m = collect(1.0:5.0)
    xs = scale_centered_at(x, m, 0.2)
    @test mean(xs; dims=2) ≈ m
    @test std(xs; dims=2) ≈ m .* 0.2
end
