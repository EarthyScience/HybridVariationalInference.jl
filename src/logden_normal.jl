"""
    neg_logden_indep_normal(obs, μ, logσ2s; σfac=1.0) 

Compute the negative Log-density of `obs` for multiple independent normal distributions,
given estimated means `μ` and estimated log of variance parameters `logσ2s`.

All the arguments should be vectors of the same length.
If `obs`,  `μ` are given as a matrix of several column-vectors, their summed
Likelihood is computed, assuming each column having the same `logσ2s`.

Keyword argument `σfac` can be increased to put more weight on achieving
a low uncertainty estimate and means closer to the observations to help
an initial fit. The obtained parameters then can be used as starting values
for a the proper fit with `σfac=1.0`.
"""
function neg_logden_indep_normal(obs::AbstractArray, μ::AbstractArray, logσ2::AbstractArray{ET}; 
    σfac=one(ET)) where ET
    # log of independent Normal distributions 
    # estimate independent uncertainty of each θM, rather than full covariance
    #nlogL = sum(σfac .* log.(σs) .+ 1 / 2 .* abs2.((obs .- μ) ./ σs))
    # problems with division by zero, better constrain log(σs) see Kendall17
    # s = log.(σs)
    # nlogL = sum(σfac .* s .+ 1 / 2 .* abs2.((obs .- μ) ./ exp.(s)))
    # optimize argument logσ2 rather than σs for performance
    #nlogL = sum(σfac .* (1/2) .* logσ2 .+ (1/2) .* exp.(.- logσ2) .* abs2.(obs .- μ))
    # specifying logσ2 instead of σ is not transforming a random variable -> no Jacobian
    #
    # obs_data = CA.getdata(obs)
    # μ_data = CA.getdata(μ)
    # nlogL = NaNMath.sum(  # observations might by NaN for missing
    #     σfac .* logσ2 .+ abs2.(obs_data .- μ_data) .* exp.(.-logσ2)) / convert(eltype(μ),2)
    # return (nlogL)
    #
    i_finobs = .! isnan.(obs)
    obs_data = CA.getdata(obs)[i_finobs]
    μ_data = CA.getdata(μ)[i_finobs]
    logσ2_fin = logσ2[i_finobs]
    nlogL = sum(  # observations might by NaN for missing
        σfac .* logσ2_fin .+ abs2.(obs_data .- μ_data) .* exp.(.-logσ2_fin)) / convert(eltype(μ),2)
    #Main.@infiltrate_main

    return (nlogL)
end

function neg_logden_indep_normal(obs::AbstractGPUArray, μ::AbstractGPUArray, logσ2::AbstractGPUArray{ET}; 
    σfac=one(ET)) where ET
    #cannot use NaNMath.sum on gpu, allocate vectors  of non-NAN
    i_finobs = .! isnan.(obs)
    obs_data = CA.getdata(obs)[i_finobs]
    μ_data = CA.getdata(μ)[i_finobs]
    logσ2_fin = logσ2[i_finobs]
    nlogL = sum(  # observations might by NaN for missing
        σfac .* logσ2_fin .+ abs2.(obs_data .- μ_data) .* exp.(.-logσ2_fin)) / convert(eltype(μ),2)
    return (nlogL)
end

# function neg_logden_indep_normal(obss::AbstractMatrix, preds::AbstractMatrix, logσ2::AbstractVector; kwargs...)
#     nlogLs = map(eachcol(obss), eachcol(preds)) do obs, μ
#         neg_logden_indep_normal(obs, μ, logσ2; kwargs...)
#     end
#     nlogL = sum(nlogLs)
#     return nlogL
# end

# function neg_logden_indep_normal(obss::AbstractMatrix, preds::AbstractMatrix, logσ2s::AbstractMatrix; kwargs...)
#     nlogLs = map(eachcol(obss), eachcol(preds), eachcol(logσ2s)) do obs, μ, logσ2
#         neg_logden_indep_normal(obs, μ, logσ2; kwargs...)
#     end
#     nlogL = sum(nlogLs)
#     return nlogL
# end


entropy_MvNormal(K, logdetΣ) = (K * log(2 * π * ℯ) + logdetΣ) / 2
# compiler figures out log(2 * π * ℯ) already, no need to tinker
#entropy_MvNormal(K, logdetΣ) = (K * (1 + log(2π)) + logdetΣ) / 2
entropy_MvNormal(Σ) = entropy_MvNormal(size(Σ, 1), logdet(Σ))

# struct MvNormalLogDensityProblem{T}
#     y::AbstractVector{T}
#     σfac::T
# end

# function MvNormalLogDensityProblem(y::AbstractVector{T}; σfac=1) where {T}
#     MvNormalLogDensityProblem(y, T(σfac))
# end

# # define a callable that unpacks parameters, and evaluates the log likelihood
# function (problem::MvNormalLogDensityProblem)(θ)
#     μ, logσ2 = θ
#     -sum(problem.σfac .* logσ2 .+  abs2.(problem.y .- μ) .* exp.(.-logσ2)) / 2
# end

