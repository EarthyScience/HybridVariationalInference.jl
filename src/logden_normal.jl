"""
    neg_logden_indep_normal(obs, μ, logσ2s; σfac=1.0) 

Compute the negative Log-density of `obs` for multiple independent normal distributions,
given estimated means `μ` and estimated log of variance parameters `logσ2s`.

All the arguments should be vectors of the same length.
If `obs`,  `μ` are given as a matrix of several column-vectors, 
the likelihood of each column is returned, assuming each column having the same `logσ2s`.
depr: If `obs`,  `μ` are given as a matrix of several column-vectors, their summed
Likelihood is computed, assuming each column having the same `logσ2s`.

Keyword argument `σfac` can be increased to put more weight on achieving
a low uncertainty estimate and means closer to the observations to help
an initial fit. The obtained parameters then can be used as starting values
for a the proper fit with `σfac=1.0`.
"""
function neg_logden_indep_normal(obs::AbstractArray{TO,DO}, μ::AbstractArray{TP,DP}, logσ2::AbstractArray{TO}; 
    i_finobs = .! isnan.(obs),
    σfac=one(TO)) where {TO, TP, DO, DP}
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
    # obs_data = CA.getdata(obs)[i_finobs]
    # μ_data = CA.getdata(μ)[i_finobs]
    # logσ2_fin = logσ2[i_finobs]
    # # nlogL = sum(  # observations might by NaN for missing
    # #     σfac .* logσ2_fin .+ abs2.(obs_data .- μ_data) .* exp.(.-logσ2_fin)) / convert(eltype(μ),2)
    # nlogL = 
    #     (σfac .* logσ2_fin .+ abs2.(obs_data .- μ_data) .* exp.(.-logσ2_fin)) ./ eltype(μ)(2)
    nlogL = 
        i_finobs .* (σfac .* logσ2 .+ abs2.(obs .- μ) .* exp.(.-logσ2)) ./ eltype(μ)(2)
    nlogL_sites = if DO == 1
        sum(nlogL) # provided a single-site vector
    elseif DO == 2    
        vec(sum(nlogL;dims=1))
    else 
        error("need to provide a matrix with observations with one column per site, " *
            "but got array of dimension $D")
    end
    #nlogL_sites = colsum_finite_obs(nlogL, obs)
    # if !all(isfinite.(nlogL_sites))
    #     @show nlogL_sites, μ
    #     @warn("encountered non-finite loglikelihood in neg_logden_indep_normal")
    #     Main.@infiltrate_main
    # end
    return (nlogL_sites)
    # function fcol(logσ2, obs, μ) 
    #     nlogL = sum(isfinite(obs[i]) ? X[i] : zero(eltype(X))  # observations might by NaN for missing
    #         σfac .* logσ2 .+ abs2.(obs .- μ) .* exp.(.-logσ2)) / convert(eltype(μ),2)
    # end
    # map(fcol, eachdol(obs), eachcol(μ), eachcol(logσ2)) 


end

# new signature with returning just one sum value rather than each column
#   and taking an optional 4rth argument of the same size as obs that can be used as
#   a temporary buffer
function neg_logden_indep_normal_sum(obs::AbstractArray{TO,DO}, 
    μ::AbstractArray, logσ2::AbstractArray, 
    obs_buffer::AbstractArray{TO,DO} = similar(obs); 
    #i_finobs = .! isnan.(obs), # TODO provide with y_unc to save allocations
    σfac=one(TO)) where {TO, DO}
    obs_buffer .= (σfac .* logσ2 .+ abs2.(obs .- μ) .* exp.(.-logσ2)) ./ eltype(μ)(2)
    sum(obs_buffer)        
end

() -> begin
    pred = (σfac .* logσ2 .+ abs2.(obs .- μ) .* exp.(.-logσ2)) ./ eltype(μ)(2)
    function alloc_tmp(σfac, logσ2, obs, μ, pred = similar(obs)) 
        @allocated loop_tmpf(10_000, σfac, logσ2, obs, μ, pred) 
    end
    function loop_tmpf(n, σfac, logσ2, obs, μ, pred = similar(obs)) 
        for _ in 1:n
            #pred .= obs .* one(eltype(obs))
            pred .= (σfac .* logσ2 .+ abs2.(obs .- μ) .* exp.(.-logσ2)) ./ eltype(μ)(2)
            sum(pred)
        end
    end
    alloc_tmp(σfac, logσ2, obs, μ) 
    alloc_tmp(σfac, logσ2, obs, μ, obs) 
    #@profview_allocs alloc_tmp(σfac, logσ2, obs, μ) 
end


function neg_logden_indep_normal(obs::AbstractGPUArray, μ::AbstractGPUArray, logσ2::AbstractGPUArray{ET}; 
    σfac=one(ET)) where ET
    error("update neg_logden_indep_normal for GPU to each solsum rather than total sum")
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


entropy_MvNormal(K::Integer, logdetΣ::T) where T = (T(K) * log(T(2) * T(π) * T(ℯ)) + logdetΣ) / T(2)
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

