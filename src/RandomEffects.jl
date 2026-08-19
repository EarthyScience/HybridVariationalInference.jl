abstract type AbstractRandomEffects end
abstract type AbstractRandomEffectsComputer{T} end

abstract type AbstractCovariancePrior{N,T} end 

struct NullRandomEffectsComputer{T} <: AbstractRandomEffectsComputer{T}
    n_par::Int
    n_site::Int
end
compute_nLranef(re::NullRandomEffectsComputer{T}, rm) where T = zero(T)
add_ranef(re::NullRandomEffectsComputer, μ, ϕq_ranef, itrain_sites) = μ
function setup_ϕq_ranef(re::NullRandomEffectsComputer{T}) where T 
    res = CA.ComponentVector(β=reshape( T[], 0, re.n_site))
end
function sample_ranef(re::NullRandomEffectsComputer{T}, ϕq_ranef, n_site, n_sample) where T
    res = fill( zero(T), re.n_par, n_site, n_sample)
end
function get_choleskyΣ_ranef(re::NullRandomEffectsComputer{T}, ϕq_ranef) where T
    zeros(T, 0, 0) # empty subset of parameters that have a randon effect
end
function get_choleskyΣ_par(re::NullRandomEffectsComputer{T}, ϕq_ranef) where T
    zeros(T, re.n_par, re.n_par) 
end

struct NullRandomEffects <: AbstractRandomEffects; end
function get_ranef_computer(
    rn::NullRandomEffects, 
    θM_keys::NTuple{NM,Symbol}, n_site::Integer, float_template::T=1.0,
    ) where {NM,T}
    NullRandomEffectsComputer{T}(length(θM_keys), Int(n_site))
end


"""
 RandomEffects(parameters; γ=0.5), η=5.0, γv = SA.SVector(γ,γ,...)) 

Each site-predicted parameter-sub-vector can have a site-specific random effect
θ_i = θ_i_ml + β_i
The random effects are estimated with the global parameters.
They are assumed to be drawn from Centered Normal distribution
with covariance Σ, whose coefficients of the upper cholesky-factor are also
estimated as global parameters.
In order to encourage low variance and low correlation, a prior of 
zero-centered Cauchy distribution (default scale 0.5) is applied to main diagonal,
and a LKJ prior (default scale 5) is applied to the correlation matrix.
Scale of the Cauchy distribution defines the MAD.
"""
struct RandomEffects{N,T} <: AbstractRandomEffects
    parameters::NTuple{N, Symbol}
    prior_Σ::AbstractCovariancePrior{N,T}
end
function RandomEffects(parameters, float_template::T=1.0; 
    γ::T=T(0.5), η::T=T(5.0), γv = SA.SVector(ntuple(_ -> γ, length(parameters)))) where T
    N = length(parameters)
    prior_Σ = CVPrior_LKJ_Cauchy(γv, η)
    RandomEffects{N,T}(parameters, prior_Σ)
end

"""

"""
struct RandomEffectsComputer{N,T,NM,NNM} <: AbstractRandomEffectsComputer{T} 
    parameters::NTuple{N, Symbol}
    prior_Σ::AbstractCovariancePrior{N,T} # changed type to float_template
    ncomp_U::Int8                         # number of components to describe correlation
    P_col::SA.SMatrix{NM, N, Bool, NNM}   # projection matrix of subset of random to all
    n_site::Int                           # number of sites to setup parameter vector
end
function get_ranef_computer(
    rn::RandomEffects{N}, θM_keys::NTuple{NM,Symbol}, n_site::Integer, float_template::T=1.0
    ) where {N,NM,T}
    prior_Σ = convert_prior(rn.prior_Σ, float_template)
    # γ = convert.(T, rn.γ)
    # η = convert(T,rn.η)
    # γ = convert.(T, rn.γ)
    # η = convert(T,rn.η)
    # prior_Σ=CVPrior_LKJ_Cauchy(γ, η)
    ncomp_U=Int8(sumn(N))
    pos = SA.SVector{N,Int8}(Tuple(findfirst(==(s), θM_keys) for s in rn.parameters))
    P_col = SA.SMatrix{NM,N}(j == k for j in 1:NM, k in pos)
    RandomEffectsComputer(rn.parameters, prior_Σ, ncomp_U, P_col, convert(Int, n_site))
end


function compute_nLranef(re::RandomEffectsComputer{N,T}, ϕq_ranef) where {N,T}
    # get cholesky factor of covariance matrix from optimized parameters
    U = get_choleskyΣ_ranef(re, ϕq_ranef)
    # compute the logdensity of the random effects given the covariance
    β = ϕq_ranef[Val(:β)]
    # βi = first(eachrow(β))
    # TODO compute for entire array (solve required only once)
    logden_rm = sum(eachrow(β)) do βi
        #log_density_mvn_cholesky(UpperTriangular(U), βi) does not work with Zygote
        log_density_mvn_cholesky(U, βi)
    end
    # compute the prior of the estimated covariance matrix
    logden_Σ = logpdf(re.prior_Σ, U)
    cost = -(logden_rm + logden_Σ)
    #@show ϕq_ranef.σ, cost
    cost
end

"""
    get_choleskyΣ_ranef(re::AbstractRandomEffectsComputer, ϕq_ranef::ComponentVector)

Return the upper cholesky factor of Covariance of the Random effects. 
"""
function get_choleskyΣ_ranef(re::RandomEffectsComputer{N,T}, ϕq_ranef) where {N,T}
    coef_Ucorr = ϕq_ranef[Val(:coef_U)] # parameterization of cholesky of correlation 
    σ = max.(T(1e-10), ϕq_ranef[Val(:σ)])  # main diagonal of cholesky factor of covariance
    U = transformU_cholesky1(coef_Ucorr; n=N) * diagm(σ)
end
function get_choleskyΣ_par(re::RandomEffectsComputer{N,T}, ϕq_ranef) where {N,T}
    U = get_choleskyΣ_ranef(re, ϕq_ranef) 
    U_full = re.P_col * U * re.P_col'
    U_full
end

"""
Assume μ of size n_par x n_site
"""
function add_ranef(re::RandomEffectsComputer, μ, ϕq_ranef, itrain_sites)
    ranef = CA.getdata(ϕq_ranef[Val(:β)][itrain_sites,:])
    # moved construction of projection matrix to RandomEffectsComputer
    P_col = re.P_col
    #ranef_full = ranef * P_col'
    ranef_full = P_col * ranef' # par-first format
    μadd = μ .+ ranef_full
    μadd
end

"""
Provide an initial estimate of the correct structure for the parameters to 
estimate for given random effects computer.
"""
function setup_ϕq_ranef(re::RandomEffectsComputer{N,T}) where {N,T}
    coef_Ucorr = uutri2vec(cholesky(I(N).*one(T)).U)
    # U = vec2utri(coef_U; n = N); U' * U
    CA.ComponentVector(
        β = zeros(T, re.n_site, N),
        coef_U = coef_Ucorr,
        σ = fill(T(0.01), N)
    )
end

"""
provide a sample n_par x n_site x n_sample, where
n_par is the dimension of parameters including those columns that have no 
random effect.
"""
function sample_ranef(re::RandomEffectsComputer{N,T}, ϕq_ranef, n_site, n_sample) where {N,T}
    U = get_choleskyΣ_ranef(re, ϕq_ranef)
    rn = randn(T, N, n_site * n_sample)
    r0 = similar(rn)
    mul!(r0, U', rn)
    out = similar(rn, size(re.P_col, 1), n_site * n_sample)
    mul!(out, re.P_col, r0)
    res = reshape(out, size(re.P_col, 1), n_site, n_sample)
end
function sample_ranef_nonmut(re::RandomEffectsComputer{N,T}, ϕq_ranef, n_site, n_sample) where {N,T}
    U = get_choleskyΣ_ranef(re, ϕq_ranef)
    rn =  randn(N, n_site * n_sample )
    r0 = U' * rn
    r = reshape(r0, N, n_site, n_sample)
    #r1 = first(eachslice(r, dims=3))
    # resv = map(eachslice(r, dims=3)) do r1
    #     r1' * re.P_col'
    # end
    # res = stack(resv)
    resr = reshape(re.P_col * r0, size(re.P_col, 1), n_site, n_sample)
    #res2 = permutedims(resr, (2,1,3))
    res, res2
end





""" 
Assigns a Cauchy-prior_Σ with scale γ to the main diagonal of the covariance
and an LKJ prior_Σ with scale η on the correlation matrix.

The default parameterization follows the 
[STAN recommendation](https://mc-stan.org/docs/2_19/stan-users-guide/multivariate-hierarchical-priors-section.html)
assigning a Cauchy scale of 0.5 and an LKJ scale of 5.0
This encourages to decrease variations correlations.
"""
struct CVPrior_LKJ_Cauchy{N,T} <: AbstractCovariancePrior{N,T}
    dCauchy::SA.SVector{N, Cauchy{T}}
    dLKJ::LKJ{T,Int8}
end

function CVPrior_LKJ_Cauchy(γ::SA.SVector{N,T}, η::T) where {N,T}
    dCauchy = Cauchy.(zero(T), γ)::SA.SVector{N, Cauchy{T}}
    dLKJ = LKJ(Int8(N), η)::LKJ{T,Int8}
    CVPrior_LKJ_Cauchy{N, T}(dCauchy, dLKJ)
end

function CVPrior_LKJ_Cauchy(n::Integer, float_template::T=1.0; 
    γ::T=T(0.5), η::T=T(5.0), γv = SA.SVector(ntuple(_ -> γ, n))) where T
    CVPrior_LKJ_Cauchy(γv, η, )
end

convert_prior(prior::CVPrior_LKJ_Cauchy{N,T}, float_template::T) where {N,T} = prior
function convert_prior(prior::CVPrior_LKJ_Cauchy{N,T}, float_template::TN) where {N,T,TN}
    #γ = SA.SVector(Tuple(T(params(d)[2]) for d in prior.dCauchy)) 
    γ = map(d -> TN(params(d)[2]), prior.dCauchy) # already returns SVector
    #η = convert.(TN,params(prior.dLKJ))
    η = TN(params(prior.dLKJ)[2])
    
    CVPrior_LKJ_Cauchy(γ, η)
end

# function rand_prior(d::CVPrior_LKJ_Cauchy{N,T}, n_sample)
# end


function Distributions.logpdf(d::CVPrior_LKJ_Cauchy{N,T}, cm) where {N,T}
    cm_s = cm + I*T(1e-8) # to deal with initial zero covariance
    τ = sqrt.(diag(cm_s))   
    logpdf_cauchy = sum(logpdf_.(d.dCauchy, τ))
    corrm = cm_s ./ (τ * τ') 
    logpdf_lkj = logpdf_(d.dLKJ, corrm)
    return logpdf_cauchy + logpdf_lkj
end

logpdf_(d::Distribution, x) = logpdf(d,x)
function ChainRulesCore.rrule(::typeof(logpdf_), d::Distribution, x)
    # avoid propagating to Distribution d but only to x, otherwise Δy is passed through
    function logpdf_pullback(Δy)
        (NoTangent(), NoTangent(), Δy)
    end
    return logpdf_(d,x), logpdf_pullback
end


