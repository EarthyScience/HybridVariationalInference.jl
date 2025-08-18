"""
Implements [`AbstractHybridProblem`](@ref) by gathering all the parts into 
one struct.

Fields:
- `θP::ComponentVector`, `θM::ComponentVector`: parameter templates
- `g::AbstractModelApplicator`, `ϕg::AbstractVector`: ML model and its parameters 
- `ϕunc::ComponentVector`: parameters for the Covariance matrix of the approximate posterior
- `f_batch`: Process-based model predicing for `n_batch` sites
- `f_allsites`: Process-based model predicing for `n_site` sites
- `priors`: AbstractDict: Prior distributions for all PBM parameters on constrained scale
- `py`: Likelihood function
- `transM::Stacked`, `transP::Stacked`: bijectors transforming from unconstrained to 
  constrained scale for site-specific and global parameters respectively.
- `train_dataloader::MLUtils.DataLoader`: providingn Tuple of matrices 
  `(xM, xP, y_o, y_unc, i_sites)`: covariates, model drivers, observations, 
  observation uncertainties and index of provided sites.
- `n_covar::Int`, `n_site::Int`, `n_batch::Int`: number covariates,
  number of sites, and number of sites within one batch
- `cor_ends::NamedTuple`: block structure in correlations, 
  defaults to  `(P = [length(θP)], M = [length(θM)])`
- `pbm_covars::NTuple{N,Symbol}`: names of global parameters used as covariates 
  in the ML model, defaults to `()`, i.e. no covariates fed into the ML model

"""
struct HybridProblem <: AbstractHybridProblem
    θP::CA.ComponentVector
    θM::CA.ComponentVector
    f_batch::Any
    f_allsites::Any
    g::AbstractModelApplicator
    ϕg::Any # depends on framework
    ϕunc::CA.ComponentVector
    priors::AbstractDict
    py::Any # any callable
    transM::Stacked
    transP::Stacked
    cor_ends::@NamedTuple{P::Vector{Int}, M::Vector{Int}} # = (P=(1,),M=(1,))
    train_dataloader::MLUtils.DataLoader
    n_covar::Int
    n_site::Int
    n_batch::Int
    pbm_covars::NTuple{_N, Symbol} where _N
    #inner constructor to constrain the types
    function HybridProblem(
            θP::CA.ComponentVector, θM::CA.ComponentVector,
            g::AbstractModelApplicator, ϕg::AbstractVector,
            f_batch, 
            f_allsites,
            priors::AbstractDict,
            py,
            transM::Stacked,
            transP::Stacked,
            # return a function that constructs the trainloader based on n_batch
            train_dataloader::MLUtils.DataLoader,
            n_covar::Int,
            n_site::Int,
            n_batch::Int,
            cor_ends::NamedTuple = (P = [length(θP)], M = [length(θM)]),
            pbm_covars::NTuple{N,Symbol} = (),
            ϕunc::CA.ComponentVector = init_hybrid_ϕunc(cor_ends, zero(eltype(θM))),
    ) where N
        new(
            θP, θM, f_batch, f_allsites, g, ϕg, ϕunc, priors, py, transM, transP, cor_ends, 
            train_dataloader, n_covar, n_site, n_batch, pbm_covars)
    end
end

# function HybridProblem(θP::CA.ComponentVector, θM::CA.ComponentVector,
#         # note no ϕg argument and g_chain unconstrained
#         g_chain, f_batch,
#         args...; rng = Random.default_rng(), kwargs...)
#     # dispatches on type of g_chain
#     g, ϕg = construct_ChainsApplicator(rng, g_chain, eltype(θM))
#     HybridProblem(θP, θM, g, ϕg, f_batch, args...; kwargs...)
# end

"""
    HybridProblem(prob::AbstractHybridProblem; scenario = ()

Gather all information from another `AbstractHybridProblem` wiht possible
updating of some of the entries.
"""
function HybridProblem(prob::AbstractHybridProblem; scenario = (),
    θP = get_hybridproblem_par_templates(prob; scenario).θP,
    θM = get_hybridproblem_par_templates(prob; scenario).θM,
    g = get_hybridproblem_MLapplicator(prob; scenario)[1],
    ϕg = get_hybridproblem_MLapplicator(prob; scenario)[2],
    f_batch = get_hybridproblem_PBmodel(prob; scenario, use_all_sites = false),
    f_allsites = get_hybridproblem_PBmodel(prob; scenario, use_all_sites = true),
    priors = get_hybridproblem_priors(prob; scenario),
    py = get_hybridproblem_neg_logden_obs(prob; scenario),
    transP = get_hybridproblem_transforms(prob; scenario).transP,
    transM = get_hybridproblem_transforms(prob; scenario).transM,
    train_dataloader = get_hybridproblem_train_dataloader(prob; scenario),
    n_covar = get_hybridproblem_n_covar(prob; scenario),
    n_site = get_hybridproblem_n_site_and_batch(prob; scenario)[1],
    n_batch = get_hybridproblem_n_site_and_batch(prob; scenario)[2],
    cor_ends = get_hybridproblem_cor_ends(prob; scenario),
    pbm_covars = get_hybridproblem_pbmpar_covars(prob; scenario),
    ϕunc = get_hybridproblem_ϕunc(prob; scenario),
    )
    HybridProblem(θP, θM, g, ϕg, f_batch, f_allsites, priors, py, transM, transP, train_dataloader,
        n_covar, n_site, n_batch, cor_ends, pbm_covars, ϕunc)
end

# """
#     update(prob::HybridProblem; ...)

# Create a copy of prob, with some parts replaced.
# """
# function update(prob::HybridProblem;
#         θP::CA.ComponentVector = prob.θP,
#         θM::CA.ComponentVector = prob.θM,
#         g::AbstractModelApplicator = prob.g, 
#         ϕg::AbstractVector = prob.ϕg,
#         ϕunc::CA.ComponentVector = prob.ϕunc,
#         f_batch = prob.f_batch,
#         f_allsites = prob.f_allsites,
#         priors::AbstractDict = prob.priors,
#         py = prob.py,
#         # transM::Union{Function, Bijectors.Transform} = prob.transM,
#         # transP::Union{Function, Bijectors.Transform} = prob.transP,
#         transM = prob.transM,
#         transP = prob.transP,
#         cor_ends::NamedTuple = prob.cor_ends,
#         pbm_covars::NTuple{N,Symbol} = prob.pbm_covars,
#         train_dataloader::MLUtils.DataLoader = prob.train_dataloader,
#         n_covar::Integer = prob.n_covar,
#         n_site::Integer = prob.n_site,
#         n_batch::Integer = prob.n_batch,
# ) where N
#     HybridProblem(θP, θM, g, ϕg, f_batch, f_allsites, priors, py, transM, transP, 
#         train_dataloader, n_covar, n_site, n_batch, cor_ends, pbm_covars, ϕunc)
# end

function get_hybridproblem_par_templates(prob::HybridProblem; scenario = ())
    (; θP = prob.θP, θM = prob.θM)
end

function get_hybridproblem_ϕunc(prob::HybridProblem; scenario = ())
    prob.ϕunc
end

function get_hybridproblem_neg_logden_obs(prob::HybridProblem; scenario = ())
    prob.py
end

function get_hybridproblem_transforms(prob::HybridProblem; scenario = ())
    (; transP = prob.transP, transM = prob.transM)
end

# function get_hybridproblem_sizes(prob::HybridProblem; scenario = ())
#     n_θM = length(prob.θM)
#     n_θP = length(prob.θP)
#     (; n_covar=prob.n_covar, n_batch=prob.n_batch, n_θM, n_θP)
# end

function get_hybridproblem_PBmodel(prob::HybridProblem; scenario = (), use_all_sites=false)
    use_all_sites ? prob.f_allsites : prob.f_batch
end

function get_hybridproblem_MLapplicator(prob::HybridProblem; scenario = ())
    prob.g, prob.ϕg
end

function get_hybridproblem_train_dataloader(prob::HybridProblem; scenario = ())
    prob.train_dataloader
end

function get_hybridproblem_cor_ends(prob::HybridProblem; scenario = ())
    prob.cor_ends
end
function get_hybridproblem_pbmpar_covars(prob::HybridProblem; scenario = ()) 
    prob.pbm_covars
end
function get_hybridproblem_n_covar(prob::HybridProblem; scenario = ())
    prob.n_covar
end
function get_hybridproblem_n_site_and_batch(prob::HybridProblem; scenario = ())
    prob.n_site, prob.n_batch
end

function get_hybridproblem_priors(prob::HybridProblem; scenario = ())
    prob.priors
end

# function get_hybridproblem_float_type(prob::HybridProblem; scenario = ()) 
#     eltype(prob.θM)
# end

"""
Get the inverse-transformation of lower and upper quantiles of a Vector of Distributions.

This can be used to get proper confidence intervals at unconstrained (log) ζ-scale
for priors on normal θ-scale for constructing a NormalScalingModelApplicator.
"""
function get_quantile_transformed(priors::AbstractVector{<:Distribution}, trans; 
    q95 = (0.05, 0.95))
    θq = ([quantile(d, q) for d in priors] for q in q95)
    lowers, uppers = inverse(trans).(θq)
end


