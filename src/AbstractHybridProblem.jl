"""
Type to dispatch constructing data and network structures
for different cases of hybrid problem setups.

For a specific prob, provide functions that specify details
- `get_hybridproblem_MLapplicator`
- `get_hybridproblem_transforms`
- `get_hybridproblem_PBmodel`
- `get_hybridproblem_neg_logden_obs`
- `get_hybridproblem_par_templates`
- `get_hybridproblem_ϕunc`
- `get_hybridproblem_train_dataloader` (may use `construct_dataloader_from_synthetic`)
- `get_hybridproblem_priors` 
- `get_hybridproblem_n_covar` 
- `get_hybridproblem_n_site` 
optionally
- `gen_hybridproblem_synthetic`
- `get_hybridproblem_float_type` (defaults to `eltype(θM)`)
- `get_hybridproblem_cor_ends` (defaults to include all correlations: 
  `(P = [length(θP)], M = [length(θM)])`)
- `get_hybridproblem_pbmpar_covars` (defaults to empty tuple)


The initial value of parameters to estimate is spread
- `ϕg`: parameter of the MLapplicator: returned by `get_hybridproblem_MLapplicator`
- `ζP`: mean of the PBmodel parameters: returned by `get_hybridproblem_par_templates`
- `ϕunc`: additional parameters of the approximte posterior: returned by `get_hybridproblem_ϕunc`
"""
abstract type AbstractHybridProblem end;

"""
    get_hybridproblem_MLapplicator([rng::AbstractRNG,] ::AbstractHybridProblem; scenario=())

Construct the machine learning model fro given problem prob and ML-Framework and 
scenario.

returns a Tuple of
- AbstractModelApplicator
- initial parameter vector
"""
function get_hybridproblem_MLapplicator end

function get_hybridproblem_MLapplicator(prob::AbstractHybridProblem; scenario = ())
    get_hybridproblem_MLapplicator(Random.default_rng(), prob; scenario)
end

"""
    get_hybridproblem_PBmodel(::AbstractHybridProblem; scenario::NTuple=())

Construct the process-based model function 
`f(θP::AbstractVector, θMs::AbstractMatrix, x) -> (AbstractVector, AbstractMatrix)`
with
- θP: calibrated parameters that are constant across site
- θMs: calibrated parameters that vary across sites, with a  column for each site
- x: drivers, indexed by site

returns a tuple of predictions with components
- first, those that are constant across sites
- second, those that vary across sites, with a column for each site
"""
function get_hybridproblem_PBmodel end

"""
    get_hybridproblem_neg_logden_obs(::AbstractHybridProblem; scenario)

Provide a `function(y_obs, ypred) -> Real` that computes the negative logdensity
of the observations, given the predictions.
"""
function get_hybridproblem_neg_logden_obs end

"""
    get_hybridproblem_par_templates(::AbstractHybridProblem; scenario)

Provide tuple of templates of ComponentVectors `θP` and `θM`.
"""
function get_hybridproblem_par_templates end

"""
    get_hybridproblem_ϕunc(::AbstractHybridProblem; scenario)

Provide a ComponentArray of the initial additional parameters of the approximate posterior.
Defaults to zero correlation and log_σ2 of 1e-10.
"""
function get_hybridproblem_ϕunc(prob::AbstractHybridProblem; scenario)
    FT = get_hybridproblem_float_type(prob; scenario) 
    cor_ends = get_hybridproblem_cor_ends(prob; scenario)
    init_hybrid_ϕunc(cor_ends, zero(FT))    
end

"""
    get_hybridproblem_transforms(::AbstractHybridProblem; scenario)

Return a NamedTupe of
- `transP`: Bijectors.Transform for the global PBM parameters, θP
- `transM`: Bijectors.Transform for the single-site PBM parameters, θM
"""
function get_hybridproblem_transforms end

# """
#     get_hybridproblem_par_templates(::AbstractHybridProblem; scenario)
# Provide a NamedTuple of number of 
# - n_covar: covariates xM
# - n_site: all sites in the data
# - n_batch: sites in one minibatch during fitting
# - n_θM, n_θP: entries in parameter vectors
# """
# function get_hybridproblem_sizes end

"""
    get_hybridproblem_n_covar(::AbstractHybridProblem; scenario)

Provide the number of covariates. 
"""
function get_hybridproblem_n_covar(::AbstractHybridProblem; scenario) end
# function get_hybridproblem_n_covar(prob::AbstractHybridProblem; scenario)
#     train_loader = get_hybridproblem_train_dataloader(Random.default_rng(), prob; scenario)
#     (xM, xP, y_o, y_unc) = first(train_loader)
#     n_covar = size(xM, 1)
#     return (n_covar)
# end


function get_hybridproblem_pbmpar_covars(::AbstractHybridProblem; scenario) 
    ()
end

"""
    get_hybridproblem_n_site(::AbstractHybridProblem; scenario)

Provide the number of sites. 
"""
function get_hybridproblem_n_site end


"""
    gen_hybridproblem_synthetic([rng,] ::AbstractHybridProblem; scenario)

Setup synthetic data, a NamedTuple of
- xM: matrix of covariates, with one column per site
- θP_true: vector global process-model parameters
- θMs_true: matrix of site-varying process-model parameters, with 
- xP: Vector of process-model drivers, with an entry per site
- y_global_true: vector of global observations
- y_true: matrix of site-specific observations with one column per site
- y_global_o, y_o: observations with added noise
"""
function gen_hybridproblem_synthetic end

"""
    get_hybridproblem_float_type(::AbstractHybridProblem; scenario)

Determine the FloatType for given Case and scenario, defaults to Float32
"""
function get_hybridproblem_float_type(prob::AbstractHybridProblem; scenario = ())
    return eltype(get_hybridproblem_par_templates(prob; scenario).θM)
end

"""
    get_hybridproblem_train_dataloader(::AbstractHybridProblem; scenario, n_batch)

Return a DataLoader that provides a tuple of
- `xM`: matrix of covariates, with one column per site
- `xP`: Iterator of process-model drivers, with one element per site
- `y_o`: matrix of observations with added noise, with one column per site
- `y_unc`: matrix `sizeof(y_o)` of uncertainty information 
- `i_sites`: Vector of indices of sites in toal sitevector for the minibatch
"""
function get_hybridproblem_train_dataloader end

"""
    construct_dataloader_from_synthetic(rng::AbstractRNG, prob::AbstractHybridProblem;
        scenario = (), n_batch)

Construct a dataloader based on `gen_hybridproblem_synthetic`. 
gdev is applied to xM.
If :f_on_gpu is in scenario tuple, gdev is also applied to `xP`, `y_o`, and `y_unc`,
to put the entire data to gpu.
Alternatively, gdev could be applied to the dataloader, then for each
iteration the subset of data is separatly transferred to gpu.
"""
function construct_dataloader_from_synthetic(rng::AbstractRNG, prob::AbstractHybridProblem;
        scenario = (), n_batch, 
        gdev = :use_gpu ∈ scenario ? gpu_device() : identity,
        )
    (; xM, xP, y_o, y_unc) = gen_hybridproblem_synthetic(rng, prob; scenario)
    n_site = size(xM,2)
    @assert length(xP) == n_site
    @assert size(y_o,2) == n_site
    @assert size(y_unc,2) == n_site
    i_sites = 1:n_site
    xM_dev = gdev(xM)
    xP_dev, y_o_dev, y_unc_dev = :f_on_gpu ∈ scenario ? 
        (gdev(xP), gdev(y_o), gdev(y_unc)) : (xP, y_o, y_unc)
    train_loader = MLUtils.DataLoader((xM_dev, xP_dev, y_o_dev, y_unc_dev, i_sites);
        batchsize = n_batch, partial = false)
    return (train_loader)
end

# function get_hybridproblem_train_dataloader(prob::AbstractHybridProblem; scenario = ())
#     rng::AbstractRNG = Random.default_rng()
#     get_hybridproblem_train_dataloader(rng, prob; scenario)
# end

"""
    get_hybridproblem_priors(::AbstractHybridProblem; scenario)

Return a dictionary of marginal prior distributions for components in `θP` and `θM`.
Defaults for each component `θ` to `Normal(θ, max(θ, 1.0))`.
"""
function get_hybridproblem_priors(prob::AbstractHybridProblem; scenario = ())
    pt = get_hybridproblem_par_templates(prob; scenario)
    θ = vcat(pt.θP, pt.θM)   
    Dict(keys(θ) .=> Normal.(CA.getdata(θ), max.(CA.getdata(θ), one(eltype(θ))))) 
end

"""
    get_hybridproblem_cor_ends(prob::AbstractHybridProblem; scenario)

Specify blocks in correlation matrices among parameters.
Returns a NamedTuple.
- `P`: correlations among global parameters
- `M`: correlations among ML-predicted parameters

Subsets ofparameters that are correlated with other but not correlated with
parameters of other subranges are specified by indicating the starting position
of each subrange.
E.g. if within global parameter vector `(p1, p2, p3)`, `p1` and `p2` are correlated, 
but parameter `p3` is not correlated with them,
then the first subrange starts at position 1 and the second subrange starts at position 3.
If there is only single block of all ML-predicted parameters being correlated 
with each other then this block starts at position 1: `(P=(1,3), M=(1,))`.
"""
function get_hybridproblem_cor_ends(prob::AbstractHybridProblem; scenario = ())
    pt = get_hybridproblem_par_templates(prob; scenario)
    (P = [length(pt.θP)], M = [length(pt.θM)])
end


function setup_PBMpar_interpreter(θP, θM, θall = vcat(θP, θM))
    keys_fixed = ((k for k in keys(θall) if (k ∉ keys(θP)) & (k ∉ keys(θM)))...,)
    θFix = θall[keys_fixed]
    intθ = ComponentArrayInterpreter(flatten1(CA.ComponentVector(; θP, θM, θFix)))
    intθ, θFix
end