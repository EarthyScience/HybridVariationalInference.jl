"""
Type to dispatch constructing data and network structures
for different cases of hybrid problem setups

For a specific prob, provide functions that specify details
- `get_hybridproblem_MLapplicator`
- `get_hybridproblem_PBmodel`
- `get_hybridproblem_neg_logden_obs`
- `get_hybridproblem_par_templates`
- `get_hybridproblem_transforms`
- `get_hybridproblem_train_dataloader` (default depends on `gen_hybridcase_synthetic`)
optionally
- `gen_hybridcase_synthetic`
- `get_hybridproblem_n_covar` (defaults to number of rows in xM in train_dataloader )
- `get_hybridproblem_float_type` (defaults to `eltype(θM)`)
- `get_hybridproblem_cor_starts` (defaults to include all correlations: `(P=(1,), M=(1,))`)
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

function get_hybridproblem_MLapplicator(prob::AbstractHybridProblem; scenario=())
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

Provide the number of covariates. Default returns the number of rows in `xM` from
`get_hybridproblem_train_dataloader`.
"""
function get_hybridproblem_n_covar(prob::AbstractHybridProblem; scenario)
    train_loader = get_hybridproblem_train_dataloader(Random.default_rng(), prob; scenario)
    (xM, xP, y_o, y_unc) = first(train_loader)
    n_covar = size(xM, 1)
    return(n_covar)
end

"""
    gen_hybridcase_synthetic([rng,] ::AbstractHybridProblem; scenario)

Setup synthetic data, a NamedTuple of
- xM: matrix of covariates, with one column per site
- θP_true: vector global process-model parameters
- θMs_true: matrix of site-varying process-model parameters, with 
- xP: Vector of process-model drivers, with an entry per site
- y_global_true: vector of global observations
- y_true: matrix of site-specific observations with one column per site
- y_global_o, y_o: observations with added noise
"""
function gen_hybridcase_synthetic end

"""
    get_hybridproblem_float_type(::AbstractHybridProblem; scenario)

Determine the FloatType for given Case and scenario, defaults to Float32
"""
function get_hybridproblem_float_type(prob::AbstractHybridProblem; scenario=())
    return eltype(get_hybridproblem_par_templates(prob; scenario).θM)
end

"""
    get_hybridproblem_train_dataloader([rng,] ::AbstractHybridProblem; scenario, n_batch)

Return a DataLoader that provides a tuple of
- `xM`: matrix of covariates, with one column per site
- `xP`: Iterator of process-model drivers, with one element per site
- `y_o`: matrix of observations with added noise, with one column per site
- `y_unc`: matrix `sizeof(y_o)` of uncertainty information 
"""
function get_hybridproblem_train_dataloader(rng::AbstractRNG, prob::AbstractHybridProblem; 
    scenario = (), n_batch = 10)
    (; xM, xP, y_o, y_unc) = gen_hybridcase_synthetic(rng, prob; scenario)
    xM_gpu = :use_Flux ∈ scenario ? CuArray(xM) : xM
    train_loader = MLUtils.DataLoader((xM_gpu, xP, y_o, y_unc), batchsize = n_batch)
    return(train_loader)
end

function get_hybridproblem_train_dataloader(prob::AbstractHybridProblem; scenario = ())
    rng::AbstractRNG = Random.default_rng()
    get_hybridproblem_train_dataloader(rng, prob; scenario)
end


"""
    get_hybridproblem_cor_starts(prob::AbstractHybridProblem; scenario)

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
function get_hybridproblem_cor_starts(prob::AbstractHybridProblem; scenario = ())
    (P=(1,), M=(1,))
end



