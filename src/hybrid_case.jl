"""
Type to dispatch constructing data and network structures
for different cases of hybrid problem setups

For a specific case, provide functions that specify details
- `get_hybridcase_MLapplicator`
- `get_hybridcase_PBmodel`
- `get_hybridcase_neg_logden_obs`
- `get_hybridcase_par_templates`
- `get_hybridcase_transforms`
- `get_hybridcase_sizes`
- `get_hybridcase_train_dataloader` (default depends on `gen_hybridcase_synthetic`)
optionally
- `gen_hybridcase_synthetic`
- `get_hybridcase_float_type` (defaults to `eltype(θM)`)
- `get_hybridcase_cor_starts` (defaults to include all correlations: `(P=(1,), M=(1,))`)
"""
abstract type AbstractHybridCase end;


"""
    get_hybridcase_MLapplicator([rng::AbstractRNG,] ::AbstractHybridCase, MLEngine; scenario=())

Construct the machine learning model fro given problem case and ML-Framework and 
scenario.

The MLEngine is a value type of a Symbol, usually the name of the module, e.g. 
`const MLengine = Val(nameof(SimpleChains))`.

returns a Tuple of
- AbstractModelApplicator
- initial parameter vector
"""
function get_hybridcase_MLapplicator end    

function get_hybridcase_MLapplicator(case::AbstractHybridCase, MLEngine; scenario=())
    get_hybridcase_MLapplicator(Random.default_rng(), case, MLEngine; scenario)
end

"""
    get_hybridcase_PBmodel(::AbstractHybridCase; scenario::NTuple=())

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
function get_hybridcase_PBmodel end

"""
    get_hybridcase_neg_logden_obs(::AbstractHybridCase; scenario)

Provide a `function(y_obs, ypred) -> Real` that computes the negative logdensity
of the observations, given the predictions.
"""
function get_hybridcase_neg_logden_obs end    


"""
    get_hybridcase_par_templates(::AbstractHybridCase; scenario)

Provide tuple of templates of ComponentVectors `θP` and `θM`.
"""
function get_hybridcase_par_templates end    


"""
    get_hybridcase_transforms(::AbstractHybridCase; scenario)

Return a NamedTupe of
- `transP`: Bijectors.Transform for the global PBM parameters, θP
- `transM`: Bijectors.Transform for the single-site PBM parameters, θM
"""
function get_hybridcase_transforms end

"""
    get_hybridcase_par_templates(::AbstractHybridCase; scenario)

Provide a NamedTuple of number of 
- n_covar: covariates xM
- n_site: all sites in the data
- n_batch: sites in one minibatch during fitting
- n_θM, n_θP: entries in parameter vectors
"""
function get_hybridcase_sizes end

"""
    gen_hybridcase_synthetic(::AbstractHybridCase, rng; scenario)

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
    get_hybridcase_float_type(::AbstractHybridCase; scenario)

Determine the FloatType for given Case and scenario, defaults to Float32
"""
function get_hybridcase_float_type(case::AbstractHybridCase; scenario=())
    return eltype(get_hybridcase_par_templates(case; scenario).θM)
end

"""
    get_hybridcase_train_dataloader(::AbstractHybridCase, rng; scenario)

Return a DataLoader that provides a tuple of
- `xM`: matrix of covariates, with one column per site
- `xP`: Iterator of process-model drivers, with one element per site
- `y_o`: matrix of observations with added noise, with one column per site
- `y_unc`: matrix `sizeof(y_o)` of uncertainty information 
"""
function get_hybridcase_train_dataloader(case::AbstractHybridCase, rng::AbstractRNG; 
    scenario = ())
    (; xM, xP, y_o, y_unc) = gen_hybridcase_synthetic(case, rng; scenario)
    (; n_batch) = get_hybridcase_sizes(case; scenario)
    xM_gpu = :use_flux ∈ scenario ? CuArray(xM) : xM
    train_loader = MLUtils.DataLoader((xM_gpu, xP, y_o, y_unc), batchsize = n_batch)
    return(train_loader)
end

"""
    get_hybridcase_cor_starts(case::AbstractHybridCase; scenario)

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
function get_hybridcase_cor_starts(case::AbstractHybridCase; scenario = ())
    (P=(1,), M=(1,))
end



