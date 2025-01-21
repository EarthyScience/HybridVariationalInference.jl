"""
Type to dispatch constructing data and network structures
for different cases of hybrid problem setups

For a specific case, provide functions that specify details
- get_hybridcase_par_templates
- get_hybridcase_transforms
- get_hybridcase_sizes
- get_hybridcase_MLapplicator
- get_hybridcase_PBmodel
optionally
- gen_hybridcase_synthetic
- get_hybridcase_FloatType (defaults to eltype(θM))
"""
abstract type AbstractHybridCase end;

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
    get_hybridcase_MLapplicator(::AbstractHybridCase, MLEngine, n_covar, n_out; scenario=())

Construct the machine learning model fro given problem case and ML-Framework and 
scenario.

The MLEngine is a value type of a Symbol, usually the name of the module, e.g. 
`const MLengine = Val(nameof(SimpleChains))`.

returns a Tuple of
- AbstractModelApplicator
- initial parameter vector
"""
function get_hybridcase_MLapplicator end    

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
    get_hybridcase_FloatType(::AbstractHybridCase; scenario)

Determine the FloatType for given Case and scenario, defaults to Float32
"""
function get_hybridcase_FloatType(case::AbstractHybridCase; scenario)
    return eltype(get_hybridcase_par_templates(case; scenario).θM)
end


