"""
    AbstractModelApplicator(x, ϕ)

Abstraction of applying a machine learning model at covariate matrix, `x`,
using parameters, `ϕ`. It returns a matrix of predictions with the same
number of rows as in `x`.    

Constructors for specifics are defined in extension packages.
Each constructor takes a special type of machine learning model and returns 
a tuple with two components:
- The applicator 
- a sample parameter vector (type  depends on the used ML-framework)

Implemented are
- `construct_SimpleChainsApplicator`
- `construct_FluxApplicator`
- `construct_LuxApplicator`
"""
abstract type AbstractModelApplicator end

function apply_model end

(app::AbstractModelApplicator)(x, ϕ) = apply_model(app, x, ϕ)

"""
    construct_ChainsApplicator([rng::AbstractRNG,] chain, float_type)
"""
function construct_ChainsApplicator end

function construct_ChainsApplicator(chain, float_type::DataType; kwargs...)
    construct_ChainsApplicator(Random.default_rng(), chain, float_type; kwargs...)
end

# function construct_SimpleChainsApplicator end
# function construct_FluxApplicator end
# function construct_LuxApplicator end


"""
    construct_3layer_MLApplicator(
        rng::AbstractRNG, case::HVI.AbstractHybridCase, <ml_engine>;
        scenario::NTuple = ())

`ml_engine` usually is of type `Val{Symbol}`, e.g. Val(:Flux). See `select_ml_engine`.       
"""
function construct_3layer_MLApplicator end

"""
    select_ml_engine(;scenario)

Returns a value type `Val{:Symbol}` to dispatch on the machine learning engine to use.
- defaults to `Val(:SimpleChains)`
- `:use_Lux ∈ scenario -> Val(:Lux)`
- `:use_Flux ∈ scenario -> Val(:Flux)`
"""
function select_ml_engine(;scenario)
    if :use_Lux ∈ scenario
        return Val(:Lux)
    elseif :use_Flux ∈ scenario
        return Val(:Flux)
    else
        # default
        return Val(:SimpleChains)
    end
end

