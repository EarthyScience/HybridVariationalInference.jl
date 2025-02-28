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
    NullModelApplicator()

Model applicator that returns its inputs. Used for testing.
"""
struct NullModelApplicator <: AbstractModelApplicator end

function apply_model(app::NullModelApplicator, x, ϕ)
    return x
end


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
        rng::AbstractRNG, prob::HVI.AbstractHybridProblem, <ml_engine>;
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

"""
    MagnitudeModelApplicator(app, y0)

Wrapper around AbstractModelApplicator that multiplies the prediction
by the absolute inverse of an initial estimate of the prediction.

This helps to keep raw predictions and weights in a similar magnitude.
"""
struct MagnitudeModelApplicator{M,A} <: AbstractModelApplicator
    app::A
    multiplier::M
end

function apply_model(app::MagnitudeModelApplicator, x, ϕ)
    #@show size(x), size(ϕ), app.multiplier
    @assert eltype(app.multiplier) == eltype(ϕ)
    apply_model(app.app, x, ϕ) .* app.multiplier
end


"""
    NormalScalingModelApplicator(app, μ, σ)
    NormalScalingModelApplicator(app, priors, transM)

Wrapper around AbstractModelApplicator that transforms each output 
(assumed in [0..1], usch as output of logistic activation function)
to a quantile of a Normal distribution. 

Length of μ, σ must correspond to the number of outputs of the wrapped ModelApplicator.

This helps to keep raw ML-predictions (in confidence bounds) and weights in a similar magnitude.
Compared to specifying bounds, this allows for the possibility (although harder to converge)
far beyond the confidence bounds.

The second constructor fits a normal distribution of the inverse-transformed 5% and 95%
quantiles of prior distributions.
"""
struct NormalScalingModelApplicator{VF,A} <: AbstractModelApplicator
    app::A
    μ::VF
    σ::VF
end

function NormalScalingModelApplicator(
    app::AbstractModelApplicator, priors::AbstractVector{<:Distribution}, transM, ET::Type)
    # need to apply transform to entire vectors each of lowers and uppers
    θq = ([quantile(d, q) for d in priors] for q in (0.05, 0.95))
    ζlower, ζupper = inverse(transM).(θq)
    #ipar = first(axes(ζlower,1))
    pars = map(axes(ζlower,1)) do ipar
        dζ = fit(Normal, @qp_l(ζlower[ipar]), @qp_u(ζupper[ipar]))
        params(dζ)
    end
    # use collect to make it an array that works with gpu
    μ = collect(ET, first.(pars))  
    σ = collect(ET, last.(pars))
    NormalScalingModelApplicator(app, μ, σ)
end

function apply_model(app::NormalScalingModelApplicator, x, ϕ)
    y_perc = apply_model(app.app, x, ϕ)
    # @show typeof(app.μ)
    # @show typeof(ϕ)
    @assert eltype(app.μ) == eltype(ϕ)
    norminvcdf.(app.μ, app.σ, y_perc) # from StatsFuns
end







