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

Implemented overloads of `construct_ChainsApplicator` for layers of 
- `SimpleChains.SimpleChain`
- `Flux.Chain`
- `Lux.Chain`
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
Construct a parametric type-stable model applicator, given
covariates, `x`, and parameters, `ϕ`.

The default returns the current model applicator.
"""
function construct_partric(app::AbstractModelApplicator, x, ϕ) 
    app
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
        scenario::Val{scen}) where scen

Construct a machine learning model for given Proglem and machine learning engine.
Implemented for machine learning extensions, such as Flux or SimpleChains.
`ml_engine` usually is of type `Val{Symbol}`, e.g. Val(:Flux). See `select_ml_engine`.       

Scenario is a value-type of `NTuple{_,Symbol}`.
"""
function construct_3layer_MLApplicator end

"""
    select_ml_engine(;scenario)

Returns a value type `Val{:Symbol}` to dispatch on the machine learning engine to use.
- defaults to `Val(:SimpleChains)`
- `:use_Lux ∈ scenario -> Val(:Lux)`
- `:use_Flux ∈ scenario -> Val(:Flux)`
"""
function select_ml_engine(; scenario::Val{scen}) where scen
    if :use_Lux ∈ scen
        return Val(:Lux)
    elseif :use_Flux ∈ scen
        return Val(:Flux)
    else
        # default
        return Val(:SimpleChains)
    end
end

"""
    MagnitudeModelApplicator(app, y0)

Wrapper around AbstractModelApplicator that multiplies the prediction
of the wrapped `app` by scalar `y0`.
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
(assumed in [0..1], such as output of logistic activation function)
to a quantile of a Normal distribution. 

Length of μ, σ must correspond to the number of outputs of the wrapped ModelApplicator.

This helps to keep raw ML-predictions (in confidence bounds) and weights in a 
similar magnitude.
Compared to specifying bounds, this allows for the possibility 
(although harder to converge) far beyond the confidence bounds.

The second constructor fits a normal distribution of the inverse-transformed 5% and 95%
quantiles of prior distributions.
"""
struct NormalScalingModelApplicator{VF,A} <: AbstractModelApplicator
    app::A
    μ::VF
    σ::VF
end
@functor NormalScalingModelApplicator

"""
    NormalScalingModelApplicator(app, lowers, uppers, FT::Type; repeat_inner::Integer = 1) 

Fit a Normal distribution to number iterators `lower` and `upper` and transform 
results of the wrapped `app` `AbstractModelApplicator`.
If `repeat_inner` is given, each fitted distribution is repeated as many times
to support independent multivariate normal distribution.

`FT` is the specific FloatType to use to construct Distributions, 
It usually corresponds to the type used in other ML-parts of the model, e.g. `Float32`.
"""
function NormalScalingModelApplicator(
    app::AbstractModelApplicator, lowers, uppers, FT::Type; 
    repeat_inner::Integer = 1) 
    pars = map(lowers, uppers) do lower, upper
        dζ = fit(Normal, @qp_l(lower), @qp_u(upper))
        params(dζ)
    end
    # use collect to make it an array that works with gpu
    μ = repeat(collect(FT, first.(pars)); inner=(repeat_inner,))  
    σ = repeat(collect(FT, last.(pars)); inner=(repeat_inner,))
    NormalScalingModelApplicator(app, μ, σ)
end

function apply_model(app::NormalScalingModelApplicator, x, ϕ)
    y_perc = apply_model(app.app, x, ϕ)
    # @show typeof(app.μ)
    # @show typeof(ϕ)
    @assert eltype(app.μ) == eltype(ϕ)
    ans = norminvcdf.(app.μ, app.σ, y_perc) # from StatsFuns
    if !all(isfinite.(ans))
        @info "NormalScalingModelApplicator.apply_model: encountered non-finite results"
        @show ans, y_perc, app.μ, app.σ
        @show app.app, x, ϕ
        error("error to print stacktrace")
    end
    ans
end







