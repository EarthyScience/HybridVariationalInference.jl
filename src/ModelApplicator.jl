"""
    AbstractModelApplicator(x, ϕ; is_testmode = false)

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

(app::AbstractModelApplicator)(x, ϕ; kwargs...) = apply_model(app, x, ϕ; kwargs...)


"""
    NullModelApplicator()

Model applicator that returns its inputs. Used for testing.
"""
struct NullModelApplicator <: AbstractModelApplicator end

function apply_model(app::NullModelApplicator, x, ϕ; kwargs...)
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

Implementations may call 
`get_numberof_inputs_outputs(prob; scenario) -> (n_input, n_output)`.
"""
function construct_3layer_MLApplicator end

function get_numberof_inputs_outputs(prob; scenario)
    n_covar = get_hybridproblem_n_covar(prob; scenario)
    n_pbm_covars = length(get_hybridproblem_pbmpar_covars(prob; scenario))
    n_input = n_covar + n_pbm_covars
    (;θM) = get_hybridproblem_par_templates(prob; scenario)
    #n_out = length(θM)
    approx = get_hybridproblem_HVIApproximation(prob; scenario)
    n_output = get_numberof_MLinputs(approx, θM)
    (;n_input, n_output)
end



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
    range_scaled::UnitRange{Int}
end
@functor MagnitudeModelApplicator (app, multiplier)

function MagnitudeModelApplicator(app::AbstractModelApplicator, multiplier; range_scaled = 1:0)
    MagnitudeModelApplicator(app, multiplier, range_scaled)
end
function apply_model(app::MagnitudeModelApplicator, x, ϕ; kwargs...)
    #@show size(x), size(ϕ), app.multiplier
    @assert eltype(app.multiplier) == eltype(ϕ)
    if !isempty(app.range_scaled)
        res = apply_model(app.app, x, ϕ; kwargs...)
        res_scaled = index_firstdim(res,app.range_scaled) .* app.multiplier
        combine_range(res, res_scaled, app.range_scaled)
    else
        apply_model(app.app, x, ϕ; kwargs...) .* app.multiplier
    end
end


"""
    NormalScalingModelApplicator(app, μ, σ; range_scaled=1:0)
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
    range_scaled::UnitRange{Int}
end
@functor NormalScalingModelApplicator (app, μ, σ)

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
    range_scaled = 1:0,
    repeat_inner::Integer = 1) 
    pars = map(lowers, uppers) do lower, upper
        dζ = fit(Normal, @qp_l(lower), @qp_u(upper))
        params(dζ)
    end
    # use collect to make it an array that works with gpu
    μ = repeat(collect(FT, first.(pars)); inner=(repeat_inner,))  
    σ = repeat(collect(FT, last.(pars)); inner=(repeat_inner,))
    app = if isempty(range_scaled) || (repeat_inner == 1)
        NormalScalingModelApplicator(app, μ, σ, range_scaled)
    else
        error("debug and implement NormalScalingModelApplicator with repeated blocks, e.g. for multivariate normal distribution with independent components")
        range_scaled_rep = repeat(range_scaled, inner=repeat_inner)
        app_sub = NormalScalingModelApplicator(app, μ, σ, range_scaled_rep[2:end])
        NormalScalingModelApplicator(app_sub, μ, σ, range_scaled_rep[2:end])
    end
end

function NormalScalingModelApplicator(
    app::AbstractModelApplicator, μ, σ; 
    range_scaled = 1:0,  # empty range indicates rescaling all outputs
    repeat_inner::Integer = 1) 
    pars = map(lowers, uppers) do lower, upper
        dζ = fit(Normal, @qp_l(lower), @qp_u(upper))
        params(dζ)
    end
    # use collect to make it an array that works with gpu
    μ = repeat(collect(FT, first.(pars)); inner=(repeat_inner,))  
    σ = repeat(collect(FT, last.(pars)); inner=(repeat_inner,))
    range_scaled_rep = repeat(range_scaled, inner=repeat_inner)
    NormalScalingModelApplicator(app, μ, σ, range_scaled_rep)
end


function apply_model(app::NormalScalingModelApplicator, x, ϕ; kwargs...)
    y_perc = apply_model(app.app, x, ϕ; kwargs...)
    # @show typeof(app.μ)
    # @show typeof(ϕ)
    @assert eltype(app.μ) == eltype(ϕ)
    ans = if !isempty(app.range_scaled)
        ans_scaled = norminvcdf.(app.μ, app.σ, index_firstdim(y_perc,app.range_scaled)) # from StatsFuns
        combine_range(y_perc, ans_scaled, app.range_scaled)
    else
        ans_scaled = norminvcdf.(app.μ, app.σ, y_perc) 
    end
    # if !all(isfinite.(ans))
    #     @info "NormalScalingModelApplicator.apply_model: encountered non-finite results"
    #     #@show ans, y_perc, app.μ, app.σ
    #     #@show app.app, x, ϕ
    #     #error("error to print stacktrace")
    # end
end

index_firstdim(v::AbstractVector, i) = v[i]
index_firstdim(v::AbstractMatrix, i) = v[i,:]

"""
    RangeScalingModelApplicator(app, y0)

Wrapper around AbstractModelApplicator assumed to predict on (0,1) with 
a linear mappting to prededfined range.
"""
struct RangeScalingModelApplicator{VF,A} <: AbstractModelApplicator
    offset::VF
    width::VF
    app::A
    range_scaled::UnitRange{Int}
end

function apply_model(app::RangeScalingModelApplicator, x, ϕ; kwargs...)
    res0 = apply_model(app.app, x, ϕ; kwargs...)
    if !isempty(app.range_scaled)
        res_scaled = index_firstdim(res0,app.range_scaled) .* app.width .+ app.offset
        combine_range(res0, res_scaled, app.range_scaled)
    else
        res0 .* app.width .+ app.offset
    end
end

function combine_range(res0, res_scaled, range_scaled)
    range_before = 1:(range_scaled[1]-1)
    range_after = (range_scaled[end]+1):size(res0,1)
    vcat(index_firstdim(res0,range_before), res_scaled, index_firstdim(res0,range_after))
end



"""
    RangeScalingModelApplicator(app, lowers, uppers, FT::Type; repeat_inner::Integer = 1) 

Provide the target ragen by vectors `lower` and `upper`. The size of both
outputs must correspond to the size of the output of `app`.

"""
function RangeScalingModelApplicator(
    app::AbstractModelApplicator, 
    lowers::VT, uppers::VT,
    FT::Type;
    range_scaled = 1:0
    ) where VT<:AbstractVector
    width = collect(FT, uppers .- lowers)
    lowersFT = collect(FT, lowers) # convert eltype
    RangeScalingModelApplicator(lowersFT, width, app, range_scaled)
end








