"""
    AbstractHVIApproximation

Provides a type hierarchy to distinguish different forms and
parameterizations of posterior approximations.   

Subtypes must implement method `get_numberof_MLinputs(approx, θM)` that 
returns the number of required outputs of the machine learning model
per site for a parameter vector `θM`.

Subtypes must implement method 
`get_numberof_θM(::AbstractHVIApproximation, ml_pred::AbstractArray)` that 
returns the number compponents mean parameters `θM` given the machine learning
model predictions, if it differs from the default implementation of
returning their number of rows.
For example, [`MeanScalingHVIApproximation`](@ref) there are more outputs than
the mean parameters with the ML model.
"""
abstract type AbstractHVIApproximation end,
function get_numberof_MLinputs end
get_numberof_θM(::AbstractHVIApproximation, ml_pred::AbstractArray) = size(ml_pred,1)

abstract type AbstractMeanHVIApproximation <: AbstractHVIApproximation end
get_numberof_MLinputs(::AbstractMeanHVIApproximation, θM) = length(θM)

# First implementation with one big sparse covariance matrix
struct MeanHVIApproximationMat <: AbstractMeanHVIApproximation end

# Reimplementation with generating random numbers for each block separately
struct MeanHVIApproximation <: AbstractMeanHVIApproximation end

# for benchmarking changes, before implementing them
struct MeanHVIApproximationDev <: AbstractMeanHVIApproximation end 


abstract type AbstractMeanVarSepHVIApproximation <: AbstractHVIApproximation end
get_numberof_MLinputs(::AbstractMeanVarSepHVIApproximation, θM) = length(θM)

struct MeanVarSepHVIApproximation <: AbstractMeanVarSepHVIApproximation end


abstract type AbstractMeanScalingHVIApproximation <: AbstractHVIApproximation end

"""
    MeanScalingHVIApproximation(scalingblocks_ends, logσ2_ζM_base)

An approximation that requires the ML model to predict a scaling factor, 
(i.e. an additive of the log) for
the mean diagonal of the covariance matrix per site in addition to 
the mean values of parameters in unconstrained space.

TODO: describe multiplication of site-factor, parameter-factor and base_variance
for each parameter.
"""
struct MeanScalingHVIApproximation{T} <: AbstractMeanScalingHVIApproximation 
    scalingblocks_ends::Vector{Int} # indices of end of blocks with the same scaling factor
    # log_var of last parameters in block (to be mulitplied by par_factor and site_factor)
    logσ2_ζM_bases::Vector{T} # already repeated for blocks in parameters
    # indexing into logσ2_par_offsets_before_end, including repeats and zeros
    idxs_par0::Vector{Int}
    # indexing into logσ2_sites including repeats
    idxs_repblocks::Vector{Int}
end
function MeanScalingHVIApproximation(scalingblocks_ends, logσ2_ζM_base::AbstractVector{T}
    ) where T
    idxs_par0 = insert_zeros(
        1:(scalingblocks_ends[end] - length(scalingblocks_ends)), scalingblocks_ends)
    length_scale_blocks = vcat(first(scalingblocks_ends), diff(scalingblocks_ends))
    idxs_repblocks = vcat((fill(i, length_scale_blocks[i]) for i in axes(length_scale_blocks,1))...)
    logσ2_ζM_bases = reduce(vcat, fill.(logσ2_ζM_base, length_scale_blocks))
    MeanScalingHVIApproximation{T}(
        scalingblocks_ends, logσ2_ζM_bases, idxs_par0, idxs_repblocks)
end

function MeanScalingHVIApproximation{T}(approx::AbstractMeanScalingHVIApproximation; 
    scalingblocks_ends = approx.scalingblocks_ends,
    logσ2_ζM_base::AbstractVector{T} = approx.logσ2_ζM_bases[scalingblocks_ends],
) where T
    MeanScalingHVIApproximation(scalingblocks_ends, logσ2_ζM_base)
end
function get_numberof_MLinputs(approx::MeanScalingHVIApproximation, θM) 
    length(θM) + length(approx.scalingblocks_ends)
end
function get_numberof_θM(approx::MeanScalingHVIApproximation, ml_pred::AbstractArray) 
    # scalingblocks_ends reports the end position of the parameters, hence, the last
    # corresponds to the number of overall parameters.
    approx.scalingblocks_ends[end]
end

