"""
    AbstractHVIApproximation

Provides a type hierarchy to distinguish different forms and
parameterizations of posterior approximations.   

Subtypes must implement method `get_numberof_MLinputs(approx, θM)` that 
returns the numberof required outputs of the machine learning model
per site for a parameter vector `θM`.
"""
abstract type AbstractHVIApproximation end,
function get_numberof_MLinputs end

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

struct MeanScalingHVIApproximation{T} <: AbstractMeanScalingHVIApproximation 
    scalingblocks_ends::Vector{Int}
    logσ2_ζM_base::Vector{T}
end
function MeanScalingHVIApproximation{T}(approx::AbstractMeanScalingHVIApproximation; 
    scalingblocks_ends = approx.scalingblocks_ends,
    logσ2_ζM_base = approx.logσ2_ζM_base,
) where T
    MeanScalingHVIApproximation{T}(scalingblocks_ends, logσ2_ζM_base)
end
function get_numberof_MLinputs(approx::MeanScalingHVIApproximation, θM) 
    length(θM) + length(approx.scalingblocks_ends)
end
