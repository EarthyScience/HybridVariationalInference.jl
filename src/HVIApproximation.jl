"""
    AbstractHVIApproximation

Provides a type hierarchy to distinguish different forms and
parameterizations of posterior approximations.    
"""
abstract type AbstractHVIApproximation end

abstract type AbstractMeanHVIApproximation <: AbstractHVIApproximation end

struct MeanHVIApproximation <: AbstractMeanHVIApproximation end
struct MeanHVIApproximationMat <: AbstractMeanHVIApproximation end


