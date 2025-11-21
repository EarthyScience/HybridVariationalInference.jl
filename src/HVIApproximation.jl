"""
    AbstractHVIApproximation

Provides a type hierarchy to distinguish different forms and
parameterizations of posterior approximations.    
"""
abstract type AbstractHVIApproximation end

abstract type AbstractMeanHVIApproximation <: AbstractHVIApproximation end

# First implementation with one big sparse covariance matrix
struct MeanHVIApproximationMat <: AbstractMeanHVIApproximation end

# Reimplementation with generating random numbers for each block separately
struct MeanHVIApproximation <: AbstractMeanHVIApproximation end

# for benchmarking changes, bevore implementing them
struct MeanHVIApproximationDev <: AbstractMeanHVIApproximation end 


