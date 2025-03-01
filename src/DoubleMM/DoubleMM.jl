module DoubleMM

using HybridVariationalInference
using HybridVariationalInference: HybridVariationalInference as HVI
using ComponentArrays: ComponentArrays as CA
using Random
using StableRNGs
using Combinatorics
using StatsFuns: logistic
using Bijectors
using CUDA
using Distributions, DistributionFits


export f_doubleMM, xP_S1, xP_S2
include("f_doubleMM.jl")


end