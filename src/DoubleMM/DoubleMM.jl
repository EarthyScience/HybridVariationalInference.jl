module DoubleMM

using HybridVariationalInference
using HybridVariationalInference: HybridVariationalInference as HVI
using ComponentArrays: ComponentArrays as CA
using Random
using Combinatorics
using StatsFuns: logistic
using Bijectors


include("f_doubleMM.jl")

export f_doubleMM, S1, S2

end