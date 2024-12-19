module DoubleMM

using HybridVariationalInference
using ComponentArrays: ComponentArrays as CA
using Random
using Combinatorics
using StatsFuns: logistic


include("f_doubleMM.jl")

export f_doubleMM, S1, S2

end