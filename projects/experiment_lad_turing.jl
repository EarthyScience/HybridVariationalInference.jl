# experiment with correction Turing.@addlogprob! +logx
#   for correcting sampling at log-transformed scale

if basename(pwd()) != "projects"; cd("projects"); end; Pkg.activate("."); 
@info "Start of doubleMM_site.jl"
@info("Number of threads: $(Threads.nthreads())")

using Test 
using StableRNGs
rng = StableRNG(115)

#---- do an DEMC inversion of the PBM model with parameters at log-scale
using DistributionFits
using PDMats
using Turing
#using MCMCChains
using FlexiChains
using DimensionalData

using AlgebraOfGraphics, CairoMakie, FigureHelpers
const aog = AlgebraOfGraphics
using ColorBrewer: ColorBrewer
using CategoricalArrays
using PairPlots
using DataFrames



# to adapt already the v0.7 syntax of FlexiChains
macro vn(args...)
    return macroexpand(__module__, :(@FlexiChains.varname($(args...))))
end

@model function mζ_flat(y,::Type{T} = Float64) where {T}
    logx ~ Normal(0.0,2.0)
    x = exp(logx)
    y ~ Normal(x, 1.0)
    #    Turing.@addlogprob! (-sum(ζP) -sum(ζMs))
end
@model function mζ_corminus(y,::Type{T} = Float64) where {T}
    logx ~ Normal(0.0,2.0)
    x = exp(logx)
    y ~ Normal(x, 1.0)
    Turing.@addlogprob! -logx
end
@model function mζ_corplus(y,::Type{T} = Float64) where {T}
    logx ~ Normal(0.0,2.0)
    x = exp(logx)
    y ~ Normal(x, 1.0)
    Turing.@addlogprob! +logx
end

n_threads = min(16, Threads.nthreads())
function sample_nocor(mod=mζ_flat) 
    @info mod
    model = mod(2.0)
    n_sample_NUTS = 1_000
    chn = sample(model, NUTS(20, 0.65), n_sample_NUTS);
    # chn = sample(model, NUTS(8, 0.65), MCMCThreads(), ceil(Integer,n_sample_NUTS/n_threads), 
    #     n_threads; initial_params=map(_ -> sample_initial_params(rng),1:n_threads))
    chn
end
include("scratch_include.jl")
mods = (;nocor=mζ_flat, cormin=mζ_corminus, corplus=mζ_corplus)
chns = map(sample_nocor, mods);
chnst = map(chn -> FlexiChains.transform_values(chn, @vn(logx) => exp => @vn(x)), chns);
st = map(summarystats, chnst)
st.corplus[@vn(x)]
Makie.plot(chnst.nocor)
Makie.plot(chnst.cormin)
Makie.plot(chnst.corplus)

# need to add logx to regain undistorted distribution of x around 2
#   with nocor or corminus distribution is distorted towards lower values

