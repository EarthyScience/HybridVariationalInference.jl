using SimpleChains, BenchmarkTools, Static, OptimizationOptimisers
import Zygote
using StatsFuns: logistic
using UnicodePlots
using Distributions
using StableRNGs
using LinearAlgebra, StatsBase, Combinatorics
using Random
using MLUtils

using Test
using HybridVariationalInference
using StableRNGs
using Random
using Statistics
using ComponentArrays: ComponentArrays as CA

using SimpleChains
import Zygote

using OptimizationOptimisers

using UnicodePlots

const EX = HybridVariationalInference.DoubleMM

() -> begin
    #const PROJECT_ROOT = pkgdir(@__MODULE__)
    _project_dir = basename(@__DIR__) == "uncNN" ? dirname(@__DIR__) : @__DIR__
    include(joinpath(_project_dir, "uncNN", "ComponentArrayInterpreter.jl"))
    include(joinpath(_project_dir, "uncNN", "util.jl")) # flatten1
end

const T = Float32
rng = StableRNG(111)

const n_covar_pc = 2
const n_covar = n_covar_pc + 3 # linear dependent
const n_site = 10^n_covar_pc
# n responses each per 200 observations
n_batch = n_site

# moved to f_doubleMM
#θP = θP_true = CA.ComponentVector(r0 = 0.3, K2=2.0)
#θM = EX.θM = CA.ComponentVector(r1 = 0.5, K1 = 0.2)

const n_θP = length(EX.θP)
const n_θM = length(EX.θM)

const int_θP = ComponentArrayInterpreter(EX.θP)
const int_θM = ComponentArrayInterpreter(EX.θM)
const int_θMs = ComponentArrayInterpreter(EX.θM, (n_batch,))
const int_θPMs_flat = ComponentArrayInterpreter(P = n_θP, Ms = n_θM * n_batch)
const int_θ = ComponentArrayInterpreter(CA.ComponentVector(;θP=EX.θP,θM=EX.θM))
# moved to f_doubleMM
# const int_θdoubleMM = ComponentArrayInterpreter(flatten1(CA.ComponentVector(;θP,θM)))
# const S1 = [1.0, 1.0, 1.0, 0.3, 0.1]
# const S2 = [1.0, 3.0, 5.0, 5.0, 5.0]
θ = CA.getdata(vcat(EX.θP, EX.θM))

const int_θPMs = ComponentArrayInterpreter(CA.ComponentVector(;EX.θP,
    θMs=CA.ComponentMatrix(zeros(n_θM, n_batch), first(CA.getaxes(EX.θM)), CA.Axis(i=1:n_batch))))

f = EX.f_doubleMM


# moved to f_doubleMM
# gen_q(InteractionsCovCor)
x_o, θMs_true0, g, q = EX.gen_q(
    rng, T, length(EX.θM), n_covar, n_site, n_θM);

# normalize to be distributed around the prescribed true values
θMs_true = int_θMs(scale_centered_at(θMs_true0, EX.θM, 0.1))

extrema(θMs_true)
histogram(vec(θMs_true[:r1,:]))
histogram(vec(θMs_true[:K1,:]))

@test isapprox(vec(mean(CA.getdata(θMs_true); dims=2)), CA.getdata(EX.θM), rtol=0.02)
@test isapprox(vec(std(CA.getdata(θMs_true); dims=2)), CA.getdata(EX.θM) .* 0.1, rtol=0.02)

# moved to f_doubleMM
#applyf(f_double, θMs_true, stack(Iterators.repeated(CA.getdata(θP_true), size(θMs_true,2))))

y_true = applyf(f, θMs_true, EX.θP)
σ_o = 0.01
#σ_o = 0.002
y_o = y_true .+ reshape(randn(length(y_true)), size(y_true)...) .* σ_o
scatterplot(vec(y_true), vec(y_o))
scatterplot(vec(log.(y_true)), vec(log.(y_o)))

# fit g to log(θ_true) ~ x_o
ϕg = ϕg0 = SimpleChains.init_params(g);
n_ϕg = length(ϕg)


#----- fit g to θMs_true
function loss_g(ϕg, x, g)
    ζMs = g(x, ϕg) # predict the log of the parameters
    θMs = exp.(ζMs)   
    loss = sum(abs2, θMs .- θMs_true)
    return loss, θMs
end
loss_g(ϕg,x_o, g)
Zygote.gradient(x-> loss_g(x, x_o, g)[1], ϕg);

optf = Optimization.OptimizationFunction((ϕg, p) -> loss_g(ϕg,x_o, g)[1],
    Optimization.AutoZygote())
optprob = Optimization.OptimizationProblem(optf, ϕg0);
res = Optimization.solve(optprob, Adam(0.02), callback=callback_loss(100), maxiters=500);

ϕg_opt1 = res.u;
scatterplot(vec(θMs_true), vec(loss_g(ϕg_opt1, x_o, g)[2]))
@test cor(vec(θMs_true), vec(loss_g(ϕg_opt1, x_o, g)[2])) > 0.9

#----------- fit q and θP to y_obs
int_ϕθP = ComponentArrayInterpreter(CA.ComponentVector(ϕg=1:length(ϕg), θP=EX.θP))
p = p0 = vcat(ϕg0, EX.θP .* 0.9);  # slightly disturb θP_true
#p = p0 = vcat(ϕg_opt1, θP_true .* 0.9);  # slightly disturb θP_true
p0c = int_ϕθP(p0); 
#gf(g,f_doubleMM, x_o, pc.ϕg, pc.θP)[1]


k = 10
# Pass the data for the batches as separate vectors wrapped in a tuple
train_loader = MLUtils.DataLoader((x_o, y_o), batchsize = k)
#l1 = loss_gf(p0, train_loader.data...)[1]

optf = Optimization.OptimizationFunction((ϕ, data) -> loss_gf(ϕ, data...)[1],
    Optimization.AutoZygote())
optprob = OptimizationProblem(optf, p0, train_loader)using SimpleChains, BenchmarkTools, Static, OptimizationOptimisers
import Zygote
using StatsFuns: logistic
using UnicodePlots
using Distributions
using StableRNGs
using LinearAlgebra, StatsBase, Combinatorics
using Random
using MLUtils

using Test
using HybridVariationalInference
using StableRNGs
using Random
using ComponentArrays: ComponentArrays as CA

const EX = HybridVariationalInference.DoubleMM

#const PROJECT_ROOT = pkgdir(@__MODULE__)
_project_dir = basename(@__DIR__) == "uncNN" ? dirname(@__DIR__) : @__DIR__
include(joinpath(_project_dir, "uncNN", "ComponentArrayInterpreter.jl"))
include(joinpath(_project_dir, "uncNN", "util.jl")) # flatten1

T = Float32
rng = StableRNG(111)

const n_covar_pc = 2
const n_covar = n_covar_pc + 3 # linear dependent
const n_site = 10^n_covar_pc
# n responses each per 200 observations
n_batch = n_site

# moved to f_doubleMM
#θP = θP_true = CA.ComponentVector(r0 = 0.3, K2=2.0)
#θM = EX.θM = CA.ComponentVector(r1 = 0.5, K1 = 0.2)

const n_θP = length(EX.θP)
const n_θM = length(EX.θM)

const int_θP = ComponentArrayInterpreter(EX.θP)
const int_θM = ComponentArrayInterpreter(EX.θM)
const int_θMs = ComponentArrayInterpreter(EX.θM, (n_batch,))
const int_θPMs_flat = ComponentArrayInterpreter(P = n_θP, Ms = n_θM * n_batch)
const int_θ = ComponentArrayInterpreter(CA.ComponentVector(;θP=EX.θP,θM=EX.θM))
# moved to f_doubleMM
# const int_θdoubleMM = ComponentArrayInterpreter(flatten1(CA.ComponentVector(;θP,θM)))
# const S1 = [1.0, 1.0, 1.0, 0.3, 0.1]
# const S2 = [1.0, 3.0, 5.0, 5.0, 5.0]
θ = CA.getdata(vcat(EX.θP, EX.θM))

const int_θPMs = ComponentArrayInterpreter(CA.ComponentVector(;EX.θP,
    θMs=CA.ComponentMatrix(zeros(n_θM, n_batch), first(CA.getaxes(EX.θM)), CA.Axis(i=1:n_batch))))

f = EX.f_doubleMM


# moved to f_doubleMM
# gen_q(InteractionsCovCor)
x_o, θMs_true0, g, q = EX.gen_q(
    rng, T, length(EX.θM), n_covar, n_site, n_θM);

# normalize to be distributed around the true values
σ_θM = EX.θM .* 0.1  # 10% around expected
dt = fit(ZScoreTransform, θMs_true0, dims=2)
θMs_true0_scaled = StatsBase.transform(dt, θMs_true0)
θMs_true = int_θMs(EX.θM .+  θMs_true0_scaled .* σ_θM)
#map(mean, eachrow(θMs_true)), map(std, eachrow(θMs_true))
#scatterplot(vec(θMs_true0), vec(θMs_true))
#scatterplot(vec(θMs_true0), vec(θMs_true0_scaled))

extrema(θMs_true)
histogram(vec(θMs_true))

# moved to f_doubleMM
#applyf(f_double, θMs_true, stack(Iterators.repeated(CA.getdata(θP_true), size(θMs_true,2))))

y_true = applyf(f_doubleMM, θMs_true, θP_true)
σ_o = 0.01
#σ_o = 0.002
y_o = y_true .+ reshape(randn(length(y_true)), size(y_true)...) .* σ_o
scatterplot(vec(y_true), vec(y_o))
scatterplot(vec(log.(y_true)), vec(log.(y_o)))

ϕg = ϕg0 = SimpleChains.init_params(g);
n_ϕg = length(ϕg)
ϕq = SimpleChains.init_params(q);
#G = SimpleChains.alloc_threaded_grad(g);
#@benchmark valgrad!($g, $mlpdloss, $x_o, $ϕg) # dropout active

#----- fit g to x_o and θMs_true
function loss_g(ϕg, x, g)
    ζMs = g(x, ϕg) # predict the log of the parameters
    θMs = exp.(ζMs)   
    loss = sum(abs2, θMs .- θMs_true)
    return loss, θMs
end
loss_g(ϕg,x_o, g)
Zygote.gradient(x-> loss_g(x, x_o, g)[1], ϕg);

optf = Optimization.OptimizationFunction((ϕg, p) -> loss_g(ϕg,x_o, g)[1],
    Optimization.AutoZygote())
optprob = Optimization.OptimizationProblem(optf, ϕg0);
res = Optimization.solve(optprob, Adam(0.02), callback=callback_loss(100), maxiters=500);

ϕg_opt1 = res.u
scatterplot(vec(θMs_true), vec(loss_g(ϕg_opt1, x_o, g)[2]))
@test cor(vec(θMs_true), vec(loss_g(ϕg_opt1, x_o, g)[2])) >  0.9

#-------- fit g and θP to x_o and y_o
int_ϕθP = ComponentArrayInterpreter(CA.ComponentVector(ϕg=1:length(ϕg), θP=θP_true))
p = p0 = vcat(ϕg0, θP_true .* 0.9);  # slightly disturb θP_true
#p = p0 = vcat(ϕg_opt1, θP_true .* 0.9);  # slightly disturb θP_true
p0c = int_ϕθP(p0); 
#gf(g,f_doubleMM, x_o, pc.ϕg, pc.θP)[1]




k = 10
# Pass the data for the batches as separate vectors wrapped in a tuple
train_loader = MLUtils.DataLoader((x_o, y_o), batchsize = k)
#l1 = loss_gf(p0, train_loader.data...)[1]

optf = Optimization.OptimizationFunction((ϕ, data) -> loss_gf(ϕ, data...)[1],
    Optimization.AutoZygote())
optprob = OptimizationProblem(optf, p0, train_loader)
# caution: larger learning rate (of 0.02) or fewer iterations -> skewed θMs_pred ~ θMs_true
res = Optimization.solve(optprob, Optimisers.Adam(0.02); callback=callback_loss(100), maxiters=2_000);
#res = Optimization.solve(optprob, Optimisers.ADAM(0.02); callback=callback_loss(100), epochs=200);



() -> begin
    loss_gf(p0)[1]
    loss_gf(vcat(ϕg, θP_true))[1]
    loss_gf(vcat(ϕg_opt1, θP_true))[1]
    loss_gf(res.u)[1]

    scatterplot(vec(loss_gf(res.u)[2]), vec(y_true))
    scatterplot(vec(loss_gf(res.u)[2]), vec(y_o))
    scatterplot(vec(y_true), vec(y_o))
end

poptc = int_ϕθP(res.u);
ϕg_opt, θP_opt = poptc.ϕg, poptc.θP;
hcat(θP_true, θP_opt, p0c.θP)
y_pred, θMs_pred = gf(g, f_doubleMM, x_o, ϕg_opt, θP_opt);
() -> begin
    scatterplot(vec(y_pred), vec(y_o))
    scatterplot(vec(y_pred), vec(y_true) )

    scatterplot(y_pred[1,:], y_true[1,:] )
    scatterplot(y_pred[2,:], y_true[2,:] )
    scatterplot(y_pred[1,:], y_o[1,:] )
    scatterplot(y_pred[2,:], y_o[2,:] )

    plt = scatterplot(θMs_true[1,:],θMs_pred[1,:])
    plt = scatterplot(θMs_true[2,:],θMs_pred[2,:])
end
#vcat(θMs_true, θMs_pred)
plt = scatterplot(vec(θMs_true), vec(θMs_pred))
#lineplot!(plt, 0.0, 1.1, identity)


# caution: larger learning rate (of 0.02) or fewer iterations -> skewed θMs_pred ~ θMs_true
res = Optimization.solve(optprob, Optimisers.Adam(0.02); callback=callback_loss(100), maxiters=2_000);
#res = Optimization.solve(optprob, Optimisers.ADAM(0.02); callback=callback_loss(100), epochs=200);



() -> begin
    loss_gf(p0)[1]
    loss_gf(vcat(ϕg, θP_true))[1]
    loss_gf(vcat(ϕg_opt1, θP_true))[1]
    loss_gf(res.u)[1]

    scatterplot(vec(loss_gf(res.u)[2]), vec(y_true))
    scatterplot(vec(loss_gf(res.u)[2]), vec(y_o))
    scatterplot(vec(y_true), vec(y_o))
end

poptc = int_ϕθP(res.u);
ϕg_opt, θP_opt = poptc.ϕg, poptc.θP;
hcat(θP_true, θP_opt, p0c.θP)
y_pred, θMs_pred = gf(g, f_doubleMM, x_o, ϕg_opt, θP_opt);
() -> begin
    scatterplot(vec(y_pred), vec(y_o))
    scatterplot(vec(y_pred), vec(y_true) )

    scatterplot(y_pred[1,:], y_true[1,:] )
    scatterplot(y_pred[2,:], y_true[2,:] )
    scatterplot(y_pred[1,:], y_o[1,:] )
    scatterplot(y_pred[2,:], y_o[2,:] )

    plt = scatterplot(θMs_true[1,:],θMs_pred[1,:])
    plt = scatterplot(θMs_true[2,:],θMs_pred[2,:])
end
#vcat(θMs_true, θMs_pred)
plt = scatterplot(vec(θMs_true), vec(θMs_pred))
#lineplot!(plt, 0.0, 1.1, identity)

