() -> begin
    using SimpleChains, BenchmarkTools, Static, OptimizationOptimisers
    import Zygote
    using StatsFuns: logistic
    using UnicodePlots
    using Distributions
    using StableRNGs
    using LinearAlgebra, StatsBase, Combinatorics
    using Random
end

using Test
using HybridVariationalInference
using StableRNGs
using Random
using Statistics
using ComponentArrays: ComponentArrays as CA

using SimpleChains
using MLUtils
import Zygote

using OptimizationOptimisers

using UnicodePlots

const EX = HybridVariationalInference.DoubleMM
const case = DoubleMM.DoubleMMCase()
const MLengine = Val(nameof(SimpleChains))
scenario=(:default,)

rng = StableRNG(111)

(; n_covar_pc, n_covar, n_site, n_batch, n_θM, n_θP) = get_case_sizes(case; scenario)

# const int_θP = ComponentArrayInterpreter(EX.θP)
# const int_θM = ComponentArrayInterpreter(EX.θM)
# const int_θPMs_flat = ComponentArrayInterpreter(P = n_θP, Ms = n_θM * n_batch)
# const int_θ = ComponentArrayInterpreter(CA.ComponentVector(;θP=EX.θP,θM=EX.θM))
# # moved to f_doubleMM
# # const int_θdoubleMM = ComponentArrayInterpreter(flatten1(CA.ComponentVector(;θP,θM)))
# # const S1 = [1.0, 1.0, 1.0, 0.3, 0.1]
# # const S2 = [1.0, 3.0, 5.0, 5.0, 5.0]
# θ = CA.getdata(vcat(EX.θP, EX.θM))

# const int_θPMs = ComponentArrayInterpreter(CA.ComponentVector(;EX.θP,
#     θMs=CA.ComponentMatrix(zeros(n_θM, n_batch), first(CA.getaxes(EX.θM)), CA.Axis(i=1:n_batch))))

# moved to f_doubleMM
# gen_q(InteractionsCovCor)
x_o, θMs_true0 = gen_cov_pred(case, rng; scenario)
# normalize to be distributed around the prescribed true values
int_θMs_sites = ComponentArrayInterpreter(EX.θM, (n_site,))
int_θMs_batch = ComponentArrayInterpreter(EX.θM, (n_batch,))
θMs_true = int_θMs_sites(scale_centered_at(θMs_true0, EX.θM, 0.1));

@test isapprox(vec(mean(CA.getdata(θMs_true); dims=2)), CA.getdata(EX.θM), rtol=0.02)
@test isapprox(vec(std(CA.getdata(θMs_true); dims=2)), CA.getdata(EX.θM) .* 0.1, rtol=0.02)


#----- fit g to θMs_true
g, ϕg0 = gen_g(case, MLengine; scenario)
n_ϕg = length(ϕg0)

function loss_g(ϕg, x, g)
    ζMs = g(x, ϕg) # predict the log of the parameters
    θMs = exp.(ζMs)   
    loss = sum(abs2, θMs .- θMs_true)
    return loss, θMs
end
loss_g(ϕg0, x_o, g)
Zygote.gradient(x-> loss_g(x, x_o, g)[1], ϕg0);

optf = Optimization.OptimizationFunction((ϕg, p) -> loss_g(ϕg,x_o, g)[1],
    Optimization.AutoZygote())
optprob = Optimization.OptimizationProblem(optf, ϕg0);
res = Optimization.solve(optprob, Adam(0.02), callback=callback_loss(100), maxiters=600);

ϕg_opt1 = res.u;
loss_g(ϕg_opt1, x_o, g)
scatterplot(vec(θMs_true), vec(loss_g(ϕg_opt1, x_o, g)[2]))
@test cor(vec(θMs_true), vec(loss_g(ϕg_opt1, x_o, g)[2])) > 0.9

#----------- fit g and θP to y_obs
f = gen_f(case; scenario)
y_true = f(EX.θP, θMs_true, zip())[2]

σ_o = 0.01
#σ_o = 0.002
y_o = y_true .+ reshape(randn(length(y_true)), size(y_true)...) .* σ_o
scatterplot(vec(y_true), vec(y_o))
scatterplot(vec(log.(y_true)), vec(log.(y_o)))

# fit g to log(θ_true) ~ x_o

int_ϕθP = ComponentArrayInterpreter(CA.ComponentVector(ϕg=1:length(ϕg0), θP=EX.θP))
p = p0 = vcat(ϕg0, EX.θP .* 0.9);  # slightly disturb θP_true
# #p = p0 = vcat(ϕg_opt1, θP_true .* 0.9);  # slightly disturb θP_true
# p0c = int_ϕθP(p0); 
# #gf(g,f_doubleMM, x_o, pc.ϕg, pc.θP)[1]


# Pass the data for the batches as separate vectors wrapped in a tuple
train_loader = MLUtils.DataLoader((
    x_o, 
    fill((), n_site), # xP
    y_o
    ), batchsize = n_batch)

loss_gf = get_loss_gf(g, f, Float32[], int_ϕθP)
l1 = loss_gf(p0, train_loader.data...)[1]

optf = Optimization.OptimizationFunction((ϕ, data) -> loss_gf(ϕ, data...)[1],
    Optimization.AutoZygote())
optprob = OptimizationProblem(optf, p0, train_loader)

res = Optimization.solve(optprob, Adam(0.02), callback=callback_loss(100), maxiters=1000);

l1, y_pred_global, y_pred, θMs = loss_gf(res.u, train_loader.data...)
scatterplot(vec(θMs_true), vec(θMs))
scatterplot(log.(vec(θMs_true)), log.(vec(θMs)))
scatterplot(vec(y_pred), vec(y_o))
hcat(EX.θP, int_ϕθP(res.u).θP)
