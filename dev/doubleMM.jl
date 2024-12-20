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

const case = DoubleMM.DoubleMMCase()
const MLengine = Val(nameof(SimpleChains))
scenario = (:default,)
rng = StableRNG(111)

par_templates = get_hybridcase_par_templates(case; scenario)

(; n_covar, n_site, n_batch, n_θM, n_θP) = get_hybridcase_sizes(case; scenario)

# const int_θP = ComponentArrayInterpreter(par_templates.θP)
# const int_θM = ComponentArrayInterpreter(par_templates.θM)
# const int_θPMs_flat = ComponentArrayInterpreter(P = n_θP, Ms = n_θM * n_batch)
# const int_θ = ComponentArrayInterpreter(CA.ComponentVector(;θP=par_templates.θP,θM=par_templates.θM))
# # moved to f_doubleMM
# # const int_θdoubleMM = ComponentArrayInterpreter(flatten1(CA.ComponentVector(;θP,θM)))
# # const S1 = [1.0, 1.0, 1.0, 0.3, 0.1]
# # const S2 = [1.0, 3.0, 5.0, 5.0, 5.0]
# θ = CA.getdata(vcat(par_templates.θP, par_templates.θM))

# const int_θPMs = ComponentArrayInterpreter(CA.ComponentVector(;par_templates.θP,
#     θMs=CA.ComponentMatrix(zeros(n_θM, n_batch), first(CA.getaxes(par_templates.θM)), CA.Axis(i=1:n_batch))))

(; xM, θP_true, θMs_true, xP, y_global_true, y_true, y_global_o, y_o) = gen_hybridcase_synthetic(
    case, rng; scenario);

@test isapprox(
    vec(mean(CA.getdata(θMs_true); dims = 2)), CA.getdata(par_templates.θM), rtol = 0.02)
@test isapprox(vec(std(CA.getdata(θMs_true); dims = 2)),
    CA.getdata(par_templates.θM) .* 0.1, rtol = 0.02)

#----- fit g to θMs_true
g, ϕg0 = gen_hybridcase_MLapplicator(case, MLengine; scenario);

function loss_g(ϕg, x, g)
    ζMs = g(x, ϕg) # predict the log of the parameters
    θMs = exp.(ζMs)
    loss = sum(abs2, θMs .- θMs_true)
    return loss, θMs
end
loss_g(ϕg0, xM, g)
Zygote.gradient(x -> loss_g(x, xM, g)[1], ϕg0);

optf = Optimization.OptimizationFunction((ϕg, p) -> loss_g(ϕg, xM, g)[1],
    Optimization.AutoZygote())
optprob = Optimization.OptimizationProblem(optf, ϕg0);
res = Optimization.solve(optprob, Adam(0.02), callback = callback_loss(100), maxiters = 600);

ϕg_opt1 = res.u;
loss_g(ϕg_opt1, xM, g)
scatterplot(vec(θMs_true), vec(loss_g(ϕg_opt1, xM, g)[2]))
@test cor(vec(θMs_true), vec(loss_g(ϕg_opt1, xM, g)[2])) > 0.9

#----------- fit g and θP to y_o
f = gen_hybridcase_PBmodel(case; scenario)

int_ϕθP = ComponentArrayInterpreter(CA.ComponentVector(
    ϕg = 1:length(ϕg0), θP = par_templates.θP))
p = p0 = vcat(ϕg0, par_templates.θP .* 0.9);  # slightly disturb θP_true

# Pass the site-data for the batches as separate vectors wrapped in a tuple
train_loader = MLUtils.DataLoader((xM, xP, y_o), batchsize = n_batch)

loss_gf = get_loss_gf(g, f, y_global_o, int_ϕθP)
l1 = loss_gf(p0, train_loader.data...)[1]

optf = Optimization.OptimizationFunction((ϕ, data) -> loss_gf(ϕ, data...)[1],
    Optimization.AutoZygote())
optprob = OptimizationProblem(optf, p0, train_loader)

res = Optimization.solve(
    optprob, Adam(0.02), callback = callback_loss(100), maxiters = 1000);

l1, y_pred_global, y_pred, θMs = loss_gf(res.u, train_loader.data...)
scatterplot(vec(θMs_true), vec(θMs))
scatterplot(log.(vec(θMs_true)), log.(vec(θMs)))
scatterplot(vec(y_pred), vec(y_o))
hcat(par_templates.θP, int_ϕθP(res.u).θP)
