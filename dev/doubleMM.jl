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

tmpf = () -> begin
    #----------- fit g and θP to y_o
    # end2end inversion
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
end

#---------- HADVI
# TODO think about good general initializations
coef_logσ2_logMs = [-5.769 -3.501; -0.01791 0.007951]
logσ2_logP = CA.ComponentVector(r0=-8.997, K2=-5.893)
mean_σ_o_MC = 0.006042

# correlation matrices
ρsP = zeros(sum(1:(n_θP-1)))
ρsM = zeros(sum(1:(n_θM-1)))

ϕunc = CA.ComponentVector(;
    logσ2_logP=logσ2_logP,
    coef_logσ2_logMs=coef_logσ2_logMs,
    ρsP,
    ρsM)
int_unc = ComponentArrayInterpreter(ϕunc)

# for a conservative uncertainty assume σ2=1e-10 and no relationship with magnitude
ϕunc0 = CA.ComponentVector(;
    logσ2_logP=fill(-10.0, n_θP),
    coef_logσ2_logMs=reduce(hcat, ([-10.0, 0.0] for _ in 1:n_θM)),
    ρsP,
    ρsM)

logσ2y = fill(2 * log(σ_o), size(y_o, 1))
n_MC = 3


#-------------- ADVI with g inside cost function
using CUDA
using TransformVariables

transPMs_batch = as(
    (P=as(Array, asℝ₊, n_θP),
    Ms=as(Array, asℝ₊, n_θM, n_batch)))
transPMs_all = as(
    (P=as(Array, asℝ₊, n_θP),
    Ms=as(Array, asℝ₊, n_θM, n_site)))
    
ϕ_true = θ = CA.ComponentVector(;
    μP=θP_true,
    ϕg=ϕg_opt,
    unc=ϕunc);
trans_gu = as(
    (μP=as(Array, asℝ₊, n_θP),
    ϕg=as(Array, n_ϕg),
    unc=as(Array, length(ϕunc))))
trans_g = as(
    (μP=as(Array, asℝ₊, n_θP),
    ϕg=as(Array, n_ϕg)))

const int_PMs_batch = ComponentArrayInterpreter(CA.ComponentVector(; θP,
    θMs=CA.ComponentMatrix(
        zeros(n_θM, n_batch), first(CA.getaxes(θM)), CA.Axis(i=1:n_batch))))

interpreters = interpreters_g = map(get_concrete,(; 
    μP_ϕg_unc=ComponentArrayInterpreter(ϕ_true), 
    PMs=int_PMs_batch,
    unc=ComponentArrayInterpreter(ϕunc)
    ))

ϕg_true_vec = CA.ComponentVector(
    TransformVariables.inverse(trans_gu, cv2NamedTuple(ϕ_true)))
ϕcg_true = interpreters.μP_ϕg_unc(ϕg_true_vec)
ϕ_ini = ζ = vcat(ϕcg_true[[:μP, :ϕg]] .* 1.2, ϕcg_true[[:unc]]);
ϕ_ini0 = ζ = vcat(ϕcg_true[:μP] .* 0.0, SimpleChains.init_params(g), ϕunc0);

neg_elbo_transnorm_gf(rng, g, f, ϕcg_true, y_o[:, 1:n_batch], x_o[:, 1:n_batch],
    transPMs_batch, map(get_concrete, interpreters);
    n_MC=8, logσ2y)
Zygote.gradient(ϕ -> neg_elbo_transnorm_gf(
        rng, g, f, ϕ, y_o[:, 1:n_batch], x_o[:, 1:n_batch],
        transPMs_batch, interpreters; n_MC=8, logσ2y), ϕcg_true)

() -> begin
    train_loader = MLUtils.DataLoader((x_o, y_o), batchsize = n_batch)

    optf = Optimization.OptimizationFunction((ζg, data) -> begin
            x_o, y_o = data
            neg_elbo_transnorm_gf(
                rng, g, f, ζg, y_o, x_o, transPMs_batch, map(get_concrete, interpreters_g); n_MC=5, logσ2y)
        end,
        Optimization.AutoZygote())
    optprob = Optimization.OptimizationProblem(optf, CA.getdata(ϕ_ini), train_loader);
    res = Optimization.solve(optprob, Optimisers.Adam(0.02), callback=callback_loss(50), maxiters=800);
    #optprob = Optimization.OptimizationProblem(optf, ϕ_ini0);
    #res = Optimization.solve(optprob, Adam(0.02), callback=callback_loss(50), maxiters=1_400);
end
        
#using Lux
ϕ = ϕcg_true |> gpu;
x_o_gpu = x_o |> gpu;
# y_o = y_o |> gpu
# logσ2y = logσ2y |> gpu
n_covar = size(x_o, 1)
g_flux = Flux.Chain(
    # dense layer with bias that maps to 8 outputs and applies `tanh` activation
    Flux.Dense(n_covar => n_covar * 4, tanh),
    Flux.Dense(n_covar * 4 => n_covar * 4, logistic),
    # dense layer without bias that maps to n outputs and `identity` activation
    Flux.Dense(n_covar * 4 => n_θM, identity, bias=false),
)
() -> begin
    using Lux
    g_lux = Lux.Chain(
        # dense layer with bias that maps to 8 outputs and applies `tanh` activation
        Lux.Dense(n_covar => n_covar * 4, tanh),
        Lux.Dense(n_covar * 4 => n_covar * 4, logistic),
        # dense layer without bias that maps to n outputs and `identity` activation
        Lux.Dense(n_covar * 4 => n_θM, identity, use_bias=false),
    )
    ps, st = Lux.setup(Random.default_rng(), g_lux)
    ps_ca = CA.ComponentArray(ps) |> gpu
    st = st |> gpu
    g_luxs = StatefulLuxLayer{true}(g_lux, nothing, st)
    g_luxs(x_o_gpu[:, 1:n_batch], ps_ca)
    ax_g = CA.getaxes(ps_ca)
    g_luxs(x_o_gpu[:, 1:n_batch], CA.ComponentArray(ϕ.ϕg, ax_g))
    interpreters = (interpreters..., ϕg = ComponentArrayInterpreter(ps_ca))
    ϕg = CA.ComponentArray(ϕ.ϕg, ax_g)
    ϕgc = interpreters.ϕg(ϕ.ϕg)
    g_gpu = g_luxs
end
g_gpu = g_flux

#Zygote.gradient(ϕg -> sum(g_gpu(x_o_gpu[:, 1:n_batch],ϕg)), ϕgc)
# Zygote.gradient(ϕg -> sum(compute_g(g_gpu, x_o_gpu[:, 1:n_batch], ϕg, interpreters)), ϕ.ϕg)
# Zygote.gradient(ϕ -> sum(tmp_gen1(g_gpu, x_o_gpu[:, 1:n_batch], ϕ, interpreters)), ϕ.ϕg)
# Zygote.gradient(ϕ -> sum(tmp_gen2(g_gpu, x_o_gpu[:, 1:n_batch], ϕ, interpreters)), CA.getdata(ϕ))
# Zygote.gradient(ϕ -> sum(tmp_gen2(g_gpu, x_o_gpu[:, 1:n_batch], ϕ, interpreters)), ϕ) |> cpu
# Zygote.gradient(ϕ -> sum(tmp_gen3(g_gpu, x_o_gpu[:, 1:n_batch], ϕ, interpreters)), ϕ) |> cpu
# Zygote.gradient(ϕ -> sum(tmp_gen4(g_gpu, x_o_gpu[:, 1:n_batch], ϕ, interpreters)[1]), ϕ) |> cpu
# generate_ζ(rng, g_gpu, f, ϕ, x_o_gpu[:, 1:n_batch], interpreters)
# Zygote.gradient(ϕ -> sum(generate_ζ(rng, g_gpu, f, ϕ, x_o_gpu[:, 1:n_batch], interpreters)[1]), ϕ) |> cpu
# include(joinpath(@__DIR__, "uncNN", "elbo.jl")) # callback_loss
# neg_elbo_transnorm_gf(rng, g_gpu, f, ϕ, y_o[:, 1:n_batch], 
#     x_o_gpu[:, 1:n_batch], transPMs_batch, interpreters; logσ2y)
# Zygote.gradient(ϕ -> sum(neg_elbo_transnorm_gf(rng, g_gpu, f, ϕ, y_o[:, 1:n_batch], 
# x_o_gpu[:, 1:n_batch], transPMs_batch, interpreters; logσ2y)[1]), ϕ) |> cpu


fcost(ϕ) = neg_elbo_transnorm_gf(rng, g_gpu, f, ϕ, y_o[:, 1:n_batch], 
    x_o_gpu[:, 1:n_batch], transPMs_batch, map(get_concrete, interpreters);
    n_MC=8, logσ2y = logσ2y)
fcost(ϕ)
gr = Zygote.gradient(fcost, ϕ) |> cpu;
Zygote.gradient(fcost, CA.getdata(ϕ))


train_loader = MLUtils.DataLoader((x_o_gpu, y_o), batchsize = n_batch)

optf = Optimization.OptimizationFunction((ζg, data) -> begin
        x_o, y_o = data
        neg_elbo_transnorm_gf(
            rng, g_gpu, f, ζg, y_o, x_o, transPMs_batch, map(get_concrete, interpreters_g); n_MC=5, logσ2y)
    end,
    Optimization.AutoZygote())
optprob = Optimization.OptimizationProblem(optf, CA.getdata(ϕ_ini) |> gpu, train_loader);
res = res_gpu = Optimization.solve(optprob, Optimisers.Adam(0.02), callback=callback_loss(50), maxiters=800);

ζ_VIc = interpreters_g.μP_ϕg_unc(res.u |> cpu)
ζMs_VI = g(x_o, ζ_VIc.ϕg)
ϕunc_VI = int_unc(ζ_VIc.unc)

hcat(θP_true, exp.(ζ_VIc.μP))
plt = scatterplot(vec(θMs_true), vec(exp.(ζMs_VI)))
#lineplot!(plt, 0.0, 1.1, identity)
# 
hcat(ϕunc, ϕunc_VI) # need to compare to MC sample
# hard to estimate for original very small theta's but otherwise good

# test predicting correct obs-uncertainty of predictive posterior
n_sample_pred = 200
intm_PMs_gen = ComponentArrayInterpreter(CA.ComponentVector(; θP,
    θMs=CA.ComponentMatrix(
        zeros(n_θM, n_site), first(CA.getaxes(θM)), CA.Axis(i=1:n_sample_pred))))

include(joinpath(@__DIR__, "uncNN", "elbo.jl")) # callback_loss
ζs, _ = generate_ζ(rng, g, f, res.u |> cpu, x_o, 
    (;interpreters..., PMs = intm_PMs_gen); n_MC=n_sample_pred)
# ζ = ζs[:,1]   
θsc = stack(ζ -> CA.getdata(CA.ComponentVector(
        TransformVariables.transform(transPMs_all, ζ))), eachcol(ζs));
y_pred = stack(map(ζ -> first(predict_y(ζ, f, transPMs_all)), eachcol(ζs)));

size(y_pred)
σ_o_post = mapslices(std, y_pred; dims=3);
#describe(σ_o_post)
vcat(σ_o, mean_σ_o_MC, mean(σ_o_post), sqrt(mean(abs2, σ_o_post)))
mean_y_pred = map(mean, eachslice(y_pred; dims=(1, 2)))
#describe(mean_y_pred - y_o)
histogram(vec(mean_y_pred - y_true)) # predictions centered around y_o (or y_true)

# look at θP, θM1 of first site
intm = ComponentArrayInterpreter(int_θdoubleMM(1:length(int_θdoubleMM)), (n_sample_pred,))
ζs1c = intm(ζs[1:length(int_θdoubleMM), :])
vcat(θP_true, θM_true)
histogram(exp.(ζs1c[:r0, :]))
histogram(exp.(ζs1c[:K2, :]))
histogram(exp.(ζs1c[:r1, :]))
histogram(exp.(ζs1c[:K1, :]))
# all parameters estimated to high (true not in cf bounds)
scatterplot(ζs1c[:r1, :], ζs1c[:K1, :])  # r1 and K1 strongly correlated (from θM)
scatterplot(ζs1c[:r0, :], ζs1c[:K2, :])  # r0 and K also correlated (from θP)
scatterplot(ζs1c[:r0, :], ζs1c[:K1, :])  # no correlation (modeled independent)

# TODO compare distributions to MC sample






