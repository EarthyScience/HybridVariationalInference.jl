using Test
using HybridVariationalInference
using HybridVariationalInference: HybridVariationalInference as HVI
using StableRNGs
using Random
using Statistics
using ComponentArrays: ComponentArrays as CA

using SimpleChains
import Flux # to allow for FluxMLEngine and cpu()
using MLUtils
import Zygote

using CUDA
using TransformVariables
using OptimizationOptimisers
using UnicodePlots

const case = DoubleMM.DoubleMMCase()
const MLengine = Val(nameof(SimpleChains))
const FluxMLengine = Val(nameof(Flux))
scenario = (:default,)
rng = StableRNG(111)

par_templates = get_hybridcase_par_templates(case; scenario)

(; n_covar, n_site, n_batch, n_θM, n_θP) = get_hybridcase_sizes(case; scenario)

(; xM, θP_true, θMs_true, xP, y_global_true, y_true, y_global_o, y_o, σ_o
) = gen_hybridcase_synthetic(case, rng; scenario);

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
res = Optimization.solve(optprob, Adam(0.02), callback = callback_loss(100), maxiters = 800);

ϕg_opt1 = res.u;
loss_g(ϕg_opt1, xM, g)
scatterplot(vec(θMs_true), vec(loss_g(ϕg_opt1, xM, g)[2]))
@test cor(vec(θMs_true), vec(loss_g(ϕg_opt1, xM, g)[2])) > 0.9

f = gen_hybridcase_PBmodel(case; scenario)

#----------- fit g and θP to y_o
() -> begin
    # end2end inversion

    int_ϕθP = ComponentArrayInterpreter(CA.ComponentVector(
        ϕg = 1:length(ϕg0), θP = par_templates.θP))
    p = p0 = vcat(ϕg0, par_templates.θP .* 0.9)  # slightly disturb θP_true

    # Pass the site-data for the batches as separate vectors wrapped in a tuple
    train_loader = MLUtils.DataLoader((xM, xP, y_o), batchsize = n_batch)

    loss_gf = get_loss_gf(g, f, y_global_o, int_ϕθP)
    l1 = loss_gf(p0, train_loader.data...)[1]

    optf = Optimization.OptimizationFunction((ϕ, data) -> loss_gf(ϕ, data...)[1],
        Optimization.AutoZygote())
    optprob = OptimizationProblem(optf, p0, train_loader)

    res = Optimization.solve(
        optprob, Adam(0.02), callback = callback_loss(100), maxiters = 1000)

    l1, y_pred_global, y_pred, θMs = loss_gf(res.u, train_loader.data...)
    scatterplot(vec(θMs_true), vec(θMs))
    scatterplot(log.(vec(θMs_true)), log.(vec(θMs)))
    scatterplot(vec(y_pred), vec(y_o))
    hcat(par_templates.θP, int_ϕθP(res.u).θP)
end

#---------- HVI
logσ2y = 2 .* log.(σ_o)
n_MC = 3
(; ϕ, transPMs_batch, interpreters, get_transPMs, get_ca_int_PMs) = init_hybrid_params(
    θP_true, θMs_true[:, 1], ϕg_opt1, n_batch; transP = asℝ₊, transM = asℝ₊);
ϕ_true = ϕ

() -> begin
    coef_logσ2_logMs = [-5.769 -3.501; -0.01791 0.007951]
    logσ2_logP = CA.ComponentVector(r0 = -8.997, K2 = -5.893)
    mean_σ_o_MC = 0.006042

    # correlation matrices
    ρsP = zeros(sum(1:(n_θP - 1)))
    ρsM = zeros(sum(1:(n_θM - 1)))

    ϕunc = CA.ComponentVector(;
        logσ2_logP = logσ2_logP,
        coef_logσ2_logMs = coef_logσ2_logMs,
        ρsP,
        ρsM)
    int_unc = ComponentArrayInterpreter(ϕunc)

    # for a conservative uncertainty assume σ2=1e-10 and no relationship with magnitude
    ϕunc0 = CA.ComponentVector(;
        logσ2_logP = fill(-10.0, n_θP),
        coef_logσ2_logMs = reduce(hcat, ([-10.0, 0.0] for _ in 1:n_θM)),
        ρsP,
        ρsM)

    transPMs_batch = as(
        (P = as(Array, asℝ₊, n_θP),
        Ms = as(Array, asℝ₊, n_θM, n_batch)))
    transPMs_allsites = as(
        (P = as(Array, asℝ₊, n_θP),
        Ms = as(Array, asℝ₊, n_θM, n_site)))

    n_ϕg = length(ϕg_opt1)
    ϕt_true = θ = CA.ComponentVector(;
        μP = θP_true,
        ϕg = ϕg_opt1,
        unc = ϕunc)
    trans_gu = as(
        (μP = as(Array, asℝ₊, n_θP),
        ϕg = as(Array, n_ϕg),
        unc = as(Array, length(ϕunc))))
    trans_g = as(
        (μP = as(Array, asℝ₊, n_θP),
        ϕg = as(Array, n_ϕg)))

    #const 
    int_PMs_batch = ComponentArrayInterpreter(CA.ComponentVector(; θP = θP_true,
        θMs = CA.ComponentMatrix(
            zeros(n_θM, n_batch), first(CA.getaxes(θMs_true)), CA.Axis(i = 1:n_batch))))

    interpreters = interpreters_g = map(get_concrete,
        (;
            μP_ϕg_unc = ComponentArrayInterpreter(ϕt_true),
            PMs = int_PMs_batch,
            unc = ComponentArrayInterpreter(ϕunc)
        ))

    ϕ_true = inverse_ca(trans_gu, ϕt_true)
end

ϕ_ini0 = ζ = vcat(ϕ_true[:μP] .* 0.0, ϕg0, ϕ_true[[:unc]]); # scratch
#
# true values
ϕ_ini = ζ = vcat(ϕ_true[[:μP, :ϕg]] .* 1.2, ϕ_true[[:unc]]); # slight disturbance
# hardcoded from HMC inversion
ϕ_ini.unc.coef_logσ2_logMs = [-5.769 -3.501; -0.01791 0.007951]
ϕ_ini.unc.logσ2_logP = CA.ComponentVector(r0 = -8.997, K2 = -5.893)
mean_σ_o_MC = 0.006042

# test cost function and gradient
() -> begin
    neg_elbo_transnorm_gf(rng, g, f, ϕ_true, y_o[:, 1:n_batch], xM[:, 1:n_batch],
        transPMs_batch, map(get_concrete, interpreters);
        n_MC = 8, logσ2y)
    Zygote.gradient(
        ϕ -> neg_elbo_transnorm_gf(
            rng, g, f, ϕ, y_o[:, 1:n_batch], xM[:, 1:n_batch],
            transPMs_batch, interpreters; n_MC = 8, logσ2y),
        CA.getdata(ϕ_true))
end

# optimize using SimpleChains
() -> begin
    train_loader = MLUtils.DataLoader((xM, y_o), batchsize = n_batch)

    optf = Optimization.OptimizationFunction(
        (ϕ, data) -> begin
            xM, y_o = data
            neg_elbo_transnorm_gf(
                rng, g, f, ϕ, y_o, xM, transPMs_batch,
                map(get_concrete, interpreters_g); n_MC = 5, logσ2y)
        end,
        Optimization.AutoZygote())
    optprob = Optimization.OptimizationProblem(optf, CA.getdata(ϕ_ini), train_loader)
    res = Optimization.solve(
        optprob, Optimisers.Adam(0.02), callback = callback_loss(50), maxiters = 800)
    #optprob = Optimization.OptimizationProblem(optf, ϕ_ini0);
    #res = Optimization.solve(optprob, Adam(0.02), callback=callback_loss(50), maxiters=1_400);
end

ϕ = ϕ_true |> Flux.gpu;
xM_gpu = xM |> Flux.gpu;
g_flux, ϕg0_flux_cpu = gen_hybridcase_MLapplicator(case, FluxMLengine; scenario);

# otpimize using LUX
() -> begin
    #using Lux
    g_lux = Lux.Chain(
        # dense layer with bias that maps to 8 outputs and applies `tanh` activation
        Lux.Dense(n_covar => n_covar * 4, tanh),
        Lux.Dense(n_covar * 4 => n_covar * 4, logistic),
        # dense layer without bias that maps to n outputs and `identity` activation
        Lux.Dense(n_covar * 4 => n_θM, identity, use_bias = false)
    )
    ps, st = Lux.setup(Random.default_rng(), g_lux)
    ps_ca = CA.ComponentArray(ps) |> gpu
    st = st |> gpu
    g_luxs = StatefulLuxLayer{true}(g_lux, nothing, st)
    g_luxs(xM_gpu[:, 1:n_batch], ps_ca)
    ax_g = CA.getaxes(ps_ca)
    g_luxs(xM_gpu[:, 1:n_batch], CA.ComponentArray(ϕ.ϕg, ax_g))
    interpreters = (interpreters..., ϕg = ComponentArrayInterpreter(ps_ca))
    ϕg = CA.ComponentArray(ϕ.ϕg, ax_g)
    ϕgc = interpreters.ϕg(ϕ.ϕg)
    g_flux = g_luxs
end

function fcost(ϕ)
    neg_elbo_transnorm_gf(rng, g_flux, f, CA.getdata(ϕ), y_o[:, 1:n_batch],
        xM_gpu[:, 1:n_batch], transPMs_batch, map(get_concrete, interpreters);
        n_MC = 8, logσ2y = logσ2y)
end
fcost(ϕ)
#Zygote.gradient(fcost, ϕ) |> cpu;
gr = Zygote.gradient(fcost, CA.getdata(ϕ));
gr_c = CA.ComponentArray(gr[1] |> Flux.cpu, CA.getaxes(ϕ)...)

train_loader = MLUtils.DataLoader((xM_gpu, y_o), batchsize = n_batch)

optf = Optimization.OptimizationFunction(
    (ϕ, data) -> begin
        xM, y_o = data
        fcost(ϕ)
        # neg_elbo_transnorm_gf(
        #     rng, g_flux, f, ϕ, y_o, xM, transPMs_batch,
        #     map(get_concrete, interpreters); n_MC = 5, logσ2y)
    end,
    Optimization.AutoZygote())
optprob = Optimization.OptimizationProblem(
    optf, CA.getdata(ϕ_ini) |> Flux.gpu, train_loader);
res = res_gpu = Optimization.solve(
    optprob, Optimisers.Adam(0.02), callback = callback_loss(50), maxiters = 800);

# start from zero 
() -> begin
    optprob = Optimization.OptimizationProblem(
        optf, CA.getdata(ϕ_ini0) |> Flux.gpu, train_loader)
    res = res_gpu = Optimization.solve(
        optprob, Optimisers.Adam(0.02), callback = callback_loss(50), maxiters = 4_000)
end

ζ_VIc = interpreters.μP_ϕg_unc(res.u |> Flux.cpu)
ζMs_VI = g_flux(xM_gpu, ζ_VIc.ϕg |> Flux.gpu) |> Flux.cpu
ϕunc_VI = interpreters.unc(ζ_VIc.unc)

hcat(θP_true, exp.(ζ_VIc.μP))
plt = scatterplot(vec(θMs_true), vec(exp.(ζMs_VI)))
#lineplot!(plt, 0.0, 1.1, identity)
# 
hcat(ϕ_ini.unc, ϕunc_VI) # need to compare to MC sample
# hard to estimate for original very small theta's but otherwise good

# test predicting correct obs-uncertainty of predictive posterior
# TODO reuse g_flux rather than g
n_sample_pred = 200
y_pred = predict_gf(rng, g_flux, f, res.u, xM_gpu, interpreters;
    get_transPMs, get_ca_int_PMs, n_sample_pred);
size(y_pred) # n_obs x n_site, n_sample_pred

σ_o_post = dropdims(std(y_pred; dims = 3), dims=3)

#describe(σ_o_post)
hcat(σ_o, fill(mean_σ_o_MC, length(σ_o)),
    mean(σ_o_post, dims = 2), sqrt.(mean(abs2, σ_o_post, dims = 2)))
mean_y_pred = map(mean, eachslice(y_pred; dims = (1, 2)))
#describe(mean_y_pred - y_o)
histogram(vec(mean_y_pred - y_true)) # predictions centered around y_o (or y_true)

# look at θP, θM1 of first site
θPM = vcat(θP_true, θMs_true[:, 1])
intm = ComponentArrayInterpreter(θPM, (n_sample_pred,))
ζs1c = intm(ζs[1:length(θPM), :])
θPM
histogram(exp.(ζs1c[:r0, :]))
histogram(exp.(ζs1c[:K2, :]))
histogram(exp.(ζs1c[:r1, :]))
histogram(exp.(ζs1c[:K1, :]))
# all parameters estimated to high (true not in cf bounds)
scatterplot(ζs1c[:r1, :], ζs1c[:K1, :])  # r1 and K1 strongly correlated (from θM)
scatterplot(ζs1c[:r0, :], ζs1c[:K2, :])  # r0 and K also correlated (from θP)
scatterplot(ζs1c[:r0, :], ζs1c[:K1, :])  # no correlation (modeled independent)

# TODO compare distributions to MC sample
