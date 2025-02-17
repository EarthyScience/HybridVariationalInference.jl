using Test
using HybridVariationalInference
using HybridVariationalInference: HybridVariationalInference as HVI
using StableRNGs
using Random
using Statistics
using ComponentArrays: ComponentArrays as CA
using Optimization
using OptimizationOptimisers # Adam
using UnicodePlots
using SimpleChains
using Flux
using MLUtils
using CUDA

rng = StableRNG(114)
scenario = NTuple{0, Symbol}()
scenario = (:use_Flux,)

#------ setup synthetic data and training data loader
(; xM, n_site, θP_true, θMs_true, xP, y_global_true, y_true, y_global_o, y_o, y_unc
) = gen_hybridcase_synthetic(rng, DoubleMM.DoubleMMCase(); scenario);
xM_cpu = xM
if :use_Flux ∈ scenario
    xM = CuArray(xM_cpu)
end
get_train_loader = (rng; n_batch, kwargs...) -> MLUtils.DataLoader((xM, xP, y_o, y_unc); 
    batchsize = n_batch, partial = false)
σ_o = exp(first(y_unc)/2)

# assign the train_loader, otherwise it eatch time creates another version of synthetic data
prob0 = HVI.update(HybridProblem(DoubleMM.DoubleMMCase(); scenario); get_train_loader)

#------- pointwise hybrid model fit
solver = HybridPointSolver(; alg = Adam(0.02), n_batch = 30)
#solver = HybridPointSolver(; alg = Adam(0.01), n_batch = 10)
#solver = HybridPointSolver(; alg = Adam(), n_batch = 200)
(; ϕ, resopt) = solve(prob0, solver; scenario,
    rng, callback = callback_loss(100), maxiters = 1200);
# update the problem with optimized parameters
prob0o = HVI.update(prob0; ϕg=cpu_ca(ϕ).ϕg, θP=cpu_ca(ϕ).θP)
y_pred_global, y_pred, θMs = gf(prob0o, xM, xP; scenario);
scatterplot(θMs_true[1,:], θMs[1,:])
scatterplot(θMs_true[2,:], θMs[2,:])

# do a few steps without minibatching, 
#   by providing the data rather than the DataLoader
solver1 = HybridPointSolver(; alg = Adam(0.01), n_batch = n_site)
(; ϕ, resopt) = solve(prob0o, solver1; scenario, rng, 
    callback = callback_loss(20), maxiters = 600);
prob1o = HVI.update(prob0o; ϕg=cpu_ca(ϕ).ϕg, θP=cpu_ca(ϕ).θP);
y_pred_global, y_pred, θMs = gf(prob1o, xM, xP; scenario);
scatterplot(θMs_true[1,:], θMs[1,:])
scatterplot(θMs_true[2,:], θMs[2,:])
prob1o.θP
scatterplot(vec(y_true), vec(y_pred))

# still overestimating θMs

() -> begin # with more iterations?
    prob2 = prob1o
    (; ϕ, resopt) = solve(prob2, solver1; scenario, rng, 
        callback = callback_loss(20), maxiters = 600);
    prob2o = update(prob2; ϕg=ϕ.ϕg, θP=ϕ.θP)
    y_pred_global, y_pred, θMs = gf(prob2o, xM, xP);
    prob2o.θP
end


#----------- fit g to true θMs 
() -> begin
    # and fit gf starting from true parameters
    prob = prob0
    g, ϕg0_cpu = get_hybridproblem_MLapplicator(prob; scenario);
    ϕg0 = (:use_Flux ∈ scenario) ? CuArray(ϕg0_cpu) : ϕg0_cpu
    (; transP, transM) = get_hybridproblem_transforms(prob; scenario)

    function loss_g(ϕg, x, g, transM; gpu_handler = HVI.default_GPU_DataHandler)
        ζMs = g(x, ϕg) # predict the log of the parameters
        ζMs_cpu = gpu_handler(ζMs)
        θMs = reduce(hcat, map(transM, eachcol(ζMs_cpu))) # transform each column
        loss = sum(abs2, θMs .- θMs_true)
        return loss, θMs
    end
    loss_g(ϕg0, xM, g, transM)

    optf = Optimization.OptimizationFunction((ϕg, p) -> loss_g(ϕg, xM, g, transM)[1],
        Optimization.AutoZygote())
    optprob = Optimization.OptimizationProblem(optf, ϕg0);
    res = Optimization.solve(optprob, Adam(0.015), callback = callback_loss(100), maxiters = 2000);

    ϕg_opt1 = res.u;
    l1, θMs = loss_g(ϕg_opt1, xM, g, transM)
    #scatterplot(θMs_true[1,:], θMs[1,:])
    scatterplot(θMs_true[2,:], θMs[2,:]) # able to fit θMs[2,:]

    prob3 = HVI.update(prob0, ϕg = Array(ϕg_opt1), θP = θP_true)
    solver1 = HybridPointSolver(; alg = Adam(0.01), n_batch = n_site)
    (; ϕ, resopt) = solve(prob3, solver1; scenario, rng, 
        callback = callback_loss(50), maxiters = 600);
    prob3o = HVI.update(prob3; ϕg=cpu_ca(ϕ).ϕg, θP=cpu_ca(ϕ).θP)
    y_pred_global, y_pred, θMs = gf(prob3o, xM, xP; scenario);
    scatterplot(θMs_true[2,:], θMs[2,:])
    prob3o.θP
    scatterplot(vec(y_true), vec(y_pred))
    scatterplot(vec(y_true), vec(y_o))
    scatterplot(vec(y_pred), vec(y_o))

    () -> begin # optimized loss is indeed lower than with true parameters
        int_ϕθP = ComponentArrayInterpreter(CA.ComponentVector(
            ϕg = 1:length(prob0.ϕg), θP = prob0.θP))
        loss_gf = get_loss_gf(prob0.g, prob0.transM, prob0.f, Float32[], int_ϕθP)
        loss_gf(vcat(prob3.ϕg, prob3.θP), xM, xP, y_o, y_unc)[1]
        loss_gf(vcat(prob3o.ϕg, prob3o.θP), xM, xP, y_o, y_unc)[1]
        #
        loss_gf(vcat(prob2o.ϕg, prob2o.θP), xM, xP, y_o, y_unc)[1]
    end
end
    
#----------- Hybrid Variational inference: HVI

using MLUtils
import Zygote

using CUDA
using Bijectors

solver = HybridPosteriorSolver(; alg = Adam(0.01), n_batch = 60, n_MC = 3)
#solver = HybridPointSolver(; alg = Adam(), n_batch = 200)
(; ϕ, θP, resopt) = solve(prob0o, solver; scenario,
    rng, callback = callback_loss(100), maxiters = 800);
# update the problem with optimized parameters
prob1o = HVI.update(prob0o; ϕg=cpu_ca(ϕ).ϕg, θP=θP)
y_pred_global, y_pred, θMs = gf(prob1o, xM, xP; scenario);
scatterplot(θMs_true[1,:], θMs[1,:])
scatterplot(θMs_true[2,:], θMs[2,:])
hcat(θP_true, θP) # all parameters overestimated


() -> begin
    #n_covar = get_hybridproblem_n_covar(prob; scenario)
    #, n_batch, n_θM, n_θP) = get_hybridproblem_sizes(prob; scenario)

    n_covar = size(xM, 1)

    #----- fit g to θMs_true
    g, ϕg0 = get_hybridproblem_MLapplicator(prob; scenario);
    (; transP, transM) = get_hybridproblem_transforms(prob; scenario)

    function loss_g(ϕg, x, g, transM)
        ζMs = g(x, ϕg) # predict the log of the parameters
        θMs = reduce(hcat, map(transM, eachcol(ζMs))) # transform each column
        loss = sum(abs2, θMs .- θMs_true)
        return loss, θMs
    end
    loss_g(ϕg0, xM, g, transM)

    optf = Optimization.OptimizationFunction((ϕg, p) -> loss_g(ϕg, xM, g, transM)[1],
        Optimization.AutoZygote())
    optprob = Optimization.OptimizationProblem(optf, ϕg0);
    res = Optimization.solve(optprob, Adam(0.02), callback = callback_loss(100), maxiters = 800);

    ϕg_opt1 = res.u;
    l1, θMs_pred = loss_g(ϕg_opt1, xM, g, transM)
    scatterplot(vec(θMs_true), vec(θMs_pred))

    f = get_hybridproblem_PBmodel(prob; scenario)
    py = get_hybridproblem_neg_logden_obs(prob; scenario)

    #----------- fit g and θP to y_o
    () -> begin
        # end2end inversion

        int_ϕθP = ComponentArrayInterpreter(CA.ComponentVector(
            ϕg = 1:length(ϕg0), θP = par_templates.θP))
        p = p0 = vcat(ϕg0, par_templates.θP .* 0.9)  # slightly disturb θP_true

        # Pass the site-data for the batches as separate vectors wrapped in a tuple
        train_loader = MLUtils.DataLoader((xM, xP, y_o, y_unc), batchsize = n_batch)

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
end

#---------- HVI
n_MC = 3
(; transP, transM) = get_hybridproblem_transforms(prob; scenario)
FT = get_hybridproblem_float_type(prob; scenario)

(; ϕ, transPMs_batch, interpreters, get_transPMs, get_ca_int_PMs) = init_hybrid_params(
    θP_true, θMs_true[:, 1], ϕg_opt1, n_batch; transP, transM);
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

ϕ_ini0 = ζ = reduce(
    vcat, (
        ϕ_true[[:μP]] .* FT(0.001), CA.ComponentVector(ϕg = ϕg0), ϕ_true[[:unc]])) # scratch
#
ϕ_ini = ζ = reduce(
    vcat, (
        ϕ_true[[:μP]] .- FT(0.1), ϕ_true[[:ϕg]] .* FT(1.1), ϕ_true[[:unc]])) # slight disturbance
# hardcoded from HMC inversion
ϕ_ini.unc.coef_logσ2_logMs = [-5.769 -3.501; -0.01791 0.007951]
ϕ_ini.unc.logσ2_logP = CA.ComponentVector(r0 = -8.997, K2 = -5.893)
mean_σ_o_MC = 0.006042

ϕ = CA.getdata(ϕ_ini) |> Flux.gpu;
xM_gpu = xM |> Flux.gpu;
scenario_flux = (scenario..., :use_Flux)
g_flux, _ = get_hybridproblem_MLapplicator(prob; scenario = scenario_flux);

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

function fcost(ϕ, xM, y_o, y_unc)
    neg_elbo_transnorm_gf(rng, CA.getdata(ϕ), g_flux, transPMs_batch, f, py,
        xM, xP, y_o, y_unc, map(get_concrete, interpreters);
        n_MC = 8)
end
fcost(ϕ, xM_gpu[:, 1:n_batch], y_o[:, 1:n_batch], y_unc[:, 1:n_batch])
#Zygote.gradient(fcost, ϕ) |> cpu;
gr = Zygote.gradient(fcost,
    CA.getdata(ϕ), CA.getdata(xM_gpu[:, 1:n_batch]),
    CA.getdata(y_o[:, 1:n_batch]), CA.getdata(y_unc[:, 1:n_batch]));
gr_c = CA.ComponentArray(gr[1] |> Flux.cpu, CA.getaxes(ϕ_ini)...)

train_loader = MLUtils.DataLoader((xM_gpu, xP, y_o, y_unc), batchsize = n_batch)
#train_loader = get_hybridproblem_train_dataloader(prob, rng; scenario = (scenario..., :use_Flux))

optf = Optimization.OptimizationFunction(
    (ϕ, data) -> begin
        xM, xP, y_o, y_unc = data
        fcost(ϕ, xM, y_o, y_unc)
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

hcat(log.(θP_true), ϕ_ini.μP, ζ_VIc.μP)
plt = scatterplot(vec(θMs_true), vec(exp.(ζMs_VI)))
#lineplot!(plt, 0.0, 1.1, identity)
# 
hcat(ϕ_ini.unc, ϕunc_VI) # need to compare to MC sample
# hard to estimate for original very small theta's but otherwise good

# test predicting correct obs-uncertainty of predictive posterior
n_sample_pred = 200

y_pred = predict_gf(rng, g_flux, f, res.u, xM_gpu, xP, interpreters;
    get_transPMs, get_ca_int_PMs, n_sample_pred);
size(y_pred) # n_obs x n_site, n_sample_pred

σ_o_post = dropdims(std(y_pred; dims = 3), dims = 3);
σ_o = exp.(y_unc[:, 1] / 2)

#describe(σ_o_post)
hcat(σ_o, fill(mean_σ_o_MC, length(σ_o)),
    mean(σ_o_post, dims = 2), sqrt.(mean(abs2, σ_o_post, dims = 2)))
# VI predicted uncertainty is smaller than HMC predicted one
mean_y_pred = map(mean, eachslice(y_pred; dims = (1, 2)))
#describe(mean_y_pred - y_o)
histogram(vec(mean_y_pred - y_true)) # predictions centered around y_o (or y_true)

# look at θP, θM1 of first site
intm_PMs_gen = get_ca_int_PMs(n_site)
ζs, _σ = HVI.generate_ζ(rng, g_flux, res.u, xM_gpu,
    (; interpreters..., PMs = intm_PMs_gen); n_MC = n_sample_pred);
ζs = ζs |> Flux.cpu;
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
