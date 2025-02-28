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
using MLDataDevices, CUDA, cuDNN, GPUArraysCore

rng = StableRNG(115)
scenario = NTuple{0, Symbol}()
scenario = (:omit_r0,)  # without omit_r0 ambiguous K2 estimated to high
scenario = (:use_Flux,)
scenario = (:use_Flux, :omit_r0)
# prob = DoubleMM.DoubleMMCase()

gdev = :use_Flux ∈ scenario ? gpu_device() : cpu_device()

#------ setup synthetic data and training data loader
(; xM, n_site, θP_true, θMs_true, xP, y_global_true, y_true, y_global_o, y_o, y_unc
) = gen_hybridcase_synthetic(rng, DoubleMM.DoubleMMCase(); scenario);
xM_cpu = xM; xM = xM_cpu |> gdev
get_train_loader = (rng; n_batch, kwargs...) -> MLUtils.DataLoader((xM, xP, y_o, y_unc);
    batchsize = n_batch, partial = false)
σ_o = exp(first(y_unc) / 2)

# assign the train_loader, otherwise it eatch time creates another version of synthetic data
prob0 = HVI.update(HybridProblem(DoubleMM.DoubleMMCase(); scenario); get_train_loader)

#------- pointwise hybrid model fit
solver = HybridPointSolver(; alg = Adam(0.01), n_batch = 30)
#solver = HybridPointSolver(; alg = Adam(0.01), n_batch = 10)
#solver = HybridPointSolver(; alg = Adam(), n_batch = 200)
n_epoch = 5
(; ϕ, resopt) = solve(prob0, solver; scenario,
    rng, callback = callback_loss(200), maxiters = n_site * n_epoch);
# update the problem with optimized parameters
prob0o = HVI.update(prob0; ϕg = cpu_ca(ϕ).ϕg, θP = cpu_ca(ϕ).θP)
y_pred_global, y_pred, θMs = gf(prob0o, xM, xP; scenario);
plt = scatterplot(θMs_true[1, :], θMs[1, :]);
lineplot!(plt, 0, 1)
scatterplot(θMs_true[2, :], θMs[2, :])
prob0o.θP
#scatterplot(vec(y_true), vec(y_o))
#scatterplot(vec(y_true), vec(y_pred))
histogram(vec(y_pred) - vec(y_true)) # predictions centered around y_o (or y_true)


# do a few steps without minibatching, 
#   by providing the data rather than the DataLoader
() -> begin
    solver1 = HybridPointSolver(; alg = Adam(0.01), n_batch = n_site)
    (; ϕ, resopt) = solve(prob0o, solver1; scenario, rng,
        callback = callback_loss(20), maxiters = 400)
    prob1o = HVI.update(prob0o; ϕg = cpu_ca(ϕ).ϕg, θP = cpu_ca(ϕ).θP)
    y_pred_global, y_pred, θMs = gf(prob1o, xM, xP; scenario)
    scatterplot(θMs_true[1, :], θMs[1, :])
    scatterplot(θMs_true[2, :], θMs[2, :])
    prob1o.θP
    scatterplot(vec(y_true), vec(y_pred))

    # still overestimating θMs and θP
end

() -> begin # with more iterations?
    prob2 = prob1o
    (; ϕ, resopt) = solve(prob2, solver1; scenario, rng,
        callback = callback_loss(20), maxiters = 600)
    prob2o = HVI.update(prob2; ϕg = collect(ϕ.ϕg), θP = ϕ.θP)
    y_pred_global, y_pred, θMs = gf(prob2o, xM, xP)
    prob2o.θP
end

#----------- fit g to true θMs 
() -> begin
    # and fit gf starting from true parameters
    prob = prob0
    g, ϕg0_cpu = get_hybridproblem_MLapplicator(prob; scenario)
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
    optprob = Optimization.OptimizationProblem(optf, ϕg0)
    res = Optimization.solve(
        optprob, Adam(0.015), callback = callback_loss(100), maxiters = 2000)

    ϕg_opt1 = res.u
    l1, θMs = loss_g(ϕg_opt1, xM, g, transM)
    #scatterplot(θMs_true[1,:], θMs[1,:])
    scatterplot(θMs_true[2, :], θMs[2, :]) # able to fit θMs[2,:]

    prob3 = HVI.update(prob0, ϕg = Array(ϕg_opt1), θP = θP_true)
    solver1 = HybridPointSolver(; alg = Adam(0.01), n_batch = n_site)
    (; ϕ, resopt) = solve(prob3, solver1; scenario, rng,
        callback = callback_loss(50), maxiters = 600)
    prob3o = HVI.update(prob3; ϕg = cpu_ca(ϕ).ϕg, θP = cpu_ca(ϕ).θP)
    y_pred_global, y_pred, θMs = gf(prob3o, xM, xP; scenario)
    scatterplot(θMs_true[2, :], θMs[2, :])
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
using Bijectors

probh = prob0o  # start from point optimized to infer uncertainty
#probh = prob1o  # start from point optimized to infer uncertainty
#probh = prob0  # start from no information
solver2 = HybridPosteriorSolver(; alg = Adam(0.01), n_batch = 40, n_MC = 3)
#solver = HybridPointSolver(; alg = Adam(), n_batch = 200)
n_epoch = 3
(; ϕ, θP, resopt, interpreters) = solve(probh, solver2; scenario,
    rng, callback = callback_loss(200), maxiters = n_site * n_epoch);
# update the problem with optimized parameters
prob1o = HVI.update(prob0o; ϕg = cpu_ca(ϕ).ϕg, θP = θP)
() -> begin # prediction with fitted parameters  (should be smaller than mean)
    y_pred_global, y_pred2, θMs = gf(prob1o, xM, xP; scenario);
    scatterplot(θMs_true[1, :], θMs[1, :])
    scatterplot(θMs_true[2, :], θMs[2, :])
    hcat(θP_true, θP) # all parameters overestimated
    histogram(vec(y_pred2) - vec(y_true)) # predicts an unsymmytric distribution
end

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

ζ_VIc = interpreters.μP_ϕg_unc(resopt.u |> Flux.cpu)
#ζMs_VI = g_flux(xM_gpu, ζ_VIc.ϕg |> Flux.gpu) |> Flux.cpu
ϕunc_VI = interpreters.unc(ζ_VIc.unc)
ϕunc_VI.ρsM
exp.(ϕunc_VI.logσ2_logP)
exp.(ϕunc_VI.coef_logσ2_logMs[1,:])


# test predicting correct obs-uncertainty of predictive posterior
n_sample_pred = 400
(; θ, y) = predict_gf(rng, prob1o, xM, xP; scenario, n_sample_pred);
size(y) # n_obs x n_site, n_sample_pred
size(θ)  # n_θP + n_site * n_θM x n_sample
σ_o_post = dropdims(std(y; dims = 3), dims = 3);
σ_o = exp.(y_unc[:, 1] / 2)

#describe(σ_o_post)
hcat(σ_o, # fill(mean_σ_o_MC, length(σ_o)),
    mean(σ_o_post, dims = 2), sqrt.(mean(abs2, σ_o_post, dims = 2)))
hcat(σ_o, fill(mean_σ_o_MC, length(σ_o)),
    mean(σ_o_post, dims = 2), sqrt.(mean(abs2, σ_o_post, dims = 2)))
# VI predicted uncertainty is smaller than HMC predicted one
mean_y_pred = map(mean, eachslice(y; dims = (1, 2)))
#describe(mean_y_pred - y_o)
histogram(vec(mean_y_pred) - vec(y_true)) # predictions centered around y_o (or y_true)
plt = scatterplot(vec(y_true), vec(mean_y_pred)); lineplot!(plt, 0, 2)
mean(mean_y_pred - y_true) # still ok

mean_θ = CA.ComponentVector(mean(CA.getdata(θ); dims=2)[:,1], CA.getaxes(θ[:,1])[1])
plt = scatterplot(θMs_true[1, :], mean_θ.Ms[1, :]); lineplot!(plt, 0, 1)
plt = scatterplot(θMs_true[2, :], mean_θ.Ms[2, :])
#lineplot!(plt, 0, 1)




# look at one observation
scatterplot(y_true[1,1], y_obs[1,1])

# look at θP, θM1 of first site
θPM = vcat(θP_true, θMs_true[:, 1])
intm = ComponentArrayInterpreter(θPM, (n_sample_pred,))
θ1c = intm(θ[1:length(θPM), :])
θPM
#histogram((θ1c[:r0, :]))
histogram((θ1c[:K2, :]))
histogram((θ1c[:r1, :]))
histogram((θ1c[:K1, :])) 
# overestimates r1 and underestimates K1
# all parameters estimated to high (true not in cf bounds)
scatterplot(θ1c[:r1, :], θ1c[:K1, :])  # r1 and K1 strongly correlated (from θM)
scatterplot(θ1c[:r0, :], θ1c[:K2, :])  # r0 and K also correlated (from θP)
scatterplot(θ1c[:r0, :], θ1c[:K1, :])  # no correlation (modeled independent)

# TODO compare distributions to MC sample
