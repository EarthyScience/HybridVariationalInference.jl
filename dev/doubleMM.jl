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
import MLDataDevices, CUDA, cuDNN, GPUArraysCore

rng = StableRNG(115)
scenario = NTuple{0, Symbol}()
scenario = (:omit_r0,)  # without omit_r0 ambiguous K2 estimated to high
scenario = (:use_Flux, :use_gpu)
scenario = (:use_Flux, :use_gpu, :omit_r0, :few_sites)
scenario = (:use_Flux, :use_gpu, :omit_r0, :few_sites, :covarK2)
scenario = (:use_Flux, :use_gpu, :omit_r0, :sites20, :covarK2)
scenario = (:use_Flux, :use_gpu, :omit_r0)
scenario = (:use_Flux, :use_gpu, :omit_r0, :covarK2)
# prob = DoubleMM.DoubleMMCase()

gdev = :use_gpu ∈ scenario ? gpu_device() : identity
cdev = gdev isa MLDataDevices.AbstractGPUDevice ? cpu_device() : identity

#------ setup synthetic data and training data loader
prob0_ = HybridProblem(DoubleMM.DoubleMMCase(); scenario);
(; xM, θP_true, θMs_true, xP, y_global_true, y_true, y_global_o, y_o, y_unc
) = gen_hybridproblem_synthetic(rng, DoubleMM.DoubleMMCase(); scenario);
n_site, n_batch = get_hybridproblem_n_site_and_batch(prob0_; scenario)
ζP_true, ζMs_true = log.(θP_true), log.(θMs_true)
i_sites = 1:n_site
n_site, n_batch = get_hybridproblem_n_site_and_batch(prob0_; scenario)
train_dataloader = MLUtils.DataLoader(
    (xM, xP, y_o, y_unc, 1:n_site);
    batchsize = n_batch, partial = false)
σ_o = exp.(y_unc[:, 1] / 2)
# assign the train_loader, otherwise it eatch time creates another version of synthetic data
prob0 = HVI.update(prob0_; train_dataloader);
#tmp = HVI.get_hybridproblem_ϕunc(prob0; scenario)

#------- pointwise hybrid model fit
solver_point = HybridPointSolver(; alg = OptimizationOptimisers.Adam(0.01))
#solver_point = HybridPointSolver(; alg = Adam(0.01), n_batch = 30)
#solver_point = HybridPointSolver(; alg = Adam(0.01), n_batch = 10)
#solver_point = HybridPointSolver(; alg = Adam(), n_batch = 200)
n_batches_in_epoch = n_site ÷ n_batch
n_epoch = 80
(; ϕ, resopt, probo) = solve(prob0, solver_point; scenario,
    rng, callback = callback_loss(n_batches_in_epoch * 10),
    maxiters = n_batches_in_epoch * n_epoch);
# update the problem with optimized parameters
prob0o = probo;
y_pred_global, y_pred, θMs = gf(prob0o, scenario);
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

() -> begin #----------- fit g to true θMs 
    # and fit gf starting from true parameters
    prob = prob0
    g, ϕg0_cpu = get_hybridproblem_MLapplicator(prob; scenario)
    ϕg0 = (:use_Flux ∈ scenario) ? gdev(ϕg0_cpu) : ϕg0_cpu
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
        loss_gf = get_loss_gf(prob0.g, prob0.transM, prob0.transP, prob0.f, Float32[], int_ϕθP)
        loss_gf(vcat(prob3.ϕg, prob3.θP), xM, xP, y_o, y_unc, i_sites)[1]
        loss_gf(vcat(prob3o.ϕg, prob3o.θP), xM, xP, y_o, y_unc, i_sites)[1]
        #
        loss_gf(vcat(prob2o.ϕg, prob2o.θP), xM, xP, y_o, y_unc, i_sites)[1]
    end
end

#----------- Hybrid Variational inference: HVI
using MLUtils
import Zygote
using Bijectors

probh = prob0o  # start from point optimized to infer uncertainty
#probh = prob1o  # start from point optimized to infer uncertainty
#probh = prob0  # start from no information
solver_post = HybridPosteriorSolver(;
    alg = OptimizationOptimisers.Adam(0.01), n_MC = 3)
#solver_point = HybridPointSolver(; alg = Adam(), n_batch = 200)
n_batches_in_epoch = n_site ÷ n_batch
n_epoch = 40
(; ϕ, θP, resopt, interpreters, probo) = solve(probh, solver_post; scenario,
    rng, callback = callback_loss(n_batches_in_epoch * 5),
    maxiters = n_batches_in_epoch * n_epoch,
    θmean_quant = 0.05);
#probh.get_train_loader(;n_batch = 50, scenario)
# update the problem with optimized parameters, including uncertainty
prob1o = probo;
n_sample_pred = 400
#(; θ, y) = predict_gf(rng, prob1o, xM, xP; scenario, n_sample_pred);
(; θ, y) = predict_gf(rng, prob1o; scenario, n_sample_pred);
(θ1, y1) = (θ, y);

() -> begin # prediction with fitted parameters  (should be smaller than mean)
    y_pred_global, y_pred2, θMs = gf(prob1o, xM, xP; scenario)
    scatterplot(θMs_true[1, :], θMs[1, :])
    scatterplot(θMs_true[2, :], θMs[2, :])
    hcat(θP_true, θP) # all parameters overestimated
    histogram(vec(y_pred2) - vec(y_true)) # predicts an unsymmytric distribution
end

#----------- continue HVI without strong prior on θmean
prob2 = HVI.update(prob1o);  # copy
function fstate_ϕunc(state)
    u = state.u |> cpu
    #Main.@infiltrate_main
    uc = interpreters.μP_ϕg_unc(u)
    uc.unc.ρsM
end
n_epoch = 100
#n_epoch = 400
(; ϕ, θP, resopt, interpreters, probo) = solve(prob2, 
    HVI.update(solver_post, n_MC = 12);
    #HVI.update(solver_post, n_MC = 30);
    scenario, rng, maxiters = n_batches_in_epoch * n_epoch,
    callback = HVI.callback_loss_fstate(n_batches_in_epoch*5, fstate_ϕunc));
prob2o = probo;

() -> begin
    using JLD2
    #fname_probos = "intermediate/probos_$(last(scenario)).jld2"
    fname_probos = "intermediate/probos800_$(last(scenario)).jld2"
    JLD2.save(fname_probos, Dict("prob1o" => prob1o, "prob2o" => prob2o))
    tmp = JLD2.load(fname_probos)
    # get_train_loader function could not be restored with JLD2
    prob1o = HVI.update(tmp["prob1o"], get_train_loader = prob0.get_train_loader);
    prob2o = HVI.update(tmp["prob2o"], get_train_loader = prob0.get_train_loader);
end

() -> begin # load the non-covar scenario
    using JLD2
    #fname_probos = "intermediate/probos_$(last(scenario)).jld2"
    fname_probos = "intermediate/probos800_omit_r0.jld2"
    tmp = JLD2.load(fname_probos)
    # get_train_loader function could not be restored with JLD2
    prob1o_indep = HVI.update(tmp["prob1o"], get_train_loader = prob0.get_train_loader);
    prob2o_indep = HVI.update(tmp["prob2o"], get_train_loader = prob0.get_train_loader);
    # test predicting correct obs-uncertainty of predictive posterior
    n_sample_pred = 400
    (; θ, y, entropy_ζ) = predict_gf(rng, prob2o_indep, xM, xP; scenario, n_sample_pred);
    (θ2_indep, y2_indep) = (θ, y)
    #(θ2_indep, y2_indep) = (θ2, y2)  # workaround to use covarK2 when loading failed
end

() -> begin # otpimize using LUX
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
exp.(ϕunc_VI.coef_logσ2_logMs[1, :])

# test predicting correct obs-uncertainty of predictive posterior
n_sample_pred = 400
(; θ, y, entropy_ζ) = predict_gf(rng, prob2o; scenario, n_sample_pred);
(θ2, y2) = (θ, y)
size(y) # n_obs x n_site, n_sample_pred
size(θ)  # n_θP + n_site * n_θM x n_sample
σ_o_post = dropdims(std(y; dims = 3), dims = 3);
σ_o = exp.(y_unc[:, 1] / 2)

#describe(σ_o_post)
hcat(σ_o, # fill(mean_σ_o_MC, length(σ_o)),
    mean(σ_o_post, dims = 2), sqrt.(mean(abs2, σ_o_post, dims = 2)))
# VI predicted uncertainty is smaller than HMC predicted one
mean_y_pred = map(mean, eachslice(y; dims = (1, 2)));
#describe(mean_y_pred - y_o)
histogram(vec(mean_y_pred) - vec(y_true)) # predictions centered around y_o (or y_true)
plt = scatterplot(vec(y_true), vec(mean_y_pred));
lineplot!(plt, 0, 2)
mean(mean_y_pred - y_true) # still ok

mean_θ = CA.ComponentVector(mean(CA.getdata(θ); dims = 2)[:, 1], CA.getaxes(θ[:, 1])[1])
plt = scatterplot(θMs_true[1, :], mean_θ.Ms[1, :]);
lineplot!(plt, 0, 1)
plt = scatterplot(θMs_true[2, :], mean_θ.Ms[2, :])
histogram(θ[:P,:])
#scatter(fig[1,1], CA.getdata(θMs_true[1, :]), CA.getdata(mean_θ.Ms[1, :])); ablines!(fig[1,1], 0, 1)
#@usingany AlgebraOfGraphices
#fig = Figure()
#draw!(fig, data(DataFrame(x=CA.getdata(θMs_true[1, :]), y = CA.getdata(mean_θ.Ms[1, :]))) * mapping(:x, :y) * visual(Scatter))
#lineplot!(plt, 0, 1)
#plt = scatterplot(θMs_true[1, :], mean_θ.Ms[1, :] - θMs_true[1, :])
#plt = scatterplot(θMs_true[2, :], mean_θ.Ms[2, :] - θMs_true[2, :])
#plt = scatterplot(mean_θ.Ms[2, :] - θMs_true[2, :], mean_θ.Ms[1, :] - θMs_true[1, :])
# mode_θ = map(mode, eachrow(θ))
# plt = scatterplot(θMs_true[1, :], mode_θ.Ms[1, :]); lineplot!(plt, 0, 1)

() -> begin # compare elbo components for mean-constrained unconstrained
    # solver_MC = HybridPosteriorSolver(; alg = Adam(0.01), n_batch = 30, n_MC = 300)
    # n_batches_in_epoch = n_site ÷ solver_MC.n_batch
    # (; ϕ, θP, resopt, interpreters) = solve(prob2o, solver_MC; scenario,
    #     rng, callback = callback_loss(n_batches_in_epoch), maxiters = 14);
    # resopt.objective
    # (; ϕ, θP, resopt, interpreters) = solve(prob1o, solver_MC; scenario,
    #     rng, callback = callback_loss(n_batches_in_epoch), maxiters = 14);
    # resopt.objective
    # probo = prob3o = HVI.update(prob2; ϕg = cpu_ca(ϕ).ϕg, θP = θP, ϕunc = cpu_ca(ϕ).unc)

    solver_post2 = HVI.update(solver_post; n_MC = 30)
    #solver_post2 = HVI.update(solver_post; n_MC=3)
    n_rep = 30
    n_batchf = n_site
    n_batchf = n_site ÷ 10
    elbo = map(1:n_rep) do i_rep
        HVI.compute_elbo_components(
            prob2o, solver_post2; scenario, n_batch = n_batchf) |> collect
    end |> x -> stack(x; dims = 1)
    elbo_c = map(1:n_rep) do i_rep
        HVI.compute_elbo_components(
            prob1o, solver_post2; scenario, n_batch = n_batchf) |> collect
    end |> x -> stack(x; dims = 1)
    #@usingany AlgebraOfGraphics
    #@usingany CairoMakie
    #const AoG = AlgebraOfGraphics
    #@usingany DataFrames
    df = vcat(
        insertcols!(DataFrame(elbo, [:nLy, :ent, :nLmean_θ]), :scenario => "unconstrained"),
        insertcols!(DataFrame(elbo_c, [:nLy, :ent, :nLmean_θ]), :scenario => "θmean")
    ) |> x -> insertcols!(x, :elbo => -x.nLy + x.ent)
    plt = data(df) * mapping(:scenario, :elbo) * visual(BoxPlot)
    fig = draw(plt).figure
    save("tmp.svg", fig)
    save("elbo_boxplot.pdf", fig)
end

() -> begin # look at distribution of parameters, predictions, and likelihood and elob at one site
    function predict_site(probo, i_site)
        (; θ, y, entropy_ζ) = predict_gf(rng, probo, xM, xP; scenario, n_sample_pred)
        y_site = y[:, i_site, :]
        θMs_i = map(i_rep -> θ[:Ms, i_rep][:, i_site], axes(θ, 2))
        r1s = map(x -> x[1], θMs_i)
        # K1s = map(x -> x[2], θMs_i)
        # invt = map(Bijectors.inverse, get_hybridproblem_transforms(probo; scenario))
        # θPs = θ[:P,:]
        # ζPs = invt.transP.(θPs)
        # ζMs = invt.transM.(θMs_i)
        # _f = get_hybridproblem_PBmodel(probo; scenario)
        # y_site = map(eachcol(θPs), θMs_i) do θP, θM
        #     y_global, y = _f(θP, reshape(θM, (length(θM), 1)), xP[[i_site]])
        #     y[:,1]
        # end |> stack
        nLs = get_hybridproblem_neg_logden_obs(
            probo; scenario).(eachcol(y_site), Ref(y_o[:, i_site]), Ref(y_unc[:, i_site]))
        (; r1s, nLs, entropy_ζ, y_site)
    end
    i_site = 1
    (r1s, nLs, ent, y_site) = predict_site(prob2o, i_site)
    (r1sc, nLsc, entc, y_sitec) = predict_site(prob1o, i_site)
    mean(nLs), mean(nLsc)
    ent, entc
    # with larger uncertaintsy (higher entropy) in unconstrained cost much lower
    mean(nLs) - ent, mean(nLsc) - entc

    #@usingany CairoMakie
    #@usingany AlgebraOfGraphics
    const aog = AlgebraOfGraphics

    # especially uncertainty is put to r1 (compensated by larger K1)
    df = DataFrame(r1 = vcat(r1s, r1sc),
        scenario = vcat(fill.(["unconstrained", "meanθ"], n_sample_pred)...))
    plt = data(df) * mapping(:r1, color = :scenario => "Scenario") * aog.density()
    plth = mapping([θMs_true[:r1, 1]]) * visual(VLines; linestyle = :dash)
    fig = Figure(; size = (640, 480))
    fig = Figure(; size = (320, 240))
    gp = fig[1, 1]
    fd = draw!(gp, plt + plth)
    legend!(
        gp, fd; tellwidth = false, halign = :right, valign = :top, margin = (10, 10, 10, 10))
    save("r1_density.pdf", fig)
    save("tmp.svg", fig)

    # observations are matched similarly well
    # with larger uncertainty the right-skewed shape of K1 leads to left-skewed y
    df = DataFrame(
        y = vcat(vec(y_site .- y_true[:, i_site]), vec(y_sitec .- y_true[:, i_site])),
        scenario = vcat(fill.(["unconstrained", "meanθ"], n_sample_pred * size(y_o, 1))...))
    plt = data(df) *
          mapping(:y => "y_predicted - y_observed", color = :scenario => "Scenario") *
          aog.density()
    plth = mapping([0.0]) * visual(VLines; linestyle = :dash)
    fig = Figure(; size = (640, 480))
    gp = fig[1, 1]
    fg = draw!(gp, plt + plth)
    legend!(
        gp, fg; tellwidth = false, halign = :right, valign = :top, margin = (10, 10, 10, 10))
    save("ys_density.pdf", fig)
    save("tmp.svg", fig)

    #slightly worse (higher) negLogLikelihood
    df = DataFrame(nL = vcat(nLs, nLsc),
        scenario = vcat(fill.(["unconstrained", "meanθ"], n_sample_pred)...))
    plt = data(df) * mapping(:nL => "-logDensity", color = :scenario => "Scenario") *
          aog.density()
    fig = Figure()
    gp = fig[1, 1]
    fg = draw!(gp, plt)
    legend!(
        gp, fg; tellwidth = false, halign = :right, valign = :top, margin = (10, 10, 10, 10))
    save("negLogDensity.pdf", fig)
    save("tmp.svg", fig)
end

() -> begin # look at θP, θM1 of first site
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
end

#---- do an DEMC inversion of the PBM model with parameters at log-scale
using DistributionFits
using PDMats
using Turing
using MCMCChains
# construct a prior on log scale that ranges roughly across 1e.3 to 10
prior_ζ = fit(Normal, @qp_ll(log(1e-2)), @qp_uu(log(10)))
prior_ζn = (n) -> MvNormal(fill(prior_ζ.μ, n), PDiagMat(fill(abs2(prior_ζ.σ), n)))
prior_ζn(3)
prob = HVI.update(prob0o);

(; θM, θP) = get_hybridproblem_par_templates(prob; scenario)
n_θM, n_θP = length.((θM, θP))
f = get_hybridproblem_PBmodel(prob; scenario)

@model function fsites(y, ::Type{T} = Float64; f, n_θP, n_θM, σ_o) where {T}
    n_obs, n_site = size(y)
    prior_ζP = prior_ζn(n_θP)
    prior_ζM_sites = fill(prior_ζn(n_site), n_θM)
    ζP ~ prior_ζP #MvNormal(n_θP, 10.0)
    # CAUTION: order of vectorizing matrix depends on order of ~
    # need to assign each variable in first site first, then second site, ...
    #   need to construct different MvNormal prior if std differs by variable
    # or need to take care when extracting samples and when constructing chains
    ζMs = Matrix{T}(undef, n_θM, n_site)
    # the first loop vectorizes θMs by columns but is much slower
    # for i_site in 1:n_site
    #     ζMs[:, i_site] ~ prior_ζn(n_θM) #MvNormal(n_site, 10.0)
    # end
    # this loop is faster, but vectorizes θMs by rows in parameter vector
    for i_par in 1:n_θM
        ζMs[i_par, :] ~ prior_ζM_sites[i_par]
    end
    # assume σ_o known, see f_MM
    #σ_o ~ truncated(Normal(0, 1); lower=0)
    #TODO specify with transPM
    # @show ζP
    # Main.@infiltrate_main # step to second time 
    y_pred = f(exp.(ζP), exp.(ζMs), xP)[2] # first is global
    for i_obs in 1:n_obs
        y[i_obs, :] ~ MvNormal(y_pred[i_obs, :], σ_o[i_obs]) # single value σ instead of variance
    end
    #Main.@infiltrate_main # step to second time 
    # θMs_MCc[:,:,1] # checking row- or column-order of θMs
    # exp.(ζMs)
    y_pred
end

model = fsites(y_o; f, n_θP, n_θM, σ_o)

# setup transformers and interpreters for forward prediction
cor_ends = get_hybridproblem_cor_ends(prob; scenario)
g, ϕg0 = get_hybridproblem_MLapplicator(prob; scenario)
ϕunc0 = get_hybridproblem_ϕunc(prob; scenario)
(; transP, transM) = get_hybridproblem_transforms(prob; scenario)
(; ϕ, transPMs_batch, interpreters, get_transPMs, get_ca_int_PMs) = init_hybrid_params(
    θP, θM, cor_ends, ϕg0, n_site; transP, transM, ϕunc0);

intm_PMs_gen = get_ca_int_PMs(n_site);
#intm_PMs_gen = get_ca_int_PMs(100);
trans_PMs_gen = get_transPMs(n_site);
#trans_PMs_gen = get_transPMs(100);

"""
ζMs in chain are all first parameter, all second parameters, ...
while ζMsin HVI and f are columns for each site, need to transform, back and forth
"""
transposeMs = (ζ, intm_PMs, back=false) -> begin
    ζc = intm_PMs(ζ)
    Ms = back ? reshape(ζc.Ms, reverse(size(ζc.Ms))) : ζc.Ms
    ζct = vcat(CA.getdata(ζc.P), vec(CA.getdata(Ms)'))
end
θ_true = vcat(CA.getdata(θP_true), vec(CA.getdata(θMs_true)));
ζ_true = log.(θ_true);
θ0_true = transposeMs(θ_true, intm_PMs_gen);
ζ0_true = log.(θ0_true);
#transposeMs(θ0_true, intm_PMs_gen, true) == θ_true

# mle_estimate = optimize(model, MLE(), θ_ini)
# mle_estimate.values

# takes ~ 25 minutes
#n_sample_NUTS = 800
n_sample_NUTS = 2000
#chain = sample(model, NUTS(), n_sample_NUTS, initial_params = ζ0_true .+ 0.001)
#n_sample_NUTS = 24
n_threads = 8
chain = sample(model, NUTS(), MCMCThreads(), ceil(Integer,n_sample_NUTS/n_threads), 
    n_threads, initial_params = fill(ζ0_true .+ 0.001, n_threads))


() -> begin
    using JLD2
    fname = "intermediate/doubleMM_chain_zeta_$(last(scenario)).jld2"
    jldsave(fname, false, IOStream; chain)
    chain = load(fname, "chain"; iotype = IOStream);
end

#ζi = first(eachrow(Array(chain)))
f_allsites = get_hybridproblem_PBmodel(prob0; scenario, use_all_sites = true)
ζs = mapreduce(ζi -> transposeMs(ζi, intm_PMs_gen, true), hcat, eachrow(Array(chain)));
(; θ, y) = HVI.predict_ζf(ζs, f_allsites, xP, trans_PMs_gen, intm_PMs_gen);
(ζs_hmc, θ_hmc, y_hmc) = (ζs, θ, y);

    
() -> begin # check that the model predicts the same as HVI-code
    _parnames = Symbol.(vcat([
        "ζP[1]"], ["ζMs[1, :][$i]" for i in 1:n_site], ["ζMs[2, :][$i]" for i in 1:n_site]))
    #chain0 = Chains(transposeMs(ζ_true, intm_PMs_gen)', _parnames);
    chain0 = Chains(ζ0_true', _parnames); # transpose here only for chain array
    y0inv = generated_quantities(model, chain0)[1, 1]
    y0pred = HVI.predict_y(ζ_true, xP, f, trans_PMs_gen, intm_PMs_gen)[2]
    y0pred .- y_true 
    y0inv .- y_true
end

() -> begin # plot chain
    #@usingany TwMakieHelpers, CairoMakie
    # θP and first θMs 
    ch = chain[:,vcat(1:n_θP, n_θP+1, n_θP+n_site+1),:];
    fig = plot_chn(ch)
    save("tmp.svg", fig)
end


mean_y_invζ = mean_y_hmc = map(mean, eachslice(y_hmc; dims = (1, 2)));
#describe(mean_y_pred - y_o)
histogram(vec(mean_y_invζ) - vec(y_true)) # predictions centered around y_o (or y_true)
plt = scatterplot(vec(y_true), vec(mean_y_invζ));
lineplot!(plt, 0, 1)
mean(mean_y_invζ - y_true) # still ok

# first site, first prediction
y_inv11 = y[1, 1, :]
histogram(y_inv11 .- y_o[1, 1])
histogram(y_inv11 .- y_true[1, 1])  

histogram(ζs[1, :])
describe(pdf.(Ref(prior_ζ), ζs[1, :]))  # only small differences 
pdf(prior_ζ, log(θ_true[1]))

mean_θ = CA.ComponentVector(mean(CA.getdata(θ); dims = 2)[:, 1], CA.getaxes(θ[:, 1])[1])
histogram(θ[:P, :] .- θP_true)  # all overestimated ? 
plt = scatterplot(θMs_true[1, :], mean_θ.Ms[1, :]);
lineplot!(plt, 0, 1)
plt = scatterplot(θMs_true[2, :], mean_θ.Ms[2, :]);
lineplot!(plt, 0, 1)

#------------------ compare HVI vs HMC sample
() -> begin   # compare against HVI sample 
    #@usingany AlgebraOfGraphics, TwPrototypes, CairoMakie, DataFrames
    makie_config = ppt_MakieConfig()
    function get_fig_size(cfg; width2height=golden_ratio, xfac=1.0)    
        cfg = makie_config
        x_inch = first(cfg.size_inches) * xfac
        y_inch = x_inch / width2height
        72 .* (x_inch, y_inch) ./ cfg.pt_per_unit # size_pt        
    end

    ζs_hvi = log.(θ2)
    ζs_hvi_indep = log.(θ2_indep)
    int_pms = interpreters.PMs
    par_pos = int_pms(1:length(int_pms))
    i_sites = 1:10
    #i_sites = 6:10
    #i_sites = 11:15
    scen = vcat(fill(:hvi,size(ζs_hvi,2)),fill(:hmc,size(ζs_hmc,2)),fill(:hvi_indep,size(ζs_hvi_indep,2)))
    dfP = mapreduce(vcat, axes(θP,1)) do i_par
        pos = par_pos.P[i_par]
        DataFrame(
            value = vcat(ζs_hvi[pos,:], ζs_hmc[pos,:], ζs_hvi_indep[pos,:]), 
            variable = keys(θP)[i_par],
            site = i_sites[1],
            Method = scen
        )
    end 
    dfMs = mapreduce(vcat, i_sites) do i_site 
        mapreduce(vcat, axes(θM,1)) do i_par
            pos = par_pos.Ms[i_par, i_site]
            DataFrame(
                value = vcat(ζs_hvi[pos,:], ζs_hmc[pos,:], ζs_hvi_indep[pos,:]),
                variable = keys(θM)[i_par],
                site = i_site,
                Method = scen
            )
        end
    end        
    df = vcat(dfP, dfMs)
    df_true = vcat(
        mapreduce(vcat, axes(θP,1)) do i_par
            DataFrame(
                value = ζP_true[i_par],
                variable = keys(θP)[i_par],
                site = i_sites[1],
                Method = :true
            )
        end,
        mapreduce(vcat, i_sites) do i_site 
            mapreduce(vcat, axes(θM,1)) do i_par
                DataFrame(
                    value = ζMs_true[i_par, i_site],
                    variable = keys(θM)[i_par],
                    site = i_site,
                    Method = :true
                )
            end
        end        
    ) # vcat
    #cf90 = (x) -> quantile(x, [0.05,0.95])
    plt = (data(subset(df, :Method => ByRow(∈((:hvi,:hmc))))) * mapping(:value=> (x -> x ) => "", color=:Method) * AlgebraOfGraphics.density(datalimits=extrema) +
           data(df_true) * mapping(:value) * visual(VLines; color=:blue, linestyle=:dash)) * 
        mapping(col=:variable => sorter(vcat(keys(θP)..., keys(θM)...)), row = (:site => nonnumeric)) 
    fig = Figure(size = get_fig_size(makie_config, xfac=1, width2height = 1/2));
    fg = draw!(fig, plt, facet=(; linkxaxes=:minimal, linkyaxes=:none,), axis=(xlabelvisible=false,));
    fig
    save("tmp.svg", fig)
    save_with_config("intermediate/compare_hmc_hvi_sites_$(last(scenario))", fig; makie_config)

    plt = (data(subset(df, :Method => ByRow(∈((:hvi, :hvi_indep))))) * mapping(:value=> (x -> x ) => "", color=:Method) * AlgebraOfGraphics.density(datalimits=extrema) +
           data(df_true) * mapping(:value) * visual(VLines; color=:blue, linestyle=:dash)) * 
        mapping(col=:variable => sorter(vcat(keys(θP)..., keys(θM)...)), row = (:site => nonnumeric)) 
    fig = Figure(size = get_fig_size(makie_config, xfac=1, width2height = 1/2));
    fg = draw!(fig, plt, facet=(; linkxaxes=:minimal, linkyaxes=:none,), axis=(xlabelvisible=false,));
    fig
    save("tmp.svg", fig)
    save_with_config("intermediate/compare_hvi_indep_sites_$(last(scenario))", fig; makie_config)

    # 
    # compare density of predictions
    y_hvi = y2
    i_obss = [1,4,8]
    #i_obss = 1:8
    dfy = mapreduce(vcat, i_obss) do i_obs
            mapreduce(vcat, i_sites) do i_site
                vcat(
                    DataFrame(
                        value = y_hmc[i_obs,i_site,:], 
                        site = i_site,
                        Method = :hmc,
                        variable = :y,
                        i_obs = i_obs,
                    ),
                    DataFrame(
                        value = y_hvi[i_obs,i_site,:], 
                        site = i_site,
                        Method = :hvi,
                        variable = :y,
                        i_obs = i_obs,
                    )
                )# vcat
            end
    end
    dfyt = mapreduce(vcat, i_obss) do i_obs
            mapreduce(vcat, i_sites) do i_site
                vcat(
                    DataFrame(
                        value = y_true[i_obs,i_site], 
                        site = i_site,
                        Method = :truth,
                        variable = :y,
                        i_obs = i_obs,
                    ),
                    DataFrame(
                        value = y_o[i_obs,i_site,:], 
                        site = i_site,
                        Method = :obs,
                        variable = :y,
                        i_obs = i_obs,
                    )
                )# vcat
            end
    end

    plt = (data(dfy) * mapping(color=:Method) * AlgebraOfGraphics.density(datalimits=extrema) +
            data(dfyt) * mapping(color=:Method) * visual(VLines;  linestyle=:dash)) * 
            #data(dfyt) * mapping(color=:Method, linestyle=:Method) * visual(VLines;  linestyle=:dash)) * 
        mapping(:value=>"", col=:i_obs => nonnumeric, row = :site => nonnumeric)

    fig = Figure(size = get_fig_size(makie_config, xfac=1, width2height = 1/2));
    f = draw!(fig, plt, 
        facet=(; linkxaxes=:minimal, linkyaxes=:none,), 
        axis=(xlabelvisible=false,yticklabelsvisible=false));
    legend!(fig[1,3], f, ; tellwidth=false, halign=:right, valign=:top) # , margin=(-10, -10, 10, 10)
    fig
    save_with_config("intermediate/compare_hmc_hvi_sites_y_$(last(scenario))", fig; makie_config)
    # hvi predicts y better, hmc fails for quite a few obs: 3,5,6

    # todo compare mean_predictions
    mean_y_hvi = map(mean, eachslice(y_hvi; dims = (1, 2)));

    
end

() -> begin # inspect correlation of residuals
    mean_ζ_hvi = map(mean, eachrow(CA.getdata(ζs_hvi)))
    r_hvi = ζs_hvi .- mean_ζ_hvi
    cor_hvi = cor(CA.getdata(r_hvi)')
    mean_ζ_hmc = map(mean, eachrow(CA.getdata(ζs_hmc)))
    r_hmc = ζs_hmc .- mean_ζ_hmc
    cor_hmc = cor(CA.getdata(r_hmc)')
    #
    hcat(cor_hvi[:,1], cor_hmc[:,1])
    # positive correlations of K2(1) in θP with K1(3) in θMs

end
    

#---- do an DEMC inversion of the PBM model with parameters at constrained scale
# construct a Normal prior that ranges roughly across 1e-2 to 10
# prior_θ = fit(LogNormal, @qp_ll(1e-2), @qp_uu(10))
# prior_θn = (n) -> MvLogNormal(fill(prior_θ.μ, n), PDiagMat(fill(abs2(prior_θ.σ), n)))
prior_θ = Normal(0, 10)
prior_θn = (n) -> MvNormal(fill(prior_θ.μ, n), PDiagMat(fill(abs2(prior_θ.σ), n)))
prior_θn(3)
prob = HVI.update(prob0o);

(; θM, θP) = get_hybridproblem_par_templates(prob; scenario)
n_θM, n_θP = length.((θM, θP))
f = get_hybridproblem_PBmodel(prob; scenario)

@model function fsites_uc(
        y, ::Type{T} = Float64; f, n_θP, n_θM, σ_o, n_obs = length(σ_o)) where {T}
    n_obs, n_site = size(y)
    prior_θP = prior_θn(n_θP)
    prior_θM_sites = fill(prior_θn(n_site), n_θM)
    θP ~ prior_θP #MvNormal(n_θP, 10.0)
    # CAUTION: order of vectorizing matrix depends on order of ~
    # need to assign each variable in first site first, then second site, ...
    #   need to construct different MvNormal prior if std differs by variable
    # or need to take care when extracting samples or specifying initial conditions
    θMs = Matrix{T}(undef, n_θM, n_site)
    # the first loop vectorizes θMs by columns but is much slower
    # for i_site in 1:n_site
    #     ζMs[:, i_site] ~ prior_ζn(n_θM) #MvNormal(n_site, 10.0)
    # end
    # this loop is faster, but vectorizes θMs by rows in parameter vector
    for i_par in 1:n_θM
        θMs[i_par, :] ~ prior_θM_sites[i_par]
    end
    # this fills in rows first, but is also slower- why?    
    #ζMs[:] ~ prior_ζn(n_θM * n_site) 
    # assume σ_o known, see f_MM
    #σ_o ~ truncated(Normal(0, 1); lower=0)
    y_pred = f(θP, θMs, xP)[2] # first is global return
    #i_obs = 1
    for i_obs in 1:n_obs
        #pdf(MvNormal(y_pred[i_obs,:], σ_o[i_obs]),y[i_obs,:])
        y[i_obs, :] ~ MvNormal(y_pred[i_obs, :], σ_o[i_obs]) # single value σ instead of variance
    end
    #Main.@infiltrate_main # step to second time 
    # θMs_MCc[:,:,1] # checking row- or column-order of θMs
    # exp.(ζMs)
    y_pred
end
model_uc = fsites_uc(y_o; f, n_θP, n_θM, σ_o)

() -> begin # check that the model predicts the same as HVI-code
    _parnames = Symbol.(vcat([
        "θP[1]"], ["θMs[1, :][$i]" for i in 1:n_site], ["θMs[2, :][$i]" for i in 1:n_site]))
    #chain0 = Chains(transposeMs(ζ_true, intm_PMs_gen)', _parnames);
    chain0 = Chains(θ0_true', _parnames); # transpose here only for chain array
    y0inv = generated_quantities(model_uc, chain0)[1, 1]
    y0pred = f(θP_true,  θMs_true, xP)[2]
    y0pred .- y_true 
    y0inv .- y_true
end

 
#θ0_true from above, tools from above

# takes long
n_sample_NUTS = 800
#n_sample_NUTS = 24
n_threads = 8
chain = sample(model_uc, NUTS(), MCMCThreads(), ceil(Integer,n_sample_NUTS/n_threads), 
    n_threads, initial_params = fill(θ0_true .+ 0.001, n_threads))

() -> begin
    using JLD2
    jldsave("intermediate/doubleMM_chain_theta.jld2", false, IOStream; chain)
    chain = load("intermediate/doubleMM_chain_theta.jld2", "chain"; iotype = IOStream)
    # plot chain as above
end

#θi = first(eachrow(Array(chain)))
θs = mapreduce(θi -> transposeMs(θi, intm_PMs_gen, true), hcat, eachrow(Array(chain)));
(; θ, y) = HVI.predict_ζf(θs, f, xP, Stacked(elementwise(identity)), intm_PMs_gen);


mean_y_invθ = map(mean, eachslice(y; dims = (1, 2)));
#describe(mean_y_pred - y_o)
histogram(vec(mean_y_invθ) - vec(y_true)) # predictions centered around y_o (or y_true)
plt = scatterplot(vec(y_true), vec(mean_y_invθ));
lineplot!(plt, 0, 2)
mean(mean_y_invθ - y_true) # still ok

# first site, first prediction
y_inv11 = y[1, 1, :]
histogram(y_inv11 .- y_o[1, 1])
histogram(y_inv11 .- y_true[1, 1])  

histogram(θs[1, :])
describe(pdf.(Ref(prior_θ), θs[1, :]))  # only small differences 
pdf(prior_θ, log(θ_true[1]))

mean_θ = CA.ComponentVector(mean(CA.getdata(θ); dims = 2)[:, 1], CA.getaxes(θ[:, 1])[1])
histogram(θ[:P, :] .- θP_true)  # all overestimated ? 
plt = scatterplot(θMs_true[1, :], mean_θ.Ms[1, :]);
lineplot!(plt, 0, 1)
plt = scatterplot(θMs_true[2, :], mean_θ.Ms[2, :]);
lineplot!(plt, 0, 1)










#---- depr?
size(chain)
θc = Array(chain)'
θinv = CA.ComponentArray(θc, (CA.getaxes(θ[:, 1])[1], CA.Axis(i = 1:size(θc, 2))))
mean_θinv = CA.ComponentVector(
    mean(CA.getdata(θinv); dims = 2)[:, 1], CA.getaxes(θ[:, 1])[1])

@assert chain[:, 1, :1] == CA.getdata(θinv[:P, :][:K2, :])
θP_true
plot = histogram(CA.getdata(θinv[:P, :][:K2, :]))

plt = scatterplot(θMs_true[1, :], mean_θinv.Ms[1, :]);
lineplot!(plt, 0, 1);
plt = scatterplot(θMs_true[2, :], mean_θinv.Ms[2, :])

y_true = f(θP_true, θMs_true, xP)[2]
yinv = map(i -> f(θinv[:P, i], θinv[:Ms, i], xP)[2], axes(θinv, 2)) |> stack
histogram(yinv[1, 1, :])
y_true[1, 1]

tmp = generated_quantities(model_uc, chain[1:10, :, :])


# reshape θMs (site x par) -> (par x site)
_intm_PMs = ComponentArrayInterpreter(
    CA.ComponentVector(P = θP_true, Ms = vec(CA.getdata(θMs_true))), (n_sample_NUTS,))
extract_parameters_fsites = (chain) -> begin
    Ac = _intm_PMs(transpose(Array(chain)))
    #θM = Ac[:Ms,:][:,1]
    θMs = mapslices(CA.getdata(Ac[:Ms, :]), dims = 1) do θM
        # (site x par) -> (par x site)
        vec(reshape(θM, n_site, :)')
    end
    vcat(Ac[:P, :], θMs)
end

#ζs_MC = transpose(max.(-10.0,Array(chain)))
#ζs_MC = transpose(Array(chain))
ζs_MC = extract_parameters_fsites(chain)
θs_MC = exp.(ζs_MC)

y_pred = y_pred_gen = stack(generated_quantities(model, chain)[:, 1])

#ax_θPMs =  _get_ComponentArrayInterpreter_axes(int_θPMs)
#intm_PMs = ComponentArrayInterpreter(ax_θPMs, n_sample_NUTS)
intm_PMs = ComponentArrayInterpreter(
    CA.ComponentVector(P = 1:n_θP, Ms = 1:(n_θM * n_site_batch)), (n_sample_NUTS,))
intm_Ps = ComponentArrayInterpreter(θP_true, (n_sample_NUTS,))
intm_Ms = ComponentArrayInterpreter(θM_true, (n_site_batch, n_sample_NUTS))
θs_MCc = intm_PMs(θs_MC)
θMs_MCc = intm_Ms(θs_MCc[:Ms, :])
θPs_MCc = intm_Ps(θs_MCc[:P, :])
ζs_MCc = intm_PMs(ζs_MC)
ζMs_MCc = intm_Ms(ζs_MCc[:Ms, :])
ζP_MCc = intm_Ps(ζs_MCc[:P, :])

# inspect correlation between physical parameter K and ML-parameter r at first (or ith) site

mean_ζP_MC = mapslices(mean, CA.getdata(ζP_MCc), dims = 2)[:, 1]
var_ζP_MC = map(x -> var(x; corrected = false), eachrow(ζP_MCc))

mean_ζMs_MC = mapslices(mean, CA.getdata(ζMs_MCc), dims = 3)[:, :, 1]
var_ζMs_MC = mapslices(x -> var(x; corrected = false), CA.getdata(ζMs_MCc), dims = 3)[
    :, :, 1]

hcat(log.(θP_true), mean_ζP_MC)

scatterplot(vec(log.(θMs_true)), vec(mean_ζMs_MC))
cor(vec(log.(θMs_true)), vec(mean_ζMs_MC))

scatterplot(mean_ζMs_MC[1, :], log.(var_ζMs_MC[1, :]))
scatterplot(mean_ζMs_MC[2, :], log.(var_ζMs_MC[2, :]))

# predictive posterior uncertainty
# θ = first(eachcol(θs_MC))
y_pred = stack(map(eachcol(θs_MC)) do θ
    θc = int_θPMs(θ)
    #θP, θMs = @view(θ[1:n_θP]), reshape(@view(θ[n_θP+1:end, :]), n_θM, :)
    θP, θMs = θc.θP, θc.θMs
    y_pred_i = applyf(f_doubleMM, θMs, θP)
end)
#hcat(y_pred[:,1,1], y_pred_gen[:,1,1])

σ_o_post_MC = mapslices(std, y_pred; dims = 3)
describe(σ_o_post_MC)
vcat(σ_o, mean(σ_o_post_MC), sqrt(mean(abs2, σ_o_post_MC)))

() -> begin
    i_site = 12
    y_pred_site = 1
    i_sample = 1
    plt = scatterplot(y_o[:, i_site], y_pred[:, i_site, i_sample])
    for i_sample in 2:20   #n_sample_NUTS
        plt = scatterplot!(plt, y_o[:, i_site], y_pred[:, i_site, i_sample])
    end
    plt
end

# correlation r1 and K1 (original and log scale)
i_site = 5
scatterplot(ζMs_MCc[:r1, i_site, :], ζMs_MCc[:K1, i_site, :])
scatterplot(θMs_MCc[:r1, i_site, :], θMs_MCc[:K1, i_site, :])

# correlation r1 and K2 (physical) (original and log scale)
scatterplot(ζMs_MCc[:r1, i_site, :], ζP_MCc[:K2, :])
scatterplot(θMs_MCc[:r1, i_site, :], θPs_MCc[:K2, :])
cor(ζMs_MCc[:r1, i_site, :], ζP_MCc[:K2, :]) # only weak

ζPM1_MCc = vcat(θPs_MCc, θMs_MCc[:, i_site, :])
cor(CA.getdata(ζPM1_MCc)') # increases with larger σ_o (repeated with changed code)

save_prefious_MCSampling = () -> begin
    jldsave(joinpath(@__DIR__, "uncNN", "intermediate", "f_doubleMM_MC.jld2"), IOStream;
        ζs_MC)
end

# TODO compare distributions to MC sample
