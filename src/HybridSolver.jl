abstract type AbstractHybridSolver end

struct HybridPointSolver{A} <: AbstractHybridSolver
    alg::A
    n_batch::Int
end

HybridPointSolver(; alg, n_batch = 10) = HybridPointSolver(alg, n_batch)
#HybridPointSolver(; alg = Adam(0.02), n_batch = 10) = HybridPointSolver(alg,n_batch)

function CommonSolve.solve(prob::AbstractHybridProblem, solver::HybridPointSolver;
        scenario, rng = Random.default_rng(), 
        gdev = :use_gpu ∈ scenario ? gpu_device() : identity, 
        cdev = gdev isa MLDataDevices.AbstractGPUDevice ? cpu_device() : identity,
        kwargs...)
    par_templates = get_hybridproblem_par_templates(prob; scenario)
    g, ϕg0 = get_hybridproblem_MLapplicator(prob; scenario)
    FT = get_hybridproblem_float_type(prob; scenario)
    (; transP, transM) = get_hybridproblem_transforms(prob; scenario)
    intϕ = ComponentArrayInterpreter(CA.ComponentVector(
        ϕg = 1:length(ϕg0), ϕP = par_templates.θP))
    #ϕ0_cpu = vcat(ϕg0, par_templates.θP .* FT(0.9))  # slightly disturb θP_true
    ϕ0_cpu = vcat(ϕg0, apply_preserve_axes(inverse(transP),par_templates.θP))
    if gdev isa MLDataDevices.AbstractGPUDevice
        ϕ0_dev = gdev(ϕ0_cpu)
        g_dev = gdev(g)
    else
        ϕ0_dev = ϕ0_cpu
        g_dev = g
    end
    train_loader = get_hybridproblem_train_dataloader(
        prob; scenario, n_batch = solver.n_batch)
    f = get_hybridproblem_PBmodel(prob; scenario)
    y_global_o = FT[] # TODO
    pbm_covars = get_hybridproblem_pbmpar_covars(prob; scenario)
    #intP = ComponentArrayInterpreter(par_templates.θP)
    loss_gf = get_loss_gf(g_dev, transM, transP, f, y_global_o, intϕ; cdev, pbm_covars)
    # call loss function once
    l1 = loss_gf(ϕ0_dev, first(train_loader)...)[1]
    # and gradient
    # xMg, xP, y_o, y_unc = first(train_loader)
    # gr1 = Zygote.gradient(
    #             p -> loss_gf(p, xMg, xP, y_o, y_unc)[1],
    #             ϕ0_dev)
    # data1 = first(train_loader)
    # Zygote.gradient(ϕ0_dev -> loss_gf(ϕ0_dev, data1...)[1], ϕ0_dev)
    optf = Optimization.OptimizationFunction((ϕ, data) -> loss_gf(ϕ, data...)[1],
        Optimization.AutoZygote())
    optprob = OptimizationProblem(optf, CA.getdata(ϕ0_dev), train_loader)
    res = Optimization.solve(optprob, solver.alg; kwargs...)
    ϕ = intϕ(res.u)
    θP = cpu_ca(apply_preserve_axes(transP, cpu_ca(ϕ).ϕP))
    probo = update(prob; ϕg = cpu_ca(ϕ).ϕg, θP)
    (; ϕ, resopt = res, probo)
end

struct HybridPosteriorSolver{A} <: AbstractHybridSolver
    alg::A
    n_batch::Int
    n_MC::Int
    n_MC_cap::Int
end
function HybridPosteriorSolver(; alg, n_batch = 10, n_MC = 12, n_MC_cap = n_MC)
    HybridPosteriorSolver(alg, n_batch, n_MC, n_MC_cap)
end
function update(solver::HybridPosteriorSolver;
        alg = solver.alg,
        n_batch = solver.n_batch,
        n_MC = solver.n_MC,
        n_MC_cap = n_MC)
    HybridPosteriorSolver(alg, n_batch, n_MC, n_MC_cap)
end

function CommonSolve.solve(prob::AbstractHybridProblem, solver::HybridPosteriorSolver;
        scenario, rng = Random.default_rng(), 
        gdev = :use_gpu ∈ scenario ? gpu_device() : identity, 
        cdev = gdev isa MLDataDevices.AbstractGPUDevice ? cpu_device() : identity,
        θmean_quant = 0.0,
        kwargs...)
    par_templates = get_hybridproblem_par_templates(prob; scenario)
    (; θP, θM) = par_templates
    cor_ends = get_hybridproblem_cor_ends(prob; scenario)
    g, ϕg0 = get_hybridproblem_MLapplicator(prob; scenario)
    ϕunc0 = get_hybridproblem_ϕunc(prob; scenario)
    (; transP, transM) = get_hybridproblem_transforms(prob; scenario)
    pbm_covars = get_hybridproblem_pbmpar_covars(prob; scenario)
    (; ϕ, transPMs_batch, interpreters, get_transPMs, get_ca_int_PMs) = init_hybrid_params(
        θP, θM, cor_ends, ϕg0, solver.n_batch; transP, transM, ϕunc0)
    if gdev isa MLDataDevices.AbstractGPUDevice
        ϕ0_dev = gdev(ϕ)
        g_dev = gdev(g) # zygote fails if  gdev is a CPUDevice, although should be non-op
    else
        ϕ0_dev = ϕ
        g_dev = g
    end
    train_loader = get_hybridproblem_train_dataloader(prob; scenario, solver.n_batch)
    f = get_hybridproblem_PBmodel(prob; scenario)
    py = get_hybridproblem_neg_logden_obs(prob; scenario)
    priors_θ_mean = construct_priors_θ_mean(
        prob, ϕ0_dev.ϕg, keys(θM), θP, θmean_quant, g_dev, transM, transP;
        scenario, get_ca_int_PMs, cdev, pbm_covars)
    y_global_o = Float32[] # TODO
    loss_elbo = get_loss_elbo(
        g_dev, transPMs_batch, f, py, y_global_o, interpreters;
        solver.n_MC, solver.n_MC_cap, cor_ends, priors_θ_mean, cdev, pbm_covars, θP)
    # test loss function once
    l0 = loss_elbo(ϕ0_dev, rng, first(train_loader)...)
    optf = Optimization.OptimizationFunction((ϕ, data) -> loss_elbo(ϕ, rng, data...)[1],
        Optimization.AutoZygote())
    optprob = OptimizationProblem(optf, CA.getdata(ϕ0_dev), train_loader)
    res = Optimization.solve(optprob, solver.alg; kwargs...)
    ϕc = interpreters.μP_ϕg_unc(res.u)
    θP = cpu_ca(apply_preserve_axes(transP, ϕc.μP))
    probo = update(prob; ϕg = cpu_ca(ϕ).ϕg, θP = θP, ϕunc = cpu_ca(ϕ).unc);
    (; ϕ = ϕc, θP, resopt = res, interpreters, probo)
end

function fit_narrow_normal(θi, prior, θmean_quant)
    p_lower, p_upper = cdf(prior, θi) .+ (-θmean_quant, +θmean_quant)
    p_lower = max(1e-3, p_lower)
    p_upper = min(1 - 1e-3, p_upper)
    q_lower, q_upper = quantile.(prior, (p_lower, p_upper))
    d = fit(Normal, @qp_l(q_lower), @qp_u(q_upper))
end

"""
Create a loss function for parameter vector ϕ, given 
- `g(x, ϕ)`: machine learning model 
- `transPMS`: transformation from unconstrained space to parameter space
- `f(θMs, θP)`: mechanistic model 
- `interpreters`: assigning structure to pure vectors, see `neg_elbo_gtf`
- `n_MC`: number of Monte-Carlo sample to approximate the expected value across distribution
- `pbm_covars`: tuple of symbols of process-based parameters provided to the ML model
- `θP`: CompoenntVector as a template to select indices of pbm_covars

The loss function takes in addition to ϕ, data that changes with minibatch
- `rng`: random generator
- `xM`: matrix of covariates, sites in columns
- `xP`: drivers for the processmodel: Iterator of size n_site
- `y_o`, `y_unc`: matrix of observations and uncertainties, sites in columns
"""
function get_loss_elbo(g, transPMs, f, py, y_o_global, interpreters;
        n_MC, n_MC_cap = n_MC, cor_ends, priors_θ_mean, cdev, pbm_covars, θP,
        )
    let g = g, transPMs = transPMs, f = f, py = py, y_o_global = y_o_global, n_MC = n_MC,
        cor_ends = cor_ends, interpreters = map(get_concrete, interpreters),
        priors_θ_mean = priors_θ_mean, cdev = cdev, 
        pbm_covar_indices = get_pbm_covar_indices(θP, pbm_covars)

        function loss_elbo(ϕ, rng, xM, xP, y_o, y_unc, i_sites)
            neg_elbo_gtf(
                rng, ϕ, g, transPMs, f, py, xM, xP, y_o, y_unc, i_sites, interpreters;
                n_MC, n_MC_cap, cor_ends, priors_θ_mean, cdev, pbm_covar_indices)
        end
    end
end

"""
Compute the components of the elbo for given initial conditions of the problems
for the first batch of the trainloader, whose `n_batch` defaults to all sites.
"""
function compute_elbo_components(
        prob::AbstractHybridProblem, solver::HybridPosteriorSolver;
        scenario, rng = Random.default_rng(), gdev = gpu_device(),
        θmean_quant = 0.0,
        n_batch = get_hybridproblem_n_site(prob; scenario),
        kwargs...)
    par_templates = get_hybridproblem_par_templates(prob; scenario)
    (; θP, θM) = par_templates
    cor_ends = get_hybridproblem_cor_ends(prob; scenario)
    g, ϕg0 = get_hybridproblem_MLapplicator(prob; scenario)
    ϕunc0 = get_hybridproblem_ϕunc(prob; scenario)
    (; transP, transM) = get_hybridproblem_transforms(prob; scenario)
    (; ϕ, transPMs_batch, interpreters, get_transPMs, get_ca_int_PMs) = init_hybrid_params(
        θP, θM, cor_ends, ϕg0, n_batch; transP, transM, ϕunc0)
    if gdev isa MLDataDevices.AbstractGPUDevice
        ϕ0_dev = gdev(ϕ)
        g_dev = gdev(g) # zygote fails if  gdev is a CPUDevice, although should be non-op
    else
        ϕ0_dev = ϕ
        g_dev = g
    end
    train_loader = get_hybridproblem_train_dataloader(prob; scenario, n_batch)
    f = get_hybridproblem_PBmodel(prob; scenario)
    py = get_hybridproblem_neg_logden_obs(prob; scenario)
    priors_θ_mean = construct_priors_θ_mean(
        prob, ϕ0_dev.ϕg, keys(θM), θP, θmean_quant, g_dev, transM;
        scenario, get_ca_int_PMs)
    xM, xP, y_o, y_unc, i_sites = first(train_loader)
    neg_elbo_gtf_components(
        rng, ϕ0_dev, g_dev, transPMs_batch, f, py, xM, xP, y_o, y_unc, i_sites, interpreters;
        solver.n_MC, solver.n_MC_cap, cor_ends, priors_θ_mean)
end

"""
In order to let mean of θ stay close to initial point parameter estimates 
construct a prior on mean θ to a Normal around initial prediction.
"""
function construct_priors_θ_mean(prob, ϕg, keysθM, θP, θmean_quant, g_dev, transM, transP;
        scenario, get_ca_int_PMs, cdev, pbm_covars)
    iszero(θmean_quant) ? [] :
    begin
        n_site = get_hybridproblem_n_site(prob; scenario)
        all_loader = get_hybridproblem_train_dataloader(prob; scenario, n_batch = n_site)
        xM_all = first(all_loader)[1]
        #Main.@infiltrate_main
        ζP = apply_preserve_axes(inverse(transP), θP)
        pbm_covar_indices = get_pbm_covar_indices(θP, pbm_covars)
        xMP_all = _append_each_covars(xM_all, CA.getdata(ζP), pbm_covar_indices) 
        θMs = gtrans(g_dev, transM, xMP_all, CA.getdata(ϕg); cdev)
        priors_dict = get_hybridproblem_priors(prob; scenario)
        priorsP = [priors_dict[k] for k in keys(θP)]
        priors_θP_mean = map(priorsP, θP) do priorsP, θPi
            fit_narrow_normal(θPi, priorsP, θmean_quant)
        end
        priorsM = [priors_dict[k] for k in keysθM]
        i_par = 1
        i_site = 1
        priors_θMs_mean = map(Iterators.product(axes(θMs)...)) do (i_par, i_site)
            #@show i_par, i_site
            fit_narrow_normal(θMs[i_par, i_site], priorsM[i_par], θmean_quant)
        end
        # concatenate to a flat vector
        int_n_site = get_ca_int_PMs(n_site)
        int_n_site(vcat(priors_θP_mean, vec(priors_θMs_mean)))
    end
end
