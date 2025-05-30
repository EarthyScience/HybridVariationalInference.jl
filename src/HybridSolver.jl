abstract type AbstractHybridSolver end

struct HybridPointSolver{A} <: AbstractHybridSolver
    alg::A
end

HybridPointSolver(; alg) = HybridPointSolver(alg)

function CommonSolve.solve(prob::AbstractHybridProblem, solver::HybridPointSolver;
    scenario, rng=Random.default_rng(),
    gdev=:use_gpu ∈ _val_value(scenario) ? gpu_device() : identity,
    cdev=gdev isa MLDataDevices.AbstractGPUDevice ? cpu_device() : identity,
    is_inferred::Val{is_infer} = Val(false),
    kwargs...
) where is_infer
    par_templates = get_hybridproblem_par_templates(prob; scenario)
    g, ϕg0 = get_hybridproblem_MLapplicator(prob; scenario)
    FT = get_hybridproblem_float_type(prob; scenario)
    (; transP, transM) = get_hybridproblem_transforms(prob; scenario)
    intϕ = ComponentArrayInterpreter(CA.ComponentVector(
        ϕg=1:length(ϕg0), ϕP=par_templates.θP))
    #ϕ0_cpu = vcat(ϕg0, par_templates.θP .* FT(0.9))  # slightly disturb θP_true
    ϕ0_cpu = vcat(ϕg0, apply_preserve_axes(inverse(transP), par_templates.θP))
    train_loader = get_hybridproblem_train_dataloader(prob; scenario)
    if gdev isa MLDataDevices.AbstractGPUDevice
        ϕ0_dev = gdev(ϕ0_cpu)
        g_dev = gdev(g)
        train_loader_dev = gdev_hybridproblem_dataloader(train_loader; scenario, gdev)
    else
        ϕ0_dev = ϕ0_cpu
        g_dev = g
        train_loader_dev = train_loader
    end
    f = get_hybridproblem_PBmodel(prob; scenario, use_all_sites=false)
    y_global_o = FT[] # TODO
    pbm_covars = get_hybridproblem_pbmpar_covars(prob; scenario)
    n_site, n_batch = get_hybridproblem_n_site_and_batch(prob; scenario)
    #intP = ComponentArrayInterpreter(par_templates.θP)
    loss_gf = get_loss_gf(g_dev, transM, transP, f, y_global_o, intϕ;
        cdev, pbm_covars, n_site_batch=n_batch)
    # call loss function once
    l1 = is_infer ? 
        Test.@inferred(loss_gf(ϕ0_dev, first(train_loader_dev)...))[1] : 
        # using ShareAdd; @usingany Cthulhu
        # @descend_code_warntype loss_gf(ϕ0_dev, first(train_loader_dev)...)
        loss_gf(ϕ0_dev, first(train_loader_dev)...)[1]
    # and gradient
    # xMg, xP, y_o, y_unc = first(train_loader_dev)
    # gr1 = Zygote.gradient(
    #             p -> loss_gf(p, xMg, xP, y_o, y_unc)[1],
    #             ϕ0_dev)
    # Zygote.gradient(ϕ0_dev -> loss_gf(ϕ0_dev, data1...)[1], ϕ0_dev)
    optf = Optimization.OptimizationFunction((ϕ, data) -> loss_gf(ϕ, data...)[1],
        Optimization.AutoZygote())
    optprob = OptimizationProblem(optf, CA.getdata(ϕ0_dev), train_loader_dev)
    res = Optimization.solve(optprob, solver.alg; kwargs...)
    ϕ = intϕ(res.u)
    θP = cpu_ca(apply_preserve_axes(transP, cpu_ca(ϕ).ϕP))
    probo = update(prob; ϕg=cpu_ca(ϕ).ϕg, θP)
    (; ϕ, resopt=res, probo)
end

struct HybridPosteriorSolver{A} <: AbstractHybridSolver
    alg::A
    n_MC::Int
    n_MC_cap::Int
end
function HybridPosteriorSolver(; alg, n_MC=12, n_MC_cap=n_MC)
    HybridPosteriorSolver(alg, n_MC, n_MC_cap)
end
function update(solver::HybridPosteriorSolver;
    alg=solver.alg,
    n_MC=solver.n_MC,
    n_MC_cap=n_MC)
    HybridPosteriorSolver(alg, n_MC, n_MC_cap)
end

function CommonSolve.solve(prob::AbstractHybridProblem, solver::HybridPosteriorSolver;
    scenario::Val{scen}, rng=Random.default_rng(),
    gdev=:use_gpu ∈ _val_value(scenario) ? gpu_device() : identity,
    cdev=gdev isa MLDataDevices.AbstractGPUDevice ? cpu_device() : identity,
    θmean_quant=0.0,
    is_inferred::Val{is_infer} = Val(false),
    kwargs...
) where {scen, is_infer}
    par_templates = get_hybridproblem_par_templates(prob; scenario)
    (; θP, θM) = par_templates
    cor_ends = get_hybridproblem_cor_ends(prob; scenario)
    g, ϕg0 = get_hybridproblem_MLapplicator(prob; scenario)
    ϕunc0 = get_hybridproblem_ϕunc(prob; scenario)
    (; transP, transM) = get_hybridproblem_transforms(prob; scenario)
    pbm_covars = get_hybridproblem_pbmpar_covars(prob; scenario)
    n_site, n_batch = get_hybridproblem_n_site_and_batch(prob; scenario)
    hpints = HybridProblemInterpreters(prob; scenario)
    (; ϕ, transPMs_batch, interpreters, get_transPMs, get_ca_int_PMs) = init_hybrid_params(
        θP, θM, cor_ends, ϕg0, hpints; transP, transM, ϕunc0)
    int_unc = interpreters.unc
    int_μP_ϕg_unc = interpreters.μP_ϕg_unc
    transMs = StackedArray(transM, n_batch)
    #
    train_loader = get_hybridproblem_train_dataloader(prob; scenario)
    if gdev isa MLDataDevices.AbstractGPUDevice
        ϕ0_dev = gdev(ϕ)
        g_dev = gdev(g) # zygote fails if  gdev is a CPUDevice, although should be non-op
        train_loader_dev = gdev_hybridproblem_dataloader(train_loader; scenario, gdev)
    else
        ϕ0_dev = ϕ
        g_dev = g
        train_loader_dev = train_loader
    end
    f = get_hybridproblem_PBmodel(prob; scenario, use_all_sites=false)
    py = get_hybridproblem_neg_logden_obs(prob; scenario)

    priors_θP_mean, priors_θMs_mean = construct_priors_θ_mean(
        prob, ϕ0_dev.ϕg, keys(θM), θP, θmean_quant, g_dev, transM, transP;
        scenario, get_ca_int_PMs, gdev, cdev, pbm_covars)
    y_global_o = Float32[] # TODO

    loss_elbo = get_loss_elbo(
        g_dev, transP, transMs, f, py, y_global_o;
        solver.n_MC, solver.n_MC_cap, cor_ends, priors_θP_mean, priors_θMs_mean, cdev,
        pbm_covars, θP, int_unc, int_μP_ϕg_unc)
    # test loss function once
    #Main.@infiltrate_main
    l0 = is_infer ? 
        (Test.@inferred loss_elbo(ϕ0_dev, rng, first(train_loader_dev)...)) :
        loss_elbo(ϕ0_dev, rng, first(train_loader_dev)...)
    optf = Optimization.OptimizationFunction((ϕ, data) -> first(loss_elbo(ϕ, rng, data...)),
        Optimization.AutoZygote())
    optprob = OptimizationProblem(optf, CA.getdata(ϕ0_dev), train_loader_dev)
    res = Optimization.solve(optprob, solver.alg; kwargs...)
    ϕc = interpreters.μP_ϕg_unc(res.u)
    θP = cpu_ca(apply_preserve_axes(transP, ϕc.μP))
    probo = update(prob; ϕg=cpu_ca(ϕc).ϕg, θP=θP, ϕunc=cpu_ca(ϕc).unc)
    (; ϕ=ϕc, θP, resopt=res, interpreters, probo)
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
- `θP`: ComponentVector as a template to select indices of pbm_covars

The loss function takes in addition to ϕ, data that changes with minibatch
- `rng`: random generator
- `xM`: matrix of covariates, sites in columns
- `xP`: drivers for the processmodel: Iterator of size n_site
- `y_o`, `y_unc`: matrix of observations and uncertainties, sites in columns
"""
function get_loss_elbo(g, transP, transMs, f, py, y_o_global;
    n_MC, n_MC_mean = max(n_MC,20), n_MC_cap=n_MC, 
    cor_ends, priors_θP_mean, priors_θMs_mean, cdev, pbm_covars, θP,
    int_unc, int_μP_ϕg_unc,
)
    let g = g, transP = transP, transMs = transMs, f = f, py = py, y_o_global = y_o_global, 
        n_MC = n_MC, n_MC_cap = n_MC_cap, n_MC_mean = n_MC_mean,
        cor_ends = cor_ends,
        int_unc = get_concrete(int_unc), int_μP_ϕg_unc = get_concrete(int_μP_ϕg_unc),
        priors_θP_mean = priors_θP_mean, priors_θMs_mean = priors_θMs_mean, cdev = cdev,
        pbm_covar_indices = get_pbm_covar_indices(θP, pbm_covars),
        trans_mP=StackedArray(transP, n_MC_mean), 
        trans_mMs=StackedArray(transMs.stacked, n_MC_mean)

        function loss_elbo(ϕ, rng, xM, xP, y_o, y_unc, i_sites)
            neg_elbo_gtf(
                rng, ϕ, g, f, py, xM, xP, y_o, y_unc, i_sites;
                int_unc, int_μP_ϕg_unc,
                n_MC, n_MC_cap, n_MC_mean, cor_ends, priors_θP_mean, priors_θMs_mean,
                cdev, pbm_covar_indices, transP, transMs, trans_mP, trans_mMs,
            )
        end
    end
end

"""
Compute the components of the elbo for given initial conditions of the problems
for the first batch of the trainloader.
"""
function compute_elbo_components(
    prob::AbstractHybridProblem, solver::HybridPosteriorSolver;
    scenario, rng=Random.default_rng(), gdev=gpu_device(),
    θmean_quant=0.0,
    use_all_sites=false,
    kwargs...)
    n_site, n_batch = get_hybridproblem_n_site_and_batch(prob; scenario)
    par_templates = get_hybridproblem_par_templates(prob; scenario)
    (; θP, θM) = par_templates
    cor_ends = get_hybridproblem_cor_ends(prob; scenario)
    g, ϕg0 = get_hybridproblem_MLapplicator(prob; scenario)
    ϕunc0 = get_hybridproblem_ϕunc(prob; scenario)
    (; transP, transM) = get_hybridproblem_transforms(prob; scenario)
    (; ϕ, transPMs_batch, interpreters, get_transPMs, get_ca_int_PMs) = init_hybrid_params(
        θP, θM, cor_ends, ϕg0, n_batch; transP, transM, ϕunc0)
    train_loader = get_hybridproblem_train_dataloader(prob; scenario)
    if gdev isa MLDataDevices.AbstractGPUDevice
        ϕ0_dev = gdev(ϕ)
        g_dev = gdev(g) # zygote fails if  gdev is a CPUDevice, although should be non-op
        train_loader_dev = gdev_hybridproblem_dataloader(train_loader; scenario, gdev)
    else
        ϕ0_dev = ϕ
        g_dev = g
        train_loader_dev = train_loader
    end
    f = get_hybridproblem_PBmodel(prob; scenario, use_all_sites)
    py = get_hybridproblem_neg_logden_obs(prob; scenario)
    priors_θ_mean = construct_priors_θ_mean(
        prob, ϕ0_dev.ϕg, keys(θM), θP, θmean_quant, g_dev, transM;
        scenario, get_ca_int_PMs, gdev, cdev, pbm_covars)
    # TODO replace train_loader.data by proper function that pulls all the data
    xM, xP, y_o, y_unc, i_sites = use_all_sites ? train_loader_dev.data : first(train_loader_dev)
    neg_elbo_gtf_components(
        rng, ϕ0_dev, g_dev, transPMs_batch, f, py, xM, xP, y_o, y_unc, i_sites, interpreters;
        solver.n_MC, solver.n_MC_cap, cor_ends, priors_θ_mean)
end

"""
In order to let mean of θ stay close to initial point parameter estimates 
construct a prior on mean θ to a Normal around initial prediction.
"""
function construct_priors_θ_mean(prob, ϕg, keysθM, θP, θmean_quant, g_dev, transM, transP;
    scenario::Val{scen}, get_ca_int_PMs, gdev, cdev, pbm_covars) where {scen}
    iszero(θmean_quant) ? ([],[]) :
    begin
        n_site, n_batch = get_hybridproblem_n_site_and_batch(prob; scenario)
        # all_loader = MLUtils.DataLoader(
        #     get_hybridproblem_train_dataloader(prob; scenario).data, batchsize = n_site)
        # xM_all = first(all_loader)[1]
        is_gpu = :use_gpu ∈ scen
        xM_all_cpu = get_hybridproblem_train_dataloader(prob; scenario).data[1]
        xM_all = is_gpu ? gdev(xM_all_cpu) : xM_all_cpu
        ζP = apply_preserve_axes(inverse(transP), θP)
        pbm_covar_indices = get_pbm_covar_indices(θP, pbm_covars)
        xMP_all = _append_each_covars(xM_all, CA.getdata(ζP), pbm_covar_indices)
        transMs = StackedArray(transM, n_site)
        # ζMs = g_dev(xMP_all, CA.getdata(ϕg))'  # transpose to par-last for StackedArray
        # ζMs_cpu = cdev(ζMs)
        # θMs = transMs(ζMs_cpu)
        θMs = gtrans(g_dev, transMs, xMP_all, CA.getdata(ϕg); cdev=cpu_device())
        priors_dict = get_hybridproblem_priors(prob; scenario)
        priorsP = [priors_dict[k] for k in keys(θP)]
        priors_θP_mean = map(priorsP, θP) do priorsP, θPi
            fit_narrow_normal(θPi, priorsP, θmean_quant)
        end
        priorsM = [priors_dict[k] for k in keysθM]
        i_par = 1
        i_site = 1
        priors_θMs_mean = map(Iterators.product(axes(θMs)...)) do (i_site, i_par)
            #@show i_par, i_site
            fit_narrow_normal(θMs[i_site, i_par], priorsM[i_par], θmean_quant)
        end
        # # concatenate to a flat vector
        # int_n_site = get_ca_int_PMs(n_site)
        # int_n_site(vcat(priors_θP_mean, vec(priors_θMs_mean)))
        priors_θP_mean, priors_θMs_mean
    end
end
