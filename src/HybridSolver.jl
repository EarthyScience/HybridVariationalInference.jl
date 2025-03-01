abstract type AbstractHybridSolver end

struct HybridPointSolver{A} <: AbstractHybridSolver
    alg::A
    n_batch::Int
end

HybridPointSolver(; alg, n_batch = 10) = HybridPointSolver(alg, n_batch)
#HybridPointSolver(; alg = Adam(0.02), n_batch = 10) = HybridPointSolver(alg,n_batch)

function CommonSolve.solve(prob::AbstractHybridProblem, solver::HybridPointSolver;
        scenario, rng = Random.default_rng(), dev = gpu_device(), kwargs...)
    par_templates = get_hybridproblem_par_templates(prob; scenario)
    g, ϕg0 = get_hybridproblem_MLapplicator(prob; scenario)
    FT = get_hybridproblem_float_type(prob; scenario)
    (; transP, transM) = get_hybridproblem_transforms(prob; scenario)
    int_ϕθP = ComponentArrayInterpreter(CA.ComponentVector(
        ϕg = 1:length(ϕg0), θP = par_templates.θP))
    #p0_cpu = vcat(ϕg0, par_templates.θP .* FT(0.9))  # slightly disturb θP_true
    p0_cpu = vcat(ϕg0, par_templates.θP)
    p0 = p0_cpu
    g_dev = g
    if dev isa MLDataDevices.AbstractGPUDevice
        p0 = dev(p0_cpu)
        g_dev = dev(g)
    end
    train_loader = get_hybridproblem_train_dataloader(
        prob; scenario, n_batch = solver.n_batch)
    f = get_hybridproblem_PBmodel(prob; scenario)
    y_global_o = FT[] # TODO
    loss_gf = get_loss_gf(g_dev, transM, f, y_global_o, int_ϕθP)
    # call loss function once
    l1 = loss_gf(p0, first(train_loader)...)[1]
    # and gradient
    # xMg, xP, y_o, y_unc = first(train_loader)
    # gr1 = Zygote.gradient(
    #             p -> loss_gf(p, xMg, xP, y_o, y_unc)[1],
    #             p0)
    # data1 = first(train_loader)
    # Zygote.gradient(p0 -> loss_gf(p0, data1...)[1], p0)
    optf = Optimization.OptimizationFunction((ϕ, data) -> loss_gf(ϕ, data...)[1],
        Optimization.AutoZygote())
    optprob = OptimizationProblem(optf, CA.getdata(p0), train_loader)
    res = Optimization.solve(optprob, solver.alg; kwargs...)
    (; ϕ = int_ϕθP(res.u), resopt = res)
end

struct HybridPosteriorSolver{A} <: AbstractHybridSolver
    alg::A
    n_batch::Int
    n_MC::Int
end
function HybridPosteriorSolver(; alg, n_batch = 10, n_MC = 3)
    HybridPosteriorSolver(alg, n_batch, n_MC)
end
function update(solver::HybridPosteriorSolver; 
    alg = solver.alg,
    n_batch = solver.n_batch,
    n_MC = solver.n_MC)
    HybridPosteriorSolver(alg, n_batch, n_MC)
end

function CommonSolve.solve(prob::AbstractHybridProblem, solver::HybridPosteriorSolver;
        scenario, rng = Random.default_rng(), dev = gpu_device(), 
        θmean_quant = 0.0,        
        kwargs...)
    par_templates = get_hybridproblem_par_templates(prob; scenario)
    (; θP, θM) = par_templates
    cor_ends = get_hybridproblem_cor_ends(prob; scenario)
    g, ϕg0 = get_hybridproblem_MLapplicator(prob; scenario)
    ϕunc0 = get_hybridproblem_ϕunc(prob; scenario)
    (; transP, transM) = get_hybridproblem_transforms(prob; scenario)
    (; ϕ, transPMs_batch, interpreters, get_transPMs, get_ca_int_PMs) = init_hybrid_params(
        θP, θM, cor_ends, ϕg0, solver.n_batch; transP, transM, ϕunc0)
    if dev isa MLDataDevices.AbstractGPUDevice
        ϕ0_dev = dev(ϕ)
        g_dev = dev(g) # zygote fails if  dev is a CPUDevice, although should be non-op
    else
        ϕ0_dev = ϕ
        g_dev = g
    end
    train_loader = get_hybridproblem_train_dataloader(prob; scenario, solver.n_batch)
    f = get_hybridproblem_PBmodel(prob; scenario)
    py = get_hybridproblem_neg_logden_obs(prob; scenario)
    priors_θ_mean = iszero(θmean_quant) ? [] : begin
        # in order to let mean θ stay close to initial point parameter estimates 
        # construct a prior on mean θ to a Normal around initial prediction
        n_site = get_hybridproblem_n_site(prob; scenario)
        all_loader = get_hybridproblem_train_dataloader(prob; scenario, n_batch = n_site)
        xM_all = first(all_loader)[1]
        # ensure that the thrainloader returns the same underlying data as all_loader
        xM1 = first(train_loader)[1]
        @assert xM1 == xM_all[:,1:size(xM1,2)]
        θMs = gtrans(g_dev, transM, xM_all, CA.getdata(ϕ0_dev.ϕg))
        priors_dict = get_hybridproblem_priors(prob; scenario)
        #Main.@infiltrate_main
        priorsP = [priors_dict[k] for k in keys(θP)]
        priors_θP_mean = map(priorsP, θP) do priorsP, θPi
            fit_narrow_normal(θPi, priorsP, θmean_quant)
        end
        priorsM = [priors_dict[k] for k in keys(θM)]
        i_par = 1; i_site = 1
        priors_θMs_mean = map(Iterators.product(axes(θMs)...)) do (i_par, i_site)
            #@show i_par, i_site
            fit_narrow_normal(θMs[i_par,i_site], priorsM[i_par], θmean_quant) 
        end
        # concatenate to a flat vector
        int_n_site = get_ca_int_PMs(n_site)
        int_n_site(vcat(priors_θP_mean, vec(priors_θMs_mean)))
    end
    y_global_o = Float32[] # TODO
    loss_elbo = get_loss_elbo(
        g_dev, transPMs_batch, f, py, y_global_o, interpreters; 
        solver.n_MC, cor_ends, priors_θ_mean)
    # test loss function once
    l0 = loss_elbo(ϕ0_dev, rng, first(train_loader)...)
    optf = Optimization.OptimizationFunction((ϕ, data) -> loss_elbo(ϕ, rng, data...)[1],
        Optimization.AutoZygote())
    optprob = OptimizationProblem(optf, CA.getdata(ϕ0_dev), train_loader)
    res = Optimization.solve(optprob, solver.alg; kwargs...)
    ϕc = interpreters.μP_ϕg_unc(res.u)
    (; ϕ = ϕc, θP = cpu_ca(apply_preserve_axes(transP, ϕc.μP)), resopt = res, interpreters)
end

function fit_narrow_normal(θi, prior, θmean_quant) 
    p_lower, p_upper = cdf(prior, θi) .+ (-θmean_quant, + θmean_quant)
    p_lower = max(1e-3, p_lower)
    p_upper = min(1 - 1e-3, p_upper)
    q_lower, q_upper = quantile.(prior, (p_lower, p_upper))
    d = fit(Normal, @qp_l(q_lower), @qp_u(q_upper))
end


"""
Create a loss function for parameter vector ϕ, given 
- g(x, ϕ): machine learning model 
- transPMS: transformation from unconstrained space to parameter space
- f(θMs, θP): mechanistic model 
- interpreters: assigning structure to pure vectors, see neg_elbo_transnorm_gf
- n_MC: number of Monte-Carlo sample to approximate the expected value across distribution

The loss function takes in addition to ϕ, data that changes with minibatch
- rng: random generator
- xM: matrix of covariates, sites in columns
- xP: drivers for the processmodel: Iterator of size n_site
- y_o, y_unc: matrix of observations and uncertainties, sites in columns
"""
function get_loss_elbo(g, transPMs, f, py, y_o_global, interpreters; 
    n_MC, cor_ends, priors_θ_mean)
    let g = g, transPMs = transPMs, f = f, py = py, y_o_global = y_o_global, n_MC = n_MC,
        cor_ends = cor_ends, interpreters = map(get_concrete, interpreters),
        priors_θ_mean = priors_θ_mean
        function loss_elbo(ϕ, rng, xM, xP, y_o, y_unc, i_sites)
            neg_elbo_transnorm_gf(rng, ϕ, g, transPMs, f, py,
                xM, xP, y_o, y_unc, i_sites, interpreters; n_MC, cor_ends, priors_θ_mean)
        end
    end
end
