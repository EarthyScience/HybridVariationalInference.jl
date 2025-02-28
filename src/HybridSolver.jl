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
    train_loader = get_hybridproblem_train_dataloader(rng, prob; scenario, solver.n_batch)
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

function CommonSolve.solve(prob::AbstractHybridProblem, solver::HybridPosteriorSolver;
        scenario, rng = Random.default_rng(), dev = gpu_device(), kwargs...)
    par_templates = get_hybridproblem_par_templates(prob; scenario)
    (; θP, θM) = par_templates
    cor_ends = get_hybridproblem_cor_ends(prob; scenario)
    g, ϕg0 = get_hybridproblem_MLapplicator(prob; scenario)
    (; transP, transM) = get_hybridproblem_transforms(prob; scenario)
    (; ϕ, transPMs_batch, interpreters, get_transPMs, get_ca_int_PMs) = init_hybrid_params(
        θP, θM, cor_ends, ϕg0, solver.n_batch; transP, transM)
    if dev isa MLDataDevices.AbstractGPUDevice
        ϕ0_dev = dev(ϕ)
        g_dev = dev(g) # zygote fails if  dev is a CPUDevice, although should be non-op
    else
        ϕ0_dev = ϕ
        g_dev = g
    end
    train_loader = get_hybridproblem_train_dataloader(rng, prob; scenario, solver.n_batch)
    f = get_hybridproblem_PBmodel(prob; scenario)
    py = get_hybridproblem_neg_logden_obs(prob; scenario)
    y_global_o = Float32[] # TODO
    loss_elbo = get_loss_elbo(
        g_dev, transPMs_batch, f, py, y_global_o, interpreters; solver.n_MC, cor_ends)
    # test loss function once
    l0 = loss_elbo(ϕ0_dev, rng, first(train_loader)...)
    optf = Optimization.OptimizationFunction((ϕ, data) -> loss_elbo(ϕ, rng, data...)[1],
        Optimization.AutoZygote())
    optprob = OptimizationProblem(optf, CA.getdata(ϕ0_dev), train_loader)
    res = Optimization.solve(optprob, solver.alg; kwargs...)
    ϕc = interpreters.μP_ϕg_unc(res.u)
    (; ϕ = ϕc, θP = cpu_ca(apply_preserve_axes(transP, ϕc.μP)), resopt = res, interpreters)
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
function get_loss_elbo(g, transPMs, f, py, y_o_global, interpreters; n_MC, cor_ends)
    let g = g, transPMs = transPMs, f = f, py = py, y_o_global = y_o_global, n_MC = n_MC,
        cor_ends = cor_ends, interpreters = map(get_concrete, interpreters)

        function loss_elbo(ϕ, rng, xM, xP, y_o, y_unc)
            neg_elbo_transnorm_gf(rng, ϕ, g, transPMs, f, py,
                xM, xP, y_o, y_unc, interpreters; n_MC, cor_ends)
        end
    end
end
