abstract type AbstractHybridSolver end

struct HybridPointSolver{A} <: AbstractHybridSolver
    alg::A
    n_batch::Int
end

HybridPointSolver(; alg, n_batch = 10) = HybridPointSolver(alg,n_batch)
#HybridPointSolver(; alg = Adam(0.02), n_batch = 10) = HybridPointSolver(alg,n_batch)


function CommonSolve.solve(prob::AbstractHybridProblem, solver::HybridPointSolver; 
    scenario, rng = Random.default_rng(), kwargs...)
    par_templates = get_hybridproblem_par_templates(prob; scenario)
    g, ϕg0 = get_hybridproblem_MLapplicator(prob; scenario);
    FT = get_hybridproblem_float_type(prob; scenario)
    (; transP, transM) = get_hybridproblem_transforms(prob; scenario)
    int_ϕθP = ComponentArrayInterpreter(CA.ComponentVector(
        ϕg = 1:length(ϕg0), θP = par_templates.θP))
    #p0_cpu = vcat(ϕg0, par_templates.θP .* FT(0.9))  # slightly disturb θP_true
    p0_cpu = vcat(ϕg0, par_templates.θP)  
    p0 = (:use_Flux ∈ scenario) ? CuArray(p0_cpu) : p0_cpu
    train_loader = get_hybridproblem_train_dataloader(rng, prob; scenario, solver.n_batch)    
    f = get_hybridproblem_PBmodel(prob; scenario)
    y_global_o = FT[] # TODO
    loss_gf = get_loss_gf(g, transM, f, y_global_o, int_ϕθP)
    #l1 = loss_gf(p0, train_loader...)[1]
    # Zygote.gradient(p0 -> loss_gf(p0, data1...)[1], p0)
    optf = Optimization.OptimizationFunction((ϕ, data) -> loss_gf(ϕ, data...)[1],
        Optimization.AutoZygote())
    optprob = OptimizationProblem(optf, p0, train_loader)
    res = Optimization.solve(optprob, solver.alg; kwargs...)
    (;ϕ = int_ϕθP(res.u), resopt = res)
end



struct HybridPosteriorSolver{A} <: AbstractHybridSolver
    alg::A
    n_batch::Int
    n_MC::Int

end
HybridPosteriorSolver(; alg, n_batch = 10, n_MC = 3) = HybridPointSolver(alg, n_batch, n_MC)

function CommonSolve.solve(prob::AbstractHybridProblem, solver::HybridPosteriorSolver; 
    scenario, rng = Random.default_rng(), kwargs...)
    par_templates = get_hybridproblem_par_templates(prob; scenario)
    g, ϕg0 = get_hybridproblem_MLapplicator(prob; scenario);
    (; transP, transM) = get_hybridproblem_transforms(prob; scenario)
    (; ϕ, transPMs_batch, interpreters, get_transPMs, get_ca_int_PMs) = init_hybrid_params(
        θP_true, θMs_true[:, 1], ϕg0, solver.n_batch; transP, transM);
    use_gpu = (:use_Flux ∈ scenario)
    # ϕd = use_gpu ? CuArray(ϕ) : ϕ
    # train_loader = get_hybridproblem_train_dataloader(rng, prob; scenario, solver.n_batch)    
    # f = get_hybridproblem_PBmodel(prob; scenario)
    # y_global_o = Float32[] # TODO
    # loss_gf = get_loss_gf(g, transM, f, y_global_o, int_ϕθP)
    # optf = Optimization.OptimizationFunction((ϕ, data) -> loss_gf(ϕ, data...)[1],
    #     Optimization.AutoZygote())
    # optprob = OptimizationProblem(optf, p0, train_loader)
    # res = Optimization.solve(optprob, solver.alg; kwargs...)
end

