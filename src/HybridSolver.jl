abstract type AbstractHybridSolver end

struct HybridPointSolver{A} <: AbstractHybridSolver
    alg::A
end

HybridPointSolver(; alg) = HybridPointSolver(alg)

function CommonSolve.solve(prob::AbstractHybridProblem, solver::HybridPointSolver;
    scenario=Val(()), rng=Random.default_rng(),
    gdevs = nothing, # get_gdev_MP(scenario)
    is_inferred::Val{is_infer} = Val(false),
    ad_backend_loss = AutoZygote(),
    epochs,
    epochs_callback = max(1, floor(Int, epochs/10)),
    callback = (state, loss_val) -> false,
    is_omitting_NaNbatches = false,
    is_omit_priors::Val{omit_priors} = Val(false),
    clusters::AbstractVector{<:Integer} = 
        1:first(get_hybridproblem_n_site_and_batch(prob; scenario)),
    cluster_rep = 1,
    kwargs...
) where {is_infer, omit_priors}
    gdevs = isnothing(gdevs) ? get_gdev_MP(scenario) : gdevs
    pt = get_hybridproblem_par_templates(prob; scenario)
    g, ϕg0 = get_hybridproblem_MLapplicator(prob; scenario)
    # TODO: separate parameters from problem description - right now optimized
    ϕq0 = get_hybridproblem_ϕq(prob; scenario)
    (; transP, transM) = get_hybridproblem_transforms(prob; scenario)
    n_site, n_batch = get_hybridproblem_n_site_and_batch(prob; scenario)
    ranef_spec = get_hybridproblem_ranef(prob; scenario)     
    ranef = get_ranef_computer(
        ranef_spec, keys(pt.θM), n_site, one(eltype(ϕq0)))
    ϕq_ranef = setup_ϕq_ranef(ranef)
    if :ranef ∉ keys(ϕq0)
        ϕq0 = CA.ComponentVector(ϕq0, ranef = ϕq_ranef)
    else
        @assert size(ϕq0[Val(:ranef)]) == size(ϕq_ranef)
    end
    ϕP0 = ϕq0[Val(:μP)]
    #intϕ = ComponentArrayInterpreter(CA.ComponentVector(ϕg=1:length(ϕg0), ϕq=ϕq0))
    #ϕ0_cpu = vcat(ϕg0, pt.θP .* FT(0.9))  # slightly disturb θP_true
    ϕ0_cpu = CA.ComponentVector(ϕg=ϕg0, ϕq=ϕq0)
    intϕ = ComponentArrayInterpreter(ϕ0_cpu)
    n_sites_cluster = [count(==(element),clusters) for element in 1:maximum(clusters)]
    frac_cluster_all = (1 / cluster_rep) ./ n_sites_cluster[clusters] 
    train_loader = get_hybridproblem_train_dataloader(prob; scenario)
    test_data = get_hybridproblem_test_data(prob; scenario) 
    # i_test = rand(1:n_site, Integer(floor(n_site/10)))
    # test_data = map(train_loader.data) do data_comp
    #     ndims(data_comp) == 2 ? data_comp[:, i_test] : data_comp[i_test]
    # end
    gdev = gdevs.gdev_M
    if gdev isa MLDataDevices.AbstractGPUDevice
        ϕ0_dev = gdev(ϕ0_cpu)
        g_dev = gdev(g)
        train_loader_dev = gdev_hybridproblem_dataloader(train_loader; gdevs)
        test_data_dev = gdev_hybridproblem_data(
            test_data[keys(test_data)[1:5]]; gdevs)
    else
        ϕ0_dev = ϕ0_cpu
        g_dev = g
        train_loader_dev = train_loader
        test_data_dev = test_data[keys(test_data)[1:5]]
    end
    f = get_hybridproblem_PBmodel(prob; scenario)
    ftest = create_nsite_applicator(f, size(test_data[1],2))
    if gdevs.gdev_P isa MLDataDevices.AbstractGPUDevice
        f_dev = gdevs.gdev_P(f) 
        ftest_dev = gdevs.gdev_P(ftest) 
    else
        f_dev = f
        ftest_dev = ftest
    end
    py = get_hybridproblem_neg_logden_obs(prob; scenario)
    pbm_covars = get_hybridproblem_pbmpar_covars(prob; scenario)
    n_site_test = size(test_data[1],2)
    priors = get_hybridproblem_priors(prob; scenario)
    priorsP = Tuple(priors[k] for k in keys(pt.θP))
    priorsM = Tuple(priors[k] for k in keys(pt.θM))
    #intP = ComponentArrayInterpreter(pt.θP)
    # intθP = ComponentArrayInterpreter(pt.θP)
    # intθM = ComponentArrayInterpreter(pt.θM)
    penalty_computer = get_hybridproblem_penalty_computer(prob; scenario)
    loss_gf = get_loss_gf(g_dev, transM, transP, f_dev,  py, intϕ;
        n_site_batch=n_batch, 
        par_templates = pt, 
        cdev=infer_cdev(gdevs), pbm_covars, 
        priorsP, priorsM, is_omit_priors, penalty_computer,
        #intθM, intθP, 
        frac_cluster_all, ranef,
        )
    loss_gf_test = get_loss_gf(g_dev, transM, transP, ftest_dev,  py, intϕ;
        n_site_batch=n_site_test,
        par_templates = pt,
        cdev=infer_cdev(gdevs), pbm_covars, 
        priorsP, priorsM, is_omit_priors, penalty_computer,
        #intθM, intθP, 
        frac_cluster_all, ranef,
        )
    # call loss function once
    l1 = is_infer ? 
        Test.@inferred(loss_gf(ϕ0_dev, first(train_loader_dev)...; is_testmode=true))[1] : 
        # using ShareAdd; @usingany Cthulhu
        # @descend_code_warntype loss_gf(ϕ0_dev, first(train_loader_dev)...)
        loss_gf(ϕ0_dev, first(train_loader_dev)...; is_testmode=true)[1]
    l1t = loss_gf_test(ϕ0_dev, test_data_dev...; is_testmode=true)[1]
    # and gradient
    # xMg, xP, y_o, y_unc = first(train_loader_dev)
    # gr1 = Zygote.gradient(
    #             p -> loss_gf(p, xMg, xP, y_o, y_unc)[1],
    #             ϕ0_dev)
    # Zygote.gradient(ϕ0_dev -> loss_gf(ϕ0_dev, data1...)[1], ϕ0_dev)
    # if is_omitting_NaNbatches 
    #     # implement training loop by hand to skip minibatches with NaN gradients
    #     ps = CA.getdata(ϕ0_dev)
    #     opt_st_new = Optimisers.setup(solver.alg, ps)
    #     n_skips = 0
    #     # prepare DI.gradient, need to access and update outside cope data_batch
    #     # because cannot redefine fopt_loss_gf
    #     data_batch = first(train_loader_dev)
    #     is_testmode = false
    #     function fopt_loss_gf(ϕ) 
    #         #@show first(data_batch[5], 2)
    #         loss_gf(ϕ, data_batch...; is_testmode)[1]
    #     end
    #     ad_prep = DI.prepare_gradient(fopt_loss_gf, ad_backend_loss, zero(ps))
    #     grad = similar(ps)
    #     stime = time()
    #     for epoch in 1:epochs
    #         is_testmode = false
    #         #i,data_batch = first(enumerate(loader))
    #         for (i, data_batch_) in enumerate(train_loader_dev)
    #             data_batch = data_batch_  # propagate outside for to scope of fopt_loss_gf
    #             DI.gradient!(fopt_loss_gf, grad, ad_prep, ad_backend_loss, ps)    
    #             if any(isnan.(grad))
    #                 n_skips += 1
    #                 #println("Skipped NaN : Batch $i")
    #                 print(",$i")
    #             else
    #                 Optimisers.update!(opt_st_new, ps, grad)
    #             end
    #         end
    #         ttime = time() - stime
    #         # compute loss for test data
    #         l = loss_gf_test(ps, test_data_dev...; is_testmode = true)
    #         println()
    #         @show round(ttime, digits=1), epoch, l.nLy, l.neg_log_prior, l.loss_penalty
    #         # TODO log 
    #     end
    #     res = nothing  
    #     ϕ = intϕ(ps)
    # else
        loss_test = let test_data_dev = test_data_dev
           (state) -> loss_gf_test(state.u, test_data_dev...; is_testmode=true) 
        end
        callback_epochs = get_callback_epochs(epochs_callback; 
            n_site, n_batch, callback, loss_test)
        optf = Optimization.OptimizationFunction((ϕ, data) -> loss_gf(ϕ, data...; is_testmode=false)[1],
            ad_backend_loss)
        # use CA.getdata(ϕ0_dev), i.e. the plain vector to avoid recompiling for specific CA
        # loss_gf re-attaches the axes
        optprob = OptimizationProblem(optf, CA.getdata(ϕ0_dev), train_loader_dev)
        res = Optimization.solve(optprob, solver.alg; 
            epochs, callback = callback_epochs, kwargs...)
        ϕ = intϕ(res.u)
    # end
    #θP = !isempty(ϕ.ϕP) ? cpu_ca(apply_preserve_axes(transP, cpu_ca(ϕ).ϕq.μP)) : CA.ComponentVector{eltype(ϕ)}()
    # TODO check which components live on gpu and which on cpu
    ϕq_opt = ComponentArrayInterpreter(ϕq0)(gdevs.gdev_P(ϕ.ϕq))
    probo = HybridProblem(prob; ϕg=cpu_ca(ϕ).ϕg, ϕq=ϕq_opt)
    (; ϕ, resopt=res, probo)
end

function get_callback_epochs(epochs_callback; 
    n_site, n_batch, callback, loss_test)
    n_per_epoch = n_site ÷ n_batch    
    if epochs_callback == 0 
        callback_epochs = callback
    else
        callback_epochs = function(state, l)
            if (state.iter == 1) || (state.iter % (n_per_epoch * epochs_callback) == 0)
                l_test = loss_test(state)[1]
                println("epoch = $((state.iter) ÷ n_per_epoch), iter = $(state.iter), loss_train=$l, loss_test=$l_test")
            end
            return callback(state, l)
        end
    end
end


struct HybridPosteriorSolver{A} <: AbstractHybridSolver
    alg::A
    n_MC::Int
    n_MC_cap::Int
end
function HybridPosteriorSolver(; alg, n_MC=12, n_MC_cap=n_MC)
    HybridPosteriorSolver(alg, n_MC, n_MC_cap)
end
function HybridPosteriorSolver(solver::HybridPosteriorSolver;
    alg=solver.alg,
    n_MC=solver.n_MC,
    n_MC_cap=n_MC)
    HybridPosteriorSolver(alg, n_MC, n_MC_cap)
end

"""
    solve(prob::AbstractHybridProblem, solver::HybridPosteriorSolver; epochs, ...)

Perform the inversion of HVI Problem.

Arguemtns
- `prob`: The AbstractHybridProblem to solve.
- `scenario`: Scenario to query prob, defaults to `Val(())`.
- `epochs`: number of epochs to train, i.e. number of passes through the whole dataset.

Optional keyword arguments
- `rng`: Random generator, defaults to `Random.default_rng()`.
- `gdevs`: `NamedTuple` `(;gdev_M, gdev_P)` functions to move
  computation and data of ML model on and PBM respectively
  to gpu (e.g. `gpu_device()` or cpu (`identity`). 
  defaults to [`get_gdev_MP`](@ref)`(scenario)`
- `θmean_quant` default to `0.0`: deprecated
- `is_inferred`: set to `Val(true)` to activate type stability checks
- `is_omit_priors`: set to `Val(true)` to omit priors in the loss computation, which can be useful for debugging or if priors are not implemented for a specific scenario (e.g. on gpu)
- `clusters`: vector of cluster assignments for each site, defaults to each site being its own cluster. Clusters are used to compute the loss in a way that accounts for clustering of sites, which can be useful if there are many sites and the number of Monte Carlo samples is limited.
- `cluster_rep`: number of times to repeat each cluster in the loss computation, defaults to 1. Repeating clusters can be useful to effectively increase the number of Monte Carlo samples when the number of clusters is small.
- `epochs_callback`: number of epochs between progress output on evaluating testdata

Returns a `NamedTuple` of
- `probo`: A copy of the HybridProblem, with updated optimized parameters
- `interpreters`:  TODO
- `ϕ`: the optimized HVI parameters: a `ComponentVector` with entries
  - `ϕg`: The ML model parameter vector, 
  - `ϕq`: `ComponentVector` of non-ML parameters, including 
    `μP`: `ComponentVector` of the mean global PBM parameters at unconstrained scale
- `θP`: `ComponentVector` of the mean global PBM parameters at constrained scale
- `resopt`: the structure returned by `Optimization.solve`. It can contain
  more information on convergence.
"""
function CommonSolve.solve(prob::AbstractHybridProblem, solver::HybridPosteriorSolver;
    epochs,
    scenario::Val{scen}=Val(()), rng=Random.default_rng(),
    gdevs = get_gdev_MP(scenario), 
    θmean_quant=0.0,
    is_inferred::Val{is_infer} = Val(false),
    is_omit_priors::Val{omit_priors} = Val(false),
    approx = prob.approx,
    clusters::AbstractVector{<:Integer} = 
        1:first(get_hybridproblem_n_site_and_batch(prob; scenario)),
    cluster_rep = 1, 
    epochs_callback = max(1, floor(Int, epochs/10)),
    callback = (state, loss_val) -> false,
    kwargs...
) where {scen, is_infer, omit_priors}
    pt = get_hybridproblem_par_templates(prob; scenario)
    cor_ends = get_hybridproblem_cor_ends(prob; scenario)
    g, ϕg0 = get_hybridproblem_MLapplicator(prob; scenario)
    (; transP, transM) = get_hybridproblem_transforms(prob; scenario)
    pbm_covars = get_hybridproblem_pbmpar_covars(prob; scenario)
    n_site, n_batch = get_hybridproblem_n_site_and_batch(prob; scenario)
    n_sites_cluster = [count(==(element),clusters) for element in 1:maximum(clusters)]
    frac_cluster_all = (1 / cluster_rep) ./ n_sites_cluster[clusters] 
    ranef_spec = get_hybridproblem_ranef(prob; scenario)     
    ranef = get_ranef_computer(
        ranef_spec, keys(pt.θM), n_site, one(eltype(pt.θM)))
    ϕq_ranef = setup_ϕq_ranef(ranef)
    ϕq = CA.ComponentVector(get_hybridproblem_ϕq(prob; scenario), ranef = ϕq_ranef)
    (; ϕ, interpreters) = init_hybrid_params(ϕg0, ϕq)
    int_ϕq = interpreters.ϕq
    int_ϕg_ϕq = interpreters.ϕg_ϕq
    priors = get_hybridproblem_priors(prob; scenario)
    priorsP = Tuple(priors[k] for k in keys(pt.θP))
    priorsM = Tuple(priors[k] for k in keys(pt.θM))
    zero_prior_logdensity = omit_priors ? 0f0 : get_zero_prior_logdensity(
        priorsP, priorsM, pt.θP, pt.θM)     
    train_loader = get_hybridproblem_train_dataloader(prob; scenario)
    test_data = get_hybridproblem_test_data(prob; scenario) 
    # i_test = rand(1:n_site, Integer(floor(n_site/10)))
    # test_data = map(train_loader.data) do data_comp
    #     ndims(data_comp) == 2 ? data_comp[:, i_test] : data_comp[i_test]
    # end
    n_batch_test = size(test_data[1],2)
    if first(train_loader)[1] isa CA.ComponentArray
        @warn("ML model covariates (1) were provided as ComponentArray. " * 
        "Consider providing them as a plain array.")
    end
    if first(train_loader)[2] isa CA.ComponentArray
        @warn("PBM drivers (2) were provided as ComponentArray. " * 
        "Consider providing them as a plain array.")
    end
    if gdevs.gdev_M isa MLDataDevices.AbstractGPUDevice
        ϕ0_dev = gdevs.gdev_M(ϕ)
        g_dev = gdevs.gdev_M(g) # zygote fails if  gdev is a CPUDevice, although should be non-op
        train_loader_dev = gdev_hybridproblem_dataloader(train_loader; gdevs)
        test_data_dev = gdev_hybridproblem_data(test_data[keys(test_data)[1:5]]; gdevs)
    else
        ϕ0_dev = ϕ
        g_dev = g
        train_loader_dev = train_loader
        test_data_dev = test_data[keys(test_data)[1:5]]
    end
    f = get_hybridproblem_PBmodel(prob; scenario)
    f_test = create_nsite_applicator(f, n_batch_test)
    if gdevs.gdev_P isa MLDataDevices.AbstractGPUDevice
        f_dev = gdevs.gdev_P(f) #fmap(gdevs.gdev_P, f)
        f_test_dev = gdevs.gdev_P(f_test)
    else
        f_dev = f
        f_test_dev = f_test
    end

    py = get_hybridproblem_neg_logden_obs(prob; scenario)

    penalty_computer = get_hybridproblem_penalty_computer(prob; scenario)
    # intθP = ComponentArrayInterpreter(pt.θP)
    # intθMs = ComponentArrayInterpreter((n_batch,), pt.θM)

    loss_elbo = get_loss_elbo(
        g_dev, transP, transM, f_dev, py, n_batch;
        n_MC = solver.n_MC, n_MC_cap = solver.n_MC_cap, cor_ends,  
        cdev=infer_cdev(gdevs), pbm_covars, 
        par_templates = pt,
        #pt.θP, 
        int_ϕq, int_ϕg_ϕq, priorsP, priorsM,
        is_omit_priors, zero_prior_logdensity, approx, penalty_computer, 
        ranef,
        #intθMs, intθP,
        frac_cluster_all,
        )
    loss_elbo_test = get_loss_elbo(
        g_dev, transP, transM, f_test_dev, py, n_batch_test;
        solver.n_MC, solver.n_MC_cap, cor_ends, 
        cdev=infer_cdev(gdevs), pbm_covars, 
        par_templates = pt,
        #pt.θP, 
        int_ϕq, int_ϕg_ϕq, priorsP, priorsM,
        is_omit_priors, zero_prior_logdensity, approx, penalty_computer, 
        ranef,
        #intθMs, intθP,
        frac_cluster_all,
        )

    # test loss function once
    # tmp = first(train_loader_dev)
    # using ShareAdd
    # @usingany Cthulhu
    # @descend_code_warntype loss_elbo(ϕ0_dev, rng, first(train_loader_dev)...)
    # omit for type stability in AD
    @assert length(first(train_loader_dev)) == 5
    l0 = 
    #is_infer ? 
    #     (Test.@inferred loss_elbo(ϕ0_dev, rng, first(train_loader_dev)...; is_testmode=true)) :
        loss_elbo(ϕ0_dev, rng, first(train_loader_dev)...; is_testmode=false)
    l0t = loss_elbo_test(ϕ0_dev, rng, test_data_dev...; is_testmode=true)
    loss_test = let test_data_dev = test_data_dev
        (state) -> loss_elbo_test(state.u, rng, test_data_dev...; is_testmode=true) 
    end
    callback_epochs = get_callback_epochs(epochs_callback; 
        n_site, n_batch, callback, loss_test)
    optf = Optimization.OptimizationFunction(
        (ϕ, data) -> first(loss_elbo(ϕ, rng, data...; is_testmode=false)),
        Optimization.AutoZygote())
    optprob = OptimizationProblem(optf, CA.getdata(ϕ0_dev), train_loader_dev)
    res = Optimization.solve(optprob, solver.alg; callback = callback_epochs, epochs, kwargs...)
    ϕc = interpreters.ϕg_ϕq(cpu_device()(res.u))
    ϕq = ϕc[Val(:ϕq)]; 
    ϕg = ϕc[Val(:ϕg)]; 
    probo = HybridProblem(prob; ϕg, ϕq)
    θP = get_hybridproblem_θP(probo)
    (; probo, interpreters, ϕ=ϕc, θP, resopt=res)
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
- `f(θMs_tr, θP)`: mechanistic model 
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
function get_loss_elbo(g, transP, transM, f, py, n_batch;
    n_MC, n_MC_mean = max(n_MC,20), n_MC_cap=n_MC, 
    cor_ends, cdev, pbm_covars, 
    par_templates, 
    #θP::AbstractVector{T},
    int_ϕq, int_ϕg_ϕq,
    priorsP, priorsM, penalty_computer = ZeroPenaltyComputer(),
    is_omit_priors, zero_prior_logdensity, approx,
    ranef::AbstractRandomEffectsComputer,
    #intθMs, intθP,
    frac_cluster_all,
) 
    T = eltype(par_templates.θP)
    intθP = ComponentArrayInterpreter(par_templates.θP)
    intθMs = ComponentArrayInterpreter((n_batch,), par_templates.θM)
    transMs = StackedArray(transM, n_batch)

    let g = g, transP = transP, transMs = transMs, f = f, py = py, 
        n_MC = n_MC, n_MC_cap = n_MC_cap, n_MC_mean = n_MC_mean,
        cor_ends = cor_ends,
        int_ϕq = get_concrete(int_ϕq), int_ϕg_ϕq = get_concrete(int_ϕg_ϕq),
        cdev = cdev,
        pbm_covar_indices = get_pbm_covar_indices(par_templates.θP, pbm_covars),
        trans_mP=StackedArray(transP, n_MC_mean), 
        trans_mMs=StackedArray(transMs.stacked, n_MC_mean),
        priorsP=priorsP, priorsM=priorsM, penalty_computer=penalty_computer,
        is_omit_priors = is_omit_priors, zero_prior_logdensity = zero_prior_logdensity,
        approx = approx,
        intθMs = get_concrete(intθMs), intθP = get_concrete(intθP),
        ranef = ranef
        frac_cluster_all = convert.(T, frac_cluster_all)


        function loss_elbo(ϕ, rng::Random.AbstractRNG, xM, xP, y_o, y_unc, i_sites; is_testmode)
            #ϕc = int_ϕg_ϕq(ϕ)
            neg_elbo_gtf(
                rng, ϕ, g, f, py, xM, xP, y_o, y_unc, i_sites;
                int_ϕq, int_ϕg_ϕq,
                n_MC, n_MC_cap, n_MC_mean, cor_ends, 
                cdev, pbm_covar_indices, transP, transMs, trans_mP, trans_mMs,
                priorsP, priorsM, penalty_computer, #ϕg = ϕc.ϕg, ϕq = ϕc.ϕq,
                is_testmode, is_omit_priors, zero_prior_logdensity, approx,
                intθMs, intθP, ranef, frac_cluster_all,
            )
        end
    end
end


function compute_elbo_components(
    prob::AbstractHybridProblem, solver::HybridPosteriorSolver; 
    scenario, kwargs...
    )
    train_loader = get_hybridproblem_train_dataloader(prob; scenario)
    data = train_loader.data 
    compute_elbo_components(
        prob::AbstractHybridProblem, solver::HybridPosteriorSolver, data; 
        scenario, kwargs...
        )
end

"""
Compute the components of the elbo for given initial conditions of the problems
for the first batch of the trainloader.
"""
function compute_elbo_components(
    prob::AbstractHybridProblem, solver::HybridPosteriorSolver, data::Tuple;
    scenario, rng=Random.default_rng(), gdev=gpu_device(),
    θmean_quant=0.0,
    kwargs...)
    n_site, n_batch = get_hybridproblem_n_site_and_batch(prob; scenario)
    pt = get_hybridproblem_par_templates(prob; scenario)
    (; θP, θM) = pt
    cor_ends = get_hybridproblem_cor_ends(prob; scenario)
    g, ϕg0 = get_hybridproblem_MLapplicator(prob; scenario)
    ϕq = get_hybridproblem_ϕq(prob; scenario)
    (; transP, transM) = get_hybridproblem_transforms(prob; scenario)
    (; ϕ, interpreters) = init_hybrid_params(ϕg0, ϕq)
    if gdev isa MLDataDevices.AbstractGPUDevice
        ϕ0_dev = gdev(ϕ)
        g_dev = gdev(g) # zygote fails if  gdev is a CPUDevice, although should be non-op
        data_dev = gdev_hybridproblem_data(data; scenario, gdev)
    else
        ϕ0_dev = ϕ
        g_dev = g
        data_dev = data
    end
    (xM, xP, y_o, y_unc, i_sites) = data_dev
    n_site_pred = size(xP,2)
    @assert size(xM, 2) == n_site_pred
    @assert size(y_o, 2) == n_site_pred
    @assert size(y_unc, 2) == n_site_pred
    @assert length(i_sites) == n_site_pred
    f_batch = get_hybridproblem_PBmodel(prob; scenario)
    f = (n_site_pred == n_batch) ? f : create_nsite_applicator(f_batch, n_site_pred)
    py = get_hybridproblem_neg_logden_obs(prob; scenario)
    priors_θ_mean = nothing
    # priors_θ_mean = construct_priors_θ_mean(
    #     prob, ϕ0_dev.ϕg, keys(θM), θP, θmean_quant, g_dev, transM;
    #     scenario, gdev, cdev, pbm_covars)
    neg_elbo_gtf_components(
        rng, ϕ0_dev, g_dev, transPMs_batch, f, py, xM, xP, y_o, y_unc, i_sites, interpreters;
        solver.n_MC, solver.n_MC_cap, cor_ends, priors_θ_mean)
end

