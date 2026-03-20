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
    is_omitting_NaNbatches = false,
    is_omit_priors::Val{omit_priors} = Val(false),
    kwargs...
) where {is_infer, omit_priors}
    gdevs = isnothing(gdevs) ? get_gdev_MP(scenario) : gdevs
    pt = get_hybridproblem_par_templates(prob; scenario)
    g, ϕg0 = get_hybridproblem_MLapplicator(prob; scenario)
    FT = get_hybridproblem_float_type(prob; scenario)
    (; transP, transM) = get_hybridproblem_transforms(prob; scenario)
    ϕq0 = get_hybridproblem_ϕq(prob; scenario)
    ϕP0 = ϕq0[Val(:μP)]
    intϕ = ComponentArrayInterpreter(CA.ComponentVector(ϕg=1:length(ϕg0), ϕP=ϕP0))
    #ϕ0_cpu = vcat(ϕg0, pt.θP .* FT(0.9))  # slightly disturb θP_true
    ϕ0_cpu = vcat(ϕg0, ϕP0)
    n_site, n_batch = get_hybridproblem_n_site_and_batch(prob; scenario)
    train_loader = get_hybridproblem_train_dataloader(prob; scenario)
    #TODO provide different test data
    # TODO use 1/10 of the training data
    # currently HybridProblem returns only applciators of size n_batch and n_site
    # i_test = rand(1:n_site, Integer(floor(n_site/10)))
    # test_data = map(train_loader.data) do data_comp
    #     ndims(data_comp) == 2 ? data_comp[:, i_test] : data_comp[i_test]
    # end
    test_data = train_loader.data
    gdev = gdevs.gdev_M
    if gdev isa MLDataDevices.AbstractGPUDevice
        ϕ0_dev = gdev(ϕ0_cpu)
        g_dev = gdev(g)
        train_loader_dev = gdev_hybridproblem_dataloader(train_loader; gdevs)
        test_data_dev = gdev_hybridproblem_data(test_data; gdevs)
    else
        ϕ0_dev = ϕ0_cpu
        g_dev = g
        train_loader_dev = train_loader
        test_data_dev = test_data
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
    zero_prior_logdensity = omit_priors ? 0f0 : get_zero_prior_logdensity(
        priorsP, priorsM, pt.θP, pt.θM)     
    #intP = ComponentArrayInterpreter(pt.θP)
    loss_gf = get_loss_gf(g_dev, transM, transP, f_dev,  py, intϕ;
        n_site_batch=n_batch, 
        cdev=infer_cdev(gdevs), pbm_covars, 
        priorsP, priorsM, is_omit_priors, zero_prior_logdensity,)
    loss_gf_test = get_loss_gf(g_dev, transM, transP, ftest_dev,  py, intϕ;
        n_site_batch=n_site_test,
        cdev=infer_cdev(gdevs), pbm_covars, 
        priorsP, priorsM, is_omit_priors, zero_prior_logdensity,)
    # call loss function once
    l1 = is_infer ? 
        Test.@inferred(loss_gf(ϕ0_dev, first(train_loader_dev)...; is_testmode=true))[1] : 
        # using ShareAdd; @usingany Cthulhu
        # @descend_code_warntype loss_gf(ϕ0_dev, first(train_loader_dev)...)
        loss_gf(ϕ0_dev, first(train_loader_dev)...; is_testmode=true)[1]
    # and gradient
    # xMg, xP, y_o, y_unc = first(train_loader_dev)
    # gr1 = Zygote.gradient(
    #             p -> loss_gf(p, xMg, xP, y_o, y_unc)[1],
    #             ϕ0_dev)
    # Zygote.gradient(ϕ0_dev -> loss_gf(ϕ0_dev, data1...)[1], ϕ0_dev)
    if is_omitting_NaNbatches 
        # implement training loop by hand to skip minibatches with NaN gradients
        ps = CA.getdata(ϕ0_dev)
        opt_st_new = Optimisers.setup(solver.alg, ps)
        n_skips = 0
        # prepare DI.gradient, need to access and update outside cope data_batch
        # because cannot redefine fopt_loss_gf
        data_batch = first(train_loader_dev)
        is_testmode = false
        function fopt_loss_gf(ϕ) 
            #@show first(data_batch[5], 2)
            loss_gf(ϕ, data_batch...; is_testmode)[1]
        end
        ad_prep = DI.prepare_gradient(fopt_loss_gf, ad_backend_loss, zero(ps))
        grad = similar(ps)
        stime = time()
        for epoch in 1:epochs
            is_testmode = false
            #i,data_batch = first(enumerate(loader))
            for (i, data_batch_) in enumerate(train_loader_dev)
                data_batch = data_batch_  # propagate outside for to scope of fopt_loss_gf
                DI.gradient!(fopt_loss_gf, grad, ad_prep, ad_backend_loss, ps)    
                if any(isnan.(grad))
                    n_skips += 1
                    #println("Skipped NaN : Batch $i")
                    print(",$i")
                else
                    Optimisers.update!(opt_st_new, ps, grad)
                end
            end
            ttime = time() - stime
            # compute loss for test data
            l = loss_gf_test(ps, test_data_dev...; is_testmode = true)
            println()
            @show round(ttime, digits=1), epoch, l.nLy, l.neg_log_prior, l.loss_penalty
            # TODO log 
        end
        res = nothing  
        ϕ = intϕ(ps)
    else
        optf = Optimization.OptimizationFunction((ϕ, data) -> loss_gf(ϕ, data...; is_testmode=false)[1],
            ad_backend_loss)
        # use CA.getdata(ϕ0_dev), i.e. the plain vector to avoid recompiling for specific CA
        # loss_gf re-attaches the axes
        optprob = OptimizationProblem(optf, CA.getdata(ϕ0_dev), train_loader_dev)
        res = Optimization.solve(optprob, solver.alg; epochs, kwargs...)
        ϕ = intϕ(res.u)
    end
    θP = !isempty(ϕ.ϕP) ? cpu_ca(apply_preserve_axes(transP, cpu_ca(ϕ).ϕP)) : CA.ComponentVector{eltype(ϕ)}()
    probo = HybridProblem(prob; ϕg=cpu_ca(ϕ).ϕg, θP)
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
function HybridPosteriorSolver(solver::HybridPosteriorSolver;
    alg=solver.alg,
    n_MC=solver.n_MC,
    n_MC_cap=n_MC)
    HybridPosteriorSolver(alg, n_MC, n_MC_cap)
end

"""
    solve(prob::AbstractHybridProblem, solver::HybridPosteriorSolver; ...)

Perform the inversion of HVI Problem.

Optional keyword arguments
- `prob`: The AbstractHybridProblem to solve.
- `scenario`: Scenario to query prob, defaults to `Val(())`.
- `rng`: Random generator, defaults to `Random.default_rng()`.
- `gdevs`: `NamedTuple` `(;gdev_M, gdev_P)` functions to move
  computation and data of ML model on and PBM respectively
  to gpu (e.g. `gpu_device()` or cpu (`identity`). 
  defaults to [`get_gdev_MP`](@ref)`(scenario)`
- `θmean_quant` default to `0.0`: deprecated
- `is_inferred`: set to `Val(true)` to activate type stability checks

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
    scenario::Val{scen}=Val(()), rng=Random.default_rng(),
    gdevs = get_gdev_MP(scenario), 
    θmean_quant=0.0,
    is_inferred::Val{is_infer} = Val(false),
    is_omit_priors::Val{omit_priors} = Val(false),
    approx = prob.approx,
    kwargs...
) where {scen, is_infer, omit_priors}
    pt = get_hybridproblem_par_templates(prob; scenario)
    cor_ends = get_hybridproblem_cor_ends(prob; scenario)
    g, ϕg0 = get_hybridproblem_MLapplicator(prob; scenario)
    (; transP, transM) = get_hybridproblem_transforms(prob; scenario)
    pbm_covars = get_hybridproblem_pbmpar_covars(prob; scenario)
    n_site, n_batch = get_hybridproblem_n_site_and_batch(prob; scenario)
    ϕq = get_hybridproblem_ϕq(prob; scenario)
    (; ϕ, interpreters) = init_hybrid_params(ϕg0, ϕq)
    int_ϕq = interpreters.ϕq
    int_ϕg_ϕq = interpreters.ϕg_ϕq
    transMs = StackedArray(transM, n_batch)
    priors = get_hybridproblem_priors(prob; scenario)
    priorsP = Tuple(priors[k] for k in keys(pt.θP))
    priorsM = Tuple(priors[k] for k in keys(pt.θM))
    zero_prior_logdensity = omit_priors ? 0f0 : get_zero_prior_logdensity(
        priorsP, priorsM, pt.θP, pt.θM)     
    train_loader = get_hybridproblem_train_dataloader(prob; scenario)
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
    else
        ϕ0_dev = ϕ
        g_dev = g
        train_loader_dev = train_loader
    end
    f = get_hybridproblem_PBmodel(prob; scenario)
    if gdevs.gdev_P isa MLDataDevices.AbstractGPUDevice
        f_dev = gdevs.gdev_P(f) #fmap(gdevs.gdev_P, f)
    else
        f_dev = f
    end

    py = get_hybridproblem_neg_logden_obs(prob; scenario)

    priors_θP_mean, priors_θMs_mean = construct_priors_θ_mean(
        prob, ϕ0_dev.ϕg, keys(pt.θM), pt.θP, θmean_quant, g_dev, transM, transP;
        scenario, gdevs, pbm_covars)

    loss_elbo = get_loss_elbo(
        g_dev, transP, transMs, f_dev, py;
        solver.n_MC, solver.n_MC_cap, cor_ends, priors_θP_mean, priors_θMs_mean, 
        cdev=infer_cdev(gdevs), pbm_covars, pt.θP, int_ϕq, int_ϕg_ϕq, priorsP, priorsM,
        is_omit_priors, zero_prior_logdensity, approx,
        )
    # test loss function once
    # tmp = first(train_loader_dev)
    # using ShareAdd
    # @usingany Cthulhu
    # @descend_code_warntype loss_elbo(ϕ0_dev, rng, first(train_loader_dev)...)
    # omit for type stability in AD
    l0 = 
    #is_infer ? 
    #     (Test.@inferred loss_elbo(ϕ0_dev, rng, first(train_loader_dev)...; is_testmode=true)) :
        loss_elbo(ϕ0_dev, rng, first(train_loader_dev)...; is_testmode=false)
    optf = Optimization.OptimizationFunction(
        (ϕ, data) -> first(loss_elbo(ϕ, rng, data...; is_testmode=false)),
        Optimization.AutoZygote())
    optprob = OptimizationProblem(optf, CA.getdata(ϕ0_dev), train_loader_dev)
    res = Optimization.solve(optprob, solver.alg; kwargs...)
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
function get_loss_elbo(g, transP, transMs, f, py;
    n_MC, n_MC_mean = max(n_MC,20), n_MC_cap=n_MC, 
    cor_ends, priors_θP_mean, priors_θMs_mean, cdev, pbm_covars, θP,
    int_ϕq, int_ϕg_ϕq,
    priorsP, priorsM, floss_penalty = zero_penalty_loss,
    is_omit_priors, zero_prior_logdensity, approx,
)
    let g = g, transP = transP, transMs = transMs, f = f, py = py, 
        n_MC = n_MC, n_MC_cap = n_MC_cap, n_MC_mean = n_MC_mean,
        cor_ends = cor_ends,
        int_ϕq = get_concrete(int_ϕq), int_ϕg_ϕq = get_concrete(int_ϕg_ϕq),
        priors_θP_mean = priors_θP_mean, priors_θMs_mean = priors_θMs_mean, cdev = cdev,
        pbm_covar_indices = get_pbm_covar_indices(θP, pbm_covars),
        trans_mP=StackedArray(transP, n_MC_mean), 
        trans_mMs=StackedArray(transMs.stacked, n_MC_mean),
        priorsP=priorsP, priorsM=priorsM, floss_penalty=floss_penalty,
        is_omit_priors = is_omit_priors, zero_prior_logdensity = zero_prior_logdensity,
        approx = approx

        function loss_elbo(ϕ, rng, xM, xP, y_o, y_unc, i_sites; is_testmode)
            #ϕc = int_ϕg_ϕq(ϕ)
            neg_elbo_gtf(
                rng, ϕ, g, f, py, xM, xP, y_o, y_unc, i_sites;
                int_ϕq, int_ϕg_ϕq,
                n_MC, n_MC_cap, n_MC_mean, cor_ends, priors_θP_mean, priors_θMs_mean,
                cdev, pbm_covar_indices, transP, transMs, trans_mP, trans_mMs,
                priorsP, priorsM, floss_penalty, #ϕg = ϕc.ϕg, ϕq = ϕc.ϕq,
                is_testmode, is_omit_priors, zero_prior_logdensity, approx,
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
    priors_θ_mean = construct_priors_θ_mean(
        prob, ϕ0_dev.ϕg, keys(θM), θP, θmean_quant, g_dev, transM;
        scenario, gdev, cdev, pbm_covars)
    neg_elbo_gtf_components(
        rng, ϕ0_dev, g_dev, transPMs_batch, f, py, xM, xP, y_o, y_unc, i_sites, interpreters;
        solver.n_MC, solver.n_MC_cap, cor_ends, priors_θ_mean)
end

"""
In order to let mean of θ stay close to initial point parameter estimates 
construct a prior on mean θ to a Normal around initial prediction.
"""
function construct_priors_θ_mean(prob, ϕg, keysθM, θP, θmean_quant, g_dev, transM, transP;
    scenario::Val{scen}, gdevs, pbm_covars,
    ) where {scen}
    iszero(θmean_quant) ? ([],[]) :
    begin
        gdev=gdevs.gdev_M
        #cdev=infer_cdev(gdevs)
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
        # θMs_tr = transMs(ζMs_cpu)
        θMs_tr = gtrans(
            g_dev, transMs, xMP_all, CA.getdata(ϕg); cdev=cpu_device(), is_testmode = true)
        priors_dict = get_hybridproblem_priors(prob; scenario)
        priorsP = [priors_dict[k] for k in keys(θP)]
        priors_θP_mean = map(priorsP, θP) do priorsP, θPi
            fit_narrow_normal(θPi, priorsP, θmean_quant)
        end
        priorsM = Tuple(priors_dict[k] for k in keysθM) 
        i_par = 1
        i_site = 1
        priors_θMs_mean = map(Iterators.product(axes(θMs_tr)...)) do (i_site, i_par)
            #@show i_par, i_site
            fit_narrow_normal(θMs_tr[i_site, i_par], priorsM[i_par], θmean_quant)
        end
        # # concatenate to a flat vector
        # int_n_site = get_ca_int_PMs(n_site)
        # int_n_site(vcat(priors_θP_mean, vec(priors_θMs_mean)))
        priors_θP_mean, priors_θMs_mean
    end
end
