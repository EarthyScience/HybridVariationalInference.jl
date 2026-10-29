"""
A Solver that solves for the mode of hybrid variational inference problem
([`AbstractHybridProblem`](@ref)).


It does not perform the full posterior inference, 
but only finds the optimal parameters of the variational distribution. 
This is useful for initializing the full posterior inference.

- `alg`: The optimization algorithm to use for solving the problem, e.g. `Adam()`
"""
struct HybridPointSolverSite{NMC,A} <: AbstractHybridSolver
    alg::A
    n_MC_val::Val{NMC}
end
function HybridPointSolverSite(;alg = Optimisers.Adam(), n_MC_val=Val(3)) 
    HybridPointSolverSite(alg, n_MC_val)
end

function CommonSolve.solve(prob::AbstractHybridProblem, solver::HybridPointSolverSite{NMC_ELBO}, 
    loss_mode::AbstractElboLossMode, ϕ0;
    scenario=Val(()), rng=Random.default_rng(),
    epochs,
    epochs_callback = max(1, floor(Int, epochs/10)),
    show_train_loss::Val{TRAIN} = Val(false),
) where {NMC_ELBO, TRAIN}
    NMC = loss_mode isa NegLogDensity ? 1 : NMC_ELBO
    train_loader = get_hybridproblem_train_dataloader(prob; scenario)
    # TODO replace key itrain_sites by i_indiv_train 
    tmp = get_hybridproblem_test_data(prob; scenario) 
    indiv_args_test = (;tmp[keys(tmp)[1:4]]..., i_indiv_train = tmp.itrain_sites)
    indiv_args1 = NamedTuple{keys(indiv_args_test)}(first(train_loader))
    # TODO replace n_site by n_indiv
    n_obs, n_site = size(indiv_args1.y_o)
    n_indiv_test = size(indiv_args_test.y_o,2)
    #
    par_templates = get_hybridproblem_par_templates(prob; scenario)
    n_θP = length(par_templates.θP)
    n_θM = length(par_templates.θM)
    # TODO: separate parameters from problem description - right now optimized
     g, ϕg0 = get_hybridproblem_MLapplicator(prob; scenario)
    (; transP, transM) = get_hybridproblem_transforms(prob; scenario)
    cor_ends = get_hybridproblem_cor_ends(prob; scenario)
    approx1 = get_hybridproblem_HVIApproximation(prob; scenario)
    #
    # TODO make init_hybrid_ϕunc already return proper ϕqPc1 and ϕqIc1
    ϕq, approx = init_hybrid_ϕunc(approx1, cor_ends; transM, par_templates.θM, n_site)
    ϕqPc1 = CA.ComponentVector(
        μζP = log.(CA.getdata(par_templates.θP)),
        #logσ_ζP = ones(n_θP) .* log(0.01),
        logσ_ζP = ϕq.logσ_ζP,
        ρsP = ϕq.ρsP,
        )
    ϕqIc1 = CA.ComponentVector(
        logσ_ζM_offsets = ϕq.logσ_ζM_offsets,
        ρsM = ϕq.ρsM,
        )
    intϕqP = get_concrete(ComponentArrayInterpreter(ϕqPc1))
    intϕqI = get_concrete(ComponentArrayInterpreter(ϕqIc1))
    if ϕ0 == NamedTuple()
        ϕqP = CA.getdata(ϕqPc1)
        ϕqI = CA.getdata(ϕqIc1)
        ϕ0 = (;ϕg = CA.getdata(ϕg0), ϕqP, ϕqI) # copied - not modified
    else
        @assert length(ϕ0.ϕqP) == length(ϕqPc1)
        @assert length(ϕ0.ϕqI) == length(ϕqIc1)
    end
    #
    diffchunk = ForwardDiff.Chunk(8)
    pbm = create_nsite_applicator(get_hybridproblem_PBmodel(prob; scenario), NMC; diffchunk)      
    nljoint_args_fix =(;
         f! = pbm, 
         fLogLik = get_hybridproblem_neg_logden_obs(prob; scenario)
         )
    n_M = get_numberof_MLinputs(approx, par_templates.θM)
    n_cov = get_hybridproblem_n_covar(prob; scenario)
    pbm_covars = get_hybridproblem_pbmpar_covars(prob; scenario)
    n_covP = length(pbm_covars)

    h0 = prepare_elbo_helpers(approx, ϕ0.ϕg, ϕ0.ϕqP; n_θP, n_θM, n_site, n_MC=NMC, n_cov, n_covP, n_M, cor_ends, n_obs);
    h0_test = prepare_elbo_helpers(approx, ϕ0.ϕg, ϕ0.ϕqP; n_θP, n_θM, n_site = n_indiv_test, n_MC=NMC, n_cov, n_covP, n_M, cor_ends, n_obs);
    if TRAIN
        indiv_args_train =  NamedTuple{keys(indiv_args_test)}(train_loader.data)
        n_indiv_train = size(indiv_args_train.xM,2)
        h0_train = prepare_elbo_helpers(approx, ϕ0.ϕg, ϕ0.ϕqP; n_θP, n_θM, n_site = n_indiv_train, n_MC=NMC, n_cov, n_covP, n_M, cor_ends, n_obs);
    end
    rnormPM = prepare_rnorm(ϕ0.ϕqP; n_θP, n_θM, n_MC=NMC, n_site)
    #
    py = get_hybridproblem_neg_logden_obs(prob; scenario)
    # priors = get_hybridproblem_priors(prob; scenario)
    # priorsP = Tuple(priors[k] for k in keys(par_templates.θP))
    # priorsM = Tuple(priors[k] for k in keys(par_templates.θM))
    pbm_covar_indices = SA.SVector{length(pbm_covars)}([findfirst(==(k), keys(par_templates.θP)) for k in pbm_covars])        
    sample_args = (;
        approx, g, is_testmode = false, pbm_covar_indices, intϕqP, intϕqI, cor_ends, 
        transP, transM)
    sample_args_test = (;sample_args..., is_testmode = true)
    #nljoint_args_fix_test = (;nljoint_args_fix..., f! = ) 
    ϕ = map(copy, ϕ0) # do not modify the initial guess
    # randnPM!(rng, rnormPM)
    #primal = neg_elbo_sites!(h0, loss_mode, ϕ, rnormPM, sample_args, indiv_args1, nljoint_args_fix)
    # res0, gradh0 = grad_neg_elbo_sites(
    #     h0, (;), loss_mode, ϕ, rnormPM, sample_args, indiv_args1, nljoint_args_fix; solver.n_MC_val,
    #     #n_workers = 3,
    #     )
    #
    opt_state = Optimisers.setup(solver.alg, ϕ);  # initialise this optimiser's momentum etc.
    n_skips = 0
    # epoch = 1
    gradh = (;)
    stime = time()
    for epoch in 1:epochs
        is_testmode = false
        #i,data_batch_ = first(enumerate(train_loader))
        for (i, data_batch_) in enumerate(train_loader)
            indiv_args = NamedTuple{keys(indiv_args_test)}(data_batch_)
            #randnPM!(rng, rnormPM)   # only required for NegElbo()
            grad_cv, gradh = grad_neg_elbo_sites(
            #grad_cv, gradh = grad_neg_elbo_sites_enzyme(
                h0, gradh, loss_mode, ϕ, rnormPM, sample_args, indiv_args, nljoint_args_fix; 
                solver.n_MC_val,
                #n_workers = 3,
                )
            #grad_cv.ϕqP .= zero(eltype(grad_cv.ϕqP)) # TODO remove fixing P gradient
            if any(map(x -> any(isnan.(x)), grad_cv))
                @show grad_cv.ϕqP, grad_cv.ϕqI
                error("encountered non-finite gradient grad_v")
                n_skips += 1
                #println("Skipped NaN : Batch $i")
                print(",$i")
            else
                Optimisers.update!(opt_state, ϕ, grad_cv)
            end
        end
        if (epoch % epochs_callback) == 0
            ttime = time() - stime
            # compute loss for test data
            loss_test = neg_elbo_sites!(h0_test, loss_mode, ϕ, rnormPM, sample_args_test, indiv_args_test, nljoint_args_fix)
            @show round(ttime, digits=1), epoch, loss_test
            #@show exp.(ϕ.ϕqP[1:2]), grad_cv.ϕqP
            if TRAIN
                loss_train = neg_elbo_sites!(h0_train, loss_mode, ϕ, rnormPM, sample_args_test, indiv_args_train, nljoint_args_fix)
                @show loss_train
            end
        end
    end
    (;ϕ, h0, h0_test)
end

function predict_point!(elbo_helpers, prob::AbstractHybridProblem, ϕ; scenario
    ,keys_predict_indiv = (:ζsM, :θsM, :y))
    h0 = elbo_helpers
    train_loader = get_hybridproblem_train_dataloader(prob; scenario)
    keys_indiv = (:xM, :xP, :y_o, :y_unc, :i_indiv_train)
    indiv_args1 = NamedTuple{keys_indiv}(first(train_loader))
    # TODO replace n_site by n_indiv
    n_indiv_train, n_batch  = get_hybridproblem_n_site_and_batch(prob; scenario)
    n_obs, n_site = size(indiv_args1.y_o)
    #
    par_templates = get_hybridproblem_par_templates(prob; scenario)
    n_θP = length(par_templates.θP)
    n_θM = length(par_templates.θM)
    g, ϕg = get_hybridproblem_MLapplicator(prob; scenario)
    # TODO: separate parameters from problem description - right now optimized
    (; transP, transM) = get_hybridproblem_transforms(prob; scenario)
    cor_ends = get_hybridproblem_cor_ends(prob; scenario)
    approx1 = get_hybridproblem_HVIApproximation(prob; scenario)
    #
    # TODO make init_hybrid_ϕunc already return proper ϕqPc1 and ϕqIc1
    ϕq, approx = init_hybrid_ϕunc(approx1, cor_ends; transM, par_templates.θM, n_site)
    ϕqPc1 = CA.ComponentVector(
        μζP = log.(CA.getdata(par_templates.θP)),
        #logσ_ζP = ones(n_θP) .* log(0.01),
        logσ_ζP = ϕq.logσ_ζP,
        ρsP = ϕq.ρsP,
        )
    ϕqIc1 = CA.ComponentVector(
        logσ_ζM_offsets = ϕq.logσ_ζM_offsets,
        ρsM = ϕq.ρsM,
        )
    ϕqP = ϕqP2 = CA.getdata(ϕqPc1)
    ϕqI = CA.getdata(ϕqIc1)
    intϕqP = get_concrete(ComponentArrayInterpreter(ϕqPc1))
    intϕqI = get_concrete(ComponentArrayInterpreter(ϕqIc1))
    #
    diffchunk = ForwardDiff.Chunk(8)
    pbm = create_nsite_applicator(get_hybridproblem_PBmodel(prob; scenario), 1; diffchunk)      
    nljoint_args_fix =(;
         f! = pbm, 
         fLogLik = get_hybridproblem_neg_logden_obs(prob; scenario)
         )
    n_M = get_numberof_MLinputs(approx, par_templates.θM)
    n_cov = get_hybridproblem_n_covar(prob; scenario)
    pbm_covars = get_hybridproblem_pbmpar_covars(prob; scenario)
    n_covP = length(pbm_covars)
    rnormPM = prepare_rnorm(ϕqP; n_θP, n_θM, n_MC=1, n_site)
    #
    py = get_hybridproblem_neg_logden_obs(prob; scenario)
    # priors = get_hybridproblem_priors(prob; scenario)
    # priorsP = Tuple(priors[k] for k in keys(par_templates.θP))
    # priorsM = Tuple(priors[k] for k in keys(par_templates.θM))
    pbm_covar_indices = SA.SVector{length(pbm_covars)}([findfirst(==(k), keys(par_templates.θP)) for k in pbm_covars])        
    sample_args = (;
        approx, g, is_testmode = false, pbm_covar_indices, intϕqP, intϕqI, cor_ends, 
        transP, transM)
    sample_args_test = (;sample_args..., is_testmode = true)
    #nljoint_args_fix_test = (;nljoint_args_fix..., f! = ) 
    # convert from DiffCache to normal vector
    h0p = if h0.helpers_sites[1][1] isa PAT.DiffCache 
        h0p = (; h0..., helpers_sites = map_leaves_nt(x -> PAT.get_tmp(x, par_templates.θM), h0.helpers_sites))
    else
        h0
    end
    @assert n_batch == size(indiv_args1.xM,2)
    #
    keys_predict_pop = (:ζsP, :θsP)
    h1 = h0p.helpers_sites[1]
    # do not know the number of training sites sum of sites in batch, need to
    #   reduce by vcat
    hi_agg_batch = NamedTuple{keys_predict_indiv}(map(keys_predict_indiv) do key
        repeat(h1[key][:,1], 1, n_batch)  # extend columns to n_site_train
    end)
    red = make_tuple_reducer(hcat)
    hi_agg = mapfoldl(red, enumerate(train_loader), init=map(x -> x[:,1:0],hi_agg_batch)) do (i_batch, data_batch)
        indiv_args = NamedTuple{keys_indiv}(data_batch)
        neg_elbo_sites!(h0p, NegLogDensity(), ϕ, rnormPM, sample_args, indiv_args, nljoint_args_fix)
        for key in keys_predict_indiv
            for ii in 1:n_batch
                hi_agg_batch[key][:,ii] .= h0p.helpers_sites[ii][key]
            end
        end
        hi_agg_batch
    end
    h0p[keys_predict_pop], hi_agg
end

