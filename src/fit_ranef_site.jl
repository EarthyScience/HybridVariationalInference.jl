"""
Loss function for random effects, given fixed ζ.
"""
function get_loss_ran_tr_f(rng, prob, itrain_site; scenario = Val((:default,)),
    train_data = NamedTuple{(:xM, :xP, :y, :y_unc, :itrain_site)}(
        get_hybridproblem_train_dataloader(prob; scenario).data[:]),
    is_omit_priors::Val{omit_priors} = Val(false),        
    frac_cluster = 1.0
    ) where omit_priors
    # provide xM explicitly
    res_predict_point = predict_point_hvi(rng, prob; train_data.xM, train_data.xP, )
    n_site = size(res_predict_point.θMs_tr, 1)
    (;transM, transP) = get_hybridproblem_transforms(prob; scenario)
    transMs = StackedArray(transM, n_site)
    ζMs = inverse(transMs)(res_predict_point.θMs_tr)'
    θP = CA.getdata(res_predict_point.θP)
    intP = ComponentArrayInterpreter(res_predict_point.θP)
    intMs1_tr = ComponentArrayInterpreter(res_predict_point.θMs_tr[[1],:])
    #ζMsc = intMs(ζMs)
    #ζP = inverse(transP)(res_predict_point.θP)
    pt = get_hybridproblem_par_templates(prob; scenario)
    ranef_spec = get_hybridproblem_ranef(prob; scenario)     
    ranef = get_ranef_computer(
        ranef_spec, keys(pt.θM), n_site, one(eltype(ζMs)))
    f_batch = get_hybridproblem_PBmodel(prob; scenario)
    f = create_nsite_applicator(f_batch, 1)
    xP_itrain = train_data.xP[:,[itrain_site]]
    ϕqc = prob.ϕq
    ϕqc_cache = PreallocationTools.DiffCache(copy(ϕqc)) # copy to not modify orig
    ϕq_ranef = prob.ϕq.ranef
    intϕq_ranef = ComponentArrayInterpreter(ϕq_ranef)
    priors = get_hybridproblem_priors(prob; scenario)
    priorsP = Tuple(priors[k] for k in keys(pt.θP))
    priorsM = Tuple(priors[k] for k in keys(pt.θM))
    zero_prior_logdensity = omit_priors ? zero(eltype(pt.θP)) : get_zero_prior_logdensity(
    priorsP, priorsM, pt.θP, pt.θM)   
    penalty_computer = get_hybridproblem_penalty_computer(prob; scenario)

    let f=f,
        y_o = train_data.y[:,[itrain_site]],
        y_unc = train_data.y_unc[:,[itrain_site]],
        py = get_hybridproblem_neg_logden_obs(prob; scenario),
        ranef = ranef, ζMs = ζMs, θP = θP,
        transMs = transMs,
        xP_itrain = xP_itrain,
        intϕq_ranef = intϕq_ranef,
        ϕqc_cache = ϕqc_cache, 
        is_omit_priors = is_omit_priors,
        zero_prior_logdensity = zero_prior_logdensity,
        priorsP = priorsP, priorsM = priorsM,
        frac_cluster = eltype(ζMs)(frac_cluster),
        penalty_computer = penalty_computer,
        intMs_tr = intMs1_tr, intP = intP,
        rng = rng

        function loss_ran_tr_f(ranef_itrain::AbstractVector{T}) where T            # uses mutation -> use NelderMead for few parameters
            ϕqc1 = PreallocationTools.get_tmp(ϕqc_cache, ranef_itrain)
            ϕqc1.ranef.β[itrain_site,:] .= ranef_itrain  # mutation
            # β0 = ϕq_ranef.β
            # β = [r == itrain_site ? ranef_itrain[c] : β0[r,c] for r in axes(β0,1), c in axes(β0,2)]
            # ϕq_ranef1 = CA.ComponentVector(;ϕq_ranef..., β)
            ζMs_tr_ranef = add_ranef(ranef, ζMs, ϕqc1.ranef, [itrain_site])'
            θMs_tr_ranef = transMs(ζMs_tr_ranef)[[itrain_site],:]
            # do not add sampled alleatoric error here
            y_pred, addq_pred = f(
                θP, θMs_tr_ranef, xP_itrain)
            () -> begin    
                # using ShareAdd
                # @usingany UnicodePlots
                pl = scatterplot(log10.(xP_itrain[:,1]), y_o[:,1]; label="obs", title="site $(itrain_site)")
                scatterplot!(pl, log10.(xP_itrain[:,1]), y_pred[:,1]; label="pred")
                θMs_tr_ranef
            end
            nLy = if !all(isfinite.(y_pred[isfinite.(y_o)]))                
                #@warn "encountered non-finite y_pred"
                # random effect so large, that infinite after transformation
                T(1e6)
            else
                res_py = py(y_o, y_pred, y_unc)[1]
                # if !isfinite(res_py)
                #     @warn "encountered non-finite res_py"
                #     Main.@infiltrate_main
                # end 
                res_py
            end
            #
            nLprior_P, nLprior_Ms =
                # @descend_code_warntype (
                compute_priors_logdensity(priorsP, priorsM, θP, θMs_tr_ranef,
                    is_omit_priors, zero_prior_logdensity)
            nLprior_M = nLprior_Ms[1] * frac_cluster
            #
            loss_penalties = first(compute_penalty(penalty_computer,
                y_pred, addq_pred, intMs_tr(θMs_tr_ranef), intP(θP), 
                [itrain_site], ϕqc1))

            loss_penalty = loss_penalties[1] # * frac_cluster
            #
            nLjoint_pen = nLy + nLprior_M + loss_penalty #+ nLRanef
            if !isfinite(nLjoint_pen)
                nLjoint_pen = typeof(nLjoint_pen)(1e5)
            end
            #@show nLjoint_pen, nLy
            (;nLjoint_pen, nLy, nLprior_M, loss_penalty)
        end
    end
end
