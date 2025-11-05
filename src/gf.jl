# Point solver where ML directly predicts PBL parameters, rather than their
# distribution.

# """
# Map process base model (PBM), `f`, across each site.

# ## Arguments
# - `f(θ, xP, args...; intθ1, kwargs...)`: Process based model for single site
  
#   Make sure to hint the type, so that results can be inferred.
# - `θMst`: transposed model parameters across sites matrix: (n_parM, n_site_batch)
# - `θP`: transposed model parameters that do not differ by site: (n_parP,)
# - `θFix`: Further parameter required by f that are not calibrated.
# - `xP`: Model drivers: Matrix with n_site_batch columns.
#   If provided a ComponentArray with labeled rows, f can then access `xP[:key]`.
# - `intθ1`: ComponentArrayInterpreter that can be applied to θ, 
#   so that entries can be extracted.

# See test_HybridProblem of using this function to construct a PBM function that
# can predict across all sites.
# """
# function map_f_each_site(
#     f, θMst::AbstractMatrix, θP::AbstractVector, θFix::AbstractVector, xP, args...; 
#     intθ1::AbstractComponentArrayInterpreter, kwargs...
# )
#     # predict several sites with same global parameters θP and fixed parameters θFix
#     it1 = eachcol(CA.getdata(θMst))
#     it2 = eachcol(xP)
#     _θM = first(it1)
#     _x_site = first(it2)
#     TXS = typeof(_x_site)
#     TY = typeof(f(vcat(θP, _θM, θFix), _x_site, args...; intθ1, kwargs...))
#     #TY = typeof(f(vcat(θP, _θM, θFix), _x_site; intθ1))
#     yv = map(it1, it2) do θM, x_site
#         x_site_typed = x_site::TXS
#         f(vcat(θP, θM, θFix), x_site_typed, args...; intθ1, kwargs...)
#     end::Vector{TY}
#     y = stack(yv)
#     return(y)
# end
# function map_f_each_site(f, θMs::AbstractMatrix, θPs::AbstractMatrix, θFix::AbstractVector, xP, args...; kwargs...)
#     # do not call f with matrix θ, because .* with vectors S1 would go wrong
#     yv = map(eachcol(θMs), eachcol(θPs), xP) do θM, θP, xP_site
#         f(vcat(θP, θM, θFix), xP_site, args...; kwargs...)
#     end
#     y = stack(yv)
#     return(y)
# end
# #map_f_each_site(f_double, θMs_true, stack(Iterators.repeated(CA.getdata(θP_true), size(θMs_true,2))))

"""
    predict_point_hvi([rng], prob::AbstractHybridProblem)

Prediction function for hybrid variational inference parameter model that omits
the sampling step but returns the prediction at the mean in unconstrained space.

## Arguments
- `prob`: The problem for which to predict

## Keyword arguments
- `scenario`
- `gdevs`
- `xM`: covariates for the machine-learning model (ML): Matrix (n_θM x n_site_pred).
  Possibility to override the default from `get_hybridproblem_train_dataloader`.
- `xP`: model drivers for process based model (PBM): Matrix with (n_site_pred) rows.
  Possibility to override the default from `get_hybridproblem_train_dataloader`.

Returns an NamedTuple `(; y, θMs, θP)` with entries
- `y`: Matrix `(n_obs, n_site)` of model predictions.
- `θP`: ComponentVector of PBM model parameters
  that are kept constant across sites.
- `θMs`: ComponentMatrix `(n_site, n_θM)` of PBM model parameters
  that vary by site.
"""
function predict_point_hvi(rng, prob::AbstractHybridProblem; scenario=Val(()), 
    gdevs = get_gdev_MP(scenario), 
    xM = nothing, xP = nothing,
    is_testmode = true,
    kwargs...
    )
    if isnothing(xM) || isnothing(xP)
        dl = get_hybridproblem_train_dataloader(prob; scenario)
        dl_dev = gdev_hybridproblem_dataloader(dl; gdevs)
        xM_dl, xP_dl = dl_dev.data[1:2]
        xM = isnothing(xM) ? xM_dl : xM
        xP = isnothing(xP) ? xP_dl : xP
    end
    y_pred, θMs, θP = gf(prob, xM, xP; scenario, gdevs, is_testmode, kwargs...)    
    θPc = ComponentArrayInterpreter(prob.θP)(θP)
    θMsc = ComponentArrayInterpreter((size(θMs,1),), prob.θM)(θMs)
    (;y_pred, θMs=θMsc, θP=θPc)
end



# function gf(prob::AbstractHybridProblem; scenario = Val(()), kwargs...)
#     train_loader = get_hybridproblem_train_dataloader(prob; scenario)
#     train_loader_dev = gdev_hybridproblem_dataloader(train_loader; scenario)
#     xM, xP = train_loader_dev.data[1:2]
#     gf(prob, xM, xP; scenario, kwargs...)
# end
"""
composition f ∘ transM ∘ g: mechanistic model after machine learning parameter prediction
"""
function gf(prob::AbstractHybridProblem, xM::AbstractMatrix, xP::AbstractMatrix; 
    scenario = Val(()), 
    gdevs = nothing, #get_gdev_MP(scenario), 
    is_inferred::Val{is_infer} = Val(false),
    kwargs...
) where is_infer
    gdevs = isnothing(gdevs) ? get_gdev_MP(scenario) : gdevs
    g, ϕg = get_hybridproblem_MLapplicator(prob; scenario)
    n_site, n_batch = get_hybridproblem_n_site_and_batch(prob; scenario)
    n_site_pred = size(xP,2)
    @assert size(xM, 2) == n_site_pred
    f_batch = get_hybridproblem_PBmodel(prob; scenario)
    f = (n_site_pred == n_batch) ? f : create_nsite_applicator(f_batch, n_site_pred)
    if gdevs.gdev_P isa MLDataDevices.AbstractGPUDevice
        f_dev = gdevs.gdev_P(f) #fmap(gdevs.gdev_P, f)
    else
        f_dev = f
    end
    (; θP, θM) = get_hybridproblem_par_templates(prob; scenario)
    (; transP, transM) = get_hybridproblem_transforms(prob; scenario)
    transMs = StackedArray(transM, n_site_pred)
    intP = ComponentArrayInterpreter(θP)
    pbm_covars = get_hybridproblem_pbmpar_covars(prob; scenario)
    pbm_covar_indices = CA.getdata(intP(1:length(intP))[pbm_covars])
    ζP = inverse(transP)(θP)
    gdev, cdev = gdevs.gdev_M, infer_cdev(gdevs)
    g_dev, ϕg_dev, xM_dev, ζP_dev =  gdev(g), gdev(ϕg), gdev(CA.getdata(xM)), gdev(CA.getdata(ζP))
    # most of the properties of prob are not type-inferred
    # hence result is not type-inferred, but may test at this context
    res = is_infer ? 
        Test.@inferred( gf(
            g_dev, transMs, transP, f_dev, xM_dev, xP, ϕg_dev, ζP_dev, pbm_covar_indices; 
            cdev, kwargs...)) :
        gf(g_dev, transMs, transP, f_dev, xM_dev, xP, ϕg_dev, ζP_dev, pbm_covar_indices; 
            cdev, kwargs...)
end

function gf(g::AbstractModelApplicator, transMs, transP, f, xM, xP, ϕg, ζP; 
    cdev, pbm_covars, 
    intP = ComponentArrayInterpreter(ζP), kwargs...)
    pbm_covar_indices = intP(1:length(intP))[pbm_covars]
    gf(g, transM, transP, f, xM, xP, ϕg, ζP, pbm_covar_indices; kwargs...)
end



function gf(g::AbstractModelApplicator, transMs, transP, f, xM, xP, ϕg, ζP, 
    pbm_covar_indices::AbstractVector{<:Integer}; 
    cdev, is_testmode)
    # @show first(xM,5)
    # @show first(ϕg,5)

    # if ζP isa SubArray #&& (cdev isa MLDataDevices.AbstractCPUDevice) 
    #     # otherwise Zyote fails on cpu_handler
    #     ζP = copy(ζP)
    # end
    #xMP = _append_PBM_covars(xM, intP(ζP), pbm_covars) 
    xMP = _append_each_covars(xM, CA.getdata(ζP), pbm_covar_indices)
    θMs = gtrans(g, transMs, xMP, ϕg; cdev, is_testmode)
    # transPM = RRuleMonitor("transP", ζP -> transP(ζP))
    # θP = transPM(CA.getdata(ζP))
    θP = transP(CA.getdata(ζP))
    θP_cpu = cdev(θP) 
    y_pred = f(θP_cpu, θMs, xP)
    # fM = RRuleMonitor("f in gf", (θP_cpu) -> f(θP_cpu, θMs, xP), DI.AutoForwardDiff())
    # y_pred = fM(θP_cpu) 
    # fM = RRuleMonitor("f in gf", (θP_cpu, θMs) -> f(θP_cpu, θMs, xP))
    # y_pred = fM(θP_cpu, θMs) # very slow large JvP with θMs
    return y_pred, θMs, θP_cpu
end

"""
composition transM ∘ g: transformation after machine learning parameter prediction
Provide a `transMs = StackedArray(transM, n_batch)`
"""
function gtrans(g, transMs, xMP, ϕg; cdev, is_testmode)
    # TODO remove after removing gf
    # predict the log of the parameters
    ζMst = g(xMP, ϕg; is_testmode)
    ζMs = ζMst' 
    ζMs_cpu = cdev(ζMs)
    θMs = transMs(ζMs_cpu)
    if !all(isfinite.(θMs))
        @info "gtrans: encountered non-finite parameters"
        #@show θMs, ζMs_cpu, transMs
        #@show xMP, ϕg, is_testmode
        #TODO save xMP, ϕg, is_testmode using JLD2
    end
    θMs
    #θMs = reduce(hcat, map(transM, eachcol(ζMs_cpu))) # transform each row
end

"""
Create a loss function for given
- g(x, ϕ): machine learning model 
- transM: transformation of parameters at unconstrained space
- f(θMs, θP): mechanistic model 
- py: `function(y_pred, y_obs, y_unc)` to compute negative log-likelihood, i.e. cost
- intϕ: interpreter attaching axis with components ϕg and ϕP
- intP: interpreter attaching axis to ζP = ϕP with components used by f,
  The default, uses `intϕ(ϕ)` as a template
- kwargs: additional keyword arguments passed to `gf`, such as `gdev` or `pbm_covars`

The loss function `loss_gf(ϕ, xM, xP, y_o, y_unc, i_sites)` takes   
- parameter vector ϕ
- xM: matrix of covariate, sites in the batch are in columns
- xP: iteration of drivers for each site
- y_o: matrix of observations, sites in columns
- y_unc: vector of uncertainty information for each observation
  Currently, hardcoes squared error loss of `(y_pred .- y_o) ./ σ`, 
  with `σ = exp.(y_unc ./ 2)`.
- i_sites: index of sites in the batch

and returns a NamedTuple of 
- `nLjoint`: the negative-log of the joint parameter probability (Likelihood * prior)
- `y_pred`: predicted values
- `θMs`, `θP`: PBM-parameters 
- `nLy`: negative log-Likelihood of y_pred
- `neg_log_prior`: negative log-prior of `θMs` and `θP`
- `neg_log_prior`: negative log-prior of `θMs` and `θP`
"""
function get_loss_gf(g, transM, transP, f, py,  
    intϕ::AbstractComponentArrayInterpreter,
    intP::AbstractComponentArrayInterpreter = ComponentArrayInterpreter(
        intϕ(1:length(intϕ)).ϕP);
    cdev=cpu_device(),
    pbm_covars, n_site_batch, 
    priorsP, priorsM, floss_penalty = zero_penalty_loss,
    is_omit_priors = false,
    kwargs...)

    let g = g, transM = transM, transP = transP, f = f, 
        intϕ = get_concrete(intϕ),
        transMs = StackedArray(transM, n_site_batch),
        is_omit_priors = is_omit_priors,
        cdev = cdev,
        pbm_covar_indices = CA.getdata(intP(1:length(intP))[pbm_covars]),
        priorsP = priorsP, priorsM = priorsM, floss_penalty = floss_penalty,
        cpu_dev = cpu_device() # real cpu, different form infer_cdev(gdevs) that maybe idenetity
        #, intP = get_concrete(intP)
        #inv_transP = inverse(transP), kwargs = kwargs
        function loss_gf(ϕ, xM, xP, y_o, y_unc, i_sites; is_testmode)
            ϕc = intϕ(ϕ)
            # GPUArraysCore.allowscalar(() -> if !all(isfinite.(ϕ))
            #     @show ϕc.ϕP
            #     error("invokded loss function loss_gf with non-finite parameters")
            # end)
            # μ_ζP = ϕc.ϕP
            # xMP = _append_each_covars(xM, CA.getdata(μ_ζP), pbm_covar_indices)
            # ϕ_M = g(xMP, CA.getdata(ϕc.ϕg))
            # μ_ζMs = ϕ_M'
            # ζP_cpu = cdev(CA.getdata(μ_ζP))
            # ζMs_cpu = cdev(CA.getdata(μ_ζMs))
            # y_pred, _, _ = apply_f_trans(ζP_cpu, ζMs_cpu, f, xP; transM, transP)
            if !all(isfinite.(ϕ)) 
                @info "loss_gf: encountered non-finite ϕ"
                @show ϕc.ϕP
                #Main.@infiltrate_main
            end
            y_pred, θMs_pred, θP_pred = gf(
                g, transMs, transP, f, xM, xP, CA.getdata(ϕc.ϕg), CA.getdata(ϕc.ϕP), 
                pbm_covar_indices; cdev, is_testmode, kwargs...)
            #σ = exp.(y_unc ./ 2)
            #nLy = sum(abs2, (y_pred .- y_o) ./ σ) 
            nLy = py( y_pred, y_o, y_unc)
            # logpdf is not typestable for Distribution{Univariate, Continuous}
            logpdf_t = (prior, θ) -> logpdf(prior, θ)::eltype(θP_pred)
            logpdf_tv = (prior, θ::AbstractVector) -> begin
                map(Base.Fix1(logpdf, prior), θ)::Vector{eltype(θP_pred)}
            end
            #Main.@infiltrate_main
            #Maybe: move priors to GPU, for now need to move θ to cpu
            # currently does not work on gpu, moving to dpu has problems with gradient
            #    need to specify is_omit_priors if PBM is on GPU
            neg_log_prior = if is_omit_priors
                    zero(nLy)
            else
                nLP = if isempty(θP_pred) 
                    zero(nLy)
                else
                    θP_pred_cpu = CA.getdata(θP_pred)
                    -sum(logpdf_t.(priorsP, θP_pred_cpu))
                end
                θMs_pred_cpu = CA.getdata(θMs_pred)
                nLM = -sum(map((priorMi, θMi) -> sum(
                    logpdf_tv(priorMi, θMi)), priorsM, eachcol(θMs_pred_cpu)))
                nLP + nLM
            end
            # neg_log_prior = is_omit_priors ? zero(nLy) :
            #     (isempty() ? zero(nLy) : ) +
            #     -sum(map((priorMi, θMi) -> sum(
            #         logpdf_tv(priorMi, θMi)), priorsM, eachcol(θMs_pred_cpu))) 
            #neg_log_prior = min(sqrt(floatmax(neg_log_prior0)), neg_log_prior0)                
            if !isfinite(neg_log_prior)
                @info "loss_gf: encountered non-finite prior density"
                @show θP_pred, θMs_pred, ϕc.ϕP
                error("debug get_loss_gf")
            end
            ϕunc = eltype(θP_pred)[]  # no uncertainty parameters optimized
            loss_penalty = floss_penalty(y_pred, θMs_pred, θP_pred, ϕc.ϕg, ϕunc)
            #@show nLy, neg_log_prior, loss_penalty
            nLjoint_pen = nLy + neg_log_prior + loss_penalty
            return (;nLjoint_pen, y_pred, θMs_pred, θP_pred, nLy, neg_log_prior, loss_penalty)
        end
    end
end

# function tmp_fcost(is,intθ,fneglogden )
#     fcost = let is = is, intθ = intθ,fneglogden=fneglogden
#         fcost_inner = (θvec, xPM, y_o, y_unc) -> begin
#             θ = hcat(CA.getdata(θvec.P[is]), CA.getdata(θvec.Ms'))
#             y = DoubleMM.f_doubleMM(θ, xPM, intθ)
#             #y = CP.DoubleMM.f_doubleMM(θ, xPM, θpos)
#             res = fneglogden(y_o, y', y_unc)
#             res
#         end
#     end    
# end
