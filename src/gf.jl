# Point solver where ML directly predicts PBL parameters, rather than their
# distribution.

"""
Map process base model (PBM), `f`, across each site.

## Arguments
- `f(θ, xP, args...; intθ1, kwargs...)`: Process based model for single site
  
  Make sure to hint the type, so that results can be inferred.
- `θMst`: transposed model parameters across sites matrix: (n_parM, n_site_batch)
- `θP`: transposed model parameters that do not differ by site: (n_parP,)
- `θFix`: Further parameter required by f that are not calibrated.
- `xP`: Model drivers: Matrix with n_site_batch columns.
  If provided a ComponentArray with labeled rows, f can then access `xP[:key]`.
- `intθ1`: ComponentArrayInterpreter that can be applied to θ, 
  so that entries can be extracted.

See test_HybridProblem of using this function to construct a PBM function that
can predict across all sites.
"""
function map_f_each_site(
    f, θMst::AbstractMatrix, θP::AbstractVector, θFix::AbstractVector, xP, args...; 
    intθ1::AbstractComponentArrayInterpreter, kwargs...
)
    # predict several sites with same global parameters θP and fixed parameters θFix
    it1 = eachcol(CA.getdata(θMst))
    it2 = eachcol(xP)
    _θM = first(it1)
    _x_site = first(it2)
    TXS = typeof(_x_site)
    TY = typeof(f(vcat(θP, _θM, θFix), _x_site, args...; intθ1, kwargs...))
    #TY = typeof(f(vcat(θP, _θM, θFix), _x_site; intθ1))
    yv = map(it1, it2) do θM, x_site
        x_site_typed = x_site::TXS
        f(vcat(θP, θM, θFix), x_site_typed, args...; intθ1, kwargs...)
    end::Vector{TY}
    y = stack(yv)
    return(y)
end
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
composition f ∘ transM ∘ g: mechanistic model after machine learning parameter prediction
"""
function gf(prob::AbstractHybridProblem; scenario = Val(()), kwargs...)
    train_loader = get_hybridproblem_train_dataloader(prob; scenario)
    train_loader_dev = gdev_hybridproblem_dataloader(train_loader; scenario)
    xM, xP = train_loader_dev.data[1:2]
    gf(prob, xM, xP; scenario, kwargs...)
end
function gf(prob::AbstractHybridProblem, xM::AbstractMatrix, xP::AbstractMatrix; 
    scenario = Val(()), 
    gdev = :use_gpu ∈ _val_value(scenario) ? gpu_device() : identity, 
    cdev = gdev isa MLDataDevices.AbstractGPUDevice ? cpu_device() : identity,
    is_inferred::Val{is_infer} = Val(false),
    kwargs...
) where is_infer
    g, ϕg = get_hybridproblem_MLapplicator(prob; scenario)
    n_site, n_batch = get_hybridproblem_n_site_and_batch(prob; scenario)
    is_predict_batch = (n_batch == size(xP,2))
    n_site_pred = is_predict_batch ? n_batch : n_site
    @assert size(xP, 2) == n_site_pred
    @assert size(xM, 2) == n_site_pred
    f = get_hybridproblem_PBmodel(prob; scenario, use_all_sites = !is_predict_batch)
    (; θP, θM) = get_hybridproblem_par_templates(prob; scenario)
    (; transP, transM) = get_hybridproblem_transforms(prob; scenario)
    transMs = StackedArray(transM, n_site_pred)
    intP = ComponentArrayInterpreter(θP)
    pbm_covars = get_hybridproblem_pbmpar_covars(prob; scenario)
    pbm_covar_indices = CA.getdata(intP(1:length(intP))[pbm_covars])
    ζP = inverse(transP)(θP)
    g_dev, ϕg_dev, ζP_dev =  (gdev(g), gdev(ϕg), gdev(CA.getdata(ζP))) 
    # most of the properties of prob are not type-inferred
    # hence result is not type-inferred, but may test at this context
    res = is_infer ? 
        Test.@inferred( gf(
            g_dev, transMs, transP, f, xM, xP, ϕg_dev, ζP_dev, pbm_covar_indices; 
            cdev, kwargs...)) :
        gf(g_dev, transMs, transP, f, xM, xP, ϕg_dev, ζP_dev, pbm_covar_indices; 
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
    cdev)
    # @show first(xM,5)
    # @show first(ϕg,5)

    # if ζP isa SubArray #&& (cdev isa MLDataDevices.AbstractCPUDevice) 
    #     # otherwise Zyote fails on cpu_handler
    #     ζP = copy(ζP)
    # end
    #xMP = _append_PBM_covars(xM, intP(ζP), pbm_covars) 
    xMP = _append_each_covars(xM, CA.getdata(ζP), pbm_covar_indices)
    θMs = gtrans(g, transMs, xMP, ϕg; cdev)
    θP = transP(CA.getdata(ζP))
    θP_cpu = cdev(θP) 
    y_pred_global, y_pred = f(θP_cpu, θMs, xP)
    return y_pred_global, y_pred, θMs, θP_cpu
end

"""
composition transM ∘ g: transformation after machine learning parameter prediction
Provide a `transMs = StackedArray(transM, n_batch)`
"""
function gtrans(g, transMs, xMP::T, ϕg; cdev) where T
    # TODO remove after removing gf
    # predict the log of the parameters
    ζMst = g(xMP, ϕg)::T   # problem of Flux model applicator restructure 
    ζMs = ζMst' 
    ζMs_cpu = cdev(ζMs)
    θMs = transMs(ζMs_cpu)
    #θMs = reduce(hcat, map(transM, eachcol(ζMs_cpu))) # transform each row
end


"""
Create a loss function for given
- g(x, ϕ): machine learning model 
- transM: transforamtion of parameters at unconstrained space
- f(θMs, θP): mechanistic model 
- y_o_global: site-independent observations
- intϕ: interpreter attaching axis with components ϕg and ϕP
- intP: interpreter attaching axis to ζP = ϕP with components used by f
- kwargs: additional keyword arguments passed to gf, such as gdev or pbm_covars

The loss function `loss_gf(ϕ, xM, xP, y_o, y_unc, i_sites)` takes   
- parameter vector ϕ
- xM: matrix of covariate, sites in the batch are in columns
- xP: iteration of drivers for each site
- y_o: matrix of observations, sites in columns
- y_unc: vector of uncertainty information for each observation
- i_sites: index of sites in the batch
"""
function get_loss_gf(g, transM, transP, f, y_o_global, 
    intϕ::AbstractComponentArrayInterpreter,
    intP::AbstractComponentArrayInterpreter = ComponentArrayInterpreter(
        intϕ(1:length(intϕ)).ϕP);
    cdev=cpu_device(),
    pbm_covars, n_site_batch, kwargs...)

    let g = g, transM = transM, transP = transP, f = f, y_o_global = y_o_global, 
        intϕ = get_concrete(intϕ),
        transMs = StackedArray(transM, n_site_batch),
        cdev = cdev,
        pbm_covar_indices = CA.getdata(intP(1:length(intP))[pbm_covars])
        #, intP = get_concrete(intP)
        #inv_transP = inverse(transP), kwargs = kwargs
        function loss_gf(ϕ, xM, xP, y_o, y_unc, i_sites)
            σ = exp.(y_unc ./ 2)
            ϕc = intϕ(ϕ)
            # μ_ζP = ϕc.ϕP
            # xMP = _append_each_covars(xM, CA.getdata(μ_ζP), pbm_covar_indices)
            # ϕ_M = g(xMP, CA.getdata(ϕc.ϕg))
            # μ_ζMs = ϕ_M'
            # ζP_cpu = cdev(CA.getdata(μ_ζP))
            # ζMs_cpu = cdev(CA.getdata(μ_ζMs))
            # y_pred, _, _ = apply_f_trans(ζP_cpu, ζMs_cpu, f, xP; transM, transP)
            y_pred_global, y_pred, θMs, θP = gf(
                g, transMs, transP, f, xM, xP, CA.getdata(ϕc.ϕg), CA.getdata(ϕc.ϕP), 
                pbm_covar_indices; cdev, kwargs...)
            loss = sum(abs2, (y_pred .- y_o) ./ σ) #+ sum(abs2, y_pred_global .- y_o_global)
            return loss, y_pred, θMs, θP
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
