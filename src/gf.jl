function applyf(f, θMs::AbstractMatrix, θP::AbstractVector, θFix::AbstractVector, xP, args...; kwargs...)
    # predict several sites with same global parameters θP and fixed parameters θFix
    yv = map(eachcol(θMs), xP) do θM, x_site
        f(vcat(θP, θM, θFix), x_site, args...; kwargs...)
    end
    y = stack(yv)
    return(y)
end
function applyf(f, θMs::AbstractMatrix, θPs::AbstractMatrix, θFix::AbstractVector, xP, args...; kwargs...)
    # do not call f with matrix θ, because .* with vectors S1 would go wrong
    yv = map(eachcol(θMs), eachcol(θPs), xP) do θM, θP, xP_site
        f(vcat(θP, θM, θFix), xP_site, args...; kwargs...)
    end
    y = stack(yv)
    return(y)
end
#applyf(f_double, θMs_true, stack(Iterators.repeated(CA.getdata(θP_true), size(θMs_true,2))))

"""
composition f ∘ transM ∘ g: mechanistic model after machine learning parameter prediction
"""
function gf(prob::AbstractHybridProblem, args...; scenario = (), kwargs...)
    train_loader = get_hybridproblem_train_dataloader(prob; scenario)
    train_loader_dev = gdev_hybridproblem_dataloader(train_loader; scenario)
    xM, xP = train_loader_dev.data[1:2]
    gf(prob, xM, xP, args...; kwargs...)
end
function gf(prob::AbstractHybridProblem, xM::AbstractMatrix, xP::AbstractVector, args...; 
    scenario = (), 
    gdev = :use_gpu ∈ scenario ? gpu_device() : identity, 
    cdev = gdev isa MLDataDevices.AbstractGPUDevice ? cpu_device() : identity,
    kwargs...)
    g, ϕg = get_hybridproblem_MLapplicator(prob; scenario)
    n_site, n_batch = get_hybridproblem_n_site_and_batch(prob; scenario)
    is_predict_batch = (n_batch == length(xP))
    n_site_pred = is_predict_batch ? n_batch : n_site
    @assert length(xP) == n_site_pred
    @assert size(xM, 2) == n_site_pred
    f = get_hybridproblem_PBmodel(prob; scenario, use_all_sites = !is_predict_batch)
    (; θP, θM) = get_hybridproblem_par_templates(prob; scenario)
    (; transP, transM) = get_hybridproblem_transforms(prob; scenario)
    intP = ComponentArrayInterpreter(θP)
    pbm_covars = get_hybridproblem_pbmpar_covars(prob; scenario)
    pbm_covar_indices = CA.getdata(intP(1:length(intP))[pbm_covars])
    ζP = inverse(transP)(θP)
    g_dev, ϕg_dev, ζP_dev =  (gdev(g), gdev(ϕg), gdev(CA.getdata(ζP))) 
    gf(g_dev, transM, transP, f, xM, xP, ϕg_dev, ζP_dev, pbm_covar_indices; cdev, kwargs...)
end

function gf(g::AbstractModelApplicator, transM, transP, f, xM, xP, ϕg, ζP; 
    cdev = identity, pbm_covars, 
    intP = ComponentArrayInterpreter(ζP), kwargs...)
    pbm_covar_indices = intP(1:length(intP))[pbm_covars]
    gf(g, transM, transP, f, xM, xP, ϕg, ζP, pbm_covar_indices; kwargs...)
end


function gf(g::AbstractModelApplicator, transM, transP, f, xM, xP, ϕg, ζP, pbm_covar_indices::AbstractVector{<:Integer}; 
    cdev = identity)
    # @show first(xM,5)
    # @show first(ϕg,5)

    # if ζP isa SubArray #&& (cdev isa MLDataDevices.AbstractCPUDevice) 
    #     # otherwise Zyote fails on cpu_handler
    #     ζP = copy(ζP)
    # end
    #xMP = _append_PBM_covars(xM, intP(ζP), pbm_covars) 
    xMP = _append_each_covars(xM, CA.getdata(ζP), pbm_covar_indices)
    θMs = gtrans(g, transM, xMP, ϕg; cdev)
    θP = transP(CA.getdata(ζP))
    θP_cpu = cdev(θP) 
    y_pred_global, y_pred = f(θP_cpu, θMs, xP)
    return y_pred_global, y_pred, θMs, θP_cpu
end

"""
composition transM ∘ g: transformation after machine learning parameter prediction
"""
function gtrans(g, transM, xMP, ϕg; cdev = identity)
    ζMs = g(xMP, ϕg) # predict the log of the parameters
    ζMs_cpu = cdev(ζMs)
    # TODO move to gpu, Zygote needs to work with transM
    θMs = reduce(hcat, map(transM, eachcol(ζMs_cpu))) # transform each column
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

The loss function `loss_gf(p, xM, xP, y_o, y_unc, i_sites)` takes   
- parameter vector p
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
    pbm_covars, kwargs...)

    let g = g, transM = transM, transP = transP, f = f, y_o_global = y_o_global, 
        intϕ = get_concrete(intϕ),
        pbm_covar_indices = CA.getdata(intP(1:length(intP))[pbm_covars])
        #, intP = get_concrete(intP)
        #inv_transP = inverse(transP), kwargs = kwargs
        function loss_gf(p, xM, xP, y_o, y_unc, i_sites)
            σ = exp.(y_unc ./ 2)
            pc = intϕ(p)
            y_pred_global, y_pred, θMs, θP = gf(
                g, transM, transP, f, xM, xP, CA.getdata(pc.ϕg), CA.getdata(pc.ϕP), 
                pbm_covar_indices; kwargs...)
            loss = sum(abs2, (y_pred .- y_o) ./ σ) + sum(abs2, y_pred_global .- y_o_global)
            return loss, y_pred_global, y_pred, θMs, θP
        end
    end
end


() -> begin
    loss_gf(p, xM, y_o)
    Zygote.gradient(x -> loss_gf(x, xM, y_o)[1], p)
end
