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
function gf(prob::AbstractHybridProblem, xM, xP, args...; 
    scenario = (), 
    gdev = :use_gpu ∈ scenario ? gpu_device() : identity, 
    cdev = gdev isa MLDataDevices.AbstractGPUDevice ? cpu_device() : identity,
    kwargs...)
    g, ϕg = get_hybridproblem_MLapplicator(prob; scenario)
    f = get_hybridproblem_PBmodel(prob; scenario)
    (; θP, θM) = get_hybridproblem_par_templates(prob; scenario)
    (; transP, transM) = get_hybridproblem_transforms(prob; scenario)
    g_dev, ϕg_dev, θP_dev =  (gdev(g), gdev(ϕg), gdev(CA.getdata(θP))) 

    gf(g_dev, transM, f, xM, xP, ϕg_dev, θP_dev; cdev, kwargs...)
end

function gf(g, transM, f, xM, xP, ϕg, θP; 
    cdev = identity)
    # @show first(xM,5)
    # @show first(ϕg,5)
    if θP isa SubArray && (cdev isa MLDataDevices.AbstractCPUDevice) 
        # otherwise Zyote fails on cpu_handler
        θP = copy(θP)
    end
    θP_cpu = cdev(CA.getdata(θP)) 
    θMs = gtrans(g, transM, xM, ϕg; cdev)
    y_pred_global, y_pred = f(θP_cpu, θMs, xP)
    return y_pred_global, y_pred, θMs
end

"""
composition transM ∘ g: transformation after machine learning parameter prediction
"""
function gtrans(g, transM, xM, ϕg; cdev = identity)
    ζMs = g(xM, ϕg) # predict the log of the parameters
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
- int_ϕθP: interpreter attachin axis with compponents ϕg and pc.θP
- kwargs: additional keyword arguments passed to gf, such as gdev

The loss function `loss_gf(p, xM, xP, y_o, y_unc, i_sites)` takes   
- parameter vector p
- xM: matrix of covariate, sites in the batch are in columns
- xP: iteration of drivers for each site
- y_o: matrix of observations, sites in columns
- y_unc: vector of uncertainty information for each observation
- i_sites: index of sites in the batch
"""
function get_loss_gf(g, transM, f, y_o_global, int_ϕθP::AbstractComponentArrayInterpreter; kwargs...)
    let g = g, transM = transM, f = f, int_ϕθP = int_ϕθP, y_o_global = y_o_global, kwargs = kwargs
        function loss_gf(p, xM, xP, y_o, y_unc, i_sites)
            σ = exp.(y_unc ./ 2)
            pc = int_ϕθP(p)
            y_pred_global, y_pred, θMs = gf(
                g, transM, f, xM, xP, CA.getdata(pc.ϕg), CA.getdata(pc.θP); kwargs...)
            loss = sum(abs2, (y_pred .- y_o) ./ σ) + sum(abs2, y_pred_global .- y_o_global)
            return loss, y_pred_global, y_pred, θMs
        end
    end
end


() -> begin
    loss_gf(p, xM, y_o)
    Zygote.gradient(x -> loss_gf(x, xM, y_o)[1], p)
end
