function applyf(f, θMs::AbstractMatrix, θP::AbstractVector, θFix::AbstractVector, xP)
    # predict several sites with same global parameters θP and fixed parameters θFix
    yv = map(eachcol(θMs), xP) do θM, x_site
        f(vcat(θP, θM, θFix), x_site)
    end
    y = stack(yv)
    return(y)
end
function applyf(f, θMs::AbstractMatrix, θPs::AbstractMatrix, θFix::AbstractVector, xP)
    # do not call f with matrix θ, because .* with vectors S1 would go wrong
    yv = map(eachcol(θMs), eachcol(θPs), xP) do θM, θP, xP_site
        f(vcat(θP, θM, θFix), xP_site)
    end
    y = stack(yv)
    return(y)
end
#applyf(f_double, θMs_true, stack(Iterators.repeated(CA.getdata(θP_true), size(θMs_true,2))))

"""
composition f ∘ transM ∘ g: mechanistic model after machine learning parameter prediction
"""
function gf(g, transM, f, xM, xP, ϕg, θP; 
    gpu_handler = default_GPU_DataHandler)
    # @show first(xM,5)
    # @show first(ϕg,5)
    ζMs = g(xM, ϕg) # predict the log of the parameters
    ζMs_cpu = gpu_handler(ζMs)
    if θP isa SubArray && !(gpu_handler isa NullGPUDataHandler) 
        # otherwise Zyote fails on gpu_handler
        θP = copy(θP)
    end
    θP_cpu = gpu_handler(CA.getdata(θP))
    θMs = reduce(hcat, map(transM, eachcol(ζMs_cpu))) # transform each column
    y_pred_global, y_pred = f(θP_cpu, θMs, xP)
    return y_pred_global, y_pred, θMs
end

function gf(prob::AbstractHybridProblem, xM, xP, args...; 
    scenario = (), dev = gpu_device(), kwargs...)
    g, ϕg = get_hybridproblem_MLapplicator(prob; scenario)
    f = get_hybridproblem_PBmodel(prob; scenario)
    (; θP, θM) = get_hybridproblem_par_templates(prob; scenario)
    (; transP, transM) = get_hybridproblem_transforms(prob; scenario)
    g_dev, ϕg_dev, θP_dev =  (dev(g), dev(ϕg), dev(CA.getdata(θP))) 
    gf(g_dev, transM, f, xM, xP, ϕg_dev, θP_dev; kwargs...)
end

"""
Create a loss function for parameter vector p, given 
- g(x, ϕ): machine learning model 
- f(θMs, θP): mechanistic model 
- xM: matrix of covariates, sites in columns
- y_o: matrix of observations, sites in columns
- int_ϕθP: interpreter attachin axis with compponents ϕg and pc.θP
"""
function get_loss_gf(g, transM, f, y_o_global, int_ϕθP::AbstractComponentArrayInterpreter)
    let g = g, transM = transM, f = f, int_ϕθP = int_ϕθP, y_o_global = y_o_global
        function loss_gf(p, xM, xP, y_o, y_unc)
            σ = exp.(y_unc ./ 2)
            pc = int_ϕθP(p)
            y_pred_global, y_pred, θMs = gf(
                g, transM, f, xM, xP, CA.getdata(pc.ϕg), CA.getdata(pc.θP))
            loss = sum(abs2, (y_pred .- y_o) ./ σ) + sum(abs2, y_pred_global .- y_o_global)
            return loss, y_pred_global, y_pred, θMs
        end
    end
end


() -> begin
    loss_gf(p, xM, y_o)
    Zygote.gradient(x -> loss_gf(x, xM, y_o)[1], p)
end
