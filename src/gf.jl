function applyf(f, θMs::AbstractMatrix, θP::AbstractVector, x)
    # predict several sites with same global parameters θP
    yv = map(eachcol(θMs), x) do θM, x_site
        f(vcat(θP, θM), x_site)
    end
    y = stack(yv)
    return(y)
end
function applyf(f, θMs::AbstractMatrix, θPs::AbstractMatrix, xP)
    # do not call f with matrix θ, because .* with vectors S1 would go wrong
    yv = map(eachcol(θMs), eachcol(θPs), xP) do θM, θP, xP_site
        f(vcat(θP, θM), xP_site)
    end
    y = stack(yv)
    return(y)
end
#applyf(f_double, θMs_true, stack(Iterators.repeated(CA.getdata(θP_true), size(θMs_true,2))))

"""
composition f ∘ transM ∘ g: mechanistic model after machine learning parameter prediction
"""
function gf(g, transM, f, xM, xP, ϕg, θP)
    # @show first(xM,5)
    # @show first(ϕg,5)
    ζMs = g(xM, ϕg) # predict the log of the parameters
    # @show first(ζMs,5)
    θMs = reduce(hcat, map(transM, eachcol(ζMs))) # transform each column
    y_pred_global, y_pred = f(θP, θMs, xP)
    return y_pred_global, y_pred, θMs
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
    let g = g, transM = transM, f = f, int_ϕθP = int_ϕθP
        function loss_gf(p, xM, xP, y_o, y_unc)
            σ = exp.(y_unc ./ 2)
            pc = int_ϕθP(p)
            y_pred_global, y_pred, θMs = gf(g, transM, f, xM, xP, pc.ϕg, pc.θP)
            loss = sum(abs2, (y_pred .- y_o) ./ σ) + sum(abs2, y_pred_global .- y_o_global)
            return loss, y_pred_global, y_pred, θMs
        end
    end
end

() -> begin
    loss_gf(p, xM, y_o)
    Zygote.gradient(x -> loss_gf(x, xM, y_o)[1], p)
end
