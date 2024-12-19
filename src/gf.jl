function applyf(f, θMs::AbstractMatrix, θP::AbstractVector, x)
    # predict several sites with same physical parameters
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
composition f ∘ g: mechanistic model after machine learning parameter prediction
"""
function gf(g, f, x, xP, ϕg, θP)
    ζMs = g(x, ϕg) # predict the log of the parameters
    θMs = exp.(ζMs)
    y_pred_global, y_pred = f(θP, θMs, xP)
    return y_pred_global, y_pred, θMs
end

"""
Create a loss function for parameter vector p, given 
- g(x, ϕ): machine learning model 
- f(θMs, θP): mechanistic model 
- x_o: matrix of covariates, sites in columns
- y_o: matrix of observations, sites in columns
- int_ϕθP: interpreter attachin axis with compponents ϕg and pc.θP
"""
function get_loss_gf(g, f, y_o_global, int_ϕθP::AbstractComponentArrayInterpreter)
    let g = g, f = f, int_ϕθP = int_ϕθP
        function loss_gf(p, x_o, xP, y_o)
            pc = int_ϕθP(p)
            y_pred_global, y_pred, θMs = gf(g, f, x_o, xP, pc.ϕg, pc.θP)
            #Main.@infiltrate_main
            loss = sum(abs2, y_pred .- y_o) + sum(abs2, y_pred_global .- y_o_global)
            return loss, y_pred_global, y_pred, θMs
        end
    end
end

() -> begin
    loss_gf(p, x_o, y_o)
    Zygote.gradient(x -> loss_gf(x, x_o, y_o)[1], p)
end
