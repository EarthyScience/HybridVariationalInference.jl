function applyf(f, θMs::AbstractMatrix, θP::AbstractVector)
    # predict several sites with same physical parameters
    yv = map(eachcol(θMs)) do θM
        f(vcat(θP, θM))
    end
    y = reduce(hcat, yv)
end
function applyf(f, θMs::AbstractMatrix, θPs::AbstractMatrix)
    # do not call f with matrix θ, because .* with vectors S1 would go wrong
    yv = map(eachcol(θMs), eachcol(θPs)) do θM, θP
        f(vcat(θP, θM))
    end
    y = reduce(hcat, yv)
end
#applyf(f_double, θMs_true, stack(Iterators.repeated(CA.getdata(θP_true), size(θMs_true,2))))

"""
composition f ∘ g: mechanistic model after machine learning parameter prediction
"""
function gf(g, f, x, ϕg, θP)
    ζMs = g(x, ϕg) # predict the log of the parameters
    θMs = exp.(ζMs)
    y_pred = applyf(f, θMs, θP)
    return y_pred, θMs
end

"""
Create a loss function for parameter vector p, given 
- g(x, ϕ): machine learning model 
- f(θMs, θP): mechanistic model 
- x_o: matrix of covariates, sites in columns
- y_o: matrix of observations, sites in columns
- int_ϕθP: interpreter attachin axis with compponents ϕg and pc.θP
"""
function get_loss_gf(g, f, x_o, y_o, int_ϕθP::AbstractComponentArrayInterpreter)
    let g = g, f = f, x_o = x_o, y_o = y_o, int_ϕθP
        function loss_gf(p)
            pc = int_ϕθP(p)
            y_pred, θMs = gf(g, f, x_o, pc.ϕg, pc.θP)
            loss = sum(abs2, y_pred .- y_o)
            return loss, y_pred, θMs
        end
    end
end

() -> begin
    loss_gf(p, x_o, y_o)
    Zygote.gradient(x -> loss_gf(x, x_o, y_o)[1], p)
end
