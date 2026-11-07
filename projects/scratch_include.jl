@model function fsites(y, ::Type{T} = Float64; f, n_θP, n_θM, xP, xM, σ_o) where {T}
    n_obs, n_site = size(y)
    prior_ζP = prior_ζn(n_θP)
    prior_ζM_sites = fill(prior_ζn(n_site), n_θM)
    ζP ~ prior_ζP 
    # CAUTION: order of vectorizing matrix depends on order of ~
    # need to assign each variable in first site first, then second site, ...
    #   need to construct different MvNormal prior if std differs by variable
    # or need to take care when extracting samples and when constructing chains
    ζMs = Matrix{T}(undef, n_θM, n_site)
    # the first loop vectorizes θMs_tr by columns but is much slower
    # for i_site in 1:n_site
    #     ζMs[:, i_site] ~ prior_ζn(n_θM) #MvNormal(n_site, 10.0)
    # end
    # this loop is faster, but vectorizes θMs_tr by rows in parameter vector
    for i_par in 1:n_θM
        ζMs[i_par, :] ~ prior_ζM_sites[i_par]
    end
    # assume σ_o known, see f_MM
    #σ_o ~ truncated(Normal(0, 1); lower=0)
    #TODO specify with transPM
    # @show ζP
    ζPm = repeat(ζP, 1, 1)
    #j = 1
    y_pred = Matrix{T}(undef, n_obs, n_site)
    for j in 1:n_site
        HVI.apply_model!(@view(y_pred[:,j:j]), f, exp.(ζPm), exp.(ζMs[:,j:j]), xP[:,j])
    end
    # i_obs = 1
    for i_obs in 1:n_obs
        y[i_obs, :] ~ MvNormal(y_pred[i_obs, :], σ_o[i_obs]) # single value σ instead of variance
    end
    # θMs_MCc[:,:,1] # checking row- or column-order of θMs_tr
    # exp.(ζMs)
    y_pred
end

@model function f_resid(y, ::Type{T} = Float64; pbm, θP_true, θMs_true, xP, σ_o, σ_r = 1.0) where T
    n_obs, n_site = size(y)
    n_θP = length(θP_true)
    n_θM = size(θMs_true, 1)
    # residuals on log scale: θ = θ_true * exp(r) stays positive
    rP ~ filldist(Normal(0.0, σ_r), n_θP)
    rMs ~ filldist(Normal(0.0, σ_r), n_θM, n_site) # single matrix-valued variable
    # Float64 (AdvancedHMC metric is Float64), Dual during ForwardDiff gradient
    θPm = reshape(CA.getdata(θP_true) .* exp.(rP), :, 1) # 1 column matrix
    y_pred = Matrix{T}(undef, n_obs, n_site)
    for j in 1:n_site
        θMm = reshape(CA.getdata(θMs_true)[:, j] .* exp.(rMs[:, j]), :, 1)
        # use a view: y_pred[:,j:j] would be a copy and y_pred would stay undef
        HVI.apply_model!(view(y_pred, :, j:j), pbm, θPm, θMm, xP[:, j])
    end
    for i_obs in 1:n_obs
        y[i_obs, :] ~ MvNormal(y_pred[i_obs, :], σ_o[i_obs]) # single value σ instead of variance
    end
    y_pred
end
