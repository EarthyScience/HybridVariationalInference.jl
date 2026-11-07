@model function fsites(y, ::Type{T} = Float32; f, n_θP, n_θM, σ_o) where {T}
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
        HVI.apply_model!(y_pred[:,j:j], f, exp.(ζPm), exp.(ζMs[:,j:j]), xP[:,j])
    end
    # i_obs = 1
    for i_obs in 1:n_obs
        y[i_obs, :] ~ MvNormal(y_pred[i_obs, :], σ_o[i_obs]) # single value σ instead of variance
    end
    # θMs_MCc[:,:,1] # checking row- or column-order of θMs_tr
    # exp.(ζMs)
    y_pred
end

@model function fallea(y, ::Type{T} = Float32; θP_true, θMs_true, pbm, xP, σ_o) where {T}
    n_obs, n_site = size(y)
    # TODO 
    n_θP = length(θP_true)
    n_θM = size(θMs_true,1)
    rP ~ MvNormal(Zeros(n_θP), I*10)
    rM = Matrix{T}(undev, n_θM, n_site)
    θP = θP_true .+ rP
    θPm = repeat(θP,1,1) # 1 column matrix
    y_pred = Matrix{T}(undef, n_obs, n_site)
    for j in 1:n_site
        rM[:,j] ~ MvNormal(Zeros(n_θM), I*10)
        θM = θM_true[:,j] .+ rM[j]
        θMm = repeat(θM,1,1)
        HVI.apply_model!(y_pred[:,j:j], pbm, θPm, θMm, xP[:,j])
    end
    for i_obs in 1:n_obs
        y[i_obs, :] ~ MvNormal(y_pred[i_obs, :], σ_o[i_obs]) # single value σ instead of variance
    end
    y_pred
end

