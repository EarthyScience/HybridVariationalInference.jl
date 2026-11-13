@model function f_resid(y, ::Type{T} = Float64; pbm, θP_true, θMs_true, xP, σ_o, prior_resP, prior_resMs, helpers_channel) where T
    n_obs, n_site = size(y)
    n_θP = length(θP_true)
    n_θM = size(θMs_true, 1)
    rP ~ prior_resP #filldist(prior_res, n_θP)
    rMs ~ prior_resMs
    with_channel_element(helpers_channel) do el_dc
        # rather than providing reshape to pbm, construct arrays and copy to 
        # θsP = Matrix{T}(undef, size(θP_true,1), 1)
        # θsMs = Array{T}(undef, size(θMs_true,1), 1, size(θMs_true,2))
        # y_pred = Array{T}(undef, n_obs, 1, n_site)
        local el, θsP, θsMs, y_pred
        el = map(x -> PAT.get_tmp(x, rP), el_dc)
        (; θsP, θsMs, y_pred) = el
        @assert size(y_pred) == (n_obs, 1, n_site)
        @view(θsP[:,1]) .= CA.getdata(θP_true) .* rP
        @view(θsMs[:,1,:]) .= CA.getdata(θMs_true) .* rMs
        for j in 1:n_site
            HVI.apply_model!(view(y_pred, :, :, j), pbm, θsP, @view(θsMs[:,:,j]), @view(xP[:, j]))
        end
        for i_obs in 1:n_obs
            y[i_obs, :] ~ MvNormal(view(y_pred, i_obs, 1, :), σ_o[i_obs]) # single value σ instead of variance
        end
        y_pred
    end
end

@model function f_resid_agg(y, ::Type{T} = Float64; pbm, θP_true, θMs_true, xP, σ_o, prior_resP, prior_resM, helpers_channel) where T
    n_obs, n_site = size(y)
    n_θP = length(θP_true)
    n_θM = size(θMs_true, 1)
    rP ~ prior_resP #filldist(prior_res, n_θP)
    rM ~ prior_resM
    with_channel_element(helpers_channel) do el_dc
        # rather than providing reshape to pbm, construct arrays and copy to 
        # θsP = Matrix{T}(undef, size(θP_true,1), 1)
        # θsMs = Array{T}(undef, size(θMs_true,1), 1, size(θMs_true,2))
        # y_pred = Array{T}(undef, n_obs, 1, n_site)
        local el, θsP, θsMs, y_pred
        el = map(x -> PAT.get_tmp(x, rP), el_dc)
        (; θsP, θsMs, y_pred) = el
        @assert size(y_pred) == (n_obs, 1, n_site)
        @view(θsP[:,1]) .= CA.getdata(θP_true) .* rP
        @view(θsMs[:,1,:]) .= CA.getdata(θMs_true) .* rM
        # θP = reshape(CA.getdata(θP_true) .* exp.(rP), size(θP_true,1), 1)
        # θMs = reshape(CA.getdata(θMs_true) .* exp.(rMs), size(θMs_true,1), 1, size(θMs_true,2))
        # rP = Matrix{T}(undef, n_θP, 1)
        # rP[:,1] ~ prior_resP #filldist(prior_res, n_θP)
        # rMs = Array{T}(undef, n_θM, 1, n_site)
        # rMs[:,1,:] ~ prior_resMs #filldist(prior_res, n_θM, n_site) # single matrix-valued variable
        # θP = CA.getdata(θP_true) .* exp.(rP)
        # θMs = reshape(CA.getdata(θMs_true), size(θMs_true,1),1,:) .* exp.(rMs)
        for j in 1:n_site
            HVI.apply_model!(view(y_pred, :, :, j), pbm, θsP, @view(θsMs[:,:,j]), @view(xP[:, j]))
        end
        for i_obs in 1:n_obs
            y[i_obs, :] ~ MvNormal(view(y_pred, i_obs, 1, :), σ_o[i_obs]) # single value σ instead of variance
        end
        y_pred
    end
end

# provide a prior around true values
@model function fsites(y, ::Type{T} = Float64; pbm, xP, σ_o, prior_ζP, prior_ζMs, helpers_channel) where {T}
    n_obs, n_site = size(y)
    # ζP = rand(prior_ζP)
    # ζMs = rand(prior_ζMs)
    ζP ~ prior_ζP 
    ζMs ~ prior_ζMs
    with_channel_element(helpers_channel) do el_dc
        local el, θsP, θsMs, y_pred
        el = map(x -> PAT.get_tmp(x, ζP), el_dc)
        (; θsP, θsMs, y_pred) = el
        @assert size(y_pred) == (n_obs, 1, n_site)
        @view(θsP[:,1]) .= exp.(ζP)
        @view(θsMs[:,1,:]) .= exp.(ζMs)
        for j in 1:n_site
            HVI.apply_model!(view(y_pred, :, :, j), pbm, θsP, @view(θsMs[:,:,j]), @view(xP[:, j]))
        end
        for i_obs in 1:n_obs
            y[i_obs, :] ~ MvNormal(view(y_pred, i_obs, 1, :), σ_o[i_obs]) # single value σ instead of variance
        end
        y_pred
    end
end

