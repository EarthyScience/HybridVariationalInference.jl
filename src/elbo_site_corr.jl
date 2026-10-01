function sample_ζsP!(ζsP, logσ_ζP, approx::AbstractMeanHVIApproximation,rnormP, ϕqc::AbstractVector{T}, cor_endsP) where T
    # TODO replace by proper sampling of full covariance matrix
    μζP = CA.getdata(view(ϕqc,Val(:μζP)))
    logσ_ζP .= view(ϕqc, Val(:logσ_ζP))
    # ρsP = view(ϕqc, Val(:ρsP))
    # UP = transformU_block_cholesky1(ρsP, cor_endsP)
    # ζsP * diagm(v) is the same as ζsP .* v'
    ζsP .= μζP .+ (rnormP .* exp.(logσ_ζP)')
    nothing
end

# with Vector, all MCs have the same mean
function sample_ζsM!(ζsM, logσ_ζM, ::AbstractMeanHVIApproximation, rnorm, ϕqIc::AbstractVector{T}, ϕm::AbstractVector, cor_endsM, buffer_nθM::AbstractVector) where T
    # TODO replace by proper sampling of full covariance matrix
    # TODO add scaling by factor in ϕm / dispatch by approach
    n_θM, n_MC = size(ζsM)
    logσ_ζM .= view(ϕqIc, Val(:logσ_ζM))
    @assert size(buffer_nθM) == (n_θM,)
    scale = buffer_nθM
    @. scale = exp(logσ_ζM / T(2))
    μζM = view(ϕm, 1:n_θM)           # view of the mean block (n_θM × n_MC)
    ζsM .= μζM .+ (rnorm .* scale')    # does not allocate
    # @inbounds for j in 1:n_MC
    #     for i in 1:n_θM
    #         ζsM[i,j] = ϕm[i] + rnorm[i,j] * scale[i]
    #     end
    # end
    nothing
end

# with Matrix, there is a site mean for each mc-sample
function sample_ζsM!(ζsM, logσ_ζM, ::DiagonalHVIApproximation, rnorm, ϕqc::AbstractVector{T}, ϕm::AbstractMatrix, buffer_nθM::AbstractVector) where T
    n_θM, n_MC = size(ζsM)
    @assert size(rnorm) == (n_θM, n_MC)
    logσ_ζM .= view(ϕqc, Val(:logσ_ζM))
    @assert size(ϕm,1) >= n_θM
    @assert size(ϕm,2) == n_MC
    # TODO avoid allocation with subsetting non-last column
    # μζM = ϕm[1:n_θM,:]
    @assert size(buffer_nθM) == (n_θM,)
    scale = buffer_nθM
    @. scale = exp(logσ_ζM / T(2))
    μζM = view(ϕm, 1:n_θM, :)           # view of the mean block (n_θM × n_MC)
    ζsM .= μζM .+ (rnorm .* scale')       # does not allocate
    # @inbounds for j in 1:n_MC
    #     for i in 1:n_θM
    #         ζsM[i,j] = ϕm[i,j] + rnorm[i,j] * scale[i]
    #     end
    # end
    nothing         
end

