function sample_ζsP!(ζsP, logσ_ζP, approx::AbstractMeanHVIApproximation, rnorm, 
    ϕqc::AbstractVector{T}, cor_endsP, sample_buffers::NamedTuple) where T
    μζP = CA.getdata(view(ϕqc,Val(:μζP)))
    logσ_ζP .= view(ϕqc, Val(:logσ_ζP))
    n_θP, n_MC = size(ζsP)
    @assert size(rnorm) == (n_θP, n_MC)
    @assert size(μζP) == (n_θP,)
    ρsP = view(ϕqc, Val(:ρsP))
    zcor_ends = OneBasedVectorWithZero(cor_endsP)
    # ib = 2
    ρ_start = 1
    for ib in axes(cor_endsP, 1)
        r = (zcor_ends[ib-1]+1):zcor_ends[ib]
        μζP_r = view_ϕm(μζP, r)           # dispatch
        rnorm_r = view(rnorm, r, :)
        logσ_ζP_r = view(logσ_ζP, r)
        ρ_end = ρ_start-1 + sumn(length(r)-1)
        #U = UpperTriangular(diagm(ones(length(r)))) # TODO preallocate
        U = sample_buffers.Us[ib] # preallocated UpperTriangular matrix
        ρsP_r = view(ρsP, ρ_start:ρ_end)
        _setU_scaled!(U, ρsP_r)
        # rotate the noise in place into ζsM[r,:], then scale and add the mean
        ζsP_r = view(ζsP, r, :)
        mul!(ζsP_r, U', rnorm_r)
        ζsP_r .= μζP_r .+ exp.(logσ_ζP_r) .* ζsP_r
        ρ_start = ρ_end + 1
    end
    @assert ρ_start-1 == length(ρsP)
    nothing         
end

function prepare_sample_buffers(approx::AbstractMeanHVIApproximation, cor_endsP)
    zcor_endsP = OneBasedVectorWithZero(cor_endsP)
    # make a Tuple, so to be handled by getdiffcache
    Us = Tuple(begin
        nb = zcor_endsP[ib] - zcor_endsP[ib-1]
        U = UpperTriangular(diagm(ones(nb))) 
    end for ib in axes(cor_endsP, 1))
    (; Us)
end

function sample_ζsM!(ζsM, logσ_ζM, approx::AbstractMeanHVIApproximation, rnorm, 
    ϕqc::AbstractVector{T}, ϕm::Union{AbstractVector, AbstractMatrix}, 
    cor_endsM, sample_buffers::NamedTuple) where T
    n_θM, n_MC = size(ζsM)
    @assert size(rnorm) == (n_θM, n_MC)
    #logσ_ζM .= view(ϕqc, Val(:logσ_ζM))
    logσ_ζM .= get_marginal_logσ(approx, ϕqc, ϕm)
    @assert size(ϕm,1) >= n_θM
    assert_ϕm(ϕm, n_MC) # dispatch on vector or matrix
    ρsM = view(ϕqc, Val(:ρsM))
    zcor_ends = OneBasedVectorWithZero(cor_endsM)
    # ib = 1
    ρ_start = 1
    for ib in axes(cor_endsM, 1)
        r = (zcor_ends[ib-1]+1):zcor_ends[ib]
        μζM_r = view_ϕm(ϕm, r)           # dispatch
        rnorm_r = view(rnorm, r, :)
        logσ_ζM_r = view(logσ_ζM, r)
        ρ_end = ρ_start-1 + sumn(length(r)-1)
        #U = UpperTriangular(diagm(ones(length(r)))) # TODO preallocate
        U = sample_buffers.Us[ib] # preallocated UpperTriangular matrix
        ρsM_r = view(ρsM, ρ_start:ρ_end)
        _setU_scaled!(U, ρsM_r)
        # rotate the noise in place into ζsM[r,:], then scale and add the mean
        ζsM_r = view(ζsM, r, :)
        mul!(ζsM_r, U', rnorm_r)
        ζsM_r .= μζM_r .+ exp.(logσ_ζM_r) .* ζsM_r
        ρ_start = ρ_end + 1
    end
    @assert ρ_start-1 == length(ρsM)
    nothing         
end

@inline get_marginal_logσ(::MeanHVIApproximation, ϕqc, ϕm) = view(ϕqc, Val(:logσ_ζM))  
@inline function get_marginal_std(::MeanUniScalingHVIApproximation, ϕqc, ϕm) 
    n_θM = length(ϕqc[Val(:logσ_ζM)])
    ϕm_scaling = ϕm[n_θM+1]
    logσ2_par_offsets = OneBasedVectorWithZero(ϕqc[Val(:logσ2_ζM_offsets)]) # zero based 
    @assert length(logσ2_par_offsets) + 1 == n_θM
    logσ2_site_offset = logit(ϕm_scaling) # (0..1)->(-Inf, +Inf), 0.5->0
    #
    logσ2_ζM_base = 0.0 # TODO provide by Approx helper 
    logσ2_ζMs = logσ2_ζM_base .+ logσ2_par_offsets[0:end] .+ logσ2_site_offset
    logσ2_ζMs
end

function prepare_ind_sample_buffers(approx::AbstractMeanHVIApproximation, cor_endsM)
    zcor_endsM = OneBasedVectorWithZero(cor_endsM)
    # make a Tuple, so to be handled by getdiffcache
    Us = Tuple(begin
        nb = zcor_endsM[ib] - zcor_endsM[ib-1]
        U = UpperTriangular(diagm(ones(nb))) 
    end for ib in axes(cor_endsM, 1))
    (; Us)
end


function _setU_scaled!(U::AbstractMatrix{T}, ρ::AbstractVector{T}) where {T};
    _vec2uutri!(U, ρ)
    U[1,1] = one(T)  # first reset to one (not set in _vec2uutri!)
    local n = size(U, 1)
    @inbounds for j in 2:n
        U[j,j] = one(T)  # first reset to one (not set in _vec2uutri!)
        view(U, 1:j, j) ./= sqrt(sum(abs2, view(U, 1:j, j))) 
    end
    #@assert diag(U' * U) ≈ ones(T, n)
    U
end

function _setρ_unscaled!(ρ::AbstractVector{T}, U::AbstractMatrix{T}) where {T};
    local n = size(U, 1)
    @inbounds for j in 2:n
        scale = U[j,j]
        U[1:j,j] ./= scale
    end
    _uutri2vec!(ρ, U)
    ρ
end

function _vec2uutri!(m::AbstractMatrix{T}, v::AbstractVector{T}) where {T}
    local n = size(m,1)
    @assert size(m,2) == n
    @assert length(v) == sumn(n-1)
    local k = 1
    @inbounds for i in 1:(n-1), j in (i+1):n
        m[i,j] = v[k]
        k += 1
    end
    m
end

function _uutri2vec!(v::AbstractVector{T}, m::AbstractMatrix{T}) where {T}
    local n = size(m,1)
    @assert size(m,2) == n
    @assert length(v) == sumn(n-1)
    local k = 1
    @inbounds for i in 1:(n-1), j in (i+1):n
        v[k] = m[i,j]
        k += 1
    end
    v
end

