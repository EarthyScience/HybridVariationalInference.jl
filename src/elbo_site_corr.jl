function sample_ζsP!(ζsP, logσ_ζP, approx::AbstractMeanHVIApproximation,rnormP, ϕqc::AbstractVector{T}, cor_endsP) where T
    # TODO replace by proper sampling of full covariance matrix
    μζP = CA.getdata(view(ϕqc,Val(:μζP)))
    logσ_ζP .= view(ϕqc, Val(:logσ_ζP))
    # ρsP = view(ϕqc, Val(:ρsP))
    # UP = transformU_block_cholesky1(ρsP, cor_endsP)
    # ζsP * diagm(v) is the same as ζsP .* v'
    # diagm(v) * ζsP   is the same as ζsP .* v'
    ζsP .= μζP .+ (rnormP .* exp.(logσ_ζP)')
    nothing
end

function sample_ζsM!(ζsM, logσ_ζM, ::AbstractMeanHVIApproximation, rnorm, 
    ϕqc::AbstractVector{T}, ϕm::Union{AbstractVector, AbstractMatrix}, 
    cor_endsM, sample_buffers::NamedTuple) where T
    n_θM, n_MC = size(ζsM)
    @assert size(rnorm) == (n_θM, n_MC)
    logσ_ζM .= view(ϕqc, Val(:logσ_ζM))
    @assert size(ϕm,1) >= n_θM
    assert_ϕm(ϕm, n_MC) # dispatch on vector or matrix
    ρsM = view(ϕqc, Val(:ρsM))
    zcor_endsM = OneBasedVectorWithZero(cor_endsM)
    # ib = 1
    ρ_start = 1
    for ib in axes(cor_endsM, 1)
        r = (zcor_endsM[ib-1]+1):zcor_endsM[ib]
        #scale_r = view(scale, r) 
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

@inline assert_ϕm(ϕm::AbstractVector, n_MC) = nothing
@inline assert_ϕm(ϕm::AbstractMatrix, n_MC) = size(ϕm,2) == n_MC
@inline view_ϕm(ϕm::AbstractMatrix, r::UnitRange{Int}) = view(ϕm, r, :)
@inline view_ϕm(ϕm::AbstractVector, r::UnitRange{Int}) = view(ϕm, r)

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
    nothing
end

function _setρ_unscaled!(ρ::AbstractVector{T}, U::AbstractMatrix{T}) where {T};
    local n = size(U, 1)
    @inbounds for j in 2:n
        scale = U[j,j]
        U[1:j,j] ./= scale
    end
    _uutri2vec!(ρ, U)
    nothing
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
    nothing
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
    nothing
end

