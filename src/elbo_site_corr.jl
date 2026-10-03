function sample_ζsP!(ζsP, logσ_ζP, 
    ::Union{MeanHVIApproximation,MeanUniScalingHVIApproximation},
    rnorm, ϕqc::AbstractVector{T}, cor_endsP, sample_buffers::NamedTuple) where T
    μζP = CA.getdata(view(ϕqc,Val(:μζP)))
    logσ_ζP .= view(ϕqc, Val(:logσ_ζP))
    n_θP, n_MC = size(ζsP)
    @assert size(rnorm) == (n_θP, n_MC)
    @assert size(μζP) == (n_θP,)
    ρsP = view(ϕqc, Val(:ρsP))
    zcor_ends = OneBasedVectorWithZero(cor_endsP)
    # ib = 2
    ρ_start = 1
    Ul = sample_buffers.U
    for ib in axes(cor_endsP, 1)
        r = (zcor_ends[ib-1]+1):zcor_ends[ib]
        μζP_r = view_ϕm(μζP, r)           # dispatch
        rnorm_r = view(rnorm, r, :)
        logσ_ζP_r = view(logσ_ζP, r)
        ρ_end = ρ_start-1 + sumn(length(r)-1)
        #U = UpperTriangular(diagm(ones(length(r)))) # TODO preallocate
        #U = sample_buffers.Us[ib] # preallocated UpperTriangular matrix
        U = LinearAlgebra.UpperTriangular(view(Ul, 1:length(r), 1:length(r)))
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

# TODO unify prepare_sample_buffers and prepare_ind_sample_buffers
function prepare_sample_buffers(
    approx::Union{MeanHVIApproximation,MeanUniScalingHVIApproximation}, 
    cor_ends, template_TF::AbstractArray{TF}) where TF
    # zcor_ends = OneBasedVectorWithZero(cor_ends)
    # # make a Tuple, so to be handled by getdiffcache
    # Us = Tuple(begin
    #     nb = zcor_ends[ib] - zcor_ends[ib-1]
    #     U = UpperTriangular(diagm(ones(nb))) 
    # end for ib in axes(cor_ends, 1))
    n_θM = cor_ends[end]
    U = Matrix{TF}(undef, n_θM, n_θM)
    (;U)
end

function sample_ζsM!(ζsM, logσ_ζM, approx::Union{MeanHVIApproximation,MeanUniScalingHVIApproximation}, rnorm, 
    ϕqc::AbstractVector{T}, ϕm::Union{AbstractVector, AbstractMatrix}, 
    cor_endsM, sample_buffers::NamedTuple) where T
    n_θM, n_MC = size(ζsM)
    @assert size(rnorm) == (n_θM, n_MC)
    #logσ_ζM .= view(ϕqc, Val(:logσ_ζM))
    set_marginal_logσ!(logσ_ζM, approx, ϕqc, ϕm)
    @assert size(ϕm,1) >= n_θM
    assert_ϕm(ϕm, n_MC) # dispatch on vector or matrix
    ρsM = view(ϕqc, Val(:ρsM))
    zcor_ends = OneBasedVectorWithZero(cor_endsM)
    Ul = sample_buffers.U
    # ib = 1
    ρ_start = 1
    for ib in axes(cor_endsM, 1)
        r = (zcor_ends[ib-1]+1):zcor_ends[ib]
        μζM_r = view_ϕm(ϕm, r)           # dispatch
        rnorm_r = view(rnorm, r, :)
        logσ_ζM_r = view(logσ_ζM, r)
        ρ_end = ρ_start-1 + sumn(length(r)-1)
        #U = UpperTriangular(diagm(ones(length(r)))) # TODO preallocate
        #U = sample_buffers.Us[ib] # preallocated UpperTriangular matrix
        U = LinearAlgebra.UpperTriangular(view(Ul, 1:length(r), 1:length(r)))
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

function set_marginal_logσ!(logσ_ζMs, ::MeanHVIApproximation, ϕqc, ϕm) 
    logσ_ζMs .= view(ϕqc, Val(:logσ_ζM))  
end
function set_marginal_logσ!(logσ_ζMs, ::MeanUniScalingHVIApproximation, ϕqc, ϕm) 
    n_θM = length(view(ϕqc, Val(:logσ_ζM_offsets))) + 1
    ϕm_scaling = ϕm[n_θM+1]
    logσ_par_offsets = OneBasedVectorWithZero(view(ϕqc, Val(:logσ_ζM_offsets))) # zero based 
    @assert length(logσ_par_offsets) + 1 == n_θM
    logσ_site_offset = logit(ϕm_scaling) # (0..1)->(-Inf, +Inf), 0.5->0
    #
    logσ_ζM_base = log(0.06) # TODO provide by Approx helper 
    logσ_ζMs .= logσ_ζM_base .+ @view(logσ_par_offsets[0:end]) .+ logσ_site_offset
    #logσ_ζMs .= logσ_ζM_base .+ logσ_par_offsets[0:end] .+ logσ_site_offset
    #exp.(logσ_ζMs)
    logσ_ζMs
end

function prepare_ind_sample_buffers(
    approx::Union{MeanHVIApproximation,MeanUniScalingHVIApproximation}, 
    cor_endsM, template_TF::AbstractArray{TF}) where TF
    n_θM = cor_endsM[end]
    U = Matrix{TF}(undef, n_θM, n_θM)
    (; U)
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

