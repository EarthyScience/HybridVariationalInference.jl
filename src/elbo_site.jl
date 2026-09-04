function neg_elbo_sites(rng::AbstractRNG, elbo_helpers::NamedTuple, args; kwargs...)
    CP.randnPM!(rng, h)    
    neg_elbo_sites!(h, args...; kwargs...)
end

function randnPM!(rng, rnorm::NamedTuple)
    randn!(rng, rnorm.P) # n_P * n_MC
    for i in 1:length(rnorm.M)
        randn!(rng, rnorm.M[i])
    end
    nothing
end

"""
elbo_helpers need to be initialized with new random numbers
in h.ζsP and hi.ζsM_dc by calling randnPM! before.
By this way, we can compute the derivative corresponding to the forward pass
"""
function neg_elbo_sites!(
    elbo_helpers::NamedTuple,      # tuple of preallocated arrays
    rnormPM::NamedTuple,          # tuple of random numbers
    ϕg::AbstractVector{TG}, ϕqP::AbstractVector{TF}, ϕqI::AbstractVector{TF}, g, 
    pbm_covar_indices::Union{Nothing,AbstractVector{<:Number}}, 
    args...;
    i_sites_train,     # indices of sites in training set
    intϕqP, intϕqI,
    xM,
    is_testmode, 
    kwargs...
) where {TG, TF}
    h = elbo_helpers # preallocated μζP, dμζP, ζsP, ϕms, xMP, dxMP
    @assert size(rnormPM.P) == size(h.ζsP)
    @assert size(rnormPM.M[1]) == size(h.helpers_sites[1].ζsM_dc.du)
    ϕqPc = intϕqP(ϕqP) 
    ϕqIc = intϕqI(ϕqI)
    sample_ζsP!(h.ζsP, h.logσ2_ζP, rnormPM.P, ϕqPc) # n_P * n_MC
    # (n_M x n_sit)  or (n_M x n_MC x n_sit)    
    ϕms_buffer_key = isnothing(pbm_covar_indices) ? :ϕms : :ϕms_mcs
    g_apply!(h[ϕms_buffer_key], ϕg, xM, h.ζsP, pbm_covar_indices, g, h.xMP, is_testmode) 
    ϕm_it = eachslice(h[ϕms_buffer_key]; dims = ndims(h[ϕms_buffer_key]))
    template = ϕqI # only important for gradient
    # TODO supply all arguments to SL!
    function SL!(hi, rnormM, i_site_train, ϕm) 
            #randn!(rng, hi.ζsM_dc) # n_M * n_MC
            ζsM = PAT.get_tmp(hi.ζsM_dc, template)
            logσ2_ζM = PAT.get_tmp(hi.logσ2_ζM_dc, template)
            buffer_nθM = PAT.get_tmp(hi.buffer_nθM_dc, template)
            #ζsM, logσ2_ζM, rnorm, ϕqc::AbstractVector{T}, ϕm::AbstractMatrix, buffer_nθM::AbstractVector
            
            sample_ζsM!(ζsM, logσ2_ζM, rnormM, ϕqIc, ϕm, buffer_nθM)
            # first component needs to be the full elbo
            Lζi(h.ζsP, ζsM, logσ2_ζM, args...; i_site_train, kwargs...)
    end
    res_site = map(SL!, h.helpers_sites, rnormPM.M, i_sites_train, ϕm_it)
    # E = sum(x -> x.E, res_site)
    # loglik = sum(x -> x.loglik, res_site)
    # costTrans = sum(x -> x.costTrans, res_site)
    elbo = sum(first, res_site) + sum(h.logσ2_ζP)
    (; elbo, ζsP=copy(h.ζsP), ϕm=copy(h[ϕms_buffer_key]), res_site)
end

function prepare_rnorm(::AbstractVector{TF}; n_θP, n_θM, n_site, n_MC) where TF
    (;
        P = Matrix{TF}(undef, n_θP, n_MC),
        M = Tuple(Matrix{TF}(undef, n_θM, n_MC) for i in 1:n_site),
    )
end

function prepare_elbo_helpers(ϕg::AbstractArray{TG}, ::AbstractArray{TF};
    n_θP, n_θM, n_site, n_MC, n_cov, n_covP, n_M
    ) where {TG, TF}
    h = (;
        ζsP = Matrix{TF}(undef, n_θP, n_MC),
        logσ2_ζP = Vector{TF}(undef, n_θP),
        ϕms = Matrix{TF}(undef, n_M, n_site),
        ϕms_mcs = Array{TF,3}(undef, n_M, n_MC, n_site),
        # ϕms = Matrix{TG}(undef, n_M, n_site),
        # ϕms_mcs = Array{TG,3}(undef, n_M, n_MC, n_site),
        xMP = Matrix{TG}(undef, (n_cov + n_covP), n_MC * n_site),
        dϕg = Vector{TG}(undef, length(ϕg)),
    )
    helpers_sites = Tuple((;
        ζsM_dc = PAT.DiffCache(Matrix{TF}(undef, n_θM, n_MC)),
        logσ2_ζM_dc = PAT.DiffCache(Vector{TF}(undef, n_θM)),
        buffer_nθM_dc = PAT.DiffCache(Vector{TF}(undef, n_θM)),
    ) for i in 1:n_site)
    h = (;h... , helpers_sites, 
        dxMP  = zero(h.xMP),      # shadow for xMP
        dϕms = zero(h.ϕms),
        dϕms_mcs = zero(h.ϕms_mcs),
        ∂elbo_∂ζP = zero(h.ζsP),
        ∂elbo_∂ϕm_∂ζP = zero(h.ζsP),
    )   
end

function check_elbo_helpers(h::NamedTuple, xM::AbstractMatrix, pbm_covar_indices;
    n_ϕg
    )
    n_cov, n_site = size(xM)
    n_covP = isnothing(pbm_covar_indices) ? 0 : length(pbm_covar_indices)
    n_θP, n_MC = size(h.ζsP)
    n_M = size(h.ϕms, 1)
    @assert size(h.ζsP) == (n_θP, n_MC )
    @assert size(h.dϕg) == (n_ϕg,)
    @assert size(h.logσ2_ζP) == (n_θP,)
    @assert size(h.ϕms) == (n_M, n_site)
    @assert size(h.ϕms_mcs) == (n_M, n_MC, n_site)
    @assert size(h.xMP) == ((n_cov + n_covP), n_MC * n_site) 
    @assert size(h.dϕg) == (n_ϕg,)
    @assert size(h.dxMP) == size(h.xMP)
    @assert size(h.dϕms) == size(h.ϕms)
    @assert size(h.dϕms_mcs) == size(h.ϕms_mcs)
    @assert size(h.∂elbo_∂ζP) == size(h.ζsP)
    @assert size(h.∂elbo_∂ϕm_∂ζP) == size(h.ζsP)
    #
    @assert length(h.helpers_sites) == n_site
    hi = h.helpers_sites[1]
    n_θM = size(hi.ζsM_dc.du, 1)
    @assert size(hi.ζsM_dc.du) == (n_θM, n_MC)
    @assert size(hi.logσ2_ζM_dc.du) == (n_θM,)
    @assert size(hi.buffer_nθM_dc.du) == (n_θM,)
end




function sample_ζsP!(ζsP, logσ2_ζP, rnormP, ϕqc::AbstractVector{T}) where T
    # TODO replace by proper sampling of full covariance matrix
    μζP = CA.getdata(ϕqc[Val(:μζP)])
    logσ2_ζP .= view(ϕqc, Val(:logσ2_ζP))
    # ζsP * diagm(v) is the same as ζsP .* v'
    ζsP .= μζP .+ (rnormP .* exp.(logσ2_ζP ./ T(2))')
    nothing
end

# with Vector, all MCs have the same mean
function sample_ζsM!(ζsM, logσ2_ζM, rnorm, ϕqIc::AbstractVector{T}, ϕm::AbstractVector, buffer_nθM::AbstractVector) where T
    # TODO replace by proper sampling of full covariance matrix
    # TODO add scaling by factor in ϕm / dispatch by approach
    n_θM, n_MC = size(ζsM)
    logσ2_ζM .= view(ϕqIc, Val(:logσ2_ζM))
    @assert size(buffer_nθM) == (n_θM,)
    scale = buffer_nθM
    @. scale = exp(logσ2_ζM / T(2))
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
function sample_ζsM!(ζsM, logσ2_ζM, rnorm, ϕqc::AbstractVector{T}, ϕm::AbstractMatrix, buffer_nθM::AbstractVector) where T
    n_θM, n_MC = size(ζsM)
    @assert size(rnorm) == (n_θM, n_MC)
    logσ2_ζM .= view(ϕqc, Val(:logσ2_ζM))
    @assert size(ϕm,1) >= n_θM
    @assert size(ϕm,2) == n_MC
    # TODO avoid allocation with subsetting non-last column
    # μζM = ϕm[1:n_θM,:]
    @assert size(buffer_nθM) == (n_θM,)
    scale = buffer_nθM
    @. scale = exp(logσ2_ζM / T(2))
    μζM = view(ϕm, 1:n_θM, :)           # view of the mean block (n_θM × n_MC)
    ζsM .= μζM .+ (rnorm .* scale')       # does not allocate
    # @inbounds for j in 1:n_MC
    #     for i in 1:n_θM
    #         ζsM[i,j] = ϕm[i,j] + rnorm[i,j] * scale[i]
    #     end
    # end
    nothing         
end

# if pbm_covar_indices is nothing, return only a Matrix (n_m x n_site)
# otherwise return an Array (n_m x n_MC x n_site)
function g_apply_oop(ϕg::AbstractVector{TG}, xM::AbstractMatrix{TG}, 
    ζsP::AbstractMatrix{TF}, pbm_covar_indices::Nothing, 
    g::AbstractModelApplicator,
    xMP::AbstractMatrix;
    is_testmode::Bool=false
    ) where {TG, TF}
    ϕm1 = apply_model(g, xM, ϕg; is_testmode)
end
function g_apply_oop(ϕg::AbstractVector{TG}, xM::AbstractMatrix{TG}, 
    ζsP::AbstractMatrix{TF}, pbm_covar_indices::AbstractVector{<:Number}, 
    g::AbstractModelApplicator,
    xMP::AbstractMatrix;
    is_testmode::Bool=false
    ) where {TG, TF}
    if length(pbm_covar_indices) == 0
        n_θP, n_MC = size(ζsP)
        ϕm1 = g_apply_oop(ϕg, xM, ζsP, nothing, g, xMP; is_testmode)
        # Reshape to (n_rows × 1 × n_cols) then repeat n_MC times along dim 2
        ϕms = repeat(reshape(ϕm1, size(ϕm1, 1), 1, size(ϕm1, 2)), 1, n_MC, 1)        
    else
        n_cov, n_site = size(xM)
        n_θP, n_MC = size(ζsP)
        ζsPc = if eltype(xM) !== eltype(ζsP)
            convert.(eltype(xM), ζsP[pbm_covar_indices,:]) 
        else
            ζsP[pbm_covar_indices,:] # know that ζsPc and xMP not modified no copy needed
        end
        # repeat driver columns each n_MC times
        # repeat global parameters matrix n_site times
        # to run the ML model once for n_MC x n_site inputs
        xMP = vcat(repeat(xM, inner = SA.SA[1, n_MC]), repeat(ζsPc, 1, n_site))
        ϕm_long = apply_model(g, xMP, ϕg; is_testmode)
        ϕm = reshape(ϕm_long, :, n_MC, n_site)
    end
end

# function g_apply(ϕg::AbstractVector{TG}, xM::AbstractMatrix{TG}, 
#     ζP::AbstractVector{TF}, pbm_covar_indices::AbstractVector{<:Number}, 
#     g::AbstractModelApplicator,
#     xMP::AbstractMatrix, is_testmode::Bool=false
#     ) where {TG, TF}
#     length(pbm_covar_indices) == 0 && return apply_model(g, xM, ϕg; is_testmode)
#     update_xMP!(xMP, xM, ζP, pbm_covar_indices)
#     return apply_model(g, xMP, ϕg; is_testmode)
# end
function g_apply!(ϕm::AbstractMatrix{TF}, ϕg::AbstractVector{TG}, xM::AbstractMatrix{TG}, 
    ζsP::AbstractMatrix{TF}, pbm_covar_indices::Nothing, 
    g::AbstractModelApplicator,
    xMP::AbstractMatrix,
    is_testmode::Bool
    ) where {TG, TF}
        apply_model!(ϕm, g, xM, ϕg; is_testmode) # allocates view
        return nothing
end
function g_apply!(ϕm::AbstractArray{TF,3}, ϕg::AbstractVector{TG}, xM::AbstractMatrix{TG}, 
    ζsP::AbstractMatrix{TF}, pbm_covar_indices::AbstractVector{<:Number}, 
    g::AbstractModelApplicator,
    xMP::AbstractMatrix,
    is_testmode::Bool
    ) where {TG, TF}
    if length(pbm_covar_indices) == 0 
        n_M, n_MC, n_site = size(ϕm)
        @assert size(ζsP,2) == n_MC
        y1 = view(ϕm,:,1,:)
        apply_model!(y1, g, xM, ϕg; is_testmode) # allocates view
        for j in 2:n_MC
            ϕm[:,j,:] .= y1
        end
        return nothing
    end
    update_xMP!(xMP, xM, ζsP, pbm_covar_indices)
    yr = reshape(ϕm, size(ϕm,1),:) # view collapses 3'r dim, apply_model! updates underlying y
    apply_model!(yr, g, xMP, ϕg; is_testmode)
    return nothing # no need to return primal for proper gradients of return to whatever
    #return y
end
function update_xMP!(xMP::AbstractMatrix{TG}, 
    xM::AbstractMatrix{TG}, ζsP::AbstractMatrix{TF}, pbm_covar_indices::AbstractVector{<:Number}
    ) where {TG, TF}
    n_θP, n_MC = size(ζsP)
    n_cov, n_site = size(xM)
    n_covP = length(pbm_covar_indices)
    @assert size(xMP) == ((n_cov + n_covP) , n_site * n_MC)
    @inbounds for i in 1:n_site
        for j in 1:n_MC
            # Copy xM block
            ic = (i-1)*n_MC + j
            for k in 1:n_cov
                xMP[k, ic] = xM[k, i]
            end
            # Fill pbm covariates block
            for (k, idx) in enumerate(pbm_covar_indices)
                xMP[n_cov + k, ic] = TG === TF ? ζsP[idx,j] : convert(TG, ζsP[idx,j])
            end
        end
    end
end

function Lζi(
    ζsP::AbstractMatrix,
    ζsM::AbstractMatrix,
    logσ2_ζM::AbstractVector; 
    # f, py,
    # xP, y_ob, y_unc, itrain_sites::AbstractVector{<:Number};
    # cor_ends, # =(P=(1,),M=(1,))
    # int_ϕg_ϕq::AbstractComponentArrayInterpreter,
    # int_ϕq::AbstractComponentArrayInterpreter,
    # transP, transMs, 
    # priorsP, priorsM,
    # penalty_computer = ZeroPenaltyComputer(),
    # is_omit_priors,
    # zero_prior_logdensity,
    # approx::AbstractHVIApproximation,
    # intθP, intθMs,
    # ranef::AbstractRandomEffectsComputer,
    # frac_cluster_all,
    i_site_train,
) 
    elbo_site = 5 * sum(ζsP) + 3 * sum(ζsM) + sum(logσ2_ζM)
    (; E=elbo_site,)
    # ζMs = sample_ζMs(zMs, ϕMs, intθMs)
    # ϕc = int_ϕg_ϕq(ϕ)
    # VT= typeof(@view(ϕ[1:1]))
    # ϕg = CA.getdata(ϕc[Val(:ϕq)])
    # ϕqc = ϕc[Val(:ϕq)]
    # #ϕq = CA.getdata(ϕqc)::VT
    # if(!all(isfinite.(ϕ)))
    #     @show ϕqc
    #     @show ϕg
    #     error("encountered non-finite optimized parameters")
    # end
    # ζsP, ζsMs_tr, σ = generate_ζ(approx, rng, g, ϕ, xM; n_MC, cor_ends, pbm_covar_indices,
    #     int_ϕq, int_ϕg_ϕq, is_testmode, itrain_sites, ranef)
    # ζsP_cpu = cdev(ζsP) # fetch to CPU, because for <1000 sites (n_batch) this is faster
    # ζsMs_tr_cpu = cdev(ζsMs_tr) # fetch to CPU, because for <1000 sites (n_batch) this is faster
    # #
    # # maybe: translate ζ once and supply to both neg_elbo and negloglik_meanθ
    # loss_comps = neg_elbo_ζtf(
    #     ζsP_cpu[:,1:n_MC], ζsMs_tr_cpu[:,:,1:n_MC], σ, f, py, xP, y_ob, y_unc;
    #     n_MC_cap, transP, transMs, priorsP, priorsM, 
    #     penalty_computer, ϕg, ϕqc, is_omit_priors, zero_prior_logdensity, 
    #     itrain_sites, intθMs, intθP, ranef, frac_cluster_all)
end