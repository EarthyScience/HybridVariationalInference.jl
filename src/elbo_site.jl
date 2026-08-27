function neg_elbo_sites(rng, ϕg::AbstractVector{TG}, ϕq::AbstractVector{TF}, g, 
    pbm_covar_indices::Union{Nothing,AbstractVector{<:Number}}, 
    elbo_helpers::NamedTuple,      # tuple of preallocated arrays
    args...;
    n_MC=3, 
    i_sites_train,     # indices of sites in training set
    intϕq,
    xM,
    is_testmode, 
    kwargs...
) where {TG, TF}
    h = elbo_helpers # preallocated μζP, dμζP, ζsP, ϕms, xMP, dxMP
    ϕqc = intϕq(ϕq) 
    μζP = ϕqc[Val(:μζP)]
    randn!(rng, h.ζsP) # n_P * n_MC
    sample_ζsP!(h.ζsP, ϕqc)
    # (n_M x n_sit)  or (n_M x n_MC x n_sit)    
    ϕm_buffer_key = isnothing(pbm_covar_indices) ? :ϕms : :ϕms_mcs
    g_apply!(h[ϕm_buffer_key], ϕg, xM, h.ζsP, pbm_covar_indices, g, h.xMP) 
    ϕm_it = eachslice(h[ϕm_buffer_key]; dims = ndims(h[ϕm_buffer_key]))
    res_site = map(i_sites_train, h.helpers_sites, ϕm_it
              ) do i_site_train,   hi,              ϕm
        randn!(rng, hi.ζsM) # n_M * n_MC
        #hi.ζsM .= one(eltype(hi.ζsM))
        sample_ζsM!(hi.ζsM, ϕqc, ϕm)
        # first component needs to be the full elbo
        Lζi(h.ζsP, hi.ζsM, args...; i_site_train, kwargs...)
    end
    # E = sum(x -> x.E, res_site)
    # loglik = sum(x -> x.loglik, res_site)
    # costTrans = sum(x -> x.costTrans, res_site)
    elbo = sum(first, res_site)
    (; elbo, res_site)
end



function sample_ζsP!(ζsP, ϕqc)
    # TODO replace by proper sampling of full covariance matrix
    # for now just add the mean
    ζsP .+= ϕqc[Val(:μζP)]
end

# with Vector, all MCs have the same mean
function sample_ζsM!(ζsM, ϕqc, ϕm::AbstractVector)
    # TODO replace by proper sampling of full covariance matrix
    # for now just add the mean
    n_θM, n_MC = size(ζsM)
    μM = ϕm[1:n_θM]
    ζsM .+= μM
end

# with Matrix, there is a site mean for each mc-sample
function sample_ζsM!(ζsM, ϕqc, ϕm::AbstractMatrix)
    # TODO replace by proper sampling of full covariance matrix
    # for now just add the mean
    n_θM, n_MC = size(ζsM)
    @assert size(ϕm,1) >= n_θM
    @assert size(ϕm,2) == n_MC
    @inbounds for j in 1:n_MC
        ζsM[:,j] .+= ϕm[1:n_θM,j]
    end
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
function g_apply!(y::AbstractMatrix{TG}, ϕg::AbstractVector{TG}, xM::AbstractMatrix{TG}, 
    ζsP::AbstractMatrix{TF}, pbm_covar_indices::Nothing, 
    g::AbstractModelApplicator,
    xMP::AbstractMatrix, is_testmode::Bool=false
    ) where {TG, TF}
        apply_model!(y, g, xM, ϕg; is_testmode) # allocates view
        return nothing
end
function g_apply!(y::AbstractArray{TG,3}, ϕg::AbstractVector{TG}, xM::AbstractMatrix{TG}, 
    ζsP::AbstractMatrix{TF}, pbm_covar_indices::AbstractVector{<:Number}, 
    g::AbstractModelApplicator,
    xMP::AbstractMatrix, is_testmode::Bool=false
    ) where {TG, TF}
    if length(pbm_covar_indices) == 0 
        n_M, n_MC, n_site = size(y)
        @assert size(ζsP,2) == n_MC
        y1 = view(y,:,1,:)
        apply_model!(y1, g, xM, ϕg; is_testmode) # allocates view
        for j in 2:n_MC
            y[:,j,:] .= y1
        end
        return nothing
    end
    update_xMP!(xMP, xM, ζsP, pbm_covar_indices)
    yr = reshape(y, size(y,1),:) # view collapses 3'r dim, apply_model! updates underlying y
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
    @show n_covP, size(xMP)
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
    ζsP::AbstractMatrix{TF},
    ζsM::AbstractMatrix{TF}; 
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
) where {TF}
    5 * sum(ζsP) + 3 * sum(ζsM)
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