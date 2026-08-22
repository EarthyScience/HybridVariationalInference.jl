function neg_elbo_sites(rng, ϕg::AbstractVector{TG}, ϕq::AbstractVector{TF}, args...;
    n_MC=3, 
    pbm_covar_indices,
    i_sites_train,     # indices of sites in training set
    elbo_helpers,      # tuple of preallocated arrays
    xM,
    kwargs...
) where {TG, TF}
    h = elbo_helpers # preallocated μζP, dμζP, ζsP, ϕms, xMP, dxMP
    extract_μζP!(h.μζP, ϕq) # n_p
    randn!(rng, h.ζsP) # n_P * n_MC
    sample_ζsP!(h.ζsP, ϕq, intθq)
    g_apply!(h.ϕms, ϕg, xM, h.ζP, pbm_covar_indices, g, h.xMP) # n_M x n_MC x n_site
    res_site = map(i_sites_train, h.helpers_sites, eachcol(h.ϕms)
              ) do i_site_train,   hi,              ϕm
        randn!(rng, hi.ζsM) # n_M * n_MC
        # first component needs to be the full elbo
        Lζi(ϕq, ϕm, h.ζsP, hi.ζsM, args...; i_site_train, kwargs...)
    end
    # E = sum(x -> x.E, res_site)
    # loglik = sum(x -> x.loglik, res_site)
    # costTrans = sum(x -> x.costTrans, res_site)
    elbo = sum(first, res_site)
    (; elbo, res_site)
end

function g_apply_zygote(ϕg::AbstractVector{TG}, xM::AbstractMatrix{TG}, 
    ζP::AbstractVector{TF}, pbm_covar_indices::AbstractVector{<:Number}, 
    g::AbstractModelApplicator,
    xMP::AbstractMatrix, is_testmode::Bool=false
    ) where {TG, TF}
    if length(pbm_covar_indices) == 0
        ϕm1 = apply_model(g, xM, ϕg; is_testmode)
    else
        n_cov, n_site = size(xM)
        ζPc = if eltype(xM) !== eltype(ζP)
            convert.(eltype(xM), ζP[pbm_covar_indices])
        else
            ζP[pbm_covar_indices]
            #copy(ζP[pbm_covar_indices]) # know that ζPc and xMP not modified
        end
        xMP = vcat(xM, repeat(ζPc, 1, n_site))        
        ϕm1 = apply_model(g, xMP, ϕg; is_testmode)
    end
end

function g_apply(ϕg::AbstractVector{TG}, xM::AbstractMatrix{TG}, 
    ζP::AbstractVector{TF}, pbm_covar_indices::AbstractVector{<:Number}, 
    g::AbstractModelApplicator,
    xMP::AbstractMatrix, is_testmode::Bool=false
    ) where {TG, TF}
    length(pbm_covar_indices) == 0 && return apply_model(g, xM, ϕg; is_testmode)
    update_xMP!(xMP, xM, ζP, pbm_covar_indices)
    return apply_model(g, xMP, ϕg; is_testmode)
end
function g_apply!(y, ϕg::AbstractVector{TG}, xM::AbstractMatrix{TG}, 
    ζP::AbstractVector{TF}, pbm_covar_indices::AbstractVector{<:Number}, 
    g::AbstractModelApplicator,
    xMP::AbstractMatrix, is_testmode::Bool=false
    ) where {TG, TF}
    length(pbm_covar_indices) == 0 && return apply_model!(y, g, xM, ϕg; is_testmode)
    update_xMP!(xMP, xM, ζP, pbm_covar_indices)
    return apply_model!(y, g, xMP, ϕg; is_testmode)
end
function update_xMP!(xMP::AbstractMatrix{TG}, 
    xM::AbstractMatrix{TG}, ζP::AbstractVector{TF}, pbm_covar_indices::AbstractVector{<:Number}
    ) where {TG, TF}
    n_cov, n_site = size(xM)
    n_covP = length(pbm_covar_indices)
    @assert size(xMP) == (n_cov + n_covP, n_site)
    @inbounds for j in 1:n_site
        # Copy xM block
        for i in 1:n_cov
            xMP[i, j] = xM[i, j]
        end
        # Fill pbm covariates block
        for (k, idx) in enumerate(pbm_covar_indices)
            xMP[n_cov + k, j] = TG === TF ? ζP[idx] : convert(TG, ζP[idx])
        end
    end
end


function Lζi(
    ϕsM::AbstractMatrix{TF}, 
    ϕq::AbstractVector{TF}, 
    ζsP::AbstractMatrix{TF}, 
    zMs::AbstractMatrix{TF}, 
    g, f, py,
    xM::AbstractMatrix, xP, y_ob, y_unc, itrain_sites::AbstractVector{<:Number};
    cor_ends, # =(P=(1,),M=(1,))
    int_ϕg_ϕq::AbstractComponentArrayInterpreter,
    int_ϕq::AbstractComponentArrayInterpreter,
    transP, transMs, 
    priorsP, priorsM,
    penalty_computer = ZeroPenaltyComputer(),
    is_testmode,
    is_omit_priors,
    zero_prior_logdensity,
    approx::AbstractHVIApproximation,
    intθP, intθMs,
    ranef::AbstractRandomEffectsComputer,
    frac_cluster_all,
    i_site_train,
) where {TG, TF}
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