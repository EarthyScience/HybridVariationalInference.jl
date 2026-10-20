"""
elbo_helpers need to be initialized with new random numbers
in h.ζP and hi.ζM by calling randnPM! before.
By this way, we can compute the derivative corresponding to the forward pass
"""
function neg_pointloss_sites!(
    pointloss_helpers::NamedTuple,      # tuple of preallocated arrays
    ϕ::NamedTuple,
    sample_args::NamedTuple,
    indiv_args::NamedTuple,
    nljoint_args_fix::NamedTuple;
) 
    # for AD do not put it into closure
    h = pointloss_helpers # preallocated μζP, dμζP, ζsP, ϕms, xMP, dxMP
    (;ϕg, ϕqP, ϕqI) = ϕ
    (;g, is_testmode, pbm_covar_indices, intϕqP, intϕqI, cor_ends, transP, transM) = sample_args
    @assert keys(indiv_args)[1] == :xM
    xM = indiv_args.xM; r2end = 2:length(indiv_args)
    nljoint_args_inds = NamedTuple{keys(indiv_args)[r2end]}(values(indiv_args)[r2end])
    use_dc = h.helpers_sites[1].ζM isa PAT.DiffCache
    (n_M,) = use_dc ? size(h.helpers_sites[1].ζM.du) : size(h.helpers_sites[1].ζM)
    ϕqPc = intϕqP(ϕqP) 
    ϕqIc = intϕqI(ϕqI)
    pbm_covar_indices = !isnothing(pbm_covar_indices) && isempty(pbm_covar_indices) ? nothing : pbm_covar_indices
    #
    h.ζP = CA.getdata(view(ϕqc,Val(:μζP)))
    # (n_M x n_site)  
    # TODO g_apply without n_MC
    g_apply!(h.ϕms, ϕg, xM, h.ζP, pbm_covar_indices, g, h.xMP, is_testmode) 
    exp_ladJacTP = transformζ!(h.θP, transP, h.ζP)  # return value captures ladJacT
    # so that one can provide its gradient to the pullback
    ϕm_it = eachscol(h.ϕms)
    template = ϕqI # only important for gradient
    θP = h.θP
    keys_nljoint_args_indi_val = Val(keys(nljoint_args_inds))
    # closure with approx, transM, and kwargs
    function compute_nelboi_z_cl!(hi, rnormM, nljoint_args_indi_tup, ϕm) 
        #nljoint_args_indi = NamedTuple{keys_nljoint_args_indi}(nljoint_args_indi_tup)
        nljoint_args_indi = add_names(nljoint_args_indi_tup, keys_nljoint_args_indi_val)
        compute_pointlossi_z!(hi, nljoint_args_indi, ϕm, 
        ϕqIc, θP, cor_ends.M, transM, nljoint_args_fix) 
    end
    #res_site = map(compute_pointlossi_z!, h.helpers_sites, rnormPM.M, i_sites_train, ϕm_it)
    #MAYBE: distributed mapreduce: 
    #   https://docs.julialang.org/en/v1/stdlib/Distributed/#Distributed.@distributed
    #   https://github.com/SupaeroDataScience/DE/blob/main/notebooks/Introduction%20to%20MapReduce.ipynb
    elbo_z = mapreduce(compute_nelboi_z_cl!, +, h.helpers_sites, rnormPM.M, zip_eachlastdims(nljoint_args_inds), ϕm_it)
    # E = sum(x -> x.E, res_site)
    # loglik = sum(x -> x.loglik, res_site)
    # costTrans = sum(x -> x.costTrans, res_site)
    #elbo = sum(first, res_site) - sum(h.logσ_ζP)
    elbo = elbo_z - exp_ladJacTP - sum(h.logσ_ζP)
end

add_names(tup, ::Val{KEYS}) where KEYS = NamedTuple{KEYS}(tup)

function compute_pointlossi_z!(hi, 
    nljoint_args_indi, ϕm::AbstractArray, ϕqIc::AbstractArray, θP, 
    cor_endsM, transM, nljoint_args_fix, ) 
    # DM is 1 (vector) if no population covariates - constant across MC
    #       2 (Matrix) if g is driven by population covariates - differs across MC
    # on update -> sync corresponding function within grad_neg_elbo_sites
    if hi.ζM isa PAT.DiffCache
        # Gradiend called twice because of different dimensions need to pick the Dual
        template = eltype(ϕqIc) <: ForwardDiff.Dual ? ϕqIc : θP
        hi = map_leaves_nt(x -> PAT.get_tmp(x, template), hi)
    end
    n_θM = cor_endsM[end]
    hi.ζM .= view(ϕm, 1:n_θM, :)    
    transformζ!(hi.θM, transM, hi.ζM)  
    exp_nL = exp_nLi!(hi.y, θP, hi.θM, nljoint_args_indi, nljoint_args_fix)[1]
    pointlossi = exp_nL 
end

function prepare_pointloss_helpers(approx::AbstractHVIApproximation, 
    ϕg::AbstractArray{TG}, template_TF::AbstractArray{TF};
    n_θP, n_θM, n_site, n_MC, n_cov, n_covP, n_M, cor_ends, n_obs,
    use_diff_cache::Val{use_dc} = Val(true),
    diffchunk::ForwardDiff.Chunk{chunk} = ForwardDiff.Chunk(8),
    ) where {TG, TF, use_dc, chunk}
    his = Tuple((;
        ζM = Vector{TF}(undef, n_θM),
        θM = Vector{TF}(undef, n_θM),
        y = Vector{TF}(undef, n_obs),
        #sample_buffers = prepare_ind_sample_buffers(approx, cor_ends.M, template_TF),
    ) for i in 1:n_site)
    helpers_sites = use_dc ? map_leaves_nt(x -> PAT.DiffCache(x, chunk), his) : his
    h = (;
        ζP = Vector{TF}(undef, n_θP),
        θP = Vector{TF}(undef, n_θP),
        logσ_ζP = Vector{TF}(undef, n_θP),
        ϕms = Matrix{TF}(undef, n_M, n_site),        
        ϕms_mcs2D_buffer = Matrix{TF}(undef, n_M, n_MC * n_site),        
        xMP = Matrix{TG}(undef, (n_cov + n_covP), n_MC * n_site),
        sample_buffers = prepare_sample_buffers(approx, cor_ends.P, template_TF),
        #diffchunk,
        helpers_sites,
    )
end

function check_pointloss_helpers(h::NamedTuple, xM::AbstractMatrix, pbm_covar_indices;
    n_ϕg
    )
    n_cov, n_site = size(xM)
    n_covP = isnothing(pbm_covar_indices) ? 0 : length(pbm_covar_indices)
    n_θP, n_MC = size(h.ζP)
    n_M = size(h.ϕms, 1)
    @assert size(h.ζP) == (n_θP, n_MC )
    @assert size(h.θP) == (n_θP, n_MC )
    #@assert size(h.dϕg) == (n_ϕg,)
    @assert size(h.logσ_ζP) == (n_θP,)
    @assert size(h.ϕms) == (n_M, n_site)
    @assert size(h.ϕms_mcs) == (n_M, n_MC, n_site)
    @assert size(h.xMP) == ((n_cov + n_covP), n_MC * n_site) 
    #
    @assert length(h.helpers_sites) == n_site
    hi = h.helpers_sites[1]
    n_θM = size(hi.ζM.du, 1)
    n_obs = size(hi.y.du, 1)
    @assert size(hi.ζM.du) == (n_θM, n_MC)
    @assert size(hi.θsM.du) == (n_θM, n_MC)
    @assert size(hi.y.du) == (n_obs, n_MC)
    @assert size(hi.logσ_ζM.du) == (n_θM,)
end


"""
compute the expected value of the negative log joint density of observations
and parameters
"""
function exp_nLi!(
    y_pred::AbstractVector, 
    θP::AbstractVector,
    θM::AbstractVector,
    nljoint_args_indi::NamedTuple,
    nljoint_args_fix::NamedTuple,
    # f, py,
    # priorsP, priorsM,
    # penalty_computer = ZeroPenaltyComputer(),
    # is_omit_priors,
    # zero_prior_logdensity,
    # intθP, intθMs,
    # ranef::AbstractRandomEffectsComputer,
    # frac_cluster_all,
) 
    (n_obs,) = size(y_pred)
    (;xP, y_o, y_unc, i_indiv_train) = nljoint_args_indi
    @assert size(y_o) == size(y_unc) == (n_obs,)
    (;f!,fLogLik) = nljoint_args_fix
    #θsPc = CA.ComponentMatrix(θsP, axθP, CA.FlatAxis())
    #θsMc = CA.ComponentMatrix(θsM, axθM, CA.FlatAxis())
    apply_model!(y_pred, f!, θP, θM, xP)
    nL_reg = sum(y_pred) / n_MC
    nL = fLogLik(y_pred, y_o, y_unc) / n_MC
    # nL = (5 * sum(θsP) + 3 * sum(θsM)) / n_MC
    (; nL,)
    # loss_comps = neg_elbo_ζtf(
    #     ζsP_cpu[:,1:n_MC], ζsMs_tr_cpu[:,:,1:n_MC], σ, f, py, xP, y_ob, y_unc;
    #     n_MC_cap, transP, transMs, priorsP, priorsM, 
    #     penalty_computer, ϕg, ϕqc, is_omit_priors, zero_prior_logdensity, 
    #     itrain_sites, intθMs, intθP, ranef, frac_cluster_all)
end

function fθpop_dummy1!(pred, θs, xPc::CA.ComponentVector{T}) where T
    # variant that computes the model jointly across parameter vectors in a broadcast
    #   using @row_valviews
    local n_θP, n_θM, n_MC, θsPc, θsMc
    a1 = view(θs, Val(:a1), :)
    @row_valviews a1,a2,a3,b1,b2,b3 = θs
    # regression to nL = (5 * sum(θsP) + 3 * sum(θsM)) / n_MC
    #n_θP = 3; n_θM = 3
    n_obs = length(view(xPc, Val(:S1)))
    n_θ, n_MC = size(θs)
    pred .= zero(eltype(pred))
    for θa in (a1,a2,a3)
        pred .+= (xPc ./ xPc) .* T(5) .* θa'
    end
    for θb in (b1,b2,b3)
        pred .+= (xPc ./ xPc) .* T(3) .* θb'
    end
    pred ./= n_obs
    pred
end

function fθpop_dummy2!(pred, θs, xPc::CA.ComponentVector{T}) where T
    # variant that runs the model separately for each parameter vector
    #   using eachcol(θs) and @unpack_val
    #   using view(pred,:,j) to avoid intermediate allocations
    pred .= zero(eltype(pred))
    n_obs = length(view(xPc, Val(:S1)))
    for (j,θ) in enumerate(eachcol(θs))
        @unpack_val a1,a2,a3,b1,b2,b3 = θ # now scalar parameters
        for θa in (a1,a2,a3)
            view(pred,:,j) .+= (xPc ./ xPc) .* T(5) .* θa
        end
        for θb in (b1,b2,b3)
            view(pred,:,j) .+= (xPc ./ xPc) .* T(3) .* θb
        end
    end
    pred ./= n_obs
    pred
end

function fLogLik_dummy(y_pred, y_o, y_unc)
    sum(y_pred)
end

