"""
elbo_helpers need to be initialized with new random numbers
in h.ζP and hi.ζM by calling randnPM! before.
By this way, we can compute the derivative corresponding to the forward pass
"""
function pointloss_sites!(
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
    use_dc = h.helpers_sites[1].ζMm isa PAT.DiffCache
    (n_θM,) = use_dc ? size(h.helpers_sites[1].ζMm.du) : size(h.helpers_sites[1].ζMm)
    ϕqPc = intϕqP(ϕqP) 
    ϕqIc = intϕqI(ϕqI)
    pbm_covar_indices = !isnothing(pbm_covar_indices) && isempty(pbm_covar_indices) ? nothing : pbm_covar_indices
    #
    h.ζPm .= CA.getdata(view(ϕqPc,Val(:μζP)))
    # (n_M x n_site)  
    h_ϕms = isnothing(pbm_covar_indices) ? view(h.ϕmsa,:,1,:) : h.ϕmsa
    g_apply!(h_ϕms, ϕg, xM, h.ζPm, pbm_covar_indices, g, h.xMP, is_testmode) 
    exp_ladJacTP = transformζ!(h.θPm, transP, h.ζPm)  # return value captures ladJacT
    # so that one can provide its gradient to the pullback
    ϕm_it = eachslice(h.ϕmsa; dims=3)
    template = ϕqI # only important for gradient
    θPm = h.θPm
    keys_nljoint_args_indi_val = Val(keys(nljoint_args_inds))
    # closure with approx, transM, and kwargs
    function compute_pointlossi_z_cl!(hi, nljoint_args_indi_tup, ϕm) 
        #nljoint_args_indi = NamedTuple{keys_nljoint_args_indi}(nljoint_args_indi_tup)
        nljoint_args_indi = name_tuple(nljoint_args_indi_tup, keys_nljoint_args_indi_val)
        compute_pointlossi_z!(hi, nljoint_args_indi, ϕm, 
        ϕqIc, θPm, cor_ends.M, transM, nljoint_args_fix) 
    end
    #res_site = map(compute_pointlossi_z!, h.helpers_sites, rnormPM.M, i_sites_train, ϕm_it)
    #MAYBE: distributed mapreduce: 
    #   https://docs.julialang.org/en/v1/stdlib/Distributed/#Distributed.@distributed
    #   https://github.com/SupaeroDataScience/DE/blob/main/notebooks/Introduction%20to%20MapReduce.ipynb
    #zind = Test.@inferred zip_eachlastdims(nljoint_args_inds)
    #tmp = Test.@inferred zip(h.helpers_sites, zind, ϕm_it)
    zind = zip_eachlastdims(nljoint_args_inds)
    pointloss_z = mapreduce(compute_pointlossi_z_cl!, +, h.helpers_sites, zind, ϕm_it)
    pointloss = pointloss_z - exp_ladJacTP 
end

function compute_pointlossi_z!(hi, 
    nljoint_args_indi, ϕms::AbstractMatrix, ϕqIc::AbstractVector, θPm::AbstractMatrix, 
    cor_endsM, transM, nljoint_args_fix, ) 
    # DM is 1 (vector) if no population covariates - constant across MC
    #       2 (Matrix) if g is driven by population covariates - differs across MC
    # on update -> sync corresponding function within grad_neg_elbo_sites
    if hi.ζMm isa PAT.DiffCache
        # Gradiend called twice because of different dimensions need to pick the Dual
        template = eltype(ϕqIc) <: ForwardDiff.Dual ? ϕqIc : θPm
        hi = map_leaves_nt(x -> PAT.get_tmp(x, template), hi)
    end
    n_θM = cor_endsM[end]
    hi.ζMm .= view(ϕms, 1:n_θM, :)    # μζM in first n_θM entries
    transformζ!(hi.θMm, transM, hi.ζMm)  
    exp_nL = exp_nLi!(hi.ym, θPm, hi.θMm, nljoint_args_indi, nljoint_args_fix)[1]
    pointlossi = exp_nL 
end

function prepare_pointloss_helpers(approx::AbstractHVIApproximation, 
    ϕg::AbstractArray{TG}, template_TF::AbstractArray{TF};
    n_θP, n_θM, n_site, n_MC, n_cov, n_covP, n_M, cor_ends, n_obs,
    use_diff_cache::Val{use_dc} = Val(true),
    diffchunk::ForwardDiff.Chunk{chunk} = ForwardDiff.Chunk(8),
    ) where {TG, TF, use_dc, chunk}
    his = Tuple((;
        ζMm = Matrix{TF}(undef, n_θM, 1),
        θMm = Matrix{TF}(undef, n_θM, 1),
        ym = Matrix{TF}(undef, n_obs,1),
    ) for i in 1:n_site)
    helpers_sites = use_dc ? map_leaves_nt(x -> PAT.DiffCache(x, chunk), his) : his
    h = (;
        ζPm = Matrix{TF}(undef, n_θP,1),
        θPm = Matrix{TF}(undef, n_θP,1),
        ϕmsa = Array{TF}(undef, n_M, 1, n_site),  
        xMP = Matrix{TG}(undef, (n_cov + n_covP), n_site),
        helpers_sites,
    )
end

function check_pointloss_helpers(h::NamedTuple, xM::AbstractMatrix, pbm_covar_indices)
    n_cov, n_site = size(xM)
    n_covP = isnothing(pbm_covar_indices) ? 0 : length(pbm_covar_indices)
    (n_θP,_) = size(h.ζPm)
    n_M = size(h.ϕmsa, 1)
    @assert size(h.ζPm) == (n_θP, 1)
    @assert size(h.θPm) == (n_θP, 1)
    @assert size(h.ϕmsa) == (n_M, 1, n_site)
    @assert size(h.xMP) == ((n_cov + n_covP), n_site) 
    #
    @assert length(h.helpers_sites) == n_site
    hi = h.helpers_sites[1]
    n_θM = size(hi.ζMm.du, 1)
    n_obs = size(hi.ym.du, 1)
    @assert size(hi.ζMm.du) == (n_θM, 1)
    @assert size(hi.θMm.du) == (n_θM, 1)
    @assert size(hi.ym.du) == (n_obs, 1)
end


