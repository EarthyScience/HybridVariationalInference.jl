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
    h.ζP .= CA.getdata(view(ϕqPc,Val(:μζP)))
    # (n_M x n_site)  
    # TODO g_apply without n_MC
    ϕms = isnothing(pbm_covar_indices) ? view(h.ϕms,:,:,1) : h.ϕms
    g_apply!(ϕms, ϕg, xM, h.ζP, pbm_covar_indices, g, h.xMP, is_testmode) 
    exp_ladJacTP = transformζ!(h.θP, transP, h.ζP)  # return value captures ladJacT
    # so that one can provide its gradient to the pullback
    ϕm_it = eachcol(view(h.ϕms,:,:,1))
    template = ϕqI # only important for gradient
    θP = h.θP
    keys_nljoint_args_indi_val = Val(keys(nljoint_args_inds))
    # closure with approx, transM, and kwargs
    function compute_pointlossi_z_cl!(hi, nljoint_args_indi_tup, ϕm) 
        #nljoint_args_indi = NamedTuple{keys_nljoint_args_indi}(nljoint_args_indi_tup)
        nljoint_args_indi = name_tuple(nljoint_args_indi_tup, keys_nljoint_args_indi_val)
        compute_pointlossi_z!(hi, nljoint_args_indi, ϕm, 
        ϕqIc, θP, cor_ends.M, transM, nljoint_args_fix) 
    end
    #res_site = map(compute_pointlossi_z!, h.helpers_sites, rnormPM.M, i_sites_train, ϕm_it)
    #MAYBE: distributed mapreduce: 
    #   https://docs.julialang.org/en/v1/stdlib/Distributed/#Distributed.@distributed
    #   https://github.com/SupaeroDataScience/DE/blob/main/notebooks/Introduction%20to%20MapReduce.ipynb
    #zind = Test.@inferred zip_eachlastdims(nljoint_args_inds)
    zind = zip_eachlastdims(nljoint_args_inds)
    pointloss_z = mapreduce(compute_pointlossi_z_cl!, +, h.helpers_sites, zind, ϕm_it)
    pointloss = pointloss_z - exp_ladJacTP 
end

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
        ζM = Matrix{TF}(undef, n_θM, 1),
        θM = Matrix{TF}(undef, n_θM, 1),
        y = Matrix{TF}(undef, n_obs,1),
    ) for i in 1:n_site)
    helpers_sites = use_dc ? map_leaves_nt(x -> PAT.DiffCache(x, chunk), his) : his
    h = (;
        ζP = Matrix{TF}(undef, n_θP,1),
        θP = Matrix{TF}(undef, n_θP,1),
        ϕms = Array{TF}(undef, n_M, n_site,1),        
        xMP = Matrix{TG}(undef, (n_cov + n_covP), n_site),
        helpers_sites,
    )
end

function check_pointloss_helpers(h::NamedTuple, xM::AbstractMatrix, pbm_covar_indices)
    n_cov, n_site = size(xM)
    n_covP = isnothing(pbm_covar_indices) ? 0 : length(pbm_covar_indices)
    (n_θP,) = size(h.ζP)
    n_M = size(h.ϕms, 1)
    @assert size(h.ζP) == (n_θP, 1)
    @assert size(h.θP) == (n_θP, 1)
    @assert size(h.ϕms) == (n_M, n_site,1)
    @assert size(h.xMP) == ((n_cov + n_covP), n_site) 
    #
    @assert length(h.helpers_sites) == n_site
    hi = h.helpers_sites[1]
    n_θM = size(hi.ζM.du, 1)
    n_obs = size(hi.y.du, 1)
    @assert size(hi.ζM.du) == (n_θM, 1)
    @assert size(hi.θM.du) == (n_θM, 1)
    @assert size(hi.y.du) == (n_obs, 1)
end


