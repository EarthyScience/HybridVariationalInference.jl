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
    approx::AbstractHVIApproximation,
    rnormPM::NamedTuple,          # tuple of random numbers
    ϕg::AbstractVector{TG}, ϕqP::AbstractVector{TF}, ϕqI::AbstractVector{TF}, g, 
    pbm_covar_indices::Union{Nothing,AbstractVector{<:Number}}, 
    args...;
    i_sites_train,     # indices of sites in training set
    intϕqP, intϕqI,
    xM,
    cor_ends,
    is_testmode, 
    kwargs...
) where {TG, TF}
    # for AD do not put it into closure
    h = elbo_helpers # preallocated μζP, dμζP, ζsP, ϕms, xMP, dxMP
    @assert size(rnormPM.P) == size(h.ζsP)
    use_dc = h.helpers_sites[1].ζsM isa PAT.DiffCache
    n_M, n_MC = use_dc ? size(h.helpers_sites[1].ζsM.du) : size(h.helpers_sites[1].ζsM_dc)
    @assert size(rnormPM.M[1]) == (n_M, n_MC)
    ϕqPc = intϕqP(ϕqP) 
    ϕqIc = intϕqI(ϕqI)
    pbm_covar_indices = !isnothing(pbm_covar_indices) && isempty(pbm_covar_indices) ? nothing : pbm_covar_indices
    #
    sample_ζsP!(h.ζsP, h.logσ_ζP, approx, rnormPM.P, ϕqPc, cor_ends.P, h.sample_buffers) # n_P * n_MC
    # (n_M x n_sit)  or (n_M x n_MC x n_sit)    
    ϕms_buffer_key = isnothing(pbm_covar_indices) ? :ϕms : :ϕms_mcs
    #g_apply!(h[ϕms_buffer_key], ϕg, xM, h.ζsP, pbm_covar_indices, g, h.xMP, is_testmode, h.ϕms_mcs2D_buffer) 
    g_apply!(h[ϕms_buffer_key], ϕg, xM, h.ζsP, pbm_covar_indices, g, h.xMP, is_testmode) 
    exp_ladJacTP = transformζ(h.θsP, h.ζsP)  # return value captures ladJacT
    # so that one can provide its gradient to the pullback
    ϕm_it = eachslice(h[ϕms_buffer_key]; dims = ndims(h[ϕms_buffer_key]))
    template = ϕqI # only important for gradient
    θsP = h.θsP
    # closure with approx and kwargs
    function compute_elboi_z_cl!(hi, rnormM, i_site_train, ϕm) 
        compute_nelboi_z!(hi, approx, rnormM, i_site_train, ϕm, 
        ϕqIc, θsP, cor_ends.M; kwargs...) 
    end
    #res_site = map(compute_nelboi_z!, h.helpers_sites, rnormPM.M, i_sites_train, ϕm_it)
    #MAYBE: distributed mapreduce: 
    #   https://docs.julialang.org/en/v1/stdlib/Distributed/#Distributed.@distributed
    #   https://github.com/SupaeroDataScience/DE/blob/main/notebooks/Introduction%20to%20MapReduce.ipynb
    elbo_z = mapreduce(compute_elboi_z_cl!, +, h.helpers_sites, rnormPM.M, i_sites_train, ϕm_it)
    # E = sum(x -> x.E, res_site)
    # loglik = sum(x -> x.loglik, res_site)
    # costTrans = sum(x -> x.costTrans, res_site)
    #elbo = sum(first, res_site) - sum(h.logσ_ζP)
    elbo = elbo_z - exp_ladJacTP - sum(h.logσ_ζP)
    (; elbo, ζsP=copy(h.ζsP), ϕm=copy(h[ϕms_buffer_key]))
end

function compute_nelboi_z!(hi, approx::AbstractHVIApproximation,
    rnormM, i_site_train, ϕm, ϕqIc::AbstractArray{TF}, θsP, cor_endsM;
    kwargs...) where TF
    # on update -> sync corresponding function within grad_neg_elbo_sites
    if hi.ζsM isa PAT.DiffCache
        hi = map_leaves_nt(x -> PAT.get_tmp(x, ϕqIc), hi)
    end
    #ζsM, logσ_ζM, rnorm, ϕqc::AbstractVector{T}, ϕm::AbstractMatrix, buffer_nθM::AbstractVector
    sample_ζsM!(hi.ζsM, hi.logσ_ζM, approx, rnormM, ϕqIc, ϕm, cor_endsM, hi.sample_buffers)
    exp_ladJacTM = transformζ(hi.θsM, hi.ζsM)  # return value captures ladJacT
    # first component needs to be the full elbo
    exp_nL = exp_nLi(θsP, hi.θsM; i_site_train, kwargs...)[1]
    elbozi = exp_nL - exp_ladJacTM - sum(hi.logσ_ζM)
end
# get_tmp_rec_(x::PAT.DiffCache{<:AbstractArray}, template) = PAT.get_tmp(x, template)
# function get_tmp_rec_(x::Union{Tuple,NamedTuple}, template) 
#     map(xi -> get_tmp_rec_(xi, template), x)
# end


function prepare_rnorm(::AbstractVector{TF}; n_θP, n_θM, n_site, n_MC) where TF
    (;
        P = Matrix{TF}(undef, n_θP, n_MC),
        M = Tuple(Matrix{TF}(undef, n_θM, n_MC) for i in 1:n_site),
    )
    # discussed transforming Tuple{Matrix} to a 3D Array.
    # This would run, but make it more difficult and brittle for Enzyme,
    # which would work on a view rather than Matrix. Moreover, accessing
    # the view yields a small performance cost. So keep the Tuple-pattern 
end

function prepare_elbo_helpers(approx::AbstractHVIApproximation, 
    ϕg::AbstractArray{TG}, template_TF::AbstractArray{TF};
    n_θP, n_θM, n_site, n_MC, n_cov, n_covP, n_M, cor_ends,
    use_diff_cache::Val{use_dc} = Val(true),
    diffchunk::ForwardDiff.Chunk{chunk} = ForwardDiff.Chunk(8),
    ) where {TG, TF, use_dc, chunk}
    his = Tuple((;
        ζsM = Matrix{TF}(undef, n_θM, n_MC),
        θsM = Matrix{TF}(undef, n_θM, n_MC),
        logσ_ζM = Vector{TF}(undef, n_θM),
        sample_buffers = prepare_ind_sample_buffers(approx, cor_ends.M, template_TF),
    ) for i in 1:n_site)
    helpers_sites = use_dc ? map_leaves_nt(x -> PAT.DiffCache(x, chunk), his) : his
    h = (;
        ζsP = Matrix{TF}(undef, n_θP, n_MC),
        θsP = Matrix{TF}(undef, n_θP, n_MC),
        logσ_ζP = Vector{TF}(undef, n_θP),
        ϕms = Matrix{TF}(undef, n_M, n_site),        
        ϕms_mcs = Array{TF,3}(undef, n_M, n_MC, n_site),
        ϕms_mcs2D_buffer = Matrix{TF}(undef, n_M, n_MC * n_site),        
        xMP = Matrix{TG}(undef, (n_cov + n_covP), n_MC * n_site),
        sample_buffers = prepare_sample_buffers(approx, cor_ends.P, template_TF),
        diffchunk,
        helpers_sites,
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
    @assert size(h.θsP) == (n_θP, n_MC )
    #@assert size(h.dϕg) == (n_ϕg,)
    @assert size(h.logσ_ζP) == (n_θP,)
    @assert size(h.ϕms) == (n_M, n_site)
    @assert size(h.ϕms_mcs) == (n_M, n_MC, n_site)
    @assert size(h.xMP) == ((n_cov + n_covP), n_MC * n_site) 
    #
    @assert length(h.helpers_sites) == n_site
    hi = h.helpers_sites[1]
    n_θM = size(hi.ζsM.du, 1)
    @assert size(hi.ζsM.du) == (n_θM, n_MC)
    @assert size(hi.θsM.du) == (n_θM, n_MC)
    @assert size(hi.logσ_ζM.du) == (n_θM,)
end

function sample_ζsP!(ζsP, logσ_ζP, ::DiagonalHVIApproximation, rnormP, 
    ϕqc::AbstractVector{T}, cor_endsP, sample_buffers::NamedTuple) where T
    μζP = CA.getdata(view(ϕqc,Val(:μζP)))
    logσ_ζP .= view(ϕqc, Val(:logσ_ζP))
    ζsP .= μζP .+ (exp.(logσ_ζP) .* rnormP)
    nothing
end
prepare_sample_buffers(approx::DiagonalHVIApproximation, cor_endsP, template_TF) = (;)

function sample_ζsM!(ζsM, logσ_ζM, ::DiagonalHVIApproximation, rnorm, 
    ϕqc::AbstractVector{T}, ϕm::Union{AbstractVector, AbstractMatrix}, 
    cor_endsM, sample_buffers) where T
    n_θM, n_MC = size(ζsM)
    @assert size(rnorm) == (n_θM, n_MC)
    @assert size(ϕm,1) >= n_θM
    assert_ϕm(ϕm, n_MC) # dispatch on vector or matrix
    logσ_ζM .= view(ϕqc, Val(:logσ_ζM))
    μζM = view_ϕm(ϕm, 1:n_θM)           # dispatch
    ζsM .= μζM .+ (exp.(logσ_ζM) .* rnorm)       # does not allocate
    # @inbounds for j in 1:n_MC
    #     for i in 1:n_θM
    #         ζsM[i,j] = ϕm[i,j] + rnorm[i,j] * scale[i]
    #     end
    # end
    nothing         
end
@inline assert_ϕm(ϕm::AbstractVector, n_MC) = nothing
@inline assert_ϕm(ϕm::AbstractMatrix, n_MC) = size(ϕm,2) == n_MC
@inline view_ϕm(ϕm::AbstractMatrix, r::Union{Colon,UnitRange{Int}}) = view(ϕm, r, :)
@inline view_ϕm(ϕm::AbstractVector, r::Union{Colon,UnitRange{Int}}) = view(ϕm, r)
prepare_ind_sample_buffers(approx::DiagonalHVIApproximation, cor_endsM, template_TF) = (;)

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

function g_apply!(ϕm::AbstractMatrix{TF}, ϕg::AbstractVector{TG}, xM::AbstractMatrix{TG}, 
    ζsP::AbstractMatrix{TF}, pbm_covar_indices::Nothing, 
    g::AbstractModelApplicator,
    xMP::AbstractMatrix,
    is_testmode::Bool,
    ϕms_mcs2D_buffer = nothing, # only required for 3D case
    ) where {TG, TF}
        apply_model!(ϕm, g, xM, ϕg; is_testmode) # allocates view
        return nothing
end
function g_apply!(ϕm::AbstractArray{TF,3}, ϕg::AbstractVector{TG}, xM::AbstractMatrix{TG},
    ζsP::AbstractMatrix{TF}, pbm_covar_indices::AbstractVector{<:Number},
    g::AbstractModelApplicator, xMP::AbstractMatrix, is_testmode::Bool,
    ϕms_mcs2D_buffer = reshape(ϕm, size(ϕm,1), :) # allocates 48 bytes for an escaping view, preallocate 
    ) where {TG, TF}
    update_xMP!(xMP, xM, ζsP, pbm_covar_indices)
    #Main.@infiltrate_main
    if pointer(parent(ϕms_mcs2D_buffer)) != pointer(parent(ϕm))
        # if supplied a preallocated array (rather than using the default view) theń copy
        copyto!(ϕms_mcs2D_buffer, ϕm) # in order to call apply_model only once, stack n_sites * n_MC
    end
    apply_model!(ϕms_mcs2D_buffer, g, xMP, ϕg; is_testmode)
    if pointer(parent(ϕms_mcs2D_buffer)) != pointer(parent(ϕm))
        copyto!(ϕm, ϕms_mcs2D_buffer) 
    end
    return nothing
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

function transformζ(θs, ζs::AbstractArray{TF}) where TF
    # TODO implement user-defined parameter transformation
    n_MC = size(ζs,2)
    θs .= exp.(ζs)
    ladJacT = sum(ζs) / n_MC
end

"""
compute the expected value of the neative log joint density of observations
and parameters
"""
function exp_nLi(
    θsP::AbstractMatrix,
    θsM::AbstractMatrix;
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
    n_MC = size(θsP,2)
    nL = (5 * sum(θsP) + 3 * sum(θsM)) / n_MC
    (; nL=nL,)
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