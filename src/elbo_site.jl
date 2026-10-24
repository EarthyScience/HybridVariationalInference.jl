function randnPM!(rng, rnorm::NamedTuple)
    randn!(rng, rnorm.P) # n_P * n_MC
    for i in 1:length(rnorm.M)
        randn!(rng, rnorm.M[i])
    end
    nothing
end

"""
    Dispatch to make some part of neg_elbo_sites! polymorphic
"""
abstract type AbstractElboLossMode end
struct NegElbo <: AbstractElboLossMode end
struct NegLogDensity <: AbstractElboLossMode end

"""
elbo_helpers need to be initialized with new random numbers
in h.ζsP and hi.ζsM by calling randnPM! before.
By this way, we can compute the derivative corresponding to the forward pass
"""
function neg_elbo_sites!(
    elbo_helpers::NamedTuple,      # tuple of preallocated arrays
    loss_mode::AbstractElboLossMode,
    ϕ::NamedTuple,
    rnormPM::NamedTuple,
    sample_args::NamedTuple,
    indiv_args::NamedTuple,
    nljoint_args_fix::NamedTuple;
) 
    # for AD do not put it into closure
    h = elbo_helpers # preallocated μζP, dμζP, ζsP, ϕms, xMP, dxMP
    (;ϕg, ϕqP, ϕqI) = ϕ
    (;approx, g, is_testmode, pbm_covar_indices, intϕqP, intϕqI, cor_ends, transP, transM) = sample_args
    @assert keys(indiv_args)[1] == :xM
    xM = indiv_args.xM; r2end = 2:length(indiv_args)
    nljoint_args_inds = NamedTuple{keys(indiv_args)[r2end]}(values(indiv_args)[r2end])
    @assert size(rnormPM.P) == size(h.ζsP)
    use_dc = h.helpers_sites[1].ζsM isa PAT.DiffCache
    n_θM, n_MC = use_dc ? size(h.helpers_sites[1].ζsM.du) : size(h.helpers_sites[1].ζsM)
    @assert size(rnormPM.M[1]) == (n_θM, n_MC)
    ϕqPc = intϕqP(ϕqP) 
    ϕqIc = intϕqI(ϕqI)
    pbm_covar_indices = !isnothing(pbm_covar_indices) && isempty(pbm_covar_indices) ? nothing : pbm_covar_indices
    #
    set_ζsP!(h.ζsP, h.logσ_ζP, loss_mode, approx, rnormPM.P, ϕqPc, cor_ends.P, h.sample_buffers) # n_P * n_MC
    # (n_M x n_sit)  or (n_M x n_MC x n_sit)    
    ϕms_buffer_key = isnothing(pbm_covar_indices) ? :ϕms : :ϕms_mcs
    #g_apply!(h[ϕms_buffer_key], ϕg, xM, h.ζsP, pbm_covar_indices, g, h.xMP, is_testmode, h.ϕms_mcs2D_buffer) 
    g_apply!(h[ϕms_buffer_key], ϕg, xM, h.ζsP, pbm_covar_indices, g, h.xMP, is_testmode) 
    exp_ladJacTP = transformζ!(h.θsP, transP, h.ζsP)  # return value captures ladJacT
    # so that one can provide its gradient to the pullback
    ϕm_it = eachslice(h[ϕms_buffer_key]; dims = ndims(h[ϕms_buffer_key]))
    template = ϕqI # only important for gradient
    θsP = h.θsP
    keys_nljoint_args_indi_val = Val(keys(nljoint_args_inds))
    # closure with approx, transM, and kwargs
    function compute_nelboi_z_cl!(hi, rnormM, nljoint_args_indi_tup, ϕm) 
        #nljoint_args_indi = NamedTuple{keys_nljoint_args_indi}(nljoint_args_indi_tup)
        nljoint_args_indi = name_tuple(nljoint_args_indi_tup, keys_nljoint_args_indi_val)
        compute_nelboi_z!(hi, loss_mode, approx, rnormM, nljoint_args_indi, ϕm, 
        ϕqIc, θsP, cor_ends.M, transM, nljoint_args_fix) 
    end
    #res_site = map(compute_nelboi_z!, loss_mode, h.helpers_sites, rnormPM.M, i_sites_train, ϕm_it)
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

@inline function set_ζsP!(
    ζsP, logσ_ζP, ::NegElbo, approx, rnormP, ϕqPc, cor_endsP, sample_buffers) 
    sample_ζsP!(ζsP, logσ_ζP, approx, rnormP, ϕqPc, cor_endsP, sample_buffers) # n_P * n_MC    
end
@inline function set_ζsP!(
    ζsP, logσ_ζP::AbstractMatrix{T}, ::NegLogDensity, approx, rnormP, ϕqPc, cor_endsP, sample_buffers) where T
    ζsP .= CA.getdata(view(ϕqPc,Val(:μζP)))
    logσ_ζP .= zero(T) # omit loss component of entropy of distribution
end

name_tuple(tup, ::Val{KEYS}) where KEYS = NamedTuple{KEYS}(tup)


function compute_nelboi_z!(hi, loss_mode::AbstractElboLossMode, approx::AbstractHVIApproximation,
    rnormM, nljoint_args_indi, ϕm::AbstractArray, ϕqIc::AbstractArray, θsP, 
    cor_endsM, transM, nljoint_args_fix, ) 
    # DM is 1 (vector) if no population covariates - constant across MC
    #       2 (Matrix) if g is driven by population covariates - differs across MC
    # on update -> sync corresponding function within grad_neg_elbo_sites
    if hi.ζsM isa PAT.DiffCache
        # Gradiend called twice because of different dimensions need to pick the Dual
        template = eltype(ϕqIc) <: ForwardDiff.Dual ? ϕqIc : θsP
        hi = map_leaves_nt(x -> PAT.get_tmp(x, template), hi)
    end
    #ζsM, logσ_ζM, rnorm, ϕqc::AbstractVector{T}, ϕm::AbstractMatrix, buffer_nθM::AbstractVector
    set_ζsM!(hi.ζsM, hi.logσ_ζM, loss_mode, approx, rnormM, ϕqIc, ϕm, cor_endsM, hi.sample_buffers)
    exp_ladJacTM = transformζ!(hi.θsM, transM, hi.ζsM)  # return value captures ladJacT
    # first component needs to be the full elbo
    exp_nL = exp_nLi!(hi.y, θsP, hi.θsM, nljoint_args_indi, nljoint_args_fix, hi)[1]
    nelboi_z = exp_nL - exp_ladJacTM - sum(hi.logσ_ζM)
end

@inline function set_ζsM!(ζsM, logσ_ζM, ::NegElbo, approx, rnormM, ϕqIc, ϕm, cor_endsM, sample_buffers)
    sample_ζsM!(ζsM, logσ_ζM, approx, rnormM, ϕqIc, ϕm, cor_endsM, sample_buffers)
end
@inline function set_ζsM!(ζsM, logσ_ζM::AbstractArray{T}, ::NegLogDensity, approx, rnormM, ϕqIc, ϕm, cor_endsM, sample_buffers) where T
    error("implement set_ζsM! for NegLogDensity: care for n_θM")
    # TODO infer n_θM
    ζMm .= view(ϕm, 1:n_θM, :)    # μζM in first n_θM entries
    logσ_ζM .= zero(T)
    nothing
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
    n_θP, n_θM, n_site, n_MC, n_cov, n_covP, n_M, cor_ends, n_obs,
    use_diff_cache::Val{use_dc} = Val(true),
    diffchunk::ForwardDiff.Chunk{chunk} = ForwardDiff.Chunk(8),
    ) where {TG, TF, use_dc, chunk}
    his = Tuple((;
        ζsM = Matrix{TF}(undef, n_θM, n_MC),
        θsM = Matrix{TF}(undef, n_θM, n_MC),
        logσ_ζM = Vector{TF}(undef, n_θM),
        y = Matrix{TF}(undef, n_obs, n_MC),
        # TODO move buffers to Channel object
        y_buffer = Matrix{TF}(undef, n_obs, n_MC),
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
        #diffchunk,
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
    n_obs = size(hi.y.du, 1)
    @assert size(hi.ζsM.du) == (n_θM, n_MC)
    @assert size(hi.θsM.du) == (n_θM, n_MC)
    @assert size(hi.y.du) == (n_obs, n_MC)
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
    ϕms_mcs2D_buffer = nothing, # only required for pbm_covar_indices not nothing
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
    #stack the n_site * n_MC and copy back to array afterwards, if view points to other
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
            # for each site i there are n_MC columns, j, with the same xM
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

function g_apply!(ϕm::AbstractMatrix{TF}, ϕg::AbstractVector{TG}, xM::AbstractMatrix{TG}, 
    ζP::AbstractVector{TF}, pbm_covar_indices::AbstractVector{<:Number}, 
    g::AbstractModelApplicator,
    xMP::AbstractMatrix,
    is_testmode::Bool,
    ϕms_mcs2D_buffer = reshape(ϕm, size(ϕm,1), :) # allocates 48 bytes for an escaping view, preallocate 
    ) where {TG, TF}
    update_xMP!(xMP, xM, ζP, pbm_covar_indices) # ζP vector version
    @assert size(xMP,2) == size(ϕm,2) # same number of sites
    apply_model!(ϕm, g, xMP, ϕg; is_testmode)
    return nothing
end
function update_xMP!(xMP::AbstractMatrix{TG}, 
    xM::AbstractMatrix{TG}, ζP::AbstractVector{TF}, pbm_covar_indices::AbstractVector{<:Number}
    ) where {TG, TF}
    (n_θP,) = size(ζP)
    n_cov, n_site = size(xM)
    n_covP = length(pbm_covar_indices)
    @assert size(xMP) == ((n_cov + n_covP) , n_site)
    @inbounds for i in 1:n_site
        # Copy xM block
        for k in 1:n_cov
            xMP[k, i] = xM[k, i]
        end
        # Fill pbm covariates block
        for (k, idx) in enumerate(pbm_covar_indices)
            xMP[n_cov + k, i] = (TG === TF) ? ζP[idx] : convert(TG, ζP[idx])
        end
    end
end



function transformζ!(θs, trans, ζs::AbstractArray{TF}) where TF
    n_MC = size(ζs,2)
    # θs1 = exp.(ζs)
    # ladJacT1 = sum(ζs) / n_MC
    _, ladJacT_sum = with_logabsdet_jacobian_stacked!(θs, trans, ζs)  # return value captures ladJacT
    ladJacT = ladJacT_sum / n_MC
end

"""
compute the expected value of the negative log joint density of observations
and parameters
"""
function exp_nLi!(
    y_pred::AbstractMatrix, 
    θsP::AbstractMatrix,
    θsM::AbstractMatrix,
    nljoint_args_indi::NamedTuple,
    nljoint_args_fix::NamedTuple,
    hi::NamedTuple,
    # f, py,
    # priorsP, priorsM,
    # penalty_computer = ZeroPenaltyComputer(),
    # is_omit_priors,
    # zero_prior_logdensity,
    # intθP, intθMs,
    # ranef::AbstractRandomEffectsComputer,
    # frac_cluster_all,
) 
    n_obs, n_MC = size(y_pred)
    (;xP, y_o, y_unc, i_indiv_train) = nljoint_args_indi
    @assert size(y_o) == size(y_unc) == (n_obs,)
    @assert size(θsP,2) == size(θsM,2) == n_MC
    (;f!,fLogLik) = nljoint_args_fix
    #θsPc = CA.ComponentMatrix(θsP, axθP, CA.FlatAxis())
    #θsMc = CA.ComponentMatrix(θsM, axθM, CA.FlatAxis())
    apply_model!(y_pred, f!, θsP, θsM, xP)
    nL_reg = sum(y_pred) / n_MC
    nL = fLogLik(y_pred, y_o, y_unc, hi.y_buffer) / n_MC
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

function fLogLik_dummy(y_pred, y_o, y_unc, y_buffer=similar(y_pred))
    sum(y_pred)
end

