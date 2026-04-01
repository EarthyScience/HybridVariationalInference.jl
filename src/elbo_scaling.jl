# Similar to MeanHVIApproximationMat 
# but ML model predicts a scaling factor for a group of variance parameters
# ϕq element logσ2_ζM_offsets contains a vector of log-offsets, i.e. multipliers, 
#   for each block of ML scaled parameters
#   the log-offset for the last entry in each block is stored in approx.logσ2_ζM_base

function sample_ζresid_norm(approx::AbstractMeanScalingHVIApproximation,
    i_sites,
    zP::AbstractMatrix, zMs::AbstractMatrix, 
    ϕm::TM, ϕq::AbstractVector{T};
    int_ϕq=get_concrete(ComponentArrayInterpreter(ϕq)),
    cor_ends
) where {T,TM<:AbstractMatrix{T}}
    ϕuncc = ϕqc = int_ϕq(CA.getdata(ϕq))
    # add 0 as last logσ2_par_offset-par in block
    logσ2_par_offsets_before_end = ϕqc[Val(:logσ2_ζM_offsets)]
    # insert zeros at the end of each block of parameters
    logσ2_par_offsets = insert_zeros(logσ2_par_offsets_before_end, approx.scalingblocks_ends)::typeof(logσ2_par_offsets_before_end) 
    b_ends = ChainRulesCore.ignore_derivatives(approx.scalingblocks_ends)
    length_scale_blocks = vcat(first(b_ends), diff(b_ends))
    n_scale_blocks = length(b_ends)
    n_par = size(ϕm,1) - n_scale_blocks
    ζMs = ϕm[1:n_par,:]
    logσ2_sites = ϕm[(n_par+1):end,:]
    ζP = ϕqc[Val(:μP)]
    n_θP, n_θMs, (n_θM, n_batch) = length(ζP), length(ζMs), size(ζMs)
    # do not create a UpperTriangular Matrix of an AbgeneraGÜUArray in transformU_cholesky1
    ρsP = isempty(ϕuncc[Val(:ρsP)]) ? similar(ϕuncc[Val(:ρsP)]) : ϕuncc[Val(:ρsP)] # required by zygote
    UP = transformU_block_cholesky1(ρsP, cor_ends.P)
    ρsM = isempty(ϕuncc[Val(:ρsM)]) ? similar(ϕuncc[Val(:ρsM)]) : ϕuncc[Val(:ρsM)] # required by zygote
    # cholesky factor of the correlation: diag(UM' * UM) .== 1
    # coefficients ρsM can be larger than 1, still yielding correlations <1 in UM' * UM
    UM = transformU_block_cholesky1(ρsM, cor_ends.M)
    #
    logσ2_site_offsets = repeat_rows_by_counts(logσ2_sites, length_scale_blocks)
    logσ2_ζM_bases = reduce(vcat, fill.(approx.logσ2_ζM_base, length_scale_blocks))
    logσ2_ζMs = logσ2_ζM_bases .+ logσ2_par_offsets .+ logσ2_site_offsets
    #
    logσ2_ζP = vec(CA.getdata(ϕuncc[Val(:logσ2_ζP)]))
    # CUDA cannot multiply BlockDiagonal * Diagonal, construct already those blocks
    σMs = exp.(logσ2_ζMs ./ T(2))
    σP = exp.(logσ2_ζP ./ T(2))
    # BlockDiagonal does work with CUDA, but not with combination of Zygote and CUDA
    # need to construct full matrix for CUDA
    Uσ, diagUσ = _compute_choleskyfactor(UP, UM, σP, σMs, n_batch) # inferred only BlockDiagonal
    #diagUσ = diag(Uσ)::typeof(σP)   # elements of the diagonal: standard deviations
    n_MC = size(zP, 1) 
    # is this multiplication efficient if Uσ is not concrete but only sumtype BlockDiagonal?
    urandn = hcat(zP, zMs)
    ζ_resids_parfirst = (Uσ' * urandn') #::typeof(urandn) # n_par x n_MC
    #ζ_resids_parfirst = (urandn * Uσ)' #::typeof(urandn) # n_par x n_MC
    #ζ_resids_parfirst = urandn' * Uσ # n_MC x n_par
    # need to handle empty(ζP) explicitly, otherwise Zygote tries to take gradient
    ζP_resids = isempty(ζP) ? ζ_resids_parfirst[1:0, :] : ζ_resids_parfirst[1:n_θP, :]
    ζMs_parfirst_resids = reshape(ζ_resids_parfirst[(n_θP+1):end, :], n_θM, n_batch, n_MC)
    ζP_resids, ζMs_parfirst_resids, diagUσ
    # #map(std, eachcol(ζ_resids_parfirst[:, 3:8]))
    # ζ_resid = transpose_mPMs_sitefirst(ζ_resids_parfirst; intm_PMs_parfirst)
    # #map(std, eachcol(ζ_resid[:, 3:8])) # all ~ 0.1 in sample_ζresid_norm cpu
    # #map(std, eachcol(ζ_resid[:, 2 + n_batch .+ (-1:5)])) # all ~ 100, except first two
    # # returns AbstractGPUuArrays to either continue on GPU or need to transfer to CPU
    # ζ_resid, diagUσ
end

# repeat rows of a matrix by per-row counts, non-mutating (Zygote-friendly)
function repeat_rows_by_counts(A::AbstractMatrix, counts::AbstractVector{<:Integer})
    @assert length(counts) == size(A,1) "Need to provide a count for each row ($counts, $(size(A,1)))."
    if isempty(A)
        return similar(A, 0, size(A,2))
    end
    idx = vcat((fill(i, counts[i]) for i in axes(counts,1))...)
    return A[idx, :]
end



