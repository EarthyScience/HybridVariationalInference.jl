function sample_ζresid_norm(app::MeanHVIApproximation, z::AbstractMatrix, ζP::TP, ζMs::TM,
    ϕq::AbstractVector;
    int_ϕq=get_concrete(ComponentArrayInterpreter(ϕq)),
    cor_ends
) where {T,TP<:AbstractVector{T},TM<:AbstractMatrix{T}}
    ϕuncc = int_ϕq(CA.getdata(ϕq))
    n_θP, n_θMs, (n_θM, n_batch) = length(ζP), length(ζMs), size(ζMs)
    # do not create a UpperTriangular Matrix of an AbstractGÜUArray in transformU_cholesky1
    ρsP = isempty(ϕuncc[Val(:ρsP)]) ? similar(ϕuncc[Val(:ρsP)]) : ϕuncc[Val(:ρsP)] # required by zygote
    UPs, rangesP = transformU_blocks_cholesky1(ρsP, cor_ends.P) 
    UP = transformU_block_cholesky1(ρsP, cor_ends.P)
    ρsM, rangesM = isempty(ϕuncc[Val(:ρsM)]) ? similar(ϕuncc[Val(:ρsM)]) : ϕuncc[Val(:ρsM)] # required by zygote
    # cholesky factor of the correlation: diag(UM' * UM) .== 1
    # coefficients ρsM can be larger than 1, still yielding correlations <1 in UM' * UM
    UMs = transformU_block_cholesky1(ρsM, cor_ends.M)
    cf = ϕuncc[Val(:coef_logσ2_ζMs)]
    logσ2_logMs = vec(cf[1, :] .+ cf[2, :] .* ζMs)
    logσ2_ζP = vec(CA.getdata(ϕuncc[Val(:logσ2_ζP)]))
    # CUDA cannot multiply BlockDiagonal * Diagonal, construct already those blocks
    σMs = reshape(exp.(logσ2_logMs ./ 2), n_θM, :)
    σP = exp.(logσ2_ζP ./ 2)
    # create random numbers from U diag(σ) z =  (σ .* z)
    

    # # BlockDiagonal does work with CUDA, but not with combination of Zygote and CUDA
    # # need to construct full matrix for CUDA
    # Uσ, diagUσ = _compute_choleskyfactor(UP, UM, σP, σMs, n_batch) # inferred only BlockDiagonal
    # #diagUσ = diag(Uσ)::typeof(σP)   # elements of the diagonal: standard deviations
    # n_MC = size(urandn, 2) # TODO transform urandn
    # # is this multiplication efficient if Uσ is not concrete but only sumtype BlockDiagonal?
    # ζ_resids_parfirst = (Uσ' * urandn) #::typeof(urandn) # n_par x n_MC
    # #ζ_resids_parfirst = urandn' * Uσ # n_MC x n_par
    # # need to handle empty(ζP) explicitly, otherwise Zygote tries to take gradient
    # ζP_resids = isempty(ζP) ? ζ_resids_parfirst[1:0, :] : ζ_resids_parfirst[1:n_θP, :]
    # ζMs_parfirst_resids = reshape(ζ_resids_parfirst[(n_θP+1):end, :], n_θM, n_batch, n_MC)


    ζP_resids, ζMs_parfirst_resids, diagUσ
    # #map(std, eachcol(ζ_resids_parfirst[:, 3:8]))
    # ζ_resid = transpose_mPMs_sitefirst(ζ_resids_parfirst; intm_PMs_parfirst)
    # #map(std, eachcol(ζ_resid[:, 3:8])) # all ~ 0.1 in sample_ζresid_norm cpu
    # #map(std, eachcol(ζ_resid[:, 2 + n_batch .+ (-1:5)])) # all ~ 100, except first two
    # # returns AbstractGPUuArrays to either continue on GPU or need to transfer to CPU
    # ζ_resid, diagUσ
end