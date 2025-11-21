# Re-Implementation of MeanHVIApproximationVec where random numbers are
# generated for each block in the correlation matrix separately.
#
# Unfortunately, having many smaller matrix multiplications, the derivative
# computation is slower than with 
# one big matrix multiplication of the big sparse blocked covariance matrix.
# As benchmarked 251121 for 20 sites in test_sample_zeta.
# Both on cpu and gpu
#
# For a large number of sites in a forward run across all sites
# this implementaion is faster on gpu, but slower than the Vec
# implementation on CPU (that uses triangular and blocked matrices)
# 
# Hence, currently, rather use the MeanHVIApproximationVec CPU implementation.
#   for gradient on few sites (1.6ms cpu vs 5.4ms gpu) 
#   and forward runs for many sites (760mus cpu vs 439ms gpu unit!)

function sample_ζresid_norm(app::MeanHVIApproximation, 
    zP::AbstractMatrix, zMs::AbstractMatrix, ζP::TP, ζMs::TM,
    ϕq::AbstractVector;
    int_ϕq=get_concrete(ComponentArrayInterpreter(ϕq)),
    cor_ends
) where {T,TP<:AbstractVector{T},TM<:AbstractMatrix{T}}
    ϕunc = CA.getdata(ϕq)
    ϕuncc = int_ϕq(ϕunc)
    n_θP, n_θMs, (n_θM, n_batch), n_MC = length(ζP), length(ζMs), size(ζMs), size(zP,1)
    # do not create a UpperTriangular Matrix of an AbstractGÜUArray in transformU_cholesky1
    ρsP = isempty(ϕuncc[Val(:ρsP)]) ? similar(ϕuncc[Val(:ρsP)]) : ϕuncc[Val(:ρsP)] # required by zygote
    UPs, rangesP = transformU_blocks_cholesky1(ρsP, cor_ends.P) 
    ρsM = isempty(ϕuncc[Val(:ρsM)]) ? similar(ϕuncc[Val(:ρsM)]) : ϕuncc[Val(:ρsM)] # required by zygote
    # cholesky factor of the correlation: diag(UM' * UM) .== 1
    # coefficients ρsM can be larger than 1, still yielding correlations <1 in UM' * UM
    UMs, rangesM = transformU_blocks_cholesky1(ρsM, cor_ends.M)
    cf = ϕuncc[Val(:coef_logσ2_ζMs)]
    logσ2_logMs = vec(cf[1, :] .+ cf[2, :] .* ζMs)
    logσ2_ζP = vec(CA.getdata(ϕuncc[Val(:logσ2_ζP)]))
    # CUDA cannot multiply BlockDiagonal * Diagonal, construct already those blocks
    σMs = reshape(exp.(logσ2_logMs ./ 2), n_θM, :)
    σP = exp.(logσ2_ζP ./ 2)
    # create random numbers from U diag(σ) z =  U (σ .* z)
    #  for each block separately
    #Ui, ri = first(zip(UPs, rangesP))
    ζP1 = ChainRulesCore.@ignore_derivatives similar(CA.getdata(ϕunc), size(zP'))
    #@benchmark typeof(zP .* ϕq[1:1])  # more allocations and more time
    local ζP_resids = if isempty(ζP)
        ζP1 # provide init of correct empty matrix type
    else
        mapreduce(vcat, UPs, rangesP) do Ui, ri
            Diagonal(σP[ri]) * Ui' * zP[:,ri]' 
        end
    end::typeof(ζP1)
    #
    #Ui, ri = first(zip(UMs, rangesM))
    zMs_subjects = reshape(zMs, (n_MC, n_θM, n_batch))
    #σM, zM = first(zip(eachcol(σMs), eachslice(zMs_subjects; dims=3)))
    cat3 = (x,y) -> cat(x,y,dims=3)
    # map across subjects (n_batch)
    #ζMs_vec = map(eachcol(σMs), eachslice(zMs_subjects; dims=3)) do σM, zM
    ζMs1 = ChainRulesCore.@ignore_derivatives similar(ϕunc, size(zMs_subjects[:,:,1]'))
    fBlock = let UMs = UMs, rangesM = collect(rangesM) # without collect, type unstable
        function fBlock_inner(σM, zM)::typeof(ζMs1)
            # generator does not work with Zygote
            # vcat(...) is not faster than mapreduce
            # g = map( UMs, rangesM) do Ui, ri  # n_θM, n_MC
            #     #(zM[:,ri]' * Ui * diagm(σM[ri]))' 
            #     diagm(σM[ri]) * Ui' * zM[:, ri]'
            # end
            # vcat(g...)
            mapreduce(vcat, UMs, rangesM) do Ui, ri  # n_θM, n_MC
                # (zM[:,ri]' * (Ui * Diagonal(σM[ri])))' 
                # Diagonal is faster than diagm  and works with GPU 
                Diagonal(σM[ri]) * Ui' * zM[:, ri]'   # transpose to not return adjoint
            end
        end
    end
    #ζMs1 = ChainRulesCore.@ignore_derivatives fBlock(σMs[:,1], zMs_subjects[:,:,1])
    #TMR = ChainRulesCore.@ignore_derivatives typeof(similar(ϕq, size(zMs)))
    #TB = Base.infer_return_type(fBlock, (typeof(σMs[:,1]), typeof(zMs_subjects[:,:,1])))
    #ζMs_vec = map(fBlock, eachcol(σMs[:,2:end]), eachslice(zMs_subjects[:,:,2:end]; dims=3), init = ζMs1) 
    #ζMs_vec = map((σM, zM) -> fBlock(σM, zM)::TMR, eachcol(σMs), eachslice(zMs_subjects; dims=3)) 
    ζMs_vec = map((σM, zM) -> fBlock(σM, zM), eachcol(σMs), eachslice(zMs_subjects; dims=3)) 
    #zM = zMs_subjects[:,:,1]
    #ζMs_vec = [fBlock(σMs[:,i], zM) for i in axes(σMs, 2)] 
    #ζMs_vec = [fBlock(σMs[:,1], zM)]
    #ζMs_vec = [fBlock(σMs[:,1], zM) for i in 1:n_batch]
    #ζMs_vec = [fBlock(σMs[:,1], zM) for i in axes(σMs, 2)]
    #ζMs_vec = [fBlock(σM, zM) for  (σM, zM) in zip(eachcol(σMs), eachslice(zMs_subjects; dims=3))] 
    # concatenate so that n_MC is last dimension
    local ζMs_parfirst_resids = stack(ζMs_vec; dims = 2 ) # n_θM, n_batch, n_MC
    #size(ζMs_parfirst_resids)

    # std(ζMs_parfirst_resids[1,1,:])
    # std(ζMs_parfirst_resids[1,end,:])
    # σzMs_stacked = reshape(σzMs, (n_θM, n_batch * n_MC))
    # ζMs_resids_stacked = mapreduce(vcat, UMs, rangesM) do Ui, ri
    #     #Ui * σzMs_stacked[ri, :]
    #     Uσ = Ui * σMs[ri,:]
    #     Uσ' * zMs[:,ri]' 
    #     diagm(σMs[ri,:]) * Ui' * zMs[:,ri]' 
    # end
    # ζMs_parfirst_resids = reshape(ζMs_resids_stacked, n_θM, n_batch, n_MC)
    #
    diagUσ = vcat(σP, vec(σMs))
    ζP_resids, ζMs_parfirst_resids, diagUσ
end
