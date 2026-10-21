
function grad_pointloss_sites(
    pointloss_helpers::NamedTuple,      # tuple of preallocated arrays
    grad_pointloss_helpers::NamedTuple,  # tuple of preallocated arrays pullback closures
    ϕ::NamedTuple,
    sample_args::NamedTuple,
    indiv_args::NamedTuple,
    nljoint_args_fix::NamedTuple;
    # approx::AbstractHVIApproximation,
    # rnormPM::NamedTuple,          # tuple of random numbers
    # ϕg::AbstractVector{TG}, ϕqP::AbstractVector{TF}, ϕqI::AbstractVector{TF}, g, 
    # pbm_covar_indices::Union{Nothing,AbstractVector{<:Number}}, 
    # args...;
    # nljoint_args_inds,     # indices of sites in training set
    # intϕqP, intϕqI,
    # xM,
    # cor_ends,
    # transP::Stacked, transM::Stacked,
    # is_testmode, 
    n_workers = 1, # TODO compute from executor and Distributed.nworkers and nthreads,...
    executor::Transducers.Executor = Transducers.SequentialEx(),
) 
    h = pointloss_helpers # preallocated μζP, dμζP, ζsP, ϕms, xMP, dxMP
    (;ϕg, ϕqP, ϕqI) = ϕ
    TF = eltype(ϕqP)
    (;approx, g, is_testmode, pbm_covar_indices, intϕqP, intϕqI, cor_ends, transP, transM) = sample_args
    @assert keys(indiv_args)[1] == :xM
    xM = indiv_args.xM; r2end = 2:length(indiv_args)
    nljoint_args_inds = NamedTuple{keys(indiv_args)[r2end]}(values(indiv_args)[r2end])
    use_ϕm_matrix = isnothing(pbm_covar_indices)
    ϕqPc = intϕqP(ϕqP) 
    ϕqIc = intϕqI(ϕqI)
    #ϕm_buffer_key = use_ϕm_matrix ? :ϕms : :ϕms_mcs
    #h_ϕms = h[ϕm_buffer_key]
    h_ϕms = isnothing(pbm_covar_indices) ? view(h.ϕmsa,:,1,:) : h.ϕmsa
    check_pointloss_helpers(h, xM, pbm_covar_indices)
    n_cov, n_site = size(xM)
    n_θP, n_MC = size(h.ζPm)
    @assert n_MC == 1
    ax_θP = get_axis_θP(nljoint_args_fix.f!)
    ax_θM = get_axis_θM(nljoint_args_fix.f!)
    n_θM = axis_length(ax_θM)
    n_M = get_numberof_MLinputs(approx, 1:n_θM)
    nljoint_args_indi = NamedTuple{keys(nljoint_args_inds)}(
        first(zip_eachlastdims(nljoint_args_inds)))
    gradh = !isempty(grad_pointloss_helpers) ? grad_pointloss_helpers : 
    Test.@inferred prepare_gradpointloss_helpers(
    #prepare_gradpointloss_helpers(
        ϕqIc, selectdim(h_ϕms, ndims(h_ϕms), 1), h.ζPm, ϕg, ϕqPc, approx; 
        pbm_covar_indices, n_workers,
        h, nljoint_args_indi,
        n_site, n_M, n_cov, cor_ends, transP, transM, nljoint_args_fix, )
    #
    h.ζPm .= CA.getdata(view(ϕqPc,Val(:μζP)))
    g_apply!(h_ϕms, ϕg, xM, h.ζPm, pbm_covar_indices, g, h.xMP, is_testmode) 
    ladJacTP = transformζ!(h.θPm, transP, h.ζPm)  # return value captures ladJacT
    #
    # parallel ForwardDiffGradient through forwarddiff_grad_pointlossi_z!
    cl = ForwardDiffGradPointlossiZCl(
        approx, ϕqIc, h.θPm, gradh.dϕmvecs, gradh.hw_channel, cor_ends.M, transM,
        Val(keys(nljoint_args_inds)), nljoint_args_fix)    
    ϕm_it = eachslice(h.ϕmsa; dims = ndims(h.ϕmsa))
    init = (;
        dϕqI = zero(TF) .* SA.SVector{axis_length(CA.getaxes(ϕqIc)[1])}(ϕqIc),
        dθsP = zero(TF) .* SA.SVector{axis_length(ax_θP)*1}(h.θPm), )
    res = Folds.mapreduce(cl, make_tuple_reducer(+), 
        zip(h.helpers_sites, zip_eachlastdims(nljoint_args_inds), ϕm_it, 1:n_site),
        executor; init)
    #
    # copy SharedArray into preallocated plain array for ∂elbo_∂θP
    ∂elbo_∂ϕqm = gradh.∂elbo_∂ϕqm 
    copyto!(∂elbo_∂ϕqm, gradh.dϕmvecs) # copy from shared array
    # copy StaticVector into preallocated ComponentVector
    ∂elbo_∂θP = gradh.∂elbo_∂θP
    copyto!(∂elbo_∂θP, res.dθsP)
    copyto!(gradh.dϕqIc, res.dϕqI)
    # other gradients are one, calculated from formula
    gradh.∂elbo_∂logσ_ζP .= -one(TF)  
    ∂elbo_∂ladJacTP = -one(TF)
    #
    # pullback gradients of ϕqm -> gradh.dϕg and gradh.dζsP
    gradh.pullback_g_apply!(
        gradh.dϕg, gradh.∂elbo_∂ϕm_∂ζP, h_ϕms, ∂elbo_∂ϕqm, 
        ϕg, xM, h.ζPm, pbm_covar_indices, g, is_testmode)
    #
    # pullback gradients of ∂elbo_∂θP to ∂elbo_∂θP_∂ζP
    ∂elbo_∂θP_∂ζP = gradh.∂elbo_∂θP_∂ζP
    gradh.pullback_cl_transformζsP!(
        ∂elbo_∂θP_∂ζP, 
        ∂elbo_∂θP,
        ∂elbo_∂ladJacTP,
        h.θPm,
        transP, 
        h.ζPm,
        )
    #
    # pullback gradients of ∂elbo_∂ζP, ∂elbo_∂ϕm_∂ζP to  dϕqP
    #   for μζP (first n_θM entries) its the sum of the woggles, others are zero
    gradh.dϕqPc[1:n_θM] .= ∂elbo_∂θP_∂ζP + gradh.∂elbo_∂ϕm_∂ζP
    gradh.dϕqPc[(n_θM+1):end] .= zero(TF)
    (;dϕqPc = gradh.dϕqPc, dϕqIc = gradh.dϕqIc, dϕg = gradh.dϕg), gradh
end

"""
Callable to make deliver arguments that do not differ by individual to Foldl.mapreduce.
"""
struct ForwardDiffGradPointlossiZCl{KEYS, TA, TI, Tθ, TD, THWC, TC, TM, TAF}
    approx::TA
    ϕqIc::TI
    θPm::Tθ
    dϕmvecs::TD
    hw_channel::THWC
    corendsM::TC
    transM::TM
    # need to store KEYS in type parameter otherwise allocation in NamedTuple(keys)
    keys_nljoint_args_indi::Val{KEYS}
    nljoint_args_fix::TAF
end
function (f::ForwardDiffGradPointlossiZCl{KEYS})(tup) where KEYS
    local hi, nljoint_args_indi_tup, ϕm, i
    hi, nljoint_args_indi_tup, ϕm, i = tup
    #nljoint_args_indi = NamedTuple{f.keys_nljoint_args_indi}(nljoint_args_indi_tup)
    nljoint_args_indi = NamedTuple{KEYS}(nljoint_args_indi_tup)
    forwarddiff_grad_pointloss_z!(hi, f.approx, nljoint_args_indi, ϕm, i,
        f.ϕqIc, f.θPm, f.dϕmvecs, f.hw_channel, f.corendsM, f.transM, 
        f.nljoint_args_fix,
        nothing,
        #true,
        )
end


# ϕm vector version
# TOOD pass closure object rather than many parameters
function forwarddiff_grad_pointloss_z!(hi, approx::AbstractHVIApproximation, 
    nljoint_args_indi, ϕm::AbstractMatrix{TF}, i::Integer, 
    ϕqIc::AbstractVector, θP::AbstractMatrix, dϕmvecs, hw_channel::Channel, cor_endsM, transM::Stacked, 
    nljoint_args_fix,
    omit_gradient=nothing) where {TF}
    # need to calls to ForwardDiff.gradient! to avoid reshape
    # group arrays of same dimensionality (here grad_ϕm and grad_ϕqI) into a Vcat
    ax_ϕqI = CA.getaxes(ϕqIc)[1]
    ax_ϕP = get_axis_θP(nljoint_args_fix.f!)
    with_channel_element(hw_channel) do hwi_tup
        hwi, hwi_dc = hwi_tup
        #grad_conf = hwi.grad_conf
        pointlossi_z_vec = make_pointlossz_lazy_cls(hi, nljoint_args_indi, 
            cor_endsM, transM, nljoint_args_fix,
            ax_ϕqI, 
            )
        grad_θP, grad_ϕm, grad_ϕqI = if isnothing(omit_gradient)
            local θP_ϕm_ϕqI
            # transform ϕqI to a 1-column matrix to vcat it to θP and ϕm
            ϕqI = if strides(CA.getdata(ϕqIc))[1] == 1
                # to avoid allocations with reshape
                Base.ReshapedArray(CA.getdata(ϕqIc), (length(CA.getdata(ϕqIc)), 1), ())                
            else
                reshape(CA.getdata(ϕqIc), :, 1)
            end
            () -> begin
                tmpf(ϕqI) = mapreduce(identity, +, view(ϕqI,:,1))
                @allocated tmpf(ϕqI) # zero
            end
            θP_ϕm_ϕqI = Vcat(θP, ϕm, ϕqI)    
            tmp = similar(θP_ϕm_ϕqI)
            ForwardDiff.gradient!(hwi.grad_θsP_ϕm_ϕqI, pointlossi_z_vec, θP_ϕm_ϕqI, hwi.grad_conf_point)
            hwi.grad_θsP_ϕm_ϕqI.args[1], hwi.grad_θsP_ϕm_ϕqI.args[2], hwi.grad_θsP_ϕm_ϕqI.args[3]
        else
            θP, ϕm, CA.getdata(ϕqIc)
        end
        copyto!(view(dϕmvecs, :, i), grad_ϕm) 
        # returning SVector helps avoiding allocations during reduce
        local res
        res = (;
            dϕqI = SA.SVector{axis_length(CA.getaxes(ϕqIc)[1])}(grad_ϕqI),
            dθsP = SA.SVector{axis_length(ax_ϕP)*1}(grad_θP),
        )
    end
end

function make_pointlossz_lazy_cls(hi, nljoint_args_indi, 
    cor_endsM, transM, nljoint_args_fix,
    ax_ϕqI::CA.Axis, 
    )
    # mat, is a Vcat(θP, ϕm, ϕqI)
    function pointlossi_cl_vec(mat::Vcat) 
        local θP_dual, ϕm_dual, ϕqIc_dual
        θP_dual = mat.args[1]
        ϕm_dual = mat.args[2]
        #ϕqIc_dual = CA.ComponentVector(vec(mat.args[3]), ax_ϕqI)
        ϕqIc_dual = CA.ComponentVector(view(mat.args[3],:,1), ax_ϕqI)
        compute_pointlossi_z!(
            hi, nljoint_args_indi,
            ϕm_dual, ϕqIc_dual, θP_dual, 
            cor_endsM, transM, nljoint_args_fix,
        )[1]
    end
    pointlossi_cl_vec
end


function prepare_gradpointloss_helpers(
    ϕqIc, ϕm::AbstractArray{TF,ND}, θsP::AbstractMatrix,
    ϕg::AbstractVector{TG}, ϕqPc::AbstractVector{TF},
    approx::AbstractHVIApproximation; 
    pbm_covar_indices, n_workers,
    h, nljoint_args_indi,
    n_site, n_M, n_cov, cor_ends,
    transP::Stacked, transM::Stacked,
    nljoint_args_fix,
    diffchunk::ForwardDiff.Chunk{chunk} = ForwardDiff.Chunk(8),
    ) where {TG, TF, ND, chunk}
    hi1 = h.helpers_sites[1]
    ax_θP = get_axis_θP(nljoint_args_fix.f!)
    ax_θM = get_axis_θM(nljoint_args_fix.f!)
    # TODO get n_site, n_cov from 
    n_covP =  isnothing(pbm_covar_indices) ? 0 : length(pbm_covar_indices)
    n_ϕmvec = length(ϕm) #use_ϕm_matrix ? n_M : n_M * n_MC
    (n_θP,n_MC) = size(θsP)
    @assert n_MC == 1
    @assert n_θP == axis_length(ax_θP)
    # n_M = size(ϕm,1)
    # To sync across procs/threads, use a Channel 
    # https://juliafolds2.github.io/OhMyThreads.jl/stable/literate/tls/tls/#The-safe-way:-Channel
    #    
    @assert length(nljoint_args_indi.xP) > 1
    # pointlossi_z = make_pointlossiz_cl(hi1, approx, rnormMi, nljoint_args_indi, CA.getaxes(cv_grad), 
    #     cor_ends.M, transM, nljoint_args_fix) 
    hw_dcs_raw = (;
            ϕm_dc = similar(ϕm),   
            # callocation on using ComponentArray (_restructure), need to use plain vector
            ϕqI_dc = similar(CA.getdata(ϕqIc)), 
            θP_dc = similar(θsP),
    )
    # make sure, each worker gets its own copy, use similar
    # TODO remove hw_dcs, not needed because Vcat of all vectors
    hw_dcs = (hw_dcs_raw) -> map_leaves_nt(x -> PAT.DiffCache(similar(x), chunk), hw_dcs_raw) 
    pointlossi_z_vec = make_pointlossz_lazy_cls(hi1, nljoint_args_indi, 
        cor_ends.M, transM, nljoint_args_fix,
        CA.getaxes(ϕqIc)[1], 
        ) 
    local grad_θsP_ϕm_ϕqI1, diffchunks_point
    ϕqIm = reshape(CA.getdata(ϕqIc),:,1)
    ϕm_mat = (ND == 2) ? ϕm : reshape(ϕm, :, 1)
    grad_θsP_ϕm_ϕqI1 = grad_θsP_ϕm_ϕqI(θsP, ϕm_mat, ϕqIm)
    # n_M is currently not static, hence diffchunks_vec and gradientConfig not inferred
    #   however, n_θM is a lower bound, so use this one to set a minimum length
    #   chunk size might be not as large as possible, but safely not larger than paramvector
    diffchunks_point = ForwardDiff.Chunk(min(chunk, 
        #axis_length(ax_θP) + n_M + axis_length(CA.getaxes(ϕqIc)[1])))
        axis_length(ax_θP) + axis_length(ax_θM) + axis_length(CA.getaxes(ϕqIc)[1])))
    function grad_conf_point() 
        cfg_vec = ForwardDiff.GradientConfig(pointlossi_z_vec, grad_θsP_ϕm_ϕqI1, diffchunks_point)
        update_duals_Vcat3(cfg_vec, grad_θsP_ϕm_ϕqI1)
    end
    get_helpers_worker = () -> begin
        (;
            grad_conf_point = grad_conf_point(), 
            grad_θsP_ϕm_ϕqI = grad_θsP_ϕm_ϕqI(θsP, ϕm_mat, ϕqIm),
        )
    end
    h1 = get_helpers_worker()
    h1_dc = hw_dcs(hw_dcs_raw)
    h1_tup = (h1, h1_dc)
    # put tuples of h1 an d1_dc into the channel
    hw_channel = Channel{typeof(h1_tup)}(n_workers) # workers + parallel threads
    put!(hw_channel, h1_tup)
    foreach(2:n_workers) do _
        put!(hw_channel, (get_helpers_worker(), hw_dcs(hw_dcs_raw)))
    end 
    (;
        dϕg = similar(ϕg),
        dϕqPc = similar(ϕqPc),
        dϕqIc = similar(ϕqIc),
        ∂elbo_∂θP = similar(θsP),
        ∂elbo_∂ϕm_∂ζP = similar(θsP),
        ∂elbo_∂logσ_ζP = Vector{TF}(undef, size(θsP,1)),
        ∂elbo_∂θP_∂ζP = Matrix{TF}(undef, size(θsP)),
        ∂elbo_∂ϕqm = Array{TF}(undef, size(ϕm)..., n_site),
        hw_channel,
        pullback_cl_transformζsP! = get_pullback_cl_transformζ!(CA.getdata(ϕqPc); n_θ = n_θP, n_MC),
        pullback_g_apply! = get_pullback_g_apply(
            ϕg, ϕqPc; n_θP, n_cov, n_covP, n_MC, n_site, n_M),
        # SharedArrays last because of problems with ReferenceRevision
        dϕmvecs = SharedArrays.SharedArray{TF}(n_ϕmvec, n_site), 
    )
end
# need similar to make a copy for each worker
grad_θsP_ϕm_ϕqI(θP, ϕm, ϕqIc) = Vcat(similar(θP), similar(ϕm), similar(CA.getdata(ϕqIc))) 

