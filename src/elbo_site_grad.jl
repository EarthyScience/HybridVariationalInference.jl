
function grad_neg_elbo_sites(
    elbo_helpers::NamedTuple,      # tuple of preallocated arrays
    grad_elbo_helpers::NamedTuple,  # tuple of preallocated arrays pullback closures
    ϕ::NamedTuple,
    rnormPM::NamedTuple,
    sample_args::NamedTuple,
    indiv_args::NamedTuple,
    nljoint_args_fix::NamedTuple;
    n_MC_val::Val{NMC},     # need to pass by value for type inference on chunks
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
) where NMC
    h = elbo_helpers # preallocated μζP, dμζP, ζsP, ϕms, xMP, dxMP
    (;ϕg, ϕqP, ϕqI) = ϕ
    TF = eltype(ϕqP)
    (;approx, g, is_testmode, pbm_covar_indices, intϕqP, intϕqI, cor_ends, transP, transM) = sample_args
    @assert keys(indiv_args)[1] == :xM
    xM = indiv_args.xM; r2end = 2:length(indiv_args)
    nljoint_args_inds = NamedTuple{keys(indiv_args)[r2end]}(values(indiv_args)[r2end])
    use_ϕm_matrix = isnothing(pbm_covar_indices)
    ϕqPc = intϕqP(ϕqP) 
    ϕqIc = intϕqI(ϕqI)
    ϕm_buffer_key = use_ϕm_matrix ? :ϕms : :ϕms_mcs
    h_ϕm = h[ϕm_buffer_key]
    check_elbo_helpers(h, xM, pbm_covar_indices; n_ϕg = length(ϕg))
    n_cov, n_site = size(xM)
    n_θP, n_MC = size(h.ζsP)
    @assert n_MC == NMC
    ax_θP = get_axis_θP(nljoint_args_fix.f!)
    ax_θM = get_axis_θM(nljoint_args_fix.f!)
    nljoint_args_indi = NamedTuple{keys(nljoint_args_inds)}(
        first(zip_eachlastdims(nljoint_args_inds)))
    gradh = !isempty(grad_elbo_helpers) ? grad_elbo_helpers : 
    #Test.@inferred prepare_gradelbo_helpers(
    prepare_gradelbo_helpers(
        ϕqIc, selectdim(h_ϕm, ndims(h_ϕm), 1), h.ζsP, ϕg, ϕqPc, approx; 
        pbm_covar_indices, n_workers,
        h, rnormMi = rnormPM.M[1], nljoint_args_indi,
        n_site, n_cov, cor_ends, transP, transM, nljoint_args_fix, n_MC_val)
    hw_channel = gradh.hw_channel
    #
    sample_ζsP!(h.ζsP, h.logσ_ζP, approx, rnormPM.P, ϕqPc, cor_ends.P, h.sample_buffers) # n_P * n_MC
    g_apply!(h_ϕm, ϕg, xM, h.ζsP, pbm_covar_indices, g, h.xMP, is_testmode) 
    ladJacTP = transformζ!(h.θsP, transP, h.ζsP)  # return value captures ladJacT
    #
    # parallel ForwardDiffGradient through forwarddiff_grad_nelboi_z!
    # fill the site-independent args of each worker's preallocated input Vcat
    # (θsP = arg[1], ϕqIc = arg[3]); the per-site ϕm (arg[2]) is copied inside
    # forwarddiff_grad_nelboi_z!. Cannot be done in prepare_gradelbo_helpers,
    # because the helpers may be passed as argument (to be reused across calls)
    with_each_channel_element(hw_channel) do hwi_tup
        hwi, _ = hwi_tup
        copyto!(subarray(hwi.θsP_ϕm_ϕqI_vv,Val(1)), h.θsP)
        copyto!(subarray(hwi.θsP_ϕm_ϕqI_vv,Val(3)), CA.getdata(ϕqIc))
    end
    cl = ForwardDiffGradNelboiZCl(
        approx, ϕqIc, gradh.dϕmvecs, gradh.hw_channel, cor_ends.M, transM,
        Val(keys(nljoint_args_inds)), nljoint_args_fix, n_MC_val)    
    ϕm_it = eachslice(h_ϕm; dims = ndims(h_ϕm))
        # _a1 = Test.@inferred first(zip(h.helpers_sites, rnormPM.M, zip_eachlastdims(nljoint_args_inds), ϕm_it, 1:n_site))
        # Test.@inferred cl(_a1)
    init = (;
        dϕqI = zero(TF) .* SA.SVector{axis_length(CA.getaxes(ϕqIc)[1])}(ϕqIc),
        dθsP = zero(TF) .* SA.SVector{axis_length(ax_θP)*NMC}(h.θsP), )
    # a1_ = first(zip(h.helpers_sites, rnormPM.M, zip_eachlastdims(nljoint_args_inds), ϕm_it, 1:n_site))
    # Test.@inferred cl(a1_)
    res = Folds.mapreduce(cl, make_tuple_reducer(+), 
        zip(h.helpers_sites, rnormPM.M, zip_eachlastdims(nljoint_args_inds), ϕm_it, 1:n_site),
        executor; init)
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
        gradh.dϕg, gradh.∂elbo_∂ϕm_∂ζP, h_ϕm, ∂elbo_∂ϕqm, 
        ϕg, xM, h.ζsP, pbm_covar_indices, g, is_testmode)
    #
    # pullback gradients of ∂elbo_∂θP to ∂elbo_∂θP_∂ζP
    ∂elbo_∂θP_∂ζP = gradh.∂elbo_∂θP_∂ζP
    gradh.pullback_cl_transformζsP!(
        ∂elbo_∂θP_∂ζP, 
        ∂elbo_∂θP,
        ∂elbo_∂ladJacTP,
        h.θsP,
        transP, 
        h.ζsP,
        )
    #
    # pullback gradients of ∂elbo_∂ζP, ∂elbo_∂ϕm_∂ζP, and ∂elbo_∂logσ_ζP to dϕqP
    gradh.pullback_cl_sample_ζsP!(
        gradh.dϕqPc, 
        ∂elbo_∂θP_∂ζP + gradh.∂elbo_∂ϕm_∂ζP, gradh.∂elbo_∂logσ_ζP,
        h.ζsP, h.logσ_ζP, approx, rnormPM.P, ϕqPc, cor_ends.P, h.sample_buffers,
        )
    (;dϕqPc = gradh.dϕqPc, dϕqIc = gradh.dϕqIc, dϕg = gradh.dϕg), gradh
end

"""
Callable to make deliver arguments that do not differ by individual to Foldl.mapreduce.
"""
struct ForwardDiffGradNelboiZCl{KEYS, NMC, TA, Tϕq, TD, THWC, TC, TM, TAF}
    approx::TA
    ϕqIc::Tϕq   # TODO remove from closure, provided with preallocated VcatVecViews
    dϕmvecs::TD
    hw_channel::THWC
    cor_endsM::TC
    transM::TM
    # need to store KEYS in type parameter otherwise allocation in NamedTuple(keys)
    keys_nljoint_args_indi::Val{KEYS}
    nljoint_args_fix::TAF
    n_MC_val::Val{NMC}
end
function (f::ForwardDiffGradNelboiZCl{KEYS})(tup) where KEYS
    hi, rnormM, nljoint_args_indi_tup, ϕm, i = tup
    #nljoint_args_indi = NamedTuple{f.keys_nljoint_args_indi}(nljoint_args_indi_tup)
    nljoint_args_indi = NamedTuple{KEYS}(nljoint_args_indi_tup)
    forwarddiff_grad_nelboi_z!(hi, rnormM, nljoint_args_indi, ϕm, i, f,
        nothing,
        #true,
        )
end

function forwarddiff_grad_nelboi_z!(hi, rnormM, 
    nljoint_args_indi, ϕm::AbstractArray{TF,ND}, i::Int, cl::ForwardDiffGradNelboiZCl{KEYS, NMC},
    omit_gradient=nothing) where {TF, ND, KEYS, NMC}
    # need to calls to ForwardDiff.gradient! to avoid reshape
    # group arrays of same dimensionality (here grad_ϕm and grad_ϕqI) into a Vcat
    ax_ϕqI = CA.getaxes(cl.ϕqIc)[1]
    ax_ϕP = get_axis_θP(cl.nljoint_args_fix.f!)
    with_channel_element(cl.hw_channel) do hwi_tup
        local hwi, hwi_dc, nelboi_vv, θsP_ϕm_ϕqI_vv, grad_θsP, grad_ϕm, grad_ϕqI
        hwi, hwi_dc = hwi_tup
        #grad_conf = hwi.grad_conf
        nelboi_vv = make_nelboiz_lazy_cls_vv(hi, cl.approx, rnormM, nljoint_args_indi, 
            cl.cor_endsM, cl.transM, cl.nljoint_args_fix, ax_ϕqI)
        # grads are vector-shaped, but copyto! dϕmvecs and SAVector use those anyhow           
        grad_θsP, grad_ϕm, grad_ϕqI = if isnothing(omit_gradient)
            # θsP_ϕm_ϕqI is preallocated per worker (hwi.θsP_ϕm_ϕqI).
            # Only ϕm (arg[2]) varies per site; θsP (arg[1]) and ϕqIc (arg[3])
            # are filled once before the mapreduce in grad_neg_elbo_sites.
            θsP_ϕm_ϕqI_vv = hwi.θsP_ϕm_ϕqI_vv
            copyto!(subarray(θsP_ϕm_ϕqI_vv,Val(2)), ϕm)
            ForwardDiff.gradient!(hwi.grad_θsP_ϕm_ϕqI_vv, nelboi_vv, θsP_ϕm_ϕqI_vv, 
                hwi.grad_conf_nelbo_vv)
            subarray(hwi.grad_θsP_ϕm_ϕqI_vv, Val(1)), 
                subarray(hwi.grad_θsP_ϕm_ϕqI_vv, Val(2)), 
                subarray(hwi.grad_θsP_ϕm_ϕqI_vv, Val(3))
        else
            θsP = subarray(hwi.θsP_ϕm_ϕqI_vv, Val(1))
            ϕqI = subarray(hwi.θsP_ϕm_ϕqI_vv, Val(3))
            θsP, ϕm, ϕqI
        end
        copyto!(view(cl.dϕmvecs, :, i), grad_ϕm) 
        # returning SVector helps avoiding allocations during reduce
        local res
        res = (;
            dϕqI = SA.SVector{axis_length(CA.getaxes(cl.ϕqIc)[1])}(grad_ϕqI),
            dθsP = SA.SVector{axis_length(ax_ϕP)*NMC}(grad_θsP),
        )
    end
end

# TODO remove unused arguments
function make_nelboiz_lazy_cls_vv(hi, approx::AbstractHVIApproximation, 
    rnormM, nljoint_args_indi, 
    cor_endsM, transM, nljoint_args_fix,
    ax_ϕqI::CA.Axis
    ) 
    # VcatVecViews(θsP, ϕm, ϕqI)
    function nelboiz_cl(vv::VcatVecViews) 
        local θsP_dual, ϕm_dual, ϕqIc_dual
        # reshape by copying into preallocated properly shaped arrays
        θsP_dual = subarray(vv,Val(1))
        ϕm_dual = subarray(vv,Val(2))
        ϕqIc_dual = CA.ComponentVector(subarray(vv,Val(3)), ax_ϕqI)
        compute_nelboi_z!(
            hi, approx, rnormM, nljoint_args_indi,
            ϕm_dual, ϕqIc_dual, θsP_dual, 
            cor_endsM, transM, nljoint_args_fix,
        )[1]
    end
end

function get_pullback_cl_sample_ζsP(::AbstractArray{TF}; n_θP, n_MC, sample_buffers) where TF
    #dϕqc, dζsP, dlogσ_ζP, ζsP, logσ_ζP, rnormP, ϕqc)
    ζsP_ = Matrix{TF}(undef, n_θP, n_MC)
    logσ_ζP_ = Vector{TF}(undef, n_θP)
    dζsP_ = similar(ζsP_)  # allocate space for derivatives
    dlogσ_ζP_ = similar(logσ_ζP_)
    drnormP = similar(ζsP_)
    dsample_buffers = map_leaves_nt(similar, sample_buffers) # allocate space for derivatives
    #
    function pullback_cl_sample_ζsP!(dϕqc, dζsP, dlogσ_ζP, ζsP, logσ_ζP, 
        approx::AbstractHVIApproximation, rnormP, ϕqc, cor_endsP, sample_buffers::NamedTuple)
        Enzyme.make_zero!(dϕqc)
        Enzyme.make_zero!(dsample_buffers)
        fill!(dϕqc, 0)            # the derivative to compute 
        copyto!(dζsP_, dζsP)      # input cotangents (modified in-place by Enzyme)
        copyto!(dlogσ_ζP_, dlogσ_ζP)
        copyto!(ζsP_, ζsP)        # modified in-place by Enzyme
        copyto!(logσ_ζP_, logσ_ζP)
        EnzBuffers = isempty(sample_buffers) ? Enzyme.Const(sample_buffers) : 
            Enzyme.Duplicated(sample_buffers, dsample_buffers)
        Enzyme.autodiff(
            Enzyme.Reverse,
            sample_ζsP!,
            Enzyme.Duplicated(ζsP_, dζsP_),  
            Enzyme.Duplicated(logσ_ζP_, dlogσ_ζP_),  
            Enzyme.Const(approx),
            Enzyme.DuplicatedNoNeed(rnormP, drnormP),  
            Enzyme.Duplicated(ϕqc, dϕqc),   
            Enzyme.Const(cor_endsP),
            EnzBuffers,
        )
    end
end

function get_pullback_g_apply(::AbstractArray{TG}, ::AbstractArray{TF}; 
    n_θP, n_cov, n_covP, n_MC, n_site, n_M,
    ) where {TG, TF}
    xMP_ = Matrix{TG}(undef, (n_cov + n_covP), n_MC * n_site)
    dxMP_ = similar(xMP_)
    ϕms_ = Matrix{TF}(undef, n_M, n_site)
    ϕms_mcs_ = Array{TF,3}(undef, n_M, n_MC, n_site)
    dϕms_ = similar(ϕms_)
    dϕms_mcs_ = similar(ϕms_mcs_)
    #dζsP_ = Matrix{TF}(undef, n_θP, n_MC)
    #
    function pullback_g_apply!(dϕg, dζsP, ϕms, dϕm, ϕg, xM, ζsP,
                               pbm_covar_indices, g, is_testmode)
        ϕms_buffer = isnothing(pbm_covar_indices) ? ϕms_ : ϕms_mcs_
        dϕms_buffer = isnothing(pbm_covar_indices) ? dϕms_ : dϕms_mcs_
        # assert that buffers were constructed with correct sizes
        n_cov_f, n_site_f = size(xM)
        n_covP_f = isnothing(pbm_covar_indices) ? 0 : length(pbm_covar_indices)
        n_MC_f = size(ζsP,2)
        @assert (n_cov, n_covP, n_MC, n_site) == (n_cov_f, n_covP_f, n_MC_f, n_site_f)
        @assert size(dϕms_buffer) == size(dϕm)
    
        fill!(dϕg,  zero(eltype(dϕg)))
        fill!(dζsP, zero(eltype(dζsP))) # also output cotangent
        fill!(dxMP_, zero(eltype(dxMP_)))
        # primal will be updated as in the forward, but shadow needs to be preserved
        copyto!(ϕms_buffer, ϕms) # copy to avoid modifying ϕm (although should be the same)
        copyto!(dϕms_buffer, dϕm) # copy to avoid modifying dϕm
        #copyto!(dζsP_, dζsP) # output, does not need to be preserved
        #
        Enzyme.autodiff(
            Enzyme.Reverse,
            #does not make SimpleChains work - currently cannot use SimpleChains
            #Enzyme.set_runtime_activity(Enzyme.Reverse), # TODO only activate for SimpleChains
            g_apply!,
            Enzyme.Duplicated(ϕms_buffer, dϕms_buffer),
            Enzyme.Duplicated(ϕg, dϕg),
            Enzyme.Const(xM),
            Enzyme.Duplicated(ζsP, dζsP),
            Enzyme.Const(pbm_covar_indices),
            Enzyme.Const(g),
            Enzyme.Duplicated(xMP_, dxMP_),
            Enzyme.Const(is_testmode),
        )
    end
end

# two return values: primal and derivative 
#   given coderiv dy, parameters and helpers
# dϕm -> dϕg, dζsP
function get_pullback_g_apply_ϕms_mcs2D_buffer(::AbstractArray{TG}, ::AbstractArray{TF}; 
    n_θP, n_cov, n_covP, n_MC, n_site, n_M,
    ) where {TG, TF}
    xMP_ = Matrix{TG}(undef, (n_cov + n_covP), n_MC * n_site)
    dxMP_ = similar(xMP_)
    ϕms_ = Matrix{TF}(undef, n_M, n_site)
    ϕms_mcs_ = Array{TF,3}(undef, n_M, n_MC, n_site)
    dϕms_ = similar(ϕms_)
    dϕms_mcs_ = similar(ϕms_mcs_)
    #dζsP_ = Matrix{TF}(undef, n_θP, n_MC)
    ϕms_mcs2D_buffer_ = Matrix{TF}(undef, n_M, n_MC * n_site)
    dϕms_mcs2D_buffer_ = similar(ϕms_mcs2D_buffer_)
    #
    function pullback_g_apply_ϕms_mcs2D_buffer!(dϕg, dζsP, ϕms, dϕm, ϕg, xM, ζsP,
                               pbm_covar_indices, g, is_testmode, ϕms_mcs2D_buffer)
        ϕms_buffer = isnothing(pbm_covar_indices) ? ϕms_ : ϕms_mcs_
        dϕms_buffer = isnothing(pbm_covar_indices) ? dϕms_ : dϕms_mcs_
        # assert that buffers were constructed with correct sizes
        n_cov_f, n_site_f = size(xM)
        n_covP_f = isnothing(pbm_covar_indices) ? 0 : length(pbm_covar_indices)
        n_MC_f = size(ζsP,1)
        @assert (n_cov, n_covP, n_MC, n_site) == (n_cov_f, n_covP_f, n_MC_f, n_site_f)
        @assert size(dϕms_buffer) == size(dϕm)
        @assert size(ϕms_mcs2D_buffer_) == size(ϕms_mcs2D_buffer)
    
        fill!(dϕg,  zero(eltype(dϕg)))
        fill!(dζsP, zero(eltype(dζsP))) # also output cotangent
        fill!(dxMP_, zero(eltype(dxMP_)))
        fill!(dϕms_mcs2D_buffer_, zero(eltype(dϕms_mcs2D_buffer_)))
        # primal will be updated as in the forward, but shadow needs to be preserved
        copyto!(ϕms_buffer, ϕms) # copy to avoid modifying ϕm (although should be the same)
        copyto!(dϕms_buffer, dϕm) # copy to avoid modifying dϕm
        #copyto!(dζsP_, dζsP) # output, does not need to be preserved
        copyto!(ϕms_mcs2D_buffer_, ϕms_mcs2D_buffer) # copy to avoid modifying dϕm
        #
        Enzyme.autodiff(
            Enzyme.Reverse, g_apply!,
            Enzyme.Duplicated(ϕms_buffer, dϕms_buffer),
            Enzyme.Duplicated(ϕg, dϕg),
            Enzyme.Const(xM),
            Enzyme.Duplicated(ζsP, dζsP),
            Enzyme.Const(pbm_covar_indices),
            Enzyme.Const(g),
            Enzyme.Duplicated(xMP_, dxMP_),
            Enzyme.Const(is_testmode),
            Enzyme.Duplicated(ϕms_mcs2D_buffer_, dϕms_mcs2D_buffer_),
        )
    end
end

function get_pullback_cl_transformζ!(::AbstractArray{TF};  n_θ, n_MC) where {TF}
    θs_buffer = Matrix{TF}(undef, n_θ, n_MC)
    dθs_buffer = similar(θs_buffer)
    # Enzyme seeds an Active return value with one, so to seed its cotangent
    # with an arbitrary dladJacTP we fold it in as a scalar factor, relying on
    # linearity of the reverse-mode adjoint: the pullback then delivers
    # dladJacTP * ∂ladJacT/∂(·) to θs and ζs, as with the former Ref seeding.
    function pullback_cl_transformζ!(dζs, dθs, dladJacTP, θs, trans, ζs)
        fill!(dζs, zero(eltype(dζs)))
        copyto!(θs_buffer, θs)
        copyto!(dθs_buffer, dθs)
        # Trick of seeding the active return value different to unity:
        # Here dladJacTP is captured from the closure (an Active-compatible scalar), 
        # the returned Active value's unit seed gets multiplied by dladJacTP, 
        # and the adjoints of θs/ζs come out as dladJacTP * ∂/∂(·)
        Enzyme.autodiff(
            Enzyme.Reverse,
            (θs_, trans_, ζs_) -> dladJacTP * transformζ!(θs_, trans_, ζs_),
            Enzyme.Duplicated(θs_buffer, dθs_buffer),
            Enzyme.Const(trans),
            Enzyme.Duplicated(ζs, dζs),
        )
    end
end

function prepare_gradelbo_helpers(
    ϕqIc, ϕm::AbstractArray{TF,ND}, θsP,
    ϕg::AbstractVector{TG}, ϕqPc::AbstractVector{TF},
    approx::AbstractHVIApproximation; 
    pbm_covar_indices, n_workers,
    h, rnormMi, nljoint_args_indi,
    n_site, n_cov, cor_ends,
    transP::Stacked, transM::Stacked,
    nljoint_args_fix,
    n_MC_val::Val{NMC},
    max_diffchunk::ForwardDiff.Chunk{chunk} = ForwardDiff.Chunk(8),
    ) where {TG, TF, ND, NMC, chunk}
    local hi1, ax_θP, ax_θM, n_covP, n_ϕmvec, n_θP, n_MC, n_M
    hi1 = h.helpers_sites[1]
    ax_θP = get_axis_θP(nljoint_args_fix.f!)
    ax_θM = get_axis_θM(nljoint_args_fix.f!)
    use_ϕm_vector = (ND == 1)
    n_covP =  isnothing(pbm_covar_indices) ? 0 : length(pbm_covar_indices)
    n_ϕmvec = length(ϕm) #use_ϕm_matrix ? n_M : n_M * n_MC
    n_θP, n_MC = size(θsP)
    @assert n_MC == NMC
    @assert n_θP == axis_length(ax_θP)
    n_M = size(ϕm,1)
    # To sync across procs/threads, use a Channel 
    # https://juliafolds2.github.io/OhMyThreads.jl/stable/literate/tls/tls/#The-safe-way:-Channel
    #    
    local hw_dcs_raw, hw_dcs
    @assert length(nljoint_args_indi.xP) > 1
    # nelboi_z = make_nelboiz_cl(hi1, approx, rnormMi, nljoint_args_indi, CA.getaxes(cv_grad), 
    #     cor_ends.M, transM, nljoint_args_fix) 
    hw_dcs_raw = (;
            ϕm_dc = similar(ϕm),   
            # callocation on using ComponentArray (_restructure), need to use plain vector
            ϕqI_dc = similar(CA.getdata(ϕqIc)), 
            θsP_dc = similar(θsP),
    )
    # make sure, each worker gets its own copy, use similar
    hw_dcs = (hw_dcs_raw) -> map_leaves_nt(x -> PAT.DiffCache(similar(x), chunk), hw_dcs_raw) 
    #
    local nelboiz_vv, θsP_ϕm_ϕqI_vv1, diffchunks_elbo_vv
    nelboiz_vv = make_nelboiz_lazy_cls_vv(hi1, approx, rnormMi, nljoint_args_indi, 
        cor_ends.M, transM, nljoint_args_fix, CA.getaxes(ϕqIc)[1], )
    θsP_ϕm_ϕqI_vv1 = θsP_ϕm_ϕqI_vv(θsP,ϕm,ϕqIc)
    # n_M >= n_θM but not static
    diffchunks_elbo_vv = ForwardDiff.Chunk(min(chunk, 
        axis_length(ax_θP)*NMC + axis_length(ax_θM)*(use_ϕm_vector ? 1 : NMC) + axis_length(CA.getaxes(ϕqIc)[1])))
    # https://juliadiff.org/ForwardDiff.jl/stable/user/advanced/#Custom-tags-and-tag-checking
    function grad_conf_nelbo_vv() 
        cfg = ForwardDiff.GradientConfig(nelboiz_vv, θsP_ϕm_ϕqI_vv1, diffchunks_elbo_vv,
            #Val{false}()  # use only for debugging, where tags clutter the output
        )
        update_duals(cfg, θsP_ϕm_ϕqI_vv1)
    end    

    local h1,h1_dc,h1_tup,hw_channel
    get_helpers_worker = () -> begin
        (;
            grad_θsP_ϕm_ϕqI_vv = θsP_ϕm_ϕqI_vv(θsP, ϕm, ϕqIc),
            grad_conf_nelbo_vv = grad_conf_nelbo_vv(),
            θsP_ϕm_ϕqI_vv = θsP_ϕm_ϕqI_vv(θsP, ϕm, ϕqIc),
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
        dθsP = similar(θsP), 
        ∂elbo_∂θP = similar(θsP),
        ∂elbo_∂ϕm_∂ζP = similar(θsP),
        ∂elbo_∂logσ_ζP = Vector{TF}(undef, size(θsP,1)),
        ∂elbo_∂θP_∂ζP = Matrix{TF}(undef, size(θsP)),
        ∂elbo_∂ϕqm = Array{TF}(undef, size(ϕm)..., n_site),
        hw_channel,
        pullback_cl_sample_ζsP! = get_pullback_cl_sample_ζsP(CA.getdata(ϕqPc); 
            n_θP, n_MC, sample_buffers = h.sample_buffers),
        pullback_cl_transformζsP! = get_pullback_cl_transformζ!(CA.getdata(ϕqPc); n_θ = n_θP, n_MC),
        pullback_g_apply! = get_pullback_g_apply(
            ϕg, ϕqPc; n_θP, n_cov, n_covP, n_MC, n_site, n_M),
        # SharedArrays last because of problems with ReferenceRevision
        dϕmvecs = SharedArrays.SharedArray{TF}(n_ϕmvec, n_site), 
        #dθsP = SharedArrays.SharedArray{TF}(n_θP, n_MC), 
    )
end
@inline θsP_ϕm_ϕqI_vv(θsP, ϕm, ϕqIc) = VcatVecViews(similar(θsP), similar(ϕm), similar(CA.getdata(ϕqIc)) ) 


