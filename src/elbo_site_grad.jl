
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
    cl = ForwardDiffGradNelboiZCl(
        approx, ϕqIc, h.θsP, gradh.dϕmvecs, gradh.dθsP, gradh.hw_channel, cor_ends.M, transM,
        Val(keys(nljoint_args_inds)), nljoint_args_fix, n_MC_val)    
    ϕm_it = eachslice(h_ϕm; dims = ndims(h_ϕm))
    init = (;
        dϕqI = zero(TF) .* SA.SVector{axis_length(CA.getaxes(ϕqIc)[1])}(ϕqIc),
        dθsP = zero(TF) .* SA.SVector{axis_length(ax_θP)*NMC}(h.θsP), )
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
struct ForwardDiffGradNelboiZCl{KEYS, TA, Tϕq, Tθ, TD, TP, THWC, TC, TM, TAF, NMC}
    approx::TA
    ϕqIc::Tϕq
    θsP::Tθ
    dϕmvecs::TD
    dθsP::TP
    hw_channel::THWC
    corendsM::TC
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
    forwarddiff_grad_nelboi_z!(hi, f.approx, rnormM, nljoint_args_indi, ϕm, i,
        f.ϕqIc, f.θsP, f.dϕmvecs, f.dθsP, f.hw_channel, f.corendsM, f.transM, 
        f.nljoint_args_fix, f.n_MC_val,
        nothing,
        #true,
        )
end


# ϕm vector version
# TOOD pass closure object rather than many parameters
function forwarddiff_grad_nelboi_z!(hi, approx::AbstractHVIApproximation, rnormM, 
    nljoint_args_indi, ϕm::AbstractVector{TF}, i, 
    ϕqIc, θsP, dϕmvecs, dθsP, hw_channel::Channel, cor_endsM, transM::Stacked, 
    nljoint_args_fix, n_MC_val::Val{NMC},
    omit_gradient=nothing) where {TF, NMC}
    # need to calls to ForwardDiff.gradient! to avoid reshape
    # group arrays of same dimensionality (here grad_ϕm and grad_ϕqI) into a Vcat
    ax_ϕqI = CA.getaxes(ϕqIc)[1]
    ax_ϕP = get_axis_θP(nljoint_args_fix.f!)
    with_channel_element(hw_channel) do hwi_tup
        hwi, hwi_dc = hwi_tup
        #grad_conf = hwi.grad_conf
        nelboi_z_mat, nelboi_z_vec = make_nelboiz_lazy_cls(hi, hwi_dc, approx, rnormM, nljoint_args_indi, 
            ϕm, ϕqIc, θsP,
            cor_endsM, transM, nljoint_args_fix,
            ax_ϕqI, Val(1), 
            )
        grad_θsP = if isnothing(omit_gradient)
            ForwardDiff.gradient!(hwi.grad_θsP, nelboi_z_mat, CA.getdata(θsP), hwi.grad_conf_mat)
            hwi.grad_θsP
        else
            θsP
        end
        grad_ϕm, grad_ϕqI = if isnothing(omit_gradient)
            local ϕm_ϕqI
            ϕm_ϕqI = Vcat(ϕm, CA.getdata(ϕqIc))    
            ForwardDiff.gradient!(hwi.grad_ϕm_ϕqI, nelboi_z_vec, ϕm_ϕqI, hwi.grad_conf_vec)
            hwi.grad_ϕm_ϕqI.args[1], hwi.grad_ϕm_ϕqI.args[2]
        else
            ϕm, CA.getdata(ϕqIc)
        end
        copyto!(view(dϕmvecs, :, i), grad_ϕm) 
        # returning SVector helps avoiding allocations during reduce
        local res
        res = (;
            dϕqI = SA.SVector{axis_length(CA.getaxes(ϕqIc)[1])}(grad_ϕqI),
            dθsP = SA.SVector{axis_length(ax_ϕP)*NMC}(grad_θsP),
        )
    end
end


#@noinline CP.forwarddiff_grad_nelboi_z!(hi, approx, rnormM, nljoint_args_indi, ϕm_, i,  
#    ϕqIc_, θsP_, dϕmvecs_, dθsP_, hw_channel, cor_endsM, transM, nljoint_args_fix, omit_gradient)


# ϕm matrix version
function forwarddiff_grad_nelboi_z!(hi, approx::AbstractHVIApproximation, rnormM, 
    nljoint_args_indi, ϕm::AbstractMatrix{TF}, i, 
    ϕqIc, θsP, dϕmvecs, dθsP, hw_channel::Channel, cor_endsM, transM::Stacked, 
    nljoint_args_fix, n_MC_val::Val{NMC},
    omit_gradient=nothing) where {TF, NMC}
    # need to calls to ForwardDiff.gradient! to avoid reshape
    # group arrays of same dimensionality (here grad_ϕm and grad_ϕqI) into a Vcat
    ax_ϕqI = CA.getaxes(ϕqIc)[1]
    ax_ϕqP = get_axis_θP(nljoint_args_fix.f!)
    with_channel_element(hw_channel) do hwi_tup
        hwi, hwi_dc = hwi_tup
        #grad_conf = hwi.grad_conf
        nelboi_z_mat, nelboi_z_vec = make_nelboiz_lazy_cls(hi, hwi_dc, approx, rnormM, nljoint_args_indi, 
            ϕm, ϕqIc, θsP,
            cor_endsM, transM, nljoint_args_fix,
            ax_ϕqI, Val(2), ## 2-dimensional ϕm
            )
        grad_θsP, grad_ϕm = if isnothing(omit_gradient)
            local θsP_ϕm
            θsP_ϕm = Vcat(θsP, ϕm)    
            ForwardDiff.gradient!(hwi.grad_θsP_ϕm, nelboi_z_mat, θsP_ϕm, hwi.grad_conf_mat)
            hwi.grad_θsP_ϕm.args[1], hwi.grad_θsP_ϕm.args[2]
        else
            θsP, ϕm
        end
        grad_ϕqI = if isnothing(omit_gradient)
            # TODO try reusing a view into grad_ϕm_ϕqI
            ForwardDiff.gradient!(hwi.grad_ϕqI, nelboi_z_vec, CA.getdata(ϕqIc), hwi.grad_conf_vec)
            hwi.grad_ϕqI
        else
            CA.getdata(ϕqIc)
        end
        copyto!(view(dϕmvecs, :, i), grad_ϕm) 
        # cannot use SVector for 
        # dθsP .+= grad_θsP # SharedArray, 
        # # returning SVector helps avoiding allocations during reduce
        # local dϕqIc
        # dϕqIc = SA.SVector{axis_length(CA.getaxes(ϕqIc)[1])}(grad_ϕqI)
        local res
        res = (;
            dϕqI = SA.SVector{axis_length(CA.getaxes(ϕqIc)[1])}(grad_ϕqI),
            dθsP = SA.SVector{axis_length(ax_ϕqP)*NMC}(grad_θsP),
        )
    end
end


function make_nelboiz_lazy_cls(hi, hwi_dc, approx::AbstractHVIApproximation, rnormM, nljoint_args_indi, 
    ϕm, ϕqIc, θsP,
    cor_endsM, transM, nljoint_args_fix,
    ax_ϕqI::CA.Axis, ndim_ϕ::Val{1}, 
    )
    # mat, here holds only θsP
    # vec, is a Vcat(ϕqI, ϕm)
    # no views involved here
    function nelboiz_cl_mat(mat::AbstractMatrix) 
        local θsP_dual, ϕm_dual, ϕqIc_dual
        θsP_dual = mat
        ϕm_dual = copyto!(PAT.get_tmp(hwi_dc.ϕm_dc, mat), ϕm)
        ϕqIc_dual = CA.ComponentArray(
            copyto!(PAT.get_tmp(hwi_dc.ϕqI_dc, mat), ϕqIc), CA.getaxes(ϕqIc))
        compute_nelboi_z!(
            hi, approx, rnormM, nljoint_args_indi,
            ϕm_dual, ϕqIc_dual, θsP_dual, 
            cor_endsM, transM, nljoint_args_fix,
        )[1]
    end
    function nelboiz_cl_vec(vec::Vcat) 
        local θsP_dual, ϕm_dual, ϕqIc_dual
        ϕm_dual = vec.args[1]
        ϕqIc_dual = CA.ComponentVector(vec.args[2], ax_ϕqI)
        θsP_dual = copyto!(PAT.get_tmp(hwi_dc.θsP_dc, vec), θsP) # reshape(reinterpred(
        compute_nelboi_z!(
            hi, approx, rnormM, nljoint_args_indi,
            ϕm_dual, ϕqIc_dual, θsP_dual, 
            cor_endsM, transM, nljoint_args_fix,
        )[1]
    end
    nelboiz_cl_mat, nelboiz_cl_vec
end

# ϕm Matrix case
function make_nelboiz_lazy_cls(hi, hwi_dc, approx::AbstractHVIApproximation, rnormM, nljoint_args_indi, 
    ϕm, ϕqIc, θsP,
    cor_endsM, transM, nljoint_args_fix,
    ax_ϕqI::CA.Axis, ndim_ϕ::Val{2}, 
    )
    # mat is Vcat(θsP, ϕm)
    # vec is ϕqI
    function nelboiz_cl_mat(mat::Vcat) 
        local θsP_dual, ϕm_dual, ϕqIc_dual
        θsP_dual = mat.args[1]
        ϕm_dual = mat.args[2]
        ϕqIc_dual = CA.ComponentArray(
            copyto!(PAT.get_tmp(hwi_dc.ϕqI_dc, mat), ϕqIc), CA.getaxes(ϕqIc))
        compute_nelboi_z!(
            hi, approx, rnormM, nljoint_args_indi,
            ϕm_dual, ϕqIc_dual, θsP_dual, 
            cor_endsM, transM, nljoint_args_fix,
        )[1]
    end
    function nelboiz_cl_vec(vec::AbstractVector) 
        local θsP_dual, ϕm_dual, ϕqIc_dual
        ϕqIc_dual = CA.ComponentVector(vec, ax_ϕqI)
        ϕm_dual = copyto!(PAT.get_tmp(hwi_dc.ϕm_dc, vec), ϕm)
        θsP_dual = copyto!(PAT.get_tmp(hwi_dc.θsP_dc, vec), θsP)
        compute_nelboi_z!(
            hi, approx, rnormM, nljoint_args_indi,
            ϕm_dual, ϕqIc_dual, θsP_dual, 
            cor_endsM, transM, nljoint_args_fix,
        )[1]
    end
    nelboiz_cl_mat, nelboiz_cl_vec
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
    diffchunk::ForwardDiff.Chunk{chunk} = ForwardDiff.Chunk(8),
    ) where {TG, TF, ND, NMC, chunk}
    hi1 = h.helpers_sites[1]
    ax_θP = get_axis_θP(nljoint_args_fix.f!)
    ax_θM = get_axis_θM(nljoint_args_fix.f!)
    # TODO get n_site, n_cov from 
    cv_grad = CA.ComponentArray(; 
        ϕqIc, 
        ϕm, # = selectdim(ϕms, ndims(ϕm), 1), 
        θsP,
    )
    # ϕqIc = cv_grad[Val(:ϕqIc)]
    # ϕm = cv_grad[Val(:ϕm)]
    # θsP = cv_grad[Val(:θsP)]
    use_ϕm_vector = isnothing(pbm_covar_indices) # for one site, provide singe ϕm across MC
    n_covP =  use_ϕm_vector ? 0 : length(pbm_covar_indices)
    n_ϕmvec = length(ϕm) #use_ϕm_matrix ? n_M : n_M * n_MC
    n_θP, n_MC = size(θsP)
    @assert n_MC == NMC
    @assert n_θP == axis_length(ax_θP)
    n_M = size(ϕm,1)
    # To sync across procs/threads, use a Channel 
    # https://juliafolds2.github.io/OhMyThreads.jl/stable/literate/tls/tls/#The-safe-way:-Channel
    #    
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
    nelboi_z_mat, nelboi_z_vec = make_nelboiz_lazy_cls(hi1, hw_dcs(hw_dcs_raw), approx, rnormMi, nljoint_args_indi, 
        ϕm, ϕqIc, θsP,
        cor_ends.M, transM, nljoint_args_fix,
        CA.getaxes(ϕqIc)[1], Val(ND),
        ) 
    grad_conf_mat, grad_conf_vec = if use_ϕm_vector
        local grad_ϕm_ϕqI1, diffchunks_vec, diffchunks_mat
        diffchunk_vec = ForwardDiff.Chunk(min(chunk, 
            axis_length(CA.getaxes(ϕqIc)[1]) + axis_length(ax_θM)))
        diffchunk_mat = ForwardDiff.Chunk(min(chunk, 
            axis_length(CA.getaxes(ϕqPc)[1])))
        grad_conf_mat_ϕmvec() = ForwardDiff.GradientConfig(nelboi_z_mat, CA.getdata(θsP), diffchunk_mat)
        grad_ϕm_ϕqI1 = grad_ϕm_ϕqI(ϕm, ϕqIc)
        function grad_conf_vec_ϕmvec() 
            cfg_vec = ForwardDiff.GradientConfig(nelboi_z_vec, grad_ϕm_ϕqI1, diffchunk_vec)
            update_duals_Vcat2(cfg_vec, grad_ϕm_ϕqI1)
        end
        grad_conf_mat_ϕmvec, grad_conf_vec_ϕmvec
    else
        local rad_θP_ϕm1, diffchunks_vec, diffchunks_mat
        diffchunks_vec = ForwardDiff.Chunk(min(chunk, 
            axis_length(CA.getaxes(ϕqIc)[1])))
        diffchunks_mat = ForwardDiff.Chunk(min(chunk, 
            NMC*(axis_length(CA.getaxes(ϕqPc)[1]) + axis_length(ax_θM))))
        grad_conf_vec_ϕmmat() = ForwardDiff.GradientConfig(nelboi_z_vec, CA.getdata(ϕqIc), diffchunks_vec)
        rad_θP_ϕm1 = grad_θsP_ϕm(ϕm, θsP)
        function grad_conf_mat_ϕmmat() 
            cfg_mat = ForwardDiff.GradientConfig(nelboi_z_mat, rad_θP_ϕm1, diffchunks_mat)
            update_duals_Vcat2(cfg_mat, rad_θP_ϕm1)
        end
        grad_conf_mat_ϕmmat, grad_conf_vec_ϕmmat
    end
    get_helpers_worker = () -> begin
        (;
            grad_conf_vec = grad_conf_vec(), 
            grad_conf_mat = grad_conf_mat(),
            grad_θsP = similar(θsP),
            grad_ϕm_ϕqI = grad_ϕm_ϕqI(ϕm, ϕqIc),
            grad_θsP_ϕm= grad_θsP_ϕm(ϕm, θsP),
            grad_ϕqI = similar(CA.getdata(ϕqIc)),
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
@inline grad_ϕm_ϕqI(ϕm::AbstractVector, ϕqIc) = Vcat(similar(ϕm), similar(CA.getdata(ϕqIc))) 
@inline grad_ϕm_ϕqI(ϕm::AbstractMatrix, ϕqIc) = Vcat(similar(ϕm[1:0,:1]), similar(CA.getdata(ϕqIc))) # not used but preallocated
@inline grad_θsP_ϕm(ϕm::AbstractVector, θsP) = Vcat(similar(θsP[:,1:0]), similar(θsP[:,1:0])) # not used but preallocated
@inline grad_θsP_ϕm(ϕm::AbstractMatrix, θsP) = Vcat(similar(θsP), similar(ϕm))


@inline function update_duals_Vcat2(cfg::ForwardDiff.GradientConfig{T,V,N}, vc::Vcat) where {T,V,N}
    # similar Vcat returns a vector, need to convert dual of cfg to Vcat
    # no view, but copy with index_at_dim
    @assert length(vc.args) == 2
    duals_vcat = Vcat(index_at_dim(cfg.duals, axes(vc.args[1], 1); dim=1), 
                      index_at_dim(cfg.duals, axes(vc.args[2], 1); dim=1));
    cfg = ForwardDiff.GradientConfig{T,V,N,typeof(duals_vcat)}(cfg.seeds, duals_vcat)
    @assert cfg.duals isa Vcat
    cfg
end
@inline function update_duals_Vcat3(cfg::ForwardDiff.GradientConfig{T,V,N}, vc::Vcat) where {T,V,N}
    # similar Vcat returns a vector, need to convert dual of cfg to Vcat
    # no view, but copy with index_at_dim
    @assert length(vc.args) == 3
    duals_vcat = Vcat(index_at_dim(cfg.duals, axes(vc.args[1], 1); dim=1), 
                      index_at_dim(cfg.duals, axes(vc.args[2], 1); dim=1),
                      index_at_dim(cfg.duals, axes(vc.args[3], 1); dim=1),
                      );
    cfg = ForwardDiff.GradientConfig{T,V,N,typeof(duals_vcat)}(cfg.seeds, duals_vcat)
    @assert cfg.duals isa Vcat
    cfg
end




