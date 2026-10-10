function grad_neg_elbo_sites(
    elbo_helpers::NamedTuple,      # tuple of preallocated arrays
    grad_elbo_helpers::NamedTuple,  # tuple of preallocated arrays pullback closures
    ϕ::NamedTuple,
    rnormPM::NamedTuple,
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
    nljoint_args_indi = NamedTuple{keys(nljoint_args_inds)}(
        first(zip_eachlastdims(nljoint_args_inds)))
    gradh = !isempty(grad_elbo_helpers) ? grad_elbo_helpers : prepare_gradelbo_helpers(
        ϕqIc, selectdim(h_ϕm, ndims(h_ϕm), 1), h.ζsP, ϕg, ϕqPc, approx; 
        pbm_covar_indices, n_workers,
        h, rnormMi = rnormPM.M[1], nljoint_args_indi,
        h.diffchunk, n_site, n_cov, cor_ends, transP, transM, nljoint_args_fix)
    hw_channel = gradh.hw_channel
    #
    sample_ζsP!(h.ζsP, h.logσ_ζP, approx, rnormPM.P, ϕqPc, cor_ends.P, h.sample_buffers) # n_P * n_MC
    g_apply!(h_ϕm, ϕg, xM, h.ζsP, pbm_covar_indices, g, h.xMP, is_testmode) 
    ladJacTP = transformζ!(h.θsP, transP, h.ζsP)  # return value captures ladJacT
    #
    # parallel ForwardDiffGradient through forwarddiff_grad_nelboi_z!
    cl = ForwardDiffGradNelboiZCl(
        approx, ϕqIc, h.θsP, gradh.dϕmvecs, gradh.hw_channel, cor_ends.M, transM,
        Val(keys(nljoint_args_inds)), nljoint_args_fix,)    
    ϕm_it = eachslice(h_ϕm; dims = ndims(h_ϕm))
    init = with_channel_element(hw_channel) do hwi 
        dθsP = Tuple(zero(SA.SVector{axis_length(nljoint_args_fix.ax_θP)}(grad_θP)) for 
            grad_θP in eachcol(gradh.∂elbo_∂θP))
        res = (; 
            dϕqIc = zero(SA.SVector{axis_length(CA.getaxes(ϕqIc)[1])}(ϕqIc)),
            dθsP, 
        )
        # (; dϕqIc = zero(static_cv_getproperty(hwi.inputs_cv, Val(:ϕqIc))),
        # dθsP = zero(static_cv_getproperty(hwi.inputs_cv, Val(:θsP))))
    end
    # sum gradient across sites for ϕqIc and θsP
    # because of non-static n_MC, reduce a Tuple of cols for dθsP
    function red_grad(x,y) 
        (;
            dϕqIc = x.dϕqIc + y.dϕqIc,
            dθsP = map(+, x.dθsP, y.dθsP)# its a Tuple of columns
        )
    end
    gacc = Folds.mapreduce(cl, red_grad, #make_tuple_reducer(+), 
        zip(h.helpers_sites, rnormPM.M, zip_eachlastdims(nljoint_args_inds), ϕm_it, 1:n_site),
        executor; init)
    ∂elbo_∂ϕqI = gacc.dϕqIc # tuple access
    # copy Tuple of SVector columns into preallocated array for ∂elbo_∂θP
    ∂elbo_∂θP = gradh.∂elbo_∂θP
    for i in axes(∂elbo_∂θP,2)
        copyto!(view(∂elbo_∂θP,:,i), gacc.dθsP[i])
    end
    #∂elbo_∂ϕqm = reshape(cl.dϕmvecs, size(h_ϕm)) # shared array   
    ∂elbo_∂ϕqm = gradh.∂elbo_∂ϕqm #preallocate to avoid copy in reshape of SharedArray
    copyto!(∂elbo_∂ϕqm, cl.dϕmvecs) # reshape from shared array
    #gradh.∂elbo_∂logσ_ζP .= -ones(TF, length(h.logσ_ζP))  
    gradh.∂elbo_∂logσ_ζP .= -one(TF)  
    ∂elbo_∂ladJacTP = -one(TF)
    #
    # pullback gradients of ϕqm -> gradh.dϕg and gradh.dζsP
    # gradh.pullback_g_apply!(
    #     gradh.dϕg, gradh.∂elbo_∂ϕm_∂ζP, h_ϕm, ∂elbo_∂ϕqm, 
    #     ϕg, xM, h.ζsP, pbm_covar_indices, g, is_testmode, h.ϕms_mcs2D_buffer)
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
    dϕqP = gradh.dϕqP
    #pullback_sample_ζsP!(
    gradh.pullback_cl_sample_ζsP!(
        dϕqP, 
        ∂elbo_∂θP_∂ζP + gradh.∂elbo_∂ϕm_∂ζP, gradh.∂elbo_∂logσ_ζP,
        h.ζsP, h.logσ_ζP, approx, rnormPM.P, ϕqPc, cor_ends.P, h.sample_buffers,
        )
    (;dϕqP, dϕqI = ∂elbo_∂ϕqI, dϕg = gradh.dϕg), gradh
end

"""
Callable to make deliver arguments that do not differ by individual to Foldl.mapreduce.
"""
struct ForwardDiffGradNelboiZCl{KEYS, TA, Tϕq, Tθ, TD, THWC, TC, TM, TAF}
    approx::TA
    ϕqIc::Tϕq
    θsP::Tθ
    dϕmvecs::TD
    hw_channel::THWC
    corendsM::TC
    transM::TM
    # need to store KEYS in type parameter otherwise allocation in NamedTuple(keys)
    keys_nljoint_args_indi::Val{KEYS}
    nljoint_args_fix::TAF
end
function (f::ForwardDiffGradNelboiZCl{KEYS})(tup) where KEYS
    hi, rnormM, nljoint_args_indi_tup, ϕm, i = tup
    #nljoint_args_indi = NamedTuple{f.keys_nljoint_args_indi}(nljoint_args_indi_tup)
    nljoint_args_indi = NamedTuple{KEYS}(nljoint_args_indi_tup)
    # forwarddiff_grad_nelboi_z!(hi, f.approx, rnormM, nljoint_args_indi, ϕm, i,
    #     f.ϕqIc, f.θsP, f.dϕmvecs, f.hw_channel, f.corendsM, f.transM, f.nljoint_args_fix,
    #     nothing,
    #     #true,
    #     )
    forwarddiff_grad_nelboi_z!(hi, f.approx, rnormM, nljoint_args_indi, ϕm, i,
        f.ϕqIc, f.θsP, f.dϕmvecs, f.hw_channel, f.corendsM, f.transM, f.nljoint_args_fix,
        nothing,
        #true,
        )
end

# ϕm vector version
function forwarddiff_grad_nelboi_z!(hi, approx::AbstractHVIApproximation, rnormM, 
    nljoint_args_indi, ϕm::AbstractVector{TF}, i, 
    ϕqIc, θsP, dϕmvecs, hw_channel::Channel, cor_endsM, transM::Stacked, 
    nljoint_args_fix,
    omit_gradient=nothing) where {TF}
    # need to calls to ForwardDiff.gradient! to avoid reshape
    # group arrays of same dimensionality (here grad_ϕm and grad_ϕqI) into a Vcat
    ax_ϕqI = CA.getaxes(ϕqIc)[1]
    with_channel_element(hw_channel) do hwi
        #grad_conf = hwi.grad_conf
        nelboi_z_mat, nelboi_z_vec = make_nelboiz_lazy_cls(hi, approx, rnormM, nljoint_args_indi, 
            ϕm, ϕqIc, θsP,
            cor_endsM, transM, nljoint_args_fix,
            ax_ϕqI, Val(1), 
            )
        grad_θsP = if isnothing(omit_gradient)
            ForwardDiff.gradient!(hwi.grad_θsP, nelboi_z_mat, CA.getdata(θsP), hwi.grad_conf_mat)
            hwi.grad_θsP
        else
            CA.getdata(θsP)
        end
        grad_ϕm, grad_ϕqI = if isnothing(omit_gradient)
            ϕm_ϕqI = Vcat(CA.getdata(ϕm), CA.getdata(ϕqIc))    
            ForwardDiff.gradient!(hwi.grad_ϕm_ϕqI, nelboi_z_vec, ϕm_ϕqI, hwi.grad_conf_vec)
            hwi.grad_ϕm_ϕqI.args[1], hwi.grad_ϕm_ϕqI.args[2]

        else
            CA.getdata(ϕm), ϕqIc
        end
        copyto!(view(dϕmvecs, :, i), grad_ϕm) 
        # returning SVector helps avoiding allocations during reduce
        # n_MC not static -> inferred Tuple{Vararg{StaticArraysCore.SVector{n_θP, Float64}}}
        dθsP = Tuple(SA.SVector{axis_length(nljoint_args_fix.ax_θP)}(grad_θP) for grad_θP in eachcol(grad_θsP))
        res = (; 
            dϕqIc = SA.SVector{axis_length(CA.getaxes(ϕqIc)[1])}(grad_ϕqI),
            dθsP, 
        )
    end
end

function make_nelboiz_lazy_cls(hi, approx::AbstractHVIApproximation, rnormM, nljoint_args_indi, 
    ϕm, ϕqIc, θsP,
    cor_endsM, transM, nljoint_args_fix,
    ax_ϕqI::CA.Axis, ndim_ϕ::Val{1}, 
    )
    # mat, here holds only θsP
    # vec, is a Vcat(ϕqI, ϕm)
    # no views involved here
    function nelboiz_cl_mat(mat::AbstractMatrix) 
        θsP_dual = mat
        compute_nelboi_z!(
            hi, approx, rnormM, nljoint_args_indi,
            ϕm, ϕqIc, θsP_dual, 
            cor_endsM, transM, nljoint_args_fix,
        )[1]
        
    end
    function nelboiz_cl_vec(vec) 
        ϕm_dual = vec.args[1]
        ϕqI_dual = CA.ComponentVector(vec.args[2], ax_ϕqI)
        compute_nelboi_z!(
            hi, approx, rnormM, nljoint_args_indi,
            ϕm_dual, ϕqI_dual, θsP, 
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
        n_MC_f = size(ζsP,1)
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
    diffchunk, n_site, n_cov,
    cor_ends,
    transP::Stacked, transM::Stacked,
    nljoint_args_fix,
    ) where {TG, TF, ND}
    hi1 = h.helpers_sites[1]
    # TODO get n_site, n_cov diffchunk from h
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
    @assert n_θP == axis_length(nljoint_args_fix.ax_θP)
    n_M = size(ϕm,1)
    # To sync across procs/threads, use a Channel 
    # https://juliafolds2.github.io/OhMyThreads.jl/stable/literate/tls/tls/#The-safe-way:-Channel
    #    
    @assert length(nljoint_args_indi.xP) > 1
    # nelboi_z = make_nelboiz_cl(hi1, approx, rnormMi, nljoint_args_indi, CA.getaxes(cv_grad), 
    #     cor_ends.M, transM, nljoint_args_fix) 
    nelboi_z_mat, nelboi_z_vec = make_nelboiz_lazy_cls(hi1, approx, rnormMi, nljoint_args_indi, 
        ϕm, ϕqIc, θsP,
        cor_ends.M, transM, nljoint_args_fix,
        CA.getaxes(ϕqIc)[1], Val(ND),
        ) 
    if use_ϕm_vector
        grad_ϕm_ϕqI() = Vcat(similar(CA.getdata(ϕm)), similar(CA.getdata(ϕqIc))) 
        grad_conf_mat() = ForwardDiff.GradientConfig(nelboi_z_mat, CA.getdata(θsP), diffchunk)
        grad_ϕm_ϕqI1 = grad_ϕm_ϕqI()
        function grad_conf_vec() 
            cfg_vec = ForwardDiff.GradientConfig(nelboi_z_vec, grad_ϕm_ϕqI1, diffchunk)
            # similar Vcat returns a vector, need to convert dual of cfg to Vcat
            duals_vcat = Vcat(cfg_vec.duals[axes(grad_ϕm_ϕqI1.args[1],1)], cfg_vec.duals[axes(grad_ϕm_ϕqI1.args[2],1)]);
            T,V,N = typeof(cfg_vec).parameters[1:3]
            cfg = ForwardDiff.GradientConfig{T,V,N,typeof(duals_vcat)}(cfg_vec.seeds, duals_vcat)
            @assert cfg.duals isa Vcat
            cfg
        end
    else
        error("implement gradient config for matrix case")
    end
    get_helpers_worker = () -> begin
        (;
            # TODO remove inputs_cv and grads_v and test with @inferred
            inputs_cv = similar(cv_grad), # collecting inputs into single cv
            grads_v = similar(CA.getdata(cv_grad)), # buffer to store gradient
            #grad_conf = ForwardDiff.GradientConfig(nelboi_z, CA.getdata(cv_grad), diffchunk),
            grad_conf_vec = grad_conf_vec(), 
            grad_conf_mat = grad_conf_mat(),
            grad_θsP = similar(θsP),
            grad_ϕm_ϕqI = grad_ϕm_ϕqI(),
        )
    end
    h1 = get_helpers_worker()
    hw_channel = Channel{typeof(h1)}(n_workers) # workers + parallel threads
    put!(hw_channel, h1)
    foreach(2:n_workers) do _
        put!(hw_channel, get_helpers_worker())
    end 
    (;
        dϕg = similar(ϕg),
        ∂elbo_∂θP = similar(θsP),
        ∂elbo_∂ϕm_∂ζP = similar(θsP),
        ∂elbo_∂logσ_ζP = Vector{TF}(undef, size(θsP,1)),
        ∂elbo_∂θP_∂ζP = Matrix{TF}(undef, size(θsP)),
        dϕqP = similar(ϕqPc),
        dϕmvecs = SharedArrays.SharedArray{TF}(n_ϕmvec, n_site), 
        ∂elbo_∂ϕqm = Array{TF}(undef, size(ϕm)..., n_site),
        ax_θP = nljoint_args_fix.ax_θP,
        hw_channel,
        pullback_cl_sample_ζsP! = get_pullback_cl_sample_ζsP(CA.getdata(ϕqPc); 
            n_θP, n_MC, sample_buffers = h.sample_buffers),
        pullback_cl_transformζsP! = get_pullback_cl_transformζ!(CA.getdata(ϕqPc); n_θ = n_θP, n_MC),
        pullback_g_apply! = get_pullback_g_apply(
            ϕg, ϕqPc; n_θP, n_cov, n_covP, n_MC, n_site, n_M),
    )
end




