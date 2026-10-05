function grad_neg_elbo_sites(
    elbo_helpers::NamedTuple,      # tuple of preallocated arrays
    grad_elbo_helpers::NamedTuple,  # tuple of preallocated arrays pullback closures
    approx::AbstractHVIApproximation,
    rnormPM::NamedTuple,          # tuple of random numbers
    ϕg::AbstractVector{TG}, ϕqP::AbstractVector{TF}, ϕqI::AbstractVector{TF}, g, 
    pbm_covar_indices::Union{Nothing,AbstractVector{<:Number}}, 
    args...;
    i_sites_train,     # indices of sites in training set
    intϕqP, intϕqI,
    xM,
    cor_ends,
    transP::Stacked, transM::Stacked,
    is_testmode, 
    n_workers = 1, # TODO compute from executor and Distributed.nworkers and nthreads,...
    executor::Transducers.Executor = Transducers.SequentialEx(),
    kwargs...
) where {TG, TF}
    use_ϕm_matrix = isnothing(pbm_covar_indices)
    ϕqPc = intϕqP(ϕqP) 
    ϕqIc = intϕqI(ϕqI)
    h = elbo_helpers # preallocated μζP, dμζP, ζsP, ϕms, xMP, dxMP
    ϕm_buffer_key = use_ϕm_matrix ? :ϕms : :ϕms_mcs
    h_ϕm = h[ϕm_buffer_key]
    check_elbo_helpers(h, xM, pbm_covar_indices; n_ϕg = length(ϕg))
    n_cov, n_site = size(xM)
    n_θP, n_MC = size(h.ζsP)
    gradh = !isempty(grad_elbo_helpers) ? grad_elbo_helpers : prepare_gradelbo_helpers(
        ϕqIc, selectdim(h_ϕm, ndims(h_ϕm), 1), h.ζsP, ϕg, ϕqPc, approx; 
        pbm_covar_indices, n_workers,
        h, rnormM1 = rnormPM.M[1], i_site_train1 = 1,
        h.diffchunk, n_site, n_cov, cor_ends, transP, transM)
    hw_channel = gradh.hw_channel
    #
    sample_ζsP!(h.ζsP, h.logσ_ζP, approx, rnormPM.P, ϕqPc, cor_ends.P, h.sample_buffers) # n_P * n_MC
    g_apply!(h_ϕm, ϕg, xM, h.ζsP, pbm_covar_indices, g, h.xMP, is_testmode) 
    ladJacTP = transformζ!(h.θsP, transP, h.ζsP)  # return value captures ladJacT
    #
    # parallel ForwardDiffGradient through forwarddiff_grad_nelboi_z!
    cl = ForwardDiffGradNelboiZCl(approx, ϕqIc, h.θsP, gradh.dϕmvecs, gradh.hw_channel, cor_ends.M, transM)    
    ϕm_it = eachslice(h_ϕm; dims = ndims(h_ϕm))
    init = with_channel_element(hw_channel) do hwi 
        (; dϕqIc = zero(static_cv_getproperty(hwi.inputs_cv, Val(:ϕqIc))),
        dθsP = zero(static_cv_getproperty(hwi.inputs_cv, Val(:θsP))))
    end
    gacc = Folds.mapreduce(cl, make_tuple_reducer(+), 
        zip(h.helpers_sites, rnormPM.M, i_sites_train, ϕm_it, axes(i_sites_train,1)),
        executor; init)
    ∂elbo_∂ϕqI = gacc.dϕqIc # tuple access
    ∂elbo_∂θP = gacc.dθsP
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
struct ForwardDiffGradNelboiZCl{TA, Tϕq, Tθ, TD, THWC, TC, TM}
    approx::TA
    ϕqIc::Tϕq
    θsP::Tθ
    dϕmvecs::TD
    hw_channel::THWC
    corendsM::TC
    transM::TM
end
function (f::ForwardDiffGradNelboiZCl)(tup)
    hi, rnormM, i_site_train, ϕm, i = tup
    forwarddiff_grad_nelboi_z!(hi, f.approx, rnormM, i_site_train, ϕm, i,
        f.ϕqIc, f.θsP, f.dϕmvecs, f.hw_channel, f.corendsM, f.transM, nothing)
end

function forwarddiff_grad_nelboi_z!(hi, approx::AbstractHVIApproximation, rnormM, i_site_train, ϕm, i, 
    ϕqIc, θsP, dϕmvecs, hw_channel::Channel, cor_endsM, transM::Stacked, omit_gradient=nothing) 
    # aggregate all the derivatives to allow a single call to ForwardDiff.gradient
    #   reshape ϕm and ζsP into a vector to avoid allocations in cv[Val(:ζsP)]
    with_channel_element(hw_channel) do hwi
    #local hwi = take!(hw_channel)
        grad_conf = hwi.grad_conf
        inputs_cv = hwi.inputs_cv
        #grad_ax = hwi.grad_ax
        inputs_v = CA.getdata(inputs_cv) # flat backing storage, shares memory with inputs
        #inputs = gradhi.inputs_cv
        view(inputs_cv, Val(:ϕqIc)) .= ϕqIc
        view(inputs_cv, Val(:ϕm)) .= ϕm
        view(inputs_cv, Val(:θsP)) .= θsP
        nelboi_z = make_nelboiz_cl(hi, approx, rnormM, i_site_train, CA.getaxes(inputs_cv), cor_endsM, transM)
        # write the gradient into the preallocated per-worker buffer to avoid
        # the result-vector allocation in ForwardDiff.gradient
        grads_flat = if isnothing(omit_gradient)
            ForwardDiff.gradient!(hwi.grads_v, nelboi_z, inputs_v, grad_conf)
            hwi.grads_v
        else
            inputs_v
        end
        # grads = !isnothing(omit_gradient) ? inputs : ForwardDiff.gradient(
        #     nelboi_z, inputs)
        # rebuild the ComponentArray as a zero-copy view of the flat partials
        grads = CA.ComponentArray(grads_flat, CA.getaxes(inputs_cv))
        copyto!(view(dϕmvecs, :, i), view(grads, Val(:ϕm))) # second storage, leads to wrong results
        # returning SVector helps avoiding allocations during reduce
        res = (; dϕqIc = static_cv_getproperty(grads, Val(:ϕqIc)), 
            dθsP = static_cv_getproperty(grads, Val(:θsP)))
    end
    #put!(hw_channel, hwi) 
    #res
end

function make_nelboiz_cl(hi, approx::AbstractHVIApproximation, rnormM, i_site_train, ax_inputs, cor_endsM, transM)
    function nelboiz_cl(cv) 
        cv_ = CA.ComponentArray(cv, ax_inputs)
        compute_nelboi_z!(
            hi, approx, rnormM, i_site_train,
            view(cv_, Val(:ϕm)), view(cv_, Val(:ϕqIc)), view(cv_, Val(:θsP)), cor_endsM, transM,
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
    ϕqIc, ϕm, θsP,
    ϕg::AbstractVector{TG}, ϕqPc::AbstractVector{TF},
    approx::AbstractHVIApproximation; 
    pbm_covar_indices, n_workers,
    h, rnormM1, i_site_train1,
    diffchunk, n_site, n_cov,
    cor_ends,
    transP::Stacked, transM::Stacked,
    ) where {TG, TF}
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
    use_ϕm_matrix = isnothing(pbm_covar_indices)
    n_covP =  use_ϕm_matrix ? 0 : length(pbm_covar_indices)
    n_ϕmvec = length(ϕm) #use_ϕm_matrix ? n_M : n_M * n_MC
    n_θP, n_MC = size(θsP)
    n_M = size(ϕm,1)
    # To sync across procs/threads, use a Channel 
    # https://juliafolds2.github.io/OhMyThreads.jl/stable/literate/tls/tls/#The-safe-way:-Channel
    #    
    nelboi_z = make_nelboiz_cl(hi1, approx, rnormM1, i_site_train1, CA.getaxes(cv_grad), 
        cor_ends.M, transM) 
    get_helpers_worker = () -> begin
        (;
            inputs_cv = similar(cv_grad), # collecting inputs into single cv
            grads_v = similar(CA.getdata(cv_grad)), # buffer to store gradient
            grad_conf = ForwardDiff.GradientConfig(nelboi_z, CA.getdata(cv_grad), diffchunk)
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
        ∂elbo_∂ϕm_∂ζP = similar(θsP),
        ∂elbo_∂logσ_ζP = Vector{TF}(undef, size(θsP,1)),
        ∂elbo_∂θP_∂ζP = Matrix{TF}(undef, size(θsP)),
        dϕqP = similar(ϕqPc),
        dϕmvecs = SharedArrays.SharedArray{TF}(n_ϕmvec, n_site), 
        ∂elbo_∂ϕqm = Array{TF}(undef, size(ϕm)..., n_site),
        hw_channel,
        pullback_cl_sample_ζsP! = get_pullback_cl_sample_ζsP(CA.getdata(ϕqPc); 
            n_θP, n_MC, sample_buffers = h.sample_buffers),
        pullback_cl_transformζsP! = get_pullback_cl_transformζ!(CA.getdata(ϕqPc); n_θ = n_θP, n_MC),
        pullback_g_apply! = get_pullback_g_apply(
            ϕg, ϕqPc; n_θP, n_cov, n_covP, n_MC, n_site, n_M),
    )
end



