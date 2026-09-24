function grad_neg_elbo_sites(
    elbo_helpers::NamedTuple,      # tuple of preallocated arrays
    grad_elbo_helpers::NamedTuple,  # tuple of preallocated arrays pullback closures
    rnormPM::NamedTuple,          # tuple of random numbers
    ϕg::AbstractVector{TG}, ϕqP::AbstractVector{TF}, ϕqI::AbstractVector{TF}, g, 
    pbm_covar_indices::Union{Nothing,AbstractVector{<:Number}}, 
    args...;
    i_sites_train,     # indices of sites in training set
    intϕqP, intϕqI,
    xM,
    is_testmode, 
    executor::Transducers.Executor = Transducers.SequentialEx(),
    kwargs...
) where {TG, TF}
    use_ϕm_matrix = isnothing(pbm_covar_indices)
    ϕqPc = intϕqP(ϕqP) 
    ϕqIc = intϕqI(ϕqI)
    h = elbo_helpers # preallocated μζP, dμζP, ζsP, ϕms, xMP, dxMP
    ϕm_buffer_key = use_ϕm_matrix ? :ϕms : :ϕms_mcs
    h_ϕm = h[ϕm_buffer_key]
    gradh = grad_elbo_helpers # here, as object, avoid closure so that can debug easier
    check_elbo_helpers(h, xM, pbm_covar_indices; n_ϕg = length(ϕg))
    check_gradelbo_helpers(gradh, ϕqI, h_ϕm, h.ζsP; n_ϕg = length(ϕg))
    sample_ζsP!(h.ζsP, h.logσ_ζP, rnormPM.P, ϕqPc) # n_P * n_MC
    g_apply!(h_ϕm, ϕg, xM, h.ζsP, pbm_covar_indices, g, h.xMP, is_testmode) 
    ladJacTP = transformζ(h.θsP, h.ζsP)  # return value captures ladJacT
    #
    # # compute the gradients of SL! using ForwardDiff
    # initialize GradientConfig of each element in the Channel
    hw_channel = gradh.hw_channel
    # hw_channel.n_avail_items
    if with_channel_element(x -> isnothing(x.grad_conf[]), hw_channel)
        nelboi_z = _make_nelboi_z_f(
            h.helpers_sites[1], rnormPM.M[1], i_sites_train[1];
            grad_ax = with_channel_element(hw_channel) do hwi
                hwi.grad_ax
            end)
        for i in 1:hw_channel.n_avail_items
            with_channel_element(hw_channel) do hwi
                hwi.grad_conf[] = ForwardDiff.GradientConfig(
                    nelboi_z, CA.getdata(hwi.cv_grad_nelboi), h.diffchunk)
            end
        end
    end
    #hw1 = take!(hw_channel); put!(hw_channel, hw1)
    cl = ForwardDiffGradNelboiZCl(ϕqIc, h.θsP, gradh.dϕmvecs, gradh.hw_channel)    
    ϕm_it = eachslice(h_ϕm; dims = ndims(h_ϕm))
    # let forwarddiff_grad_nelboi_z! directly write into array also in distributed
    #executor = Transducers.DistributedEx()
    # cannot avoid allocations in reducing function of mapreduce
    #    adding to SharedArrays dϕqIc and dθsPvec inside could lead to race conditions
    #    maybe let them store to preallocated SharedMatrix with site columns and sum after
    init = with_channel_element(hw_channel) do hwi 
        (;
        dϕqIc = zero(static_cv_getproperty(hwi.cv_grad_nelboi, Val(:ϕqIc))),
        dθsP = zero(static_cv_getproperty(hwi.cv_grad_nelboi, Val(:θsP)))
        )
    end
    reducer = make_tuple_reducer(+)
    #tmp = (@allocated reducer(init, init)) # check no allocations during reduction
    gacc = Folds.mapreduce(
        #forwarddiff_grad_nelboi_z_cl!, 
        cl,
        reducer, 
        zip(h.helpers_sites, rnormPM.M, i_sites_train, ϕm_it, axes(i_sites_train,1)),
        executor; init)
    # alloc_mapreduce = (@allocated Folds.mapreduce(
    #     (tup) -> forwarddiff_grad_nelboi_z!(tup...), 
    #     reducer, 
    #     zip(h.helpers_sites, gradh.helpers_sites, rnormPM.M, i_sites_train, ϕm_it, axes(i_sites_train,1)),
    #     executor; init))
    # @show alloc_mapreduce, alloc_mapreduce / length(rnormPM.M)
    # ∂elbo_∂ϕqI = view(gradh.gacc, Val(:ϕqIc))
    # ∂elbo_∂θP = reshape(view(gradh.gacc, Val(:θsPvec)), size(h.θsP))
    #∂elbo_∂ϕqm = reshape(view(gradh.gacc, Val(:ϕmsvec)), size(h_ϕm))
    ∂elbo_∂ϕqI = gacc.dϕqIc # tuple access
    ∂elbo_∂θP = gacc.dθsP
    ∂elbo_∂ϕqm = reshape(cl.dϕmvecs, size(h_ϕm))
    gradh.∂elbo_∂logσ_ζP .= -ones(TF, length(h.logσ_ζP))  
    ∂elbo_∂ladJacTP = -one(TF)
    #
    # pullback gradients of ϕqm -> gradh.dϕg and gradh.dζsP
    grad_elbo_helpers.pullback_g_apply!(
        gradh.dϕg, gradh.∂elbo_∂ϕm_∂ζP, h_ϕm, ∂elbo_∂ϕqm, 
        ϕg, xM, h.ζsP, pbm_covar_indices, g, is_testmode)
    #
    # pullback gradients of ∂elbo_∂θP to ∂elbo_∂θP_∂ζP
    ∂elbo_∂θP_∂ζP = Matrix{TF}(undef, size(∂elbo_∂θP)) # TODO avoid allocation
    grad_elbo_helpers.pullback_cl_transformζsP!(
        ∂elbo_∂θP_∂ζP, 
        ∂elbo_∂θP,
        ∂elbo_∂ladJacTP,
        h.θsP,
        h.ζsP,
        )
    #
    # pullback gradients of ∂elbo_∂ζP, ∂elbo_∂ϕm_∂ζP, and ∂elbo_∂logσ_ζP to dϕqP
    dϕqP = similar(ϕqPc) # TODO avoid allocation
    #pullback_sample_ζsP!(
    grad_elbo_helpers.pullback_cl_sample_ζsP!(
        dϕqP, 
        ∂elbo_∂θP_∂ζP + gradh.∂elbo_∂ϕm_∂ζP, gradh.∂elbo_∂logσ_ζP,
        h.ζsP, h.logσ_ζP, rnormPM.P, ϕqPc
        )

    # ∂ζsP∂ϕqc = zeros(eltype(ζsP), n_θP * n_MC, length(ϕqc))
    # n_θP, n_MC = size(h.ζsP)
    # ∂ζsP∂ϕqc = zeros(eltype(ζsP), n_θP * n_MC, length(ϕqc))
    # #pullback_sample_ζsP!(∂ζsP∂ϕqc, h.ζsP, h.logσ_ζP, h.rnormP, ϕqPc)
    (;dϕqP, dϕqI = ∂elbo_∂ϕqI, dϕg = gradh.dϕg)
end

"""
Callable to make deliver uncahged arguments to Foldl.mapreduce.
"""
struct ForwardDiffGradNelboiZCl{Tϕq, Tθ, TD, THWC}
    ϕqIc::Tϕq
    θsP::Tθ
    dϕmvecs::TD
    hw_channel::THWC
end
function (f::ForwardDiffGradNelboiZCl)(tup)
    hi, rnormM, i_site_train, ϕm, i = tup
    forwarddiff_grad_nelboi_z!(hi, rnormM, i_site_train, ϕm, i,
        f.ϕqIc, f.θsP, f.dϕmvecs; f.hw_channel)
end

function forwarddiff_grad_nelboi_z!(hi, rnormM, i_site_train, ϕm, i, 
    ϕqIc, θsP, dϕmvecs, omit_gradient=nothing; hw_channel) 
    # aggregate all the derivatives to allow a single call to ForwardDiff.gradient
    #   reshape ϕm and ζsP into a vector to avoid allocations in cv[Val(:ζsP)]
    # Use pre-extracted views to avoid wrapper allocations
    # hw_channel.n_avail_items
    with_channel_element(hw_channel) do hwi
        grad_conf = hwi.grad_conf[]
        inputs = hwi.cv_grad_nelboi
        grad_ax = hwi.grad_ax
        flat = CA.getdata(inputs) # flat backing storage, shares memory with inputs
        #inputs = gradhi.cv_grad_nelboi
        view(inputs, Val(:ϕqIc)) .= ϕqIc
        view(inputs, Val(:ϕm)) .= ϕm
        view(inputs, Val(:θsP)) .= θsP
        # supply something other than nothing to omit gradient to check allocations
        #grads = ForwardDiff.gradient(
        nelboi_z = _make_nelboi_z_f(hi, rnormM, i_site_train; grad_ax)
        grads_flat = !isnothing(omit_gradient) ? flat : ForwardDiff.gradient(
            nelboi_z, flat, grad_conf)
        # grads = !isnothing(omit_gradient) ? inputs : ForwardDiff.gradient(
        #     nelboi_z, inputs)
        # rebuild the ComponentArray as a zero-copy view of the flat partials
        grads = CA.ComponentArray(grads_flat, grad_ax)
        copyto!(view(dϕmvecs, :, i), view(grads, Val(:ϕm))) # second storage, leads to wrong results
        # returning SVector helps avoiding allocations during reduce
        (; dϕqIc = static_cv_getproperty(grads, Val(:ϕqIc)), 
            dθsP = static_cv_getproperty(grads, Val(:θsP)))
    end
end

function _make_nelboi_z_f(hi, rnormM, i_site_train; grad_ax)
    # the Flat Vector cv is the flat backing storage of a ComponentArray; rebuild the
    # ComponentArray as a zero-copy view so that `Val`-keyed access keeps working
    cv -> begin
        cv_ = CA.ComponentArray(cv, grad_ax)
        compute_nelboi_z!(
            hi, rnormM, i_site_train,
            view(cv_, Val(:ϕm)), view(cv_, Val(:ϕqIc)), view(cv_, Val(:θsP)),
        )[1]
    end
end

function pullback_sample_ζsP!(dϕqc, dζsP, dlogσ_ζP, ζsP, logσ_ζP, rnormP, ϕqc)
    ζsP_ = copy(ζsP) # TODO pass buffers to avoid allocation
    dζsP_ = copy(dζsP)
    logσ_ζP_ = copy(logσ_ζP) # TODO pass buffers to avoid allocation
    dlogσ_ζP_ = copy(dlogσ_ζP)
    drnormP = Enzyme.make_zero(rnormP)

    fill!(dϕqc, 0)
    Enzyme.autodiff(
        Enzyme.Reverse,
        sample_ζsP!,
        Enzyme.Duplicated(ζsP_, dζsP_),  
        Enzyme.Duplicated(logσ_ζP_, dlogσ_ζP_),  
        Enzyme.DuplicatedNoNeed(rnormP, drnormP),  
        Enzyme.Duplicated(ϕqc, dϕqc),   
    )
    nothing
end

function get_pullback_cl_sample_ζsP(::AbstractArray{TF}; n_θP, n_MC) where TF
    #dϕqc, dζsP, dlogσ_ζP, ζsP, logσ_ζP, rnormP, ϕqc)
    ζsP_ = Matrix{TF}(undef, n_θP, n_MC)
    logσ_ζP_ = Vector{TF}(undef, n_θP)
    dζsP_ = similar(ζsP_)  # allocate space for derivatives
    dlogσ_ζP_ = similar(logσ_ζP_)
    drnormP = similar(ζsP_)
    #
    function pullback_cl_sample_ζsP!(dϕqc, dζsP, dlogσ_ζP, ζsP, logσ_ζP, rnormP, ϕqc)
        Enzyme.make_zero!(dϕqc)
        fill!(dϕqc, 0)            # the derivative to compute 
        copyto!(dζsP_, dζsP)      # input cotangents (modified in-place by Enzyme)
        copyto!(dlogσ_ζP_, dlogσ_ζP)
        copyto!(ζsP_, ζsP)        # modified in-place by Enzyme
        copyto!(logσ_ζP_, logσ_ζP)
        Enzyme.autodiff(
            Enzyme.Reverse,
            sample_ζsP!,
            Enzyme.Duplicated(ζsP_, dζsP_),  
            Enzyme.Duplicated(logσ_ζP_, dlogσ_ζP_),  
            Enzyme.DuplicatedNoNeed(rnormP, drnormP),  
            Enzyme.Duplicated(ϕqc, dϕqc),   
        )
    end
end

# two return values: primal and derivative 
#   given coderiv dy, parameters and helpers
# dϕm -> dϕg, dζsP
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
        #
        fill!(dϕg,  zero(eltype(dϕg)))
        fill!(dζsP, zero(eltype(dζsP))) # also output cotangent
        fill!(dxMP_, zero(eltype(dxMP_)))
        # primal will be updated as in the forward, but shadow needs to be preserved
        copyto!(ϕms_buffer, ϕms) # copy to avoid modifying ϕm (although should be the same)
        copyto!(dϕms_buffer, dϕm) # copy to avoid modifying dϕm
        #copyto!(dζsP_, dζsP) # output, does not need to be preserved

        Enzyme.autodiff(
            Enzyme.Reverse, g_apply!,
            Enzyme.Duplicated(ϕms_buffer, dϕms_buffer),
            Enzyme.Duplicated(ϕg, dϕg),
            Enzyme.Const(xM),
            Enzyme.Duplicated(ζsP, dζsP),
            Enzyme.Const(pbm_covar_indices),
            Enzyme.Const(g),
            Enzyme.Duplicated(xMP_, dxMP_),
            Enzyme.Const(is_testmode))
    end
end

function get_pullback_cl_transformζ!(::AbstractArray{TF};  n_θ, n_MC) where {TF}
    θs_buffer = Matrix{TF}(undef, n_θ, n_MC)
    dθs_buffer = similar(θs_buffer)
    # Enzyme seeds an Active return value with one, so to seed its cotangent
    # with an arbitrary dladJacTP we fold it in as a scalar factor, relying on
    # linearity of the reverse-mode adjoint: the pullback then delivers
    # dladJacTP * ∂ladJacT/∂(·) to θs and ζs, as with the former Ref seeding.
    function pullback_cl_transformζ!(dζs, dθs, dladJacTP, θs, ζs)
        fill!(dζs, zero(eltype(dζs)))
        copyto!(θs_buffer, θs)
        copyto!(dθs_buffer, dθs)
        # Trick of seeding the active return value different to unity:
        # Here dladJacTP is captured from the closure (an Active-compatible scalar), 
        # the returned Active value's unit seed gets multiplied by dladJacTP, 
        # and the adjoints of θs/ζs come out as dladJacTP * ∂/∂(·)
        Enzyme.autodiff(
            Enzyme.Reverse,
            (θs_, ζs_) -> dladJacTP * transformζ(θs_, ζs_),
            Enzyme.Duplicated(θs_buffer, dθs_buffer),
            Enzyme.Duplicated(ζs, dζs),
        )
    end
end

function prepare_gradelbo_helpers(
    ϕg::AbstractVector{TG}, ϕqPc::AbstractVector{TF}, ϕqIc::AbstractVector{TF}; 
    n_θP, n_θM, n_MC, n_cov, pbm_covar_indices, n_site, n_M, n_workers,
    ) where {TG, TF}
    use_ϕm_matrix = isnothing(pbm_covar_indices)
    n_covP =  use_ϕm_matrix ? 0 : length(pbm_covar_indices)
    n_ϕmvec = use_ϕm_matrix ? n_M : n_M * n_MC
    # To sync across procs/threads, use a Channel 
    # https://juliafolds2.github.io/OhMyThreads.jl/stable/literate/tls/tls/#The-safe-way:-Channel
    get_helpers_worker = () -> begin
        cv_grad_nelboi = CA.ComponentArray(; 
            ϕqIc, 
            ϕm = use_ϕm_matrix ? Vector{TF}(undef, n_M) : Matrix{TF}(undef, n_M, n_MC), 
            θsP = Matrix{TF}(undef, n_θP, n_MC),
        )
        (;
            cv_grad_nelboi,
            # axis used to rebuild the ComponentArray as a zero-copy view of the
            # flat vector handed to ForwardDiff.gradient
            grad_ax = CA.getaxes(cv_grad_nelboi),
            # grad_conf = convert(Union{Base.RefValue{Nothing},Base.RefValue{ForwardDiff.GradientConfig}}, 
            #     Ref(nothing))::Union{Base.RefValue{Nothing},Base.RefValue{ForwardDiff.GradientConfig}}
            #Base.RefValue{Union{Nothing, <:ForwardDiff.GradientConfig}}(nothing),
            #grad_conf = grad_conf_n,
            grad_conf = Ref{Union{Nothing, ForwardDiff.GradientConfig}}(nothing)
        )
    end
    h1 = get_helpers_worker()
    hw_channel = Channel{typeof(h1)}(n_workers) # workers + parallel threads
    put!(hw_channel, h1)
    foreach(2:n_workers) do _
        put!(hw_channel, get_helpers_worker())
    end    
    (;
        dϕg = Vector{TG}(undef, length(ϕg)),
        #∂elbo_∂ζP = Matrix{TF}(undef, n_θP, n_MC),
        ∂elbo_∂ϕm_∂ζP = Matrix{TF}(undef, n_θP, n_MC),
        ∂elbo_∂logσ_ζP = Vector{TF}(undef, n_θP),
        #helpers_sites = map(x -> PAT.DiffCache(x), hi), his),
        dϕmvecs = SharedArrays.SharedArray{TF}(n_ϕmvec, n_site), 
        # helpers_sites = his,
        # helpers_workers = hws,
        hw_channel,
        #
        pullback_cl_sample_ζsP! = get_pullback_cl_sample_ζsP(CA.getdata(ϕqPc); n_θP, n_MC),
        pullback_cl_transformζsP! = get_pullback_cl_transformζ!(CA.getdata(ϕqPc); n_θ = n_θP, n_MC),
        pullback_g_apply! = get_pullback_g_apply(
            ϕg, ϕqPc; n_θP, n_cov, n_covP, n_MC, n_site, n_M),
    )
end

function check_gradelbo_helpers(gradh::NamedTuple, ϕqI, ϕms, ζsP;
    n_ϕg
    )
    # n_cov, n_site = size(xM)
    # n_covP = isnothing(pbm_covar_indices) ? 0 : length(pbm_covar_indices)
    n_site = size(ϕms)[end]
    n_ϕmvec= prod(size(ϕms)[1:(end-1)])
    n_θP, n_MC = size(gradh.∂elbo_∂ϕm_∂ζP)
    @assert size(gradh.dϕg) == (n_ϕg,)
    #@assert size(gradh.∂elbo_∂ζP) == (n_θP, n_MC)
    @assert size(gradh.∂elbo_∂ϕm_∂ζP) == (n_θP, n_MC)
    @assert size(gradh.∂elbo_∂logσ_ζP) == (n_θP,)
    @assert size(gradh.dϕmvecs) == (n_ϕmvec,n_site)
    with_channel_element(gradh.hw_channel) do hwi
        @assert size(hwi.cv_grad_nelboi.ϕqIc) == (length(ϕqI),)
        @assert size(hwi.cv_grad_nelboi.θsP) == size(ζsP)
        @assert size(hwi.cv_grad_nelboi.ϕm) == size(ϕms)[1:(end-1)]
        # @assert hwi.grad_conf isa Union{
        #     Base.RefValue{Nothing},Base.RefValue{ForwardDiff.GradientConfig}}
        @assert hwi.grad_conf isa Base.RefValue{Union{Nothing, ForwardDiff.GradientConfig}}
    end
end




