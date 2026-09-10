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
    kwargs...
) where {TG, TF}
    use_ϕm_matrix = isnothing(pbm_covar_indices)
    ϕm_buffer_key = use_ϕm_matrix ? :ϕms : :ϕms_mcs
    ϕqPc = intϕqP(ϕqP) 
    ϕqIc = intϕqI(ϕqI)

    h = elbo_helpers # preallocated μζP, dμζP, ζsP, ϕms, xMP, dxMP
    gradh = grad_elbo_helpers # here, as object, avoid closure so that can debug easier
    check_elbo_helpers(h, xM, pbm_covar_indices; n_ϕg = length(ϕg))
    check_gradelbo_helpers(gradh, ϕqI, h[ϕm_buffer_key], h.ζsP; n_ϕg = length(ϕg))
    sample_ζsP!(h.ζsP, h.logσ_ζP, rnormPM.P, ϕqPc) # n_P * n_MC
    g_apply!(h[ϕm_buffer_key], ϕg, xM, h.ζsP, pbm_covar_indices, g, h.xMP, is_testmode) 
    logdetTP = transformζ(h.θsP, h.ζsP)  # return value captures logdetT
    #
    # compute the gradients of SL! using ForwardDiff
    θsPvec = vec(h.θsP) # avoid putting entire h into closure
    sizeθsP = size(h.θsP)
    function forwarddiff_grad_elboi_z!(tup) # closure with ϕqIc, θsPvec, sizeθsP, kwargs
        hi, rnormM, i_site_train, ϕm = tup  # use tup to satisfy signature of mapfoldl
        # aggregate all the derivatives to allow a single call to ForwardDiff.gradient
        #   reshape ϕm and ζsP into a vector to avoid allocations in cv[Val(:ζsP)]
        #   TODO avoid allocation by buffer
        inputs = CA.ComponentArray(; ϕqIc, ϕmvec = vec(ϕm), θsPvec = θsPvec)
        grads = ForwardDiff.gradient(
            cv -> compute_elboi_z_vec!(
                hi, rnormM, i_site_train, cv[Val(:ϕmvec)], cv[Val(:ϕqIc)], cv[Val(:θsPvec)], 
                size(ϕm), sizeθsP; kwargs...
                )[1], inputs)
        grads
    end
    gradh.gacc.ϕqIc .= zero(TF) # accumulating + across mapfoldl
    gradh.gacc.θsPvec .= zero(TF)
    function get_onetime_reducer()
        local i_red = 1 
        function reducer(x,y) 
            x.ϕqIc += y.ϕqIc
            x.θsPvec += y.θsPvec
            x.ϕmsvec[:,i_red] .= y.ϕmvec
            i_red += 1   # captured in reducer closure, can only execute reducer once
            x
        end
    end
    ϕm_it = eachslice(h[ϕm_buffer_key]; dims = ndims(h[ϕm_buffer_key]))
    mapfoldl(forwarddiff_grad_elboi_z!, get_onetime_reducer(), 
        zip(h.helpers_sites, rnormPM.M, i_sites_train, ϕm_it);
        init = gradh.gacc
        )
    ∂elbo_∂ϕqI = view(gradh.gacc, Val(:ϕqIc))
    ∂elbo_∂ϕqm = reshape(view(gradh.gacc, Val(:ϕmsvec)), size(h[ϕm_buffer_key]))
    ∂elbo_∂θP = reshape(view(gradh.gacc, Val(:θsPvec)), size(h.θsP))
    gradh.∂elbo_∂logσ_ζP .= -ones(TF, length(h.logσ_ζP))  
    ∂elbo_∂logdetTP = +one(TF)
    #
    # pullback gradients of ϕqm -> gradh.dϕg and gradh.dζsP
    grad_elbo_helpers.pullback_g_apply!(
        gradh.dϕg, gradh.∂elbo_∂ϕm_∂ζP, h[ϕm_buffer_key], ∂elbo_∂ϕqm, 
        ϕg, xM, h.ζsP, pbm_covar_indices, g, is_testmode)
    #
    # pullback gradients of ∂elbo_∂θP to ∂elbo_∂θP_∂ζP
    ∂elbo_∂θP_∂ζP = similar(∂elbo_∂θP) # TODO avoid allocation
    grad_elbo_helpers.pullback_cl_transformζsP!(
        ∂elbo_∂θP_∂ζP, 
        ∂elbo_∂θP,
        ∂elbo_∂logdetTP,
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

function compute_elboi_z_vec!(hi, rnormM, i_site_train, ϕmvec, ϕqIc, θsPvec, 
    sizeϕm, sizeθsP; kwargs...
    ) 
    ϕm = reshape(ϕmvec, sizeϕm)
    θsP = reshape(θsPvec, sizeθsP) # view for plain arrays h.ζsP
    compute_elboi_z!(hi, rnormM, i_site_train, ϕm, ϕqIc, θsP; kwargs...) 
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
    # with an arbitrary dlogdetTP we fold it in as a scalar factor, relying on
    # linearity of the reverse-mode adjoint: the pullback then delivers
    # dlogdetTP * ∂logdetT/∂(·) to θs and ζs, as with the former Ref seeding.
    function pullback_cl_transformζ!(dζs, dθs, dlogdetTP, θs, ζs)
        fill!(dζs, zero(eltype(dζs)))
        copyto!(θs_buffer, θs)
        copyto!(dθs_buffer, dθs)
        # Trick of seeding the active return value different to unity:
        # Here dlogdetTP is captured from the closure (an Active-compatible scalar), 
        # the returned Active value's unit seed gets multiplied by dlogdetTP, 
        # and the adjoints of θs/ζs come out as dlogdetTP * ∂/∂(·)
        Enzyme.autodiff(
            Enzyme.Reverse,
            (θs_, ζs_) -> dlogdetTP * transformζ(θs_, ζs_),
            Enzyme.Duplicated(θs_buffer, dθs_buffer),
            Enzyme.Duplicated(ζs, dζs),
        )
    end
end

function prepare_gradelbo_helpers(
    ϕg::AbstractVector{TG}, ϕqPc::AbstractVector{TF}, ϕqIc::AbstractVector{TF}; 
    n_θP, n_θM, n_MC, n_cov, n_covP, n_site, n_M
    ) where {TG, TF}
    use_ϕm_matrix = (n_covP == 0)
    (;
        dϕg = Vector{TG}(undef, length(ϕg)),
        #∂elbo_∂ζP = Matrix{TF}(undef, n_θP, n_MC),
        ∂elbo_∂ϕm_∂ζP = Matrix{TF}(undef, n_θP, n_MC),
        ∂elbo_∂logσ_ζP = Vector{TF}(undef, n_θP),
        gacc = CA.ComponentVector( 
            ϕqIc = copy(ϕqIc), 
            ϕmsvec = use_ϕm_matrix ? Matrix{TF}(undef, n_M, n_site) :
                Matrix{TF}(undef, n_M * n_MC, n_site),
            θsPvec = Vector{TF}(undef, n_θM * n_MC)
        ),
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
    n_θP, n_MC = size(gradh.∂elbo_∂ϕm_∂ζP)
    @assert size(gradh.dϕg) == (n_ϕg,)
    #@assert size(gradh.∂elbo_∂ζP) == (n_θP, n_MC)
    @assert size(gradh.∂elbo_∂ϕm_∂ζP) == (n_θP, n_MC)
    @assert size(gradh.∂elbo_∂logσ_ζP) == (n_θP,)
    @assert size(gradh.gacc.ϕqIc) == (length(ϕqI),)
    @assert size(gradh.gacc.ϕmsvec,2) == n_site
    @assert length(gradh.gacc.ϕmsvec) == length(ϕms)
    @assert length(gradh.gacc.θsPvec) == length(ζsP)
end




