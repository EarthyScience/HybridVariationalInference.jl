function grad_neg_elbo_sites(
    elbo_helpers::NamedTuple,      # tuple of preallocated arrays
    pullbacks::NamedTuple,          # tuple of pullback closures
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
    h = elbo_helpers # preallocated μζP, dμζP, ζsP, ϕms, xMP, dxMP
    check_elbo_helpers(h, xM, pbm_covar_indices; n_ϕg = length(ϕg))
    ϕqPc = intϕqP(ϕqP) 
    ϕqIc = intϕqI(ϕqI)
    sample_ζsP!(h.ζsP, h.logσ2_ζP, rnormPM.P, ϕqPc) # n_P * n_MC
    ϕm_buffer_key = isnothing(pbm_covar_indices) ? :ϕms : :ϕms_mcs
    g_apply!(h[ϕm_buffer_key], ϕg, xM, h.ζsP, pbm_covar_indices, g, h.xMP, is_testmode) 
    #
    # compute the gradients of SL! using ForwardDiff
    function SL!(hi, ϕqIc, ϕm, ζsPvec, rnormM, sizeζsP, template, args...;
            i_site_train, kwargs...
            )
            ζsM = PAT.get_tmp(hi.ζsM_dc, template)
            logσ2_ζM = PAT.get_tmp(hi.logσ2_ζM_dc, template)
            buffer_nθM = PAT.get_tmp(hi.buffer_nθM_dc, template)
            sample_ζsM!(ζsM, logσ2_ζM, rnormM, ϕqIc, ϕm, buffer_nθM)
            # first component needs to be the full elbo
            ζsP = reshape(ζsPvec, sizeζsP) # view for plain arrays h.ζsP
            Lζi(ζsP, ζsM, logσ2_ζM, args...; i_site_train, kwargs...)
    end
    function forwarddiff_grad_SL!(hi, rnormM, i_site_train, ϕm)
        # aggregate all the derivatives to allow a single call to ForwardDiff.gradient
        #   reshape ζsP into a vector to avoid allocations in cv[Val(:ζsP)]
        #   TODO avoid allocation by buffer
        inputs = CA.ComponentArray(; ϕqIc, ϕm = ϕm, ζsPvec = vec(h.ζsP))
        grads = ForwardDiff.gradient(
            cv -> SL!(hi, cv[Val(:ϕqIc)], cv[Val(:ϕm)], cv[Val(:ζsPvec)], rnormM, size(h.ζsP), 
            CA.getdata(cv), args...; i_site_train, kwargs...)[1], inputs)
        grads
    end

    ϕm_it = eachslice(h[ϕm_buffer_key]; dims = ndims(h[ϕm_buffer_key]))
    grads_ϕ = map(forwarddiff_grad_SL!, h.helpers_sites, rnormPM.M, i_sites_train, ϕm_it)
    #
    # MAYBE avoid allocation by using a buffers
    ∂elbo_∂ϕqI = mapreduce( g -> g[Val(:ϕqIc)], +, grads_ϕ; 
        init = @view(grads_ϕ[1][Val(:ϕqIc)]).* 0)
    lcat = (x,y) -> cat(x,y; dims = ndims(h[ϕm_buffer_key]))
    ∂elbo_∂ϕqm = mapreduce(g -> g[Val(:ϕm)], lcat, grads_ϕ;
        #init = @view(grads_ϕ[1][Val(:ϕm)]).* 0
        )::typeof(h[ϕm_buffer_key])
    h.∂elbo_∂ζP .= reshape(mapreduce(g -> g[Val(:ζsPvec)], +, grads_ϕ;
        init = @view(grads_ϕ[1][Val(:ζsPvec)]).* 0),
        size(h.ζsP))
    ∂elbo_∂logσ2_ζP = ones(length(h.logσ2_ζP)) # TODO update when properly computing elbo
    #
    # pullback gradients of ϕqm -> h.dϕg and h.dζsP
    pullbacks.pullback_g_apply!(
        h[ϕm_buffer_key], h.dϕg, h.∂elbo_∂ϕm_∂ζP, ∂elbo_∂ϕqm, 
        ϕg, xM, h.ζsP, pbm_covar_indices, g, is_testmode)
    #
    # pullback gradients of ∂elbo_∂ζP, ∂elbo_∂ϕm_∂ζP, and ∂elbo_∂logσ2_ζP to dϕqP
    dϕqP = similar(ϕqPc)
    #pullback_sample_ζsP!(
    pullbacks.pullback_cl_sample_ζsP!(
        dϕqP, 
        h.∂elbo_∂ζP + h.∂elbo_∂ϕm_∂ζP, ∂elbo_∂logσ2_ζP,
        h.ζsP, h.logσ2_ζP, rnormPM.P, ϕqPc
        )

    # ∂ζsP∂ϕqc = zeros(eltype(ζsP), n_θP * n_MC, length(ϕqc))
    # n_θP, n_MC = size(h.ζsP)
    # ∂ζsP∂ϕqc = zeros(eltype(ζsP), n_θP * n_MC, length(ϕqc))
    # #pullback_sample_ζsP!(∂ζsP∂ϕqc, h.ζsP, h.logσ2_ζP, h.rnormP, ϕqPc)
    (;dϕqP, dϕqI = ∂elbo_∂ϕqI, dϕg = h.dϕg)
end

function pullback_sample_ζsP!(dϕqc, dζsP, dlogσ2_ζP, ζsP, logσ2_ζP, rnormP, ϕqc)
    ζsP_ = copy(ζsP) # TODO pass buffers to avoid allocation
    dζsP_ = copy(dζsP)
    logσ2_ζP_ = copy(logσ2_ζP) # TODO pass buffers to avoid allocation
    dlogσ2_ζP_ = copy(dlogσ2_ζP)
    drnormP = Enzyme.make_zero(rnormP)

    fill!(dϕqc, 0)
    Enzyme.autodiff(
        Enzyme.Reverse,
        sample_ζsP!,
        Enzyme.Duplicated(ζsP_, dζsP_),  
        Enzyme.Duplicated(logσ2_ζP_, dlogσ2_ζP_),  
        Enzyme.DuplicatedNoNeed(rnormP, drnormP),  
        Enzyme.Duplicated(ϕqc, dϕqc),   
    )
    nothing
end

function get_pullback_cl_sample_ζsP(::AbstractArray{TF}; n_θP, n_MC) where TF
    #dϕqc, dζsP, dlogσ2_ζP, ζsP, logσ2_ζP, rnormP, ϕqc)
    ζsP_ = Matrix{TF}(undef, n_θP, n_MC)
    logσ2_ζP_ = Vector{TF}(undef, n_θP)
    dζsP_ = similar(ζsP_)  # allocate space for derivatives
    dlogσ2_ζP_ = similar(logσ2_ζP_)
    drnormP = similar(ζsP_)
    #
    function pullback_cl_sample_ζsP!(dϕqc, dζsP, dlogσ2_ζP, ζsP, logσ2_ζP, rnormP, ϕqc)
        Enzyme.make_zero!(dϕqc)
        fill!(dϕqc, 0)            # the derivative to compute 
        copyto!(dζsP_, dζsP)      # input cotangents (modified in-place by Enzyme)
        copyto!(dlogσ2_ζP_, dlogσ2_ζP)
        copyto!(ζsP_, ζsP)        # modified in-place by Enzyme
        copyto!(logσ2_ζP_, logσ2_ζP)
        Enzyme.autodiff(
            Enzyme.Reverse,
            sample_ζsP!,
            Enzyme.Duplicated(ζsP_, dζsP_),  
            Enzyme.Duplicated(logσ2_ζP_, dlogσ2_ζP_),  
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
    function pullback_g_apply!(ϕms, dϕg, dζsP, dϕm, ϕg, xM, ζsP,
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

function prepare_pullbacks(ϕg, ϕqP; n_θP, n_MC, n_cov, n_covP, n_site, n_M)
    (;
        pullback_cl_sample_ζsP! = get_pullback_cl_sample_ζsP(ϕqP; n_θP, n_MC),
        pullback_g_apply! = get_pullback_g_apply(
            ϕg, ϕqP; n_θP, n_cov, n_covP, n_MC, n_site, n_M),
    )
end

