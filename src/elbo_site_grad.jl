function grad_neg_elbo_sites(
    elbo_helpers::NamedTuple,      # tuple of preallocated arrays
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
    #
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
    ∂elbo_∂ϕqm = mapreduce(g -> g[Val(:ϕm)], hcat, grads_ϕ;
        #init = @view(grads_ϕ[1][Val(:ϕm)]).* 0
        )::AbstractMatrix{TF}
    h.∂elbo_∂ζP .= reshape(mapreduce(g -> g[Val(:ζsPvec)], +, grads_ϕ;
        init = @view(grads_ϕ[1][Val(:ζsPvec)]).* 0),
        size(h.ζsP))
    ∂elbo_∂logσ2_ζP = ones(length(h.logσ2_ζP)) # TODO update when properly computing elbo
    #
    # pullback gradients of ϕqm to h.dϕg and h.dζsP
    ϕm_ = copy(h[ϕm_buffer_key]) # primal actually not needed -> simplify pullback_g_apply!
    Enzyme.make_zero!(h.dϕg)
    Enzyme.make_zero!(h.∂elbo_∂ϕm_∂ζP)
    pullback_g_apply!(
        ϕm_, h.dϕg, h.∂elbo_∂ϕm_∂ζP, ∂elbo_∂ϕqm, 
        ϕg, xM, h.ζsP, pbm_covar_indices, g, h, is_testmode)
    #
    # pullback gradients of ∂elbo_∂ζP, ∂elbo_∂ϕm_∂ζP, and ∂elbo_∂logσ2_ζP to dϕqP
    dϕqP = zero(ϕqPc)
    pullback_sample_ζsP!(dϕqP, 
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


# two return values: primal and derivative 
#   given coderiv dy, parameters and helpers
function pullback_g_apply!(ϕm, dϕg, dζsP, dϕm, ϕg, xM, ζsP, 
    pbm_covar_indices::Union{Nothing, AbstractArray}, g, h,
    is_testmode
    )
    # need to set shadows to zero before each gradient computation
    fill!(dϕg,     zero(eltype(dϕg)))
    fill!(dζsP,     zero(eltype(dζsP)))
    fill!(h.dxMP,  zero(eltype(h.dxMP)))
    dϕm_buffer_key = isnothing(pbm_covar_indices) ? :dϕms : :dϕms_mcs
    copyto!(h[dϕm_buffer_key], dϕm) # copy to avoid modifying dϕm
    # Enzyme already accumulates to dϕg
    Enzyme.autodiff(
        Enzyme.Reverse,
        g_apply!,
        Enzyme.Duplicated(ϕm, h[dϕm_buffer_key]),
        Enzyme.Duplicated(ϕg, dϕg),
        Enzyme.Const(xM),
        Enzyme.Duplicated(ζsP, dζsP),
        Enzyme.Const(pbm_covar_indices),
        Enzyme.Const(g),
        Enzyme.Duplicated(h.xMP, h.dxMP),
        Enzyme.Const(is_testmode),
    )
end

