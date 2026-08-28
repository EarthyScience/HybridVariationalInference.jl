function grad_neg_elbo_sites(
    elbo_helpers::NamedTuple,      # tuple of preallocated arrays
    ϕg::AbstractVector{TG}, ϕq::AbstractVector{TF}, g, 
    pbm_covar_indices::Union{Nothing,AbstractVector{<:Number}}, 
    args...;
    n_MC=3, 
    i_sites_train,     # indices of sites in training set
    intϕq,
    xM,
    is_testmode, 
    kwargs...
) where {TG, TF}
    h = elbo_helpers
    ϕqc = intϕq(ϕq) 
    n_θP, n_MC = size(h.ζsP)
    ∂ζsP∂ϕqc = zeros(eltype(ζsP), n_θP * n_MC, length(ϕqc))
    pullback_sample_ζsP!(∂ζsP∂ϕqc, h.ζsP, ϕqc)
    ϕm_buffer_key = isnothing(pbm_covar_indices) ? :ϕms : :ϕms_mcs
    # can  only pull back after dϕm = computed dL/dϕm
    # pullback_g_apply!(h[ϕm_buffer_key], h.dϕg, h.dζsP, dϕm, ϕg, xM, ζsP, 
    #pbm_covar_indices, g, h)
    g_apply!(h[ϕm_buffer_key], ϕg, xM, h.ζsP, pbm_covar_indices, g, h.xMP) 
    function SL!(hi, i_site_train, ϕm) 
            #randn!(rng, hi.ζsM) # n_M * n_MC
            sample_ζsM!(hi.ζsM, ϕqc, ϕm)
            # first component needs to be the full elbo
            Lζi(h.ζsP, hi.ζsM, args...; i_site_train, kwargs...)
    end
    i = 1
    hi = h.helpers_sites[i]
    i_site_train  = i_sites_train[i]
    res = ForwardDiff.gradient(ϕm -> SL!(hi, i_site_train, ϕm), h[ϕm_buffer_key])
end

function primal_pullback_sample_ζsP!(ζsP, logσ2_ζP, ϕqc)
    # ζsP is mutated, so store previous value
    fwd, rev = Enzyme.autodiff_thunk(
        Enzyme.ReverseSplitNoPrimal,
        Enzyme.Const{typeof(sample_ζsP!)},
        Enzyme.Const,                        # no scalar return
        Enzyme.Duplicated{typeof(ζsP)},      # mutated output (noise in, sample out)
        Enzyme.Duplicated{typeof(logσ2_ζP)}, # mutated intermediate buffer
        Enzyme.Duplicated{typeof(ϕqc)}       # active input
    )    
    # need to copy so that they are not modified in the reverse pass
    ζsP_orig = copy(ζsP) 
    ζsP_ = copy(ζsP) 
    logσ2_ζP_ = copy(logσ2_ζP) 
    dζsP_ = zero(ζsP_)
    dlogσ2_ζP_ = zero(logσ2_ζP_)   
    args_fwd = ( # only allocate Duplicated once
        Enzyme.Const(sample_ζsP!),
        Enzyme.Duplicated(ζsP_, dζsP_),
        Enzyme.Duplicated(logσ2_ζP_, dlogσ2_ζP_),
    )
    function pullback_cl_sample_ζsP!(dϕqc, dζsP, dlogσ2_ζP)
        copyto!(ζsP_, ζsP_orig)  # reset to initial random seed
        copyto!(dζsP_, dζsP)  # seed cotangent on ζsP
        copyto!(dlogσ2_ζP_, dlogσ2_ζP)  # seed cotangent on ζsP
        fill!(CA.getdata(dϕqc), 0)
        Dup_ϕqc = Enzyme.Duplicated(ϕqc, dϕqc)
        tape, _, _ = fwd( args_fwd...,  Dup_ϕqc)
        rev( args_fwd...,  Dup_ϕqc, tape )
        nothing
    end    
    # execute the primal function to update the results
    sample_ζsP!(ζsP, logσ2_ζP, ϕqc)    
    return pullback_cl_sample_ζsP!
end

 function pullback_sample_ζsP!(dϕqc, dζsP, dlogσ2_ζP, ζsP, logσ2_ζP, ϕqc)
    ζsP_ = copy(ζsP) # TODO pass buffers to avoid allocation
    dζsP_ = copy(dζsP)
    logσ2_ζP_ = copy(logσ2_ζP) # TODO pass buffers to avoid allocation
    dlogσ2_ζP_ = copy(dlogσ2_ζP)
    fill!(dϕqc, 0)
    Enzyme.autodiff(
        Enzyme.Reverse,
        sample_ζsP!,
        Enzyme.Duplicated(ζsP_, dζsP_),  
        Enzyme.Duplicated(logσ2_ζP_, dlogσ2_ζP_),  
        Enzyme.Duplicated(ϕqc, dϕqc),   
    )
    nothing
end


# two return values: primal and derivative 
#   given coderiv dy, parameters and helpers
function pullback_g_apply!(ϕm, dϕg, dζsP, dϕm, ϕg, xM, ζsP, 
    pbm_covar_indices::Union{Nothing, AbstractArray}, g, h)
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
    )
end

