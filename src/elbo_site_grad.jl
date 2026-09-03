function grad_neg_elbo_sites(
    elbo_helpers::NamedTuple,      # tuple of preallocated arrays
    ϕg::AbstractVector{TG}, ϕqP::AbstractVector{TF}, ϕqI::AbstractVector{TF}, g, 
    pbm_covar_indices::Union{Nothing,AbstractVector{<:Number}}, 
    args...;
    n_MC=3, 
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
    #- sample_ζsP!(h.ζsP, h.logσ2_ζP, h.rnormP, ϕqPc) # n_P * n_MC
    pullback_cl_sample_ζsP! = primal_pullback_sample_ζsP!(
        h.ζsP, h.logσ2_ζP, h.rnormP, ϕqPc) 
    #
    ϕm_buffer_key = isnothing(pbm_covar_indices) ? :ϕms : :ϕms_mcs
    g_apply!(h[ϕm_buffer_key], ϕg, xM, h.ζsP, pbm_covar_indices, g, h.xMP) 
    #
    # compute the gradients of SL! using ForwardDiff
    function SL!(hi, ϕqIc, ϕm, ζsPvec, sizeζsP, template, args...;
            i_site_train, kwargs...
            )
            ζsM = PAT.get_tmp(hi.ζsM_dc, template)
            logσ2_ζM = PAT.get_tmp(hi.logσ2_ζM_dc, template)
            buffer_nθM = PAT.get_tmp(hi.buffer_nθM_dc, template)
            sample_ζsM!(ζsM, logσ2_ζM, hi.rnormM, ϕqIc, ϕm, buffer_nθM)
            # first component needs to be the full elbo
            ζsP = reshape(ζsPvec, sizeζsP) # view for plain arrays h.ζsP
            Lζi(ζsP, ζsM, logσ2_ζM, args...; i_site_train, kwargs...)
    end
    function forwarddiff_grad_SL!(hi, ϕm, ϕqIc, ζsP, args...; i_site_train, kwargs...)
        # aggregate all the derivatives to allow a single call to ForwardDiff.gradient
        #   reshape ζsP into a vector to avoid allocations in cv[Val(:ζsP)]
        inputs = CA.ComponentArray(; ϕqIc,  ϕm = ϕm, ζsPvec = vec(ζsP))
        grads = ForwardDiff.gradient(
            cv -> SL!(hi, cv[Val(:ϕqIc)], cv[Val(:ϕm)], cv[Val(:ζsPvec)], size(ζsP), 
            CA.getdata(cv), args...; i_site_train, kwargs...)[1], inputs)
        grads
    end

    ϕm_it = eachslice(h[ϕm_buffer_key]; dims = ndims(h[ϕm_buffer_key]))
    grads_ϕ = map((hi, i_site_train, ϕm) -> forwarddiff_grad_SL!(
        hi, ϕm, ϕqIc, h.ζsP, args...; i_site_train, kwargs...), 
        h.helpers_sites, i_sites_train, ϕm_it)
    # MAYBE avoid allocation by using a buffers
    ∂elbo_∂ϕqI = mapreduce( g -> g[Val(:ϕqIc)], +, grads_ϕ; 
        init = @view(grads_ϕ[1][Val(:ϕqIc)]).* 0)
    ∂elbo_∂ϕqm = mapreduce(g -> g[Val(:ϕm)], hcat, grads_ϕ;
        #init = @view(grads_ϕ[1][Val(:ϕm)]).* 0
        )::AbstractMatrix{TF}
    ∂elbo_∂ζsP = reshape(mapreduce(g -> g[Val(:ζsPvec)], +, grads_ϕ;
        init = @view(grads_ϕ[1][Val(:ζsPvec)]).* 0),
        size(h.ζsP))
    ∂elbo_∂logσ2_ζP = ones(length(h.logσ2_ζP)) # TODO update when properly computing elbo
    #
    # backpropagate gradients of ϕqm to h.dϕg and h.dζsP
    ϕm_ = copy(h[ϕm_buffer_key]) # primal actually not needed -> simplify pullback_g_apply!
    Enzyme.make_zero!(h.dϕg)
    Enzyme.make_zero!(h.∂elbo_∂ϕm_∂ζP)
    pullback_g_apply!(
        ϕm_, h.dϕg, h.∂elbo_∂ϕm_∂ζP, ∂elbo_∂ϕqm, 
        ϕg, xM, h.ζsP, pbm_covar_indices, g, h)
    #h.dϕg
    #h.∂elbo_∂ϕm_∂ζP
    #
    # dϕqP
    ∂elbo_∂logσ2_ζP_∂ϕqP = zero(ϕqPc)
    pullback_cl_sample_ζsP!(∂elbo_∂logσ2_ζP_∂ϕqP, h.∂elbo_∂ζP .* zero(TF), ∂elbo_∂logσ2_ζP)
    ∂elbo_∂ζP_∂ϕqP = zero(ϕqPc)
    pullback_cl_sample_ζsP!(∂elbo_∂ζP_∂ϕqP, h.∂elbo_∂ζP, ∂elbo_∂logσ2_ζP .* zero(TF))
    ∂elbo_∂ϕm_∂ζP_∂ϕqP = zero(ϕqPc)
    pullback_cl_sample_ζsP!(∂elbo_∂ϕm_∂ζP_∂ϕqP, h.∂elbo_∂ϕm_∂ζP, ∂elbo_∂logσ2_ζP .* zero(TF))
    dϕqP = ∂elbo_∂logσ2_ζP_∂ϕqP + ∂elbo_∂ζP_∂ϕqP + ∂elbo_∂ϕm_∂ζP_∂ϕqP
    tmp2 = zero(ϕqPc)
    pullback_cl_sample_ζsP!(tmp2, 
        h.∂elbo_∂ζP + h.∂elbo_∂ϕm_∂ζP, 
        ∂elbo_∂logσ2_ζP)

    # ∂ζsP∂ϕqc = zeros(eltype(ζsP), n_θP * n_MC, length(ϕqc))
    # n_θP, n_MC = size(h.ζsP)
    # ∂ζsP∂ϕqc = zeros(eltype(ζsP), n_θP * n_MC, length(ϕqc))
    # #pullback_sample_ζsP!(∂ζsP∂ϕqc, h.ζsP, h.logσ2_ζP, h.rnormP, ϕqPc)
    (;dϕqP, dϕqI = ∂elbo_∂ϕqI, dϕqg = h.dϕg)
end

function prepare_elbo_helpers(ϕg::AbstractArray{TG}, ϕqP::AbstractArray{TF};
    n_θP, n_θM, n_site, n_MC, n_cov, n_covP, n_M
    ) where {TG, TF}
    h = (;
        rnormP = Matrix{TF}(undef, n_θP, n_MC),
        ζsP = Matrix{TF}(undef, n_θP, n_MC),
        logσ2_ζP = Vector{TF}(undef, n_θP),
        ϕms = Matrix{TF}(undef, n_M, n_site),
        ϕms_mcs = Array{TF,3}(undef, n_M, n_MC, n_site),
        # ϕms = Matrix{TG}(undef, n_M, n_site),
        # ϕms_mcs = Array{TG,3}(undef, n_M, n_MC, n_site),
        xMP = Matrix{TG}(undef, (n_cov + n_covP), n_MC * n_site),
        dϕg = Vector{TG}(undef, length(ϕg)),
    )
    helpers_sites = Tuple((;
        rnormM = Matrix{TF}(undef, n_θM, n_MC),
        ζsM_dc = PAT.DiffCache(Matrix{TF}(undef, n_θM, n_MC)),
        logσ2_ζM_dc = PAT.DiffCache(Vector{TF}(undef, n_θM)),
        buffer_nθM_dc = PAT.DiffCache(Vector{TF}(undef, n_θM)),
    ) for i in 1:n_site)
    h = (;h... , helpers_sites, 
        dxMP  = zero(h.xMP),      # shadow for xMP
        dϕms = zero(h.ϕms),
        dϕms_mcs = zero(h.ϕms_mcs),
        ∂elbo_∂ζP = zero(h.ζsP),
        ∂elbo_∂ϕm_∂ζP = zero(h.ζsP),
    )   
end


function check_elbo_helpers(h::NamedTuple, xM::AbstractMatrix, pbm_covar_indices;
    n_ϕg
    )
    n_cov, n_site = size(xM)
    n_covP = isnothing(pbm_covar_indices) ? 0 : length(pbm_covar_indices)
    n_θP, n_MC = size(h.rnormP)
    n_M = size(h.ϕms, 1)
    @assert size(h.rnormP) == (n_θP, n_MC )
    @assert size(h.ζsP) == (n_θP, n_MC )
    @assert size(h.dϕg) == (n_ϕg,)
    @assert size(h.logσ2_ζP) == (n_θP,)
    @assert size(h.ϕms) == (n_M, n_site)
    @assert size(h.ϕms_mcs) == (n_M, n_MC, n_site)
    @assert size(h.xMP) == ((n_cov + n_covP), n_MC * n_site) 
    @assert size(h.dϕg) == (n_ϕg,)
    @assert size(h.dxMP) == size(h.xMP)
    @assert size(h.dϕms) == size(h.ϕms)
    @assert size(h.dϕms_mcs) == size(h.ϕms_mcs)
    @assert size(h.∂elbo_∂ζP) == size(h.ζsP)
    @assert size(h.∂elbo_∂ϕm_∂ζP) == size(h.ζsP)
    #
    @assert length(h.helpers_sites) == n_site
    hi = h.helpers_sites[1]
    n_θM = size(hi.rnormM, 1)
    @assert size(hi.rnormM) == (n_θM, n_MC)
    @assert size(hi.ζsM_dc.du) == (n_θM, n_MC)
    @assert size(hi.logσ2_ζM_dc.du) == (n_θM,)
    @assert size(hi.buffer_nθM_dc.du) == (n_θM,)
end

function primal_pullback_sample_ζsP!(ζsP, logσ2_ζP, rnormP, ϕqc)
    # ζsP is mutated, so store previous value
    fwd, rev = Enzyme.autodiff_thunk(
        Enzyme.ReverseSplitNoPrimal,
        Enzyme.Const{typeof(sample_ζsP!)},
        Enzyme.Const,                        # no scalar return
        Enzyme.Duplicated{typeof(ζsP)},      # mutated output (noise in, sample out)
        Enzyme.Duplicated{typeof(logσ2_ζP)}, # mutated intermediate buffer
        Enzyme.DuplicatedNoNeed{typeof(rnormP)},  
        Enzyme.Duplicated{typeof(ϕqc)}       # active input
    )    
    # # need to copy so that they are not modified in the reverse pass
    # ζsP_orig = copy(ζsP) 
    ζsP_ = copy(ζsP) 
    logσ2_ζP_ = copy(logσ2_ζP) 
    dζsP_ = zero(ζsP_)
    dlogσ2_ζP_ = zero(logσ2_ζP_)   
    drnormP = zero(rnormP)   
    args_fwd = ( # only allocate Duplicated once
        Enzyme.Const(sample_ζsP!),
        Enzyme.Duplicated(ζsP_, dζsP_),
        Enzyme.Duplicated(logσ2_ζP_, dlogσ2_ζP_),
        Enzyme.DuplicatedNoNeed(rnormP, drnormP),  
    )
    function pullback_cl_sample_ζsP!(dϕqPc, dζsP, dlogσ2_ζP)
        #copyto!(ζsP_, ζsP_orig)  # reset to initial random seed
        copyto!(dζsP_, dζsP)  # seed cotangent on ζsP
        copyto!(dlogσ2_ζP_, dlogσ2_ζP)  # seed cotangent on ζsP
        fill!(CA.getdata(dϕqPc), 0)
        fill!(drnormP, 0)
        Dup_ϕqc = Enzyme.Duplicated(ϕqc, dϕqPc)
        tape, _, _ = fwd( args_fwd...,  Dup_ϕqc)
        rev( args_fwd...,  Dup_ϕqc, tape )
        nothing
    end    
    # execute the primal function to update the results
    sample_ζsP!(ζsP, logσ2_ζP, rnormP, ϕqc)    
    return pullback_cl_sample_ζsP!
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

