struct DoubleMMCase <: AbstractHybridProblem end

const θP = CA.ComponentVector{Float32}(r0 = 0.3, K2 = 2.0)
const θM = CA.ComponentVector{Float32}(r1 = 0.5, K1 = 0.2)
const θall = vcat(θP, θM)

const θP_nor0 = θP[(:K2,)]
θP_nor0_K1 = θM[(:K1,)]
θM_nor0_K1 = vcat(θM[(:r1,)], θP[(:K2,)])


const xP_S1 = Float32[0.5, 0.5, 0.5, 0.5, 0.4, 0.3, 0.2, 0.1]
const xP_S2 = Float32[1.0, 3.0, 4.0, 5.0, 5.0, 5.0, 5.0, 5.0]

int_xP1 = ComponentArrayInterpreter(CA.ComponentVector(S1 = xP_S1, S2 = xP_S2))

# const transP = elementwise(exp)
# const transM = elementwise(exp)

# const transMS = Stacked(elementwise(identity), elementwise(exp))

const int_θdoubleMM = ComponentArrayInterpreter(flatten1(CA.ComponentVector(; θP, θM)))

"""
    f_doubleMM(θc::CA.ComponentVector{ET}, x) where ET

Example process based model (PBM) predicts a double-monod constrained rate
for different substrate concentration vectors, `x.S1`, and `x.S2` for a single site.
θc is a ComponentVector with scalar parameters as components: `r0`, `r1`, `K1`, and `K2`

It predicts a rate for each entry in concentrations:
`y = r0 .+ r1 .* x.S1 ./ (K1 .+ x.S1) .* x.S2 ./ (K2 .+ x.S2)`.

It is defined as 
```julia
function f_doubleMM(θc::ComponentVector{ET}, x) where ET
    # extract parameters not depending on order, i.e whether they are in θP or θM
    # r0 = θc[:r0]
    (r0, r1, K1, K2) = map((:r0, :r1, :K1, :K2)) do par
        getdata(θc[par])::ET
    end
    y = r0 .+ r1 .* x.S1 ./ (K1 .+ x.S1) .* x.S2 ./ (K2 .+ x.S2)
    return (y)
end
```
"""
function f_doubleMM(θc::CA.ComponentVector{ET}, x) where ET
    # extract parameters not depending on order, i.e whether they are in θP or θM
    GPUArraysCore.allowscalar() do # index to scalar parameter in parameter vector
        #θc = intθ1(θ)
        #using ComponentArrays: ComponentArrays as CA
        #r0, r1, K1, K2 = θc[(:r0, :r1, :K1, :K2)] # does not work on Zygote+GPU
        # (r0, r1, K1, K2) = map((:r0, :r1, :K1, :K2)) do par
        #     # vector will be repeated when broadcasted by a matrix
        #     CA.getdata(θc[par])::ET
        # end
        @unpack r0, r1, K1, K2 = θc
        # r0 = θc[:r0]
        # r1 = θc[:r1]
        # K1 = θc[:K1]
        # K2 = θc[:K2]
        y = r0 .+ r1 .* x.S1 ./ (K1 .+ x.S1) .* x.S2 ./ (K2 .+ x.S2)
        return (y)
    end
end

"""
    f_doubleMM_sites(θc::CA.ComponentMatrix, xPc::CA.ComponentMatrix)

Example process based model (PBM) that predicts for a batch of sites.

Arguments
- `θc`: parameters with one row per site and symbolic column index 
- `xPc`: model drivers with one column per site, and symbolic row index

Returns a matrix `(n_obs x n_site)` of predictions.

```julia
function f_doubleMM_sites(θc::ComponentMatrix, xPc::ComponentMatrix)
    # extract several covariates from xP
    ST = typeof(CA.getdata(xPc)[1:1,:])  # workaround for non-type-stable Symbol-indexing
    S1 = (CA.getdata(xPc[:S1,:])::ST)   
    S2 = (CA.getdata(xPc[:S2,:])::ST)
    #
    # extract the parameters as vectors that are row-repeated into a matrix
    VT = typeof(CA.getdata(θc)[:,1])   # workaround for non-type-stable Symbol-indexing
    n_obs = size(S1, 1)
    rep_fac = ones_similar_x(xPc, n_obs)      # to reshape into matrix, avoiding repeat
    (r0, r1, K1, K2) = map((:r0, :r1, :K1, :K2)) do par
        p1 = CA.getdata(θc[:, par]) ::VT
        #(r0 .* rep_fac)'    # move to computation below to save allocation
        #repeat(p1', n_obs)  # matrix: same for each concentration row in S1
    end
    #
    # each variable is a matrix (n_obs x n_site)
    #r0 .+ r1 .* S1 ./ (K1 .+ S1) .* S2 ./ (K2 .+ S2)
    (r0 .* rep_fac)' .+ (r1 .* rep_fac)' .* S1 ./ ((K1 .* rep_fac)' .+ S1) .* S2 ./ ((K2 .* rep_fac)' .+ S2)
end
```
"""
function f_doubleMM_sites(θc::CA.ComponentMatrix, xPc::CA.ComponentMatrix)
    # extract several covariates from xP
    # ST = typeof(CA.getdata(xPc)[1:1,:])  # workaround for non-type-stable Symbol-indexing
    # S1 = (CA.getdata(xPc[:S1,:])::ST)   
    # S2 = (CA.getdata(xPc[:S2,:])::ST)
    S1 = view(xPc, Val(:S1), :)
    S2 = view(xPc, Val(:S2), :)

    # S1 = @view CA.getdata(xPc[Val(:S1),:])
    # S2 = @view CA.getdata(xPc[Val(:S2),:])
    is_valid = isfinite.(S1) .&& isfinite.(S2)
    #
    # extract the parameters as vectors that are row-repeated into a matrix
    # VT = typeof(CA.getdata(θc)[:,1])   # workaround for non-type-stable Symbol-indexing
    # #n_obs = size(S1, 1)
    # #rep_fac = HVI.ones_similar_x(xPc, n_obs) # to reshape into matrix, avoiding repeat
    # #is_dummy = isnan.(S1) .|| isnan.(S2)
    
    # (r0, r1, K1, K2) = map((:r0, :r1, :K1, :K2)) do par
    #     p1 = CA.getdata(θc[:, par]) ::VT
    #     #Main.@infiltrate_main
    #     # tmp = Zygote.gradient(p1 -> sum(repeat_rowvector_dummy(p1', is_dummy)), p1)[1]
    #     #p1_mat = repeat_rowvector_dummy(p1', is_dummy)
    #     p1_mat = is_valid .* p1' # places zeros in dummy positions, prevents gradients there
    #     #repeat(p1', n_obs)  # matrix: same for each concentration row in S1
    #     #(rep_fac .* p1')    # move to computation below to save allocation
    # end
    #
    r0 = is_valid .* CA.getdata(θc[:, Val(:r0)])'
    r1 = is_valid .* CA.getdata(θc[:, Val(:r1)])'
    K1 = is_valid .* CA.getdata(θc[:, Val(:K1)])'
    K2 = is_valid .* CA.getdata(θc[:, Val(:K2)])'
    #
    #, r1, K1, K2) = map((:r0, :r1, :K1, :K2)) do par

    # each variable is a matrix (n_obs x n_site)
    r0 .+ r1 .* S1 ./ (K1 .+ S1) .* S2 ./ (K2 .+ S2)
    #(rep_fac .* r0') .+ (rep_fac .* r1') .* S1 ./ ((rep_fac .* K1') .+ S1) .* S2 ./ ((rep_fac .* K2') .+ S2)
end

# function f_doubleMM_sites(θc::CA.ComponentMatrix, xPc::CA.ComponentMatrix)
#     # extract the parameters as vectors
#     VT = typeof(CA.getdata(θc)[:,1])   # workaround for non-type-stable Symbol-indexing
#     (r0, r1, K1, K2) = map((:r0, :r1, :K1, :K2)) do par
#         CA.getdata(θc[:, par]) ::VT
#     end
#     #
#     # extract several covariates from xP
#     # S1 = (xPc[:S1,:])'   # transform site-last -> site-first dimension
#     # S2 = (xPc[:S2,:])'
#     #Main.@infiltrate_main

#     ST = typeof(CA.getdata(xPc)[1:1,:])  # workaround for non-type-stable Symbol-indexing
#     S1 = (CA.getdata(xPc[:S1,:])::ST)'   # transform site-last -> site-first dimension
#     S2 = (CA.getdata(xPc[:S2,:])::ST)'
#     #
#     y = r0 .+ r1 .* S1 ./ (K1 .+ S1) .* S2 ./ (K2 .+ S2)
#     return (CA.getdata(y)') # transform site-first -> site-last dimension
# end



# function f_doubleMM(
#         θ::AbstractMatrix{T}, x; intθ::HVI.AbstractComponentArrayInterpreter) where T
#     # provide θ for n_row sites
#     # provide x.S1 as Matrix n_site x n_obs
#     # extract parameters not depending on order, i.e whether they are in θP or θM
#     θc = intθ(θ)
#     @assert size(x.S1, 1) == size(θ, 1)  # same number of sites
#     @assert size(x.S1) == size(x.S2)   # same number of observations
#     #@assert length(x.s2 == n_obs)
#     # problems on AD on GPU with indexing CA may be related to printing result, use ";"
#     VT = typeof(θ[:,1])   # workaround for non-stable Symbol-indexing CAMatrix 
#     #VT = first(Base.return_types(getindex, Tuple{typeof(θ),typeof(Colon()),typeof(1)}))
#     (r0, r1, K1, K2) = map((:r0, :r1, :K1, :K2)) do par
#         # vector will be repeated when broadcasted by a matrix
#         CA.getdata(θc[:, par]) ::VT
#     end
#     # r0 = CA.getdata(θc[:,:r0])  # vector will be repeated when broadcasted by a matrix
#     # r1 = CA.getdata(θc[:,:r1])
#     # K1 = CA.getdata(θc[:,:K1])
#     # K2 = CA.getdata(θc[:,:K2])
#     y = r0 .+ r1 .* x.S1 ./ (K1 .+ x.S1) .* x.S2 ./ (K2 .+ x.S2)
#     return (y)
# end

# function f_doubleMM(θ::AbstractMatrix, x::NamedTuple, θpos::NamedTuple) 
#     # provide θ for n_row sites
#     # provide x.S1 as Matrix n_site x n_obs
#     # extract parameters not depending on order, i.e whether they are in θP or θM
#         @assert size(x.S1,1) == size(θ,1)  # same number of sites
#         @assert size(x.S1) == size(x.S2)   # same number of observations
#         (r0, r1, K1, K2) = map((:r0, :r1, :K1, :K2)) do par
#               # vector will be repeated when broadcasted by a matrix
#               CA.getdata(θ[:,θpos[par]])
#         end        
#         # r0 = CA.getdata(θ[:,θpos.r0])  # vector will be repeated when broadcasted by a matrix
#         # r1 = CA.getdata(θ[:,θpos.r1])
#         # K1 = CA.getdata(θ[:,θpos.K1])
#         # K2 = CA.getdata(θ[:,θpos.K2])
#         #y = r0 .+ r1
#         #y = x.S1 + x.S2
#         #y = (K1 .+ x.S1)
#         #y = r1 .* x.S1 ./ (K1 .+ x.S1) 
#         y = r0 .+ r1 .* x.S1 ./ (K1 .+ x.S1) .* x.S2 ./ (K2 .+ x.S2)
#         return (y)
# end

function HVI.get_hybridproblem_par_templates(
        ::DoubleMMCase; scenario::Val{scen}) where {scen}
    if (:no_globals ∈ scen)
        return ((; θP = CA.ComponentVector{Float32}(), θM))
    end
    if (:omit_r0 ∈ scen)
        #return ((; θP = θP_nor0, θM, θf = θP[(:K2r)]))
        if (:K1global ∈ scen)
            # scenario of K1 global but K2 site-dependent to inspect correlations^
            return ((; θP = θP_nor0_K1, θM = θM_nor0_K1))
        end
        return ((; θP = θP_nor0, θM))
    end
    #(; θP, θM, θf = eltype(θP)[])
    (; θP, θM)
end

# function HVI.get_hybridproblem_par_templates(::DoubleMMCase; scenario::NTuple = ())
#     if (:omit_r0 ∈ scenario)
#         #return ((; θP = θP_nor0, θM, θf = θP[(:K2r)]))
#         return ((; θP = θP_nor0, θM))
#     end
#     #(; θP, θM, θf = eltype(θP)[])
#     (; θP, θM)
# end

function HVI.get_hybridproblem_priors(::DoubleMMCase; scenario::Val{scen}) where {scen}
    Dict(keys(θall) .=> fit.(LogNormal, θall, QuantilePoint.(θall .* 3, 0.95), Val(:mode)))
end

function HVI.get_hybridproblem_MLapplicator(
        prob::HVI.DoubleMM.DoubleMMCase; scenario::Val{scen}) where {scen}
    rng = StableRNGs.StableRNG(111)
    get_hybridproblem_MLapplicator(rng, prob; scenario)
end

function HVI.get_hybridproblem_MLapplicator(
        rng::AbstractRNG, prob::HVI.DoubleMM.DoubleMMCase; scenario::Val{scen},
) where {scen}
    ml_engine = select_ml_engine(; scenario)
    g_nomag, ϕ_g0 = construct_3layer_MLApplicator(rng, prob, ml_engine; scenario)
    # construct normal distribution from quantiles at unconstrained scale
    priors_dict = get_hybridproblem_priors(prob; scenario)
    (; θM) = get_hybridproblem_par_templates(prob; scenario)
    priors = Tuple(priors_dict[k] for k in keys(θM))
    (; transM) = get_hybridproblem_transforms(prob; scenario)
    lowers, uppers = HVI.get_quantile_transformed(priors, transM)
    #n_site, n_batch = get_hybridproblem_n_site_and_batch(prob; scenario)
    g = if (:use_rangescaling ∈ scen)
        RangeScalingModelApplicator(g_nomag, lowers, uppers, eltype(ϕ_g0))
    else
        NormalScalingModelApplicator(g_nomag, lowers, uppers, eltype(ϕ_g0))
    end
    return g, ϕ_g0
end

function HVI.get_hybridproblem_pbmpar_covars(
        ::DoubleMMCase; scenario::Val{scen}) where {scen}
    if (:covarK2 ∈ scen)
        if (:K1global ∈ scen)
            return (:K1,)
        end
        return (:K2,)
    end
    ()
end

function HVI.get_hybridproblem_transforms(
        prob::DoubleMMCase; scenario::Val{scen}) where {scen}
    _θP, _θM = get_hybridproblem_par_templates(prob; scenario)
    if (:stackedMS ∈ scen)
        return (; transP = Stacked((HVI.Exp(),), (1:length(_θP),)),
            transM = Stacked((identity, HVI.Exp()), (1:1, 2:length(_θM))))
    elseif (:transIdent ∈ scen)
        # identity transformations, should AD on GPU
        return (; transP = Stacked((identity,), (1:length(_θP),)),
            transM = Stacked((identity,), (1:length(_θM),)))
    end
    (; transP = Stacked((HVI.Exp(),), (1:length(_θP),)),
        transM = Stacked((HVI.Exp(),), (1:length(_θM),)))
end

# function HVI.get_hybridproblem_sizes(::DoubleMMCase; scenario::Val{scen}) where scen
#     n_covar_pc = 2
#     n_covar = n_covar_pc + 3 # linear dependent
#     #n_site = 10^n_covar_pc
#     n_batch = 10
#     n_θM = length(θM)
#     n_θP = length(θP)
#     #(; n_covar, n_site, n_batch, n_θM, n_θP)
#     (; n_covar, n_batch, n_θM, n_θP)
# end

# defining the PBmodel as a closure with let leads to problems of JLD2 reloading
# Define all the variables additional to the ones passed curing the call by
# a dedicated Closure object and define the PBmodel as a callable
# struct DoubleMMCaller{CLT}
#     cl::CLT
# end

function HVI.get_hybridproblem_PBmodel(prob::DoubleMMCase; scenario::Val{scen}) where {scen}
    # θall defined in this module above
    # TODO check and test for population or sites, currently return only site specific
    pt = get_hybridproblem_par_templates(prob; scenario)
    keys_fixed = Tuple(k for k in setdiff(keys(θall), (keys(pt.θP)..., keys(pt.θM)...))) 
    θFix = isempty(keys_fixed) ? CA.ComponentVector{eltype(θall)}() : θall[keys_fixed]
    xPvec = int_xP1(vcat(xP_S1, xP_S2))
    if (:useSitePBM ∈ scen)
        PBMSiteApplicator(f_doubleMM; pt.θP, pt.θM, θFix, xPvec)
    else
        n_site, n_batch = get_hybridproblem_n_site_and_batch(prob; scenario)
        PBMPopulationApplicator(f_doubleMM_sites, n_batch; pt.θP, pt.θM, θFix, xPvec)
    end
end

# function(caller::DoubleMMCaller)(θP::AbstractVector, θMs::AbstractMatrix, xP)
#     cl = caller.cl
#     @assert size(xP, 2) == cl.n_site_batch
#     @assert size(θMs, 1) == cl.n_site_batch
#     # # convert vector of tuples to tuple of matricesByRows
#     # # need to supply xP as vectorOfTuples to work with DataLoader
#     # # k = first(keys(xP[1]))
#     # xPM = (; zip(keys(xP[1]), map(keys(xP[1])) do k
#     #     #stack(map(r -> r[k], xP))' 
#     #     stack(map(r -> r[k], xP); dims = 1)
#     # end)...)
#     #xPM = map(transpose, xPM1)
#     #xPc = int_xPb(CA.getdata(xP))
#     #xPM = (S1 = xPc[:,:S1], S2 = xPc[:,:S2]) # problems with Zygote
#     # make sure the same order of columns as in intθ
#     # reshape big matrix into NamedTuple of drivers S1 and S2 
#     #   for broadcasting need sites in rows
#     #xPM = map(p -> CA.getdata(xP[p,:])', pos_xP)get_hybridproblem_PBmodel
#     xPM = map(p -> CA.getdata(xP)'[:, p], cl.pos_xP)
#     θFixd = (θP isa GPUArraysCore.AbstractGPUVector) ? cl.θFix_dev : cl.θFix
#     θ = hcat(CA.getdata(θP[cl.isP]), CA.getdata(θMs), θFixd)
#     pred_sites = f_doubleMM(θ, xPM; cl.intθ)'
#     pred_global = eltype(pred_sites)[]
#     return pred_global, pred_sites
# end

function HVI.get_hybridproblem_neg_logden_obs(::DoubleMMCase; scenario::Val)
    neg_logden_indep_normal
end

# function HVI.get_hybridproblem_float_type(::DoubleMMCase; scenario)
#     return Float32
# end

# two observations more?
# const xP_S1 = Float32[0.5, 0.5, 0.5, 0.5, 0.5, 0.4, 0.3, 0.1]
# const xP_S2 = Float32[1.0, 2.0, 3.0, 4.0, 5.0, 5.0, 5.0, 5.0]

HVI.get_hybridproblem_n_covar(prob::DoubleMMCase; scenario::Val) = 5
function HVI.get_hybridproblem_n_site_and_batch(prob::DoubleMMCase;
        scenario::Val{scen}) where {scen}
    n_batch = 20
    n_site = 800
    if (:few_sites ∈ scen)
        n_site = 100
    elseif (:sites20 ∈ scen)
        n_site = 20
    end
    (n_site, n_batch)
end

function HVI.get_hybridproblem_train_dataloader(prob::DoubleMMCase; scenario::Val{scen},
        rng::AbstractRNG = StableRNG(111), kwargs...
) where {scen}
    n_site, n_batch = get_hybridproblem_n_site_and_batch(prob; scenario)
    if (:driverNAN ∈ scen)
        (; xM, xP, y_o, y_unc) = gen_hybridproblem_synthetic(rng, prob; scenario)
        n_site = size(xM,2)
        i_sites = 1:n_site
        # set the last two entries of the S1 drivers and observations of the second site NaN
        view(@view(xP[:S1,2]), 7:8) .= NaN
        y_o[7:8,2] .= NaN
        train_loader = MLUtils.DataLoader((CA.getdata(xM), CA.getdata(xP), y_o, y_unc, i_sites);
            batchsize = n_batch, partial = false)
    else
        construct_dataloader_from_synthetic(rng, prob; scenario, n_batch, kwargs...)
    end
end

function HVI.gen_hybridproblem_synthetic(rng::AbstractRNG, prob::DoubleMMCase;
        scenario::Val{scen}) where {scen}
    n_covar_pc = 2
    n_site, n_batch = get_hybridproblem_n_site_and_batch(prob; scenario)
    n_covar = get_hybridproblem_n_covar(prob; scenario)
    n_θM = length(θM)
    FloatType = get_hybridproblem_float_type(prob; scenario)
    xM, θMs_true0 = gen_cov_pred(rng, FloatType, n_covar_pc, n_covar, n_site, n_θM;
        rhodec = 8, is_using_dropout = false)
    int_θMs_sites = ComponentArrayInterpreter(θM, (n_site,))
    # normalize to be distributed around the prescribed true values
    θMs_true = int_θMs_sites(scale_centered_at(θMs_true0, θM, FloatType(0.1)))
    f_batch = get_hybridproblem_PBmodel(prob; scenario)
    f = create_nsite_applicator(f_batch, n_site)
    #xP = fill((; S1 = xP_S1, S2 = xP_S2), n_site)
    int_xP_sites = ComponentArrayInterpreter(int_xP1, (n_site,))
    xP = int_xP_sites(vcat(repeat(xP_S1, 1, n_site), repeat(xP_S2, 1, n_site)))
    #xP[:S1,:]
    #θP = get_θP(prob) # for DoubleMMCase par_templates gives correct θP
    θP = get_hybridproblem_θP(prob; scenario)
    y_true = f(θP, θMs_true', xP)
    σ_o = FloatType(0.01)
    #σ_o = FloatType(0.002)
    logσ2_o = FloatType(2) .* log.(σ_o)
    #σ_o = 0.002
    y_o = y_true .+ randn(rng, FloatType, size(y_true)) .* σ_o
    (;
        xM,
        θP_true = θP,
        θMs_true,
        xP,
        y_true,
        y_o,
        y_unc = fill(logσ2_o, size(y_o))
    )
end

function HVI.get_hybridproblem_cor_ends(prob::DoubleMMCase; scenario::Val{scen}) where {scen}
    pt = get_hybridproblem_par_templates(prob; scenario)
    if (:neglect_cor ∈ scen)
        # one block for each parameter, i.e. neglect all correlations
        (P = 1:length(pt.θP), M = 1:length(pt.θM))
    else 
        # single big blocks  
        (P = [length(pt.θP)], M = [length(pt.θM)])
    end
end

function HVI.get_hybridproblem_ϕq(prob::DoubleMMCase; scenario)
    FT = get_hybridproblem_float_type(prob; scenario) 
    cor_ends = get_hybridproblem_cor_ends(prob; scenario)
    ϕunc = init_hybrid_ϕunc(MeanHVIApproximationMat(), cor_ends, zero(FT))    
    # for DoubleMMCase templates gives the correct values
    θP = get_hybridproblem_par_templates(prob; scenario).θP
    transP = get_hybridproblem_transforms(prob; scenario).transP
    ϕq = HVI.update_μP_by_θP(ϕunc, θP, transP)
end



