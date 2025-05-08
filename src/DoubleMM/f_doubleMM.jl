struct DoubleMMCase <: AbstractHybridProblem end

const θP = CA.ComponentVector{Float32}(r0 = 0.3, K2 = 2.0)
const θM = CA.ComponentVector{Float32}(r1 = 0.5, K1 = 0.2)
const θall = vcat(θP, θM)

const θP_nor0 = θP[(:K2,)]

const xP_S1 = Float32[0.5, 0.5, 0.5, 0.5, 0.4, 0.3, 0.2, 0.1]
const xP_S2 = Float32[1.0, 3.0, 4.0, 5.0, 5.0, 5.0, 5.0, 5.0]

int_xP1 = ComponentArrayInterpreter(CA.ComponentVector(S1 = xP_S1, S2 = xP_S2))

# const transP = elementwise(exp)
# const transM = elementwise(exp)

# const transMS = Stacked(elementwise(identity), elementwise(exp))

const int_θdoubleMM = ComponentArrayInterpreter(flatten1(CA.ComponentVector(; θP, θM)))

function f_doubleMM(θ::AbstractVector, x; intθ1)
    # extract parameters not depending on order, i.e whether they are in θP or θM
    y = GPUArraysCore.allowscalar() do
        θc = intθ1(θ)
        #using ComponentArrays: ComponentArrays as CA
        #r0, r1, K1, K2 = θc[(:r0, :r1, :K1, :K2)] # does not work on Zygote+GPU
        (r0, r1, K1, K2) = map((:r0, :r1, :K1, :K2)) do par
            # vector will be repeated when broadcasted by a matrix
            CA.getdata(θc[par])
        end
        # r0 = θc[:r0]
        # r1 = θc[:r1]
        # K1 = θc[:K1]
        # K2 = θc[:K2]
        y = r0 .+ r1 .* x.S1 ./ (K1 .+ x.S1) .* x.S2 ./ (K2 .+ x.S2)
    end
    return (y)
end

function f_doubleMM(
        θ::AbstractMatrix{T}, x; intθ::HVI.AbstractComponentArrayInterpreter) where T
    # provide θ for n_row sites
    # provide x.S1 as Matrix n_site x n_obs
    # extract parameters not depending on order, i.e whether they are in θP or θM
    θc = intθ(θ)
    @assert size(x.S1, 1) == size(θ, 1)  # same number of sites
    @assert size(x.S1) == size(x.S2)   # same number of observations
    #@assert length(x.s2 == n_obs)
    # problems on AD on GPU with indexing CA may be related to printing result, use ";"
    VT = typeof(θ[:,1])   # workaround for non-stable Symbol-indexing CAMatrix 
    #VT = first(Base.return_types(getindex, Tuple{typeof(θ),typeof(Colon()),typeof(1)}))
    (r0, r1, K1, K2) = map((:r0, :r1, :K1, :K2)) do par
        # vector will be repeated when broadcasted by a matrix
        CA.getdata(θc[:, par]) ::VT
    end
    # r0 = CA.getdata(θc[:,:r0])  # vector will be repeated when broadcasted by a matrix
    # r1 = CA.getdata(θc[:,:r1])
    # K1 = CA.getdata(θc[:,:K1])
    # K2 = CA.getdata(θc[:,:K2])
    y = r0 .+ r1 .* x.S1 ./ (K1 .+ x.S1) .* x.S2 ./ (K2 .+ x.S2)
    return (y)
end

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
    if (:omit_r0 ∈ scen)
        #return ((; θP = θP_nor0, θM, θf = θP[(:K2r)]))
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
    Dict(keys(θall) .=> fit.(LogNormal, θall, QuantilePoint.(θall .* 3, 0.95)))
end

function HVI.get_hybridproblem_MLapplicator(
        prob::HVI.DoubleMM.DoubleMMCase; scenario::Val{scen}) where {scen}
    rng = StableRNGs.StableRNG(111)
    get_hybridproblem_MLapplicator(rng, prob; scenario)
end

function HVI.get_hybridproblem_MLapplicator(
        rng::AbstractRNG, prob::HVI.DoubleMM.DoubleMMCase; scenario::Val{scen},
        use_all_sites = false
) where {scen}
    ml_engine = select_ml_engine(; scenario)
    g_nomag, ϕ_g0 = construct_3layer_MLApplicator(rng, prob, ml_engine; scenario)
    # construct normal distribution from quantiles at unconstrained scale
    priors_dict = get_hybridproblem_priors(prob; scenario)
    (; θM) = get_hybridproblem_par_templates(prob; scenario)
    priors = [priors_dict[k] for k in keys(θM)]
    (; transM) = get_hybridproblem_transforms(prob; scenario)
    lowers, uppers = HVI.get_quantile_transformed(
        priors::AbstractVector{<:Distribution}, transM)
    n_site, n_batch = get_hybridproblem_n_site_and_batch(prob; scenario)
    n_site_batch = use_all_sites ? n_site : n_batch
    g = NormalScalingModelApplicator(
        g_nomag, lowers, uppers, eltype(ϕ_g0))
    return g, ϕ_g0
end

function HVI.get_hybridproblem_pbmpar_covars(
        ::DoubleMMCase; scenario::Val{scen}) where {scen}
    if (:covarK2 ∈ scen)
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

# function HVI.get_hybridproblem_PBmodel(prob::DoubleMMCase; scenario::NTuple = (),
#     gdev = :f_on_gpu ∈ scenario ? gpu_device() : identity, 
#     )
#     #fsite = (θ, x_site) -> f_doubleMM(θ)  # omit x_site drivers
#     par_templates = get_hybridproblem_par_templates(prob; scenario)
#     intθ, θFix = setup_PBMpar_interpreter(par_templates.θP, par_templates.θM, θall)
#     let θFix = gdev(θFix), intθ = get_concrete(intθ)
#         function f_doubleMM_with_global(θP::AbstractVector, θMs::AbstractMatrix, xP)
#             pred_sites = map_f_each_site(f_doubleMM, θMs, θP, θFix, xP, intθ)
#             pred_global = eltype(pred_sites)[]
#             return pred_global, pred_sites
#         end
#     end
# end

function HVI.get_hybridproblem_PBmodel(prob::DoubleMMCase; scenario::Val{scen},
        use_all_sites = false,
        gdev = :f_on_gpu ∈ HVI._val_value(scenario) ? gpu_device() : identity
) where {scen}
    n_site, n_batch = get_hybridproblem_n_site_and_batch(prob; scenario)
    n_site_batch = use_all_sites ? n_site : n_batch
    #fsite = (θ, x_site) -> f_doubleMM(θ)  # omit x_site drivers
    par_templates = get_hybridproblem_par_templates(prob; scenario)
    intθ1, θFix1 = setup_PBMpar_interpreter(par_templates.θP, par_templates.θM, θall)
    θFix = repeat(θFix1', n_site_batch)
    intθ = get_concrete(ComponentArrayInterpreter((n_site_batch,), intθ1))
    #int_xPb = ComponentArrayInterpreter((n_site_batch,), int_xP1)
    isP = repeat(axes(par_templates.θP, 1)', n_site_batch)
    let θFix = θFix, θFix_dev = gdev(θFix), intθ = get_concrete(intθ), isP = isP,
        n_site_batch = n_site_batch,
        #int_xPb=get_concrete(int_xPb),
        pos_xP = get_positions(int_xP1)

        function f_doubleMM_with_global(θP::AbstractVector, θMs::AbstractMatrix, xP)
            @assert size(xP, 2) == n_site_batch
            @assert size(θMs, 1) == n_site_batch
            # # convert vector of tuples to tuple of matricesByRows
            # # need to supply xP as vectorOfTuples to work with DataLoader
            # # k = first(keys(xP[1]))
            # xPM = (; zip(keys(xP[1]), map(keys(xP[1])) do k
            #     #stack(map(r -> r[k], xP))' 
            #     stack(map(r -> r[k], xP); dims = 1)
            # end)...)
            #xPM = map(transpose, xPM1)
            #xPc = int_xPb(CA.getdata(xP))
            #xPM = (S1 = xPc[:,:S1], S2 = xPc[:,:S2]) # problems with Zygote
            # make sure the same order of columns as in intθ
            # reshape big matrix into NamedTuple of drivers S1 and S2 
            #   for broadcasting need sites in rows
            #xPM = map(p -> CA.getdata(xP[p,:])', pos_xP)
            xPM = map(p -> CA.getdata(xP)'[:, p], pos_xP)
            θFixd = (θP isa GPUArraysCore.AbstractGPUVector) ? θFix_dev : θFix
            θ = hcat(CA.getdata(θP[isP]), CA.getdata(θMs), θFixd)
            pred_sites = f_doubleMM(θ, xPM; intθ)'
            pred_global = eltype(pred_sites)[]
            return pred_global, pred_sites
        end
        # function f_doubleMM_with_global(θP::AbstractVector, θMs::AbstractMatrix, xP)
        #     # TODO
        #     pred_sites = f_doubleMM(θMs, θP, θFix, xP, intθ)
        #     pred_global = eltype(pred_sites)[]
        #     return pred_global, pred_sites
        # end
    end
end

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

function HVI.get_hybridproblem_train_dataloader(prob::DoubleMMCase; scenario::Val,
        rng::AbstractRNG = StableRNG(111), kwargs...
)
    n_site, n_batch = get_hybridproblem_n_site_and_batch(prob; scenario)
    construct_dataloader_from_synthetic(rng, prob; scenario, n_batch, kwargs...)
end

function HVI.gen_hybridproblem_synthetic(rng::AbstractRNG, prob::DoubleMMCase;
        scenario::Val{scen}) where {scen}
    n_covar_pc = 2
    n_site, n_batch = get_hybridproblem_n_site_and_batch(prob; scenario)
    n_covar = get_hybridproblem_n_covar(prob; scenario)
    n_θM = length(θM)
    FloatType = get_hybridproblem_float_type(prob; scenario)
    par_templates = get_hybridproblem_par_templates(prob; scenario)
    #XXTODO transform θMs_true
    xM, θMs_true0 = gen_cov_pred(rng, FloatType, n_covar_pc, n_covar, n_site, n_θM;
        rhodec = 8, is_using_dropout = false)
    int_θMs_sites = ComponentArrayInterpreter(θM, (n_site,))
    # normalize to be distributed around the prescribed true values
    θMs_true = int_θMs_sites(scale_centered_at(θMs_true0, θM, FloatType(0.1)))
    f = get_hybridproblem_PBmodel(prob; scenario, gdev = identity, use_all_sites = true)
    #xP = fill((; S1 = xP_S1, S2 = xP_S2), n_site)
    int_xPn = ComponentArrayInterpreter(int_xP1, (n_site,))
    xP = int_xPn(vcat(repeat(xP_S1, 1, n_site), repeat(xP_S2, 1, n_site)))
    #xP[:S1,:]
    θP = par_templates.θP
    #θint = ComponentArrayInterpreter( (size(θMs_true,2),), CA.getaxes(vcat(θP, θMs_true[:,1])))
    y_global_true, y_true = f(θP, θMs_true', xP) 
    σ_o = FloatType(0.01)
    #σ_o = FloatType(0.002)
    logσ2_o = FloatType(2) .* log.(σ_o)
    #σ_o = 0.002
    y_global_o = y_global_true .+ randn(rng, FloatType, size(y_global_true)) .* σ_o
    y_o = y_true .+ randn(rng, FloatType, size(y_true)) .* σ_o
    (;
        xM,
        θP_true = θP,
        θMs_true,
        xP,
        y_global_true,
        y_true,
        y_global_o,
        y_o,
        y_unc = fill(logσ2_o, size(y_o))
    )
end

