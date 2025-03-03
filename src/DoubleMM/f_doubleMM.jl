struct DoubleMMCase <: AbstractHybridProblem end

const θP = CA.ComponentVector{Float32}(r0 = 0.3, K2 = 2.0)
const θM = CA.ComponentVector{Float32}(r1 = 0.5, K1 = 0.2)
const θall = vcat(θP, θM)

const θP_nor0 = θP[(:K2,)]

const transP = elementwise(exp)
const transM = elementwise(exp)

const transMS = Stacked(elementwise(identity), elementwise(exp))

const int_θdoubleMM = ComponentArrayInterpreter(flatten1(CA.ComponentVector(; θP, θM)))

function f_doubleMM(θ::AbstractVector, x)
    # extract parameters not depending on order, i.e whether they are in θP or θM
    θc = int_θdoubleMM(θ)
    r0, r1, K1, K2 = θc[(:r0, :r1, :K1, :K2)]
    y = r0 .+ r1 .* x.S1 ./ (K1 .+ x.S1) .* x.S2 ./ (K2 .+ x.S2)
    return (y)
end

function HVI.get_hybridproblem_par_templates(::DoubleMMCase; scenario::NTuple = ())
    if (:omit_r0 ∈ scenario)
        return ((; θP = θP_nor0, θM))
    end
    (; θP, θM)
end

function HVI.get_hybridproblem_priors(::DoubleMMCase; scenario = ())
    Dict( keys(θall) .=> fit.(LogNormal, θall, QuantilePoint.(θall .* 3, 0.95)))
end


function HVI.get_hybridproblem_MLapplicator(
        rng::AbstractRNG, prob::HVI.DoubleMM.DoubleMMCase; scenario = ())
    ml_engine = select_ml_engine(; scenario)
    g_nomag, ϕ_g0 = construct_3layer_MLApplicator(rng, prob, ml_engine; scenario)
    # multiply by inverse of initial estimate on unconstrained scale
    # ζM_cpu = inverse(transM)(θM)
    # ζM = (:use_Flux ∈ scenario) ? CuArray(ζM_cpu) : ζM_cpu
    # g = MagnitudeModelApplicator(g_nomag, ζM)
    # construct normal distribution from quantiles at unconstrained scale
    priors_dict = get_hybridproblem_priors(prob; scenario)
    priors = [priors_dict[k] for k in keys(θM)]
    (;transM) = get_hybridproblem_transforms(prob; scenario)
    g = NormalScalingModelApplicator(g_nomag, priors, transM, eltype(ϕ_g0))
    return g, ϕ_g0
end



function HVI.get_hybridproblem_transforms(::DoubleMMCase; scenario::NTuple = ())
    if (:stackedMS ∈ scenario)
        return ((; transP, transM = transMS))
    end
    (; transP, transM)
end

# function HVI.get_hybridproblem_sizes(::DoubleMMCase; scenario = ())
#     n_covar_pc = 2
#     n_covar = n_covar_pc + 3 # linear dependent
#     #n_site = 10^n_covar_pc
#     n_batch = 10
#     n_θM = length(θM)
#     n_θP = length(θP)
#     #(; n_covar, n_site, n_batch, n_θM, n_θP)
#     (; n_covar, n_batch, n_θM, n_θP)
# end

function HVI.get_hybridproblem_PBmodel(prob::DoubleMMCase; scenario::NTuple = ())
    #fsite = (θ, x_site) -> f_doubleMM(θ)  # omit x_site drivers
    par_templates = get_hybridproblem_par_templates(prob; scenario)
    keys_fixed = ((k for k in keys(θall) if
    (k ∉ keys(par_templates.θP)) & (k ∉ keys(par_templates.θM)))...,)
    let θFix = θall[keys_fixed]
        function f_doubleMM_with_global(θP::AbstractVector, θMs::AbstractMatrix, x)
            pred_sites = applyf(f_doubleMM, θMs, θP, θFix, x)
            pred_global = eltype(pred_sites)[]
            return pred_global, pred_sites
        end
    end
end

function HVI.get_hybridproblem_neg_logden_obs(::DoubleMMCase; scenario::NTuple = ())
    neg_logden_indep_normal
end

# function HVI.get_hybridproblem_float_type(::DoubleMMCase; scenario)
#     return Float32
# end

const xP_S1 = Float32[0.5, 0.5, 0.5, 0.5, 0.4, 0.3, 0.2, 0.1]
const xP_S2 = Float32[1.0, 3.0, 4.0, 5.0, 5.0, 5.0, 5.0, 5.0]

# two observations more?
# const xP_S1 = Float32[0.5, 0.5, 0.5, 0.5, 0.5, 0.4, 0.3, 0.1]
# const xP_S2 = Float32[1.0, 2.0, 3.0, 4.0, 5.0, 5.0, 5.0, 5.0]

function HVI.gen_hybridcase_synthetic(rng::AbstractRNG, prob::DoubleMMCase;
        scenario = ())
    n_covar_pc = 2
    n_site = 800
    n_covar = 5
    n_θM = length(θM)
    FloatType = get_hybridproblem_float_type(prob; scenario)
    par_templates = get_hybridproblem_par_templates(prob; scenario)
    xM, θMs_true0 = gen_cov_pred(rng, FloatType, n_covar_pc, n_covar, n_site, n_θM;
        rhodec = 8, is_using_dropout = false)
    int_θMs_sites = ComponentArrayInterpreter(θM, (n_site,))
    # normalize to be distributed around the prescribed true values
    θMs_true = int_θMs_sites(scale_centered_at(θMs_true0, θM, FloatType(0.1)))
    f = get_hybridproblem_PBmodel(prob; scenario)
    xP = fill((; S1 = xP_S1, S2 = xP_S2), n_site)
    θP = par_templates.θP
    y_global_true, y_true = f(θP, θMs_true, xP)
    σ_o = FloatType(0.01)
    #σ_o = FloatType(0.002)
    logσ2_o = FloatType(2) .* log.(σ_o)
    #σ_o = 0.002
    y_global_o = y_global_true .+ randn(rng, FloatType, size(y_global_true)) .* σ_o
    y_o = y_true .+ randn(rng, FloatType, size(y_true)) .* σ_o
    (;
        xM,
        n_site,
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
