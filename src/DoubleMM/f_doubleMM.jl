struct DoubleMMCase <: AbstractHybridProblem end


θP = CA.ComponentVector{Float32}(r0 = 0.3, K2 = 2.0)
θM = CA.ComponentVector{Float32}(r1 = 0.5, K1 = 0.2)

transP = elementwise(exp)
transM = Stacked(elementwise(identity), elementwise(exp))


const int_θdoubleMM = ComponentArrayInterpreter(flatten1(CA.ComponentVector(; θP, θM)))

function f_doubleMM(θ::AbstractVector, x)
    # extract parameters not depending on order, i.e whether they are in θP or θM
    θc = int_θdoubleMM(θ)
    r0, r1, K1, K2 = θc[(:r0, :r1, :K1, :K2)]
    y = r0 .+ r1 .* x.S1 ./ (K1 .+ x.S1) .* x.S2 ./ (K2 .+ x.S2)
    return (y)
end

function HVI.get_hybridproblem_par_templates(::DoubleMMCase; scenario::NTuple = ())
    (; θP, θM)
end

function HVI.get_hybridproblem_transforms(::DoubleMMCase; scenario::NTuple = ())
    (; transP, transM)
end

function HVI.get_hybridproblem_neg_logden_obs(::DoubleMMCase; scenario::NTuple = ())
    neg_logden_indep_normal
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

function HVI.get_hybridproblem_PBmodel(::DoubleMMCase; scenario::NTuple = ())
    #fsite = (θ, x_site) -> f_doubleMM(θ)  # omit x_site drivers
    function f_doubleMM_with_global(θP::AbstractVector, θMs::AbstractMatrix, x)
        pred_sites = applyf(f_doubleMM, θMs, θP, x)
        pred_global = eltype(pred_sites)[]
        return pred_global, pred_sites
    end
end

# function HVI.get_hybridproblem_float_type(::DoubleMMCase; scenario)
#     return Float32
# end

const xP_S1 = Float32[1.0, 1.0, 1.0, 1.0, 0.4, 0.3, 0.1]
const xP_S2 = Float32[1.0, 3.0, 4.0, 5.0, 5.0, 5.0, 5.0]

function HVI.gen_hybridcase_synthetic(rng::AbstractRNG, prob::DoubleMMCase;
        scenario = ())
    n_covar_pc = 2
    n_site = 200
    n_covar = 5
    n_θM = length(θM)
    FloatType = get_hybridproblem_float_type(prob; scenario)
    xM, θMs_true0 = gen_cov_pred(rng, FloatType, n_covar_pc, n_covar, n_site, n_θM;
        rhodec = 8, is_using_dropout = false)
    int_θMs_sites = ComponentArrayInterpreter(θM, (n_site,))
    # normalize to be distributed around the prescribed true values
    θMs_true = int_θMs_sites(scale_centered_at(θMs_true0, θM, FloatType(0.1)))
    f = get_hybridproblem_PBmodel(prob; scenario)
    xP = fill((;S1=xP_S1, S2=xP_S2), n_site)
    y_global_true, y_true = f(θP, θMs_true, xP)
    σ_o = FloatType(0.01)
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
        y_unc = fill(logσ2_o, size(y_o)),
    )
end

function HVI.get_hybridproblem_MLapplicator(
    rng::AbstractRNG, prob::HVI.DoubleMM.DoubleMMCase; scenario = ())
    ml_engine = select_ml_engine(; scenario)
    construct_3layer_MLApplicator(rng, prob, ml_engine; scenario)
end






