struct DoubleMMCase <: AbstractHybridCase end

const S1 = [1.0, 1.0, 1.0, 1.0, 0.4, 0.3, 0.1]
const S2 = [1.0, 3.0, 4.0, 5.0, 5.0, 5.0, 5.0]

θP = CA.ComponentVector(r0 = 0.3, K2 = 2.0)
θM = CA.ComponentVector(r1 = 0.5, K1 = 0.2)

const int_θdoubleMM = ComponentArrayInterpreter(flatten1(CA.ComponentVector(; θP, θM)))

function f_doubleMM(θ::AbstractVector)
    # extract parameters not depending on order, i.e whether they are in θP or θM
    θc = int_θdoubleMM(θ)
    r0, r1, K1, K2 = θc[(:r0, :r1, :K1, :K2)]
    y = r0 .+ r1 .* S1 ./ (K1 .+ S1) .* S2 ./ (K2 .+ S2)
    return (y)
end

function HybridVariationalInference.get_hybridcase_par_templates(::DoubleMMCase; scenario::NTuple = ())
    (; θP, θM)
end

function HybridVariationalInference.get_hybridcase_sizes(::DoubleMMCase; scenario = ())
    n_covar_pc = 2
    n_covar = n_covar_pc + 3 # linear dependent
    n_site = 10^n_covar_pc
    n_batch = 10
    n_θM = length(θM)
    n_θP = length(θP)
    (; n_covar, n_site, n_batch, n_θM, n_θP)
end

function HybridVariationalInference.gen_hybridcase_PBmodel(::DoubleMMCase; scenario::NTuple = ())
    fsite = (θ, x_site) -> f_doubleMM(θ)  # omit x_site drivers
    function f_doubleMM_with_global(θP::AbstractVector, θMs::AbstractMatrix, x)
        pred_sites = applyf(fsite, θMs, θP, x)
        pred_global = eltype(pred_sites)[]
        return pred_global, pred_sites
    end
end

function HybridVariationalInference.get_hybridcase_FloatType(::DoubleMMCase; scenario)
    return Float32
end

function HybridVariationalInference.gen_hybridcase_synthetic(case::DoubleMMCase, rng::AbstractRNG;
        scenario = ())
    n_covar_pc = 2
    (; n_covar, n_site, n_batch, n_θM, n_θP) = get_hybridcase_sizes(case; scenario)
    FloatType = get_hybridcase_FloatType(case; scenario)
    xM, θMs_true0 = gen_cov_pred(rng, FloatType, n_covar_pc, n_covar, n_site, n_θM;
        rhodec = 8, is_using_dropout = false)
    int_θMs_sites = ComponentArrayInterpreter(θM, (n_site,))
    # normalize to be distributed around the prescribed true values
    θMs_true = int_θMs_sites(scale_centered_at(θMs_true0, θM, 0.1))
    f = gen_hybridcase_PBmodel(case; scenario)
    xP = fill((), n_site)
    y_global_true, y_true = f(θP, θMs_true, zip())
    σ_o = 0.01
    #σ_o = 0.002
    y_global_o = y_global_true .+ randn(rng, size(y_global_true)) .* σ_o
    y_o = y_true .+ randn(rng, size(y_true)) .* σ_o
    (;
        xM,
        θP_true = θP,
        θMs_true,
        xP,
        y_global_true,
        y_true,
        y_global_o,
        y_o,
        σ_o = fill(σ_o, size(y_true,1)),
    )
end
