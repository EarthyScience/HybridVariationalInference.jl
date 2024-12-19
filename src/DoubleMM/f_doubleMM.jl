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

function HybridVariationalInference.gen_f(::DoubleMMCase; scenario::NTuple = ())
    fsite = (θ, x_site) -> f_doubleMM(θ)  # omit x_site drivers
    function f_doubleMM_with_global(θP::AbstractVector, θMs::AbstractMatrix, x)
        pred_sites = applyf(fsite, θMs, θP, x)
        pred_global = eltype(pred_sites)[]
        return pred_global, pred_sites
    end
end

function HybridVariationalInference.get_case_sizes(::DoubleMMCase; scenario = ())
    n_covar_pc = 2
    n_covar = n_covar_pc + 3 # linear dependent
    n_site = 10^n_covar_pc
    n_batch = 10
    n_θM = length(θM)
    n_θP = length(θP)
    (; n_covar_pc, n_covar, n_site, n_batch, n_θM, n_θP)
end

function HybridVariationalInference.get_case_FloatType(::DoubleMMCase; scenario)
    return Float32
end

function HybridVariationalInference.gen_cov_pred(case::DoubleMMCase, rng::AbstractRNG;
        scenario = ())
    (; n_covar_pc, n_covar, n_site, n_batch, n_θM, n_θP) = get_case_sizes(case; scenario)
    FloatType = get_case_FloatType(case; scenario)
    gen_cov_pred(rng, FloatType, n_covar_pc, n_covar, n_site, n_θM;
        rhodec = 8, is_using_dropout = false)
end
