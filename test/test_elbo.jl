#using LinearAlgebra, BlockDiagonals

using Test
using Zygote
using HybridVariationalInference
using HybridVariationalInference: HybridVariationalInference as CP
using StableRNGs
using CUDA
using GPUArraysCore: GPUArraysCore
using Random
using SimpleChains
using ComponentArrays: ComponentArrays as CA
using TransformVariables

#CUDA.device!(4)
rng = StableRNG(111)

const case = DoubleMM.DoubleMMCase()
const MLengine = Val(nameof(SimpleChains))
scenario = (:default,)

#θsite_true = get_hybridcase_par_templates(case; scenario)
g, ϕg0 = gen_hybridcase_MLapplicator(case, MLengine; scenario);
f = gen_hybridcase_PBmodel(case; scenario)

(; n_covar, n_site, n_batch, n_θM, n_θP) = get_hybridcase_sizes(case; scenario)

(; xM, θP_true, θMs_true, xP, y_global_true, y_true, y_global_o, y_o
) = gen_hybridcase_synthetic(case, rng; scenario);

# correlation matrices
ρsP = zeros(sum(1:(n_θP-1)))
ρsM = zeros(sum(1:(n_θM-1)))

() -> begin
    coef_logσ2_logMs = [-5.769 -3.501; -0.01791 0.007951]
    logσ2_logP = CA.ComponentVector(r0=-8.997, K2=-5.893)
    #mean_σ_o_MC = 0.006042

    ϕunc = CA.ComponentVector(;
        logσ2_logP=logσ2_logP,
        coef_logσ2_logMs=coef_logσ2_logMs,
        ρsP,
        ρsM)
end


# for a conservative uncertainty assume σ2=1e-10 and no relationship with magnitude
ϕunc0 = CA.ComponentVector(;
    logσ2_logP=fill(-10.0, n_θP),
    coef_logσ2_logMs=reduce(hcat, ([-10.0, 0.0] for _ in 1:n_θM)),
    ρsP,
    ρsM)
#int_unc = ComponentArrayInterpreter(ϕunc0)

transPMs_batch = as(
    (P=as(Array, asℝ₊, n_θP),
    Ms=as(Array, asℝ₊, n_θM, n_batch)))
transPMs_all = as(
    (P=as(Array, asℝ₊, n_θP),
    Ms=as(Array, asℝ₊, n_θM, n_site)))
    
ϕ_true = θ = CA.ComponentVector(;
    μP=θP_true,
    ϕg=ϕg0, #ϕg_opt,  # here start from randomized
    unc=ϕunc);
trans_gu = as(
    (μP=as(Array, asℝ₊, n_θP),
    ϕg=as(Array, length(ϕg0)),
    unc=as(Array, length(ϕunc))))
trans_g = as(
    (μP=as(Array, asℝ₊, n_θP),
    ϕg=as(Array, length(ϕg0))))

int_PMs_batch = ComponentArrayInterpreter(CA.ComponentVector(; θP_true,
    θMs=CA.ComponentMatrix(
        zeros(n_θM, n_batch), first(CA.getaxes(θMs_true)), CA.Axis(i=1:n_batch))))

interpreters = map(get_concrete,(; 
    μP_ϕg_unc=ComponentArrayInterpreter(ϕ_true), 
    PMs=int_PMs_batch,
    unc=ComponentArrayInterpreter(ϕunc0)
    ))

ϕg_true_vec = CA.ComponentVector(
    TransformVariables.inverse(trans_gu, CP.cv2NamedTuple(ϕ_true)))
ϕcg_true = interpreters.μP_ϕg_unc(ϕg_true_vec)
ϕ_ini = ζ = vcat(ϕcg_true[[:μP, :ϕg]] .* 1.2, ϕcg_true[[:unc]]);
ϕ_ini0 = ζ = vcat(ϕcg_true[:μP] .* 0.0, ϕg0, ϕunc0);

neg_elbo_transnorm_gf(rng, g, f, ϕcg_true, y_o[:, 1:n_batch], xM[:, 1:n_batch],
    transPMs_batch, map(get_concrete, interpreters);
    n_MC=8, logσ2y)
Zygote.gradient(ϕ -> neg_elbo_transnorm_gf(
        rng, g, f, ϕ, y_o[:, 1:n_batch], x_o[:, 1:n_batch],
        transPMs_batch, interpreters; n_MC=8, logσ2y), ϕcg_true)

    @testset "generate_ζ" begin
        ϕ = CA.getdata(ϕ_cpu)
        n_sample_pred = 200
        intm_PMs_gen = ComponentArrayInterpreter(CA.ComponentVector(; θP_true,
            θMs=CA.ComponentMatrix(
                zeros(n_θM, n_site), first(CA.getaxes(θMs_true)), CA.Axis(i=1:n_sample_pred))))
        int_μP_ϕg_unc=ComponentArrayInterpreter(ϕ_true)
        interpreters = (; PMs = intm_PMs_gen, μP_ϕg_unc = int_μP_ϕg_unc  )
        ζs, _ = CP.generate_ζ(rng, g, f, ϕ, xM, interpreters; n_MC=n_sample_pred)
        
    end;

