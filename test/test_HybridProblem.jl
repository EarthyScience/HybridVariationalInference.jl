using Test
using HybridVariationalInference
using StableRNGs
using Random
using Statistics
using ComponentArrays: ComponentArrays as CA
using Bijectors

using SimpleChains
using MLUtils
import Zygote

using OptimizationOptimisers

const MLengine = Val(nameof(SimpleChains))

construct_problem = () -> begin
    FT = Float32
    θP = CA.ComponentVector{FT}(r0=0.3, K2=2.0)
    θM = CA.ComponentVector{FT}(r1=0.5, K1=0.2)
    transP = elementwise(exp)
    transM = Stacked(elementwise(identity), elementwise(exp))
    cov_starts = (P=(1, 2), M=(1)) # assume r0 independent of K2
    n_covar = 5
    n_batch = 10
    int_θdoubleMM = get_concrete(ComponentArrayInterpreter(
        flatten1(CA.ComponentVector(; θP, θM))))
    function f_doubleMM(θ::AbstractVector, x)
        # extract parameters not depending on order, i.e whether they are in θP or θM
        θc = int_θdoubleMM(θ)
        r0, r1, K1, K2 = θc[(:r0, :r1, :K1, :K2)]
        y = r0 .+ r1 .* x.S1 ./ (K1 .+ x.S1) .* x.S2 ./ (K2 .+ x.S2)
        return (y)
    end
    function f_doubleMM_with_global(θP::AbstractVector, θMs::AbstractMatrix, x)
        pred_sites = applyf(f_doubleMM, θMs, θP, x)
        pred_global = eltype(pred_sites)[]
        return pred_global, pred_sites
    end
    n_out = length(θM)
    g_chain = SimpleChain(
        static(n_covar), # input dimension (optional)
        # dense layer with bias that maps to 8 outputs and applies `tanh` activation
        TurboDense{true}(tanh, n_covar * 4),
        TurboDense{true}(tanh, n_covar * 4),
        # dense layer without bias that maps to n outputs and `identity` activation
        TurboDense{false}(identity, n_out)
    )
    # g, ϕg = construct_SimpleChainsApplicator(g_chain)
    #
    rng = StableRNG(111)
    # dependency on DeoubleMMCase -> take care of changes in covariates
    (; xM, n_site, θP_true, θMs_true, xP, y_global_true, y_true, y_global_o, y_o, y_unc
    ) = gen_hybridcase_synthetic(rng, DoubleMM.DoubleMMCase())
    py = neg_logden_indep_normal
    train_loader = MLUtils.DataLoader((xM, xP, y_o, y_unc), batchsize=n_batch)
    HybridProblem(θP, θM, g_chain, f_doubleMM_with_global, py,
        transM, transP, train_loader, cov_starts)
end
prob = construct_problem();
scenario = (:default,)

@testset "loss_gf" begin
    #----------- fit g and θP to y_o
    rng = StableRNG(111)
    g, ϕg0 = get_hybridcase_MLapplicator(prob, MLengine; scenario)
    train_loader = get_hybridcase_train_dataloader(rng, prob; scenario)
    (xM, xP, y_o, y_unc) = first(train_loader)
    f = get_hybridcase_PBmodel(prob; scenario)
    par_templates = get_hybridcase_par_templates(prob; scenario)
    (;transM, transP) = get_hybridcase_transforms(prob; scenario)

    int_ϕθP = ComponentArrayInterpreter(CA.ComponentVector(
        ϕg=1:length(ϕg0), θP=par_templates.θP))
    p = p0 = vcat(ϕg0, par_templates.θP .* 0.8)  # slightly disturb θP_true

    # Pass the site-data for the batches as separate vectors wrapped in a tuple

    y_global_o = Float64[]
    loss_gf = get_loss_gf(g, transM, f, y_global_o, int_ϕθP)
    l1 = loss_gf(p0, first(train_loader)...)
    gr = Zygote.gradient(p -> loss_gf(p, train_loader.data...)[1], p0)
    @test gr[1] isa Vector

    () -> begin
        optf = Optimization.OptimizationFunction((ϕ, data) -> loss_gf(ϕ, data...)[1],
            Optimization.AutoZygote())
        optprob = OptimizationProblem(optf, p0, train_loader)

        res = Optimization.solve(
            #        optprob, Adam(0.02), callback = callback_loss(100), maxiters = 1000);
            optprob, Adam(0.02), maxiters=1000)

        l1, y_pred_global, y_pred, θMs_pred = loss_gf(res.u, train_loader.data...)
        @test isapprox(par_templates.θP, int_ϕθP(res.u).θP, rtol=0.11)
    end
end

using CUDA
import Flux

@testset "neg_elbo_transnorm_gf cpu" begin
    rng = StableRNG(111)
    g, ϕg0 = get_hybridcase_MLapplicator(prob, MLengine)
    train_loader = get_hybridcase_train_dataloader(rng, prob)
    (xM, xP, y_o, y_unc) = first(train_loader)
    n_batch = size(y_o, 2)
    f = get_hybridcase_PBmodel(prob)
    (θP0, θM0) = get_hybridcase_par_templates(prob)
    (; transP, transM) = get_hybridcase_transforms(prob)
    py = get_hybridcase_neg_logden_obs(prob)

    (; ϕ, transPMs_batch, interpreters, get_transPMs, get_ca_int_PMs) = init_hybrid_params(
        θP0, θM0, ϕg0, n_batch; transP, transM)
    ϕ_ini = ϕ

    py = get_hybridcase_neg_logden_obs(prob)

    cost = neg_elbo_transnorm_gf(rng, g, f, py, ϕ_ini, y_o, y_unc,
        xM, xP, transPMs_batch, map(get_concrete, interpreters);
        n_MC=8)
    @test cost isa Float64
    gr = Zygote.gradient(
        ϕ -> neg_elbo_transnorm_gf(rng, g, f, py, ϕ, y_o, y_unc,
            xM, xP, transPMs_batch, map(get_concrete, interpreters);
            n_MC=8),
        CA.getdata(ϕ_ini))
    @test gr[1] isa Vector

    if CUDA.functional()
        @testset "neg_elbo_transnorm_gf gpu" begin
            g, ϕg0 = begin
                n_covar = size(xM, 1)
                n_out = length(θM0)
                g_chain = Flux.Chain(
                    # dense layer with bias that maps to 8 outputs and applies `tanh` activation
                    Flux.Dense(n_covar => n_covar * 4, tanh),
                    Flux.Dense(n_covar * 4 => n_covar * 4, tanh),
                    # dense layer without bias that maps to n outputs and `identity` activation
                    Flux.Dense(n_covar * 4 => n_out, identity, bias=false)
                )
                construct_ChainsApplicator(g_chain, eltype(θM0))
            end
            ϕ_ini.ϕg = ϕg0
            ϕ = CuArray(CA.getdata(ϕ_ini))
            xMg = CuArray(xM)
            cost = neg_elbo_transnorm_gf(rng, g, f, py, ϕ, y_o, y_unc,
                xMg, xP, transPMs_batch, map(get_concrete, interpreters);
                n_MC=8)
            @test cost isa Float64
            gr = Zygote.gradient(
                ϕ -> neg_elbo_transnorm_gf(rng, g, f, py, ϕ, y_o, y_unc,
                    xMg, xP, transPMs_batch, map(get_concrete, interpreters);
                    n_MC=8),
                ϕ)
            @test gr[1] isa CuVector
            @test eltype(gr[1]) == get_hybridcase_float_type(prob)
        end
    end
end
