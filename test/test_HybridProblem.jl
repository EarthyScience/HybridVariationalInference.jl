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
    S1 = [1.0, 1.0, 1.0, 1.0, 0.4, 0.3, 0.1]
    S2 = [1.0, 3.0, 4.0, 5.0, 5.0, 5.0, 5.0]
    θP = CA.ComponentVector{Float32}(r0 = 0.3, K2 = 2.0)
    θM = CA.ComponentVector{Float32}(r1 = 0.5, K1 = 0.2)    
    transP = elementwise(exp)
    transM = Stacked(elementwise(identity), elementwise(exp))
    n_covar = 5
    n_batch = 10
    int_θdoubleMM = get_concrete(ComponentArrayInterpreter(
        flatten1(CA.ComponentVector(; θP, θM))))
    function f_doubleMM(θ::AbstractVector)
        # extract parameters not depending on order, i.e whether they are in θP or θM
        θc = int_θdoubleMM(θ)
        r0, r1, K1, K2 = θc[(:r0, :r1, :K1, :K2)]
        y = r0 .+ r1 .* S1 ./ (K1 .+ S1) .* S2 ./ (K2 .+ S2)
        return (y)
    end
    fsite = (θ, x_site) -> f_doubleMM(θ)  # omit x_site drivers
    function f_doubleMM_with_global(θP::AbstractVector, θMs::AbstractMatrix, x)
        pred_sites = applyf(fsite, θMs, θP, x)
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
            TurboDense{false}(identity, n_out),
        )
    g = construct_SimpleChainsApplicator(g_chain)
    ϕg = SimpleChains.init_params(g_chain, eltype(θM));
    HybridProblem(θP, θM, transM, transP, n_covar, n_batch, f_doubleMM_with_global, g, ϕg)
end
prob = construct_problem();
case_syn = DoubleMM.DoubleMMCase()
scenario = (:default,)

par_templates = get_hybridcase_par_templates(prob; scenario)

(; n_covar, n_batch, n_θM, n_θP) = get_hybridcase_sizes(prob; scenario)

rng = StableRNG(111)
(; xM, n_site, θP_true, θMs_true, xP, y_global_true, y_true, y_global_o, y_o
) = gen_hybridcase_synthetic(case_syn, rng; scenario);

@testset "loss_g" begin
    g, ϕg0 = get_hybridcase_MLapplicator(prob, MLengine; scenario);

    function loss_g(ϕg, x, g)
        ζMs = g(x, ϕg) # predict the log of the parameters
        θMs = exp.(ζMs)
        loss = sum(abs2, θMs .- θMs_true)
        return loss, θMs
    end
    loss_g(ϕg0, xM, g)
    Zygote.gradient(x -> loss_g(x, xM, g)[1], ϕg0);

    optf = Optimization.OptimizationFunction((ϕg, p) -> loss_g(ϕg, xM, g)[1],
        Optimization.AutoZygote())
    optprob = Optimization.OptimizationProblem(optf, ϕg0);
    #res = Optimization.solve(optprob, Adam(0.02), callback = callback_loss(100), maxiters = 600);
    res = Optimization.solve(optprob, Adam(0.02), maxiters = 600);

    ϕg_opt1 = res.u;
    pred = loss_g(ϕg_opt1, xM, g)
    θMs_pred = pred[2]
    #scatterplot(vec(θMs_true), vec(θMs_pred))
    @test cor(vec(θMs_true), vec(θMs_pred)) > 0.9
end

@testset "loss_gf" begin
    #----------- fit g and θP to y_o
    g, ϕg0 = get_hybridcase_MLapplicator(prob, MLengine; scenario);
    f = get_hybridcase_PBmodel(prob; scenario)

    int_ϕθP = ComponentArrayInterpreter(CA.ComponentVector(
        ϕg = 1:length(ϕg0), θP = par_templates.θP))
    p = p0 = vcat(ϕg0, par_templates.θP .* 0.8);  # slightly disturb θP_true

    # Pass the site-data for the batches as separate vectors wrapped in a tuple
    train_loader = MLUtils.DataLoader((xM, xP, y_o), batchsize = n_batch)

    loss_gf = get_loss_gf(g, f, y_global_o, int_ϕθP)
    l1 = loss_gf(p0, train_loader.data...)[1]

    optf = Optimization.OptimizationFunction((ϕ, data) -> loss_gf(ϕ, data...)[1],
        Optimization.AutoZygote())
    optprob = OptimizationProblem(optf, p0, train_loader)

    res = Optimization.solve(
#        optprob, Adam(0.02), callback = callback_loss(100), maxiters = 1000);
        optprob, Adam(0.02), maxiters = 1000);

    l1, y_pred_global, y_pred, θMs_pred = loss_gf(res.u, train_loader.data...)
    @test isapprox(par_templates.θP, int_ϕθP(res.u).θP, rtol = 0.11)
    @test cor(vec(θMs_true), vec(θMs_pred)) > 0.9

    () -> begin
        scatterplot(vec(θMs_true), vec(θMs_pred))
        scatterplot(log.(vec(θMs_true)), log.(vec(θMs_pred)))
        scatterplot(vec(y_pred), vec(y_o))
        hcat(par_templates.θP, int_ϕθP(p0).θP, int_ϕθP(res.u).θP)
    end
end
