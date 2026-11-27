# updated doubleMM.jl with using HybridSolveSite

if basename(pwd()) != "projects"; cd("projects"); end; Pkg.activate("."); 
@info "Start of doubleMM_site.jl"
@info("Number of threads: $(Threads.nthreads())")

using Test 
using HybridVariationalInference
using HybridVariationalInference: HybridVariationalInference as HVI
using StableRNGs
using Random
using Statistics
using ComponentArrays: ComponentArrays as CA
using Optimization
using OptimizationOptimisers # Adam
using UnicodePlots
using Lux
using MLUtils
using StatsBase
#import MLDataDevices, CUDA, cuDNN, GPUArraysCore

rng = StableRNG(115)
scenario = Val((:use_Lux,:usePopPBM,:use_rangescaling,:omit_r0,))  # without omit_r0 ambiguous K2 estimated to high
#scenario = Val((:use_Lux,:usePopPBM,:use_rangescaling,:omit_r0,:covarK2))  # with allow corr population
#scenario = Val((:use_Lux,:usePopPBM,:use_rangescaling,:covarK2,:notomit_r0))  
# prob = DoubleMM.DoubleMMCase()
# scenario = Val((:use_Lux,:usePopPBM,:use_rangescaling,:omit_r0,:covarK2,:fewsites))  # only 80 sites
# not omitting r0 and few sites leads no-unique sampling across chains and large 
#  discrepancies in inferred P.K2 and I.r1 and K1
#scenario = Val((:use_Lux,:usePopPBM,:use_rangescaling,:covarK2,:fewsites,:notomit_r0))  # only 80 sites

# :allsites: use n_batch = n_site
#scenario = Val((:use_Lux,:usePopPBM,:use_rangescaling,:allsites,:omit_r0,))  
#scenario = Val((:use_Lux,:usePopPBM,:use_rangescaling,:allsites,:omit_r0,:covarK2))  
#scenario = Val((:use_Lux,:usePopPBM,:use_rangescaling,:forthbatch,:omit_r0,))  
#scenario = Val((:use_Lux,:usePopPBM,:use_rangescaling,:forthbatch,:omit_r0,:covarK2))  

# using UniScaling
#scenario = Val((:use_Lux,:usePopPBM,:use_rangescaling,:uniscaling,:omit_r0,))  # without omit_r0 ambiguous K2 estimated to high


#------ setup synthetic data and training data loader
function create_problem(; scenario)
    rng = StableRNG(115) # fix so that returns the same data for a given scenario
    prob0_ = HybridProblem(DoubleMM.DoubleMMCase(); scenario);
    (; xM, θP_true, θMs_true, xP,  y_true, y_o, y_unc
    ) = gen_hybridproblem_synthetic(rng, DoubleMM.DoubleMMCase(); scenario);
    n_indiv = size(xM, 2)
    n_site_test = n_indiv ÷ 10    
    iall_indiv_test = sample(rng, 1:n_indiv, n_site_test, replace=false)
    iall_indiv_train = setdiff(1:n_indiv, iall_indiv_test)
    _, n_batch = get_hybridproblem_n_indiv_train_and_batch(prob0_; scenario)
    n_indiv_train = length(iall_indiv_train)
    if :fewsites ∈ HVI._val_value(scenario) 
        n_indiv_train = 80
        iall_indiv_train = iall_indiv_train[1:n_indiv_train]
        n_batch = min(n_indiv_train, n_batch)
    end
    if :allsites ∈ HVI._val_value(scenario)
        n_batch = n_indiv_train
    end
    if :forthbatch ∈ HVI._val_value(scenario)
        n_batch = n_indiv_train ÷ 5
    end
    itrain_sites = zeros(n_indiv)
    itrain_sites[iall_indiv_test] .= 1:length(iall_indiv_test)
    iall = vcat(iall_indiv_train, iall_indiv_test)
    test_data = (; xM = xM[:, iall_indiv_test], xP = xP[:, iall_indiv_test], y_o = y_o[:, iall_indiv_test],
        y_unc = y_unc[:, iall_indiv_test], itrain_sites = zeros(Int, length(iall_indiv_test)))
    train_data = (; xM = xM[:, iall_indiv_train], xP = xP[:, iall_indiv_train], y_o = y_o[:, iall_indiv_train],
        y_unc = y_unc[:, iall_indiv_train], itrain_sites = 1:length(iall_indiv_train))
    train_dataloader = MLUtils.DataLoader(
        (train_data.xM, train_data.xP, train_data.y_o, train_data.y_unc, train_data.itrain_sites),
        batchsize = n_batch, partial = false, shuffle=true)
    σ_o = exp.(y_unc[:, 1] / 2)
    # assign the train_loader, otherwise it each time creates another version of synthetic data
    prob0 = HybridProblem(prob0_; train_dataloader, test_data, n_indiv_train = n_indiv_train, n_indiv_batch = n_batch)
    @assert get_hybridproblem_test_data(prob0) == test_data
    # alldata and all_θMs_true may encompass more than iall_indiv_train, iall_indiv_test with fewsites
    θMs_true_orig = copy(θMs_true)  # original order
    θMs_true_train = θMs_true[:,iall_indiv_train]
    θMs_true_test = θMs_true[:,iall_indiv_test]
    (;prob0, train_data, test_data, θP_true, θMs_true_train, θMs_true_test, θMs_true_orig, σ_o)
end 
# make sure to use the same data
(;prob0, train_data, test_data, θP_true, θMs_true_train, θMs_true_test, θMs_true_orig, σ_o) = create_problem(; scenario);
tmp = create_problem(; scenario = Val((HVI._val_value(scenario)..., :all_sites)));
@assert tmp.train_data == train_data
@assert tmp.θMs_true_train == θMs_true_train

    #tmp = HVI.get_hybridproblem_ϕq(prob0; scenario)
    #prob0.covar

function fit_pointwise(prob0::AbstractHybridProblem, args...; kwargs...)
    rng = Random.default_rng()
    fit_pointwise(rng, prob0, args...; kwargs...)
end
function fit_pointwise(rng::AbstractRNG, prob0::AbstractHybridProblem; 
        n_epoch = 1_000, η=0.001, patience = 50)
    #------- pointwise hybrid model fit
    solver_point = HybridSolverSite(; alg = OptimizationOptimisers.Adam(η))
    #solver_point = HybridPointSolver(; alg = Adam(0.01), n_batch = 30)
    #solver_point = HybridPointSolver(; alg = Adam(0.01), n_batch = 10)
    #solver_point = HybridPointSolver(; alg = Adam(), n_batch = 200)
    n_site, n_batch = get_hybridproblem_n_indiv_train_and_batch(prob0)    
    n_batches_in_epoch = n_site ÷ n_batch
    # n_epoch = 120
    # n_epoch = 600
    res_point = solve(rng, prob0, solver_point, NegLogDensity(); scenario, epochs = n_epoch, patience);
    return(res_point)
    
    # TODO adjust to new sampling
    hp, hi_agg = HVI.predict_point(rng, prob0, res_point.ϕc; scenario)
    y_pred = hi_agg.y
    θMs = hi_agg.θsM
    # @descend_code_warntype gf(prob0o; scenario)
    #@usingany UnicodePlots
    plt = scatterplot(θMs_true_train[1,:], θMs[1,:]);
    lineplot!(plt, 0, 1)
    scatterplot(θMs_true_train[2,:], θMs[2,:])
    hp.θsP
    #scatterplot(vec(y_true), vec(y_o))
    #scatterplot(vec(y_true), vec(y_pred))
    UnicodePlots.histogram(vec(y_pred) - vec(y_true)) # predictions centered around y_o (or y_true)
    res_point
end
# res_point = fit_pointwise(prob0);

#----------- Hybrid Variational inference: HVI
# using MLUtils
# using Bijectors
using JLD2
function interactive_elbo(prob0::AbstractHybridProblem, args...; kwargs...)
    rng = Random.default_rng()
    interactive_elbo(rng, prob0, args...; kwargs...)
end
function interactive_elbo(rng::AbstractRNG, prob0::AbstractHybridProblem; 
    ϕ0c = res_point.ϕc, n_epoch=1200, η=0.0002, patience=50,
    n_MC_val=Val(3), n_MC_test_val=Val(300),
    loss_mode = NegElbo(), p_old = 0.9,)
    #----------- HVI without strong prior on θmean
    # lower learning rate
    solver_post =  HybridSolverSite(; alg = OptimizationOptimisers.Adam(η), n_MC_val, n_MC_test_val)
    #ϕ0c = res_point.ϕc
    #n_epoch = 1200
    #exp.(res_point.ϕc.ϕqP.μζP)
    res_elbo1 = solve(rng, prob0, solver_post, loss_mode; ϕ0c, scenario, 
        epochs=n_epoch, patience, p_old,
        #show_train_loss = Val(true)
        );
    @show res_elbo1.epoch_best, res_elbo1.epoch_old
    return res_elbo1

    () -> begin
        res_elbo = res_elbo1;
        exp.(res_elbo.ϕc.ϕqP.μζP)
        n_sample_pred = 60
        hp_stat, hi_agg_stat = HVI.predict_stat(rng, prob0, res_elbo.ϕc; scenario, n_sample_pred);
        hp_stat.θsP[1,:]  
        hi_agg_stat.θsM[1,:,1]
        hp = map(x -> x[:,Val(:mean)], hp_stat)
        hi_agg = map(x -> x[:,Val(:mean),:], hi_agg_stat)
        @test hp.θsP ≈ θP_true atol=0.2
        @test mean(abs.(hi_agg.θsM .- θMs_true_train)) < 0.05
        hi_std = map(x -> x[:,Val(:std),:], hi_agg_stat)
        hi_std.θsM[:,1]  # indiv_1
        hi_std.θsM[:,2]  # indiv_2  same magnitude
        hi_agg.θsM[:,1]  
        hi_agg.θsM[:,2]  
    end

    () -> begin
        # continue with higher n_MC
        solver_post_cor =  HybridSolverSite(; 
            alg = OptimizationOptimisers.Adam(0.005), n_MC_val=Val(60))
        n_epoch = 200
        res_elbo2 = solve(prob0, solver_post_cor, NegElbo(); ϕ0c = res_elbo.ϕc, scenario, 
            epochs=n_epoch, );
        #n_epoch = 20
        res_covar = solve(prob0, solver_post_cor, HVI.NegLogDensityCovar(); ϕ0c = res_elbo1.ϕc, scenario, 
            epochs=n_epoch,
            #show_train_loss = Val(true)
            );
        res_covar2 = solve(prob0, solver_post_cor, HVI.NegLogDensityCovar(); ϕ0c = res_covar.ϕc, scenario, 
            epochs=1,
            #show_train_loss = Val(true)
            );

        res_elbo = solve(prob0, solver_post_cor, NegElbo(); ϕ0c = res_covar.ϕc, scenario, 
            epochs=n_epoch, );
        # gets worth on test loss - overfitting? separate train and test data
    end

    () -> begin # store and reload optimized problem
        #fname_probos = "intermediate/probos_$(last(scenario)).jld2"
        fname_elbo60 = "intermediate/elbo60_$(last(HVI._val_value(scenario))).jld2"
        @show fname_elbo60
        #JLD2.save(fname_probos, Dict("prob1o" => prob1o, "prob2o" => prob2o))
        JLD2.save(fname_elbo60, Dict("prob0"=> prob0, "res_elbo" => res_elbo))
        #prob, res_elbo = JLD2.load(fname_elbo60, "prob", "res_elbo")
        prob0, res_elbo = JLD2.load(fname_elbo60, "prob0", "res_elbo")
    end

    () -> begin # load the non-covar scenario, and neglect_cor scenario
        #using JLD2
        scenario_indep = Val(Tuple(s for s in HVI._val_value(scenario) if s != :covarK2))
        fname_probos_indep = "intermediate/probos800_$(last(HVI._val_value(scenario_indep))).jld2"
        #fname_probos = "intermediate/probos800_omit_r0.jld2"
        tmp = JLD2.load(fname_probos_indep)
        prob2o_indep = tmp["prob2o"]
        # test predicting correct obs-uncertainty of predictive posterior
        n_sample_pred = 400
        (; y, θsP, θsMs_tr) = predict_hvi(rng, prob2o_indep; scenario = scenario_indep, n_sample_pred);
        (y2_indep, θsP2_indep, θsMs2_tr_indep) = (y, θsP, θsMs_tr);
        #θsMs2_tr_indep .- θsMs2_tr
        #(θ2_indep, y2_indep) = (θ2, y2)  # workaround to use covarK2 when loading failed
        #
        scenario_neglect_cor = Val((HVI._val_value(scenario)..., :neglect_cor))
        fname_probos_neglect_cor = "intermediate/probos800_$(last(HVI._val_value(scenario_neglect_cor))).jld2"
        #fname_probos = "intermediate/probos800_omit_r0.jld2"
        tmp = JLD2.load(fname_probos_neglect_cor)
        prob2o_neglect_cor = tmp["prob2o"]
        # test predicting correct obs-uncertainty of predictive posterior
        n_sample_pred = 400
        (; y, θsP, θsMs_tr) = predict_hvi(rng, prob2o_neglect_cor; scenario = scenario_neglect_cor, n_sample_pred);
        (y2_neglect_cor, θsP2_neglect_cor, θsMs2_tr_neglect_cor) = (y, θsP, θsMs_tr);
        #
        scenario_K1global = Val((HVI._val_value(scenario)..., :K1global))
        fname_probos_K1global = "intermediate/probos800_$(last(HVI._val_value(scenario_K1global))).jld2"
        #fname_probos = "intermediate/probos800_omit_r0.jld2"
        tmp = JLD2.load(fname_probos_K1global)
        prob2o_K1global = tmp["prob2o"]
        # test predicting correct obs-uncertainty of predictive posterior
        n_sample_pred = 400
        (; y, θsP, θsMs_tr) = predict_hvi(rng, prob2o_K1global; scenario = scenario_K1global, n_sample_pred);
        (y2_K1global, θsP2_K1global, θsMs2_tr_K1global) = (y, θsP, θsMs_tr);
    end

    () -> begin # optimize using LUX
        #using Lux
        g_lux = Lux.Chain(
            # dense layer with bias that maps to 8 outputs and applies `tanh` activation
            Lux.Dense(n_covar => n_covar * 4, tanh),
            Lux.Dense(n_covar * 4 => n_covar * 4, logistic),
            # dense layer without bias that maps to n outputs and `identity` activation
            Lux.Dense(n_covar * 4 => n_θM, identity, use_bias = false)
        )
        ps, st = Lux.setup(Random.default_rng(), g_lux)
        ps_ca = CA.ComponentArray(ps) |> gpu
        st = st |> gpu
        g_luxs = StatefulLuxLayer{true}(g_lux, nothing, st)
        g_luxs(xM_gpu[:, 1:n_batch], ps_ca)
        ax_g = CA.getaxes(ps_ca)
        g_luxs(xM_gpu[:, 1:n_batch], CA.ComponentArray(ϕ.ϕg, ax_g))
        interpreters = (interpreters..., ϕg = ComponentArrayInterpreter(ps_ca))
        ϕg = CA.ComponentArray(ϕ.ϕg, ax_g)
        ϕgc = interpreters.ϕg(ϕ.ϕg)
        g_flux = g_luxs
    end

    () -> begin
        exp.(res_elbo.ϕc.ϕqP.μζP)
        exp.(res_point.ϕc.ϕqP.μζP)    

        exp.(res_elbo.ϕc.ϕqP.logσ_ζP)
        exp.(res_elbo.ϕc.ϕqI.coef_logσ_ζMs[:,1])

        # test predicting correct obs-uncertainty of predictive posterior
        n_sample_pred = 400
        hp_samp, hi_agg_samp = HVI.sample_posterior(rng, prob0, res_elbo.ϕc; 
            scenario, 
            n_sample_pred,
            );
        y = y2 = hi_agg_samp.y
        θsP = θsP2 = hp_samp.θsP
        θsMs = θsMs2 = hi_agg_samp.θsM
        θsMs12 = hi_agg_samp.θsM[1:2,:,:]
        nt12 = (;
            θP1 = θsP[1,:],
            θsM11 = θsMs[1,:,1],
            θsM21 = θsMs[2,:,1],
            θsM12 = θsMs[1,:,2],
            θsM22 = θsMs[2,:,2],
        )
        # using PairPlots
        # using CairoMakie
        fig = pairplot(nt12)
        save("../tmp/test.png", fig)

        size(y) # n_obs x  n_sample_pred x n_site,
        size(θsMs)  # n_θM x n_sample x n_site 
        σ_o_post = dropdims(std(CA.getdata(y); dims = 2), dims = 2);
        σ_o = exp.(y_unc[:, 1] / 2)

        #describe(σ_o_post)
        hcat(σ_o, # fill(mean_σ_o_MC, length(σ_o)),
            mean(σ_o_post, dims = 2), sqrt.(mean(abs2, σ_o_post, dims = 2)))
        # VI predicted uncertainty is smaller than HMC predicted one
        mean_y_pred = map(mean, eachslice(y; dims = (1, 3)));
        #describe(mean_y_pred - y_o)
        histogram(vec(mean_y_pred) - vec(y_true)) # predictions centered around y_o (or y_true)
        plt = scatterplot(vec(y_true), vec(mean_y_pred));
        lineplot!(plt, 0, 2)
        mean(mean_y_pred - y_true) # still ok

        mean_θMs = CA.ComponentArray(mean(CA.getdata(θsMs); dims = 2)[:,1,:], CA.getaxes(θMs_true))
        plt = scatterplot(θMs_true_train[1,:], mean_θMs[1,:]);
        lineplot!(plt, 0, 1)
        plt = scatterplot(θMs_true_train[2,:], mean_θMs[2,:])
        histogram(θsP)
    end
    #scatter(fig[1,1], CA.getdata(θMs_true_train[1, :]), CA.getdata(mean_θ.Ms[1, :])); ablines!(fig[1,1], 0, 1)
    #@usingany AlgebraOfGraphices
    #fig = Figure()
    #draw!(fig, data(DataFrame(x=CA.getdata(θMs_true_train[1, :]), y = CA.getdata(mean_θ.Ms[1, :]))) * mapping(:x, :y) * visual(Scatter))
    #lineplot!(plt, 0, 1)
    #plt = scatterplot(θMs_true_train[1, :], mean_θ.Ms[1, :] - θMs_true_train[1, :])
    #plt = scatterplot(θMs_true_train[2, :], mean_θ.Ms[2, :] - θMs_true_train[2, :])
    #plt = scatterplot(mean_θ.Ms[2, :] - θMs_true_train[2, :], mean_θ.Ms[1, :] - θMs_true_train[1, :])
    # mode_θ = map(mode, eachrow(θ))
    # plt = scatterplot(θMs_true_train[1, :], mode_θ.Ms[1, :]); lineplot!(plt, 0, 1)

    () -> begin # compare elbo components for mean-constrained unconstrained
        # solver_MC = HybridPosteriorSolver(; alg = Adam(0.01), n_batch = 30, n_MC = 300)
        # n_batches_in_epoch = n_site ÷ solver_MC.n_batch
        # (; ϕ, θP, resopt, interpreters) = solve(prob2o, solver_MC; scenario,
        #     rng, callback = callback_loss(n_batches_in_epoch), maxiters = 14);
        # resopt.objective
        # (; ϕ, θP, resopt, interpreters) = solve(prob1o, solver_MC; scenario,
        #     rng, callback = callback_loss(n_batches_in_epoch), maxiters = 14);
        # resopt.objective
        # probo = prob3o = HybridProblem(prob2; ϕg = cpu_ca(ϕ).ϕg, θP = θP, ϕq = cpu_ca(ϕ).ϕq)

        solver_post2 = HybridPosteriorSolver(solver_post; n_MC = 30)
        #solver_post2 = HybridPosteriorSolver(solver_post; n_MC = 3)
        n_rep = 30
        n_batchf = n_indiv_all
        n_batchf = n_indiv_all ÷ 10
        elbo = map(1:n_rep) do i_rep
            HVI.compute_elbo_components(
                prob2o, solver_post2; scenario, n_batch = n_batchf) |> collect
        end |> x -> stack(x; dims = 1)
        elbo_c = map(1:n_rep) do i_rep
            HVI.compute_elbo_components(
                prob1o, solver_post2; scenario, n_batch = n_batchf) |> collect
        end |> x -> stack(x; dims = 1)
        #@usingany AlgebraOfGraphics
        #@usingany CairoMakie
        #const AoG = AlgebraOfGraphics
        #@usingany DataFrames
        df = vcat(
            insertcols!(DataFrame(elbo, [:nLy, :ent, :nLmean_θ]), :scenario => "unconstrained"),
            insertcols!(DataFrame(elbo_c, [:nLy, :ent, :nLmean_θ]), :scenario => "θmean")
        ) |> x -> insertcols!(x, :elbo => -x.nLy + x.ent)
        plt = data(df) * mapping(:scenario, :elbo) * visual(BoxPlot)
        fig = draw(plt).figure
        save("tmp.svg", fig)
        save("elbo_boxplot.pdf", fig)
    end

    () -> begin # look at distribution of parameters, predictions, and likelihood and elob at one site
        # compare prob1o (with constraining theta to be near original mean) to unconstrained HVI
        function predict_site(probo, i_site)
            (; y, θsP, θsMs_tr, entropy_ζ) = predict_hvi(rng, probo; scenario, n_sample_pred)
            y_site = y[:, i_site, :]
            θMs_i = CA.ComponentArray(θsMs_tr[i_site,:,:], (CA.getaxes(θMs_true_train)[1], CA.FlatAxis()))
            r1s = θMs_i[:r1,:]
            # K1s = map(x -> x[2], θMs_i)
            # invt = map(Bijectors.inverse, get_hybridproblem_transforms(probo; scenario))
            # θPs = θ[:P,:]
            # ζPs = invt.transP.(θPs)
            # ζMs = invt.transM.(θMs_i)
            # pbm_spec = get_hybridproblem_PBModelSpec(probo; scenario)
            # _f = TODO
            # y_site = map(eachcol(θPs), θMs_i) do θP, θM
            #       y = _f(θP, reshape(θM, (length(θM), 1)), xP[[i_site]])
            #     y[:,1]
            # end |> stack
            nLs = get_hybridproblem_neg_logden_obs(
                probo; scenario).(eachcol(y_site), Ref(y_o[:, i_site]), Ref(y_unc[:, i_site]))
            (; r1s, nLs, entropy_ζ, y_site)
        end
        i_site = 1
        (r1s, nLs, ent, y_site) = predict_site(prob2o, i_site)
        (r1sc, nLsc, entc, y_sitec) = predict_site(prob1o, i_site) # result from point-solver
        mean(nLs), mean(nLsc)
        ent, entc
        # with larger uncertaintsy (higher entropy) in unconstrained cost much lower
        mean(nLs) - ent, mean(nLsc) - entc

        #@usingany CairoMakie
        #@usingany AlgebraOfGraphics
        #@usingany DataFrames

        # especially uncertainty is put to r1 (compensated by larger K1)
        df = DataFrame(r1 = vcat(r1s, r1sc),
            scenario = vcat(fill.(["unconstrained", "meanθ"], n_sample_pred)...))
        plt = data(df) * mapping(:r1, color = :scenario => "Scenario") * aog.density()
        plth = mapping([θMs_true_train[:r1, 1]]) * visual(VLines; linestyle = :dash)
        fig = Figure(; size = (640, 480))
        fig = Figure(; size = (320, 240))
        gp = fig[1, 1]
        fd = draw!(gp, plt + plth)
        legend!(
            gp, fd; tellwidth = false, halign = :right, valign = :top, margin = (10, 10, 10, 10))
        save("r1_density.pdf", fig)
        save("tmp.svg", fig)

        # observations are matched similarly well
        # with larger uncertainty the right-skewed shape of K1 leads to left-skewed y
        df = DataFrame(
            y = vcat(vec(y_site .- y_true[:, i_site]), vec(y_sitec .- y_true[:, i_site])),
            scenario = vcat(fill.(["unconstrained", "meanθ"], n_sample_pred * size(y_o, 1))...))
        plt = data(df) *
            mapping(:y => "y_predicted - y_observed", color = :scenario => "Scenario") *
            aog.density()
        plth = mapping([0.0]) * visual(VLines; linestyle = :dash)
        fig = Figure(; size = (640, 480))
        gp = fig[1, 1]
        fg = draw!(gp, plt + plth)
        legend!(
            gp, fg; tellwidth = false, halign = :right, valign = :top, margin = (10, 10, 10, 10))
        save("ys_density.pdf", fig)
        save("tmp.svg", fig)

        #slightly worse (higher) negLogLikelihood
        df = DataFrame(nL = vcat(nLs, nLsc),
            scenario = vcat(fill.(["unconstrained", "meanθ"], n_sample_pred)...))
        plt = data(df) * mapping(:nL => "-logDensity", color = :scenario => "Scenario") *
            aog.density()
        fig = Figure()
        gp = fig[1, 1]
        fg = draw!(gp, plt)
        legend!(
            gp, fg; tellwidth = false, halign = :right, valign = :top, margin = (10, 10, 10, 10))
        save("negLogDensity.pdf", fig)
        save("tmp.svg", fig)
    end

    () -> begin # look at θP, θM1 of first site
        θPM = vcat(θP_true, θMs_true_train[:, 1])
        intm = ComponentArrayInterpreter(θPM, (n_sample_pred,))
        θ1c = intm(θ[1:length(θPM), :])
        θPM
        #histogram((θ1c[:r0, :]))
        histogram((θ1c[:K2, :]))
        histogram((θ1c[:r1, :]))
        histogram((θ1c[:K1, :]))
        # overestimates r1 and underestimates K1
        # all parameters estimated to high (true not in cf bounds)
        scatterplot(θ1c[:r1, :], θ1c[:K1, :])  # r1 and K1 strongly correlated (from θM)
        scatterplot(θ1c[:r0, :], θ1c[:K2, :])  # r0 and K also correlated (from θP)
        scatterplot(θ1c[:r0, :], θ1c[:K1, :])  # no correlation (modeled independent)
    end
end # interactive_elbo
# res_elbo = interactive_elbo(prob0; n_epoch=8_000, patience = 1_000);
# res_elbo = interactive_elbo(prob0; n_epoch=8_000, patience = 1_000, loss_mode = NegElboFixϕgμI());
# @usingany UnicodePlots
# UnicodePlots.scatterplot(1:length(res_elbo.loss_test_hist), last.(res_elbo.loss_test_hist))

#---- do an DEMC inversion of the PBM model with parameters at log-scale
using DistributionFits
using PDMats
using Turing
#using MCMCChains
using FlexiChains
using DimensionalData
import PreallocationTools as PAT
using JLD2

# to adapt already the v0.7 syntax of FlexiChains
macro vn(args...)
    return macroexpand(__module__, :(@FlexiChains.varname($(args...))))
end
include("scratch_include.jl")

prob = HybridProblem(prob0);
(; θM, θP) = get_hybridproblem_par_templates(prob; scenario)
n_θP, n_θM  = length.((θP, θM))

# invert for the training and test set together
is_invert_train_only = false
#is_invert_train_only = true
if is_invert_train_only 
    all_data = train_data
    θMs_true = θMs_true_train
else
    # concatenate all components of test and train data preserving x-ComponentAxis
    cat_comp(x::CA.ComponentMatrix, y::CA.ComponentMatrix) = HVI.hcat_cm(x,y)
    cat_comp(x, y) = cat(x, y; dims = ndims(x))
    all_data = NamedTuple{keys(train_data)}(map(cat_comp, train_data, test_data))
    θMs_true = CA.ComponentArray(
        hcat(CA.getdata(θMs_true_train), CA.getdata(θMs_true_test)), CA.getaxes(θMs_true_train))
end
n_indiv = size(θMs_true,2)
n_indiv_all = size(train_data.xM,2)
UnicodePlots.scatterplot(θMs_true_train[1,:], θMs_true[1,1:n_indiv_all])

# construct a prior on log scale that ranges roughly across 1e.3 to 10
# Turing does not work well with Float32, switch all to Float64
# prior_ζ = fit(Normal{Float32}, @qp_ll(log(1e-2)), @qp_uu(log(10)))
# prior_ζn = (n) -> MvNormal(fill(prior_ζ.μ, n), PDiagMat(fill(abs2(prior_ζ.σ), n)))
prior_ζ = fit(Normal, @qp_ll(log(1e-2)), @qp_uu(log(10)))
prior_ζn = (n) -> MvNormal(fill(prior_ζ.μ, n), PDiagMat(fill(abs2(prior_ζ.σ), n)))
prior_ζn(3)
init_res = fit_mean_relerror(LogNormal, 1.0, 0.01)
#prior_res = fit_mean_relerror(LogNormal, 1.0, 0.1)
prior_res = Uniform(0.6, 1.4)
prior_resP = filldist(prior_res, n_θP)
prior_resMs = filldist(prior_res, n_θM, n_indiv)
prior_resM = filldist(prior_res, n_θM)
rand(prior_resP)
rand(prior_resMs)
#@assert eltype(rand(prior_ζn(3))) == Float32
@assert eltype(rand(prior_ζn(3))) == Float64

#i = 1
# uniform prior on log-scale causes problems, try using a very broad Normal
function setup_priors(θMs_true)
    local n_indiv
    n_indiv = size(θMs_true,2)
    prior_ζP = product_distribution(
        #map(i -> fit_mean_relerror(LogNormal, θP_true[i], 0.5), 1:n_θP) 
        map(i -> Normal(log(θP_true[i]), 5.0), 1:n_θP) 
    )
    prior_ζMs = product_distribution(
        #[fit_mean_relerror(LogNormal, θMs_true[i,indiv], 0.5) for i in 1:n_θM, indiv in 1:n_indiv]
        [Normal(log(θMs_true[i,indiv]), 5.0) for i in 1:n_θM, indiv in 1:n_indiv]
    )
    prior_θP = product_distribution(
        #map(i -> fit_mean_relerror(LogNormal, θP_true[i], 0.5), 1:n_θP) 
        map(i -> Normal(θP_true[i], 20.0), 1:n_θP) 
    )
    prior_θMs = product_distribution(
        #[fit_mean_relerror(LogNormal, θMs_true[i,indiv], 0.5) for i in 1:n_θM, indiv in 1:n_indiv]
        [Normal(θMs_true[i,indiv], 20.0) for i in 1:n_θM, indiv in 1:n_indiv]
    )

    init_θP = product_distribution(
        map(i -> fit_mean_relerror(LogNormal, θP_true[i], 0.01), 1:n_θP)
    )
    init_θMs = product_distribution(
        [fit_mean_relerror(LogNormal, θMs_true[i,indiv], 0.01) for i in 1:n_θM, indiv in 1:n_indiv]
    )
    log.(rand(init_θMs))
    rand(prior_ζMs)
    (;prior_ζP, prior_ζMs, prior_θP, prior_θMs, init_θP, init_θMs)
end
(;prior_ζP, prior_ζMs, prior_θP, prior_θMs, init_θP, init_θMs) = setup_priors(θMs_true)

pbm_spec = get_hybridproblem_PBModelSpec(prob; scenario)
import ForwardDiff
#f = get_PBModelApplicator(pbm_spec, θP, Val(1), ForwardDiff.Chunk(25)) # single prediction
f = get_PBModelApplicator(pbm_spec, zeros(0), Val(1), ForwardDiff.Chunk(12)) # single prediction
# for Turing inversion cannot rely on loop across sites, use the older PBMPopulationApplicatorTr 
θFix = pbm_spec.θFix
f_sites_tr = PBMPopulationApplicatorTr(DoubleMM.f_doubleMM_sites, n_indiv; θP, θM, θFix, xPvec = all_data.xP[:,1])
pred = HVI.apply_model(f_sites_tr, θP_true, θMs_true', all_data.xP)[1]
# there is concurrent write to get_tmp(Fixm_dc) across threads
#   but, all threads write the same value, so may use same pbm
# pbm_channel = Channel{typeof(f)}(8)
# put!(pbm_channel, f)
# for i in 2:8
#     f = get_PBModelApplicator(pbm_spec, zeros(0), Val(1), ForwardDiff.Chunk(12)) # single prediction
#     put!(pbm_channel, f)
# end

n_threads = min(16, Threads.nthreads())
n_obs, n_indiv_all = size(all_data.y_o)
n_indiv_train = size(train_data.y_o,2)
n_indiv_test = size(test_data.y_o,2)
function create_helper(;
    diffchunk::ForwardDiff.Chunk{chunk} = ForwardDiff.Chunk(12),
    use_diff_cache::Val{use_dc} = Val(true)
) where {chunk, use_dc}
    θsP = Matrix{Float64}(undef, size(θP_true,1), 1)
    θsMs = Array{Float64}(undef, size(θMs_true,1), 1, size(θMs_true,2))
    y_pred = Array{Float64}(undef, n_obs, 1, n_indiv_all)
    helpers_raw = (;θsP, θsMs, y_pred)
    helpers = use_dc ? map(x -> PAT.DiffCache(x, chunk), helpers_raw) : helpers_raw
end
h1 = create_helper()
helpers_channel = Channel{typeof(h1)}(n_threads)
put!(helpers_channel, h1)
for i in 2:n_threads
    local hi
    hi = create_helper()
    put!(helpers_channel, hi)
end


# # # TODO rename f_batch to pbmodel
# include("scratch_include.jl")
# model = fsites(y_o; f, n_θP, n_θM, xP = convert.(Float64, xP), xM = convert.(Float64, xM), σ_o)
# tmp = sample(model, NUTS(0,0.65), 2, initial_params = InitFromParams((
#     ζP = convert.(Float64, log.(CA.getdata(θP_true))), 
#     ζMs = convert.(Float64, log.(CA.getdata(θMs_true_train)))))
# )
# #ζ0_true .+ eltype(ζ0_true)(0.001))

# f_resid with sampling a multiplicative residual around each θ
function sample_resid() 
    @info "sample_resid"
    include("scratch_include.jl")
    model = f_resid(y_o; pbm = f, θP_true, θMs_true_train, xP, σ_o, prior_resP, prior_resMs, helpers_channel)
    #model = f_resid(y_o; pbm_channel, θP_true, θMs_true_train, xP, σ_o)
    #model = f_resid(y_o; f_sites_tr, θP_true, θMs_true_train, xP, σ_o)
    sample_initial_params = (rng=StableRNG(1234)) -> InitFromParams((
        # Float64: AdvancedHMC metric is Float64
        rP = ones(n_θP) .* rand(rng, filldist(init_res, n_θP)), 
        rMs = ones(n_θM, n_indiv_all) .* rand(rng, filldist(init_res, n_θM, n_indiv_all))
    ))
    initial_params = sample_initial_params()
    adtype = AutoForwardDiff(chunksize=12)
    #tmp = sample(model, NUTS(0, 0.65; adtype), 2; initial_params)
    #return nothing

    n_sample_NUTS = 2_000
    #n_sample_NUTS = 1_000
    #n_sample_NUTS = 16
    chn = sample(model, NUTS(8, 0.65), MCMCThreads(), ceil(Integer,n_sample_NUTS/n_threads), 
        n_threads; initial_params=map(_ -> sample_initial_params(rng),1:n_threads))
    # chn = sample(model, NUTS(8, 0.65), MCMCThreads(), ceil(Integer,n_sample_NUTS/n_threads), 
    #     n_threads; initial_params=fill(initial_params, n_threads))
    #fname = "intermediate/doubleMM_chain_rPMs2_$(last(HVI._val_value(scenario))).jld2" # LogNormal prior
    fname_resid = "intermediate/doubleMM_chain_rPMs3_$(last(HVI._val_value(scenario))).jld2" # Uniform prior
    jldsave(fname_resid, false, IOStream; chn)
end
#sample_resid()

# f_resid_agg: sample only a single residual added to all sites
function sample_agg() # fast can be done interactive
    @info "sample_agg"
    include("scratch_include.jl")
    model = f_resid_agg(y_o; pbm = f, θP_true, θMs_true_train, xP, σ_o, prior_resP, prior_resM, helpers_channel)
    #model = f_resid(y_o; pbm_channel, θP_true, θMs_true_train, xP, σ_o)
    #model = f_resid(y_o; f_sites_tr, θP_true, θMs_true_train, xP, σ_o)
    sample_initial_params = (rng=StableRNG(1234)) -> InitFromParams((
        # Float64: AdvancedHMC metric is Float64
        rP = ones(n_θP) .* rand(rng, filldist(init_res, n_θP)), 
        rM = ones(n_θM) .* rand(rng, filldist(init_res, n_θM))
    ))
    initial_params = sample_initial_params()
    #initial_params.params[@vn(rMs)]
    adtype = AutoForwardDiff()
    #adtype = AutoEnzyme() # does not work
    # import Mooncake
    # adtype = AutoMooncake() 
    tmp = sample(model, NUTS(0, 0.65), 2; initial_params)

    n_threads = min(16, Threads.nthreads())
    n_sample_NUTS = 2_000
    #n_sample_NUTS = 1_000
    #n_sample_NUTS = 16
    chn = sample(model, NUTS(8, 0.65), MCMCThreads(), ceil(Integer,n_sample_NUTS/n_threads), 
        n_threads; initial_params=map(_ -> sample_initial_params(rng),1:n_threads))
    # chn = sample(model, NUTS(8, 0.65), MCMCThreads(), ceil(Integer,n_sample_NUTS/n_threads), 
    #     n_threads; initial_params=fill(initial_params, n_threads))

    #fname = "intermediate/doubleMM_chain_rPMs2_$(last(HVI._val_value(scenario))).jld2" # LogNormal prior
    fname_agg = "intermediate/doubleMM_chain_rPMs_agg_$(last(HVI._val_value(scenario))).jld2" # Uniform prior
    jldsave(fname_agg, false, IOStream; chn)
end

# fsitesζ_flat: sample on log-scale without correcting logden for transformation
function sample_ζ_flatnormalprior()
    @info "sample_ζ_flatnormalprior"
    include("scratch_include.jl")
    model = fsitesζ_flat(y_o; pbm = f, xP, σ_o, prior_ζP, prior_ζMs, helpers_channel)
    #model = f_resid(y_o; pbm_channel, θP_true, θMs_true_train, xP, σ_o)
    #model = f_resid(y_o; f_sites_tr, θP_true, θMs_true_train, xP, σ_o)
    sample_initial_params = (rng=StableRNG(1234)) -> InitFromParams((
        ζP = log.(rand(init_θP)),
        ζMs = log.(rand(init_θMs)),
    ))
    initial_params = sample_initial_params()
    adtype = AutoForwardDiff(chunksize=12)
    #tmp = sample(model, NUTS(0, 0.65; adtype), 2; initial_params)
    #return nothing

    n_sample_NUTS = 1_500
    #n_sample_NUTS = 16
    chn = sample(model, NUTS(8, 0.65), MCMCThreads(), ceil(Integer,n_sample_NUTS/n_threads), 
        n_threads; initial_params=map(_ -> sample_initial_params(rng),1:n_threads))
    # chn = sample(model, NUTS(8, 0.65), MCMCThreads(), ceil(Integer,n_sample_NUTS/n_threads), 
    #     n_threads; initial_params=fill(initial_params, n_threads))
    #fname = "intermediate/doubleMM_chain_rPMs2_$(last(HVI._val_value(scenario))).jld2" # LogNormal prior
    fname_flatnormal = "intermediate/doubleMM_chain_zeta_normal_$(last(HVI._val_value(scenario))).jld2" # Uniform prior
    jldsave(fname_flatnormal, false, IOStream; chn)
end
#sample_ζ_flatnormalprior()


# fsitesθ: sample θ on original untransformed scale
function sample_fsitesθ()
    @info "sample_fsitesθ"
    include("scratch_include.jl")
    model = fsitesθ(all_data.y_o; pbm = f, all_data.xP, σ_o, prior_θP, prior_θMs, helpers_channel)
    sample_initial_params = (rng=StableRNG(1234)) -> InitFromParams((
        θP = rand(init_θP),
        θMs = (rand(init_θMs)),
    ))
    initial_params = sample_initial_params()
    adtype = AutoForwardDiff(chunksize=12)
    #tmp = sample(model, NUTS(0, 0.65; adtype), 2; initial_params)
    #return nothing
    n_sample_NUTS = 700
    #n_sample_NUTS = 12
    chn = sample(model, NUTS(8, 0.65), MCMCThreads(), ceil(Integer,n_sample_NUTS/n_threads), 
        n_threads; initial_params=map(_ -> sample_initial_params(rng),1:n_threads))
    # chn = sample(model, NUTS(8, 0.65), MCMCThreads(), ceil(Integer,n_sample_NUTS/n_threads), 
    #     n_threads; initial_params=fill(initial_params, n_threads))
    #fname = "intermediate/doubleMM_chain_rPMs2_$(last(HVI._val_value(scenario))).jld2" # LogNormal prior
    train_suffix = is_invert_train_only ? "trainonly_" : ""
    fname_theta = "intermediate/doubleMM_chain_theta_$(train_suffix)$(last(HVI._val_value(scenario))).jld2" # Uniform prior
    jldsave(fname_theta, false, IOStream; chn)
    @info "saved chn to $fname_theta"
end
#sample_fsitesθ()

function sample_fsitesθ_cont(rng = StableRNG(0815))
    @info "sample_fsitesθ_cont"
    include("scratch_include.jl")
    fname_chn = "intermediate/doubleMM_chain_theta_$(last(HVI._val_value(scenario))).jld2" # Uniform prior
    chn = load(fname_chn, "chn"; iotype = IOStream);
    n_iter = iter = FlexiChains.niters(chn)
    n_chain = min(n_threads, FlexiChains.nchains(chn))
    # extract then last sample from each chain    
    initial_params = map(1:n_chain) do i_chain
        InitFromParams(NamedTuple(map(FlexiChains.parameters(chn)) do par
            Symbol(par) => chn[par, iter=n_iter, chain=i_chain]
        end))
    end
    model = fsitesθ(y_o; pbm = f, xP, σ_o, prior_θP, prior_θMs, helpers_channel)
    adtype = AutoForwardDiff(chunksize=12)
    #tmp = sample(rng, model, NUTS(0, 0.65; adtype), 2; initial_params=initial_params[1])
    n_sample_NUTS = 2_000
    #n_sample_NUTS = 16
    chn = sample(rng, model, NUTS(8, 0.65), MCMCThreads(), ceil(Integer,n_sample_NUTS/n_chain), 
        n_threads; initial_params)
    fname_thetac = "intermediate/doubleMM_chain_thetac_$(last(HVI._val_value(scenario))).jld2" # Uniform prior
    jldsave(fname_thetac, false, IOStream; chn)
    @info "saved chn to $fname_thetac"
    #chn12 = cut2Ms(chn)
    #Makie.plot(chn12)
end
#sample_fsitesθ_cont()

# start sampling theta but start from sample of zeta2
function sample_fsitesθ2()
    @info "sample_fsitesθ2"
    include("scratch_include.jl")
    fname_chn = "intermediate/doubleMM_chain_zeta_normal_$(last(HVI._val_value(scenario))).jld2" # Uniform prior
    chn = merge_chains(load(fname_chn, "chn"; iotype = IOStream));
    chnt = compute_θchain(chn)
    n_iter = FlexiChains.niters(chn)
    init_θP = (iter) -> chnt[@vn(θP), iter=iter, chain=1]
    init_θMs = (iter) -> chnt[@vn(θMs), iter=iter, chain=1]
    rng3 = StableRNG(0815)
    init_θMs(3)
    sample_initial_params = (rng) -> begin
        iter = rand(rng, 1:n_iter) # needs to be consistent across P and Ms
        InitFromParams((
            θP = init_θP(iter),
            θMs = init_θMs(iter),
        ))
    end
    initial_params_ = map(_ ->sample_initial_params(rng3), 1:n_threads)
    model = fsitesθ(y_o; pbm = f, xP, σ_o, prior_θP, prior_θMs, helpers_channel)
    adtype = AutoForwardDiff(chunksize=12)
    #tmp = sample(rng3, model, NUTS(0, 0.65; adtype), 2; initial_params=sample_initial_params(rng3))
    n_sample_NUTS = 700
    #n_sample_NUTS = 16
    chn = sample(rng3, model, NUTS(8, 0.65), MCMCThreads(), ceil(Integer,n_sample_NUTS/n_threads), 
        n_threads; initial_params=map(_ -> sample_initial_params(rng3),1:n_threads))
    fname_theta2 = "intermediate/doubleMM_chain_theta2_$(last(HVI._val_value(scenario))).jld2" # Uniform prior
    jldsave(fname_theta2, false, IOStream; chn)
    @info "saved chn to $fname_theta2"
end
#sample_fsitesθ2()

# fsitesζ: sample ζ but apply adjustment of logdensity for the back-transformation
function sample_fsitesζ()
    @info "sample_fsitesζ"
    include("scratch_include.jl")
    model = fsitesζ(y_o; pbm = f, xP, σ_o, prior_ζP, prior_ζMs, helpers_channel)
    sample_initial_params = (rng=StableRNG(1234)) -> InitFromParams((
        ζP = log.(rand(init_θP)),
        ζMs = log.(rand(init_θMs)),
    ))
    initial_params = sample_initial_params()
    adtype = AutoForwardDiff(chunksize=12)
    #tmp = sample(model, NUTS(0, 0.65; adtype), 2; initial_params)
    #return nothing
    n_sample_NUTS = 1_500
    #n_sample_NUTS = 16
    chn = sample(model, NUTS(8, 0.65), MCMCThreads(), ceil(Integer,n_sample_NUTS/n_threads), 
        n_threads; initial_params=map(_ -> sample_initial_params(rng),1:n_threads))
    # chn = sample(model, NUTS(8, 0.65), MCMCThreads(), ceil(Integer,n_sample_NUTS/n_threads), 
    #     n_threads; initial_params=fill(initial_params, n_threads))
    #fname = "intermediate/doubleMM_chain_rPMs2_$(last(HVI._val_value(scenario))).jld2" # LogNormal prior
    fname_zeta = "intermediate/doubleMM_chain_zeta_$(last(HVI._val_value(scenario))).jld2" # Uniform prior
    jldsave(fname_zeta, false, IOStream; chn)
    @info "saved chn to $fname_theta"
end
#sample_fsitesζ()

# same as fsitesζ but deviding transformation correction of ζMs by n_indiv
function sample_fsitesζ2()
    @info "sample_fsitesζ2"
    include("scratch_include.jl")
    local rng3, model, sample_initial_params, initial_params, adtype, n_sample_NUTS, chn
    model = fsitesζ2(y_o; pbm = f, xP, σ_o, prior_ζP, prior_ζMs, helpers_channel)
    rng3 = StableRNG(1234)
    sample_initial_params = (rng=StableRNG(1234)) -> InitFromParams((
        ζP = log.(rand(rng, init_θP)),
        ζMs = log.(rand(rng, init_θMs)),
    ))
    initial_params = sample_initial_params(rng3)
    adtype = AutoForwardDiff(chunksize=12)
    #tmp = sample(model, NUTS(0, 0.65; adtype), 2; initial_params)
    #return nothing
    n_sample_NUTS = 1_500
    #n_sample_NUTS = 16
    chn = sample(rng3, model, NUTS(8, 0.65), MCMCThreads(), ceil(Integer,n_sample_NUTS/n_threads), 
        n_threads; initial_params=map(_ -> sample_initial_params(rng3),1:n_threads))
    # chn = sample(model, NUTS(8, 0.65), MCMCThreads(), ceil(Integer,n_sample_NUTS/n_threads), 
    #     n_threads; initial_params=fill(initial_params, n_threads))
    #fname = "intermediate/doubleMM_chain_rPMs2_$(last(HVI._val_value(scenario))).jld2" # LogNormal prior
    fname_zeta2 = "intermediate/doubleMM_chain_zeta2_$(last(HVI._val_value(scenario))).jld2" # Uniform prior
    jldsave(fname_zeta2, false, IOStream; chn)
    @info "saved chn to $fname_theta"
end
#sample_fsitesζ2()


() -> begin # explore FlexiChains    
    keys(chn)
    FlexiChains.parameters(chn)
    chn[:rMs] # same as stack = false
    chn[:rMs, stack=false] # 2x8 DimArray of DimArray 2 x 800
    chn[:rMs, stack=true] # single DimArray 2x8x2x800
    chn[@vn(rMs[1,:]), stack=true]
    #chnm = FlexiChains.merge_chains(chn)
    # combine into a single chain, looses information in iter
    n_sample_, n_chain_ = size(chn[:rMs]) # same as stack = false
    chnm = vcat((chn[chain=i] for i in 1:n_chain_)...)
    chnm[iter = 1:7]
    chnm[@vn(rMs[1,:]), stack=true][:,1,:] # resid_θ[1]: n_sample x n_indiv 
    chnm[@vn(rMs[:,1]), stack=true][:,1,:] # site 1: θ: n_sample x θ

    x = DimArray(chn_s1)[:,1,:]
    # @usingany UnicodePlots

    sm = summarystats(chn)
    keys(sm)
    sm[@vn(rP[1])]
    n_sample_, n_chain_ = size(chn[:rMs]) # same as stack = false

    pws = FlexiChains.values_at(chn; iter=2, chain=1)
    FlexiChains.values_at(chn; iter=2, chain=1)

    chnt = FlexiChains.transform_values(chnm,
        @vn(rP) => (rP -> θP_true .* rP) => @vn(θP),
        @vn(rMs) => (rMs -> θMs_true_train .* rMs) => @vn(θMs),
    )
    chnt[:θP]
    chnt[@vn(θMs[:,1])]
    chn_s1 = chnt[[@vn(θP), @vn(θMs[:,1])]]
    # using 
    plot(chn_s1)
    

end





# """
# ζMs in chn are all first parameter, all second parameters, ...
# while ζMsin HVI and f are columns for each site, need to transform, back and forth
# """
# transposeMs = (ζ, intm_PMs, back=false) -> begin
#     ζc = intm_PMs(ζ)
#     Ms = back ? reshape(ζc.Ms, reverse(size(ζc.Ms))) : ζc.Ms
#     ζct = vcat(CA.getdata(ζc.P), vec(CA.getdata(Ms)'))
# end
# # θ_true = vcat(CA.getdata(θP_true), vec(CA.getdata(θMs_true_train)));
# # ζ_true = log.(θ_true);
# θ0_true = vcat(CA.getdata(θP_true), vec(CA.getdata(θMs_true_train'))); # note the transpose

# ζ0_true = log.(θ0_true);
# #transposeMs(θ0_true, intm_PMs_gen, true) == θ_true

# # mle_estimate = optimize(model, MLE(), θ_ini)
# # mle_estimate.values

# # takes ~ 25 minutes
# #n_sample_NUTS = 800
# n_sample_NUTS = 2000
# tmp = sample(model, NUTS(0,0.65), 2, initial_params = ζ0_true .+ 0.001)
# #chn = sample(model, NUTS(), n_sample_NUTS, initial_params = ζ0_true .+ 0.001)
# #n_sample_NUTS = 24
# n_threads = 8
# chn = sample(model, NUTS(), MCMCThreads(), ceil(Integer,n_sample_NUTS/n_threads), 
#     n_threads, initial_params = fill(ζ0_true .+ 0.001, n_threads))


using AlgebraOfGraphics, CairoMakie, FigureHelpers
const aog = AlgebraOfGraphics
using ColorBrewer: ColorBrewer
using CategoricalArrays
using PairPlots
using DataFrames


function compare_ζcorrection(rng_seed = 118)
    pt = get_hybridproblem_par_templates(prob0)
    keysP = keys(pt.θP)
    keysM = keys(pt.θM)
    fname_chains = (
        #mc_ζ_uncor = "intermediate/doubleMM_chain_zeta_normal_$(last(HVI._val_value(scenario))).jld2",
        #mc_ζflat = "intermediate/doubleMM_chain_zeta_normal_$(last(HVI._val_value(scenario))).jld2",
        #mc_ζ = "intermediate/doubleMM_chain_zeta_$(last(HVI._val_value(scenario))).jld2",
        #mc_ζ2 = "intermediate/doubleMM_chain_zeta2_$(last(HVI._val_value(scenario))).jld2",
        mc_θ_all = "intermediate/doubleMM_chain_theta_$(last(HVI._val_value(scenario))).jld2",
        mc_θ_train = "intermediate/doubleMM_chain_theta_trainonly_$(last(HVI._val_value(scenario))).jld2",
        #mc_θ = "intermediate/doubleMM_chain_thetac_$(last(HVI._val_value(scenario))).jld2",
        # from different starting conditions
        #mc_θ2 = "intermediate/doubleMM_chain_theta2_$(last(HVI._val_value(scenario))).jld2",
    )
    chns = map(fname_chn -> load(fname_chn, "chn"; iotype = IOStream), fname_chains);
    chnst = map(compute_θchain, chns);
    chnst12 = map(cut2Ms, chnst);
    # keysP = keys(hp_samp.θsP[:,1])
    # keysM = keys(hi_agg_samp.θsM[:,1,1])
    include("scratch_include.jl")
    i_indiv_plot = 1:2
    chnst12 = map(chn -> cut2Ms(chn, keysP, keysM; ichains=i_indiv_plot), chnst);
    chnst12_test = map(chn -> cut2Ms(chn, keysP, keysM; ichains=i_indiv_plot, offset_chains = size(θMs_true_train,2)), chnst);
    # map(keys, chnst12)
    # (chn12, label) = first(zip(chnst12, keys(chnst12)))
    chn12 = first(chnst12)
    chn12_test = first(chnst12_test)
    pairplot(( 
        PairPlots.Series(DataFrame(chn12)[:,3:end], label=string(label))
        for (chn12, label) in zip(chnst12, keys(chnst12)))...)
    sms = map(chn -> summarystats(chn, split_varnames=false), chnst);
    #sm = sms[1]
    stdsP = map(sm -> sm[@vn(θP), stat=:std], sms)
    values(stdsP) ./ vec(stdsP[2])
    meansM = map(sm -> sm[@vn(θMs), stat=:mean], sms)
    stdsM = map(sm -> sm[@vn(θMs), stat=:std], sms)
    i_test = n_indiv_train .+ (1:n_indiv_test)
    i_par = 1
    i_par = 2
    mean_label = string(keys(θM)[i_par])
    #fig = Makie.scatter(θMs_true[1,:], meansM.mc_θ_all[1,:])
    fig = Makie.scatter(θMs_true[i_par,1:n_indiv_train], meansM.mc_θ_all[i_par,1:n_indiv_train];
        axis=(; xlabel=mean_label * " true", ylabel=mean_label * " MC all sites" ))
    Makie.ablines!(fig.axis, 0,1)
    Makie.scatter!(fig.axis, θMs_true[i_par,i_test], meansM.mc_θ_all[i_par,i_test], color="maroon")
    fig
    # two inversions yield almost the same mean and also std the train indivs
    #   with a few outliers of larger 
    fig = Makie.scatter(meansM.mc_θ_train[i_par,:], meansM.mc_θ_all[i_par,1:n_indiv_train];
        axis=(; xlabel="mean(" * mean_label * ") MC train sites", ylabel="mean(" * mean_label * ") MC all sites" ))
    Makie.ablines!(fig.axis, 0,1)
    fig
    #
    fig = Makie.scatter(stdsM.mc_θ_train[i_par,:], stdsM.mc_θ_all[i_par,1:n_indiv_train];
        axis=(; xlabel="std(" * mean_label * ") MC train sites", ylabel="std(" * mean_label * ") MC all sites" ))
    Makie.ablines!(fig.axis, 0,1)
    fig
    #
    # together with HVI
    hvi_scenarios = (
        hvi_batchall_covar = Val((:allsites, :covarK2)),
        hvi_batchall = Val((:allsites,)),
        hvi_batch10th_covar = Val((:covarK2,)),
        hvi_batch10th = Val(()),
        )
    combine_scenarios = (scenario::Val{scen}, hvi_scenario::Val{hvi_scen}) where {scen, hvi_scen} = Val((scen..., hvi_scen...))
    scenarios = map(s -> @inferred(combine_scenarios(scenario, s)), hvi_scenarios) 
    scenario_comb = @inferred Base.first(scenarios)
    #scenario_comb = @inferred first(Base.tail(scenarios))
    n_sample_pred = 400 #nrow(DataFrame(first(chnst12)))
    rng_seed = 815
    prob_elbo_dfs = map(scenarios) do scenario_comb
        @info "-- scenario = $scenario_comb"
        local rng1, prob, res_point, res_elbo
        rng1 = StableRNG(rng_seed)
        prob = create_problem(;scenario = scenario_comb)[1]
        # first fit the location
        res_point = fit_pointwise(rng1, prob; n_epoch = 10_000, patience = 200);    
        # UnicodePlots.scatterplot(Base.first.(res_point.loss_test_hist[3:end]), Base.last.(res_point.loss_test_hist[3:end]))
        # next fit the uncertainty, starting from som e
        res_elbo1 = interactive_elbo(rng1, prob; ϕ0c = res_point.ϕc_old, n_epoch=8_000, 
            patience = 200, η=0.0005,
            loss_mode = NegElboFixϕgμI(),
            );
        # UnicodePlots.scatterplot(Base.first.(res_elbo1.loss_test_hist), Base.last.(res_elbo1.loss_test_hist))
        #@assert res_elbo1.ϕc.ϕg == res_point.ϕc.ϕg
        # fit both together, with higher N_MC
        res_elbo = interactive_elbo(rng1, prob; ϕ0c = res_elbo1.ϕc, n_epoch=8_000, 
            patience = 200, η=0.0005,
            n_MC_val=Val(60)
            );
        # UnicodePlots.scatterplot(Base.first.(res_elbo.loss_test_hist), Base.last.(res_elbo.loss_test_hist))
        # res_elbo.ϕc.ϕqP.μζP[1] = log.(2.0) # try setting to true value but goes back
        #res_elbo = interactive_elbo(ϕ0c = res_elbo.ϕc, n_epoch=400); # overfitting?
        #res_elbo = interactive_elbo(ϕ0c = res_elbo.ϕc, n_epoch=2_000); # overfitting?
        hp_samp, hi_agg_samp = HVI.sample_posterior(rng1, prob, res_elbo.ϕc;
            n_sample_pred,
            );
        df_hvi = hcat(
            DataFrame(hp_samp.θsP', collect(keys(CA.getaxes(hp_samp.θsP)[1]))),
            (DataFrame(hi_agg_samp.θsM[:,:,i]', collect(string.(keys(CA.getaxes(hi_agg_samp.θsM)[1])) .* "_$i")) for i in i_indiv_plot)...,
        );
        _, batchsize = get_hybridproblem_n_indiv_train_and_batch(prob0)
        n_indiv_test = size(test_data[1],2)
        # if (n_indiv_test % batchsize) != 0
        #     @warn("Number of siztes in testdata=$n_indiv_test is not a "*
        #         "multiple of problems batchsize=$batchsize. Using all sites as " *
        #         "one batch.")
        #     batchsize = n_indiv_test
        # end
        test_dl = DataLoader(values(test_data); batchsize, partial=false)
        hp_samp_t, hi_agg_samp_t = HVI.sample_posterior(rng1, prob, res_elbo.ϕc, test_dl;
            n_sample_pred,
            );
        df_hvi_test = hcat(
            DataFrame(hp_samp_t.θsP', collect(keys(CA.getaxes(hp_samp_t.θsP)[1]))),
            (DataFrame(hi_agg_samp_t.θsM[:,:,i]', collect(string.(keys(CA.getaxes(hi_agg_samp_t.θsM)[1])) .* "_$i")) for i in i_indiv_plot)...,
        );
        (; prob, res_elbo, df_hvi, df_hvi_test, res_point, res_elbo1)
    end;
    probs = map(x -> x.prob, prob_elbo_dfs);
    res_elbos = map(x -> x.res_elbo, prob_elbo_dfs);
    () -> begin
        res_elbos1 = map(x -> x.res_elbo1, prob_elbo_dfs);
        dfs_hvi = NamedTuple{keys(res_elbos1)}(map(zip(probs, res_elbos1)) do (prob, res_elbo)
            hp_samp, hi_agg_samp = HVI.sample_posterior(rng1, prob, res_elbo.ϕc;
                n_sample_pred,
                );
            df_hvi = hcat(
                DataFrame(hp_samp.θsP', collect(keys(CA.getaxes(hp_samp.θsP)[1]))),
                DataFrame(hi_agg_samp.θsM[:,:,1]', collect(string.(keys(CA.getaxes(hi_agg_samp.θsM)[1])) .* "_1")),
                DataFrame(hi_agg_samp.θsM[:,:,2]', collect(string.(keys(CA.getaxes(hi_agg_samp.θsM)[1])) .* "_2")),
            );
        end);
    end
    dfs_hvi = map(x -> x.df_hvi, prob_elbo_dfs);
    dfs_hvi_test = map(x -> x.df_hvi_test, prob_elbo_dfs);
    # :hvi_batchall_covar 
    dfs_hvi_plot = dfs_hvi; #dfs_hvi[(:hvi_batchall, :hvi_batch10th, :hvi_batch10th_covar)];
    fig = pairplot(
        (PairPlots.Series(DataFrame(chn12)[:,3:end], label=string(label))
        for (chn12, label) in zip(chnst12, keys(chnst12)))..., 
        (PairPlots.Series(df_hvi, label=string(label))
        for (df_hvi, label) in zip(dfs_hvi_plot, keys(dfs_hvi_plot)))..., 
        PairPlots.Truth(NamedTuple{map(Symbol, (keys(first(chnst12))...,))}(
            vcat(θP_true, vec(θMs_true[:,i_indiv_plot]))),
            label="True Value")            
        );
    Label(fig[0,:], "Comparing posterior on training data")        
    fig
    () -> begin
        save("intermediate/pairplots_omitr0.svg", fig)
    end
    fig_test = pairplot(
        (PairPlots.Series(DataFrame(chn12)[:,3:end], label=string(label))
        for (chn12, label) in zip(chnst12_test, keys(chnst12_test)))..., 
        (PairPlots.Series(df_hvi, label=string(label))
        for (df_hvi, label) in zip(dfs_hvi_test, keys(dfs_hvi_test)))..., 
        PairPlots.Truth(NamedTuple{map(Symbol, (keys(first(chnst12))...,))}(
            vcat(θP_true, vec(θMs_true[:,i_indiv_plot .+ n_indiv_train]))),
            label="True Value")            
        );
    Label(fig_test[0,:], "Comparing posterior on test data")        
    fig_test
    () -> begin
        save("intermediate/pairplots_test_omitr0.svg", fig_test)
    end
end

function interactive_after_sample() 
    fname_chn = "intermediate/doubleMM_chain_rPMs_agg_$(last(HVI._val_value(scenario))).jld2" # Uniform prior
    fname_chn = "intermediate/doubleMM_chain_zeta_normal_$(last(HVI._val_value(scenario))).jld2" # Uniform prior
    fname_chn = "intermediate/doubleMM_chain_zeta_$(last(HVI._val_value(scenario))).jld2" # Uniform prior
    fname_chn = "intermediate/doubleMM_chain_zeta2_$(last(HVI._val_value(scenario))).jld2" # Uniform prior
    fname_chn2 = "intermediate/doubleMM_chain_theta_$(last(HVI._val_value(scenario))).jld2" # Uniform prior
    fname_chn2 = "intermediate/doubleMM_chain_thetac_$(last(HVI._val_value(scenario))).jld2" # Uniform prior
    chn = load(fname_chn, "chn"; iotype = IOStream);
    chn2 = load(fname_chn2, "chn"; iotype = IOStream);
    n_sample_NUTS = prod(size(Array(chn))[1:2])
    chn2_12 = cut2Ms(chn2) 

    # graph and check convergence
    include("scratch_include.jl")
    
    chn12 = cut2Ms(chn)
    Makie.plot(chn2_12)
    chn2_12 = cut2Ms(chn2) #chn2[[@vn(θP), @vn(θMs[:,1]), @vn(θMs[:,2])]]
    θP_true
    θMs_true_train[:,1:2]
    log.(θMs_true_train[:,1:2])
    log.(θP_true)
    FlexiChains.Makie.meanplot(chn12) # larger uncertainty at the start
    # slow mixing of chains, and posterior of 0.86 rather than 1 for thetaMs[1,2]
    Makie.plot(chn12[iter=1:150])
    Makie.plot(chn12[iter=10:End])
    # remove burnin
    chn = chn[iter = 10:End]
    #chn = chn[iter = 100:End]

    # merge chains and recompute θ
    n_sample_, n_chain_ = FlexiChains.niters(chn), FlexiChains.nchains(chn)
    chnt2 = merge_chains(chn2)
    chnm = merge_chains(chn)
    chnt = compute_θchain(chnm)
    # inspect mean
    chn12 = chnt[[@vn(θP), @vn(θMs[:,1]), @vn(θMs[:,2])]]
    chn2_12 = chnt2[[@vn(θP), @vn(θMs[:,1]), @vn(θMs[:,2])]]
    pairplot(DataFrame(chn12)[:,3:end], DataFrame(chn2_12)[:,3:end])
    pairplot(chnt2)
    pairplot(chn2_12)
    θMs_true_train[:,1]
    θMs_true_train[:,2]

    # compute statistics
    sm = summarystats(chn12)    

    l1 = Long(chn1);
    d = data(l1)
    m = mapping(:value, layout=:param => presorted)
    v = AlgebraOfGraphics.density()
    draw(d * m * v)
    Makie.plot()


    () -> begin
        #using JLD2
        fname = "intermediate/doubleMM_chain_zeta_$(last(HVI._val_value(scenario))).jld2"
        jldsave(fname, false, IOStream; chn)
        chn = load(fname, "chn"; iotype = IOStream);
        n_sample_NUTS = size(Array(chn),1)
    end

    () -> begin # load HMC sample for K1global scenario
        #using JLD2
        fname_K1global = "intermediate/doubleMM_chain_zeta_K1global.jld2"
        chain_K1global = load(fname_K1global, "chn"; iotype = IOStream);
        ζsP_hmc_K1global = Array(chain_K1global)[:,1:n_θP]'
        ζsMst_hmc_K1global = reshape(Array(chain_K1global)[:,(n_θP+1) : end], n_sample_NUTS, n_indiv_all, n_θM)
        ζsMs_hmc_K1global = permutedims(ζsMst_hmc_K1global, (2,3,1))
    end

    #ζi = first(eachrow(Array(chn)))
    pbm_spec = get_hybridproblem_PBModelSpec(prob; scenario)
    f = TODO
    f_allsites = create_nsite_applicator(f, n_indiv_all)
    #ζs = mapreduce(ζi -> transposeMs(ζi, intm_PMs_gen, true), hcat, eachrow(Array(chn)));
    ζsP = Array(chn)[:,1:n_θP]'
    ζsMs = reshape(Array(chn)[:,(n_θP+1) : end], n_sample_NUTS, n_indiv_all, n_θM)
    # need to reshape according to generate_ζ, with columns of parameters
    ζsMs_tr = permutedims(ζsMs, (2,3,1))
    ζsMs_tr[:,:,1]  # first sample: n_site x n_par
    ζsMs_tr[:,1,:]  # first parameter n_site x n_sample 

    trans_mP=StackedArray(transP, size(ζsP, 2))
    trans_mMs=StackedArray(transM, size(ζsMs_tr, 1) * size(ζsMs_tr, 3))
    θsP, θsMs_tr = transform_ζs(ζsP, ζsMs_tr; trans_mP, trans_mMs)
    y = f(θsP, θsMs_tr, f, xP) 
    #(; y, θsP, θsMs_tr) = HVI.apply_f_trans(ζsP, ζsMs, f_allsites, xP; transP, transM);
    (y_hmc, θsP_hmc, θsMs_hmc) = (; y, θsP, θsMs_tr);

        
    () -> begin # check that the model predicts the same as HVI-code
        _parnames = Symbol.(vcat([
            "ζP[1]"], ["ζMs[1, :][$i]" for i in i_indiv_train], ["ζMs[2, :][$i]" for i in i_indiv_train]))
        #chain0 = Chains(transposeMs(ζ_true, intm_PMs_gen)', _parnames);
        chain0 = Chains(ζ0_true', _parnames); # transpose here only for chn array
        y0inv = generated_quantities(model, chain0)[1, 1]
        y0pred = HVI.apply_f_trans(ζ_true, xP, f, trans_PMs_gen, intm_PMs_gen)[2]
        y0pred .- y_true 
        y0inv .- y_true
    end

    () -> begin # plot chn
        #@usingany FigureHelpers, CairoMakie
        # θP and first θMs_tr 
        ch = chn[:,vcat(1:n_θP, n_θP+1, n_θP+n_indiv_all+1),:];
        fig = plot_chn(ch)
        save("tmp.svg", fig)
    end

    mean_y_invζ = mean_y_hmc = map(mean, eachslice(y_hmc; dims = (1, 2)));
    #describe(mean_y_pred - y_o)
    histogram(vec(mean_y_invζ) - vec(y_true)) # predictions centered around y_o (or y_true)
    plt = scatterplot(vec(y_true), vec(mean_y_invζ));
    lineplot!(plt, 0, 1)
    mean(mean_y_invζ - y_true) # still ok

    # first site, first prediction
    y_inv11 = y[1, 1, :]
    histogram(y_inv11 .- y_o[1, 1])
    histogram(y_inv11 .- y_true[1, 1])  

    histogram(ζsP[1, :])
    describe(pdf.(Ref(prior_ζ), ζsP[1, :]))  # only small differences 
    pdf(prior_ζ, log(θP_true[1]))

    mean_θP = CA.ComponentArray(mean(θsP; dims = 2)[:, 1], CA.getaxes(θP_true))
    mean_θMs = CA.ComponentArray(mean(θsMs_tr; dims = 3)[:,:, 1], CA.getaxes(θMs_true_train'))
    histogram(θsP .- θP_true)  # all overestimated ? 
    plt = scatterplot(θMs_true_train'[:,1], mean_θMs[:,1]);
    lineplot!(plt, 0, 1)
    plt = scatterplot(θMs_true_train'[:,2], mean_θMs[:,2]);
    lineplot!(plt, 0, 1)

    #------------------ compare HVI vs HMC sample
    # reload results from run without covars, see above
    () -> begin   # compare against HVI sample 
        #@usingany AlgebraOfGraphics, FigureHelpers, TwPrototypes, CairoMakie, DataFrames
        #@usingany LaTeXStrings
        makie_config = ppt_MakieConfig(fontsize=14) # decrease standard font from 18 to 14
        paper_config = paper_MakieConfig() 
        set_default_AoGTheme!(;makie_config)

        # two same colors for hmc and hvi , additional for further unspecified labels
        cDark2 = cgrad(ColorBrewer.palette("Dark2",3),3,categorical=true)
        #color_methods = vcat([k => col for (k, col) in zip([:hmc, :hvi], cDark2[1:2])], cDark2[3], Makie.wong_colors()[2:end]);
        cpal0 = Makie.wong_colors()
        color_methods = vcat([k => col for (k, col) in zip([:hmc, :hvi], cpal0[1:2])], cpal0[3:end]);

        function lower_lastdigits(sym::Symbol,n_digit=1)
            s = string(sym)
            latexstring(s[1:(end-n_digit)] * "_" * s[(end-n_digit+1):end])
        end
        function get_fig_size(cfg; width2height=golden_ratio, xfac=1.0)    
            cfg = makie_config
            x_inch = first(cfg.size_inches) * xfac
            y_inch = x_inch / width2height
            72 .* (x_inch, y_inch) ./ cfg.pt_per_unit # size_pt        
        end

        ζsP_hvi = log.(θsP2)
        ζsP_hvi_indep = log.(θsP2_indep) 
        ζsP_hvi_neglect_cor = log.(θsP2_neglect_cor) 
        ζsP_hvi_K1global = log.(θsP2_K1global) 
        ζsP_hmc = log.(θsP_hmc)
        ζsMs_hvi = log.(θsMs2_tr)
        ζsMs_hvi_indep = log.(θsMs2_tr_indep) 
        ζsMs_hvi_neglect_cor = log.(θsMs2_tr_neglect_cor) 
        ζsMs_hvi_K1global = log.(θsMs2_tr_K1global) 
        ζsMs_hmc = log.(θsMs_hmc)
        # int_pms = interpreters.PMs
        # par_pos = int_pms(1:length(int_pms))
        #itrain_sites = 1:10
        itrain_sites = 1:5
        #itrain_sites = 6:10
        #itrain_sites = 11:15
        scen = vcat(
            fill(:hvi,size(ζsP_hvi,2)),
            fill(:hmc,size(ζsP_hmc,2)),
            fill(:hvi_indep,size(ζsP_hvi_indep,2)),
            fill(:neglect_cor,size(ζsP_hvi_neglect_cor,2)),
            )
        dfP = mapreduce(vcat, axes(θP_true,1)) do i_par
            #pos = par_pos.P[i_par]
            DataFrame(
                value = vcat(
                    ζsP_hvi[i_par, :], ζsP_hmc[i_par,:], 
                    ζsP_hvi_indep[i_par,:], ζsP_hvi_neglect_cor[i_par,:]), 
                variable = lower_lastdigits.(keys(θP_true)[i_par]),
                site = "site $(itrain_sites[1])",
                Method = scen
            )
        end 
        dfMs = mapreduce(vcat, itrain_sites) do i_site 
            mapreduce(vcat, axes(θM,1)) do i_par
                #pos = par_pos.Ms[i_par, i_site]
                DataFrame(
                    value = vcat(
                        ζsMs_hvi[i_site,i_par,:], 
                        ζsMs_hmc[i_site,i_par,:], 
                        ζsMs_hvi_indep[i_site,i_par,:],
                        ζsMs_hvi_neglect_cor[i_site,i_par,:],),
                    variable = lower_lastdigits.(keys(θM)[i_par]),
                    site = "site $(i_site)",
                    Method = scen
                )
            end
        end        
        df = vcat(dfP, dfMs)
        df_true = vcat(
            mapreduce(vcat, axes(θP,1)) do i_par
                DataFrame(
                    value = ζP_true[i_par],
                    variable = lower_lastdigits.(keys(θP)[i_par]),
                    site = "site $(itrain_sites[1])",
                    Method = :true
                )
            end,
            mapreduce(vcat, itrain_sites) do i_site 
                mapreduce(vcat, axes(θM,1)) do i_par
                    DataFrame(
                        value = ζMs_true_train[i_par, i_site],
                        variable = lower_lastdigits.(keys(θM)[i_par]),
                        site = "site $(i_site)",
                        Method = :true
                    )
                end
            end        
        ) # vcat
        #cf90 = (x) -> quantile(x, [0.05,0.95])
        plot_par_densities = (dfs; makie_config = makie_config) -> begin
            plt = (data(dfs) * mapping(:value=> (x -> x ) => "", color=:Method) * AlgebraOfGraphics.density(datalimits=extrema) +
                data(df_true) * mapping(:value => "") * visual(VLines; color=:blue, linestyle=:dash)) * 
                mapping(col=:variable => sorter(lower_lastdigits.(vcat(keys(θP)..., keys(θM)...))), 
                    row = (:site => nonnumeric)) 
                #mapping(col=:variable, row = (:site => nonnumeric)) 
            #fig = Figure(size = get_fig_size(makie_config, xfac=1, width2height = 1/2)); # 10 sites
            fig = figure_conf(1.0; makie_config);
            ffig = draw!(fig, plt, 
                facet=(; linkxaxes=:minimal, linkyaxes=:none,), 
                axis=(xlabelvisible=false,yticklabelsvisible=false),
                scales(Color = (; palette = color_methods)),
                );
            legend!(fig[length(itrain_sites),1], ffig, ; tellwidth=false, halign=:left, valign=:bottom , margin=(10, 10, 10, 10))
            fig
        end
        () -> begin
            save_with_config(joinpath(pwd(), "tmp.svg"), fig; makie_config = MakieConfig())
            save_with_config(joinpath(pwd(), "tmp"), fig; makie_config = paper_config)
            save_with_config("tmp", fig) # returns path in tmp to click on
        end
        fig = plot_par_densities(subset(df, :Method => ByRow(∈((:hvi,:hmc)))))
        save_with_config("intermediate/compare_hmc_hvi_sites_$(last(HVI._val_value(scenario)))", fig; makie_config)

        fig = plot_par_densities(subset(df, :Method => ByRow(∈((:hmc,:neglect_cor)))))
        save("tmp.svg", fig)
        save_with_config("intermediate/compare_hmc_neglectcor_sites_$(last(HVI._val_value(scenario)))", fig; makie_config)

        fig = plot_par_densities(subset(df, :Method => ByRow(∈((:hvi,:hvi_indep)))))
        save("tmp.svg", fig)
        save_with_config("intermediate/compare_hvi_indep_sites_$(last(HVI._val_value(scenario)))", fig; makie_config)

        # 
        # compare density of predictions
        y_hvi = y2
        i_obss = [1,4,8]
        #i_obss = 1:8
        dfy = mapreduce(vcat, i_obss) do i_obs
                mapreduce(vcat, itrain_sites) do i_site
                    vcat(
                        DataFrame(
                            value = y_hmc[i_obs,i_site,:], 
                            site = "site $(i_site)",
                            Method = :hmc,
                            variable = :y,
                            i_obs = i_obs,
                            y_i = latexstring("y_$(i_obs)"),
                        ),
                        DataFrame(
                            value = y_hvi[i_obs,i_site,:], 
                            site = "site $(i_site)",
                            Method = :hvi,
                            variable = :y,
                            i_obs = i_obs,
                            y_i = latexstring("y_$(i_obs)"),
                        )
                    )# vcat
                end
        end
        dfyt = mapreduce(vcat, i_obss) do i_obs
                mapreduce(vcat, itrain_sites) do i_site
                    vcat(
                        DataFrame(
                            value = y_true[i_obs,i_site], 
                            site = "site $(i_site)",
                            Reference = :truth,
                            variable = :y,
                            i_obs = i_obs,
                            y_i = latexstring("y_$(i_obs)"),
                        ),
                        DataFrame(
                            value = y_o[i_obs,i_site,:], 
                            site = "site $(i_site)",
                            Reference = :obs,
                            variable = :y,
                            i_obs = i_obs,
                            y_i = latexstring("y_$(i_obs)"),
                        )
                    )# vcat
                end
        end
        DataFrames.transform!(dfyt, :Reference => (x -> categorical(string.(x); ordered = true, levels = ["truth", "obs"])) => :Reference) #  

        plt = (data(dfy) * mapping(color=:Method) * AlgebraOfGraphics.density(datalimits=extrema) +
                data(dfyt) * mapping(color=:Reference => AlgebraOfGraphics.scale(:Reference)) * visual(VLines;  linestyle=:dash)) * 
                #data(dfyt) * mapping(linestyle=:Reference => AlgebraOfGraphics.scale(:Reference)) * visual(VLines;  linestyle=:dash)) * 
                #data(dfyt) * mapping(color=:Reference => AlgebraOfGraphics.scale(:Reference), 
                #    linestyle= :Reference => AlgebraOfGraphics.scale(:Reference)) * visual(VLines)) * # bug?
            mapping(:value=>"", col=:y_i, row = :site)

        #fig = Figure(size = get_fig_size(makie_config, xfac=1, width2height = 1/2));
        fig = figure_conf(1; makie_config);
        ffig = draw!(fig, plt, 
            facet=(; linkxaxes=:minimal, linkyaxes=:none,), 
            axis=(xlabelvisible=false,yticklabelsvisible=false),
            scales(Color = (; palette = color_methods)),
            );
        #legend!(fig[1,3], f, ; tellwidth=false, halign=:right, valign=:top) # , margin=(-10, -10, 10, 10)
        legend!(fig[1,4], ffig, ; tellwidth=true, halign=:right, valign=:top) # , margin=(-10, -10, 10, 10)
        fig
        save("tmp.svg", fig)
        save_with_config("intermediate/compare_hmc_hvi_sites_y_$(last(HVI._val_value(scenario)))", fig; makie_config)
        # hvi predicts y better, hmc fails for quite a few obs: 3,5,6

        # compare mean_predictions
        mean_y_hvi = map(mean, eachslice(y_hvi; dims = (1, 2)));
        size(y_o)
        histogram(vec(mean_y_hvi .- y_o))
    end

    #------------------------------------- correlations -----------------
    """
    Compute standard deviation and correlation for predicted parameters on unconstrained scale.

    ## Arguments
    _ζsP: n_P x n_pred matrix of draws of predicted cross-sites parameters
    _ζsMs: n_site x n_M x n_pred of draws of predicted physical parameters

    returns sdP (n_P), sdMs (n_site x n_M), cor_PMs n_P + (n_M * length(itrain_sites)) square matrix
    """
    function compute_sd_cor_PMs(_ζsP, _ζsMs; i_sites_inspect = [1,2,3])
        mP = mean(_ζsP; dims=2)
        residP = _ζsP .- mP
        sdP = vec(std(residP; dims=2))
        mMs = mean(_ζsMs; dims=3)[:,:,1]
        residMs = _ζsMs .- mMs
        sdMs = std(residMs; dims=3)[:,:,1]
        residMst = permutedims(residMs[i_sites_inspect,:,:], (2,1,3)) # n_M x n_site x n_pred
        residPMst = vcat(residP, 
            reshape(residMst, size(residMst,1)*size(residMst,2), size(residMst,3))) # n_P x n_pred
        corPMs = cor(residPMst')
        sdP, sdMs, corPMs
    end

    function draw_cor_fig(cor, method; makie_config, par_names)
        fig = figure_conf(1.3, 0.8; makie_config);
        ax = Axis(fig[1,1], 
            xticklabelsvisible=false,yticklabelsvisible=true, 
            xticksvisible=false, yticksvisible=true,
            yticks = (axes(par_names,1), par_names),
            yreversed = true,
            aspect = 1,
            title = "Corr. $method")
        hm = heatmap!(ax, cor)
        rowsize!(fig.layout, 1, Aspect(1, 1))
        Colorbar(fig[1,2], hm ; tellwidth=true, tellheight=false)
        fig
    end

    () -> begin # inspect correlation of residuals
        # get true 
        i_sites_inspect = [1,2,3]
        sdP_hmc, sdMs_hmc, corPMs_hmc = compute_sd_cor_PMs(ζsP_hmc, ζsMs_hmc; i_sites_inspect)
        sdP_hvi, sdMs_hvi, corPMs_hvi = compute_sd_cor_PMs(ζsP_hvi, ζsMs_hvi; i_sites_inspect)
        sdP_hvi_indep, sdMs_hvi_indep, corPMs_hvi_indep = compute_sd_cor_PMs(
            ζsP_hvi_indep, ζsMs_hvi_indep; i_sites_inspect)
        sdP_hvi_neglect_cor, sdMs_hvi_neglect_cor, corPMs_hvi_neglect_cor = compute_sd_cor_PMs(
            ζsP_hvi_neglect_cor, ζsMs_hvi_neglect_cor; i_sites_inspect)
        sdP_hvi_K1global, sdMs_hvi_K1global, corPMs_hvi_K1global = compute_sd_cor_PMs(
            ζsP_hvi_K1global, ζsMs_hvi_K1global; i_sites_inspect)
        sdP_hmc_K1global, sdMs_hmc_K1global, corPMs_hmc_K1global = compute_sd_cor_PMs(
            ζsP_hmc_K1global, ζsMs_hmc_K1global; i_sites_inspect)
        # no correlations of K2(global) ML parameters in inversion?
        par_names = vcat(["global $k" for k in keys(θP)], vec(["site $i $k" for k in keys(θM), i in i_sites_inspect]))
        par_names_globalK1 = vcat(["global K1" for k in keys(θP)], vec(["site $i $k" for k in (:r1, :K2), i in i_sites_inspect]))
        fig = draw_cor_fig(corPMs_hmc, "Hamiltonian Monte Carlo posterior"; makie_config, par_names)
        save_with_config("intermediate/cor_hmc_$(last(HVI._val_value(scenario)))", fig; makie_config)
        fig = draw_cor_fig(corPMs_hvi, "Hybrid Variational Inference posterior"; makie_config, par_names)
        save_with_config("intermediate/cor_hvi_$(last(HVI._val_value(scenario)))", fig; makie_config)
        fig = draw_cor_fig(corPMs_hvi_neglect_cor, "HVI neglecting block correlations"; makie_config, par_names)
        save_with_config("intermediate/cor_hvi_neglect_cor_$(last(HVI._val_value(scenario)))", fig; makie_config)
        save_with_config("tmp", fig; makie_config)
        #
        fig = draw_cor_fig(corPMs_hvi_K1global, "HVI K1 global K2 site-dependent"; makie_config, par_names = par_names_globalK1)
        save_with_config("intermediate/cor_hvi_K1global_$(last(HVI._val_value(scenario)))", fig; makie_config)
        fig = draw_cor_fig(corPMs_hmc_K1global, "HMC K1 global K2 site-dependent"; makie_config, par_names = par_names_globalK1)
        save_with_config("intermediate/cor_hmc_K1global_$(last(HVI._val_value(scenario)))", fig; makie_config)


        df_sd = reduce(vcat, map(axes(θM,1)) do i_par 
            vcat(DataFrame(
            Method = :hmc,
            par = lower_lastdigits.(keys(θM)[i_par]),
            sd = sdMs_hmc[:,i_par],
            value = ζMs_true_train'[:,i_par],
        ), DataFrame(
            Method = :hvi,
            par = lower_lastdigits.(keys(θM)[i_par]),
            sd = sdMs_hvi[:,i_par],
            value = ζMs_true_train'[:,i_par],
        ), DataFrame(
            Method = :neglect_cor,
            par = lower_lastdigits.(keys(θM)[i_par]),
            sd = sdMs_hvi_neglect_cor[:,i_par],
            value = ζMs_true_train'[:,i_par],
        ),)
        end)
        plt = data(df_sd) * mapping(color=:Method=>"", row=:par) * 
            mapping(:value=>"", :sd => "Predicted Standard Deviation") * visual(Scatter, alpha = 0.5)

        #fig = Figure(size = get_fig_size(makie_config, xfac=1, width2height = 1/2));
        fig = figure_conf(1;makie_config);
        #fig = figure_conf(1;makie_config = paper_config);
        ffig = draw!(fig, plt, scales(Color = (; palette = color_methods)); 
            facet=(; linkxaxes=:none, linkyaxes=:none,),
            )
        #legend!(fig[1,1], ffig, ;tellheight=false, tellwidth=false, halign=:right, valign=:top, margin=(10, 10, 10, 10))
        legend!(fig[1,1], ffig, ;tellheight=false, tellwidth=false, halign=:left, valign=:bottom, margin=(10, 10, 10, 10))
        save_with_config("tmp", fig; makie_config)
        #save_with_config("tmp", fig; makie_config=paper_config)
    end

    () -> begin # inspect marginal variance

        
    end
        


    () -> begin # depr---- do an DEMC inversion of the PBM model with parameters at constrained scale
        # construct a Normal prior that ranges roughly across 1e-2 to 10
        # prior_θ = fit(LogNormal, @qp_ll(1e-2), @qp_uu(10))
        # prior_θn = (n) -> MvLogNormal(fill(prior_θ.μ, n), PDiagMat(fill(abs2(prior_θ.σ), n)))
        prior_θ = Normal(0, 10)
        prior_θn = (n) -> MvNormal(fill(prior_θ.μ, n), PDiagMat(fill(abs2(prior_θ.σ), n)))
        prior_θn(3)
        prob = HybridProblem(prob0);

        (; θM, θP) = get_hybridproblem_par_templates(prob; scenario)
        n_θM, n_θP = length.((θM, θP))
        pbm_spec = get_hybridproblem_PBModelSpec(prob; scenario)
        f = TODO

        @model function fsites_uc(
                y, ::Type{T} = Float64; f, n_θP, n_θM, σ_o, n_obs = length(σ_o)) where {T}
            n_obs, n_site = size(y)
            prior_θP = prior_θn(n_θP)
            prior_θM_sites = fill(prior_θn(n_site), n_θM)
            θP ~ prior_θP #MvNormal(n_θP, 10.0)
            # CAUTION: order of vectorizing matrix depends on order of ~
            # need to assign each variable in first site first, then second site, ...
            #   need to construct different MvNormal prior if std differs by variable
            # or need to take care when extracting samples or specifying initial conditions
            θMs = Matrix{T}(undef, n_θM, n_site)
            # the first loop vectorizes θMs by columns but is much slower
            # for i_site in 1:n_site
            #     ζMs[:, i_site] ~ prior_ζn(n_θM) #MvNormal(n_site, 10.0)
            # end
            # this loop is faster, but vectorizes θMs by rows in parameter vector
            for i_par in 1:n_θM
                θMs[i_par, :] ~ prior_θM_sites[i_par]
            end
            # this fills in rows first, but is also slower- why?    
            #ζMs[:] ~ prior_ζn(n_θM * n_site) 
            # assume σ_o known, see f_MM
            #σ_o ~ truncated(Normal(0, 1); lower=0)
            y_pred = f(θP, θMs', xP)[2] # first is global return
            #i_obs = 1
            for i_obs in 1:n_obs
                #pdf(MvNormal(y_pred[i_obs,:], σ_o[i_obs]),y[i_obs,:])
                y[i_obs, :] ~ MvNormal(y_pred[i_obs, :], σ_o[i_obs]) # single value σ instead of variance
            end
            #Main.@infiltrate_main # step to second time 
            # θMs_MCc[:,:,1] # checking row- or column-order of θMs
            # exp.(ζMs)
            y_pred
        end
        model_uc = fsites_uc(y_o; f, n_θP, n_θM, σ_o)

        () -> begin # check that the model predicts the same as HVI-code
            _parnames = Symbol.(vcat([
                "θP[1]"], ["θMs[1, :][$i]" for i in i_indiv_train], ["θMs[2, :][$i]" for i in i_indiv_train]))
            #chain0 = Chains(transposeMs(ζ_true, intm_PMs_gen)', _parnames);
            chain0 = Chains(θ0_true', _parnames); # transpose here only for chn array
            y0inv = generated_quantities(model_uc, chain0)[1, 1]
            y0pred = f(θP_true,  θMs_true_train, xP)[2]
            y0pred .- y_true 
            y0inv .- y_true
        end

        
        #θ0_true from above, tools from above

        # takes long
        n_sample_NUTS = 800
        #n_sample_NUTS = 24
        n_threads = 8
        chn = sample(model_uc, NUTS(), MCMCThreads(), ceil(Integer,n_sample_NUTS/n_threads), 
            n_threads, initial_params = fill(θ0_true .+ 0.001, n_threads))

        () -> begin
            #using JLD2
            jldsave("intermediate/doubleMM_chain_theta.jld2", false, IOStream; chn)
            chn = load("intermediate/doubleMM_chain_theta.jld2", "chn"; iotype = IOStream)
            # plot chn as above
        end

        #θi = first(eachrow(Array(chn)))
        θs = mapreduce(θi -> transposeMs(θi, intm_PMs_gen, true), hcat, eachrow(Array(chn)));
        (; θ, y) = HVI.apply_f_trans(θs, f, xP, Stacked(elementwise(identity)), intm_PMs_gen);


        mean_y_invθ = map(mean, eachslice(y; dims = (1, 2)));
        #describe(mean_y_pred - y_o)
        histogram(vec(mean_y_invθ) - vec(y_true)) # predictions centered around y_o (or y_true)
        plt = scatterplot(vec(y_true), vec(mean_y_invθ));
        lineplot!(plt, 0, 2)
        mean(mean_y_invθ - y_true) # still ok

        # first site, first prediction
        y_inv11 = y[1, 1, :]
        histogram(y_inv11 .- y_o[1, 1])
        histogram(y_inv11 .- y_true[1, 1])  

        histogram(θs[1, :])
        describe(pdf.(Ref(prior_θ), θs[1, :]))  # only small differences 
        pdf(prior_θ, log(θ_true[1]))

        mean_θ = CA.ComponentVector(mean(CA.getdata(θ); dims = 2)[:, 1], CA.getaxes(θ[:, 1])[1])
        histogram(θ[:P, :] .- θP_true)  # all overestimated ? 
        plt = scatterplot(θMs_true_train[1, :], mean_θ.Ms[1, :]);
        lineplot!(plt, 0, 1)
        plt = scatterplot(θMs_true_train[2, :], mean_θ.Ms[2, :]);
        lineplot!(plt, 0, 1)
    end  # depr DEMC constrained scale


    # TODO inspect correlation between physical parameter K and ML-parameter r at first (or ith) site

    mean_ζP_MC = mapslices(mean, CA.getdata(ζP_MCc), dims = 2)[:, 1]
    var_ζP_MC = map(x -> var(x; corrected = false), eachrow(ζP_MCc))

    mean_ζMs_MC = mapslices(mean, CA.getdata(ζMs_MCc), dims = 3)[:, :, 1]
    var_ζMs_MC = mapslices(x -> var(x; corrected = false), CA.getdata(ζMs_MCc), dims = 3)[
        :, :, 1]

    hcat(log.(θP_true), mean_ζP_MC)

    scatterplot(vec(log.(θMs_true_train)), vec(mean_ζMs_MC))
    cor(vec(log.(θMs_true_train)), vec(mean_ζMs_MC))

    scatterplot(mean_ζMs_MC[1, :], log.(var_ζMs_MC[1, :]))
    scatterplot(mean_ζMs_MC[2, :], log.(var_ζMs_MC[2, :]))

    # predictive posterior uncertainty
    # θ = first(eachcol(θs_MC))
    y_pred = stack(map(eachcol(θs_MC)) do θ
        θc = int_θPMs(θ)
        #θP, θMs = @view(θ[1:n_θP]), reshape(@view(θ[n_θP+1:end, :]), n_θM, :)
        θP, θMs = θc.θP, θc.θMs
        y_pred_i = map_f_each_site(f_doubleMM, θMs, θP)
    end)
    #hcat(y_pred[:,1,1], y_pred_gen[:,1,1])

    σ_o_post_MC = mapslices(std, y_pred; dims = 3)
    describe(σ_o_post_MC)
    vcat(σ_o, mean(σ_o_post_MC), sqrt(mean(abs2, σ_o_post_MC)))

    () -> begin
        i_site = 12
        y_pred_site = 1
        i_sample = 1
        plt = scatterplot(y_o[:, i_site], y_pred[:, i_site, i_sample])
        for i_sample in 2:20   #n_sample_NUTS
            plt = scatterplot!(plt, y_o[:, i_site], y_pred[:, i_site, i_sample])
        end
        plt
    end

    # correlation r1 and K1 (original and log scale)
    i_site = 5
    scatterplot(ζMs_MCc[:r1, i_site, :], ζMs_MCc[:K1, i_site, :])
    scatterplot(θMs_MCc[:r1, i_site, :], θMs_MCc[:K1, i_site, :])

    # correlation r1 and K2 (physical) (original and log scale)
    scatterplot(ζMs_MCc[:r1, i_site, :], ζP_MCc[:K2, :])
    scatterplot(θMs_MCc[:r1, i_site, :], θPs_MCc[:K2, :])
    cor(ζMs_MCc[:r1, i_site, :], ζP_MCc[:K2, :]) # only weak

    ζPM1_MCc = vcat(θPs_MCc, θMs_MCc[:, i_site, :])
    cor(CA.getdata(ζPM1_MCc)') # increases with larger σ_o (repeated with changed code)

    save_prefious_MCSampling = () -> begin
        jldsave(joinpath(@__DIR__, "uncNN", "intermediate", "f_doubleMM_MC.jld2"), IOStream;
            ζs_MC)
    end

    # TODO compare distributions to MC sample
end # finteractive_after_sample
