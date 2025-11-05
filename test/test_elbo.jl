#using LinearAlgebra, BlockDiagonals
using LinearAlgebra

using Test
using HybridVariationalInference
using HybridVariationalInference: HybridVariationalInference as CP
using StableRNGs
using Random
using SimpleChains
using ComponentArrays: ComponentArrays as CA
#using TransformVariables
using Bijectors
using Zygote
import GPUArraysCore: GPUArraysCore
import CUDA, cuDNN
using MLDataDevices

# setup g as FluxNN on gpu
using Flux

#CUDA.device!(4)

ggdev = gpu_device()

rng = StableRNG(111)

const prob = DoubleMM.DoubleMMCase()
scenario = Val((:default,))
#scenario = Val((:covarK2,))

test_scenario = (scenario) -> begin
    probc = HybridProblem(prob; scenario);
    FT = get_hybridproblem_float_type(probc; scenario)
    par_templates = get_hybridproblem_par_templates(probc; scenario)
    int_P, int_M = map(ComponentArrayInterpreter, par_templates)
    pbm_covars = get_hybridproblem_pbmpar_covars(probc; scenario)
    pbm_covar_indices = CP.get_pbm_covar_indices(par_templates.θP, pbm_covars)
    #get_hybridproblem_

    #θsite_true = get_hybridproblem_par_templates(probc; scenario)
    n_site, n_batch = get_hybridproblem_n_site_and_batch(probc; scenario)
    # note: need to use prob rather than probc here, make sure the same
    rng = StableRNG(111)
    (; xM, θP_true, θMs_true, xP,  y_true,  y_o, y_unc) = 
        gen_hybridproblem_synthetic(rng, prob; scenario)
    tmpf = () -> begin
        # wrap inside function to not define(pollute) variables in level up
        _trainloader = get_hybridproblem_train_dataloader(probc; scenario)
        (_xM, _xP, _y_o, _y_unc, _i_sites) = _trainloader.data
        @test _xM == xM
        @test _y_o == y_o
    end; tmpf()

    # prediction by g(ϕg, XM) does not correspond to θMs_true, randomly initialized
    # only the magnitude is there because of NormalScaling and prior
    g, ϕg0 = get_hybridproblem_MLapplicator(probc; scenario)
    f = get_hybridproblem_PBmodel(probc; scenario)
    f_pred = create_nsite_applicator(f, n_site)

    n_θM, n_θP = values(map(length, par_templates))

    py = neg_logden_indep_normal

    priors = get_hybridproblem_priors(prob; scenario)
    priorsP = [priors[k] for k in keys(par_templates.θP)]
    priorsM = [priors[k] for k in keys(par_templates.θM)]

    n_MC = 3
    (; transP, transM) = get_hybridproblem_transforms(probc; scenario)
    cor_ends = get_hybridproblem_cor_ends(probc; scenario)
    # transP = elementwise(exp)
    # transM = Stacked(elementwise(identity), elementwise(exp))
    #transM = Stacked(elementwise(identity), elementwise(exp), elementwise(exp)) # test mismatch
    ϕunc0 = init_hybrid_ϕunc(cor_ends, zero(FT))
    hpints = HybridProblemInterpreters(probc; scenario)
    (; ϕ, transPMs_batch, interpreters, get_transPMs, get_ca_int_PMs) = init_hybrid_params(
        θP_true, θMs_true[:, 1], cor_ends, ϕg0, hpints; transP, transM)
    int_unc = interpreters.unc
    int_μP_ϕg_unc = interpreters.μP_ϕg_unc

    # @descend_code_warntype init_hybrid_params(θP_true, θMs_true[:, 1], cor_ends, ϕg0, n_batch; transP, transM)
    # @descend_code_warntype CA.ComponentVector(nt)
    ϕ_ini = ϕ
    transform_tools = nothing # TODO remove
    # transform_tools = @inferred CP.setup_transform_ζ(
    #     transP, transM, get_int_PMst_batch(hpints))
    int_PMs = get_int_PMst_batch(hpints)

    if ggdev isa MLDataDevices.AbstractGPUDevice
        scenario_flux = Val((CP._val_value(scenario)..., :use_Flux, :use_gpu))
        probc_dev = HybridProblem(prob; scenario = scenario_flux);
        g_flux, ϕg0_flux_cpu = get_hybridproblem_MLapplicator(
            probc_dev; scenario=scenario_flux)
        g_gpu = ggdev(g_flux)
    end

    ζsP, ζsMs, σ = @inferred (
    # @descend_code_warntype (
        CP.generate_ζ(
        rng, g, ϕ_ini, xM[:, 1:n_batch];
        n_MC, cor_ends, pbm_covar_indices,
        int_unc=interpreters.unc, int_μP_ϕg_unc=interpreters.μP_ϕg_unc, is_testmode = false)
    )

    @testset "generate_ζ $(last(CP._val_value(scenario)))" begin
        # xMtest = vcat(xM, xM[1:1,:])
        # ζ, σ = CP.generate_ζ(
        #     rng, g, ϕ_ini, xMtest[:, 1:n_batch], map(get_concrete, interpreters);
        #     n_MC = 8, cor_ends, pbm_covar_indices)
        @test ζsP isa AbstractMatrix
        @test ζsMs isa AbstractArray
        @test size(ζsP) == (n_θP, n_MC)
        @test size(ζsMs) == (n_batch, n_θM, n_MC)
        gr = Zygote.gradient(
            ϕ -> begin
                _ζsP, _ζsMs, _σ = CP.generate_ζ(
                    rng, g, ϕ, xM[:, 1:n_batch];
                    n_MC=8, cor_ends, pbm_covar_indices,
                    int_unc=interpreters.unc, int_μP_ϕg_unc=interpreters.μP_ϕg_unc,
                     is_testmode = true)
                sum(_ζsP) + sum(_ζsMs) + sum(_σ)
            end, CA.getdata(ϕ_ini))
        @test gr[1] isa Vector
    end

    if !(:covarK2 ∈ CP._val_value(scenario)) 
        # can only test distribution if g is not repeated
        @testset "generate_ζ check sd residuals $(last(CP._val_value(scenario)))" begin
            # prescribe very different uncertainties 
            ϕunc_true = copy(probc.ϕunc)
            sd_ζP_true = [0.2,20]
            sd_ζMs_a_true = [0.1,2]  # sd at_variance at θ==0
            logσ2_ζMs_b_true = [-0.3,+0.2]  # slope of log_variance with θ
            ρsP_true = [+0.8]
            ρsM_true = [-0.6]
            
            ϕunc_true.logσ2_ζP = (log ∘ abs2).(sd_ζP_true)
            ϕunc_true.coef_logσ2_ζMs[1,:] = (log ∘ abs2).(sd_ζMs_a_true)
            ϕunc_true.coef_logσ2_ζMs[2,:] = logσ2_ζMs_b_true 
            # note that the parameterization contains a transformation that
            # here only inverted for the single correlation case
            ϕunc_true.ρsP = CP.compute_cholcor_coefficient_single.(ρsP_true)
            ϕunc_true.ρsM = CP.compute_cholcor_coefficient_single.(ρsM_true)
            # check that ρsM_true = -0.6 recovered with params ϕunc_true.ρsM = -0.75
            UC = CP.transformU_cholesky1(ϕunc_true.ρsM); Σ = UC' * UC
            @test Σ[1,2] ≈ ρsM_true[1]

            probd = HybridProblem(probc;  ϕunc=ϕunc_true);
        
            _ϕ = vcat(ϕ_ini.μP, probc.ϕg, probd.ϕunc)
            #hcat(ϕ_ini, ϕ, _ϕ)[1:4,:]
            #hcat(ϕ_ini, ϕ, _ϕ)[(end-20):end,:]
            n_predict = 10_000 #8_000
            xM_batch = xM[:, 1:n_batch]
            _ζsP, _ζsMs, _σ = @inferred (
                # @descend_code_warntype (
                    CP.generate_ζ(
                    rng, g, _ϕ, xM_batch;
                    n_MC = n_predict, cor_ends, pbm_covar_indices,
                    int_unc=interpreters.unc, int_μP_ϕg_unc=interpreters.μP_ϕg_unc,
                    is_testmode = true)
                )
            ζMs_g = g(xM_batch, probc.ϕg)' # have been generated with no scaling
            function test_distζ(_ζsP, _ζsMs, ϕunc_true, ζMs_g)
                mP = mean(_ζsP; dims=2)
                residP = _ζsP .- mP
                sdP = vec(std(residP; dims=2))
                _sd_ζP_true = sqrt.(exp.(ϕunc_true.logσ2_ζP))
                @test isapprox(sdP, _sd_ζP_true; rtol=0.05)
                mMs = mean(_ζsMs; dims=3)[:,:,1]
                hcat(mMs, ζMs_g)
                # @usingany UnicodePlots
                #scatterplot(ζMs_g[:,1], mMs[:,1])
                #scatterplot(ζMs_g[:,2], mMs[:,2])
                @test cor(ζMs_g[:,1], mMs[:,1]) > 0.9
                @test cor(ζMs_g[:,2], mMs[:,2]) > 0.7
                map(axes(mMs,2)) do ipar
                    #@show ipar
                    @test isapprox(mMs[:,ipar], ζMs_g[:,ipar]; rtol=0.1)
                end
                #ζMs_true = stack(map(inverse(transM), eachcol(CA.getdata(θMs_true[:,1:n_batch]))))'
                residMs = _ζsMs .- mMs
                sdMs = std(residMs; dims=3)[:,:,1]
                # (_a,_b), mMi = first(zip(
                #     eachcol(ϕunc_true.coef_logσ2_ζMs), eachcol(mMs)))
                _sd_ζMs_true = stack(map(
                        eachcol(ϕunc_true.coef_logσ2_ζMs), eachcol(ζMs_g)) do (_a,_b), mMi
                        #eachcol(ϕunc_true.coef_logσ2_ζMs), eachcol(mMs)) do (_a,_b), mMi
                    logσ2_ζM = _a .+ mMi .* _b
                    sqrt.(exp.(logσ2_ζM))
                end)
                #ipar = 2
                #ipar = 1
                map(axes(sdMs,2)) do ipar
                    #@show ipar
                    hcat(sdMs[:,ipar], _sd_ζMs_true[:,ipar])
                    @test isapprox(sdMs[:,ipar], _sd_ζMs_true[:,ipar]; rtol=0.2)
                    # scatterplot(sdMs[:,ipar], _sd_ζMs_true[:,ipar])
                end
                i_sites_inspect = [1,2,3]
                # reshape to par-first so that can inspect correlations better
                residMst = permutedims(residMs[i_sites_inspect,:,:], (2,1,3))
                residPMst = vcat(residP, 
                    reshape(residMst, size(residMst,1)*size(residMst,2), size(residMst,3)))
                cor_PMs = cor(residPMst')
                @test cor_PMs[1,2] ≈ ρsP_true[1] atol=0.02
                @test all(.≈(cor_PMs[1:2,3:end], 0.0, atol=0.1)) # no correlations P,M
                @test cor_PMs[3,4] ≈ ρsM_true[1] atol=0.02
                @test all(.≈(cor_PMs[3:4,5:end], 0.0, atol=0.1)) # no correlations M1, M2
                @test cor_PMs[5,6] ≈ ρsM_true[1] atol=0.02
                @test all(.≈(cor_PMs[5:6,7:end], 0.0, atol=0.1)) # no correlations M1, M2
            end
            test_distζ(_ζsP, _ζsMs, ϕunc_true, ζMs_g)
            @testset "predict_hvi check sd" begin
                # test if uncertainty and reshaping is propagated
                # here inverse the predicted θs and then test distribution 
                probcu = HybridProblem(probc, ϕunc=ϕunc_true);
                n_sample_pred = 10_000 #2_400
                #n_sample_pred = 400
                (; y, θsP, θsMs, entropy_ζ) = predict_hvi(rng, probcu; scenario, n_sample_pred);
                #size(_ζsMs), size(θsMs)
                #size(_ζsP), size(θsP)
                trans_minvP = StackedArray(inverse(transP), n_sample_pred)
                _ζsP2 = trans_minvP(θsP)
                int_minvM = StackedArray(inverse(transM), n_site)
                _ζsMs2 = stack(map(eachslice(θsMs; dims=3)) do _θMs
                    int_minvM(_θMs)
                end)
                ζMs_g2 = g(xM, probcu.ϕg)' # have been generated with no scaling
                test_distζ(_ζsP2, _ζsMs2, ϕunc_true, ζMs_g2)
            end;
        end;
    end # if covarK2 in scenario

    if ggdev isa MLDataDevices.AbstractGPUDevice
        @testset "generate_ζ gpu $(last(CP._val_value(scenario)))" begin
            ϕ = ggdev(CA.getdata(ϕ_ini))
            @test g_gpu.μ isa GPUArraysCore.AbstractGPUArray
            # @test g_gpu.app isa HybridVariationalInferenceFluxExt.FluxApplicator
            xMg_batch = ggdev(xM[:, 1:n_batch])
            ζsP_d, ζsMs_d, σ_d = @inferred (
            # @descend_code_warntype (
                CP.generate_ζ(
                rng, g_gpu, ϕ, xMg_batch;
                n_MC, cor_ends, pbm_covar_indices,
                int_unc=interpreters.unc, int_μP_ϕg_unc=interpreters.μP_ϕg_unc,
                is_testmode = true))
            @test ζsP_d isa Union{GPUArraysCore.AbstractGPUMatrix,
                LinearAlgebra.Adjoint{FT,<:GPUArraysCore.AbstractGPUMatrix}}
            @test ζsMs_d isa Union{GPUArraysCore.AbstractGPUArray,
                LinearAlgebra.Adjoint{FT,<:GPUArraysCore.AbstractGPUArray}}
            @test eltype(ζsP_d) == eltype(ζsMs_d) == FT
            @test size(ζsP_d) == (n_θP, n_MC)
            @test size(ζsMs_d) == (n_batch, n_θM, n_MC)
            gr = Zygote.gradient(
                ϕ -> begin
                    _ζsP, _ζsMs, _σ = CP.generate_ζ(
                        rng, g_gpu, ϕ, xMg_batch;
                        n_MC, cor_ends, pbm_covar_indices,
                        int_unc=interpreters.unc, int_μP_ϕg_unc=interpreters.μP_ϕg_unc,
                        is_testmode = false)
                    sum(_ζsP) + sum(_ζsMs) + sum(_σ)
                end, CA.getdata(ϕ))
            @test gr[1] isa GPUArraysCore.AbstractGPUVector
        end
    end

    @testset "transform_and_logjac_ζ $(last(CP._val_value(scenario)))" begin
        # reorder Ms columns so that first parameter of all sites is first
        # # transforming entire parameter set across n_MC most efficient but
        # # does not yield logdetjac
        # intm_PMs_gen = get_ca_int_PMs(n_batch);
        # pos_intm_PMs = get_positions(intm_PMs_gen)
        # function trans_ζs_crossMC(ζs::AbstractMatrix, pos_intm_PMs::NamedTuple; n_MC = size(ζs,2))
        #     ζstMs = ζs'[1:n_MC, pos_intm_PMs.Ms'] # n_MC x n_site_batch x n_par
        #     ζstP = ζs'[1:n_MC, pos_intm_PMs.P] # n_MC x n_par
        #     transPM = extend_stacked_nrow(transP, n_MC)
        #     θsP = reshape(transPM(vec(ζstP)), size(ζstP))
        #     transMM = extend_stacked_nrow(transM, n_MC * n_batch)
        #     θsMs = reshape(transMM(vec(ζstMs)), size(ζstMs))
        #     (θsP, θsMs)            
        # end
        # (θsP, θsMs) = trans_ζs(ζs, pos_intm_PMs; n_MC)
        # @test size(θsP) == (n_MC, n_θP)
        # @test size(θsMs) == (n_MC, n_batch, n_θM)
        # map by rows
        ζP, ζMs = ζsP[:, 1], ζsMs[:, :, 1]
        n_site_batch = size(ζMs, 1)
        transMs = StackedArray(transM, n_site_batch)
        θP, θMs, logjac = @inferred CP.transform_and_logjac_ζ(ζP, ζMs; transP, transMs)
        @test size(θP) == (n_θP,)
        @test size(θMs) == (n_site_batch, n_θM)
        @test θP == transP(ζP)
        @test θMs[1, :] == transM(ζMs[1, :])
        @test θMs[end, :] == transM(ζMs[end, :])
        if ggdev isa MLDataDevices.AbstractGPUDevice
            ζPdev, ζMsdev = ggdev.((ζP, ζMs))
            θP, θMs, logjac = @inferred CP.transform_and_logjac_ζ(
                ζPdev, ζMsdev; transP, transMs)
            @test size(θP) == (n_θP,)
            @test size(θMs) == (n_site_batch, n_θM)
            gr = Zygote.gradient(ζPdev, ζMsdev) do ζPdev, ζMsdev
                θP, θMs, logjac = CP.transform_and_logjac_ζ(ζPdev, ζMsdev; transP, transMs)
                sum(θP) + sum(θMs) + logjac
            end
            @test eltype(gr[1]) == eltype(ζPdev)
            @test eltype(gr[2]) == eltype(ζMsdev)
        end
    end

    @testset "transform_ζs $(last(CP._val_value(scenario)))" begin
        n_site_batch, _, n_MC = size(ζsMs)
        trans_mP = StackedArray(transP, n_MC)
        #trans_mP = StackedArray(Stacked((identity,),(1:n_θP,)), n_MC)
        trans_mMs = StackedArray(transM, n_MC * n_site_batch)
        θsP, θsMs = @inferred CP.transform_ζs(ζsP, ζsMs; trans_mP, trans_mMs)
        #@descend_code_warntype CP.transform_ζs(ζsP, ζsMs; trans_mP, trans_mMs)
        @test size(θsP) == (n_θP, n_MC)
        @test size(θsMs) == (n_site_batch, n_θM, n_MC)
        @test θsP[:, 1] == transP(ζsP[:, 1])
        @test θsP[:, end] == transP(ζsP[:, end])
        @test θsMs[1, :, 1] == transM(ζsMs[1, :, 1]) # first parameter
        @test θsMs[end, :, 1] == transM(ζsMs[end, :, 1])
        @test θsMs[1, :, end] == transM(ζsMs[1, :, end]) # last parameter
        @test θsMs[end, :, end] == transM(ζsMs[end, :, end])
        if ggdev isa MLDataDevices.AbstractGPUDevice
            ζsPdev, ζsMsdev = ggdev.((ζsP, ζsMs))
            #trans_mP(ζsPdev)
            θsP, θsMs = @inferred CP.transform_ζs(ζsPdev, ζsMsdev; trans_mP, trans_mMs)
            gr = Zygote.gradient(ζsPdev, ζsMsdev) do ζsPdev, ζsMsdev
                θsP, θsMs = CP.transform_ζs(ζsPdev, ζsMsdev; trans_mP, trans_mMs)
                sum(θsP) + sum(θsMs)
            end
            @test eltype(gr[1]) == eltype(ζsPdev)
            @test eltype(gr[2]) == eltype(ζsMsdev)
        end
    end

    @testset "neg_elbo_gtf cpu $(last(CP._val_value(scenario)))" begin
        i_sites = 1:n_batch
        transMs = StackedArray(transM, size(ζsMs, 1))
        cost = @inferred (
        #@descend_code_warntype (
            neg_elbo_gtf(rng, ϕ_ini, g, f, py,
            xM[:, i_sites], xP[:, i_sites], y_o[:, i_sites], y_unc[:, i_sites], i_sites;
            int_unc, int_μP_ϕg_unc,
            cor_ends, pbm_covar_indices, transP, transMs, priorsP, priorsM,
            is_testmode = true)
        )
        @test cost isa Float64
        gr = Zygote.gradient(
            ϕ -> neg_elbo_gtf(rng, ϕ, g, f, py,
                xM[:, i_sites], xP[:, i_sites], y_o[:, i_sites], y_unc[:, i_sites], i_sites;
                int_unc, int_μP_ϕg_unc,
                cor_ends, pbm_covar_indices, transP, transMs, priorsP, priorsM,
                is_testmode = false),
            CA.getdata(ϕ_ini))
        @test gr[1] isa Vector
    end

    if ggdev isa MLDataDevices.AbstractGPUDevice
        @testset "neg_elbo_gtf gpu $(last(CP._val_value(scenario)))" begin
            i_sites = 1:n_batch
            transMs = StackedArray(transM, size(ζsMs, 1))
            ϕ = ggdev(CA.getdata(ϕ_ini))
            xMg_batch = ggdev(xM[:, i_sites])
            xP_batch = xP[:, i_sites] # used in f which runs on CPU
            cost = @inferred (
            #@descend_code_warntype (
                neg_elbo_gtf(rng, ϕ, g_gpu, f, py,
                xMg_batch, xP_batch, y_o[:, i_sites], y_unc[:, i_sites], i_sites;
                int_unc, int_μP_ϕg_unc,
                n_MC=3, cor_ends, pbm_covar_indices, transP, transMs, priorsP, priorsM,
                is_testmode = true,
                )
            )
            @test cost isa Float64
            gr = Zygote.gradient(
                ϕ -> neg_elbo_gtf(rng, ϕ, g_gpu, f, py,
                    xMg_batch, xP_batch, y_o[:, i_sites], y_unc[:, i_sites], i_sites;
                    int_unc, int_μP_ϕg_unc,
                    n_MC=3, cor_ends, pbm_covar_indices, transP, transMs, priorsP, priorsM,
                    is_testmode = false,
                    ),
                ϕ)
            @test gr[1] isa GPUArraysCore.AbstractGPUVector
            @test eltype(gr[1]) == FT
        end
    end

    @testset "sample_posterior apply_process_model cpu $(last(CP._val_value(scenario)))" begin
        # intm_PMs_gen = get_ca_int_PMs(n_site)
        # trans_PMs_gen = get_transPMs(n_site)
        # @test length(intm_PMs_gen) == 402
        # @test trans_PMs_gen.length_in == 402
        n_sample_pred = 30
        (; θsP, θsMs, entropy_ζ) =
        #Cthulhu.@descend_code_warntype (
            @inferred (
                sample_posterior(rng, g, ϕ_ini, xM;
                int_μP_ϕg_unc, int_unc,
                transP, transM,
                cdev = identity,
                n_sample_pred, cor_ends, pbm_covar_indices,
                is_testmode = true,
                )
            )
        @test θsP isa AbstractMatrix
        @test θsMs isa AbstractArray{T,3} where {T}
        int_mP = ComponentArrayInterpreter(int_P, (size(θsP, 2),))
        θsPc = int_mP(θsP)
        @test all(θsPc[:r0, :] .> 0)
        #
        y = @inferred f_pred(θsP, θsMs, xP)
        @test y isa Array
        @test size(y) == (size(y_o)..., n_sample_pred)
    end

    if ggdev isa MLDataDevices.AbstractGPUDevice
        @testset "predict_hvi gpu $(last(CP._val_value(scenario)))" begin
            ϕ_ini_g = ggdev(CA.getdata(ϕ_ini))
            xMg = ggdev(xM)
            n_sample_pred = 30
            (; θsP, θsMs, entropy_ζ) =
            #Cthulhu.@descend_code_warntype (
                @inferred (
                    sample_posterior(rng, g_gpu, ϕ_ini_g, xMg;
                    int_μP_ϕg_unc, int_unc,
                    transP, transM,
                    #cdev = cpu_device(),
                    cdev = identity, # do not transfer to CPU
                    n_sample_pred, cor_ends, pbm_covar_indices,
                    is_testmode = true)
                )
            # this variant without the problem, does not attach axes
            @test θsP isa AbstractMatrix
            @test θsMs isa AbstractArray{T,3} where {T}
            int_mP = ComponentArrayInterpreter(int_P, (size(θsP, 2),))
            @test all(int_mP(θsP)[:r0, :] .> 0)
            #
            xP_dev = ggdev(xP);
            f_pred_dev = ggdev(f_pred) #fmap(ggdev, f_pred)
            y = @inferred f_pred_dev(θsP, θsMs, xP_dev)
            #@benchmark f_pred_dev(θsP, θsMs, xP_dev)
            @test y isa GPUArraysCore.AbstractGPUArray
            @test size(y) == (size(y_o)..., n_sample_pred)
        end
        # @testset "predict_hvi also f on gpu" begin
        #     # currently only works with identity transformations but not elementwise(exp)
        #     transPM_ident = get_hybridproblem_transforms(probc; scenario = (scenario..., :transIdent))
        #     get_transPMs_ident = (() -> begin
        #         # wrap in function to not override get_transPMs
        #         (; get_transPMs) = init_hybrid_params(
        #             θP_true, θMs_true[:, 1], cor_ends, ϕg0, n_batch; 
        #             transP = transPM_ident.transP, transM = transPM_ident.transM);
        #         get_transPMs
        #     end)()
        #     n_sample_pred = 200
        #     ϕ = ggdev(CA.getdata(ϕ_ini))
        #     xMg = ggdev(xM)
        #     (; θ, y) = predict_hvi(rng, g_gpu, f_pred, ϕ, xMg, ggdev(xP), map(get_concrete, interpreters);
        #         get_ca_int_PMs, n_sample_pred, cor_ends, pbm_covar_indices,
        #         get_transPMs = get_transPMs_ident, 
        #         cdev = identity); # keep on gpu
        #     @test θ isa CA.ComponentMatrix 
        #     @test CA.getdata(θ) isa GPUArraysCore.AbstractGPUArray
        #     #@test CUDA.@allowscalar θ[:, 1].P.r0 > 0 # did not update ζP
        #     @test y isa GPUArraysCore.AbstractGPUArray
        #     @test size(y) == (size(y_o)..., n_sample_pred)
        # end
    end # if ggdev

end # test_scenario


test_scenario(Val((:default,)))

# with providing process parameter as additional covariate
test_scenario(Val((:covarK2,)))
