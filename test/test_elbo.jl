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

ggdev = gpu_device()


#CUDA.device!(4)
rng = StableRNG(111)

const prob = DoubleMM.DoubleMMCase()
scenario = Val((:default,))
#scenario = Val((:covarK2,))


test_scenario = (scenario) -> begin
    FT = get_hybridproblem_float_type(prob; scenario)
    par_templates = get_hybridproblem_par_templates(prob; scenario)
    int_P, int_M = map(ComponentArrayInterpreter, par_templates)
    pbm_covars = get_hybridproblem_pbmpar_covars(prob; scenario)
    pbm_covar_indices = CP.get_pbm_covar_indices(par_templates.θP, pbm_covars)


    #θsite_true = get_hybridproblem_par_templates(prob; scenario)
    n_site, n_batch = get_hybridproblem_n_site_and_batch(prob; scenario)
    (; xM, θP_true, θMs_true, xP, y_global_true, y_true, y_global_o, y_o, y_unc
    ) = gen_hybridproblem_synthetic(rng, prob; scenario);

    g, ϕg0 = @inferred get_hybridproblem_MLapplicator(prob; scenario);
    f = get_hybridproblem_PBmodel(prob; scenario, use_all_sites = false)
    f_pred = get_hybridproblem_PBmodel(prob; scenario, use_all_sites = true)

    n_θM, n_θP = values(map(length, par_templates))


    py = neg_logden_indep_normal

    n_MC = 3
    (; transP, transM) = get_hybridproblem_transforms(prob; scenario)
    cor_ends = get_hybridproblem_cor_ends(prob; scenario)
    # transP = elementwise(exp)
    # transM = Stacked(elementwise(identity), elementwise(exp))
    #transM = Stacked(elementwise(identity), elementwise(exp), elementwise(exp)) # test mismatch
    ϕunc0 = init_hybrid_ϕunc(cor_ends, zero(FT))
    hpints = HybridProblemInterpreters(prob; scenario)
    (; ϕ, transPMs_batch, interpreters, get_transPMs, get_ca_int_PMs) = init_hybrid_params(
        θP_true, θMs_true[:, 1], cor_ends, ϕg0, hpints; transP, transM);
    # @descend_code_warntype init_hybrid_params(θP_true, θMs_true[:, 1], cor_ends, ϕg0, n_batch; transP, transM)
    # @descend_code_warntype CA.ComponentVector(nt)
    ϕ_ini = ϕ
    transform_tools = @inferred CP.setup_transform_ζ(transP, transM, get_int_PMst_batch(hpints))
    int_PMs = get_int_PMst_batch(hpints)

    if ggdev isa MLDataDevices.AbstractGPUDevice
        scenario_flux = Val((CP._val_value(scenario)..., :use_Flux, :use_gpu))
        g_flux, ϕg0_flux_cpu = get_hybridproblem_MLapplicator(prob; scenario = scenario_flux)
        g_gpu = ggdev(g_flux)
    end;


    ζs, σ = @inferred (
    # @descend_code_warntype (
        CP.generate_ζ(
        rng, g, ϕ_ini, xM[:, 1:n_batch];
        n_MC, cor_ends, pbm_covar_indices, 
        int_unc = interpreters.unc, int_μP_ϕg_unc=interpreters.μP_ϕg_unc, 
        int_PMs)
    )

    @testset "generate_ζ $(last(CP._val_value(scenario)))" begin
        # xMtest = vcat(xM, xM[1:1,:])
        # ζ, σ = CP.generate_ζ(
        #     rng, g, ϕ_ini, xMtest[:, 1:n_batch], map(get_concrete, interpreters);
        #     n_MC = 8, cor_ends, pbm_covar_indices)
        @test ζs isa AbstractMatrix
        @test size(ζs) == (n_MC, length(int_PMs))
        gr = Zygote.gradient(
            # ϕ -> sum(CP.generate_ζ(
            #     rng, g, ϕ, xMtest[:, 1:n_batch], map(get_concrete, interpreters);
            #     n_MC = 8, cor_ends, pbm_covar_indices)[1]),
            ϕ -> sum(first(CP.generate_ζ(
                rng, g, ϕ, xM[:, 1:n_batch];
                n_MC = 8, cor_ends, pbm_covar_indices,
                int_unc = interpreters.unc, int_μP_ϕg_unc=interpreters.μP_ϕg_unc, int_PMs
                ))),
            CA.getdata(ϕ_ini))
        @test gr[1] isa Vector
    end;

    if ggdev isa MLDataDevices.AbstractGPUDevice
        @testset "generate_ζ gpu $(last(CP._val_value(scenario)))" begin
            ϕ = ggdev(CA.getdata(ϕ_ini))
            @test g_gpu.μ isa GPUArraysCore.AbstractGPUArray
            xMg_batch = ggdev(xM[:, 1:n_batch])
            ζ, σ = @inferred (
                # @descend_code_warntype (
                CP.generate_ζ(
                rng, g_gpu, ϕ, xMg_batch;
                n_MC, cor_ends, pbm_covar_indices,
                int_unc = interpreters.unc, int_μP_ϕg_unc=interpreters.μP_ϕg_unc, int_PMs
                ))
            @test ζ isa Union{GPUArraysCore.AbstractGPUMatrix, 
                LinearAlgebra.Adjoint{FT, <: GPUArraysCore.AbstractGPUMatrix}} 
            @test eltype(ζ) == FT
            @test size(ζs) == (n_MC, length(int_PMs))
            gr = Zygote.gradient(
                ϕ -> sum(first(CP.generate_ζ(
                    rng, g_gpu, ϕ, xMg_batch;
                    n_MC, cor_ends, pbm_covar_indices,
                    int_unc = interpreters.unc, int_μP_ϕg_unc=interpreters.μP_ϕg_unc, int_PMs
                    ))),
                ϕ)
            @test gr[1] isa GPUArraysCore.AbstractGPUVector
        end
    end

    @testset "transform ζs" begin
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
        ζ = ζs[1,:]
        θP, θMs, logjac = CP.transform_ζ(ζ, transP, transM, int_PMs)
        @test size(θP) == (n_θP,)
        @test size(θMs) == (n_batch, n_θM)
        if ggdev isa MLDataDevices.AbstractGPUDevice
            ζdev = ggdev(ζ)
            θP, θMs, logjac = @inferred CP.transform_ζ(ζdev, transP, transM, int_PMs)
            _transform_tools = @inferred CP.setup_transform_ζ(
                transP, transM, int_PMs)
            gr = Zygote.gradient(ζdev) do ζdev
                θP, θMs, logjac = CP.transform_ζ(ζdev, _transform_tools...)
                sum(θP) + sum(θMs) + logjac
            end;
            @test eltype(gr[1]) == eltype(ζ)
        end
    end


    @testset "neg_elbo_gtf cpu $(last(CP._val_value(scenario)))" begin
        i_sites = 1:n_batch
        cost = 
        @inferred (
        #@descend_code_warntype (
            neg_elbo_gtf(rng, ϕ_ini, g, f, py,
            xM[:, i_sites], xP[:,i_sites], y_o[:, i_sites], y_unc[:, i_sites], i_sites,
            map(get_concrete, interpreters), transform_tools;
            cor_ends, pbm_covar_indices)
            )
        
        @test cost isa Float64
        gr = Zygote.gradient(
            ϕ -> neg_elbo_gtf(rng, ϕ, g, f, py,
                xM[:, i_sites], xP[:,i_sites], y_o[:, i_sites], y_unc[:, i_sites], i_sites,
                map(get_concrete, interpreters), transform_tools;
                cor_ends, pbm_covar_indices),
            CA.getdata(ϕ_ini))
        @test gr[1] isa Vector
    end;

    if ggdev isa MLDataDevices.AbstractGPUDevice
        @testset "neg_elbo_gtf gpu $(last(CP._val_value(scenario)))" begin
            i_sites = 1:n_batch
            ϕ = ggdev(CA.getdata(ϕ_ini))
            xMg_batch = ggdev(xM[:, i_sites])
            xP_batch = xP[:,i_sites] # used in f which runs on CPU
            cost = @inferred (
                #@descend_code_warntype (
                 neg_elbo_gtf(rng, ϕ, g_gpu, f, py,
                xMg_batch, xP_batch, y_o[:, i_sites], y_unc[:, i_sites], i_sites,
                map(get_concrete, interpreters), transform_tools;
                n_MC = 3, cor_ends, pbm_covar_indices)
            )
            @test cost isa Float64
            gr = Zygote.gradient(
                ϕ -> neg_elbo_gtf(rng, ϕ, g_gpu, f, py,
                    xMg_batch, xP_batch, y_o[:, i_sites], y_unc[:, i_sites], i_sites,
                    map(get_concrete, interpreters), transform_tools;
                    n_MC = 3, cor_ends, pbm_covar_indices),
                ϕ)
            @test gr[1] isa GPUArraysCore.AbstractGPUVector
            @test eltype(gr[1]) == FT
        end
    end

    n_sample_pred = 200
    int_PMs = get_concrete(CP.construct_int_PMs_parfirst(int_P, int_M, size(xP,2)))
    intm_PMs = get_concrete(ComponentArrayInterpreter(int_PMs, (n_sample_pred,)))
    int_PMst = get_concrete(CP.construct_int_PMs_sitefirst(int_P, int_M, size(xP,2)))
    intm_PMst = get_concrete(ComponentArrayInterpreter(int_PMst, (n_sample_pred,)))
    int_unc = get_concrete(interpreters.unc)

    @testset "predict_gf cpu $(last(CP._val_value(scenario)))" begin
        # intm_PMs_gen = get_ca_int_PMs(n_site)
        # trans_PMs_gen = get_transPMs(n_site)
        # @test length(intm_PMs_gen) == 402
        # @test trans_PMs_gen.length_in == 402
        (; θs, y, intm_PMst) = 
        #Cthulhu.@descend_code_warntype (
        @inferred (
        predict_gf(rng, g, f_pred, ϕ_ini, xM, xP, map(get_concrete, interpreters),
            int_P, int_M;
            #get_transPMs, 
            transP, transM, 
            get_ca_int_PMs, n_sample_pred, cor_ends, pbm_covar_indices,
            int_PMs, int_PMst, intm_PMst, int_unc
            )
        )
        @test θs isa CA.ComponentMatrix
        @test θs[:,1].P.r0 > 0
        @test size(θs,2) == n_sample_pred
        @test y isa Array
        @test size(y) == (size(y_o)..., n_sample_pred)
    end

    if ggdev isa MLDataDevices.AbstractGPUDevice
        @testset "predict_gf gpu $(last(CP._val_value(scenario)))" begin
            n_sample_pred = 200
            ϕ_ini_g = ggdev(CA.getdata(ϕ_ini))
            xMg = ggdev(xM)
            (; θs, y, intm_PMst) = predict_gf(rng, g_gpu, f_pred, ϕ_ini_g, xMg, xP, map(get_concrete, interpreters),
            int_P, int_M;
            #get_transPMs, 
            transP, transM, 
            get_ca_int_PMs, n_sample_pred, cor_ends, pbm_covar_indices,
            int_PMs, int_PMst, intm_PMst, int_unc
            )
            @test θs isa CA.ComponentMatrix # only ML parameters are on gpu
            @test θs[:,1].P.r0 > 0
            @test size(θs,2) == n_sample_pred
            @test y isa Array
            @test size(y) == (size(y_o)..., n_sample_pred)
        end
        # @testset "predict_gf also f on gpu" begin
        #     # currently only works with identity transformations but not elementwise(exp)
        #     transPM_ident = get_hybridproblem_transforms(prob; scenario = (scenario..., :transIdent))
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
        #     (; θ, y) = predict_gf(rng, g_gpu, f_pred, ϕ, xMg, ggdev(xP), map(get_concrete, interpreters);
        #         get_ca_int_PMs, n_sample_pred, cor_ends, pbm_covar_indices,
        #         get_transPMs = get_transPMs_ident, 
        #         cdev = identity); # keep on gpu
        #     @test θ isa CA.ComponentMatrix 
        #     @test CA.getdata(θ) isa GPUArraysCore.AbstractGPUArray
        #     #@test CUDA.@allowscalar θ[:, 1].P.r0 > 0 # did not update ζP
        #     @test y isa GPUArraysCore.AbstractGPUArray
        #     @test size(y) == (size(y_o)..., n_sample_pred)
        # end

    end
end # test_scenario

test_scenario((:default,))

# with providing process parameter as additional covariate
test_scenario((:covarK2,))


