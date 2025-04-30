#using LinearAlgebra, BlockDiagonals

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
scenario = (:default,)
#scenario = (:covarK2,)


test_scenario = (scenario) -> begin
    FT = get_hybridproblem_float_type(prob; scenario)
    par_templates = get_hybridproblem_par_templates(prob; scenario)
    pbm_covars = get_hybridproblem_pbmpar_covars(prob; scenario)
    pbm_covar_indices = CP.get_pbm_covar_indices(par_templates.θP, pbm_covars)


    #θsite_true = get_hybridproblem_par_templates(prob; scenario)
    n_covar = 5
    n_site, n_batch = get_hybridproblem_n_site_and_batch(prob; scenario)
    (; xM, n_site, θP_true, θMs_true, xP, y_global_true, y_true, y_global_o, y_o, y_unc
    ) = gen_hybridproblem_synthetic(rng, prob; scenario);

    g, ϕg0 = get_hybridproblem_MLapplicator(prob; scenario);
    f = get_hybridproblem_PBmodel(prob; scenario, use_all_sites = false)
    f_pred = get_hybridproblem_PBmodel(prob; scenario, use_all_sites = true)

    n_θM, n_θP = values(map(length, get_hybridproblem_par_templates(prob; scenario)))


    py = neg_logden_indep_normal

    n_MC = 3
    (; transP, transM) = get_hybridproblem_transforms(prob; scenario)
    cor_ends = get_hybridproblem_cor_ends(prob; scenario)
    # transP = elementwise(exp)
    # transM = Stacked(elementwise(identity), elementwise(exp))
    #transM = Stacked(elementwise(identity), elementwise(exp), elementwise(exp)) # test mismatch
    ϕunc0 = init_hybrid_ϕunc(cor_ends, zero(FT))
    (; ϕ, transPMs_batch, interpreters, get_transPMs, get_ca_int_PMs) = init_hybrid_params(
        θP_true, θMs_true[:, 1], cor_ends, ϕg0, n_batch; transP, transM);
    ϕ_ini = ϕ

    if ggdev isa MLDataDevices.AbstractGPUDevice
        scenario_flux = (scenario..., :use_Flux, :use_gpu)
        g_flux, ϕg0_flux_cpu = get_hybridproblem_MLapplicator(prob; scenario = scenario_flux)
        g_gpu = ggdev(g_flux)
    end;

    @testset "generate_ζ" begin
        # xMtest = vcat(xM, xM[1:1,:])
        # ζ, σ = CP.generate_ζ(
        #     rng, g, ϕ_ini, xMtest[:, 1:n_batch], map(get_concrete, interpreters);
        #     n_MC = 8, cor_ends, pbm_covar_indices)
        ζ, σ = CP.generate_ζ(
            rng, g, ϕ_ini, xM[:, 1:n_batch], map(get_concrete, interpreters);
            n_MC = 8, cor_ends, pbm_covar_indices)
        @test ζ isa Matrix
        gr = Zygote.gradient(
            # ϕ -> sum(CP.generate_ζ(
            #     rng, g, ϕ, xMtest[:, 1:n_batch], map(get_concrete, interpreters);
            #     n_MC = 8, cor_ends, pbm_covar_indices)[1]),
            ϕ -> sum(CP.generate_ζ(
                rng, g, ϕ, xM[:, 1:n_batch], map(get_concrete, interpreters);
                n_MC = 8, cor_ends, pbm_covar_indices)[1]),
            CA.getdata(ϕ_ini))
        @test gr[1] isa Vector
    end;


    if ggdev isa MLDataDevices.AbstractGPUDevice
        @testset "generate_ζ gpu" begin
            ϕ = ggdev(CA.getdata(ϕ_ini))
            @test g_gpu.μ isa GPUArraysCore.AbstractGPUArray
            xMg_batch = ggdev(xM[:, 1:n_batch])
            ζ, σ = CP.generate_ζ(
                rng, g_gpu, ϕ, xMg_batch, map(get_concrete, interpreters);
                n_MC = 8, cor_ends, pbm_covar_indices)
            @test ζ isa GPUArraysCore.AbstractGPUMatrix
            @test eltype(ζ) == FT
            gr = Zygote.gradient(
                ϕ -> sum(CP.generate_ζ(
                    rng, g_gpu, ϕ, xMg_batch, map(get_concrete, interpreters);
                    n_MC = 8, cor_ends, pbm_covar_indices)[1]),
                ϕ)
            @test gr[1] isa GPUArraysCore.AbstractGPUVector
        end
    end

    @testset "neg_elbo_gtf cpu" begin
        i_sites = 1:n_batch
        cost = neg_elbo_gtf(rng, ϕ_ini, g, transPMs_batch, f, py,
            xM[:, i_sites], xP[i_sites], y_o[:, i_sites], y_unc[:, i_sites], i_sites,
            map(get_concrete, interpreters);
            cor_ends, pbm_covar_indices)
        @test cost isa Float64
        gr = Zygote.gradient(
            ϕ -> neg_elbo_gtf(rng, ϕ, g, transPMs_batch, f, py,
                xM[:, i_sites], xP[i_sites], y_o[:, i_sites], y_unc[:, i_sites], i_sites,
                map(get_concrete, interpreters);
                cor_ends, pbm_covar_indices),
            CA.getdata(ϕ_ini))
        @test gr[1] isa Vector
    end;

    if ggdev isa MLDataDevices.AbstractGPUDevice
        @testset "neg_elbo_gtf gpu" begin
            i_sites = 1:n_batch
            ϕ = ggdev(CA.getdata(ϕ_ini))
            xMg_batch = ggdev(xM[:, i_sites])
            xP_batch = xP[i_sites] # used in f which runs on CPU
            cost = neg_elbo_gtf(rng, ϕ, g_gpu, transPMs_batch, f, py,
                xMg_batch, xP_batch, y_o[:, i_sites], y_unc[:, i_sites], i_sites,
                map(get_concrete, interpreters);
                n_MC = 3, cor_ends, pbm_covar_indices)
            @test cost isa Float64
            gr = Zygote.gradient(
                ϕ -> neg_elbo_gtf(rng, ϕ, g_gpu, transPMs_batch, f, py,
                    xMg_batch, xP_batch, y_o[:, i_sites], y_unc[:, i_sites], i_sites,
                    map(get_concrete, interpreters);
                    n_MC = 3, cor_ends, pbm_covar_indices),
                ϕ)
            @test gr[1] isa GPUArraysCore.AbstractGPUVector
            @test eltype(gr[1]) == FT
        end
    end

    @testset "predict_gf cpu" begin
        n_sample_pred = n_site = 200
        intm_PMs_gen = get_ca_int_PMs(n_site)
        trans_PMs_gen = get_transPMs(n_site)
        @test length(intm_PMs_gen) == 402
        @test trans_PMs_gen.length_in == 402
        (; θ, y) = predict_gf(rng, g, f_pred, ϕ_ini, xM, xP, map(get_concrete, interpreters);
            get_transPMs, get_ca_int_PMs, n_sample_pred, cor_ends, pbm_covar_indices)
        @test θ isa CA.ComponentMatrix
        @test θ[:, 1].P.r0 > 0
        @test y isa Array
        @test size(y) == (size(y_o)..., n_sample_pred)
    end

    if ggdev isa MLDataDevices.AbstractGPUDevice
        @testset "predict_gf gpu" begin
            n_sample_pred = 200
            ϕ = ggdev(CA.getdata(ϕ_ini))
            xMg = ggdev(xM)
            (; θ, y) = predict_gf(rng, g_gpu, f_pred, ϕ, xMg, xP, map(get_concrete, interpreters);
                get_transPMs, get_ca_int_PMs, n_sample_pred, cor_ends, pbm_covar_indices)
            @test θ isa CA.ComponentMatrix # only ML parameters are on gpu
            @test θ[:, 1].P.r0 > 0
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


