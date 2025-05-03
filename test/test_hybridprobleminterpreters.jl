using Test
using HybridVariationalInference
using HybridVariationalInference: HybridVariationalInference as CP
using ComponentArrays: ComponentArrays as CA

using MLDataDevices, GPUArraysCore
import Zygote

# import CUDA, cuDNN
using Suppressor

gdev = Suppressor.@suppress gpu_device() # not loaded CUDA
cdev = cpu_device()

scenario = Val((:default,))
prob = DoubleMM.DoubleMMCase()

ints = @inferred HybridProblemInterpreters(prob; scenario)
θP, θM = @inferred get_hybridproblem_par_templates(prob; scenario)
NS, NB = @inferred get_hybridproblem_n_site_and_batch(prob; scenario)

@testset "HybridProblemInterpreters" begin
    @test (@inferred get_int_P(ints)(CA.getdata(θP))) == θP
    @test (@inferred get_int_M(ints)(CA.getdata(θM))) == θM
    #
    int_Ms_batch = get_concrete(ComponentArrayInterpreter(θM, (NB,)))
    ms_vec = 1:length(int_Ms_batch)
    @test (@inferred get_int_Ms_batch(ints)(ms_vec)) == int_Ms_batch(ms_vec)
    int_Mst_batch = get_concrete(ComponentArrayInterpreter((NB,), θM))
    @test (@inferred get_int_Mst_batch(ints)(ms_vec)) == int_Mst_batch(ms_vec)
    #
    int_Ms_site = get_concrete(ComponentArrayInterpreter(θM, (NS,)))
    ms_vec = 1:length(int_Ms_site)
    @test (@inferred get_int_Ms_site(ints)(ms_vec)) == int_Ms_site(ms_vec)
    int_Mst_site = get_concrete(ComponentArrayInterpreter((NS,), θM))
    @test (@inferred get_int_Mst_site(ints)(ms_vec)) == int_Mst_site(ms_vec)
    #
    pms_ca = CA.ComponentVector(P = θP, Ms = int_Ms_batch(1:length(int_Ms_batch)))
    pms_vec = CA.getdata(pms_ca)
    #int_PMs_batch = get_concrete(ComponentArrayInterpreter(pms_ca))
    @test (@inferred get_int_PMs_batch(ints)(pms_vec)) == pms_ca
    pmst_ca = CA.ComponentVector(P = θP, Ms = int_Mst_batch(1:length(int_Mst_batch)))
    pmst_vec = CA.getdata(pmst_ca)
    @test (@inferred get_int_PMst_batch(ints)(pmst_vec)) == pmst_ca
    #
    pms_ca = CA.ComponentVector(P = θP, Ms = int_Ms_site(1:length(int_Ms_site)))
    pms_vec = CA.getdata(pms_ca)
    @test (@inferred get_int_PMs_site(ints)(pms_vec)) == pms_ca
    pmst_ca = CA.ComponentVector(P = θP, Ms = int_Mst_site(1:length(int_Mst_site)))
    pmst_vec = CA.getdata(pmst_ca)
    @test (@inferred get_int_PMst_site(ints)(pmst_vec)) == pmst_ca
end;

@testset "stack_ca_int" begin
    int_Mst_batch = get_int_Mst_batch(ints)
    pmst_ca = CA.ComponentVector(P = θP, Ms = int_Mst_batch(1:length(int_Mst_batch)))
    n_pred = 5
    mmst_vec = repeat(CA.getdata(pmst_ca)', n_pred) # column per parameter
    int_PMst_batch = @inferred get_int_PMst_batch(ints)
    intm_PMst_batch = @inferred stack_ca_int(Val((n_pred,)), int_PMst_batch)
    mmst = @inferred intm_PMst_batch(mmst_vec)
    @test size(mmst[1, :Ms]) == (NB, length(θM))
    @test all(mmst[:, :P][:, :r0] .== pmst_ca.P.r0)
    #
    # note the use of Val here -> arrays interpreted will by Any outside the context
    @testset "stack_ca_int not inferred outside" begin 
        tmpf = (mmst_vec;
            intm_PMst_batch = @inferred stack_ca_int(Val((size(mmst_vec,1),)), int_PMst_batch)
        ) -> begin
            # good practise to help inference by providing a hint to the eltype
            (@inferred intm_PMst_batch(mmst_vec))::CA.ComponentMatrix{eltype(mmst_vec),typeof(mmst_vec)}
        end
        res = tmpf(mmst_vec)
        @test_broken @inferred tmpf(mmst_vec)
        # but supplying the extended array, its inferred in this context
        intm_PMst_batch2 = @inferred stack_ca_int(Val((size(mmst_vec,1),)), int_PMst_batch)
        @inferred tmpf(mmst_vec; intm_PMst_batch = intm_PMst_batch2)
    end
end

