using LinearAlgebra, Test
using HybridVariationalInference
using HybridVariationalInference: HybridVariationalInference as CP
using Zygote
using OptimizationOptimisers
using ComponentArrays: ComponentArrays as CA
#using SymmetricFormats
using GPUArraysCore: GPUArraysCore
#using Flux
using CUDA

A = [1.0 2.0 3.0
     2.0 1.0 4.0
     3.0 4.0 1.0] + I(3) .* 10
LA = cholesky(A).L

B = rand(3, 3) ./ 10

C = [1.0 2.0 3.2
     2.0 1.0 4.2
     3.2 4.2 1.0] + I(3) .* 10
LC = cholesky(C).L

Z = zeros(3, 3)

@testset "cholesky of blockdiagonal" begin
    # cholesky factorization of a BlockDiagonal equals BlockDiagonal of Choleskies
    X = [A Z; Z' C]
    LX = cholesky(X).L
    @test LX == [LA Z; Z LC]
end;

@testset "cholesky for of edge structure" begin
    # structure of cholesky factorization of a BlockDiagonal with edges does not carry over
    X = [A B B; B' C Z; B' Z C]
    LX = cholesky(X).L
    LX[4:6, 4:6] ≈ LX[7:9, 7:9]
    X = [A B B B;
         B' C Z Z;
         B' Z C Z;
         B' Z Z C]
    LX = cholesky(X).L
    LX[4:6, 4:6] ≈ LX[7:9, 7:9]
    # but non-zero off-diagonals at Z positions and different edge entries 
    @test_broken LX[7:9, 4:6] ≈ Z
    @test_broken LX[9:12, 4:6] ≈ LX[9:12, 7:9]
end;

@testset "invsumn" begin
    ns_orig = [1, 2, 3, 6]
    s = map(n -> sum(1:n), ns_orig)
    ns = CP.invsumn.(s)
    @test ns == ns_orig
    @test eltype(ns) == Int
    #
    @test_throws Exception invsumn(5) # 5 is not result of sum(1:n)
end;

@testset "vec2utri" begin
    v_orig = 1.0:6.0
    Uv = CP.vec2utri(v_orig)
    @test Uv isa UpperTriangular
    Zygote.gradient(v -> sum(CP.vec2utri(v)), v_orig)[1] # works nice
    #
    v2 = CP.utri2vec(Uv)
    @test v2 == v_orig
    Zygote.gradient(Uv -> sum(CP.utri2vec(Uv)), Uv)[1] # works nice
end;

@testset "vec2uutri" begin
    v = v_orig = 1.0:6.0
    vcpu = collect(v_orig)
    n = CP.invsumn(length(v)) + 1
    T = eltype(v)
    U1v = CP.vec2uutri(v_orig)
    @test U1v isa UnitUpperTriangular
    @test size(U1v, 1) == 4
    gr = Zygote.gradient(v -> sum(abs2.(CP.vec2uutri(v))) , vcpu)[1] # works nice
    # test providing keyword argument
    gr = Zygote.gradient(v -> sum(abs2.(CP.vec2uutri(v; n=4))) , vcpu)[1] # works nice
    #
    v2 = CP.uutri2vec(U1v)
    @test v2 == v_orig
    gr = Zygote.gradient(U1v -> sum(CP.uutri2vec(U1v) .* (1.0:6.0)), U1v)[1] # works nice
end;

@testset "utri2vec_pos" begin
    @test CP.utri2vec_pos(1,1) == 1
    @test CP.utri2vec_pos(1,2) == 2
    @test CP.utri2vec_pos(2,2) == 3
    @test CP.utri2vec_pos(1,3) == 4
    @test CP.utri2vec_pos(1,4) == 7
    @test CP.utri2vec_pos(5,5) == 15
    typeof(CP.utri2vec_pos(5,5)) == Int
    typeof(CP.utri2vec_pos(Int32(5),Int32(5))) == Int32
    @test_throws AssertionError CP.utri2vec_pos(2,1)
end

@testset "vec2uutri gpu" begin
    if CUDA.functional() # only run the test, if CUDA is working (not on Github ci)
        v_orig = 1.0f0:6.0f0
        v = CuArray(collect(v_orig))
        U1v = CP.vec2uutri(v)
        @test !(U1v isa UnitUpperTriangular) # on CUDA work with normal matrix
        @test U1v isa GPUArraysCore.AbstractGPUMatrix
        @test size(U1v, 1) == 4
        @test Array(U1v) == CP.vec2uutri(v_orig)
        gr = Zygote.gradient(v -> sum(abs2.(CP.vec2uutri(v))), v)[1] # works nice
        @test gr isa GPUArraysCore.AbstractGPUArray
        @test Array(gr) == (1:6) .* 2.0
        #
        v2 = CP.uutri2vec(U1v)
        @test v2 isa GPUArraysCore.AbstractGPUVector
        @test eltype(v2) == eltype(U1v)
        @test Array(v2) == v_orig
        gr = Zygote.gradient(U1v -> sum(CP.uutri2vec(U1v .* 2)), U1v)[1] # works nice
        @test gr isa GPUArraysCore.AbstractGPUArray
        @test all(diag(gr) .== 0)
        @test Array(CP.uutri2vec(gr)) == fill(2.0f0, length(v_orig))
    end
end;

@testset "transformU_cholesky1 gpu" begin
    v_orig = 1.0f0:6.0f0
    vcpu = collect(v_orig)
    ch = CP.transformU_cholesky1(vcpu)
    gr1 = Zygote.gradient(v -> sum(CP.transformU_cholesky1(v)), vcpu)[1] # works nice
    @test all(diag(ch' * ch) .≈ 1)
    if CUDA.functional() # only run the test, if CUDA is working (not on Github ci)
        v = CuArray(collect(v_orig))
        U1v = CP.transformU_cholesky1(v)
        @test !(U1v isa UnitUpperTriangular) # on CUDA work with normal matrix
        @test U1v isa GPUArraysCore.AbstractGPUMatrix
        @test Array(U1v) ≈ ch
        gr2 = Zygote.gradient(v -> sum(CP.transformU_cholesky1(v)), v)[1] # works nice
        @test Array(gr2) == gr1
    end
end;

() -> begin
    #setup for fitting of interactive blocks below
    _X = rand(3, 3)
    S = _X * _X' # know that this is Hermitian
    stdS = sqrt.(diag(S))
    C = Diagonal(1 ./ stdS) * S * Diagonal(1 ./ stdS)
    @test Diagonal(stdS) * C * Diagonal(stdS) ≈ S

    SL = cholesky(S).L
    SU = cholesky(S).U
    # CU = cholesky(C).U  # fails: its not recognized as Symmetric
    # CU = cholesky(UnitLowerTriangular(C)).U
    CU = cholesky(Symmetric(C)).U
    CL = cholesky(Symmetric(C)).L
    @test CU * Diagonal(stdS) ≈ SU
    @test Diagonal(stdS) * CL ≈ SL
end

() -> begin
    # find entries of U so that Cpred = U'*U = C
    n_x = 20
    xs = rand(3, n_x)
    ys = xs' * CU
    n_U = size(C, 1)
    fcost = (Uvec) -> begin
        U = vec2utri(Uvec; n = n_U)
        y_pred = xs' * U
        sum(abs2, ys .- y_pred)
    end
    Uvec_true = utri2vec(CU)
    Uvec0 = Uvec_true .* 1.2
    fcost(Uvec0)

    optf = Optimization.OptimizationFunction((x, p) -> fcost(x), Optimization.AutoZygote())
    optprob = Optimization.OptimizationProblem(optf, Uvec0)
    res = Optimization.solve(optprob, OptimizationOptimisers.Adam(0.02),
        callback = callback_loss(50), maxiters = 600)

    #hcat(Uvec_true, res.u, Uvec0)
    Upred = vec2utri(res.u)
    @test Upred ≈ CU
    Cpred = Upred' * Upred
    #hcat(C, Cpred)
    @test Cpred ≈ C
end

() -> begin
    # find Cholesky factor so that U' * U = C given that diag(C) == 1
    n_x = 20
    xs = rand(3, n_x)
    ys = ysC = xs' * CU
    # Us1vec = 1.0:6.0
    n_U = size(S, 1)

    fcost = fcostCT = (Us1vec) -> begin
        U = transformU_cholesky1(Us1vec; n = n_U)
        y_pred = xs' * U
        sum(abs2, ys .- y_pred)
    end
    # cannot infer true U_scaled any more
    Unscaled0 = CU ./ diag(CU)
    Us1vec0 = uutri2vec(Unscaled0)
    fcost(Us1vec0)

    optf = Optimization.OptimizationFunction((x, p) -> fcost(x), Optimization.AutoZygote())
    optprob = Optimization.OptimizationProblem(optf, Us1vec0)
    res = resCT = Optimization.solve(optprob, OptimizationOptimisers.Adam(0.02),
        callback = callback_loss(50), maxiters = 1_600)

    Upred = transformU_cholesky1(res.u; n = n_U)
    #hcat(CU', Upred)
    @test Upred ≈ CU
    fcostCT(resCT.u)
end

@testset "fit cholesky correlation" begin
    # find transformed Cholesky factor so  that recovers covariance S given its diagonal
    _X = rand(3, 3)
    S = _X * _X' # know that this is Hermitian
    n_x = 200
    xs = rand(3, n_x)
    SU = cholesky(S).U
    σ_o = 0.05
    ys_true = ysS = xs' * SU
    ys = ys_true .+ randn(n_x) .* σ_o

    Dσ = Diagonal(sqrt.(diag(S))) # assume given
    n_U = size(S, 1)

    fcost = fcostS = (Us1vec) -> begin
        U = CP.transformU_cholesky1(Us1vec; n = n_U)
        y_pred = (xs' * U) * Dσ
        sum(abs2, ys .- y_pred)
    end
    # cannot infer true U_scaled any more
    Unscaled0 = S ./ diag(S)
    Us1vec0 = CP.uutri2vec(Unscaled0)
    fcost(Us1vec0)
    #fcostS(resCT.u)  # cost of u optimized by Covar should yield small result if same x

    optf = Optimization.OptimizationFunction((x, p) -> fcost(x), Optimization.AutoZygote())
    optprob = Optimization.OptimizationProblem(optf, Us1vec0)
    res = Optimization.solve(optprob, OptimizationOptimisers.Adam(0.02),
        #callback=callback_loss(50), 
        maxiters = 800)

    Upred = CP.transformU_cholesky1(res.u; n = n_U)
    #@test Upred ≈ CU
    SUpred = Upred * Dσ
    #hcat(SUpred, SU)  
    @test SUpred≈SU atol=2e-1
    S_pred = Dσ' * Upred' * Upred * Dσ
    @test S_pred≈S atol=2e-1
end

