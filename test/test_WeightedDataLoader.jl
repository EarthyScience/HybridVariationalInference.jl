using Test
using HybridVariationalInference
using HybridVariationalInference: HybridVariationalInference as HVI
using HybridVariationalInference: WeightedObsView
using SimpleChains
#using Optimisers
#using StatsBase
using MLUtils
using Optimization
using OptimizationOptimisers   # provides Optimization-compatible Adam
import Statistics


    # ----------------------------------------------------------
    # Helper: uniform weights
    # ----------------------------------------------------------
    uniform_weights(n) = ones(Float64, n) ./ n

    # ==========================================================
    # 1. Returns a native MLUtils.DataLoader
    # ==========================================================
    @testset "native MLUtils.DataLoader" begin
        X = rand(Float32, 4, 100)
        w = uniform_weights(100)
        loader = WeightedDataLoader(X, w; batchsize=16)
        @test loader isa MLUtils.DataLoader
    end

    # ==========================================================
    # 2. Correct batch shapes and types
    # ==========================================================
    N = 200
    BS = 32

    @testset "Float32 matrix" begin
        X = rand(Float32, 8, N)
        w = uniform_weights(N)
        loader = WeightedDataLoader(X, w; batchsize=BS, shuffle=false)
        batch = first(loader)
        @test batch isa Array{Float32}
        @test size(batch, 1) == 8
        @test size(batch, 2) == BS
    end

    @testset "Float64 matrix" begin
        X = rand(Float64, 5, N)
        w = uniform_weights(N)
        loader = WeightedDataLoader(X, w; batchsize=BS, shuffle=false)
        batch = first(loader)
        @test batch isa Array{Float64}
        @test size(batch) == (5, BS)
    end

    @testset "Int array" begin
        X = rand(1:10, 3, N)
        w = uniform_weights(N)
        loader = WeightedDataLoader(X, w; batchsize=BS, shuffle=false)
        batch = first(loader)
        @test eltype(batch) <: Integer
        @test size(batch) == (3, BS)
    end

    @testset "NTuple dataset" begin
        X = rand(Float32, 4, N)
        Y = rand(Float32, 2, N)
        data = (X, Y)
        w = uniform_weights(N)
        loader = WeightedDataLoader(data, w; batchsize=BS, shuffle=false)
        batch = first(loader)
        @test batch isa Tuple
        @test length(batch) == 2
        @test batch[1] isa Array{Float32}
        @test size(batch[1]) == (4, BS)
        @test size(batch[2]) == (2, BS)
    end

    @testset "NamedTuple dataset" begin
        X = rand(Float32, 4, N)
        Y = rand(Float32, 1, N)
        data = (x=X, y=Y)
        w = uniform_weights(N)
        loader = WeightedDataLoader(data, w; batchsize=BS, shuffle=false)
        batch = first(loader)
        @test batch isa NamedTuple
        @test haskey(batch, :x) && haskey(batch, :y)
        @test size(batch.x) == (4, BS)
        @test size(batch.y) == (1, BS)
    end

    # ==========================================================
    # 3. Minority class oversampling
    # ==========================================================
    @testset "minority class oversampling" begin
        # Build an imbalanced binary dataset:
        # class 0: 190 samples, class 1: 10 samples
        n_maj = 190
        n_min = 10
        n_total = n_maj + n_min

        labels = vcat(zeros(Int, n_maj), ones(Int, n_min))

        # Inverse-frequency weights so minority is over-sampled
        weights = vcat(
            fill(1.0 / n_maj, n_maj),
            fill(1.0 / n_min, n_min)
        )
        weights ./= sum(weights)   # normalise

        loader = WeightedDataLoader(labels, weights; batchsize=1000, shuffle=false)

        # Collect several batches and measure minority fraction
        minority_fracs = Float64[]
        for (i, batch) in enumerate(loader)
            i > 5 && break
            push!(minority_fracs, Statistics.mean(batch .== 1))
        end

        avg_frac = Statistics.mean(minority_fracs)
        # With balanced weights the minority should appear ~50% of the time
        # (within reasonable Monte-Carlo tolerance)
        @test avg_frac > 0.30   # much more than the original 5%
        @test avg_frac < 0.70
    end

    # ==========================================================
    # 4. End-to-end training with Optimization.solve
    # ==========================================================
    @testset "end-to-end training with Optimization.solve" begin
        # Build a tiny regression dataset  y = W*x  with W known
        n_features = 4
        n_out = 2
        n_samples = 256
        BS_train = 64

        W_true = randn(Float32, n_out, n_features)
        X_train = randn(Float32, n_features, n_samples)
        Y_train = W_true * X_train

        w = uniform_weights(n_samples)
        loader = WeightedDataLoader((X_train, Y_train), w;
                                    batchsize=BS_train, shuffle=true)

        @test loader isa MLUtils.DataLoader

        # Simple linear model: θ is a flattened (n_out × n_features) weight matrix
        θ_init = zeros(Float32, n_out * n_features)

        # Loss: mean squared error over a single batch
        # `p` receives one batch (a Tuple) from the loader
        function loss(p, batch)
            Xb, Yb = batch          # unpack the tuple
            W = reshape(p, n_out, n_features)
            Ŷ = W * Xb
            return sum(abs2, Ŷ .- Yb) / size(Xb, 2)
        end

        # Compute initial loss on the full training set as reference
        W0 = reshape(θ_init, n_out, n_features)
        loss_init = sum(abs2, W0 * X_train .- Y_train) / n_samples

        # Set up Optimization problem with the DataLoader
        opt_func = OptimizationFunction(loss, Optimization.AutoZygote())
        prob = OptimizationProblem(opt_func, θ_init, loader)

        # Train for a few epochs
        sol = Optimization.solve(prob, OptimizationOptimisers.Adam(0.05f0);
                                 epochs=30)

        W_sol = reshape(sol.u, n_out, n_features)
        loss_final = sum(abs2, W_sol * X_train .- Y_train) / n_samples

        @test loss_final < loss_init          # loss must decrease
        @test loss_final < 0.5f0 * loss_init  # by at least 50 %
    end

    # ==========================================================
    # Core requirement: loader.data[i] returns the i-th raw array
    # ==========================================================
    @testset "getindex returns raw array" begin
        N  = 100
        X1 = rand(Float32, 8, N)
        X2 = rand(Float32, 4, N)
        X3 = rand(Float32, 2, N)

        loader = WeightedDataLoader((X1, X2, X3), uniform_weights(N);
                                    batchsize=32, partial=false)

        @test loader.data[1] === X1
        @test loader.data[2] === X2
        @test loader.data[3] === X3

        @test size(loader.data[1]) == size(X1)
        @test size(loader.data[2]) == size(X2)
        @test size(loader.data[3]) == size(X3)
    end

    # ==========================================================
    # length returns the number of arrays in the tuple
    # ==========================================================
    @testset "length returns tuple length" begin
        N  = 100
        X1 = rand(Float32, 8, N)
        X2 = rand(Float32, 4, N)
        X3 = rand(Float32, 2, N)

        loader = WeightedDataLoader((X1, X2, X3), uniform_weights(N);
                                    batchsize=32, partial=false)

        @test length(loader.data) == 3

        loader2 = WeightedDataLoader((X1, X2), uniform_weights(N);
                                     batchsize=32, partial=false)
        @test length(loader2.data) == 2
    end
