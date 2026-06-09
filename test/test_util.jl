using Test
using HybridVariationalInference: vectuptotupvec_allowmissing, vectuptotupvec, insert_zeros
using HybridVariationalInference: replace_columns_matrix
using HybridVariationalInference: HybridVariationalInference as HVI
using Zygote
using Distributions
using LinearAlgebra



@testset "OneBasedVectorWithZero" begin
    # Standard Julia 1-based vector (no underlying shift)
    v1 = HVI.OneBasedVectorWithZero([10,20,30])
    @test @inferred v1[1] == 10
    @test v1[2] == 20
    @test v1[3] == 30
    @test v1[[true,true,true]] == [10,20,30]

    v1[1] = 100
    @test v1[1] == 100
    @test v1.data[1] == 100

    # Underlying 0-based vector should still present 1-based API.
    struct ZeroBasedVector{T} <: AbstractVector{T}
        data::Vector{T}
    end
    Base.size(v::ZeroBasedVector) = size(v.data)
    Base.length(v::ZeroBasedVector) = length(v.data)
    Base.axes(v::ZeroBasedVector) = (0:length(v)-1,)
    Base.getindex(v::ZeroBasedVector, i::Integer) = v.data[i+1]
    Base.setindex!(v::ZeroBasedVector, x, i::Integer) = (v.data[i+1] = x)

    data0 = ZeroBasedVector([10,20,30])
    v0 = HVI.OneBasedVectorWithZero(data0)
    @test eltype(v0) == Int
    @test @inferred v0[1] == 10
    @test v0[2] == 20
    @test v0[3] == 30

    v0[1] = 100
    @test v0[1] == 100
    @test v0.data[0] == 100

    # bounds for non-zero indices should be 1..length for wrapper independent of underlying axis
    @test v0[0] == 0
    @test_throws BoundsError v0[-1]
    @test_throws BoundsError v0[4]
    @test_throws BoundsError v0[0] = 50

    @test collect(v0) == [100,20,30]
    @test length(v0) == 3
    @test axes(v0) == (Base.OneTo(3),)

    # gradient pass through values works
    g = Zygote.gradient(x -> sum(HVI.OneBasedVectorWithZero(x)), [1.0, 2.0, 3.0])
    @test g == ([1.0,1.0,1.0],)

    v1 = HVI.OneBasedVectorWithZero([10,20,30])
    # @usingany Cthulhu
    # @descend_code_warntype v1[0]
    @inferred v1[0]
    @test v1[0] == 0 # default value at index 0 is zero
    v1 = HVI.OneBasedVectorWithZero([10,20,30]; val_at_zero=-5)
    @test @inferred(v1[0]) == -5 # default value at index 0 is zero
    @test v1[[1,1,2,3]] == [10,10,20,30]
    @test v1[[1,0,0,3]] == [10,-5,-5,30]
    g1 = Zygote.gradient(y -> sum(HVI.OneBasedVectorWithZero(y)[[1,1,0,0,2]]), [1.0,2.0,3.0])
    @test g1 == ([2.0,1.0,0.0],)
end;

@testset "take_n!" begin
    it = HVI.drop_iterate(1:5) # initialize the iterator
    a1 = HVI.take_n!(it,3)
    @test collect(a1) == [1,2,3]
    a2 = HVI.take_n!(it,3)
    @test collect(a2) == [4,5]  # only two element left, so return those
    a3 = HVI.take_n!(it,3)
    @test collect(a3) == [] # no elements left, so return empty vector
end

@testset "insert_zeros" begin
    @test @inferred HVI.insert_zeros([1,2,3], [2,5]) == [1,0,2,3,0]
    @test @inferred HVI.insert_zeros([1,2,3], [1,5]) == [0,1,2,3,0]
    @test_throws AssertionError  HVI.insert_zeros([1,2,3], [2,4])
    @test_throws AssertionError  HVI.insert_zeros([1,2,3], [1,1,6])
    Zygote.gradient(x -> sum(HVI.insert_zeros(x, [2,5])), [1,2,3]) 

    #
    v = HVI.OneBasedVectorWithZero([10,20,30])
    idxs = HVI.insert_zeros(1:length(v), [2,5])
    res = @inferred v[idxs]
    # @usingany Cthulhu
    #@descend_code_warntype v[idxs]
    @test res == [10, 0, 20, 30, 0]
    @test Zygote.gradient(x -> sum(x[idxs]), v) == ([1.0, 1.0, 1.0],)

end;

@testset "vectuptotupvec" begin
    vectup = [(1,1.01, "string 1"), (2,2.02, "string 2")] 
    tupvec = @inferred vectuptotupvec(vectup)
    #@code_warntype vectuptotupvec_allowmissing(vectup)
    @test tupvec == ([1, 2], [1.01, 2.02], ["string 1", "string 2"])  
    @test typeof(first(tupvec)) == Vector{Int}
    # empty not allowed
    @test_throws Exception tupvec = vectuptotupvec([])
    # do not allow tuples of different types - note the Float64 in first entry
    vectupm = [(1.00,1.01, "string 1"), (2,2.02, "string 2",:asymbol)] 
    @test_throws Exception tupvecm = vectuptotupvec(vectupm)
    #
    gr = Zygote.gradient(x -> sum(vectuptotupvec(x)[1]), vectup)
end;


@testset "vectuptotupvec_allowmissing" begin
    vectup = [(1,1.01, "string 1"), (2,2.02, "string 2")] 
    tupvec = @inferred vectuptotupvec_allowmissing(vectup)
    #@code_warntype vectuptotupvec_allowmissing(vectup)
    @test tupvec == ([1, 2], [1.01, 2.02], ["string 1", "string 2"])  
    @test typeof(first(tupvec)) == Vector{Union{Missing,Int}}
    # empty not allowed
    @test_throws Exception tupvec = vectuptotupvec_allowmissing([])
    # first missing
    vectupm = [missing, (1,1.01, "string 1"), (2,2.02, "string 2")] 
    vectuptotupvec_allowmissing(vectupm)
    tupvecm = @inferred vectuptotupvec_allowmissing(vectupm)
    @test ismissing(vectupm[1]) # did not change underlying vector
    #@code_warntype vectuptotupvec_allowmissing(vectupm)
    @test isequal(tupvecm, ([missing, 1, 2], [missing, 1.01, 2.02], [missing, "string 1", "string 2"]))
    # do not allow tuples of different length
    vectupm = [(1,1.01, "string 1"), (2,2.02, "string 2",:asymbol)] 
    @test_throws Exception tupvecm = vectuptotupvec_allowmissing(vectupm)
    # do not allow tuples of different types - note the Float64 in first entry
    vectupm = [(1.00,1.01, "string 1"), (2,2.02, "string 2",:asymbol)] 
    @test_throws Exception tupvecm = vectuptotupvec_allowmissing(vectupm)
    #
    vectupm = [missing, (1,1.01, "string 1"), (2,2.02, "string 2")] 
    gr = Zygote.gradient(x -> sum(skipmissing(vectuptotupvec_allowmissing(x)[1])), vectupm)
end;

@testset "log_density_mvn_cholesky" begin
    # Test cases: different dimensions and covariance matrices
    test_cases = [
        (2, 0.1),   # 2D, small variance
        #(3, 1.0),   # 3D, moderate
        #(5, 2.0),   # 5D, larger
        (10, 0.5),  # 10D, small
    ]
    # n, scale = test_cases[3]
    for (n, scale) in test_cases
        # Generate a random positive definite covariance matrix
        A = randn(n, n)
        C = A * A' + scale * I  # Ensure positive definite
        U = cholesky(C).L'
        x = randn(n)
        log_p_custom = HVI.log_density_mvn_cholesky(U, x)
        # Compute log-density using Distributions.jl
        dist = MvNormal(zeros(n), C)
        log_p_builtin = logpdf(dist, x)
        @test isapprox(log_p_custom, log_p_builtin, atol=1e-10)
    end
    # test Zygote can handle this function
    () -> begin
        A = rand(3,3)
        U = cholesky(A * A' + I).L'
        Zygote.gradient(x -> HVI.log_density_mvn_cholesky(U, x), randn(3))
    end
end

@testset "replace_columns_matrix" begin
    # Test case 1: Basic replacement
    x = [1 2 3; 4 5 6; 7 8 9]
    col_indices = [1, 3]
    y = [10 11; 12 13; 14 15]
    result = replace_columns_matrix(x, col_indices, y)
    expected = [10 2 11; 12 5 13; 14 8 15]
    @test result ≈ expected
    tmp = Zygote.gradient(y -> sum(replace_columns_matrix(x, col_indices, y)), y)
    #    
    # Test case 2: Replace all columns
    x = [1 2; 3 4]
    col_indices = [1, 2]
    y = [5 6; 7 8]
    #
    result = replace_columns_matrix(x, col_indices, y)
    expected = [5 6; 7 8]
    @test result ≈ expected
    #    
    # Test case 3: Replace single column
    x = [1 2 3; 4 5 6]
    col_indices = [2]
    y = reshape([10; 20], :,1)
    result = replace_columns_matrix(x, col_indices, y)
    expected = [1 10 3; 4 20 6]
    @test result ≈ expected
    #
    # Test case 4: Empty replacement
    x = [1.0 2; 3 4]
    col_indices = Int[]
    y = Matrix{Float64}(undef, 2, 0)
    result = replace_columns_matrix(x, col_indices, y)
    expected = x
    @test result ≈ expected
    #    
    # Test case 5: Differentiation compatibility (Zygote)
    x = [1.0 2.0; 3.0 4.0]
    col_indices = [1]
    y = reshape([5.0; 6.0], :, 1)
    # Check that gradient can be computed
    gradient((x,y) -> sum(replace_columns_matrix(x, col_indices, y)), x,y)
end


