using Test
using HybridVariationalInference: vectuptotupvec_allowmissing, vectuptotupvec, insert_zeros
using HybridVariationalInference: HybridVariationalInference as HVI
using Zygote

@testset "insert_zeros" begin
    @test HVI.insert_zeros([1,2,3], [2,4]) == [1,0,2,0,3]
    @test HVI.insert_zeros([1,2,3], [1,1]) == [0,0,1,2,3]
    @test_throws ArgumentError HVI.insert_zeros([1,2], [4])
    @test HVI.insert_zeros([1,2,3], 2) == [1,0,2,3]
    @test @inferred HVI.insert_zeros([1,2,3,4], [1,3,4,7,9]) == [0,1,0,0,2,3,0,4,0]
    # position 9 is not available at the beginning, but after inserting 4 zeros, the vector has length 9, so it is valid
    @test_throws ArgumentError  HVI.insert_zeros([1,2,3,4], reverse([1,3,4,7,9])) 
    @test Zygote.gradient(x -> sum(HVI.insert_zeros(x, [1,3,4,7,9])), [1,2,3,4]) == ([1.0, 1.0, 1.0, 1.0],)

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

