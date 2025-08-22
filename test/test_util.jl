using Test
using HybridVariationalInference: vectuptotupvec_allowmissing, vectuptotupvec
using Zygote

@testset "vectuptotupvec" begin
    vectup = [(1,1.01, "string 1"), (2,2.02, "string 2")] 
    tupvec = @inferred vectuptotupvec(vectup)
    #@code_warntype vectuptotupvec_allowmissing(vectup)
    @test tupvec == ([1, 2], [1.01, 2.02], ["string 1", "string 2"])  
    @test typeof(first(tupvec)) == Vector{Int}
    # empty not allowed
    @test_throws Exception tupvec = vectuptotupvec([])
    # do not allow tuples of differnt types - note the Float64 in first entry
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
    # do not allow tuples of differnt types - note the Float64 in first entry
    vectupm = [(1.00,1.01, "string 1"), (2,2.02, "string 2",:asymbol)] 
    @test_throws Exception tupvecm = vectuptotupvec_allowmissing(vectupm)
    #
    vectupm = [missing, (1,1.01, "string 1"), (2,2.02, "string 2")] 
    gr = Zygote.gradient(x -> sum(skipmissing(vectuptotupvec_allowmissing(x)[1])), vectupm)
end;

