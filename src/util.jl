"""
    vectuptotupvec(vectup)
    vectuptotupvec_allowmissing(vectup)

Typesafe convert from Vector of Tuples to Tuple of Vectors.
The first variant does not allow for `missing` in `vectup`.
The second variant allows for `missing` but has `eltype` of `Union{Missing, ...}` in  
all components of the returned Tuple, also when there were not `missing` in `vectup`. 

# Arguments
* `vectup`: A Vector of identical Tuples 

# Examples
```jldoctest; output=false
vectup = [(1,1.01, "string 1"), (2,2.02, "string 2")] 
HybridVariationalInference.vectuptotupvec_allowmissing(vectup) == 
  ([1, 2], [1.01, 2.02], ["string 1", "string 2"])
# output
true
```
"""
function vectuptotupvec(vectup::AbstractVector{<:Tuple}) 
    Ti = eltype(vectup).parameters
    npar = length(Ti)
    ntuple(i -> 
        (getindex.(vectup, i))::Vector{Ti[i]}, npar)
end
function vectuptotupvec_allowmissing(
    vectup::AbstractVector{<:Union{Missing,Tuple}}) 
    Ti = nonmissingtype(eltype(vectup)).parameters
    npar = length(Ti)
    Tim = ntuple(i -> Union{Missing,Ti[i]}, npar)
    ntuple(i -> begin
        allowmissing(passmissing(getindex).(vectup, i))::Vector{Tim[i]}
    end, npar)
end

"""
    take_n!(itr, n)  

Peel off the first `n` elements of an drop-iterator `itr` and 
return them as a vector, while mutating `itr` to now start after those `n` elements.    

# Examples
```jldoctest; output=false
it = drop_iterate(1:5) # initialize the iterator

a1 = take_n!(it,3)
collect(a1) == [1,2,3]

a2 = take_n!(it,3)
collect(a2) == [4,5]  # only two element left, so return those

a3 = take_n!(it,3)
collect(a3) == [] # no elements left, so return empty vector
```
"""
function take_n!(itr::Base.RefValue{<:Base.Iterators.Drop},n)
    ans = Iterators.take(itr[], n)
    itr[] = Iterators.drop(itr[], n)
    ans
end
drop_iterate(x) = Ref(Iterators.drop(x,0))

"""
    insert_zeros(v, positions)

Return a new vector with `zero(eltype(v))` inserted at each position in `positions`.
Positions are applied in order against the growing vector (as if sequential inserts),
so later indices are interpreted on the updated result.
Only one output vector is allocated.
"""
function insert_zeros(v::AbstractVector, positions::AbstractVector{<:Integer})
    # does not work with Zygote, but its only used to create the indexing vector
    # v = [10,20,30];positions = [2, 5] # means insert zeros before original v[2] and v[4], so final output has zeros at those positions.
    @assert length(v)+ length(positions) == positions[end] "The last position in `positions` must be equal to the final length of the output vector after all insertions."
    dpos1 = diff(positions) .- 1
    @assert all(dpos1 .>= 0) "Positions must be in strictly ascending order."
    # length of blocks before insert is diff(postions) -1 
    length_blocks_beforeinsert = Iterators.flatten((first(positions) .- 1, dpos1))
    #collect(length_blocks_beforeinsert) == [1,2]
    it = drop_iterate(v)  # to allow take_n!
    #collect(HVI.take_n!(it, 4)) == v
    gen = (Iterators.flatten(
        (take_n!(it, l), zero(eltype(v)))) for l in length_blocks_beforeinsert)
    # collect(Iterators.flatten(gen)) == [10, 0, 20, 30, 0]
    return collect(Iterators.flatten(gen))
end

function ChainRulesCore.rrule(::typeof(insert_zeros), v::AbstractVector, positions::AbstractVector{<:Integer})
    y = insert_zeros(v, positions)
    # Reverse pass (pullback) for gradient of `insert_zeros`:
    # - We only propagate gradients into `v`.
    # - `positions` is treated as non-differentiable (NoTangent()).
    # We ignore the gradients for the postions, where zero was inserted
    # Otherwise, we just need to extract the corresponding positions in ȳ
    function pullback(ȳ)
        n = length(v)
        m = length(positions)
        grad_v = OneBasedVectorWithZero(ȳ[:])[1:(n+m) .∉ Ref(positions)] 
        return NoTangent(), grad_v, NoTangent()
    end
    return y, pullback
end
