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
    insert_zeros(v, positions)

Return a new vector with `zero(eltype(v))` inserted at each position in `positions`.
Positions are applied in order against the growing vector (as if sequential inserts),
so later indices are interpreted on the updated result.
Only one output vector is allocated.
"""
function insert_zeros(v::AbstractVector, positions::AbstractVector{<:Integer})
    n = length(v)
    m = length(positions) # number of insert positions

    # If there are no insertions, just copy and return the original vector.
    if m == 0
        return collect(v)
    end

    # Convert positions to an array; we can index into p in the same order.
    # (This also allows `positions` to be any AbstractVector.)
    p = positions

    # Validate each insertion position using the growing-vector semantics:
    # after k-1 insertions, the vector length is n + k - 1, so pk must be in [1, length+1].
    for (k, pk) in enumerate(positions)
        curr_len = n + k - 1
        if pk < 1 || pk > curr_len + 1
            throw(ArgumentError("insert_zeros position $pk out of bounds for length $curr_len"))
        end
    end

    # Map each sequential insert position to a bucket index in the final sequence,
    # where 0 <= x[k] <= n (0 means before first element, n means after last element).
    # Formula: p[k]-1 (zero-based insert target in original coords) minus the number of
    # previously inserted positions that fall before p[k], because they shift later indexes.
    x = [p[k] - 1 - count(j -> p[j] < p[k], 1:k-1) for k in 1:m]

    # zero_chain(i) yields one `zero(eltype(v))` per insertion that should occur in bucket i.
    zero_chain = i -> (zero(eltype(v)) for k in 1:m if x[k] == i)

    # Build output lazily by iterating i from 0..n:
    #   for each i, first emit zeros for that bucket, then emit original v[i+1] when i<n.
    # This avoids explicit vector mutation and matches sequential insert semantics in one pass.
    out_iter = Iterators.flatten(
        (i < n ? IterTools.chain(zero_chain(i), (v[i+1],)) : zero_chain(i)) for i in 0:n
    )

    # Materialize the iterator into a vector and return.
    return collect(out_iter)
end

insert_zeros(v::AbstractVector, position::Integer) = insert_zeros(v, [position])

function ChainRulesCore.rrule(::typeof(insert_zeros), v::AbstractVector, positions::AbstractVector{<:Integer})
    # Forward pass: compute output normally.
    y = insert_zeros(v, positions)

    # Reverse pass (pullback) for gradient of `insert_zeros`:
    # - We only propagate gradients into `v`.
    # - `positions` is treated as non-differentiable (NoTangent()).
    # - For each output value:
    #    * if it's one of the inserted zeros => no contribution to grad_v
    #    * if it's from original v[k] => add ȳ to grad_v[k]
    function pullback(ȳ)
        n = length(v)
        m = length(positions)

        # Compute insertion buckets x[k] as in insert_zeros.
        p = positions
        x = [p[k] - 1 - count(j -> p[j] < p[k], 1:k-1) for k in 1:m]

        # Determine how many zeros are produced in each bucket i in 0:n.
        zeros_per_bucket = zeros(Int, n + 1)
        for xi in x
            zeros_per_bucket[xi + 1] += 1
        end

        grad_v = zeros(eltype(v), n)
        src = 1   # index in original input vector v from which gradient flows
        out_idx = 1  # current index in output vector y/ȳ

        for i in 0:n
            # Step past inserted zeros for this bucket (all zero insertions from insert_zeros)
            # out_idx becomes the position of the next original value (if any) after zero insertions.
            out_idx += zeros_per_bucket[i + 1]

            if i < n
                # Map output gradient back to exactly one input element v[src].
                # ȳ[out_idx] refers to the gradient at output position corresponding to v[src]
                grad_v[src] += ȳ[out_idx]
                src += 1

                # Advance to next output index (past this original value)
                out_idx += 1
            end
        end

        return NoTangent(), grad_v, NoTangent()
    end

    return y, pullback
end
