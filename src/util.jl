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
function vectuptotupvec(vecntup::AbstractVector{<:NamedTuple{KEYS}}) where KEYS
    #vectup = values.(vecntup)   
    Ti = eltype(vecntup).parameters[2].parameters
    npar = length(Ti)
    tupvec = ntuple(i -> 
        (getindex.(vecntup, i))::Vector{Ti[i]}, npar)
    NamedTuple{KEYS}(tupvec)
end
# function vectuptotupvec_(vecntup::AbstractVector{<:NamedTuple}) 
#     vectup = values.(vecntup)
#     tupvec = vectuptotupvec(vectup)
#     NamedTuple{keys(first(vecntup))}(tupvec)
# end


"""
    take_n!(itr, n)  

Peel off the first `n` elements of an drop-iterator `itr` and 
return them as a vector, while mutating `itr` to now start after those `n` elements.    

# Examples
```jldoctest; output=false
it = HybridVariationalInference.drop_iterate(1:5) # initialize the iterator

a1 = HybridVariationalInference.take_n!(it,3)
collect(a1) == [1,2,3]

a2 = HybridVariationalInference.take_n!(it,3)
collect(a2) == [4,5]  # only two element left, so return those

a3 = HybridVariationalInference.take_n!(it,3)
collect(a3) == [] # no elements left, so return empty vector
# output
true
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
    @assert length(v) + length(positions) == positions[end] "The last position in `positions` must be equal to the final length of the output vector after all insertions."
    dpos1 = diff(positions) .- 1
    @assert all(dpos1 .>= 0) "Positions must be in strictly ascending order."
    # length of blocks before insert is diff(positions) -1 
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
    # We ignore the gradients for the positions, where zero was inserted
    # Otherwise, we just need to extract the corresponding positions in ȳ
    function pullback(ȳ)
        n = length(v)
        m = length(positions)
        grad_v = OneBasedVectorWithZero(ȳ[:])[1:(n+m) .∉ Ref(positions)] 
        return NoTangent(), grad_v, NoTangent()
    end
    return y, pullback
end

"""
    cat_namedtuple_lastdim(nt_agg, nt; along) 

Reducing function that takes two NamedTuple objects of the same type and
concatenates each component along specified dimension.

Optionally, the dimension at which to concatenate can be specified in
the `along` NamedTuple argument.
"""
function cat_namedtuple_lastdim(nt_agg::NamedTuple, nt::NamedTuple; along = map(ndims, nt)::NamedTuple) 
    NamedTuple( map(nt, keys(nt)) do comp, key
        key => cat(nt_agg[key], comp; dims = along[key])
    end)
end

"""
    index_at_dim(x::AbstractArray{T, N}, i::AbstractVector{Int}; dim::Int) where {T, N}

Index into array `x` along dimension `dim` using indices `i`, while selecting
all elements along all other dimensions.

# Arguments
- `x::AbstractArray{T, N}`: Input array of type `T` and `N` dimensions.
- `i::AbstractVector{Int}`: Vector of indices to select along dimension `dim`.
- `dim::Int`: The dimension along which to index.

# Returns
- An array of the same type as `x` with the same number of dimensions, where
  the size along `dim` is `length(i)` and all other dimensions are unchanged.

# Examples
```julia
x = reshape(1:24, 3, 2, 4)

# Index along dimension 1
index_at_dim(x, [1]; dim=1)        # 1×2×4 array

# Index along dimension 2
index_at_dim(x, [1, 3]; dim=1)     # 2×2×4 array

# Index along dimension 3
index_at_dim(x, [2, 4]; dim=3)     # 3×2×2 array
```
"""
function index_at_dim(x::AbstractArray{T, N}, i::AbstractVector{Int}; dim::Int) where {T, N}
    colons = ntuple(d -> d == dim ? i : Colon(), N)
    return x[colons...]
end



using LinearAlgebra

"""
    log_density_mvn_cholesky(x, U)

Compute the log-density of a zero-mean multivariate normal distribution
with covariance matrix C = U' * U, where U is the upper Cholesky factor.

Arguments:
- x: vector of length n (the sample)
- U: upper triangular Cholesky factor of the covariance matrix (n × n)

Returns:
- log p(x) ∈ ℝ: log-density at x
"""
function log_density_mvn_cholesky(U::AbstractMatrix{T}, x::AbstractVector{T}) where T
    n = length(x)
    # Solve L * y = x for y (forward substitution)
    y = U' \ x  # Efficient triangular solve
    # Compute ||y||^2 = y' * y
    quad_form = dot(y, y)  # or: sum(abs2, y)
    # Compute sum of log(diagonals) of U → this is log(sqrt(det(C)))
    if any(diag(U) .< 0) 
        @info("log_density_mvn_cholesky: encountered diag(U) components smaller than zero: $(diag(U))")
        # ignore_derivatives() do
        #     Main.@infiltrate_main
        # end
    end
    log_det_C_half = sum(log, diag(U))  # = 0.5 * log|C|
    # Full log-density formula
    log2π = T(1.8378770664093453)   #log(2π)
    log_density = -T(0.5) * (quad_form + T(2) * log_det_C_half + n * log2π)
    return log_density
end

# Prompt: Write a Julia function replace_values(x::Matrix, i_sites::Vector{Int}, pos::Vector{Int}, y::Matrix) that returns a new matrix where x[i_sites, pos] is replaced by y and all other values are unchanged. The function must be fully non-mutating and compatible with Zygote automatic differentiation. Use one-hot projection matrices P_row and P_col to scatter y into the full matrix space via P_row * y * P_col', and blend with x using a binary mask derived from the outer product of row and column indicator vectors. It should use matrix comprehensions to form P_row and P_col.
"""
    replace_values_matrix(x::Matrix, i_sites::Vector{<:Integer}, pos::Vector{<:Integer}, y::Matrix)

Return a new matrix where the submatrix at positions `x[i_sites, pos]` is replaced by `y`, 
while all other values remain unchanged.

This function performs a non-mutating replacement using one-hot projection matrices and 
a binary mask derived from the outer product of indicator vectors. It is designed to be 
compatible with automatic differentiation frameworks like Zygote.

### Parameters
- `x`: The input matrix of size `(m, n)` to be modified.
- `i_sites`: A vector of row indices (1-based) specifying which rows to replace.
- `pos`: A vector of column indices (1-based) specifying which columns to replace.
- `y`: The replacement matrix of size `(length(i_sites), length(pos))`.

### Returns
- A new matrix of the same size as `x`, where `x[i_sites, pos]` is replaced by `y`.

### Details
- The function constructs one-hot projection matrices `P_row` (size `m × length(i_sites)`) 
  and `P_col` (size `n × length(pos)`) using matrix comprehensions.
- The replacement values are scattered into the full matrix space via `P_row * y * P_col'`.
- A binary mask is created using the outer product of indicator vectors for `i_sites` and `pos`.
- The result is computed as `x .* (1 - mask) + (P_row * y * P_col') .* mask`, blending the original 
  matrix with the scattered replacement values.

### Example
```julia
x = [1 2 3; 4 5 6; 7 8 9]
i_sites = [1, 3]
pos = [2, 3]
y = [10 11; 12 13]

result = replace_values(x, i_sites, pos, y)
# result = [1 10 11; 4 5 6; 7 12 13]
"""
function replace_values_matrix(x::AbstractMatrix{T}, i_sites::AbstractVector{<:Integer}, pos::AbstractVector{<:Integer}, y::AbstractMatrix{T}) where T
    # Precompute projection matrices (one-hot style)
    P_row = [i == k for i in 1:size(x, 1), k in i_sites]  # n_rows × length(i_sites)
    P_col = [j == k for j in 1:size(x, 2), k in pos]      # n_cols × length(pos)
    # Project y into full matrix space: P_row * y * P_col'
    y_full = P_row * y * P_col'
    # Compute mask as outer product of indicator vectors
    # row_mask = [i in i_sites for i in 1:size(x, 1)]
    # col_mask = [j in pos     for j in 1:size(x, 2)]
    # replace_mask0 = row_mask .* col_mask'
    replace_mask = sum(P_row, dims=2) .* sum(P_col, dims=2)'
    res = (1 .- replace_mask) .* x .+ replace_mask .* y_full
    res
end

"""
    replace_columns_matrix(x::Matrix, col_indices::Vector{Int}, y::Matrix)

Return a new matrix where the specified columns of `x` are replaced by the columns of `y`.

This function performs a non-mutating column replacement using one-hot projection matrices 
and is compatible with automatic differentiation frameworks like Zygote.

### Parameters
- `x`: The input matrix of size `(m, n)` to be modified.
- `col_indices`: A vector of column indices (1-based) specifying which columns to replace.
- `y`: The replacement matrix of size `(m, length(col_indices))`.

### Returns
- A new matrix of the same size as `x`, where the columns at positions `col_indices` are replaced by the corresponding columns of `y`.

### Details
- The function constructs a one-hot projection matrix `P_col` (size `n × length(col_indices)`) 
  using matrix comprehensions, where each column corresponds to a target column index.
- The replacement values are scattered into the full matrix space via `x * (I - P_col * P_col') + y * P_col'`.
- The operation is differentiable with respect to all inputs.

### Example
```julia
x = [1 2 3; 4 5 6; 7 8 9]
col_indices = [1, 3]
y = [10 11; 12 13; 14 15]

result = replace_columns_matrix(x, col_indices, y)
#result = HVI.replace_columns_matrix(x, col_indices, y)
# result == [10 2 11; 12 5 13; 14 8 15]
```

### Notes
- All column indices must be valid (1-based).
- The function is fully non-mutating and compatible with Zygote for automatic differentiation.
- The operation is differentiable with respect to all inputs.
"""
function replace_columns_matrix(x::AbstractMatrix{T}, col_indices::AbstractVector{<:Integer}, y::AbstractMatrix{T}) where T
    # Get dimensions
    m, n = size(x)
    p = length(col_indices)
    
    # Validate inputs
    # @assert p == size(y, 2) "Number of columns in y must match length of col_indices"
    # @assert m == size(y, 1) "Number of rows in y must match number of rows in x"
    # @assert all(1 .<= col_indices .<= n) "col_indices must be valid column indices"
    
    # Create one-hot projection matrix using matrix comprehension
    # P_col: n × p matrix where each column is a one-hot vector for col_indices
    P_col = ChainRulesCore.@ignore_derivatives [
        j_it == j_col ? one(T) : zero(T) for j_it in 1:n, j_col in col_indices]
    
    # Replace columns: keep original columns (I - P_col * P_col') and replace with y * P_col'
    # This is equivalent to: x * (I - P_col * P_col') + y * P_col'
    result = x * (LinearAlgebra.I - P_col * P_col') .+ y * P_col'
    return result
end

"""
    generate_repeated_integers(n_MC::Int, n_sample_ranef::Int) -> Vector{Int}

Generate a vector of increasing integers where each integer is repeated `n_sample_ranef`
times, except possibly the last one, such that the total length of the vector is exactly
`n_MC`.

# Arguments
- `n_MC::Int`: The total length of the output vector. Must be a positive integer.
- `n_sample_ranef::Int`: The number of times each integer is repeated. Must be a positive integer.

# Returns
- `Vector{Int}`: A vector of length `n_MC` where each integer `i` appears `n_sample_ranef`
  times, except for the last integer which appears `mod(n_MC, n_sample_ranef)` times if
  `n_MC` is not a multiple of `n_sample_ranef`, and `n_sample_ranef` times otherwise.

# Examples
```julia-repl
julia> generate_repeated_integers(8, 5)
8-element Vector{Int64}:
 1, 1, 1, 1, 1, 2, 2, 2
 ```
 """
function generate_repeated_integers(n_MC::Integer, n_sample_ranef::Integer)
    # Calculate how many complete groups we need
    n_groups = ceil(Int, n_MC / n_sample_ranef)
    # Generate the full repeated vector
    full_vec = repeat(1:n_groups, inner = n_sample_ranef)
    # Trim to exactly n_MC elements
    return full_vec[1:n_MC]
end


