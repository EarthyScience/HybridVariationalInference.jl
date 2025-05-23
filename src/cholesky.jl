sumn(n::T) where {T} = n * (n + one(T)) ÷ T(2)

""" 
Inverse of s = sumn(n) for positive integer `n`.

Gives an inexact error, if given s was not such a sum.
"""
invsumn(s::T) where {T} = T(-0.5 + sqrt(1 / 4 + 2 * s)) # inversion of n*(n+1)/2

"""
Convert vector v columnwise entries of upper diagonal matrix to UppterTriangular
"""
function vec2utri(v::AbstractVector{T}; n=invsumn(length(v))) where {T}
    # works with Zygote despite of doing mutation of k (see test_cholesky_structure.jl)
    #https://groups.google.com/g/julia-users/c/UARlZBCNlng/m/6tKKxIeoEY8J
    z = zero(T)
    k = 0
    #m = T[j >= i ? (k += 1; v[k]) : z for i in 1:n, j in 1:n] # no Zygote
    #m = [j >= i ? (k += 1; convert(T,v[k])) : convert(T,z) for i in 1:n, j in 1:n] # no Zygote
    m = [j >= i ? (k += 1; v[k]) : z for i in 1:n, j in 1:n]::AbstractMatrix{T} # for typestability
    UpperTriangular(m)
end

# see ext/HybridVariationalInferenceCUDAExt for CUDA-specific implementations

""" 
Compute the (one-based) position `(row, col)` within an upper tridiagonal matrix for
given (one-based) position, `s` within a packed vector representation.
"""
function vec2utri_pos(T, s)
    col = ceil(T, -0.5 + sqrt(1 / 4 + 2 * s)) # inversion of n*(n+1)/2
    cl = col - one(T)
    s1 = cl * (cl + one(T)) ÷ T(2) # first number in column
    row = s - s1
    (row, col)
end
vec2utri_pos(s) = vec2utri_pos(Int, s)

"""
Compute the index in the vector of entries in an upper tridiagonal matrix
"""
function utri2vec_pos(row, col)
    @assert row <= col
    utri2vec_pos_unsafe(row, col)
end
function utri2vec_pos_unsafe(row::T, col::T) where {T}
    # variant that does not check for unreasonable position away from upper tridiagonal
    sc1 = T((col - one(T)) * (col) ÷ T(2))
    sc1 + row
end

function vec2uutri(v::AbstractArray{T}; kwargs...) where {T}
    m = _vec2uutri(v; kwargs...)
    UnitUpperTriangular(m)
    #TriangularRFP(m, :U) # no adjoint
    #SymmetricPacked(m) # no adjoint
    #SymmetricPacked(v) # also no adjoint - does not yet care for main diagonal
end

# """
# Convert vector v columnwise entries of upper diagonal matrix to UnitUppterTriangular

# Avoid using this repeatedly on GPU arrays, because it only works on CPU (scalar indexing).
# There is a fallback that pulls `v` to the CPU, applies, and pushes back to GPU.
# """
function _vec2uutri(
    v::AbstractVector{T}; n=invsumn(length(v)) + one(T), diag=one(T)) where {T}
    z = zero(T)
    k = 0
    m = [j > i ? (k += 1; v[k]) : i == j ? diag : z for i in 1:n, j in 1:n]::AbstractMatrix{T}
    return (m)
end


function ChainRulesCore.rrule(::typeof(vec2uutri), v::AbstractArray; kwargs...)
    # does not change the value or the derivative but just reshapes
    # -> extract the components of matrix Δy into vector 
    function vec2uutri_pullback(Δy)
        (NoTangent(), uutri2vec(Δy))
    end
    return vec2uutri(v; kwargs...), vec2uutri_pullback
end

"""
Extract entries of upper diagonal matrix of UppterTriangular to columnwise vector
"""
function utri2vec(X::AbstractMatrix{T}) where {T}
    n = size(X, 1)
    lv = sumn(n)
    i = 0
    j = 1
    [
        begin
            if i == j
                i = 0
                j += 1
            end
            i += 1
            X[i, j]
        end
        for _ in 1:lv
    ]::AbstractVector{T}
end

"""
Extract entries of upper diagonal matrix of UnitUppterTriangular to columnwise vector
"""
function uutri2vec(X::AbstractMatrix{T}) where {T}
    n = size(X, 1) - 1
    lv = sumn(n)
    i = 0
    j = 2
    if n == 0; return T[]; end # otherwise Any[] is returned
    v = [
        begin
            if i == j - 1
                i = 0
                j += 1
            end
            i += 1
            convert(T,X[i, j])
        end
        for _ in 1:lv
    ]::AbstractVector{T}
end

function ChainRulesCore.rrule(::typeof(uutri2vec), X::AbstractMatrix{T}) where {T}
    # does not change the value or the derivative but just reshapes
    # -> put the components of vector Δy into matrix
    # make sure that the gradient of main diagonal is zero rather than one
    function uutri2vec_pullback(Δy)
        (NoTangent(), vec2uutri(unthunk(Δy); diag=zero(T)))
    end
    return uutri2vec(X), uutri2vec_pullback
end

# moved to HybridVariationalInferenceCUDAExt
#function uutri2vec(X::CUDA.CuMatrix{T}; kwargs...) where {T}

"""
Takes a vector of entries of a lower UnitUpperTriangular matrix
and transforms it to an UpperTriangular that satisfies 
diag(U' * U) = 1.

This can be used to fit parameters that yield an upper Cholesky-Factor
of a Covariance matrix.

It uses the upper triangular matrix rather than the lower because it
involes a sum across columns, whereas the alternative of a lower triangular
uses sum across rows. 
Sum across columns is often faster, because entries of columns are contiguous.
"""
function transformU_cholesky1(v::AbstractVector;
    n=invsumn(length(v)) + 1
)
    U_scaled = vec2uutri(v; n)
    #Sc_inv = sqrt.(sum(abs2, U_scaled, dims=1))
    #U_scaled * Diagonal(1 ./ vec(Sc_inv))
    #U = U_scaled ./ Sc_inv
    U = U_scaled ./ sqrt.(sum(abs2, U_scaled, dims=1))
    return (UpperTriangular(U))
end
function transformU_cholesky1(
    v::GPUArraysCore.AbstractGPUVector; n=invsumn(length(v)) + 1)
    U_scaled = vec2uutri(v; n)
    U = U_scaled ./ sqrt.(sum(abs2, U_scaled, dims=1))
    # do not convert to UpperTrinangular on GPU, but full matrix
    #return (UpperTriangular(U))
    return U
end

# function transformU_block_cholesky1(v::CA.ComponentVector; 
#     ns=(invsumn(length(v[k])) + 1 for k in keys(v)) # may pass for efficiency
#     )
#     blocks = [transformU_cholesky1(v[k]; n) for (k, n) in zip(keys(v), ns)]
#     U = _create_blockdiag(v[first(keys(v))], blocks) # v only for dispatch: plain matrix for gpu
# end

"""
Compute the cholesky-factor parameter for a given single
correlation in a 2x2 matrix.
Invert the transformation of cholesky-factor parameterization.
"""
function compute_cholcor_coefficient_single(ρ)
    # invert  ρ = a / sqrt(a^2 + 1)
    sign(ρ) * sqrt(ρ^2/(1 - ρ^2))
end




"""
    get_ca_starts(vc::ComponentVector)

Return a tuple with starting positions of components in vc. 
Useful for providing information on correlactions among subranges in a vector.
"""
function get_ca_starts(vc::CA.ComponentVector)
    (1, (1 .+ cumsum((length(vc[k]) for k in front(keys(vc)))))...)
end
"omit the last n elements of an iterator"
front(itr, n=1) = Iterators.take(itr, length(itr) - n)


"""
    get_ca_ends(vc::ComponentVector)

Return a Vector with ending positions of components in vc. 
Useful for providing information on correlactions among subranges in a vector.
"""
function get_ca_ends(vc::CA.ComponentVector)
    #(cumsum(length(vc[k]) for k in keys(vc))...,)
    length(keys(vc)) == 0 ? Int[] : cumsum(length(vc[k])::Int for k in keys(vc))
end


"""
    get_cor_count(cor_ends::AbstractVector)
    get_cor_count(n_par::Integer)

Return number of correlation coefficients for a correlation matrix of size `(npar x npar)`
With blocks starting a positions given with tuple `cor_ends`.
"""
function get_cor_count(cor_ends::AbstractVector)
    sum(get_cor_counts(cor_ends))
end
function get_cor_counts(cor_ends::AbstractVector{T}) where {T}
    isempty(cor_ends) && return (zeros(T,1))
    cnt_blocks = (
        begin
            i == 1 ? cor_ends[i] : cor_ends[i] - cor_ends[i-1]
        end for i in 1:length(cor_ends)
    )
    get_cor_count.(cnt_blocks)
end
function get_cor_count(n_par::T) where T<:Number # <: Integer causes problems with  AD 
    sumn(n_par - one(T))
end


"""
    transformU_block_cholesky1(v::AbstractVector, cor_ends)

Transform a parameterization v of a blockdiagonal of upper triangular matrices
into the this matrix.
`cor_ends` is an AbstractVector of Integers specifying the last column of each block. 
E.g. For a matrix with a 3x3, a 2x2, and another single-entry block, 
the blocks start at columns (3,5,6). It defaults to a single entire block.
"""
function transformU_block_cholesky1(
    v::AbstractVector{T}, cor_ends::AbstractVector{TI}=Int[]) where {T,TI<:Integer}
    if length(cor_ends) <= 1 # if there is only one block, return it 
        # for type stability create a BlockDiagonal of a single block
        return _create_blockdiag(v, [transformU_cholesky1(v)])
    end
    cor_counts = get_cor_counts(cor_ends) # number of correlation parameters
    #@show cor_counts
    ranges = ChainRulesCore.@ignore_derivatives (
        begin
            cor_start = (i == 1 ? one(TI) : cor_counts[i-1] + one(TI))
            cor_start:cor_counts[i]
        end for i in 1:length(cor_counts)
    )
    #@show collect(ranges)
    blocks = [transformU_cholesky1(v[r]) for r in ranges]
    U = _create_blockdiag(v, blocks) # v only for dispatch: plain matrix for gpu
    return (U)
end

function _create_blockdiag(::AbstractArray{T}, blocks::AbstractArray) where {T}
    BlockDiagonal(blocks)
end


function _create_blockdiag(::GPUArraysCore.AbstractGPUArray, blocks::AbstractArray)
    # impose no special structure
    cat(blocks...; dims=(1, 2))
end

() -> begin
    tmp = sqrt.(sum(abs2, U_scaled, dims=1))
    tmp2 = sum(abs2, U_scaled, dims=1) .^ (-1 / 2)
    U_scaled * tmp'
end
