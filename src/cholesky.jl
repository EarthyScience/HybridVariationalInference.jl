sumn(n::T) where T = n*(n+one(T)) ÷ T(2)

""" 
Inverse of s = sumn(n) for positive integer `n`.

Gives an inexact error, if given s was not such a sum.
"""
invsumn(s::T) where T = T(-0.5 + sqrt(1 / 4 + 2 * s)) # inversion of n*(n+1)/2


"""
Convert vector v columnwise entries of upper diagonal matrix to UppterTriangular
"""
function vec2utri(v::AbstractVector{T}; n=invsumn(length(v))) where {T}
    # works with Zygote despite of doing mutation of k (see test_cholesky_structure.jl)
    #https://groups.google.com/g/julia-users/c/UARlZBCNlng/m/6tKKxIeoEY8J
    z = zero(T)
    k = 0
    m = [j >= i ? (k += 1; v[k]) : z for i = 1:n, j = 1:n]
    UpperTriangular(m)
end

function vec2utri(v::GPUArraysCore.AbstractGPUVector{T}; n=invsumn(length(v))) where {T}
    z = zero(T)
    k = 0
    m = CUDA.allowscalar() do
        CuArray([j >= i ? (k += 1; v[k]) : z for i = 1:n, j = 1:n])
    end
    # m = CUDA.zeros(T,n,n)
    # @cuda threads = 256 vec2utri_gpu!(m, v) # planned to put v into positions of m
    return (m)
end

function vec2utri_gpu!(m, v::AbstractVector) 
    index = threadIdx().x    # this example only requires linear indexing, so just use `x`
    stride = blockDim().x
    for i = index:stride:length(v)
        row, col = vec2utri_pos(i)
        @inbounds m[row, col] = v[i]
    end
    return nothing # important
end

""" 
Compute the (one-based) position `(row, col)` within an upper tridiagonal matrix for
given (one-based) position, `s` within a packed vector representation.
"""
function vec2utri_pos(T, s) 
    col = ceil(T, -0.5 + sqrt(1 / 4 + 2 * s)) # inversion of n*(n+1)/2
    cl = col - one(T)
    s1 = cl*(cl+one(T)) ÷ T(2) # first number in column
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
function utri2vec_pos_unsafe(row::T, col::T) where T
    # variant that does not check for unreasonable position away from upper tridiagonal
    sc1 = T((col-one(T))*(col) ÷ T(2))
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

# Avoid using this repeatetly on GPU arrays, because it only works on CPU (scalar indexing).
# There is a fallback that pulls `v` to the CPU, applies, and pushes back to GPU.
# """
function _vec2uutri(v::AbstractVector{T}; n=invsumn(length(v)) + one(T), diag=one(T)) where {T}
    z = zero(T)
    k = 0
    m = [j > i ? (k += 1; v[k]) : i == j ? diag : z for i = 1:n, j = 1:n]
    return (m)
end


#function vec2uutri(v::GPUArraysCore.AbstractGPUVector{T}; n=invsumn(length(v)) + one(T)) where {T}
#TODO remove internal conversion to CuArray to generalize to AbstractGPUVector
function vec2uutri(v::CuVector{T}; n=invsumn(length(v)) + 1, diag=one(T)) where {T}
    # _one = one(T)
    # z = zero(T)
    # k = 0
    # m = CUDA.allowscalar() do
    #     CuArray([j > i ? (k += 1; v[k]) : i == j ? _one : z for i = 1:n, j = 1:n])
    # end
    m = CUDA.zeros(T,n,n) 
    m[1:size(m, 1)+1:end] .= diag  # indexing trick for diag(m) .= one(T) 
    @cuda threads = 256 vec2uutri_gpu!(m, v)
    return (m)
end

function vec2uutri_gpu!(m::Union{CuDeviceArray, GPUArraysCore.AbstractGPUMatrix}, v::AbstractVector) 
    index = threadIdx().x    # this example only requires linear indexing, so just use `x`
    stride = blockDim().x
    for i = index:stride:length(v)
        row, col = vec2utri_pos(i)
        # for unit-upper triangle shift the column position by  one
        @inbounds m[row, col+1] = v[i]
    end
    return nothing # important
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
        end for _ in 1:lv
    ]
end



"""
Extract entries of upper diagonal matrix of UnitUppterTriangular to columnwise vector
"""
function uutri2vec(X::AbstractMatrix{T}) where {T}
    n = size(X, 1) - 1
    lv = sumn(n)
    i = 0
    j = 2
    [
        begin
            if i == j - 1
                i = 0
                j += 1
            end
            i += 1
            X[i, j]
        end for _ in 1:lv
    ]
end

function ChainRulesCore.rrule(::typeof(uutri2vec), X::AbstractMatrix{T}) where T
    # does not change the value or the derivative but just reshapes
    # -> put the components of vector Δy into matrix
    # make sure that the gradient of main diagonal is zero rather than one
    function uutri2vec_pullback(Δy) 
        (NoTangent(), vec2uutri(Δy; diag=zero(T)))
    end
    return uutri2vec(X), uutri2vec_pullback
end

#function uutri2vec(X::GPUArraysCore.AbstractGPUMatrix; kwargs...)
#TODO remove internal coersion to CuArray to extend to other AbstractGPUMatrix
# function uutri2vec(X::CuArray; kwargs...)
#     n = size(X, 1) - 1
#     lv = sumn(n)
#     i = 0
#     j = 2
#     CUDA.allowscalar() do
#         CuArray([
#             begin
#                 if i == j - 1
#                     i = 0
#                     j += 1
#                 end
#                 i += 1
#                 X[i, j]
#             end for _ in 1:lv
#         ])
#     end
# end
function uutri2vec(X::CuMatrix{T}; kwargs...) where T
    lv = sumn(size(X,1)-1)
    v = CUDA.zeros(T,lv) 
    @cuda threads=(16, 16) uutri2vec_gpu!(v, X)
    return v
end

function uutri2vec_gpu!(v::Union{CuVector,CuDeviceVector}, X::AbstractMatrix) 
    x = threadIdx().x
    y = threadIdx().y
    stride_x= blockDim().x
    stride_y = blockDim().y
    for row = x:stride_x:size(X,1)
        for col = y:stride_y:size(X,2)
            if row < col
                i = utri2vec_pos_unsafe(row, col-1)
                @inbounds v[i] = X[row, col]
            end        
        end
    end
    return nothing # important
end


"""
Takes a vector of entries of a lower UnitUpperTriangular matrix
and transformes it to an UpperTriangular that satisfies 
diag(U' * U) = 1.

This can be used to fit parameters that yield an upper Cholesky-Factor
of a Covariance matrix.

It uses the upper triangular matrix rather than the lower because it
involes a sum across columns, wheras the alternative of a lower triangular
uses sum across rows. 
Sum across columns is often faster, because entries of columns are contiguous.
"""
function transformU_cholesky1(v::AbstractVector; 
    n=invsumn(length(v)) + 1
    )
    Uscaled = vec2uutri(v; n)
    #Sc_inv = sqrt.(sum(abs2, Uscaled, dims=1))
    #Uscaled * Diagonal(1 ./ vec(Sc_inv))
    #U = Uscaled ./ Sc_inv
    U = Uscaled ./ sqrt.(sum(abs2, Uscaled, dims=1))
    return (UpperTriangular(U))
end
function transformU_cholesky1(v::GPUArraysCore.AbstractGPUVector; n=invsumn(length(v)) + 1)
    Uscaled = vec2uutri(v; n)
    U = Uscaled ./ sqrt.(sum(abs2, Uscaled, dims=1))
    # do not convert to UpperTrinangular on GPU, but full matrix
    #return (UpperTriangular(U))
    return U
end

() -> begin
    tmp = sqrt.(sum(abs2, Uscaled, dims=1))
    tmp2 = sum(abs2, Uscaled, dims=1) .^ (-1 / 2)
    Uscaled * tmp'
end
