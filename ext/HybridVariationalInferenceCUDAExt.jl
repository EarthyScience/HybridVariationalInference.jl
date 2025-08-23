module HybridVariationalInferenceCUDAExt

using HybridVariationalInference, CUDA
using HybridVariationalInference: HybridVariationalInference as HVI
using ComponentArrays: ComponentArrays as CA
using ChainRulesCore

# here, really CUDA-specific implementation, in case need to code other GPU devices
function HVI.vec2utri(v::CUDA.CuVector{T}; n=invsumn(length(v)) ) where {T}
    z = zero(T)
    k = 0
    m = CUDA.allowscalar() do
        CUDA.CuArray([j >= i ? (k += 1; v[k]) : z for i in 1:n, j in 1:n])
    end
    # m = CUDA.zeros(T,n,n)
    # @cuda threads = 256 vec2utri_gpu!(m, v) # planned to put v into positions of m
    return (m)
end

function vec2utri_gpu!(m, v::AbstractVector)
    index = threadIdx().x    # this example only requires linear indexing, so just use `x`
    stride = blockDim().x
    for i in index:stride:length(v)
        row, col = vec2utri_pos(i)
        @inbounds m[row, col] = v[i]
    end
    return nothing # important
end


#function vec2uutri(v::GPUArraysCore.AbstractGPUVector{T}; n=invsumn(length(v)) + one(T)) where {T}
function HVI.vec2uutri(v::CUDA.CuVector{T}; n=HVI.invsumn(length(v)) + 1, diag=one(T)) where {T}
    # _one = one(T)
    # z = zero(T)
    # k = 0
    # m = CUDA.allowscalar() do
    #     CUDA.CuVector([j > i ? (k += 1; v[k]) : i == j ? _one : z for i = 1:n, j = 1:n])
    # end
    m = CUDA.zeros(T, n, n)
    m[1:(size(m, 1)+1):end] .= diag  # indexing trick for diag(m) .= one(T) 
    CUDA.@cuda threads = 256 vec2uutri_gpu!(m, v)
    return (m)
end

function vec2uutri_gpu!(
    m::Union{CUDA.CuDeviceMatrix, CUDA.CuMatrix}, v::AbstractVector)
    #m::Union{CUDA.CuDeviceArray,GPUArraysCore.AbstractGPUMatrix}, v::AbstractVector)
    index = threadIdx().x    # this example only requires linear indexing, so just use `x`
    stride = blockDim().x
    for i in index:stride:length(v)
        row, col = HVI.vec2utri_pos(i)
        # for unit-upper triangle shift the column position by  one
        @inbounds m[row, col+1] = v[i]
    end
    return nothing # important
end

function HVI.uutri2vec(X::CUDA.CuMatrix{T}; kwargs...) where {T}
    lv = HVI.sumn(size(X, 1) - 1)
    v = CUDA.zeros(T, lv)
    CUDA.@cuda threads = (16, 16) uutri2vec_gpu!(v, X)
    return v
end

function uutri2vec_gpu!(v::Union{CUDA.CuVector,CUDA.CuDeviceVector}, X::AbstractMatrix)
    x = threadIdx().x
    y = threadIdx().y
    stride_x = blockDim().x
    stride_y = blockDim().y
    for row in x:stride_x:size(X, 1)
        for col in y:stride_y:size(X, 2)
            if row < col
                i = HVI.utri2vec_pos_unsafe(row, col - 1)
                @inbounds v[i] = X[row, col]
            end
        end
    end
    return nothing # important
end

function HVI._create_randn(rng, v::CUDA.CuVector{T,M}, dims...) where {T,M}
    # ignores rng
    # https://discourse.julialang.org/t/help-using-cuda-zygote-and-random-numbers/123458/4?u=bgctw
    res = ChainRulesCore.@ignore_derivatives CUDA.randn(dims...)
    res::CUDA.CuArray{T, length(dims),M}
end

function HVI.ones_similar_x(x::CuArray, size_ret = size(x))
    # call CUDA.ones rather than ones for x::CuArray
    ChainRulesCore.@ignore_derivatives CUDA.ones(eltype(x), size_ret)
end







end # module
