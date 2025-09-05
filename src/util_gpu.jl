"""
    ones_similar_x(x::AbstractArray, size_ret = size(x))

Return `ones(eltype(x), size_ret)`.
Overload this methods for specific AbstractGPUArrays to return the 
correct container type. 
See e.g. `HybridVariationalInferenceCUDAExt`
that calls `CUDA.fill` to return a `CuArray` rather than `Array`.
"""
function ones_similar_x(x::AbstractArray, size_ret = size(x))
    #ones(eltype(x), size_ret)
    Ones{eltype(x)}(size_ret)
end

function ones_similar_x(x::GPUArraysCore.AbstractGPUArray, s = size(x)) 
    backend = get_backend(x)
    ans = ChainRulesCore.@ignore_derivatives KernelAbstractions.ones(backend, eltype(x), s)
    # https://juliagpu.github.io/KernelAbstractions.jl/stable/quickstart/#Synchronization
    ChainRulesCore.@ignore_derivatives synchronize(backend)  
    ans
end

# handle containers and transformations of Arrays
ones_similar_x(x::CA.ComponentArray, s = size(x)) = ones_similar_x(CA.getdata(x), s)
ones_similar_x(x::LinearAlgebra.Adjoint, s = size(x)) = ones_similar_x(parent(x), s)
ones_similar_x(x::SubArray, s = size(x)) = ones_similar_x(parent(x), s)





