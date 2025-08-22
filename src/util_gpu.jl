"""
    ones_similar_x(x::AbstractArray, size_ret = size(x))

Return `ones(eltype(x), size_ret)`.
Overload this methods for specific AbstractGPUArrays to return the 
correct container type. 
See e.g. `HybridVariationalInferenceCUDAExt`
that calls `CUDA.fill` to return a `CuArray` rather than `Array`.
"""
function ones_similar_x(x::AbstractArray, size_ret = size(x))
    ones(eltype(x), size_ret)
end

# handle containers and transformations of Arrays
ones_similar_x(x::CA.ComponentArray, args...) = ones_similar_x(CA.getdata(x), args...)
ones_similar_x(x::LinearAlgebra.Adjoint, args...) = ones_similar_x(parent(x), args...)




