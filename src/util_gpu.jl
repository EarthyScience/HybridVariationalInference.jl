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

# TODO replace CUDA-method in extension after type stabilie is fixed
# https://github.com/JuliaGPU/KernelAbstractions.jl/issues/634
# function ones_similar_x(x::GPUArraysCore.AbstractGPUArray, s = size(x)) 
#     backend = get_backend(x)
#     ans = KernelAbstractions.ones(backend, eltype(x), s)
#     #ans = ChainRulesCore.@ignore_derivatives KernelAbstractions.ones(backend, eltype(x), s)
#     # https://juliagpu.github.io/KernelAbstractions.jl/stable/quickstart/#Synchronization
#     # ChainRulesCore.@ignore_derivatives synchronize(backend)  
#     ans
# end

# handle containers and transformations of Arrays
ones_similar_x(x::CA.ComponentArray, s = size(x)) = ones_similar_x(CA.getdata(x), s)
ones_similar_x(x::LinearAlgebra.Adjoint, s = size(x)) = ones_similar_x(parent(x), s)
ones_similar_x(x::SubArray, s = size(x)) = ones_similar_x(parent(x), s)

function repeat_rowvector_dummy(x::AbstractMatrix{T}, is_dummy::Union{BitVector,AbstractVector{Bool}}; 
    ones_vec = ones_similar_x(x, length(is_dummy)),
    ) where T
    ones_vec .* x .+ (is_dummy .* convert(T,NaN))
    #ones_vec .* x .+ (is_dummy)
end

function ChainRulesCore.rrule(::typeof(repeat_rowvector_dummy), x, is_dummy::Union{BitVector,AbstractVector{Bool}}; kwargs...)
    function repeat_rowvector_dummy_pullback(Δy)
        # only sum the partials across non-dummy rows for each column
        Δx = sum(Δy[.! is_dummy,:]; dims=1)
        (NoTangent(), Δx, NoTangent())
    end
    return repeat_rowvector_dummy(x, is_dummy; kwargs...), repeat_rowvector_dummy_pullback
end

function repeat_rowvector_dummy(x::AbstractMatrix{T}, is_dummy::Union{BitMatrix,AbstractMatrix{Bool}}; 
    ) where T
    x .+ (is_dummy .* convert(T,NaN))
end

function ChainRulesCore.rrule(::typeof(repeat_rowvector_dummy), 
    x, is_dummy::Union{BitMatrix,AbstractMatrix{Bool}}; kwargs...)
    pullback = if !any(is_dummy)
        # avoid mapping rows if there is no dummy
        function repeat_rowvector_dummy_pullback_emptybitmatrix(Δy)
            Δx = sum(Δy; dims=1)
            #Main.@infiltrate_main
            (NoTangent(), Δx, ZeroTangent())            
        end
    else
        function repeat_rowvector_dummy_pullback_bitmatrix(Δy)
            #@info "called rrule for repeat_rowvector_dummy with is_dummy matrix"
            # only sum the partials across non-dummy rows for each column
            #Δx = similar(x)  # errors in copyto! need the same as xtvec
            Δxt = similar(x')
            # TODO think of avoiding allocation of temporary vector
            # using generator or StaticArray results in scalar indexing
            xtvec = map(eachcol(Δy[:,:]), eachcol(is_dummy)) do Δyi, is_dummi_i
                sum(Δyi[.! is_dummi_i])
            end 
            # gen = (sum(Δyi[.! is_dummi_i]) for (Δyi, is_dummi_i) in zip(eachcol(Δy[:,:]), eachcol(is_dummy)))
            #xtvec = @SArray[sum(Δyi[.! is_dummi_i]) for (Δyi, is_dummi_i) in zip(eachcol(Δy[:,:]), eachcol(is_dummy))]
            # if !all(isfinite.(xtvec))
            #     @info "repeat_rowvector_dummy_pullback_bitmatrix: encountered non-finite gradients"
            # end
            copyto!(Δxt, xtvec) 
            (NoTangent(), Δxt', ZeroTangent())
        end
    end
    return repeat_rowvector_dummy(x, is_dummy; kwargs...), pullback
end






