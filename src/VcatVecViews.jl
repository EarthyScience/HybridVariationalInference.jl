"""
    VcatVecViews(A::AbstractArray{T}...) where T 

Combines to combine several heterogeneously shaped subarrays, `A`, into a
`LazyArrays.Vcat` view that behaves like an `AbstractVector{T}`.
It does not copy the subarrays and writing into it will modify the subarrays, `A`.
Access the original non-reshaped subarrays using function [`subarray`](@ref).

There is a small allocation when constructing the `Vcat`, an index computation
when accessing elements, but no computation when indexing into `subarray(vcc, i)`.

Note that `similar(vcc)` or `copy(vcc)` of a `vcc::VcatVecViews` 
only return a Vcat in which the vector-views of the subarrays are replaced by 
vectors.

It allows efficient ForwardDiff.gradient computation of a function to several 
array arguments of heterogeneous shapes by updating the duals component of
a `ForwardDiff.GradientConfig` by `update_duals`.

```julia
    #import ForwardDiff
    v = collect(1:4)
    A = rand(1:10, 3,3)
    Avv = VcatVecViews(A,v)
    compAv(A,v) = 2*sum(A) + 3*sum(v) # the hereogeneous computation
    compAvv(Avv) = compAv(subarray(Avv,Val(1)), subarray(Avv,Val(2))) # wrapper
    cfg = ForwardDiff.GradientConfig(compAvv, Avv, ForwardDiff.Chunk(8))
    # replace the duals-Component of the GradientConfig by a proper VcatVecViews
    cfgv = HVI.update_duals(cfg, Avv)
    # can also supply a VcatVecViews mutated destination for the gradient
    #   note that similar or copy of a VcatVecViews only return a Vcat of Vector
    gr = VcatVecViews(A,v) 
    ForwardDiff.gradient!(gr, compAvv, Avv, cfgv)
    @test all(subarray(gr,Val(1)) .== 2)
    @test all(subarray(gr,Val(2)) .== 3)
```
"""
const VcatVecViews{T,N,I} = Vcat{T,N,I} # its a Vcat across views

@inline function VcatVecViews(A::AbstractArray{T}...) where T 
    # if used vec(Ai) then the parent of the resulting object would be wrongly shaped 1D
    Vcat{T}((vecview(Ai) for Ai in A)...) # ApplyVector allocates
end

@inline vecview(v::AbstractVector) = v
@inline function vecview(A::AbstractArray) 
    @assert strides(A)[1] == 1
    Base.ReshapedArray(A, (length(A), 1), ())
end

"""
    subarray(vv::VcatVecViews, Val{i])
    subarray_uninferred(vv::VcatVecViews, i)

Access the i'th subarray from which the `vv::VcatVecViews` was built. 

Only provide literals or compile-time inferred integers as Valuetypes to the index, 
to make of the result type inferred by the compiler.
"""
@inline subarray_uninferred(vv::VcatVecViews, i::Integer) = parent(vv.args[i])
@inline subarray(vv::VcatVecViews, ::Val{i}) where i = parent(vv.args[i])

function update_duals(
    cfg::ForwardDiff.GradientConfig{C1,C2,C3}, 
    vv::VcatVecViews{T,N,I},
    ) where {T,N,I,C1,C2,C3}
    @assert size(vv) == size(cfg.duals)
    duals_vcat = VcatVecViews(
        (similar(parent(A), eltype(cfg.duals)) for A in vv.args)...)
    cfgv = ForwardDiff.GradientConfig{C1,C2,C3,typeof(duals_vcat)}(cfg.seeds, duals_vcat)
    cfgv
end
