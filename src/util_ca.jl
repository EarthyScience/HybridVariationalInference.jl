"""
    cpu_ca(ca::CA.ComponentArray)

Move ComponentArray form gpu to cpu.    
"""
function cpu_ca(ca::CA.ComponentArray)
    CA.ComponentArray(cpu_device()(CA.getdata(ca)), CA.getaxes(ca))
end

"""
    apply_preserve_axes(f, ca::ComponentArray)

Apply callable `f(x)` to the data inside `ca`, assume that the result has
the same shape, and return a new `ComponentArray` with the same axes
as in `ca`.
"""
function apply_preserve_axes(f, ca::CA.ComponentArray)
    CA.ComponentArray(f(CA.getdata(ca)), CA.getaxes(ca))
end
# special case of empty Sub-ComponentVector 
# apply_preserve_axes(identity, CA.ComponentVector(a=1, b=CA.ComponentVector()).b)
function apply_preserve_axes(f, ca::AbstractArray)
    @assert isempty(ca)
    CA.ComponentVector()
end

"""
    compose_axes(axtuples::NamedTuple)

Create a new 1d-axis that combines several other named axes-tuples
such as of `key = getaxes(::AbstractComponentArray)`.

The new axis consists of several ViewAxes. If an axis-tuple consists only of one axis, it is used for the view.
Otherwise a ShapedAxis is created with the axes-length of the others, essentially dropping
component information that might be present in the dimensions.
"""
function compose_axes(axtuples::NamedTuple)
    ls = map(axtuple -> Val(prod(axis_length.(axtuple))), axtuples)
    # to work on types, need to construct value types of intervals
    intervals = _construct_intervals(;lengths=ls)
    named_intervals = (;zip(keys(axtuples),intervals)...)
    axc = map(named_intervals, axtuples) do interval, axtuple
        ax = length(axtuple) == 1 ? axtuple[1] : CA.ShapedAxis(axis_length.(axtuple))
        CA.ViewAxis(_val_value(interval), ax)
    end
    CA.Axis(; axc...)
end

function _construct_intervals(;lengths) 
    reduce((ranges,length) -> _add_interval(;ranges, length), 
        Iterators.tail(lengths), init=(Val(1:_val_value(first(lengths))),))    
end
function _add_interval(;ranges, length::Val{l}) where {l}
    ind_before = last(_val_value(last(ranges)))
    (ranges...,Val(ind_before .+ (1:l)))
end
_val_value(::Val{x}) where x = x


axis_length(ax::CA.AbstractAxis) = CA.lastindex(ax) - CA.firstindex(ax) + 1
axis_length(::CA.FlatAxis) = 0
axis_length(ax::CA.UnitRange) = length(ax)
axis_length(ax::CA.ShapedAxis) = length(ax)
axis_length(ax::CA.Shaped1DAxis) = length(ax)

"""
    as_data_frame(cm::CA.ComponentMatrix) 
    as_data_frame(cm::CA.ComponentArray{T,3}) 
    as_data_frame(cm::CA.ComponentArray{T,4}) 

Converts a ComponentMatrix with scalar keys in first or second dimension to a DataFrame.
If keys are in first column, the result corresponds to transposing the first
two dimensions. 
With arrays of higher dimension, columns dim3 and dim4 are added that report
the index in this dimension.
"""
function as_data_frame end
# in ext/HybridVariationalInferenceDataFramesExt.jl to avoid DataFrames dependency 


"""
Extends CA.static_getproperty(cv, Val(:b)) and @static_unpack.
But also converts a component that is itself a ComponentVector to a StaticVector
"""
function static_cv_getproperty(cv::CA.ComponentVector, key::Val) 
    local v = view(cv, key)    
    if v isa CA.ComponentVector
        #SA.SVector{axis_length(CA.getaxes(v)[1])}(CA.getdata(v))
        SA.SVector{axis_length(CA.getaxes(v)[1])}(v)
        # attaching axis -> downstream errors "Dimension is not static. Please file a bug."
        # CA.ComponentVector(
        #     SA.SVector{axis_length(CA.getaxes(v)[1])}(v),
        #     CA.getaxes(v)
        # )
    else
        CA.static_getproperty(cv, key)
    end
end

"""
    copyto_nested!(dest::NamedTuple, src::ComponentVector) -> NamedTuple

Copy data from a `ComponentVector` into a preallocated `NamedTuple`, recursively
handling nested structures.

# Arguments
- `dest::NamedTuple`: The destination `NamedTuple` with preallocated arrays or
  nested `NamedTuple`s. The structure must match `src` exactly.
- `src::ComponentVector`: The source `ComponentVector` containing the data to copy.

# Returns
The modified `dest` `NamedTuple` (same object, mutated in-place).

# Details
This function recursively traverses both `dest` and `src` simultaneously:
- If a field in `dest` is a `NamedTuple` and the corresponding field in `src`
  is a `ComponentVector`, the function recurses into both.
- Otherwise, it copies the data from the view of `src` into the array in `dest`
  using `copyto!`.

This is useful for zero-allocation copying of `ComponentVector` data into a
preallocated `NamedTuple` structure, which can be beneficial in performance-
critical code or when interfacing with functions that expect `NamedTuple`s.

# Example
```julia
using ComponentArrays

dest = (x = zeros(3), params = (a = zeros(2), b = zeros(4)));
src = ComponentVector(dest); randn!(src)
copyto_nested!(dest, src)
```
""" 
function copyto_nested!(dest::NamedTuple, src::CA.ComponentVector)
    for (k, v) in pairs(dest)
        if v isa NamedTuple
            # Recursively handle nested NamedTuples
            sub_src = getproperty(src, k)
            copyto_nested!(v, sub_src)
        else
            # Copy the view for leaf components
            copyto!(v, view(src, k))
        end
    end
    return dest
end

