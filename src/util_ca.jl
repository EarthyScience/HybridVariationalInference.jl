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


