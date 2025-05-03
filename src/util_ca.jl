"""
    cpu_ca(ca::CA.ComponentArray)

Move ComponentArray form gpu to cpu.    
"""
function cpu_ca end
# define in FluxExt

function apply_preserve_axes(f, ca::CA.ComponentArray)
    CA.ComponentArray(f(CA.getdata(ca)), CA.getaxes(ca))
end

"""
    combine_axes(axtuples::NamedTuple)

Create a new 1d-axis that combines several other named axes-tuples
such as of `key = getaxes(::AbstractComponentArray)`.

The new axis consists of several ViewAxes. If an axis-tuple consists only of one axis, it is used for the view.
Otherwise a ShapedAxis is created wiht the axes-length of the others, essentially dropping
component information that might be present in the dimensions.
"""
function combine_axes(axtuples::NamedTuple)
    ls = map(axtuple -> Val(prod(axis_length.(axtuple))), axtuples)
    # to work on types, need to construct value types of intervals
    intervals = _construct_invervals(;lengths=ls)
    named_intervals = (;zip(keys(axtuples),_val_value(intervals))...)
    axc = map(named_intervals, axtuples) do interval, axtuple
        ax = length(axtuple) == 1 ? axtuple[1] : CA.ShapedAxis(axis_length.(axtuple))
        CA.ViewAxis(interval, ax)
    end
    CA.Axis(; axc...)
end

axis_length(ax::CA.AbstractAxis) = CA.lastindex(ax) - CA.firstindex(ax) + 1
axis_length(::CA.FlatAxis) = 0
axis_length(::CA.UnitRange) = 0


