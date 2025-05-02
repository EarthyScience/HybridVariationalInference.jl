"""
    cpu_ca(ca::CA.ComponentArray)

Move ComponentArray form gpu to cpu.    
"""
function cpu_ca end
# define in FluxExt

function apply_preserve_axes(f, ca::CA.ComponentArray)
    CA.ComponentArray(f(CA.getdata(ca)), CA.getaxes(ca))
end

axis_length(ax::CA.AbstractAxis) = CA.lastindex(ax) - CA.firstindex(ax) + 1
axis_length(::CA.FlatAxis) = 0
axis_length(::CA.UnitRange) = 0

