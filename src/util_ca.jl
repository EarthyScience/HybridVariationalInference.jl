"""
    cpu_ca(ca::CA.ComponentArray)

Move ComponentArray form gpu to cpu.    
"""
function cpu_ca end
# define in FluxExt

function apply_preserve_axes(f, ca::CA.ComponentArray)
    CA.ComponentArray(f(CA.getdata(ca)), CA.getaxes(ca))
end


