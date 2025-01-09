"""
Apply TransformVariables.inverse to ComponentArray, `ca`.

- convert `ca` to a `NamedTuple`
- apply transformation
- convert back to `ComponentArray`
"""
function inverse_ca(trans, ca::CA.AbstractArray)
    CA.ComponentArray(
        TransformVariables.inverse(trans, cv2NamedTuple(ca)),
        CA.getaxes(ca))
end

"""
Convert ComponentVector to NamedTuple of the first layer, i.e. keep 
ComponentVectors in the second level.
"""
function cv2NamedTuple(ca::CA.ComponentVector)
    g = ((k, CA.getdata(ca[k])) for k in keys(ca))
    (; g...)
end