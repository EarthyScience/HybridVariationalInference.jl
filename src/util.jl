"""
    vectuptotupvec(vectup)
    vectuptotupvec_allowmissing(vectup)

Typesafe convert from Vector of Tuples to Tuple of Vectors.
The first variant does not allow for missings in `vectup`.
The second variant allows for missing but has `eltype` of `Union{Missing, ...}` in  
all components of the returned Tuple, also when there were not missings in `vectup`. 

# Arguments
* `vectup`: A Vector of identical Tuples 

# Examples
```jldoctest; output=false, setup = :(using Distributions)
vectup = [(1,1.01, "string 1"), (2,2.02, "string 2")] 
vectuptotupvec_allowmissing(vectup) == ([1, 2], [1.01, 2.02], ["string 1", "string 2"])
# output
true
```
"""
function vectuptotupvec(vectup::AbstractVector{<:Tuple}) 
    Ti = eltype(vectup).parameters
    npar = length(Ti)
    ntuple(i -> 
        (getindex.(vectup, i))::Vector{Ti[i]}, npar)
end
function vectuptotupvec_allowmissing(
    vectup::AbstractVector{<:Union{Missing,Tuple}}) 
    Ti = nonmissingtype(eltype(vectup)).parameters
    npar = length(Ti)
    Tim = ntuple(i -> Union{Missing,Ti[i]}, npar)
    ntuple(i -> begin
        allowmissing(passmissing(getindex).(vectup, i))::Vector{Tim[i]}
    end, npar)
end
