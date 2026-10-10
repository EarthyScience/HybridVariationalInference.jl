"""
    VcatCMs(templates::Tuple, ms...)

A non-allocating single Val(symbol) element view type into a series of AbstractMatrices,
including reshpaped vies.

```julia
# the matracies to vcat
n_MC = 4; n_θP = 2; n_θM = 3
pm = rand(n_θP,n_MC)
mm = rand(n_θM,n_MC)
fm = rand(0,n_MC)
#
# templates for access into first dimension
pt = CA.ComponentVector(NamedTuple(Symbol("p"*string(i)) => i for i in 1:n_θP))
mt = CA.ComponentVector(NamedTuple(Symbol("m"*string(i)) => i for i in 1:n_θM))
ft = CA.ComponentVector{eltype(pt)}()
#
# the Vcat object
ccat = HVI.VcatCMs((pt,mt,ft), pm, mm, fm)
HVI.viewindex(ccat, Val(:p2)) == pm[2,:]
HVI.viewindex(ccat, Val(:m3)) == mm[2,:]
```
"""
struct VcatCMs{AX,T}
    axs::Val{AX}
    cms::T
end
function VcatCMs(templates::Tuple{Vararg{<:CA.ComponentVector}}, ms...) 
    @assert length(templates) == length(ms)
    axs = map(ms, templates) do m, cv # allocates, better call directly with axes
        ax = CA.getaxes(cv)[1]
        @assert size(m,1) == axis_length(ax)
        ax
    end
    VcatCMs(axs, ms...)
end
@inline function VcatCMs(axs::Tuple{Vararg{<:CA.Axis}}, ms...)
    @assert length(axs) == length(ms)
    cms = map(ms, axs) do m,ax
        @assert size(m,1) == axis_length(ax)
        CA.ComponentMatrix(m, ax, CA.FlatAxis())
    end
    VcatCMs(Val(axs), cms)
end

Base.size(ccat::VcatCMs) = (mapreduce(x -> size(x,1), +, ccat.cms), size(ccat.cms[1],2))
Base.eltype(ccat::VcatCMs) = eltype(ccat.cms[1])

Base.view(ccat::VcatCMs, valsym::Val, cols) = viewindex(ccat, valsym, cols) 

# @inline function viewindex(ccat::VcatCMs{AX}, val::Val{sym}) where {AX,sym}
#     local i = 1
#     while i <= length(AX)
#         sym ∈ keys(AX[i]) && return view(ccat.cms[i], val, :)
#         i += 1
#     end
#     error(string(sym) * " not a key of VcatCMs object")
# end
# avoid the loop because i is runtime value and AX[i] not compile time, use recursion

@inline function viewindex(ccat::VcatCMs{AX}, ::Val{sym}, cols=Colon()) where {AX,sym}
    _vi(ccat, Val(sym), AX, ccat.cms, cols)
end
@inline function _vi(ccat, ::Val{sym}, axs::Tuple{Head,Vararg{Any,N}}, cms::Tuple{CM,Vararg{Any,N}}, cols) where {sym,Head,CM,N}
    if sym ∈ keys(axs[1])
        cols isa Integer ? first(cms)[Val(sym), cols] : view(first(cms), Val(sym), cols)
    else
        _vi(ccat, Val(sym), Base.tail(axs), Base.tail(cms), cols)
    end
end
@inline _vi(ccat, ::Val{sym}, axs::Tuple{}, cms::Tuple{}, cols) where {sym} = error(
    string(sym) * " not a key of VcatCMs object")