"""
    VcatCAs(templates::Tuple, ms...)

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
ccat = HVI.VcatCAs((pt,mt,ft), pm, mm, fm)
HVI.viewindex(ccat, Val(:p2)) == pm[2,:]
HVI.viewindex(ccat, Val(:m3)) == mm[2,:]
```

Type parameters
- `AX`: Tuple of Axes for the first dim of the underlying arrays
- `NC`: dimensionality of the underlying arrys minus 1 (dimensions in addition to named one), 
- `SM`: NamedTuple: mapping symbol to number of underlying array that holds the entry, 
- `T`: types of the underlying arrays (to make type concrete)
"""
struct VcatCAs{AX, NC, SM, T}
    axs::Val{AX}
    cms::T
    colons::NTuple{NC, Colon}
    sym_map::Val{SM}
end
function VcatCAs(templates::Tuple{Vararg{<:CA.ComponentVector}}, ms...) 
    @assert length(templates) == length(ms)
    axs = map(ms, templates) do m, cv # allocates, better call directly with axes
        ax = CA.getaxes(cv)[1]
        @assert size(m,1) == axis_length(ax)
        ax
    end
    VcatCAs(axs, ms...)
end
@inline function VcatCAs(axs::Tuple{Vararg{<:CA.Axis}}, 
    ms::Vararg{<:AbstractArray{<:Any,N},NM}) where {N, NM}
    sym_map_t = ntuple(NM) do i
        ax = axs[i]
        NamedTuple{keys(ax)}(ntuple(_ -> i, length(keys(ax))))
    end
    sym_map = concat_ntuples(sym_map_t)
    # provide constructor where can provide sym_map already (for eachcol)
    VcatCAs(Val(axs), Val(sym_map), ms)
end
@inline function VcatCAs(axs_val::Val{AXS}, sym_map_val::Val{SM},
    ms::Tuple{Vararg{<:AbstractArray{<:Any,N},NM}}) where {AXS, N, NM, SM}
    @assert length(AXS) == NM
    @assert length(SM) == sum(map(x -> length(keys(x)), AXS))
    flataxes = ntuple(i -> CA.FlatAxis(), N-1)
    cms = map(ms, AXS) do m,ax
        @assert size(m,1) == axis_length(ax)
        CA.ComponentArray(CA.getdata(m), ax, flataxes...)
    end
    colons = ntuple(_->Colon(), N-1)
    VcatCAs(axs_val, cms, colons, sym_map_val)
end


function Base.size(ccat::VcatCAs{AXS,NC}) where {AXS, NC} 
    (mapreduce(x -> size(x,1), +, ccat.cms), size(ccat.cms[1])[2:(NC+1)]...)
end
Base.eltype(ccat::VcatCAs) = eltype(ccat.cms[1])

Base.view(ccat::VcatCAs, valsym::Val, cols...) = viewindex(ccat, valsym, cols...) 

# @inline function viewindex(ccat::VcatCAs{AX}, val::Val{sym}) where {AX,sym}
#     local i = 1
#     while i <= length(AX)
#         sym ∈ keys(AX[i]) && return view(ccat.cms[i], val, :)
#         i += 1
#     end
#     error(string(sym) * " not a key of VcatCAs object")
# end
# avoid the loop because i is runtime value and AX[i] not compile time, use recursion

@inline function viewindex(ccat::VcatCAs{AX,NC,SM}, ::Val{sym}, cols...) where {AX,NC,SM,sym}
    #_vi(ccat, Val(sym), AX, ccat.cms, cols...)
    local cm, v
    cm = ccat.cms[SM[sym]]
    v = view(cm, Val(sym), ccat.colons...)
    if cols == ()
        v
    elseif cols isa (NTuple{NT, <:Int} where NT) # all scalars indices -> scalar value
        v[cols...]
    else 
        view(v, cols...)
    end
end

# @inline function _vi(ccat, ::Val{sym}, axs::Tuple{Head,Vararg{Any,N}}, cms::Tuple{CM,Vararg{Any,N}}, cols...) where {sym,Head,CM,N}
#     if sym ∈ keys(axs[1])
#         # indexing with cols into array allocates, indexing into the view does not
#         # cols isa Integer ? view(first(cms), Val(sym), :)[cols] : view(view(first(cms), Val(sym), :), cols)
#         ND = ndims(first(cms))
#         local colons, v
#         #colons = (Colon() for i in 2:ND)
#         #Main.@infiltrate_main
#         #colons = ntuple(_->Colon(), ND-1)
#         v = view(first(cms), Val(sym), ccat.colons...)
#         if cols == ()
#             v
#         elseif cols isa (NTuple{N, <:Int} where N) # all scalars indices -> scalar value
#             v[cols...]
#         else 
#             view(v, cols...)
#         end
#     else
#         _vi(ccat, Val(sym), Base.tail(axs), Base.tail(cms), cols...)
#     end
# end
# @inline _vi(ccat, ::Val{sym}, axs::Tuple{}, cms::Tuple{}, cols...) where {sym} = error(
#     string(sym) * " is not a key of VcatCAs object")

@inline function Base.getindex(ccat::VcatCAs{AX,NC,SM}, ::Val{sym}, cols...) where {AX,NC,SM,sym}
    #_gi(ccat, Val(sym), AX, ccat.cms, cols...)
    local cm, v
    cm = ccat.cms[SM[sym]]
    getindex(cm, Val(sym), cols...)
end

# @inline function _gi(ccat, ::Val{sym}, axs::Tuple{Head,Vararg{Any,N}}, cms::Tuple{CM,Vararg{Any,N}}, cols...) where {sym,Head,CM,N}
#     if sym ∈ keys(axs[1])
#         ND = ndims(first(cms))
#         colons = (Colon() for i in 2:ND)
#         getindex(first(cms), Val(sym), cols...)
#     else
#         _gi(ccat, Val(sym), Base.tail(axs), Base.tail(cms), cols...)
#     end
# end
# @inline _gi(ccat, ::Val{sym}, axs::Tuple{}, cms::Tuple{}, cols...) where {sym} = error(
#     string(sym) * " is not a key of VcatCAs object")

Base.eachcol(ccat::VcatCAs{AX,2-1}) where {AX} = VcatCAEachCol(ccat)



struct VcatCAEachCol{AX,T}
    ccat::VcatCAs{AX,2-1,T}
end

Base.IteratorSize(::Type{<:VcatCAEachCol}) = Base.HasLength()
Base.length(ec::VcatCAEachCol) = size(ec.ccat)[2]
Base.eltype(ec::VcatCAEachCol) =
    VcatCAs{typeof(ec.ccat.axs),1,typeof(map(cm -> view(cm, :, 1), ec.ccat.cms))}

@inline function Base.iterate(ec::VcatCAEachCol, j::Int = 1)
    j > length(ec) && return nothing
    return (_col(ec.ccat, j), j + 1)
end

@inline function _col(ccat::VcatCAs{AX,2-1,SM}, j::Int) where {AX, SM}
    cols = map(cm -> view(CA.getdata(cm), :, j), ccat.cms)
    VcatCAs(Val(AX), Val(SM), cols)
end

