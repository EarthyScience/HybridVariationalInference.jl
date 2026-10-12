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
"""
struct VcatCAs{AX,N, T}
    axs::Val{AX}
    cms::T
    ndims::Val{N}
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
#TODO make it work with ms of different eltypes (θFix is not dual)
@inline function VcatCAs(axs::Tuple{Vararg{<:CA.Axis}}, 
    ms::Vararg{<:AbstractArray{<:Any,N}}) where {N}
    @assert length(axs) == length(ms)
    flataxes = ntuple(i -> CA.FlatAxis(), N-1)
    cms = map(ms, axs) do m,ax
        @assert size(m,1) == axis_length(ax)
        CA.ComponentArray(CA.getdata(m), ax, flataxes...)
    end
    VcatCAs(Val(axs), cms, Val(N))
end

function Base.size(ccat::VcatCAs{AXS,N}) where {AXS,N} 
    (mapreduce(x -> size(x,1), +, ccat.cms), size(ccat.cms[1])[2:N]...)
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

@inline function viewindex(ccat::VcatCAs{AX}, ::Val{sym}, cols...) where {AX,sym}
    _vi(ccat, Val(sym), AX, ccat.cms, cols...)
end
@inline function _vi(ccat, ::Val{sym}, axs::Tuple{Head,Vararg{Any,N}}, cms::Tuple{CM,Vararg{Any,N}}, cols...) where {sym,Head,CM,N}
    if sym ∈ keys(axs[1])
        # indexing with cols into array allocates, indexing into the view does not
        # cols isa Integer ? view(first(cms), Val(sym), :)[cols] : view(view(first(cms), Val(sym), :), cols)
        ND = ndims(first(cms))
        local colons, v
        colons = (Colon() for i in 2:ND)
        v = view(first(cms), Val(sym), colons...)
        if cols == ()
            v
        elseif cols isa (NTuple{N, <:Int} where N) # all scalars indices -> scalar value
            v[cols...]
        else 
            view(v, cols...)
        end
    else
        _vi(ccat, Val(sym), Base.tail(axs), Base.tail(cms), cols...)
    end
end
@inline _vi(ccat, ::Val{sym}, axs::Tuple{}, cms::Tuple{}, cols...) where {sym} = error(
    string(sym) * " is not a key of VcatCAs object")

@inline function Base.getindex(ccat::VcatCAs{AX}, ::Val{sym}, cols...) where {AX,sym}
    _gi(ccat, Val(sym), AX, ccat.cms, cols...)
end

@inline function _gi(ccat, ::Val{sym}, axs::Tuple{Head,Vararg{Any,N}}, cms::Tuple{CM,Vararg{Any,N}}, cols...) where {sym,Head,CM,N}
    if sym ∈ keys(axs[1])
        ND = ndims(first(cms))
        colons = (Colon() for i in 2:ND)
        getindex(first(cms), Val(sym), cols...)
    else
        _gi(ccat, Val(sym), Base.tail(axs), Base.tail(cms), cols...)
    end
end
@inline _gi(ccat, ::Val{sym}, axs::Tuple{}, cms::Tuple{}, cols...) where {sym} = error(
    string(sym) * " is not a key of VcatCAs object")


struct VcatCAEachCol{AX,T}
    ccat::VcatCAs{AX,2,T}
end

Base.IteratorSize(::Type{<:VcatCAEachCol}) = Base.HasLength()
Base.length(ec::VcatCAEachCol) = size(ec.ccat)[2]
Base.eltype(ec::VcatCAEachCol) =
    VcatCAs{typeof(ec.ccat.axs),1,typeof(map(cm -> view(cm, :, 1), ec.ccat.cms))}
Base.lastindex(ec::VcatCAEachCol) = _col(ec.ccat, length(ec))

@inline function Base.iterate(ec::VcatCAEachCol, j::Int = 1)
    j > length(ec) && return nothing
    return (_col(ec.ccat, j), j + 1)
end

@inline function _col(ccat::VcatCAs{AX,2}, j::Int) where {AX}
    cols = map(cm -> CA.getdata(view(cm, :, j)), ccat.cms)
    VcatCAs(AX, cols...)
end

Base.eachcol(ccat::VcatCAs{AX,2}) where {AX} = VcatCAEachCol(ccat)
