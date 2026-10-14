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
struct VcatCAs{AX, NC, T}
    axs::Val{AX}
    cms::T
    colons::NTuple{NC, Colon}
    flataxes::NTuple{NC, CA.FlatAxis}
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
    flataxes = ntuple(i -> CA.FlatAxis(), N-1)
    map(ms, axs) do m,ax
        @assert size(m,1) == axis_length(ax)
    end
    colons = ntuple(_->Colon(), N-1)
    VcatCAs(Val(axs), ms, colons, flataxes)
end


function Base.size(ccat::VcatCAs{AXS,NC}) where {AXS, NC} 
    (mapreduce(x -> size(x,1), +, ccat.cms), size(ccat.cms[1])[2:(NC+1)]...)
end
Base.eltype(ccat::VcatCAs) = eltype(ccat.cms[1])

Base.view(ccat::VcatCAs, valsym::Val, cols...) = viewindex(ccat, valsym, cols...) 


@inline function viewindex(ccat::VcatCAs{AX}, sym_val::Val{sym}, cols...) where {AX, sym}
    #_vi(ccat, Val(sym), AX, ccat.cms, cols...)
    i_val = findfirst_keyed(AX, sym_val)
    viewindex(ccat, i_val, sym_val, cols)
end
@inline function viewindex(ccat::VcatCAs{AX,NC}, ::Val{I}, ::Val{sym}, cols::Tuple) where {I, sym, AX, NC}
    local cm, v
    cm = CA.ComponentArray(ccat.cms[I], AX[I], ccat.flataxes...)
    v = view(cm, Val(sym), ccat.colons...)
    if cols == ()
        v
    elseif cols isa (NTuple{NT, <:Int} where NT) # all scalars indices -> scalar value
        v[cols...]
    else 
        view(v, cols...)
    end
end

"""
static compile-time computation for finding the index among axs that
hold the symbol. 
axs is a tuple of objects with method keys defined.

Only invoke with short tuples axs, because a method is generated for
each key.
"""
@noinline function findfirst_keyed(axs::Tuple, key::Val{KEY}) where {KEY} 
    # @noinline to avoid many methods of calling functions, but replaced by constant anyway
    i = findfirst_keyed_(axs, key, 1) # using plain Int here, works because of inlining
    Val(i)   # return Val so that can be used for type stable indexing
    # Claude Sonnet 4.6 told:
    # For the second call HVI.findfirst_keyed(axs, Val(:e)) with identical argument types:
    # No, the compiler does no work. Julia caches compiled specializations keyed by 
    # argument types. Since the types are identical to the first call, the cached compiled 
    # code (which already has the constant folded in) is looked up and reused directly.
    # The "work" of:
    #     type inference
    #     inlining findfirst_keyed_
    #     constant folding to Val{2}()
    #     code generation
    # all happened once, during the first call, and the result is in Julia's method cache.
    #  The second call with identical types is purely a cache lookup
end
@inline function findfirst_keyed_(axs::Tuple, key::Val{KEY}, i::Int) where {KEY} 
    KEY ∈ keys(first(axs)) ? i : findfirst_keyed_(Base.tail(axs), key, i+1)
end
@inline function findfirst_keyed_(axs::Tuple{}, key::Val{KEY}, i::Int) where {KEY}
    error("key " * string(KEY) * " not found in any of the axes.")
end


@inline function Base.getindex(ccat::VcatCAs{AX}, sym_val::Val{sym}, cols...) where {AX,sym}
    local ival
    i_val = findfirst_keyed(AX, sym_val)
    Base.getindex(ccat, i_val, sym_val, cols)
end

@inline function Base.getindex(ccat::VcatCAs{AX}, ::Val{I}, ::Val{sym}, cols::Tuple) where {AX, I,sym}
    local cm
    cm = CA.ComponentArray(ccat.cms[I], AX[I], ccat.flataxes...)
    getindex(cm, Val(sym), cols...)
end

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

@inline function _col(ccat::VcatCAs{AX,2-1}, j::Int) where {AX}
    cols = map(cm -> view(CA.getdata(cm), :, j), ccat.cms)
    VcatCAs(AX, cols...)
end

