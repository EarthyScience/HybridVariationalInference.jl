# using ComponentArrays: ComponentArrays as CA

"""
    AbstractComponentArrayInterpreter

Interface for Type that implements
- `as_ca(::AbstractArray, interpreter) -> ComponentArray`
- `ComponentArrays.getaxes(interpreter)`
- `Base.length(interpreter) -> Int`

When called on a vector, forwards to `as_ca`.

There is a default implementation for Base.length based on ComponentArrays.getaxes.
"""
abstract type AbstractComponentArrayInterpreter end

"""
    as_ca(v::AbstractArray, interpretor)

Returns a ComponentArray with underlying data `v`.
"""
function as_ca end

function Base.length(cai::AbstractComponentArrayInterpreter)
    prod(_axis_length.(CA.getaxes(cai)))
end

function (interpreter::AbstractComponentArrayInterpreter)(v::AbstractArray{ET}) where ET
    as_ca(v, interpreter)::CA.ComponentArray{ET}
end

"""
Concrete version of `AbstractComponentArrayInterpreter` that stores an axis
in its type signature.

Allows compiler-inferred `length` to construct StaticArrays, but requires specialization
on dispatch when provided as an argument to a function.

Use `get_concrete(cai::ComponentArrayInterpreter)` to pass a concrete version to 
performance-critical functions.
"""
struct StaticComponentArrayInterpreter{AX} <: AbstractComponentArrayInterpreter end
function as_ca(v::AbstractArray, ::StaticComponentArrayInterpreter{AX}) where {AX}
    vr = reshape(v, _axis_length.(AX))
    CA.ComponentArray(vr, AX)::CA.ComponentArray{eltype(v)}
end

function StaticComponentArrayInterpreter(component_shapes::NamedTuple)
    axs = map(component_shapes) do valx
        x = _val_value(valx)
        ax = x isa Integer ? CA.Shaped1DAxis((x,)) : CA.ShapedAxis(x)
        (ax,)
    end
    axc = compose_axes(axs)
    StaticComponentArrayInterpreter{(axc,)}()
end
function StaticComponentArrayInterpreter(ca::CA.ComponentArray)
    ax = CA.getaxes(ca)
    StaticComponentArrayInterpreter{ax}()
end

# concatenate from several other ArrayInterpreters, keep static
# did not manage to get it inferred, better use get_concrete(ComponentArrayInterpreter)
# also does not save allocations
# function StaticComponentArrayInterpreter(; kwargs...)
#     ints = values(kwargs)
#     axc = compose_axes(ints)
#     intc = StaticComponentArrayInterpreter{(axc,)}()
#     return(intc)
# end

# function Base.length(::StaticComponentArrayInterpreter{AX}) where {AX}
#     #sum(length, typeof(AX).parameters[1])
#     prod(_axis_length.(AX))
# end

function CA.getaxes(int::StaticComponentArrayInterpreter{AX}) where {AX}
    AX
end

get_concrete(cai::StaticComponentArrayInterpreter) = cai

"""
Non-Concrete version of `AbstractComponentArrayInterpreter` that avoids storing
additional type parameters.

Does not trigger specialization for Interpreters of different axes, but does
not allow compiler-inferred `length` to construct StaticArrays.

Use `get_concrete(cai::ComponentArrayInterpreter)` to pass a concrete version to 
performance-critical functions.
"""
struct ComponentArrayInterpreter <: AbstractComponentArrayInterpreter
    axes::Tuple  #{T, <:CA.AbstractAxis}
end

function as_ca(v::AbstractArray, cai::ComponentArrayInterpreter)
    vr = reshape(CA.getdata(v), _axis_length.(cai.axes))
    CA.ComponentArray(vr, cai.axes)::CA.ComponentArray{eltype(v)}
end

function CA.getaxes(cai::ComponentArrayInterpreter)
    cai.axes
end

get_concrete(cai::ComponentArrayInterpreter) = StaticComponentArrayInterpreter{cai.axes}()

"""
    ComponentArrayInterpreter(; kwargs...)
    ComponentArrayInterpreter(::AbstractComponentArray)
    
    ComponentArrayInterpreter(::AbstractComponentArray, n_dims::NTuple{N,<:Integer})
    ComponentArrayInterpreter(n_dims::NTuple{N,<:Integer}, ::AbstractComponentArray)
    ComponentArrayInterpreter(n_dims::NTuple{N,<:Integer}, ::AbstractComponentArray, m_dims::NTuple{M,<:Integer})

Construct a `ComponentArrayInterpreter <: AbstractComponentArrayInterpreter`
with components being vectors of given length or given model of a `AbstractComponentArray`.

The other constructors allow constructing arrays with additional dimensions.

'''julia
    interpreter = ComponentArrayInterpreter(; P=2, M=(2,3), Unc=5)
    v = 1.0:length(interpreter)
    interpreter(v).M == 2 .+ [1 3 5; 2 4 6]
    vm = stack((v,v .* 10, v .* 100))

    intm = ComponentArrayInterpreter(interpreter(v), (3,))
    intm(vm)[:Unc, 2]


'''
"""
function ComponentArrayInterpreter(; kwargs...)
    ComponentArrayInterpreter(values(kwargs))
end
function ComponentArrayInterpreter(component_shapes::NamedTuple)
    #component_counts = map(prod, component_shapes)
    # avoid constructing a template first, but create axes 
    # n = sum(component_counts)
    # x = 1:n
    # is_end = cumsum(component_counts)
    # #is_start = (0, is_end[1:(end-1)]...) .+ 1  # problems with Zygote
    # is_start = Iterators.flatten((1:1, is_end[1:(end-1)] .+ 1))
    # g = (reshape(x[i_start:i_end], shape) for (i_start, i_end, shape) in zip(is_start, is_end, component_shapes))
    # xc = CA.ComponentVector(; zip(propertynames(component_counts), g)...)
    # #nt = NamedTuple{propertynames(component_counts)}(g)
    # ComponentArrayInterpreter(xc)
    axs = map(x -> (x isa Integer ? CA.Shaped1DAxis((x,)) : CA.ShapedAxis(x),), component_shapes)
    ax = compose_axes(axs)
    m1 = ComponentArrayInterpreter((ax,))
end

function ComponentArrayInterpreter(vc::CA.AbstractComponentArray)
    ComponentArrayInterpreter(CA.getaxes(vc))
end

const CAorCAI = Union{CA.AbstractComponentArray, AbstractComponentArrayInterpreter}

# Attach axes to matrices and arrays of ComponentArrays
# with ComponentArrays in the first dimensions (e.g. rownames of a matrix or array)
function ComponentArrayInterpreter(ca::CAorCAI, n_dims::NTuple{N,<:Integer}) where {N}
    ComponentArrayInterpreter((), CA.getaxes(ca), n_dims)
end
# with ComponentArrays in the last dimensions (e.g. columnnames of a matrix)
function ComponentArrayInterpreter(n_dims::NTuple{N,<:Integer}, ca::CAorCAI) where {N}
    ComponentArrayInterpreter(n_dims, CA.getaxes(ca), ())
end
# with ComponentArrays in the center dimensions (e.g. columnnames of a 3D-array)
function ComponentArrayInterpreter(
    n_dims::NTuple{N,<:Integer}, ca::CAorCAI, m_dims::NTuple{M,<:Integer}) where {N,M}
    ComponentArrayInterpreter(n_dims, CA.getaxes(ca), m_dims)
end


function ComponentArrayInterpreter(
    n_dims::NTuple{N,<:Integer}, axes::NTuple{A,<:CA.AbstractAxis},
    m_dims::NTuple{M,<:Integer}) where {N,A,M}
    axes_ext = (
        map(n_dim -> CA.Axis(i=1:n_dim), n_dims)..., 
        axes..., 
        map(n_dim -> CA.Axis(i=1:n_dim), m_dims)...)
    ComponentArrayInterpreter(axes_ext)
end

# support also for other AbstractComponentArrayInterpreter types
# in a type-stable way by providing the Tuple of dimensions as a value type
"""
    stack_ca_int(cai::AbstractComponentArrayInterpreter, ::Val{n_dims})

Interpret the first dimension of an Array as a ComponentArray. Provide the Tuple
of following dimensions by a value type, e.g. `Val((n_col, n_z))`.
"""
function stack_ca_int(
    cai::IT, ::Val{n_dims}) where {IT<:AbstractComponentArrayInterpreter,n_dims}
    @assert n_dims isa NTuple{N,<:Integer} where {N}
    IT.name.wrapper(CA.getaxes(cai), n_dims)::IT.name.wrapper
end
function StaticComponentArrayInterpreter(
    axes::NTuple{A,<:CA.AbstractAxis}, n_dims::NTuple{N,<:Integer}) where {A,N}
    axes_ext = (axes..., map(n_dim -> CA.Axis(i=1:n_dim), n_dims)...)
    StaticComponentArrayInterpreter{axes_ext}()
end


function stack_ca_int(
    ::Val{n_dims}, cai::IT) where {IT<:AbstractComponentArrayInterpreter,n_dims}
    @assert n_dims isa NTuple{N,<:Integer} where {N}
    IT.name.wrapper(n_dims, CA.getaxes(cai))::IT.name.wrapper
end
function StaticComponentArrayInterpreter(
    n_dims::NTuple{N,<:Integer}, axes::NTuple{M,<:CA.AbstractAxis}) where {N,M}
    axes_ext = (map(n_dim -> CA.Axis(i=1:n_dim), n_dims)..., axes...)
    StaticComponentArrayInterpreter{axes_ext}()
end


# ambuiguity with two empty Tuples (edge prob that does not make sense)
# Empty ComponentVector with no other array dimensions -> empty componentVector
function ComponentArrayInterpreter(n_dims1::Tuple{}, n_dims2::Tuple{})
    ComponentArrayInterpreter((CA.Axis(),))
end
function StaticComponentArrayInterpreter(n_dims1::Tuple{}, n_dims2::Tuple{})
    StaticComponentArrayInterpreter{(CA.Axis(),)}()
end

# concatenate several 1d ComponentArrayInterpreters
function compose_interpreters(; kwargs...)
    compose_interpreters(values(kwargs))
end

function compose_interpreters(ints::NamedTuple)
    axtuples = map(x -> CA.getaxes(x), ints)
    axc = compose_axes(axtuples)
    intc = ComponentArrayInterpreter((axc,))
    return (intc)
end


# not exported, but required for testing
_get_ComponentArrayInterpreter_axes(::StaticComponentArrayInterpreter{AX}) where {AX} = AX
_get_ComponentArrayInterpreter_axes(cai::ComponentArrayInterpreter) = cai.axes

_axis_length(ax::CA.AbstractAxis) = lastindex(ax) - firstindex(ax) + 1
_axis_length(::CA.FlatAxis) = 0
_axis_length(::CA.UnitRange) = 0

"""
    flatten1(cv::CA.ComponentVector)

Removes the highest level of keys.
Keeps the reference to the underlying data, but changes the axis.
If first-level vector has no sub-names, an error (Aguement Error tuple must be non-empty)
is thrown.
"""
function flatten1(cv::CA.ComponentVector)
    # return a tuple of (key, value) as zip(keys, values) would do
    # gen = (((ks, cv[k][ks]) for ks in keys(cv[k])) for k in keys(cv) if !isempty(cv[k]))
    # cv_new = CA.ComponentVector(; Iterators.Flatten(gen)...)
    # benchmarks show that the vcat variant is more efficient
    if length(cv) == 0
        return CA.ComponentVector(cv, CA.FlatAxis())
    else
        gen_cvs = (cv[k] for k in keys(cv) if !isempty(cv[k]))
        cv_new = reduce(vcat, gen_cvs)
        CA.ComponentVector(cv, first(CA.getaxes(cv_new)))
    end
end

"""
    get_positions(cai::AbstractComponentArrayInterpreter)

Create a NamedTuple of integer indices for each component.
Assumes that interpreter results in a one-dimensional array, i.e. in a ComponentVector.
"""
function get_positions(cai::AbstractComponentArrayInterpreter)
    #@assert length(CA.getaxes(cai)) == 1
    cv = cai(1:length(cai))
    keys_cv = keys(cv)
    # splatting creates Problems with Zygote
    #keys_cv isa Tuple ? (; (k => CA.getdata(cv[k]) for k in keys_cv)...) : CA.getdata(cv)
    keys_cv isa Tuple ? NamedTuple{keys_cv}(map(k -> CA.getdata(cv[k]), keys_cv)) : CA.getdata(cv)
end

function tmpf(v;
    cv,
    cai::AbstractComponentArrayInterpreter=get_concrete(ComponentArrayInterpreter(cv)))
    cai(v)
end

function tmpf1(v; cai)
    caic = get_concrete(cai)
    #caic(v)
    Test.@inferred tmpf(v, cv=nothing, cai=caic)
end

function tmpf2(v; cai::AbstractComponentArrayInterpreter)
    caic = get_concrete(cai)
    #caic = cai
    cv = Test.@inferred caic(v) # inferred inside tmpf2
    #cv = caic(v) # inferred inside tmpf2
    vv = tmpf(v; cv=nothing, cai=caic)
    #vv = tmpf(v; cv)
    #cv.x
    #sum(cv) # not inferred on Union cv (axis not know)
    #cv.x::AbstractVector{eltype(vv)} # not sufficient
    # need to specify concrete return type, but can rely on eltype
    sum(vv)::eltype(vv)  # need to specify return type 
end
