# using ComponentArrays: ComponentArrays as CA

"""
    AbstractComponentArrayInterpreter

Interface for Type that implements
- `as_ca(::AbstractArray, interpreter) -> ComponentArray`
- `Base.length(interpreter) -> Int`

When called on a vector, forwards to `as_ca`.
"""
abstract type AbstractComponentArrayInterpreter end

"""
    as_ca(v::AbstractArray, interpretor)

Returns a ComponentArray with underlying data `v`.
"""
function as_ca end

(interpreter::AbstractComponentArrayInterpreter)(v::AbstractArray) = as_ca(v, interpreter)

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
    CA.ComponentArray(vr, AX)
end

function Base.length(::StaticComponentArrayInterpreter{AX}) where {AX}
    #sum(length, typeof(AX).parameters[1])
    prod(_axis_length.(AX))
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
    vr = reshape(v, _axis_length.(cai.axes))
    CA.ComponentArray(vr, cai.axes)
end

function Base.length(cai::ComponentArrayInterpreter) 
    prod(_axis_length.(cai.axes))
end

get_concrete(cai::ComponentArrayInterpreter) = StaticComponentArrayInterpreter{cai.axes}()


"""
    ComponentArrayInterpreter(; kwargs...)
    ComponentArrayInterpreter(::AbstractComponentArray)
    ComponentArrayInterpreter(::AbstractComponentArray, n_dims::NTuple{N,<:Integer})
    ComponentArrayInterpreter(n_dims::NTuple{N,<:Integer}, ::AbstractComponentArray)

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
end,
function ComponentArrayInterpreter(component_shapes::NamedTuple)
    component_counts = map(prod, component_shapes)
    n = sum(component_counts)
    x = 1:n
    is_end = cumsum(component_counts)
    is_start = (0, is_end[1:(end-1)]...) .+ 1
    #g = (x[i_start:i_end] for (i_start, i_end) in zip(is_start, is_end))
    g = (reshape(x[i_start:i_end], shape) for (i_start, i_end, shape) in zip(is_start, is_end, component_shapes))
    xc = CA.ComponentVector(; zip(propertynames(component_counts), g)...)
    ComponentArrayInterpreter(xc)
end

function ComponentArrayInterpreter(vc::CA.AbstractComponentArray)
    ComponentArrayInterpreter(CA.getaxes(vc))
end



# Attach axes to matrices and arrays of ComponentArrays
# with ComponentArrays in the first dimensions (e.g. rownames of a matrix or array)
function ComponentArrayInterpreter(
    ca::CA.AbstractComponentArray, n_dims::NTuple{N,<:Integer}) where N
    ComponentArrayInterpreter(CA.getaxes(ca), n_dims)
end
function ComponentArrayInterpreter(
    axes::NTuple{M, <:CA.AbstractAxis}, n_dims::NTuple{N,<:Integer}) where {M,N}
    axes_ext = (axes..., map(n_dim -> CA.Axis(i=1:n_dim), n_dims)...)
    ComponentArrayInterpreter(axes_ext)
end

# with ComponentArrays in the last dimensions (e.g. columnnames of a matrix)
function ComponentArrayInterpreter(
    n_dims::NTuple{N,<:Integer}, ca::CA.AbstractComponentArray) where N
    ComponentArrayInterpreter(n_dims, CA.getaxes(ca))
end
function ComponentArrayInterpreter(
    n_dims::NTuple{N,<:Integer}, axes::NTuple{M, <:CA.AbstractAxis}) where {N,M}
    axes_ext = (map(n_dim -> CA.Axis(i=1:n_dim), n_dims)..., axes...)
    ComponentArrayInterpreter(axes_ext)
end

# ambuiguity with two empty Tuples (edge prob that does not make sense)
# Empty ComponentVector with no other array dimensions -> empty componentVector
function ComponentArrayInterpreter(n_dims1::Tuple{}, n_dims2::Tuple{})
    ComponentArrayInterpreter(CA.ComponentVector())
end




# not exported, but required for testing
_get_ComponentArrayInterpreter_axes(::StaticComponentArrayInterpreter{AX}) where {AX} = AX
_get_ComponentArrayInterpreter_axes(cai::ComponentArrayInterpreter) = cai.axes


_axis_length(ax::CA.AbstractAxis) = lastindex(ax) - firstindex(ax) + 1
_axis_length(::CA.FlatAxis) = 0
_axis_length(::CA.UnitRange) = 0

"""
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
