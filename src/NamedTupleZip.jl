"""
    NamedTupleZip{Names, T<:Tuple}

A zero-allocation, splittable iterator that zips together the fields of a
`NamedTuple` of equal-length tuples, yielding a `NamedTuple` per iteration
instead of a plain `Tuple`.

Given a `NamedTuple` like `(x = (1, 2, 3), y = (4, 5, 6))`, iterating a
`NamedTupleZip` wrapping it yields `(x = 1, y = 4)`, `(x = 2, y = 5)`,
`(x = 3, y = 6)`, in order.

# Fields
- `data::T`: the tuple of subcomponent tuples (e.g. `(x_tuple, y_tuple, ...)`).
- `range::UnitRange{Int}`: the current index range this iterator covers.
  Present (rather than always `1:n`) so the iterator can be split into
  sub-ranges for parallel execution (see [`SplittablesBase.halve`](@ref)).

# Interfaces implemented
- `Base.iterate`
- `Base.length`
- `Base.eltype`
- `Base.IteratorSize` (returns `Base.HasLength()`)
- `SplittablesBase.halve` (enables use with `Folds.jl` / `Transducers.jl`
  parallel reducers)

# Performance notes
- All field names (`Names`) are compile-time type parameters, so every
  `NamedTuple{Names}(vals)` constructed during iteration is fully
  type-inferred — no dynamic dispatch, no allocation.
- `ntuple(..., Val(length(Names)))` is used internally to force
  compile-time loop unrolling when building the value tuple for each
  element.
- Splitting via `halve` only partitions the `range`; `data` is shared
  (not copied) between the two halves, so splitting itself is
  allocation-free (aside from unavoidable minimal `Task` overhead when
  used with threaded executors).

# Examples
```julia
subcomponents = (x = (1, 2, 3), y = (4, 5, 6), z = (7, 8, 9));
iter = NamedTupleZip(subcomponents);
collect(iter)
# 3-element Vector{NamedTuple{(:x, :y, :z), Tuple{Int64, Int64, Int64}}}:
#  (x = 1, y = 4, z = 7)
#  (x = 2, y = 5, z = 8)
#  (x = 3, y = 6, z = 9)
```
"""
struct NamedTupleZip{Names, T<:Tuple}
    data::T
    range::UnitRange{Int}
end

"""
    NamedTupleZip(nt::NamedTuple{Names}) where Names

Construct a [`NamedTupleZip`](@ref) from a `NamedTuple` `nt` whose fields
are tuples (or other indexable, equal-length collections). The length of
the resulting iterator is taken from the first field of `nt`.

# Arguments
- `nt::NamedTuple{Names}`: a NamedTuple whose values are equal-length,
  indexable collections (e.g. `Tuple`s).
"""
function NamedTupleZip(nt::NamedTuple{Names}) where Names
    vals = values(nt)
    n = length(first(vals))
    NamedTupleZip{Names, typeof(vals)}(vals, 1:n)
end

"""
    Base.length(z::NamedTupleZip)

Return the number of elements `z` will yield, i.e. the length of its
current index range. Note that after splitting with `halve`, this
reflects the length of the corresponding sub-range, not the original
full collection.
"""
Base.length(z::NamedTupleZip) = length(z.range)

"""
    Base.iterate(z::NamedTupleZip{Names}, i = first(z.range)) where Names

Advance the iterator. Returns `(nt, i + 1)` where `nt::NamedTuple{Names}`
is constructed from the `i`-th element of each field in `z.data`, or
`nothing` once `i` exceeds `last(z.range)`.

Construction of `nt` uses `ntuple(..., Val(length(Names)))` to ensure
compile-time unrolling and avoid allocation.
"""
function Base.iterate(z::NamedTupleZip{Names}, i=first(z.range)) where Names
    i > last(z.range) && return nothing
    vals = ntuple(j -> z.data[j][i], Val(length(Names)))
    return NamedTuple{Names}(vals), i + 1
end

"""
    Base.eltype(::Type{NamedTupleZip{Names, T}}) where {Names, T}

Return the concrete `NamedTuple` type yielded by iteration, i.e.
`NamedTuple{Names, Tuple{eltype(T1), eltype(T2), ...}}`, where `T1, T2,
...` are the element types of `T`'s fields (the original subcomponent
tuple types).

This is distinct from `T` itself, which is the type of the *input*
tuple-of-tuples, not the per-element output type.
"""
function Base.eltype(::Type{NamedTupleZip{Names, T}}) where {Names, T}
    types = ntuple(i -> eltype(fieldtype(T, i)), Val(length(T.parameters)))
    NamedTuple{Names, Tuple{types...}}
end

"""
    Base.IteratorSize(::Type{<:NamedTupleZip})

Always returns `Base.HasLength()`, since the length of a `NamedTupleZip`
is known up front from its `range` field.
"""
Base.IteratorSize(::Type{<:NamedTupleZip}) = Base.HasLength()

"""
    SplittablesBase.halve(z::NamedTupleZip{Names}) where Names

Split `z` into two `NamedTupleZip`s covering the first and second halves
of its index range, respectively. The underlying `data` tuple is shared
(not copied) between both halves — only the `range` differs.

This method is what allows `NamedTupleZip` to be used with parallel
reducers from `Folds.jl` / `Transducers.jl`, which rely on `halve` to
recursively partition work across threads/tasks.

# Example
```jldoctest
nt = (x = (1,2,3,4), y = (5,6,7,8));

z = NamedTupleZip(nt);

z1, z2 = SplittablesBase.halve(z);

collect(z1)
2-element Vector{NamedTuple{(:x, :y), Tuple{Int64, Int64}}}:
 (x = 1, y = 5)
 (x = 2, y = 6)

collect(z2)
2-element Vector{NamedTuple{(:x, :y), Tuple{Int64, Int64}}}:
 (x = 3, y = 7)
 (x = 4, y = 8)
```
"""
function SplittablesBase.halve(z::NamedTupleZip{Names}) where Names
    r1, r2 = SplittablesBase.halve(z.range)
    return NamedTupleZip{Names, typeof(z.data)}(z.data, r1),
           NamedTupleZip{Names, typeof(z.data)}(z.data, r2)
end
