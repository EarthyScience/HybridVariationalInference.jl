"""
    OneBasedVectorWithZero(data)

A thin wrapper over an `AbstractVector` that exposes a linear 1-based indexing API
mapping `v[i]` to `data[axes(data, 1)[i]]` on the underlying storage
and provides a value at index 0 (defaulting to zero) that is not stored in the underlying vector.

Example usage:
```jldoctest; output=false
v = HVI.OneBasedVectorWithZero([10,20,30])
v[1] == 10
v[2] == 20
v[3] == 30
v[0] == 0 # default value at index 0 is zero
v[[1,0,0,3]] == [10,0,0,30]
```         
"""
struct OneBasedVectorWithZero{E,V<:AbstractVector{E}} <: AbstractVector{E}
    data::V
    val_at_zero::E # optional field to store the value at index 0 if needed
end

OneBasedVectorWithZero(v::AbstractVector; val_at_zero=zero(eltype(v))) = OneBasedVectorWithZero(v, val_at_zero)
Base.size(v::OneBasedVectorWithZero) = size(v.data)
Base.length(v::OneBasedVectorWithZero) = length(v.data)
Base.eltype(v::OneBasedVectorWithZero) = eltype(v.data)
Base.axes(v::OneBasedVectorWithZero) = (Base.OneTo(length(v)),)
Base.IndexStyle(::Type{<:OneBasedVectorWithZero}) = IndexLinear()
Base.empty(v::OneBasedVectorWithZero) = OneBasedVectorWithZero(empty(v.data), v.val_at_zero)

function Base.getindex(v::OneBasedVectorWithZero, i::Integer)
    if i == 0
        return v.val_at_zero
    elseif 1 <= i <= length(v)
        return v.data[axes(v.data,1)[i]]
    else
        throw(BoundsError(v, i))
    end    
end

# Bools is a subtype of Integer, need to handle this case separately
function Base.getindex(v::OneBasedVectorWithZero, inds::AbstractVector{<:Bool})
    v.data[inds]
end

function Base.getindex(v::OneBasedVectorWithZero, inds::AbstractVector{<:Integer})
    return map(i -> getindex(v, i), inds)
end

function Base.setindex!(v::OneBasedVectorWithZero, value, i::Integer)
    # setting index 0 is not allowed
    if 1 <= i <= length(v)
        v.data[axes(v.data, 1)[i]] = value
        return v
    else
        throw(BoundsError(v, i))
    end
end

function Base.iterate(v::OneBasedVectorWithZero, state=1)
    state > length(v) ? nothing : (v[state], state + 1)
end

function ChainRulesCore.rrule(::typeof(getindex), v::OneBasedVectorWithZero, i::Integer)
    if i == 0
        y = v.val_at_zero
        function pullback0(ȳ)
            # no gradient to base vector or val_at_zero
            return NoTangent(), NoTangent(), NoTangent()
        end
        return y, pullback0
    elseif 1 <= i <= length(v)
        y = v.data[axes(v.data,1)[i]]
        function pullback(ȳ)
            dv = zero(v.data)
            dv[axes(v.data,1)[i]] += ȳ
            return NoTangent(), OneBasedVectorWithZero(dv, zero(eltype(v))), NoTangent()
        end
        return y, pullback
    else
        throw(BoundsError(v, i))
    end
end

function ChainRulesCore.rrule(::typeof(getindex), v::OneBasedVectorWithZero, inds::AbstractVector{<:Integer})
    y = getindex(v, inds)
    function pullback(ȳ)
        dv = zero(v.data)
        for (k, idx) in enumerate(inds)
            if 1 <= idx <= length(v)
                dv[axes(v.data,1)[idx]] += ȳ[k]
            end
        end
        return NoTangent(), OneBasedVectorWithZero(dv, zero(eltype(v))), NoTangent()
    end
    return y, pullback
end

if isdefined(Main, :Zygote) || isdefined(HybridVariationalInference, :Zygote)
    Zygote.@adjoint function OneBasedVectorWithZero(v::AbstractVector; val_at_zero=zero(eltype(v)))
        h = OneBasedVectorWithZero(v, val_at_zero)
        return h, Δ -> (
            Δ isa OneBasedVectorWithZero ? Δ.data : Δ,
            NoTangent()
        )
    end
end
