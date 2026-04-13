"""
    with_logabsdet_jacobians

Similar to with_logabsdet_jacobian, but returns as the second component a
vector of Jacobians of transformation of each component in x.
"""
function with_logabsdet_jacobians end;

with_logabsdet_jacobians(::typeof(identity), x) = x, zero(x)

#MAYBE: need to implement fallbacks for other Bijectors than Exp()
function with_logabsdet_jacobians(sb::Stacked, x::AbstractVector)
    if sb.length_in != length(x)
        error("input length mismatch ($(sb.length_in) != $(length(x)))")
    end
    y, logjacs = _with_logabsdet_jacobians(sb, x)
    if Bijectors.output_length(sb, length(x)) != length(y)
        error("output length mismatch ($(output_length(sb, length(x))) != $(length(y)))")
    end
    # if size(logjacs) != size(y)
    #     Main.@infiltrate_main
    # end
    @assert size(logjacs) == size(y)
    return (y, logjacs)
end
function _with_logabsdet_jacobians(sb::Stacked, x::AbstractVector)
    ys_and_logjacs = map(zip(sb.bs, sb.ranges_in)) do (b, r)
        with_logabsdet_jacobians(b, x[r])
    end
    y = reduce(vcat, map(first, ys_and_logjacs))
    #logjacs = reduce(+, map(last, ys_and_logjacs))
    logjacs = reduce(vcat, map(last, ys_and_logjacs))
    return (y, logjacs)
end

#------------------- Exp

"""
    Exp()

A bijector that applies broadcasted exponential function, i.e. `exp.(x)`.
It is equivalent to `elementwise(exp)` but works better with automatic
differentiation on GPU.
"""
struct Exp <: Bijector
end

#Functors.@functor Exp
Bijectors.transform(b::Exp, x) = exp.(x) # note the broadcast
Bijectors.transform(ib::Inverse{<:Exp}, y) = log.(y)

# `logabsdetjac`
Bijectors.logabsdetjac(b::Exp, x) = sum(x)

`with_logabsdet_jacobian`
function Bijectors.with_logabsdet_jacobian(b::Exp, x)
    return exp.(x), sum(x)
end

"""
    with_logabsdet_jacobians

Similar to with_logabsdet_jacobian, but returns as the second component a
vector of Jacobians of transformation of each component in x, rather than 
the sum.
"""
function with_logabsdet_jacobians(b::Exp, x)
    return exp.(x), x
end


# function Bijectors.with_logabsdet_jacobian(ib::Inverse{<:Exp}, y)
#     x = transform(ib, y)
#     return x, -logabsdetjac(inverse(ib), x)
# end
Bijectors.is_monotonically_increasing(::Exp) = true

#----------------------- Logistic
"""
    Logistic()

A bijector that applies broadcasted exponential function, i.e. `logit.(x)`.
It is equivalent to `elementwise(exp)` but works better with automatic
differentiation on GPU.
"""
struct Logistic <: Bijector
end

#Functors.@functor Logistic
Bijectors.transform(b::Logistic, x) = logistic.(x) # note the broadcast
Bijectors.transform(ib::Inverse{<:Logistic}, y) = logit.(y)

# `logabsdetjac`
# https://en.wikipedia.org/wiki/Logistic_function#Derivative
Bijectors.logabsdetjac(b::Logistic, x) = sum(loglogistic.(x) + log1mlogistic.(x)) 

`with_logabsdet_jacobian`
function Bijectors.with_logabsdet_jacobian(b::Logistic, x)
    return transform(b,x), logabsdetjac(b,x)
end
# function Bijectors.with_logabsdet_jacobian(ib::Inverse{<:Logistic}, y)
#     x = transform(ib, y)
#     return x, -logabsdetjac(inverse(ib), x)
# end

function with_logabsdet_jacobians(b::Logistic, x)
    return transform(b,x), loglogistic.(x) .+ log1mlogistic.(x)
end

Bijectors.is_monotonically_increasing(::Logistic) = true


"""
    StackedArray(stacked, nrow) 

A Bijectors.Transform that applies stacked to each column of an n-row matrix.
"""
struct StackedArray{S} <: Bijectors.Transform
    nrow::Int
    stacked::S
end

function StackedArray(stacked, nrow) 
    stacked_vec = extend_stacked_nrow(stacked, nrow)
    StackedArray{typeof(stacked_vec)}(nrow, stacked_vec)
end

Functors.@functor StackedArray (stacked,)

function Base.show(io::IO, b::StackedArray) 
    return print(io, "StackedArray ($(b.nrow), $(b.stacked))")
end

function Base.:(==)(b1::StackedArray, b2::StackedArray) 
    (b1.nrow == b2.nrow) && (b1.stacked == b2.stacked)
end

Bijectors.isclosedform(b::StackedArray) = isclosedform(b.stacked)

Bijectors.isinvertible(b::StackedArray) = isinvertible(b.stacked)

_transform_stackedarray(sb, x) = reshape(sb.stacked(vec(x)), size(x))
function _transform_stackedarray(sb, x::Adjoint{FT, <:GPUArraysCore.AbstractGPUArray}) where FT
    # errors with Zygote for Adjoint of GPUArray, need to copy first
    # TODO construct MWE and issue
    x_plain = copy(x)
    reshape(sb.stacked(vec(x_plain)), size(x_plain))
end
function Bijectors.transform(sb::StackedArray, x::AbstractArray{<:Real}) 
    _transform_stackedarray(sb, x)
end
    
_logabsdetjac_stackedarray(b,x) = logabsdet(b.stacked, vec(x))
function Bijectors.logabsdetjac(b::StackedArray, x::AbstractArray{<:Real})
    _logabsdetjac_stackedarray(b,x)
end

function Bijectors.with_logabsdet_jacobian(sb::StackedArray, x::AbstractArray)
    (y, logjac) = with_logabsdet_jacobian(sb.stacked, vec(x))
    ym = reshape(y, size(x))
    return (ym, logjac)
end

"""
    with_logabsdet_jacobians(sb::StackedArray, x::AbstractArray)

Return a Jacobian for each row in x.
"""
function with_logabsdet_jacobians(sb::StackedArray, x::AbstractArray)
    (y, logjacs_vec) = with_logabsdet_jacobians(sb.stacked, vec(x))
    ym = reshape(y, size(x))
    # move sum to elbo, here return all components
    #logjacs = sum(reshape(logjacs_vec, sb.nrow, :); dims = 2)[:,1]
    logjacs = reshape(logjacs_vec, sb.nrow, :)
    return (ym, logjacs)
end

function Bijectors.inverse(sb::StackedArray) 
    inv_stacked = inverse(sb.stacked)
    return StackedArray{typeof(inv_stacked)}(sb.nrow, inv_stacked)
end

"""
    extend_stacked_nrow(b::Stacked, nrow::Integer)

Create a Stacked bijectors that transforms nrow times the elements
of the original Stacked bijector.

# Example
```
X = reduce(hcat, ([x + y for x in 0:4 ] for y in 0:10:30))
b1 = CP.Exp()
b2 = identity
b = Stacked((b1,b2), (1:1,2:4))
bs = extend_stacked_nrow(b, size(X,1))
Xt = reshape(bs(vec(X)), size(X))
@test Xt[:,1] == b1(X[:,1])
@test Xt[:,2:4] == b2(X[:,2:4])
```
"""
function extend_stacked_nrow(b::Stacked, nrow::Integer)
    onet = one(eltype(first(b.ranges_in)))
    endpos = last.(b.ranges_in) .* nrow 
    startpos2 = (endpos[1:(end-1)] .+ onet)
    ranges = ntuple(length(endpos)) do i
        startpos = i == 1 ? onet : startpos2[i-1]
        startpos:endpos[i]
    end
    bs = Stacked(b.bs, ranges)
end



