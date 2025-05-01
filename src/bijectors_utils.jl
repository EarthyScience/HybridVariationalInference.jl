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
# function Bijectors.with_logabsdet_jacobian(ib::Inverse{<:Exp}, y)
#     x = transform(ib, y)
#     return x, -logabsdetjac(inverse(ib), x)
# end


Bijectors.is_monotonically_increasing(::Exp) = true


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
