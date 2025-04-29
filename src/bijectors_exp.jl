struct Exp <: Bijector
end

#Functors.@functor Exp

transform(b::Exp, x) = exp.(x) # note the broadcast
transform(ib::Inverse{<:Exp}, y) = log.(y)

# `logabsdetjac`
logabsdetjac(b::Exp, x) = sum(x)

`with_logabsdet_jacobian`
function Bijectors.with_logabsdet_jacobian(b::Exp, x)
    return transform(b, x), logabsdetjac(b, x)
end

is_monotonically_increasing(::Exp) = true
