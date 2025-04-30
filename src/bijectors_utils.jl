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
