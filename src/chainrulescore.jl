function ChainRulesCore.rrule(::typeof(as_ca), v::AbstractArray, int::AbstractComponentArrayInterpreter)
    as_ca(v, int),
    function rrule_as_ca_inner(Δ)
        vb = CA.getdata(unthunk(Δ))
        vbr = reshape(vb, size(v))
        (ChainRulesCore.NoTangent(), vbr, ChainRulesCore.NoTangent())
    end
end


# function ChainRulesCore.rrule(::typeof(concat_PMFix), θP, θMs, app)
#     nP = length(θP)
#     isP = 1:nP
#     isM = nP .+ 1:size(θMs,2)
#     #
#     concat_PMFix(θP, θMs, app),
#     function rrule_concat_PMFix_empty(Δ)
#         # when returning a gradient of size (n_bach, 0) there are errors
#         # need to live with dynamic dispatch of Union type of Matrix and ZeroTangent
#         ΔθP = isempty(θP) ?  ChainRulesCore.ZeroTangent() :
#             CA.ComponentArray(vec(sum(Δ[:,isP], dims=1)), CA.getaxes(θP))
#         ΔθMs = CA.ComponentArray(Δ[:, isM], CA.getaxes(θMs))
#         (ChainRulesCore.NoTangent(), ΔθP, ΔθMs, ChainRulesCore.NoTangent())
#     end
# end

# function ChainRulesCore.rrule(::typeof(transform_and_logjac_ζ), ζP, ζMs; transP, transMs)
#     transform_and_logjac_ζ(ζP, ζMs; transP, transMs),
#     function rrule_test(Δ)
#         Main.@infiltrate_main
#         (ChainRulesCore.NoTangent(), ΔζP, ΔζMs)
#     end
# end

