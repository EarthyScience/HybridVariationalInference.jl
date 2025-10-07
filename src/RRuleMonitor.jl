"""
    RRuleMonitor(label, f, [ad_backend::ADTypes.AbstractADType])
    
Identity mapping of Callable or function `f` that intervenes the the pullback 
and raises an error if the supplied cotangent or the jacobian 
contains non-finitie entries.

Arguments
- label: id (String, or symbol) used in the error message.
- `ad_backend`: the AD backend used in `DifferentiationInterface.jacobian`.
  Defaults to `AutoZygote().`
"""
struct RRuleMonitor{F,L,A}  <: AbstractModelApplicator
    label::L
    f::F
    ad_backend::A
    #TODO think of preparing: https://juliadiff.org/DifferentiationInterface.jl/DifferentiationInterface/stable/tutorials/basic/#Preparing-for-multiple-gradients
end

RRuleMonitor(label, f) = RRuleMonitor(label, f, DI.AutoZygote())
#RRuleMonitor(label, f) = RRuleMonitor{typeof(label), typeof(f), DI.AutoZygote}(label, f, DI.AutoZygote())

@functor RRuleMonitor

function (m::RRuleMonitor)(args...; kwargs...)
    apply_RRuleMonitor(m, args...; kwargs...)
end

function apply_RRuleMonitor(m::RRuleMonitor, args...; kwargs...)
    m.f(args...; kwargs...)
end

# AbstractModelApplicator
apply_model(m::RRuleMonitor, x, ϕ; kwargs...) = m.f(x, ϕ; kwargs...)

function ChainRulesCore.rrule(::typeof(apply_RRuleMonitor), m::RRuleMonitor, args...; kwargs...)
    function apply_RRuleMonitor_pullback(Δy)
        # if m.label == "transP" 
        #     @show Δy[:]
        # end
        if !all(isfinite.(Δy[:]))
            @info "apply_RRuleMonitor_pullback: encountered non-finite co-gradients Δy " * 
                "for RRuleMonitor $(m.label)"  
            #@show Δy[:]
            #Main.@infiltrate_main
            error("traceback")
        end
        # do not call apply_RRuleMonitor because that results in infinite recursion
        # for backends other than AutoZygote need to call jacobian for single argument function
        jacs = if m.ad_backend isa DI.AutoZygote 
            function ftmp(args_...) 
                m.f(args_...; kwargs...)
            end
            jacs = Zygote.jacobian(ftmp, args...)
        else
            Tuple(begin
                fxi = (x) -> m.f(args[1:(i-1)]..., x, args[i+1:end]...; kwargs...)
                DI.jacobian(fxi, m.ad_backend, args[i])
            end for (i,x) in enumerate(args))
        end
        for (i,jac) in enumerate(jacs)
            if !all(isfinite.(jac))
                #@show jac
                @info("apply_RRuleMonitor_pullback: encountered non-finite Jacobian " * 
                    "for argument $(i) in RRuleMonitor $(m.label)")
                #Main.@infiltrate_main
                error("traceback")
            end
        end
        projectors = (ProjectTo(arg) for arg in args)
        # if m.label == "f in gf" && Main.cnt_SteadySOCPools >= 1118
        #     Main.@infiltrate_main
        # end
        #(pr,jac,x) = first(zip(projectors,jacs,args))
        Δx = (@thunk(pr(reshape(jac' * vec(Δy), size(x)))) for (pr,jac,x) in zip(
            projectors,jacs,args))
        # Δx = Tuple(begin
        #     if size(jac',2) != size(Δy,1) 
        #         Main.@infiltrate_main
        #     end
        #     Δxi = jac' * Δy[:]
        #     Δxip = pr(reshape(Δxi, size(x)))
        # end for (pr,jac,x) in zip(projectors,jacs,args))
        (NoTangent(), NoTangent(), Δx...)
    end
    return apply_RRuleMonitor(m, args...; kwargs...), apply_RRuleMonitor_pullback
end

# # with DI support only functions of one argument
# function ChainRulesCore.rrule(::typeof(apply_RRuleMonitor), m::RRuleMonitor, x; kwargs...)
#     function apply_RRuleMonitor_pullback(Δy)
#         if !all(isfinite.(Δy))
#             @info "apply_RRuleMonitor_pullback: encountered non-finite co-gradients Δy " * 
#                 "for RRuleMonitor $(m.label)"  
#             #@show Δy[:]
#             #Main.@infiltrate_main
#             error("traceback")
#         end
#         # do not call apply_RRuleMonitor because that results in infinite recursion
#         ftmp = (x_) -> m.f(x_; kwargs...)
#         jac = DI.jacobian(ftmp, m.ad_backend, x)
#         if !all(isfinite.(jac))
#             #@show jac
#             @info("apply_RRuleMonitor_pullback: encountered non-finite Jacobian " * 
#                 "in RRuleMonitor $(m.label)")
#             #Main.@infiltrate_main
#             error("traceback")
#         end
#         pr = ProjectTo(x) 
#         if m.label == "f in gf"
#             #Main.@infiltrate_main
#         end
#         Δx = @thunk(pr(reshape(jac' * vec(Δy), size(x)))) 
#         (NoTangent(), NoTangent(), Δx)
#     end
#     return apply_RRuleMonitor(m, args...; kwargs...), apply_RRuleMonitor_pullback
# end

