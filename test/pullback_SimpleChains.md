## The actually-safe approach: treat SimpleChains as an opaque primitive

Instead of asking Zygote to re-derive the backward pass by tracing mutating code, you tell it **not to look inside** — by defining a `ChainRulesCore.rrule` (which Zygote automatically respects) that computes the forward pass normally and computes the backward pass using SimpleChains' *own, already-correct, hand-written* gradient routines. This sidesteps Zygote's mutation problem completely, because Zygote never sees the mutating code — it only sees a black-box function with a supplied derivative.

This is the same pattern used for e.g. custom CUDA kernels, `DifferentialEquations.jl` adjoints, etc.: wrap a fast imperative kernel, hand-derive/hand-call its gradient.

### Case 1: your model ends in a scalar loss (the common case)

If `m(x, ϕ)` is literally `SimpleChains.Chain` ending in a loss layer (scalar output), then `SimpleChains.valgrad!` *is already* exactly the backward pass you need (with implicit seed = 1), so the `rrule` is simple:

```julia
using ChainRulesCore, SimpleChains

function ChainRulesCore.rrule(m::SimpleChains.Chain, x, ϕ)
    y = m(x, ϕ)
    function m_pullback(ȳ)
        # ȳ should be ~1 for a scalar loss; otherwise you must rescale, see below
        g = similar(ϕ)
        SimpleChains.valgrad!(g, m, x, ϕ)   # fills g = ∂loss/∂ϕ
        x̄ = NoTangent()                      # or implement input-grad, see Case 2
        return (NoTangent(), x̄, ȳ .* g)
    end
    return y, m_pullback
end
```

Now `Zygote.gradient((x,ϕ) -> m(x,ϕ), x, ϕ)` works, because Zygote stops at the `Chain` call and uses your `rrule` instead of tracing further.

### Case 2: you need a general vector-output cotangent `ȳ`

This is exactly where your earlier `pullback_arg!` question comes back in. If `m`'s output is a vector (not a scalar loss) and you want `Zygote` to seed an arbitrary `ȳ`, `valgrad!` isn't directly applicable (it assumes a scalar loss seed). You'd need to:

1. Run the forward pass yourself, keeping the scratch buffer/layer outputs alive (mirroring what `valgrad!` does internally).
2. Walk the layers in reverse, calling `SimpleChains.pullback!`/`pullback_arg!` per layer, seeding the *last* layer's `C̄` with your supplied `ȳ` instead of the implicit loss-gradient seed.
3. Accumulate `∂L/∂ϕ` and, if needed, `∂L/∂x` from the `pullback_arg!` calls at the first layer.
4. Put that whole procedure inside the `rrule`'s pullback closure.

This is more work and leans on unexported internals (version-fragile, as discussed), but it is **more trustworthy than Enzyme's runtime-activity workaround**, because the actual arithmetic is SimpleChains' own tested backward implementation — you're just manually driving it instead of letting Enzyme/Zygote infer it.

## Practical recommendation

- **If your use case is "scalar loss, need `∂loss/∂ϕ` (and maybe `∂loss/∂x`) inside a larger Zygote-differentiated pipeline"**: write the `rrule` as in Case 1 — safe, fast, minimal risk.
- **If you need true per-output-element cotangents (Jacobian-vector products) for a vector-valued model**: either
  - do the manual reverse-layer-loop `rrule` (Case 2), validating against finite differences, or
  - just use `ForwardDiff.jacobian(x -> m(x, ϕ), x)` for the input-Jacobian piece (forward-mode handles mutating code fine, since it's pure operator overloading over dual numbers, not tracing) and keep `valgrad!` for the parameter-gradient piece.
- **Validate numerically regardless** — finite differences on a small case before trusting any of these in production, since you're intentionally working outside SimpleChains' documented/tested usage pattern (`train!`).

If you tell me whether your `m(x, ϕ)` ends in a loss layer (scalar) or produces a raw vector/array output, I can sketch the exact `rrule` for your case, including the layer-reversal loop if needed.



Good — this is a different mechanism than `ChainRulesCore.rrule`. Zygote picks up `rrule`s automatically (via `ChainRulesCore`), but **Enzyme does not use `ChainRulesCore` by default** — it has its own custom-rule interface, `EnzymeRules`. To make Enzyme skip tracing through SimpleChains' mutating internals and instead use your hand-computed VJP, you need to define `EnzymeRules.augmented_primal` and `EnzymeRules.reverse` for a wrapper function.

**Strong caveat first:** the `EnzymeRules` API (`RevConfigWidth`, `AugmentedReturn`, exact argument order, `needs_primal`/`needs_shadow`) has shifted across Enzyme/EnzymeCore versions, and is less stable/documented than `ChainRulesCore.rrule`. Treat the template below as a structural sketch to adapt against your installed version's docs/examples (`EnzymeRules` section of the Enzyme.jl docs, and the test suite in the Enzyme.jl repo have worked examples) — then validate numerically before trusting it.

## 1. Wrap the model call in a named function

Enzyme dispatches custom rules on function *types*, so give yourself a stable target:

```julia
call_m(x, ϕ) = m(x, ϕ)
```

## 2. Define the loss-seed trick (as before)

Same `DotSeedLoss` custom loss layer from before — this lets you get `ϕ̄ = Jϕᵗ ȳ` via SimpleChains' own `valgrad!` rather than tracing mutating code.

## 3. Define the EnzymeRules custom rule

```julia
using EnzymeCore
import EnzymeCore.EnzymeRules: augmented_primal, reverse, AugmentedReturn
import EnzymeCore.EnzymeRules
using SimpleChains, ForwardDiff, LinearAlgebra

function EnzymeRules.augmented_primal(
        config::EnzymeRules.RevConfigWidth{1},
        func::Const{typeof(call_m)},
        ::Type{<:EnzymeCore.Annotation},
        x::EnzymeCore.Annotation,
        ϕ::EnzymeCore.Annotation,
    )
    xval, ϕval = x.val, ϕ.val
    y = call_m(xval, ϕval)

    primal = EnzymeRules.needs_primal(config) ? y : nothing
    shadow = EnzymeRules.needs_shadow(config) ? zero(y) : nothing

    # save what reverse() needs
    tape = (xval, ϕval, shadow)
    return AugmentedReturn(primal, shadow, tape)
end

function EnzymeRules.reverse(
        config::EnzymeRules.RevConfigWidth{1},
        func::Const{typeof(call_m)},
        ::Type{<:EnzymeCore.Annotation},
        tape,
        x::EnzymeCore.Annotation,
        ϕ::EnzymeCore.Annotation,
    )
    xval, ϕval, dy = tape
    ȳ = dy   # cotangent accumulated into our shadow by downstream ops

    if !(ϕ isa Const)
        g = similar(ϕval)
        mloss = SimpleChains.add_loss(m, DotSeedLoss(ȳ))
        SimpleChains.valgrad!(g, mloss, xval, ϕval)
        ϕ.dval .+= g
    end

    if !(x isa Const)
        Jx = ForwardDiff.jacobian(x_ -> call_m(x_, ϕval), xval)
        x.dval .+= Jx' * ȳ
    end

    fill!(dy, 0)   # reset shadow if reused across calls
    return (nothing, nothing)
end
```

## 4. How this gets "seeded" — compose through a scalar reduction

This is the important conceptual point: **a custom Enzyme reverse rule for a function with array output doesn't take an externally-supplied `ȳ` directly** — its `dy` shadow is filled in automatically by whatever downstream operation consumes `y`, during the normal reverse sweep. So to get the VJP for a specific seed `ȳ`, embed the call exactly as you would mathematically:

```julia
ybar = ...                      # your chosen seed
L(x, ϕ) = dot(ybar, call_m(x, ϕ))

dx = zero(x); dϕ = zero(ϕ)
Enzyme.autodiff(Enzyme.Reverse, L, Active,
                Duplicated(x, dx), Duplicated(ϕ, dϕ))
# dx ≈ x̄ = Jxᵗ ybar
# dϕ ≈ ϕ̄ = Jϕᵗ ybar
```

Here, Enzyme differentiates `dot(ybar, ·)` using its *built-in* rule (safe, no mutation issues — `dot` isn't SimpleChains code), which writes `ybar` into the shadow `dy` of `call_m`'s output; then Enzyme calls your custom `reverse` for `call_m`, which reads that `dy` as `ȳ` and does the real work via `valgrad!`/`ForwardDiff` rather than tracing SimpleChains internals. This is exactly analogous to the `ChainRulesCore.rrule` version, just routed through Enzyme's rule system instead of Zygote's.

## 5. Validate before trusting

```julia
using FiniteDifferences
fdm = central_fdm(5, 1)
# check dϕ against finite differences of L(x, ϕ) w.r.t. ϕ
# check dx against finite differences of L(x, ϕ) w.r.t. x
```

## Practical notes

- If you only need `ϕ̄` (not `x̄`) in a given call, mark `x` as `Const(...)` instead of `Duplicated` — this both saves the `ForwardDiff.jacobian` cost and removes one source of risk.
- Replace the `ForwardDiff.jacobian` line with a manual reverse-layer loop (`pullback_arg!`) only if profiling shows it's a bottleneck — same tradeoff as in the Zygote/ChainRulesCore version.
- Double-check the exact `EnzymeRules` method signature against your installed `Enzyme`/`EnzymeCore` version — e.g. whether it's `RevConfigWidth{1}` or `Config`, whether `AugmentedReturn` takes 3 vs 4 fields, whether activity annotations are `EnzymeCore.Annotation` or `Enzyme.Annotation` — by looking at a current worked example in the Enzyme.jl repository/docs, since this is the part most likely to have shifted between versions.
- This approach entirely avoids the `runtime_activity` workaround from before, since Enzyme never traces into SimpleChains' mutating code at all — which is the safer design here.