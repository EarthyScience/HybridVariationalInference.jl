"""
    @row_valviews a, b, ... = cm

Create row views into a `ComponentMatrix` (or any matrix-like object
supporting `view(cm, Val(:name), :)`) using compile-time `Val`-dispatch
on the row names.

For each variable `name` on the left-hand side, this macro expands to:

    name = view(cm, Val(:name), :)

This mirrors the convenience syntax of `@unpack`, but instead of
extracting scalar fields, it creates *views* into the rows of a
`ComponentMatrix`, tagged with `Val(Symbol)` so that `view` can be
specialized/dispatched on the row name at compile time (avoiding the
type instability of a runtime `Symbol` lookup).

# Example
```jldoctest
import ComponentArrays as CA
n_MC = 4; n_θP = 3;
pt = CA.ComponentVector(NamedTuple(Symbol("p"*string(i)) => i for i in 1:n_θP));
m1 = rand(n_θP, n_MC);
cm = CA.ComponentMatrix(m1, CA.getaxes(pt)[1], CA.FlatAxis());
@row_valviews p1, p2, p3 = cm
p1 == cm[:p1, :]
```

This expands to:
```julia
p1 = view(cm, Val(:p1), :)
p2 = view(cm, Val(:p2), :)
p3 = view(cm, Val(:p3), :)
```

# Notes
- The right-hand side (`cm`) is evaluated once per variable; if it's an
  expensive expression, consider binding it to a variable first.
- Works with any number of variables on the left-hand side, including
  a single variable (no comma needed).
"""
macro row_valviews(ex)
    # written by Sonnet on prompt:
    #     Write a Julia macro @row_valviews a,b = cm
    # That replaces by the two lines:
    # a = view(cm, Val(:a), :)
    # b = view(cm, Val(:b), :)
    # similar of what the @unpack macro already does.
    @assert ex.head == :(=) "@row_valviews expects `a, b, ... = cm`"
    
    lhs, rhs = ex.args[1], ex.args[2]
    
    # Handle both single variable and tuple-of-variables LHS
    names = (lhs isa Expr && lhs.head == :tuple) ? lhs.args : [lhs]
    
    assignments = map(names) do name
        :($(esc(name)) = view($(esc(rhs)), Val($(QuoteNode(name))), :))
    end
    
    return Expr(:block, assignments...)
end
