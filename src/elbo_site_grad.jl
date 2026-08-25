# two return values: primal and derivative 
#   given coderiv dy, parameters and helpers
function grad_g_apply!(y, dϕg, dy, ϕg, xM, ζP, pbm_covar_indices, g, h)
    # need to set shadows to zero before each gradient computation
    fill!(dϕg,     zero(eltype(dϕg)))
    fill!(h.dxMP,  zero(eltype(h.dxMP)))
    copyto!(h.dy, dy) # copy to avoid modifying dy
    # Enzyme already accumulates to dϕg
    Enzyme.autodiff(
        Enzyme.Reverse,
        g_apply!,
        Enzyme.Duplicated(y, h.dy),
        Enzyme.Duplicated(ϕg, dϕg),
        Enzyme.Const(xM),
        Enzyme.Const(ζP),
        Enzyme.Const(pbm_covar_indices),
        Enzyme.Const(g),
        Enzyme.Duplicated(h.xMP, h.dxMP),
    )
end
