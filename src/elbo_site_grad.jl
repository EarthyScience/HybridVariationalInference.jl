# two return values: primal and derivative 
#   given coderiv dy, parameters and helpers
function grad_g_apply!(ϕm, dϕg, dϕm, ϕg, xM, ζP, 
    pbm_covar_indices::Union{Nothing, AbstractArray}, g, h)
    # need to set shadows to zero before each gradient computation
    fill!(dϕg,     zero(eltype(dϕg)))
    fill!(h.dxMP,  zero(eltype(h.dxMP)))
    dϕm_buffer_key = isnothing(pbm_covar_indices) ? :dϕms : :dϕms_mcs
    copyto!(h[dϕm_buffer_key], dϕm) # copy to avoid modifying dϕm
    # Enzyme already accumulates to dϕg
    Enzyme.autodiff(
        Enzyme.Reverse,
        g_apply!,
        Enzyme.Duplicated(ϕm, h[dϕm_buffer_key]),
        Enzyme.Duplicated(ϕg, dϕg),
        Enzyme.Const(xM),
        Enzyme.Const(ζP),
        Enzyme.Const(pbm_covar_indices),
        Enzyme.Const(g),
        Enzyme.Duplicated(h.xMP, h.dxMP),
    )
end


# function grad_g_apply!(y, dϕg, dy, ϕg, xM, ζP, pbm_covar_indices::Nothing, g, h)
#     # without with no global parameter covariates:  (covar_indices::Nothing )
#     fill!(dϕg,     zero(eltype(dϕg)))
#     copyto!(h.dy, dy) # copy to avoid modifying dy
#     # Enzyme already accumulates to dϕg
#     Enzyme.autodiff(
#         Enzyme.Reverse,
#         apply_model!,
#         Enzyme.Duplicated(y, h.dy),
#         Enzyme.Const(g),
#         Enzyme.Const(xM),
#         Enzyme.Duplicated(ϕg, dϕg),
#     )
# end
