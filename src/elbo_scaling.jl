# Similar to MeanHVIApproximationMat in elbo.jl
# but ML model predicts a scaling factor for a group of variance parameters
# ϕq element logσ2_ζM_offsets contains a vector of log-offsets, i.e. multipliers, 
#   for each block of ML scaled parameters
#   the log-offset for the last entry in each block is stored in approx.logσ2_ζM_base


function get_marginal_std(approx::AbstractMeanScalingHVIApproximation, 
        ϕqc::CA.ComponentVector{T}, ϕm::AbstractMatrix=Matrix{eltype(ϕq)}[]) where T
    # add 0 as last logσ2_par_offset-par in block
    logσ2_par_offsets_before_end = OneBasedVectorWithZero(ϕqc[Val(:logσ2_ζM_offsets)])
    logσ2_par_offsets = logσ2_par_offsets_before_end[approx.idxs_par0]
    n_scale_blocks = length(approx.scalingblocks_ends)
    n_par = size(ϕm,1) - n_scale_blocks
    ϕm_scalings = ϕm[(n_par+1):end,:]
    logσ2_sites_offset_blocks = logit.(ϕm_scalings) # (0..1)->(-Inf, +Inf), 0.5->0
    logσ2_site_offsets = logσ2_sites_offset_blocks[approx.idxs_repblocks,:]
    #
    logσ2_ζMs = approx.logσ2_ζM_bases .+ logσ2_par_offsets .+ logσ2_site_offsets
    logσ2_ζP = vec(CA.getdata(ϕqc[Val(:logσ2_ζP)]))
    σMs = exp.(logσ2_ζMs ./ T(2))
    σP = exp.(logσ2_ζP ./ T(2))
    (;σP, σMs)
end





