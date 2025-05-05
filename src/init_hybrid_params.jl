"""
    init_hybrid_params(θP, θM, ϕg, n_batch; transP=asℝ, transM=asℝ)

Setup ComponentVector of parameters to optimize, and associated tools.
Returns a NamedTuple of
- ϕ: A ComponentVector of parameters to optimize
- transPMs_batch, interpreters: Transformations and interpreters as 
  required by `neg_elbo_gtf`.
- get_transPMs: a function returning transformations `(n_site) -> (;P,Ms)`
- get_ca_int_PMs: a function returning ComponentArrayInterpreter for PMs vector 
  with PMs shaped as a matrix of `n_site` columns of `θM`

# Arguments
- `θP`, `θM`: Template ComponentVectors of global parameters and ML-predicted parameters
- `cor_ends`: NamedTuple with entries, `P`, and `M`, respectively with 
   integer vectors of ending columns of parameters blocks
- `ϕg`: vector of parameters to optimize, as returned by `get_hybridproblem_MLapplicator`
- `n_batch`: the number of sites to predicted in each mini-batch
- `transP`, `transM`: the Bijector.Transformations for the global and site-dependent 
    parameters, e.g. `Stacked(elementwise(identity), elementwise(exp), elementwise(exp))`.
    Its the transformation froing from unconstrained to constrained space: θ = Tinv(ζ),
    because this direction is used much more often.
- `ϕunc0` initial uncertainty parameters, ComponentVector with format of `init_hybrid_ϕunc.`
"""
function init_hybrid_params(θP::AbstractVector{FT}, θM::AbstractVector{FT},
        cor_ends::NamedTuple, ϕg::AbstractVector{FT}, n_batch;
        transP = elementwise(identity), transM = elementwise(identity),
        ϕunc0 = init_hybrid_ϕunc(cor_ends, zero(FT))) where {FT}
    n_θP = length(θP)
    n_θM = length(θM)
    @assert cor_ends.P[end] == n_θP
    @assert cor_ends.M[end] == n_θM
    n_ϕg = length(ϕg)
    # check translating parameters - can match length?
    _ = Bijectors.inverse(transP)(θP)
    _ = Bijectors.inverse(transM)(θM)
    ϕ = CA.ComponentVector(;
        μP = apply_preserve_axes(inverse(transP), θP),
        ϕg = ϕg,
        unc = ϕunc0)
    #
    get_transPMs = let transP = transP, transM = transM, n_θP = n_θP, n_θM = n_θM
        function get_transPMs_inner(n_site)
            transMs = ntuple(i -> transM, n_site)
            ranges = vcat(
                [1:n_θP], [(n_θP + i0 * n_θM) .+ (1:n_θM) for i0 in 0:(n_site - 1)])
            transPMs = Stacked((transP, transMs...), ranges)
            transPMs
        end
    end
    transPMs_batch = get_transPMs(n_batch)
    # ranges = (P = 1:n_θP, ϕg = n_θP .+ (1:n_ϕg), unc = (n_θP + n_ϕg) .+ (1:length(ϕunc0)))
    # inv_trans_gu = Stacked(
    #     (inverse(transP), elementwise(identity), elementwise(identity)), values(ranges))
    # ϕ = inv_trans_gu(CA.getdata(ϕt))        
    get_ca_int_PMs = let
        function get_ca_int_PMs_inner(n_site)
            ComponentArrayInterpreter(CA.ComponentVector(; P = θP,
                Ms = CA.ComponentMatrix(
                    zeros(n_θM, n_site), first(CA.getaxes(θM)), CA.Axis(i = 1:n_site))))
        end
    end
    interpreters = map(get_concrete,
        (;
            μP_ϕg_unc = ComponentArrayInterpreter(ϕ),
            PMs = get_ca_int_PMs(n_batch),
            unc = ComponentArrayInterpreter(ϕunc0)
        ))
    (; ϕ, transPMs_batch, interpreters, get_transPMs, get_ca_int_PMs)
end

"""
    init_hybrid_ϕunc(cor_ends, ρ0=0f0; logσ2_logP, coef_logσ2_ζMs, ρsP, ρsM)

Initialize vector of additional parameter of the approximate posterior.

Arguments:
- `cor_ends`: NamedTuple with entries, `P`, and `M`, respectively with 
   integer vectors of ending columns of parameters blocks
- `ρ0`: default entry for ρsP and ρsM, defaults = 0f0.
- `coef_logσ2_logM`: default column for `coef_logσ2_ζMs`, defaults to `[-10.0, 0.0]`

Returns a `ComponentVector` of 
- `logσ2_logP`: vector of log-variances of ζP (on log scale).
  defaults to -10
- `coef_logσ2_ζMs`: offset and slope for the log-variances of ζM scaling with 
   its value given by columns for each parameter in ζM, defaults to `[-10, 0]`
- `ρsP` and `ρsM`: parameterization of the upper triangular cholesky factor 
  of the correlation matrices of ζP and ζM, default to all entries `ρ0`, which defaults to zero.
"""
function init_hybrid_ϕunc(
        cor_ends::NamedTuple,
        ρ0::FT = 0.0f0,
        coef_logσ2_logM::AbstractVector{FT} = FT[-10.0, 0.0];
        logσ2_logP::AbstractVector{FT} = fill(FT(-10.0), cor_ends.P[end]),
        coef_logσ2_ζMs::AbstractMatrix{FT} = reduce(
            hcat, (coef_logσ2_logM for _ in 1:cor_ends.M[end])),
        ρsP = fill(ρ0, get_cor_count(cor_ends.P)),
        ρsM = fill(ρ0, get_cor_count(cor_ends.M)),
) where {FT}
    CA.ComponentVector(;
        logσ2_logP,
        coef_logσ2_ζMs,
        ρsP,
        ρsM)
end
