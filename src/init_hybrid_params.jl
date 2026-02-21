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
function init_hybrid_params(ϕg::AbstractVector{FT}, ϕq::AbstractVector{FT}) where {FT}
    ϕ = CA.ComponentVector(; ϕg, ϕq)
    interpreters = map(get_concrete,
        (;
            ϕg_ϕq = ComponentArrayInterpreter(ϕ),
            ϕq = ComponentArrayInterpreter(ϕq)
        ))
    (; ϕ, interpreters)
end

# function init_hybrid_params_old(θP::AbstractVector{FT}, θM::AbstractVector{FT},
#         cor_ends::NamedTuple, ϕg::AbstractVector{FT}, hpints::HybridProblemInterpreters;
#         transP = elementwise(identity), transM = elementwise(identity),
#         ϕunc0 = init_hybrid_ϕunc(cor_ends, zero(FT))) where {FT}
#     n_θP = length(θP)
#     n_θM = length(θM)
#     @assert cor_ends.P[end] == n_θP
#     @assert cor_ends.M[end] == n_θM
#     n_ϕg = length(ϕg)
#     # check translating parameters - can match length?
#     _ = Bijectors.inverse(transP)(θP)
#     _ = Bijectors.inverse(transM)(θM)
#     # TODO add and test θP
#     ϕq = update_μP_by_θP(ϕunc0, θP, transP)
#     ϕ = CA.ComponentVector(; ϕg, ϕq)
#     #
#     # get_transPMs = let transP = transP, transM = transM, n_θP = n_θP, n_θM = n_θM
#     #     function get_transPMs_inner(n_site)
#     #         transMs = ntuple(i -> transM, n_site)
#     #         ranges = vcat(
#     #             [1:n_θP], [(n_θP + i0 * n_θM) .+ (1:n_θM) for i0 in 0:(n_site - 1)])
#     #         transPMs = Stacked((transP, transMs...), ranges)
#     #         transPMs
#     #     end
#     # end
#     get_transPMs = transPMs_batch = Val(Symbol("deprecated , use stack_ca_int(intPMs)"))
#     #transPMs_batch = get_transPMs(n_batch)
#     # ranges = (P = 1:n_θP, ϕg = n_θP .+ (1:n_ϕg), unc = (n_θP + n_ϕg) .+ (1:length(ϕunc0)))
#     # inv_trans_gu = Stacked(
#     #     (inverse(transP), elementwise(identity), elementwise(identity)), values(ranges))
#     # ϕ = inv_trans_gu(CA.getdata(ϕt))        
#     get_ca_int_PMs = Val(Symbol("deprecated , use get_int_PMst_site(HybridProblemInterpreters(prob; scenario))"))
#     # get_ca_int_PMs = let
#     #     function get_ca_int_PMs_inner(n_site)
#     #         ComponentArrayInterpreter(CA.ComponentVector(; P = θP,
#     #             Ms = CA.ComponentMatrix(
#     #                 zeros(n_θM, n_site), first(CA.getaxes(θM)), CA.Shaped1DAxis((n_site,)))))
#     #     end
#     # end
#     interpreters = map(get_concrete,
#         (;
#             ϕg_ϕq = ComponentArrayInterpreter(ϕ),
#             PMs = get_int_PMst_batch(hpints),
#             ϕq = ComponentArrayInterpreter(ϕq)
#         ))
#     (; ϕ, transPMs_batch, interpreters, get_transPMs, get_ca_int_PMs)
# end

"""
    init_hybrid_ϕunc(approx::AbstractHVIApproximation, cor_ends, ρ0=0f0; 
      logσ2_ζP, coef_logσ2_ζMs, ρsP, ρsM)

Initialize vector of additional parameter of the approximate posterior.

Arguments:
- `approx`: AbstractMeanHVIApproximation, which is used. Parametrization will
  differ depending on the approximation.
- `cor_ends`: NamedTuple with entries, `P`, and `M`, respectively with 
   integer vectors of ending columns of parameters blocks
- `ρ0`: default entry for ρsP and ρsM, defaults = 0f0.
- `coef_logσ2_logM`: default column for `coef_logσ2_ζMs`, defaults to `[-10.0, 0.0]`

Returns a `ComponentVector` of 
- `logσ2_ζP`: vector of log-variances of ζP (on log scale).
  defaults to -10
- `coef_logσ2_ζMs`: offset and slope for the log-variances of ζM scaling with 
   its value given by columns for each parameter in ζM, defaults to `[-10, 0]`
- `ρsP` and `ρsM`: parameterization of the upper triangular cholesky factor 
  of the correlation matrices of ζP and ζM, default to all entries `ρ0`, which defaults to zero.
"""
function init_hybrid_ϕunc(
        approx::AbstractMeanHVIApproximation,
        cor_ends::NamedTuple,
        ρ0::FT = 0.0f0,
        coef_logσ2_logM::AbstractVector{FT} = FT[-10.0, 0.0];
        logσ2_ζP::AbstractVector{FT} = fill(FT(-10.0), cor_ends.P[end]),
        coef_logσ2_ζMs::AbstractMatrix{FT} = reduce(
            hcat, (coef_logσ2_logM for _ in 1:cor_ends.M[end])),
        ρsP = fill(ρ0, get_cor_count(cor_ends.P)),
        ρsM = fill(ρ0, get_cor_count(cor_ends.M)),
        θM::CA.ComponentVector,
        n_site::Integer = 0,
) where {FT}
    nt = (;
        logσ2_ζP,
        coef_logσ2_ζMs,
        ρsP,
        ρsM)
    ca = CA.ComponentVector(;nt...)::CA.ComponentVector
end

function init_hybrid_ϕunc(
        approx::AbstractMeanVarSepHVIApproximation,
        cor_ends::NamedTuple,
        ρ0::FT = 0.0f0,
        logσ2_ζMs::AbstractMatrix{FT} = Array{FT}(undef, 0, 0),
        logσ2_ζP::AbstractVector{FT} = fill(FT(-10.0), cor_ends.P[end]),
        ρsP = fill(ρ0, get_cor_count(cor_ends.P)),
        ρsM = fill(ρ0, get_cor_count(cor_ends.M));
        θM::CA.ComponentVector,
        n_site::Integer,
        relerr = 0.01,
) where {FT}
    logσ2_ζMs = if isempty(logσ2_ζMs) 
        # sigma is the relative error of the template of θM
        repeat(FT(2) * log.(convert(FT,relerr) .* CA.getdata(θM)), 1, n_site)
    else
        logσ2_ζMs
    end
    nt = (;
        logσ2_ζP,
        logσ2_ζMs,
        ρsP,
        ρsM)
    ca = CA.ComponentVector(;nt...)::CA.ComponentVector
end

# macro gen_unc(nt)
#     quote
#         nt_ev = $(esc(nt))
#         int_nt = StaticComponentArrayInterpreter(map(x -> Val(size(x)), nt_ev))
#         int_nt(CA.getdata(CA.ComponentVector(;nt_ev...)))
#     end
# end

