"""
    init_hybrid_params(θP, θM, ϕg, n_batch; transP=asℝ, transM=asℝ)

Setup ComponentVector of parameters to optimize, and associated tools.
Returns a NamedTuple of
- ϕ: A ComponentVector of parameters to optimize
- transPMs_batch, interpreters: Transformations and interpreters as 
  required by `neg_elbo_transnorm_gf`.
- get_transPMs: a function returning transformations `(n_site) -> (;P,Ms)`
- get_ca_int_PMs: a function returning ComponentArrayInterpreter for PMs vector 
  with PMs shaped as a matrix of `n_site` columns of `θM`

# Arguments
- `θP`, `θM`: Template ComponentVectors of global parameters and ML-predicted parameters
- `ϕg`: vector of parameters to optimize, as returned by `get_hybridcase_MLapplicator`
- `n_batch`: the number of sites to predicted in each mini-batch
- `transP`, `transM`: the Bijector.Transformations for the global and site-dependent 
    parameters, e.g. `Stacked(elementwise(identity), elementwise(exp), elementwise(exp))`.
    Its the transformation froing from unconstrained to constrained space: θ = Tinv(ζ),
    because this direction is used much more often.
"""
function init_hybrid_params(θP, θM, ϕg, n_batch; 
    transP=elementwise(identity), transM=elementwise(identity))
    n_θP = length(θP)
    n_θM = length(θM)
    n_ϕg = length(ϕg)
    # check translating parameters - can match length?
    _ = Bijectors.inverse(transP)(θP)
    _ = Bijectors.inverse(transM)(θM)
    # zero correlation matrices
    ρsP = zeros(sum(1:(n_θP - 1)))
    ρsM = zeros(sum(1:(n_θM - 1)))
    ϕunc0 = CA.ComponentVector(;
        logσ2_logP = fill(-10.0, n_θP),
        coef_logσ2_logMs = reduce(hcat, ([-10.0, 0.0] for _ in 1:n_θM)),
        ρsP,
        ρsM)
    ϕ = CA.ComponentVector(;
        μP = inverse(transP)(θP),
        ϕg = ϕg,
        unc = ϕunc0);
    #
    get_transPMs = let transP=transP, transM=transM, n_θP=n_θP, n_θM=n_θM 
        function get_transPMs_inner(n_site)
            transMs = ntuple(i -> transM, n_site)
            ranges = vcat([1:n_θP], [(n_θP + i0*n_θM) .+ (1:n_θM) for i0 in 0:(n_site-1)])
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
            ComponentArrayInterpreter(CA.ComponentVector(; P=θP,
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
    (;ϕ, transPMs_batch, interpreters, get_transPMs, get_ca_int_PMs)
end

