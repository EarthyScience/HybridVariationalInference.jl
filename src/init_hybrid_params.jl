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
- `ϕg`: vector of parameters to optimize, as returned by `gen_hybridcase_MLapplicator`
- `n_batch`: the number of sites to predicted in each mini-batch
- `transP`, `transM`: the Transformations for the global and site-dependent parameters
"""
function init_hybrid_params(θP, θM, ϕg, n_batch; transP=asℝ, transM=asℝ)
    n_θP = length(θP)
    n_θM = length(θM)
    n_ϕg = length(ϕg)
    # zero correlation matrices
    ρsP = zeros(sum(1:(n_θP - 1)))
    ρsM = zeros(sum(1:(n_θM - 1)))
    ϕunc0 = CA.ComponentVector(;
        logσ2_logP = fill(-10.0, n_θP),
        coef_logσ2_logMs = reduce(hcat, ([-10.0, 0.0] for _ in 1:n_θM)),
        ρsP,
        ρsM)
    ϕt = CA.ComponentVector(;
        μP = θP,
        ϕg = ϕg,
        unc = ϕunc0);
    #
    get_transPMs = let transP=transP, transM=transM, n_θP=n_θP, n_θM=n_θM 
        function get_transPMs_inner(n_site)
            transPMs = as(
                (P = as(Array, transP, n_θP),
                Ms = as(Array, transM, n_θM, n_site)))
        end
    end
    transPMs_batch = get_transPMs(n_batch)
    trans_gu = as(
        (μP = as(Array, asℝ₊, n_θP),
        ϕg = as(Array, n_ϕg),
        unc = as(Array, length(ϕunc0))))
    ϕ = inverse_ca(trans_gu, ϕt)        
    # trans_g = as(
    #     (μP = as(Array, asℝ₊, n_θP),
    #     ϕg = as(Array, n_ϕg)))       
    #
    get_ca_int_PMs = let 
        function get_ca_int_PMs_inner(n_site)
            ComponentArrayInterpreter(CA.ComponentVector(; θP,
            θMs = CA.ComponentMatrix(
                zeros(n_θM, n_site), first(CA.getaxes(θM)), CA.Axis(i = 1:n_site))))
        end
        
    end
    interpreters = map(get_concrete,
    (;
        μP_ϕg_unc = ComponentArrayInterpreter(ϕt),
        PMs = get_ca_int_PMs(n_batch),
        unc = ComponentArrayInterpreter(ϕunc0)
    ))
    (;ϕ, transPMs_batch, interpreters, get_transPMs, get_ca_int_PMs)
end

