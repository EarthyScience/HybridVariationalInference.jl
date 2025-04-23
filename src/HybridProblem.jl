struct HybridProblem <: AbstractHybridProblem
    θP::Any
    θM::Any
    f::Any
    g::Any
    ϕg::Any
    ϕunc::Any
    priors::Any
    py::Any
    transP::Any
    transM::Any
    cor_ends::Any # = (P=(1,),M=(1,))
    get_train_loader::Any
    n_covar::Int
    n_site::Int
    pbm_covars::NTuple
    # inner constructor to constrain the types
    function HybridProblem(
            θP::CA.ComponentVector, θM::CA.ComponentVector,
            g::AbstractModelApplicator, ϕg::AbstractVector,
            ϕunc::CA.ComponentVector,
            f::Function,
            priors::AbstractDict,
            py::Function,
            transM::Union{Function, Bijectors.Transform},
            transP::Union{Function, Bijectors.Transform},
            # return a function that constructs the trainloader based on n_batch
            get_train_loader::Function,
            n_covar::Int,
            n_site::Int,
            cor_ends::NamedTuple = (P = [length(θP)], M = [length(θM)]),
            pbm_covars::NTuple{N,Symbol} = (),
    ) where N
        new(
            θP, θM, f, g, ϕg, ϕunc, priors, py, transM, transP, cor_ends, get_train_loader,
            n_covar, n_site, pbm_covars)
    end
end

function HybridProblem(θP::CA.ComponentVector, θM::CA.ComponentVector,
        # note no ϕg argument and g_chain unconstrained
        g_chain, f::Function,
        args...; rng = Random.default_rng(), kwargs...)
    # dispatches on type of g_chain
    g, ϕg = construct_ChainsApplicator(rng, g_chain, eltype(θM))
    HybridProblem(θP, θM, g, ϕg, f, args...; kwargs...)
end

function HybridProblem(prob::AbstractHybridProblem; scenario = ())
    (; θP, θM) = get_hybridproblem_par_templates(prob; scenario)
    g, ϕg = get_hybridproblem_MLapplicator(prob; scenario)
    ϕunc = get_hybridproblem_ϕunc(prob; scenario)
    f = get_hybridproblem_PBmodel(prob; scenario)
    py = get_hybridproblem_neg_logden_obs(prob; scenario)
    (; transP, transM) = get_hybridproblem_transforms(prob; scenario)
    get_train_loader = let prob = prob, scenario = scenario
        function inner_get_train_loader(;kwargs...)
            get_hybridproblem_train_dataloader(prob; scenario, kwargs...)
        end
    end
    cor_ends = get_hybridproblem_cor_ends(prob; scenario)
    pbm_covars = get_hybridproblem_pbmpar_covars(prob; scenario)
    priors = get_hybridproblem_priors(prob; scenario)
    n_covar = get_hybridproblem_n_covar(prob; scenario)
    n_site = get_hybridproblem_n_site(prob; scenario)
    HybridProblem(θP, θM, g, ϕg, ϕunc, f, priors, py, transP, transM, get_train_loader,
        n_covar, n_site, cor_ends, pbm_covars)
end

function update(prob::HybridProblem;
        θP::CA.ComponentVector = prob.θP,
        θM::CA.ComponentVector = prob.θM,
        g::AbstractModelApplicator = prob.g, 
        ϕg::AbstractVector = prob.ϕg,
        ϕunc::CA.ComponentVector = prob.ϕunc,
        f::Function = prob.f,
        priors::AbstractDict = prob.priors,
        py::Function = prob.py,
        transM::Union{Function, Bijectors.Transform} = prob.transM,
        transP::Union{Function, Bijectors.Transform} = prob.transP,
        cor_ends::NamedTuple = prob.cor_ends,
        pbm_covars::NTuple{N,Symbol} = prob.pbm_covars,
        get_train_loader::Function = prob.get_train_loader,
        n_covar::Integer = prob.n_covar,
        n_site::Integer = prob.n_site
) where N
    HybridProblem(θP, θM, g, ϕg, ϕunc, f, priors, py, transP, transM, get_train_loader,
        n_covar, n_site, cor_ends, pbm_covars)
end

function get_hybridproblem_par_templates(prob::HybridProblem; scenario::NTuple = ())
    (; θP = prob.θP, θM = prob.θM)
end

function get_hybridproblem_ϕunc(prob::HybridProblem; scenario::NTuple = ())
    prob.ϕunc
end

function get_hybridproblem_neg_logden_obs(prob::HybridProblem; scenario::NTuple = ())
    prob.py
end

function get_hybridproblem_transforms(prob::HybridProblem; scenario::NTuple = ())
    (; transP = prob.transP, transM = prob.transM)
end

# function get_hybridproblem_sizes(prob::HybridProblem; scenario::NTuple = ())
#     n_θM = length(prob.θM)
#     n_θP = length(prob.θP)
#     (; n_covar=prob.n_covar, n_batch=prob.n_batch, n_θM, n_θP)
# end

function get_hybridproblem_PBmodel(prob::HybridProblem; scenario::NTuple = ())
    prob.f
end

function get_hybridproblem_MLapplicator(prob::HybridProblem; scenario::NTuple = ())
    prob.g, prob.ϕg
end

function get_hybridproblem_train_dataloader(prob::HybridProblem; kwargs...)
    return prob.get_train_loader(;kwargs...)
end

function get_hybridproblem_cor_ends(prob::HybridProblem; scenario = ())
    prob.cor_ends
end
function get_hybridproblem_pbmpar_covars(prob::HybridProblem; scenario = ()) 
    prob.pbm_covars
end
function get_hybridproblem_n_covar(prob::HybridProblem; scenario = ())
    prob.n_covar
end
function get_hybridproblem_n_site(prob::HybridProblem; scenario = ())
    prob.n_site
end

function get_hybridproblem_priors(prob::HybridProblem; scenario = ())
    prob.priors
end

# function get_hybridproblem_float_type(prob::HybridProblem; scenario::NTuple = ()) 
#     eltype(prob.θM)
# end
