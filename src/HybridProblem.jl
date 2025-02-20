struct HybridProblem <: AbstractHybridProblem
    θP
    θM
    f
    g
    ϕg
    py
    transP
    transM
    cor_ends # = (P=(1,),M=(1,))
    get_train_loader
    # inner constructor to constrain the types
    function HybridProblem(
            θP::CA.ComponentVector, θM::CA.ComponentVector,
            g::AbstractModelApplicator, ϕg::AbstractVector,
            f::Function,
            py::Function,
            transM::Union{Function, Bijectors.Transform},
            transP::Union{Function, Bijectors.Transform},
            #train_loader::DataLoader,
            # return a function that constructs the trainloader based on n_batch
            get_train_loader::Function,
            cor_ends::NamedTuple = (P = [length(θP)], M = [length(θM)]))
        new(θP, θM, f, g, ϕg, py, transM, transP, cor_ends, get_train_loader)
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
    f = get_hybridproblem_PBmodel(prob; scenario)
    py = get_hybridproblem_neg_logden_obs(prob; scenario)
    (; transP, transM) = get_hybridproblem_transforms(prob; scenario)
    get_train_loader = let prob = prob, scenario = scenario
        function inner_get_train_loader(rng::AbstractRNG; kwargs...)
            get_hybridproblem_train_dataloader(rng::AbstractRNG, prob; scenario, kwargs...)
        end
    end
    cor_ends = get_hybridproblem_cor_ends(prob; scenario)
    HybridProblem(θP, θM, g, ϕg, f, py, transP, transM, get_train_loader, cor_ends)
end

function update(prob::HybridProblem;
        θP::CA.ComponentVector = prob.θP,
        θM::CA.ComponentVector = prob.θM,
        g::AbstractModelApplicator = prob.g, ϕg::AbstractVector = prob.ϕg,
        f::Function = prob.f,
        py::Function = prob.py,
        transM::Union{Function, Bijectors.Transform} = prob.transM,
        transP::Union{Function, Bijectors.Transform} = prob.transP,
        get_train_loader::Function = prob.get_train_loader,
        cor_ends::NamedTuple = prob.cor_ends)
    # prob.θP = θP
    # prob.θM = θM
    # prob.f = f
    # prob.g = g
    # prob.ϕg = ϕg
    # prob.py = py
    # prob.transM = transM
    # prob.transP = transP
    # prob.cor_ends = cor_ends
    # prob.get_train_loader = get_train_loader
    HybridProblem(θP, θM, g, ϕg, f, py, transP, transM, get_train_loader, cor_ends)
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

function get_hybridproblem_train_dataloader(rng::AbstractRNG, prob::HybridProblem; kwargs...)
    return prob.get_train_loader(rng; kwargs...)
end

function get_hybridproblem_cor_ends(prob::HybridProblem; scenario = ())
    prob.cor_ends
end

# function get_hybridproblem_float_type(prob::HybridProblem; scenario::NTuple = ()) 
#     eltype(prob.θM)
# end
