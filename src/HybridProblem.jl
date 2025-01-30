struct HybridProblem <: AbstractHybridCase 
    θP
    θM
    f
    g
    ϕg
    py    
    transP
    transM
    cor_starts # = (P=(1,),M=(1,))
    train_loader
    # inner constructor to constrain the types
    function HybridProblem(
        θP::CA.ComponentVector, θM::CA.ComponentVector, 
        g::AbstractModelApplicator, ϕg::AbstractVector, 
        f::Function, 
        py::Function,
        transM::Union{Function, Bijectors.Transform}, 
        transP::Union{Function, Bijectors.Transform}, 
        train_loader::DataLoader,
        cor_starts::NamedTuple = (P=(1,), M=(1,)))
        new(θP, θM, f, g, ϕg, py, transM, transP, cor_starts, train_loader)
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

function get_hybridcase_par_templates(prob::HybridProblem; scenario::NTuple = ())
    (; θP = prob.θP, θM = prob.θM)
end

function get_hybridcase_neg_logden_obs(prob::HybridProblem; scenario::NTuple = ())
    prob.py
end

function get_hybridcase_transforms(prob::HybridProblem; scenario::NTuple = ())
    (; transP = prob.transP, transM = prob.transM)
end

# function get_hybridcase_sizes(prob::HybridProblem; scenario::NTuple = ())
#     n_θM = length(prob.θM)
#     n_θP = length(prob.θP)
#     (; n_covar=prob.n_covar, n_batch=prob.n_batch, n_θM, n_θP)
# end

function get_hybridcase_PBmodel(prob::HybridProblem; scenario::NTuple = ())
    prob.f
end

function get_hybridcase_MLapplicator(prob::HybridProblem; scenario::NTuple = ());
    prob.g, prob.ϕg
end

function get_hybridcase_train_dataloader(rng::AbstractRNG, prob::HybridProblem; scenario = ())
    return(prob.train_loader)
end

function get_hybridcase_cor_starts(prob::HybridProblem; scenario = ())
    prob.cor_starts
end

# function get_hybridcase_float_type(prob::HybridProblem; scenario::NTuple = ()) 
#     eltype(prob.θM)
# end



