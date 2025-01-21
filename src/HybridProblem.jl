struct HybridProblem <: AbstractHybridCase 
    θP
    θM
    transP
    transM
    n_covar
    n_batch
    f
    g
    ϕg
    train_loader
    # inner constructor to constrain the types
    function HybridProblem(
        θP::CA.ComponentVector, θM::CA.ComponentVector, 
        g::AbstractModelApplicator, ϕg, 
        f::Function, 
        transM::Union{Function, Bijectors.Transform}, 
        transP::Union{Function, Bijectors.Transform}, 
        n_covar::Integer, n_batch::Integer, 
        train_loader::DataLoader)
        new(θP, θM, transM, transP, n_covar, n_batch, f, g, ϕg, train_loader)
    end
end

function get_hybridcase_par_templates(prob::HybridProblem; scenario::NTuple = ())
    (; θP = prob.θP, θM = prob.θM)
end

function get_hybridcase_sizes(prob::HybridProblem; scenario::NTuple = ())
    n_θM = length(prob.θM)
    n_θP = length(prob.θP)
    (; n_covar=prob.n_covar, n_batch=prob.n_batch, n_θM, n_θP)
end

function get_hybridcase_PBmodel(prob::HybridProblem; scenario::NTuple = ())
    prob.f
end

function get_hybridcase_MLapplicator(prob::HybridProblem, ml_engine; scenario::NTuple = ());
    prob.g, prob.ϕg
end

function get_hybridcase_train_dataloader(
    prob::HybridProblem, rng::AbstractRNG = Random.default_rng(); 
    scenario = ())
    return(prob.train_loader)
end


# function get_hybridcase_FloatType(prob::HybridProblem; scenario::NTuple = ()) 
#     eltype(prob.θM)
# end



