"""
    AbstractPBMApplicator(θP::AbstractVector, θMs::AbstractMatrix, xP::AbstractMatrix)

Abstraction of applying a process-based model with 
global parameters, `x`, site-specific parameters, `θMs` (sites in columns), 
and site-specific model drivers, `xP` (sites in columns),
It returns a matrix of predictions sites in columns.    

Specific implementations need to implement function `apply_model(app, θP, θMs, xP)`.
Provided are implementations
- `NullPBMApplicator`: returning its input `θMs` for testing
- `PBMSiteApplicator`: based on a function that computes predictions per site
- `PBMPopulationApplicator`: based on a function that computes predictions for entire population
"""
abstract type AbstractPBMApplicator end

# function apply_model end  # already defined in ModelApplicator.jl for ML model

function (app::AbstractPBMApplicator)(θP::AbstractVector, θMs::AbstractMatrix, xP::AbstractMatrix) 
    apply_model(app, θP, θMs, xP)
end


"""
    NullPBMApplicator()

Process-Base-Model applicator that returns its θMs inputs. Used for testing.
"""
struct NullPBMApplicator <: AbstractPBMApplicator end

function apply_model(app::NullPBMApplicator, θP::AbstractVector, θMs::AbstractMatrix, xP)
    return CA.getdata(θMs)
end


struct PBMSiteApplicator{F, IT, IXT, VFT} <: AbstractPBMApplicator 
    fθ::F
    intθ1::IT 
    int_xPsite::IXT
    θFix::VFT # can be a CuArray instead of a Vector
end

"""
    PBMSiteApplicator(fθ, n_batch; θP, θM, θFix, xPvec)

Construct AbstractPBMApplicator from process-based model `fθ` that computes predictions
for a single site.
The Applicator combines enclosed `θFix`, with provided `θMs` and `θP` and
constructs a `ComponentVector` that can be indexed by 
symbolic parameter names, correspondning to the templates provided during
construction of the applicator.

## Arguments 
- `fθ`: process model, process model `fθ(θc, xP)`, which is agnostic of the partitioning
of parameters.
- `θP`: ComponentVector template of global process model parameters
- `θM`: ComponentVector template of individual process model parameters
- `θFix`: ComponentVector of actual fixed process model parameters
- `xPvec`::ComponentVector template of model drivers for a single site
"""
function PBMSiteApplicator(fθ; 
    θP::CA.ComponentVector, θM::CA.ComponentVector, θFix::CA.ComponentVector, 
    xPvec::CA.ComponentVector
    )
    intθ1 = get_concrete(ComponentArrayInterpreter(flatten1(CA.ComponentVector(; θP, θM, θFix))))
    int_xPsite = get_concrete(ComponentArrayInterpreter(xPvec))
    PBMSiteApplicator(fθ, intθ1, int_xPsite, CA.getdata(θFix))        
end

function apply_model(app::PBMSiteApplicator, θP::AbstractVector, θMs::AbstractMatrix, xP) 
    function apply_PBMsite(θM, xP1)
        if (CA.getdata(θP) isa GPUArraysCore.AbstractGPUArray) && 
            (!(app.θFix isa GPUArraysCore.AbstractGPUArray) || 
             !(CA.getdata(θMs) isa GPUArraysCore.AbstractGPUArray)) 
            error("concatenating GPUarrays with non-gpu arrays θFix or θMs. " *
            "May fmap PBMModelapplicators to gdev, " *
            "or compute PBMmodel on CPU")
        end
        θ = vcat(CA.getdata(θP), CA.getdata(θM), app.θFix)
        θc = app.intθ1(θ);  # show errors without ";"
        xPc = app.int_xPsite(xP1);
        ans = CA.getdata(app.fθ(θc, xPc))
        ans
    end
    # mapreduce-hcat is only typestable with init, which needs number of rows
    # https://discourse.julialang.org/t/type-instability-of-mapreduce-vs-map-reduce/121136
    # local pred_sites = mapreduce(
    #     apply_PBMsite, hcat, eachrow(θMs), eachcol(xP); init=Matrix{Float64}(undef,n_obs,0))
    θMs1, it_θMs = if (CA.getdata(θP) isa GPUArraysCore.AbstractGPUArray)
        # if working on CuArray, better materialize transpose and use eachcol for contiguous
        #   avoid eachrow, because it does produce non-strided views which are bad on GPU, 
        #   https://discourse.julialang.org/t/using-view-with-cuarrays/104057/5
        # better compute on CPU or use matrix-version of PBMModel
        θMst = copy(CA.getdata(θMs)')
        Iterators.peel(eachcol(θMst));
    else
        Iterators.peel(eachrow(CA.getdata(θMs)))
    end
    xP1, it_xP = Iterators.peel(eachcol(CA.getdata(xP)))
    obs1 = apply_PBMsite(θMs1, xP1)
    local pred_sites = mapreduce(
         apply_PBMsite, hcat, it_θMs, it_xP; init=reshape(obs1, :, 1))
    # # special case of mapreduce producing a vector rather than a matrix
    # pred_sites = !(pred_sites0 isa AbstractMatrix) ? hcat(pred_sites0) : pred_sites0
    #obs1 = apply_PBMsite(first(eachrow(θMs)), first(eachcol(xP)))
    #obs_vecs = map(apply_PBMsite, eachrow(θMs), eachcol(xP))
    #obs_vecs = (apply_PBMsite(θMs1, xP1) for (θMs1, xP1) in zip(eachrow(θMs), eachcol(xP)))
    #pred_sites = stack(obs_vecs; dims = 1)
    #pred_sites = stack(obs_vecs) # does not work with Zygote
    local pred_global = eltype(pred_sites)[] # TODO remove
    return pred_global, pred_sites
end

struct PBMPopulationApplicator{MFT, IPT, IT, IXT, F} <: AbstractPBMApplicator 
    fθpop::F
    θFixm::MFT # may be CuMatrix rather than Matrix
    isP::IPT #Matrix{Int} # transferred to CuMatrix?
    intθ::IT 
    int_xP::IXT
end

# let fmap not descend into isP
# @functor PBMPopulationApplicator (θFixm, )

"""
    PBMPopulationApplicator(fθpop, n_batch; θP, θM, θFix, xPvec)

Construct AbstractPBMApplicator from process-based model `fθ` that computes predictions
across sites for a population of size `n_batch`.
The applicator combines enclosed `θFix`, with provided `θMs` and `θP`
to a `ComponentMatrix` with parameters with one row for each site, that
can be column-indexed by Symbols.

## Arguments 
- `fθpop`: process model, process model `f(θc, xPc)`, which is agnostic of the partitioning
   of parameters into fixed, global, and individual.
    - `θc`: parameters: `ComponentMatrix` (n_site x n_par) with each row a parameter vector
    - `xPc`: observations: `ComponentMatrix` (n_obs x n_site) with each column 
    observationsfor one site
- `n_batch`: number of indiduals, i.e. rows in `θMs`
- `θP`: `ComponentVector` template of global process model parameters
- `θM`: `ComponentVector` template of individual process model parameters
- `θFix`: `ComponentVector` of actual fixed process model parameters
- `xPvec`: `ComponentVector` template of model drivers for a single site
"""
function PBMPopulationApplicator(fθpop, n_batch; 
    θP::CA.ComponentVector, θM::CA.ComponentVector, θFix::CA.ComponentVector, 
    xPvec::CA.ComponentVector
    )
    intθvec = ComponentArrayInterpreter(flatten1(CA.ComponentVector(; θP, θM, θFix)))
    int_xP_vec = ComponentArrayInterpreter(xPvec)
    isFix = repeat(axes(θFix, 1)', n_batch)
    #
    intθ = get_concrete(ComponentArrayInterpreter((n_batch,), intθvec))
    int_xP = get_concrete(ComponentArrayInterpreter(int_xP_vec, (n_batch,)))
    isP = repeat(axes(θP, 1)', n_batch)
    θFixm = CA.getdata(θFix[isFix])
    PBMPopulationApplicator(fθpop, θFixm, isP, intθ, int_xP)        
end

function apply_model(app::PBMPopulationApplicator, θP::AbstractVector, θMs::AbstractMatrix, xP) 
    if (CA.getdata(θP) isa GPUArraysCore.AbstractGPUArray) && 
        (!(app.θFixm isa GPUArraysCore.AbstractGPUArray) || 
            !(CA.getdata(θMs) isa GPUArraysCore.AbstractGPUArray)) 
        error("concatenating GPUarrays with non-gpu arrays θFixm or θMs. " *
        "May transfer PBMPopulationApplicator to gdev, " *
        "or compute PBM on CPU.")
    end
    # repeat θP and concatenate with 
    local θ = hcat(CA.getdata(θP[app.isP]), CA.getdata(θMs), app.θFixm)
    local θc = app.intθ(CA.getdata(θ))
    local xPc = app.int_xP(CA.getdata(xP))
    local pred_sites = app.fθpop(θc, xPc)
    local pred_global = eltype(pred_sites)[] # TODO remove
    return pred_global, pred_sites
end

