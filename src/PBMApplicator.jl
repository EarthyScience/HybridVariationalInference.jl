"""
Abstraction of applying a process-based model with 
global parameters, `θP`, site-specific parameters, `θMs` (sites in columns), 
and site-specific model drivers, `xP` (sites in columns),
It returns a matrix of predictions sites in columns.    

Specific implementations need to provide function `apply_model(app, θP, θMs, xP)`.
where
- `θsP` and `θsMs` are shaped according to the output of `generate_ζ`, i.e.
  `(n_site_pred x n_par x n_MC)`.
- Results are of shape `(n_obs x n_site_pred x n_MC)`.

They may also provide function `apply_model(app, θP, θMs, xP)` for a sample
of parameters, i.e. where an additional dimension is added to both `θP` and `θMs`.
However, there is a default implementation that mapreduces across these dimensions.

Provided are implementations
- `PBMSiteApplicator`: based on a function that computes predictions per site
- `PBMPopulationApplicator`: based on a function that computes predictions for entire population
- `NullPBMApplicator`: returning its input `θMs` for testing
- `DirectPBMApplicator`: based on a function that takes the same arguments as `apply_model`
"""
abstract type AbstractPBMApplicator end

# function apply_model end  # already defined in ModelApplicator.jl for ML model

function (app::AbstractPBMApplicator)(θP::AbstractArray, θMs::AbstractArray, xP::AbstractMatrix) 
    apply_model(app, θP, θMs, xP)
end

"""
    apply_model(app::AbstractPBMApplicator, θsP::AbstractVector, θsMs::AbstractMatrix, xP::AbstractMatrix) 
    apply_model(app::AbstractPBMApplicator, θsP::AbstractMatrix, θsMs::AbstractArray{ET,3}, xP) 

The first variant calls the PBM for one batch of sites.

The second variant calls the PBM for a sample of batches, and stack results.
The default implementation mapreduces the last dimension of `θsP` and θ`sMs` calling the 
first variant of `apply_model` for each sample.
"""
# docu in struct
function apply_model(app::AbstractPBMApplicator, θsP::AbstractMatrix, θsMs::AbstractArray{ET,3}, xP) where ET
    # stack does not work on GPU, see specialized method for GPUArrays below
    y_pred = stack(
     map(eachcol(CA.getdata(θsP)), eachslice(CA.getdata(θsMs), dims=3)) do θP, θMs
        app(θP, θMs, xP)
    end)
end
# function apply_model(app::AbstractPBMApplicator, θsP::GPUArraysCore.AbstractGPUMatrix, θsMs::GPUArraysCore.AbstractGPUArray{ET,3}, xP) where ET
#     # stack does not work on GPU, need to resort to slower mapreduce
#     # for type stability, apply f at first iterate to supply init to mapreduce
#     P1, Pit = Iterators.peel(eachcol(CA.getdata(θsP)));
#     Ms1, Msit = Iterators.peel(eachslice(CA.getdata(θsMs), dims=3));
#     y1 = apply_model(app, P1, Ms1, xP)[2]
#     y1a = reshape(y1, size(y1)..., 1) # add one dimension
#     y_pred = mapreduce((a,b) -> cat(a,b; dims=3), Pit, Msit; init=y1a) do θP, θMs
#         y_pred_i = app(θP, θMs, xP)
#     end
# end
function apply_model(app::AbstractPBMApplicator, θsP::GPUArraysCore.AbstractGPUMatrix, θsMs::GPUArraysCore.AbstractGPUArray{ET,3}, xP) where ET
    # stack does not work on GPU, need to resort to slower mapreduce
    # for type stability, apply f at first iterate to supply init to mapreduce
    # avoid Iterators.peel for CUDA
    y1 = apply_model(app, CA.getdata(θsP)[:,1], CA.getdata(θsMs)[:,:,1], xP)[2]
    y1a = reshape(y1, :, 1) # add one dimension
    n_sample = size(θsP,2)
    y_pred = if (n_sample == 1)
        y1a
    else
      mapreduce((a,b) -> cat(a,b; dims=3), 
        eachcol(CA.getdata(θsP)[:,2:end]), eachslice(CA.getdata(θsMs)[:,:,2:end], dims=3); 
        init=y1a) do θP, θMs
            app(θP, θMs, xP)
        end
    end
    return(y_pred)
end




"""
    NullPBMApplicator()

Process-Base-Model applicator that returns its θMs inputs. Used for testing.
"""
struct NullPBMApplicator <: AbstractPBMApplicator end

function apply_model(app::NullPBMApplicator, θP::AbstractVector, θMs::AbstractMatrix, xP)
    return CA.getdata(θMs)
end

"""
    DirectPBMApplicator()

Process-based-Model applicator that invokes directly given 
function `f(θP::AbstractVector, θMs::AbstractMatrix, xP)`.
"""
struct DirectPBMApplicator{F} <: AbstractPBMApplicator 
    f::F
end

function apply_model(app::DirectPBMApplicator, θP::AbstractVector, θMs::AbstractMatrix, xP)
    return app.f(θP, θMs, xP)
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
symbolic parameter names, corresponding to the templates provided during
construction of the applicator.

## Arguments 
- `fθ`: process model, process model `fθ(θc, xP)`, which is agnostic of the partitioning
of parameters.
- `θP`: `ComponentVector` template of global process model parameters
- `θM`: `ComponentVector` template of individual process model parameters
- `θFix`: `ComponentVector` of actual fixed process model parameters
- `xPvec`:`ComponentVector` template of model drivers for a single site
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
    return pred_sites
end

struct PBMPopulationApplicator{MFT, RFT, IT, IXT, F} <: AbstractPBMApplicator 
    fθpop::F
    θFixm::MFT # may be CuMatrix rather than Matrix
    #isP::IPT #Matrix{Int} # transferred to CuMatrix?
    rep_fac::RFT
    intθ::IT 
    int_xP::IXT
end

# let fmap not descend into isP, because indexing with isP on cpu is faster
@functor PBMPopulationApplicator (θFixm, rep_fac)

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
    #isP = repeat(axes(θP, 1)', n_batch)
    # n_site = size(θMs, 1)
    rep_fac = ones_similar_x(θP, n_batch) # to reshape into matrix, avoiding repeat
    θFixm = CA.getdata(θFix[isFix])
    PBMPopulationApplicator(fθpop, θFixm, rep_fac, intθ, int_xP)        
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
    # repeat is 2x slower for Vector and 100 times slower (with allocation) on GPU
    # app.isP on CPU is slightly faster than app.isP on GPU
    # multiplication has one more allocation on CPU and same speed, but 5x faster on GPU
    #@benchmark CA.getdata(θP[app.isP])  
    #@benchmark CA.getdata(repeat(θP', size(θMs,1))) 
    #@benchmark rep_fac .* CA.getdata(θP)'  # 
    local θ = hcat(app.rep_fac .* CA.getdata(θP)', CA.getdata(θMs), app.θFixm) 
    #local θ = hcat(CA.getdata(θP[app.isP]), CA.getdata(θMs), app.θFixm)
    #local θ = hcat(CA.getdata(repeat(θP', size(θMs,1))), CA.getdata(θMs), app.θFixm)
    local θc = app.intθ(CA.getdata(θ))
    local xPc = app.int_xP(CA.getdata(xP))
    local pred_sites = app.fθpop(θc, xPc)
    return pred_sites
end

