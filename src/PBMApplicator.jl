struct PBMPopulationApplicator{AX, MFT, IXT, F} <: AbstractPBMApplicator 
    fθpop::F
    θFixm::MFT 
    axθ::AX
    int_xP::IXT
end

# let fmap not descend into isP, because indexing with isP on cpu is faster
@functor PBMPopulationApplicator (θFixm, rep_fac)

"""
    PBMPopulationApplicator(fθpop, n_site; θP, θM, θFix, xPvec)

Construct AbstractPBMApplicator from process-based model `fθ` that computes predictions
across `n_MC` parameterizations.
The applicator combines enclosed `θFix`, with provided `θsM` and `θsP`
to a `ComponentMatrix` with parameters with one column for each parameterization, that
can be column-indexed by Symbols.

## Arguments 
- `fθpop`: process model, process model `f(θc, xPc)`, which is agnostic of the partitioning
   of parameters into fixed, global, and individual.
    - `θc`: parameters: `ComponentMatrix` (n_par x n_MC) with each row a parameter vector
    - `xPc`: observations: `ComponentVector (n_obs) with observationsfor one site
- `n_MC`: number of parameterizations, i.e. colums in `θsM`
- `θP`: `ComponentVector` template of global process model parameters
- `θM`: `ComponentVector` template of individual process model parameters
- `θFix`: `ComponentVector` of actual fixed process model parameters
- `xPvec`: `ComponentVector` template of model drivers for a single site
"""
function PBMPopulationApplicator(fθpop, n_MC; 
    θP::CA.ComponentVector, θM::CA.ComponentVector, θFix::CA.ComponentVector, 
    xPvec::CA.ComponentVector
    )
    is_MC = ones(n_MC)'
    θPm = CA.ComponentMatrix(CA.getdata(θP) .* is_MC, (CA.getaxes(θP)[1], CA.FlatAxis()))
    θMm = CA.ComponentMatrix(CA.getdata(θM) .* is_MC, (CA.getaxes(θM)[1], CA.FlatAxis()))
    θFixm = CA.ComponentMatrix(CA.getdata(θFix) .* is_MC, (CA.getaxes(θFix)[1], CA.FlatAxis()))
    θ = vcat(vcat(θPm, θMm), θFixm)
    axθ = CA.getaxes(θ)
    int_xP = get_concrete(ComponentArrayInterpreter(xPvec))
    PBMPopulationApplicator(fθpop, θFixm, axθ, int_xP)        
end

function create_nsite_applicator(app::PBMPopulationApplicator, n_site) 
    error("implenment create_nsite_applicator(app::PBMPopulationApplicator")
    # θFix = app.θFixm[1,:]
    # isFix = repeat(axes(θFix, 1)', n_site)
    # θFixm = if length(θFix) == 0
    #     CA.ComponentMatrix(θFix[isFix], (CA.FlatAxis(), CA.FlatAxis()))
    # else
    #     CA.ComponentMatrix(θFix[isFix], (CA.FlatAxis(), CA.getaxes(θFix)[1]))
    # end
    # #
    # intθ = get_concrete(ComponentArrayInterpreter((n_site,), (CA.getaxes(app.intθ)[2],),()))
    # int_xP = get_concrete(ComponentArrayInterpreter(
    #     (), (CA.getaxes(app.int_xP)[1],), (n_site,)))
    # rep_fac = ones_similar_x(θFix, n_site) # to reshape into matrix, avoiding repeat
    # PBMPopulationApplicator(app.fθpop, θFixm, rep_fac, intθ, int_xP)        
end

function apply_model!(pred, app::PBMPopulationApplicator, θsP::AbstractMatrix, θsM::AbstractMatrix, xP) 
    local data_view = Vcat(CA.getdata(θsP), CA.getdata(θsM), CA.getdata(app.θFixm))  # lazy, no copy!
    local θc = CA.ComponentArray(data_view, app.axθ)
    local xPc = app.int_xP(CA.getdata(xP))
    app.fθpop(pred, θc, xPc)
    pred
end



