struct PBMPopulationApplicator{AXS, AXP, MFT, MFTD, F} <: AbstractPBMApplicator 
    fθpop!::F
    θFixm::MFT 
    θFixm_dc::MFTD 
    axθcomp::Val{AXS} # 3-tuple of axes of θP, θM and θFix
    ax_xP::Val{AXP}
end

# let fmap not descend into isP, because indexing with isP on cpu is faster
@functor PBMPopulationApplicator (θFixm, rep_fac)

"""
    PBMPopulationApplicator(fθpop!, n_site; θP, θM, θFix, xPvec)

Construct AbstractPBMApplicator from process-based model `fθ` that computes predictions
across `n_MC` parameterizations.
The applicator combines enclosed `θFix`, with provided `θsM` and `θsP`
to a `ComponentMatrix` with parameters with one column for each parameterization, that
can be column-indexed by Symbols.

## Arguments 
- `fθpop!`: process model, process model `f!(pred, θc, xPc)`, which is agnostic of the partitioning
   of parameters into fixed, global, and individual.
    - `θc`: parameters: `ComponentMatrix` (n_par x n_MC) with each row a parameter vector
    - `xPc`: observations: `ComponentVector (n_obs) with observationsfor one site
- `n_MC`: number of parameterizations, i.e. colums in `θsM`
- `θP`: `ComponentVector` template of global process model parameters
- `θM`: `ComponentVector` template of individual process model parameters
- `θFix`: `ComponentVector` of actual fixed process model parameters
- `xPvec`: `ComponentVector` template of model drivers for a single site
"""
function PBMPopulationApplicator(fθpop!, n_MC:: Integer; 
    θP::CA.ComponentVector{FT}, θM::CA.ComponentVector, θFix::CA.ComponentVector, 
    xPvec::CA.ComponentVector,
    diffchunk::ForwardDiff.Chunk{chunk} = ForwardDiff.Chunk(8),
    ) where {FT,chunk}
    is_MC = ones(FT, n_MC)'
    # θPm = CA.ComponentMatrix(CA.getdata(θP) .* is_MC, (CA.getaxes(θP)[1], CA.FlatAxis()))
    # θMm = CA.ComponentMatrix(CA.getdata(θM) .* is_MC, (CA.getaxes(θM)[1], CA.FlatAxis()))
    θFixm = convert.(FT,CA.getdata(θFix)) .* is_MC
    θFixm_dc = PAT.DiffCache(θFixm, chunk)
    # θ = vcat(vcat(θPm, θMm), θFixm)
    axθ = (CA.getaxes(θP)[1], CA.getaxes(θM)[1], CA.getaxes(θFix)[1])
    ax_xP = CA.getaxes(xPvec)[1]
    PBMPopulationApplicator(fθpop!, θFixm, θFixm_dc, Val(axθ), Val(ax_xP))        
end

get_axis_θP(::PBMPopulationApplicator{AX}) where AX = AX[1]
get_axis_θM(::PBMPopulationApplicator{AX}) where AX = AX[2]

function create_nsite_applicator(app::PBMPopulationApplicator, n_MC;
    diffchunk::ForwardDiff.Chunk{chunk} = ForwardDiff.Chunk(8),
    ) where chunk
    is_MC = ones(eltype(app.θFixm), n_MC)'
    θFixm = CA.getdata(app.θFixm[:,1]) .* is_MC
    θFixm_dc = PAT.DiffCache(θFixm, chunk)
    PBMPopulationApplicator(app.fθpop!, θFixm, θFixm_dc, app.axθcomp, app.ax_xP)        
end

function apply_model!(pred, app::PBMPopulationApplicator{AXS,AXP}, θsP::AbstractMatrix, θsM::AbstractMatrix, xP) where {AXS,AXP}
    # function tmpf(θsP, θsM, app) 
    #     @allocated Vcat(CA.getdata(θsP), CA.getdata(θsM), CA.getdata(app.θFixm))
    # end
    # function tmpf2(θsP, θsM, app) 
    #     @allocated vcat(CA.getdata(θsP), CA.getdata(θsM), CA.getdata(app.θFixm))
    # end
    # Main.@infiltrate_main
    # tmpf(θsP, θsM, app)
    # tmpf2(θsP, θsM, app)
    #
    # local data_view = Vcat(CA.getdata(θsP), CA.getdata(θsM), CA.getdata(app.θFixm))  # lazy, no copy!
    # local θc = CA.ComponentArray(data_view, app.axθ)
    local θc, xPc, θFixm_dual
    # Symbol access into Vcat allocates in Gradient, roll own VcatCAs
    # θc = CA.ComponentMatrix(
    #     Vcat(CA.getdata(θsP), CA.getdata(θsM), CA.getdata(app.θFixm)),
    #     AXC, CA.FlatAxis())
    # TODO provide PreallocationTools for app.θFixm
    #θc = VcatCAs(AXS, CA.getdata(θsP), CA.getdata(θsM), CA.getdata(app.θFixm))
    θFixm_dual = if eltype(θsP) isa ForwardDiff.Dual
        copyto!(PAT.get_tmp(app.θFixm_dc, CA.getdata(θsP)), app.θFixm)
    else
        app.θFixm
    end
    θc = VcatCAs(AXS, CA.getdata(θsP), CA.getdata(θsM), θFixm_dual)
    #xPc = app.int_xP(CA.getdata(xP))
    #xPc = CA.ComponentArray(CA.getdata(xP), AXP, θc.flataxes...)
    xPc = CA.ComponentVector(CA.getdata(xP), AXP)
    app.fθpop!(pred, θc, xPc)  
    #app.fθpop!(pred, θc, xP)  
    pred
end



