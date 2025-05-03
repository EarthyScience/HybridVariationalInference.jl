abstract type AbstractHybridProblemInterpreters end

struct HybridProblemInterpreters{AXP, AXM, NS, NB} <: AbstractHybridProblemInterpreters 
end;

const HPInts = HybridProblemInterpreters

# function get_hybridproblem_statics(prob::AbstractHybridProblem, scenario)
#     θP, θM = get_hybridproblem_par_templates(prob; scenario)
#     NS, NB = get_hybridproblem_n_site_and_batch(prob; scenario)
#     (CA.getaxes(θP), CA.getaxes(θM), NS, NB)
# end

function HybridProblemInterpreters(prob::AbstractHybridProblem; scenario::Val)
    # make sure interred get_hybridproblem_par_templates and n_site_and_n_batch
    # error("'HybridProblemInterpreters(prob::AbstractHybridProblem; scenario)'",
    # "is not inferred at caller level. Replace by ",
    # "'HybridProblemInterpreters{get_hybridproblem_statics(prob; scenario)...}()'")
    θP, θM = get_hybridproblem_par_templates(prob; scenario)
    NS, NB = get_hybridproblem_n_site_and_batch(prob; scenario)
    HybridProblemInterpreters{CA.getaxes(θP), CA.getaxes(θM), NS, NB}()
end

function get_int_P(::HPInts{AXP}) where AXP 
    StaticComponentArrayInterpreter{AXP}()
end
function get_int_M(::HPInts{AXP,AXM}) where {AXP,AXM} 
    StaticComponentArrayInterpreter{AXM}()
end
function get_int_Ms_batch(ints::HPInts{AXP,AXM, NS, NB}) where {AXP,AXM,NS,NB}
    StaticComponentArrayInterpreter(AXM, (NB,))
end
function get_int_Mst_batch(ints::HPInts{AXP,AXM, NS, NB}) where {AXP,AXM,NS,NB}
    StaticComponentArrayInterpreter((NB,), AXM)
end
function get_int_Ms_site(ints::HPInts{AXP,AXM, NS, NB}) where {AXP,AXM,NS,NB}
    StaticComponentArrayInterpreter(AXM, (NS,))
end
function get_int_Mst_site(ints::HPInts{AXP,AXM, NS, NB}) where {AXP,AXM,NS,NB}
    StaticComponentArrayInterpreter((NS,), AXM)
end

function get_int_PMs_batch(ints::HPInts{AXP,AXM, NS, NB}) where {AXP,AXM,NS,NB}
    AX_MS = CA.getaxes(get_int_Ms_batch(ints))
    AX_PMs = combine_axes((;P=AXP, Ms=AX_MS))
    StaticComponentArrayInterpreter{(AX_PMs,)}()
end
function get_int_PMst_batch(ints::HPInts{AXP,AXM, NS, NB}) where {AXP,AXM,NS,NB}
    AX_MS = CA.getaxes(get_int_Mst_batch(ints)) # note the t after Ms
    AX_PMs = combine_axes((;P=AXP, Ms=AX_MS))
    StaticComponentArrayInterpreter{(AX_PMs,)}()
end
function get_int_PMs_site(ints::HPInts{AXP,AXM, NS, NB}) where {AXP,AXM,NS,NB}
    AX_MS = CA.getaxes(get_int_Ms_site(ints))
    AX_PMs = combine_axes((;P=AXP, Ms=AX_MS))
    StaticComponentArrayInterpreter{(AX_PMs,)}()
end
function get_int_PMst_site(ints::HPInts{AXP,AXM, NS, NB}) where {AXP,AXM,NS,NB}
    AX_MS = CA.getaxes(get_int_Mst_site(ints)) # note the t after Ms
    AX_PMs = combine_axes((;P=AXP, Ms=AX_MS))
    StaticComponentArrayInterpreter{(AX_PMs,)}()
end

