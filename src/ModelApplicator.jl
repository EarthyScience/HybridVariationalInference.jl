abstract type AbstractModelApplicator end

function apply_model end

(app::AbstractModelApplicator)(x, ϕ) = apply_model(app, x, ϕ)

function construct_SimpleChainsApplicator end
function construct_FluxApplicator end
function construct_LuxApplicator end


