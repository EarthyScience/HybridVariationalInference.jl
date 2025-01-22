"""
    AbstractModelApplicator(x, ϕ)

Abstraction of applying a machine learning model at covariate matrix, `x`,
using parameters, `ϕ`. It returns a matrix of predictions with the same
number of rows as in `x`.    

Constructors for specifics are defined in extension packages.
Each constructor takes a special type of machine learning model and returns 
a tuple with two components:
- The applicator 
- a sample parameter vector (type  depends on the used ML-framework)

Implemented are
- `construct_SimpleChainsApplicator`
- `construct_FluxApplicator`
- `construct_LuxApplicator`
"""
abstract type AbstractModelApplicator end

function apply_model end

(app::AbstractModelApplicator)(x, ϕ) = apply_model(app, x, ϕ)

function construct_SimpleChainsApplicator end
function construct_FluxApplicator end
function construct_LuxApplicator end


