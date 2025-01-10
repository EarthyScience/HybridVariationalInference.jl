"""
Cost function (ELBO) for hybrid model with batched sites.

It generates n_MC samples for each site, and uses these to compute the
expected value of the likelihood of observations.

## Arguments
- rng: random number generator (ignored on CUDA, if ϕ is a AbstractGPUArray)
- g: machine learning model
- f: mechanistic model
- ϕ: flat vector of parameters 
  including parameter of f (ϕ_P), of g (ϕ_Ms), and of VI (ϕ_unc),
  interpreted by interpreters.μP_ϕg_unc and interpreters.PMs
- y_ob: matrix of observations (n_obs x n_site_batch)
- x: matrix of covariates (n_cov x n_site_batch)
- transPMs: Transformations with components P, Ms, similar to interpreters
- n_MC: number of MonteCarlo samples from the distribution of parameters to simulate
  using the mechanistic model f.
- logσ2y: observation uncertainty (log of the variance)
"""
function neg_elbo_transnorm_gf(rng, g, f, ϕ::AbstractVector, y_ob, x::AbstractMatrix,
    transPMs, interpreters::NamedTuple; 
    n_MC=3, logσ2y, gpu_data_handler = get_default_GPUHandler())
    ζs, logdetΣ = generate_ζ(rng, g, f, ϕ, x, interpreters; n_MC)
    ζs_cpu = gpu_data_handler(ζs) # differentiable fetch to CPU in Flux package extension
    #ζi = first(eachcol(ζs_cpu))
    nLy = reduce(+, map(eachcol(ζs_cpu)) do ζi
        y_pred_i, logjac = predict_y(ζi, f, transPMs)
        nLy1 = neg_logden_indep_normal(y_ob, y_pred_i, logσ2y)
        nLy1 - logjac
    end) / n_MC
    ent = entropy_MvNormal(size(ζs, 1), logdetΣ)  # defined in logden_normal
    nLy - ent
end

"""
    predict_gf(rng, g, f, ϕ::AbstractVector, xM::AbstractMatrix, interpreters;
        get_transPMs, get_ca_int_PMs, n_sample_pred=200, 
        gpu_data_handler=get_default_GPUHandler())

Prediction function for hybrid model. Returns an Array `(n_obs, n_site, n_sample_pred)`.
"""
function predict_gf(rng, g, f, ϕ::AbstractVector, xM::AbstractMatrix, interpreters;
    get_transPMs, get_ca_int_PMs, n_sample_pred=200, 
    gpu_data_handler=get_default_GPUHandler())
    n_site = size(xM, 2)
    intm_PMs_gen = get_ca_int_PMs(n_site)
    tans_PMs_gen = get_transPMs(n_site)
    ζs, _ = generate_ζ(rng, g, f, CA.getdata(ϕ), CA.getdata(xM),
    (; interpreters..., PMs = intm_PMs_gen); n_MC = n_sample_pred)
    ζs_cpu = gpu_data_handler(ζs) #
    y_pred = stack(map(ζ -> first(predict_y(ζ, f, tans_PMs_gen)), eachcol(ζs_cpu)));
    y_pred
end

"""
Generate samples of (inv-transformed) model parameters, ζ, and Log-Determinant
of their distribution.

Adds the MV-normally distributed residuals, retrieved by `sample_ζ_norm0`
to the means extracted from parameters and predicted by the machine learning
model. 
"""
function generate_ζ(rng, g, f, ϕ::AbstractVector, x::AbstractMatrix,
    interpreters::NamedTuple; n_MC=3)
    # see documentation of neg_elbo_transnorm_gf
    ϕc = interpreters.μP_ϕg_unc(CA.getdata(ϕ))
    μ_ζP = ϕc.μP
    ϕg = ϕc.ϕg
    μ_ζMs0 = g(x, ϕg) # TODO provide μ_ζP to g
    ζ_resid, logdetΣ = sample_ζ_norm0(rng, μ_ζP, μ_ζMs0, ϕc.unc; n_MC)
    #ζ_resid, logdetΣ = sample_ζ_norm0(rng, ϕ[1:2], reshape(ϕ[2 .+ (1:20)],2,:), ϕ[(end-length(interpreters.unc)+1):end], interpreters.unc; n_MC)
    ζ = stack(map(eachcol(ζ_resid)) do r
        rc = interpreters.PMs(r)
        ζP = μ_ζP .+ rc.θP
        μ_ζMs = μ_ζMs0 # g(x, ϕc.ϕ) # TODO provide ζP to g
        ζMs = μ_ζMs .+ rc.θMs
        vcat(ζP, vec(ζMs))
    end)
    ζ, logdetΣ
end

"""
Extract relevant parameters from θ and return n_MC generated draws
together with the logdet of the transformation.

Necessary typestable information on number of compponents are provided with 
ComponentMarshellers
- marsh_pmu(n_θP, n_θMs, Unc=n_θUnc) 
- marsh_batch(n_batch) 
- marsh_unc(n_UncP, n_UncM, n_UncCorr)
"""
function sample_ζ_norm0(rng::Random.AbstractRNG, ζP::AbstractVector, ζMs::AbstractMatrix, ϕunc::AbstractVector, args...;
    n_MC=3) 
    n_θP, n_θMs = length(ζP), length(ζMs)
    urand = _create_random(rng, CA.getdata(ζP), n_θP + n_θMs, n_MC)
    sample_ζ_norm0(urand, ζP, ζMs, ϕunc, args...)
end

function sample_ζ_norm0(urand::AbstractMatrix, ζP::AbstractVector{T}, ζMs::AbstractMatrix, 
    ϕunc::AbstractVector, int_unc = ComponentArrayInterpreter(ϕunc);
    ) where {T}
    ϕuncc = int_unc(CA.getdata(ϕunc))
    n_θP, n_θMs, (n_θM, n_batch) = length(ζP), length(ζMs), size(ζMs) 
    # make sure to not create a UpperTriangular Matrix of an CuArray in transformU_cholesky1
    UP = transformU_cholesky1(ϕuncc.ρsP)
    UM = transformU_cholesky1(ϕuncc.ρsM)
    cf = ϕuncc.coef_logσ2_logMs
    logσ2_logMs = vec(cf[1, :] .+ cf[2, :] .* ζMs)
    logσ2_logP = vec(CA.getdata(ϕuncc.logσ2_logP))
    # CUDA cannot multiply BlockDiagonal * Diagonal, construct already those blocks
    σMs = reshape(exp.(logσ2_logMs ./ 2), n_θM, :)
    σP = exp.(logσ2_logP ./ 2)
    # BlockDiagonal does work with CUDA, but not with combination of Zygote and CUDA
    # need to construct full matrix for CUDA
    Uσ = _create_blockdiag(UP, UM, σP, σMs, n_batch) 
    ζ_resid = Uσ' * urand
    logdetΣ = 2 .* sum(log.(diag(Uσ)))
    # returns CuArrays to either continue on GPU or need to transfer to CPU
    ζ_resid, logdetΣ
end

function _create_blockdiag(UP::AbstractMatrix{T}, UM, σP, σMs, n_batch) where {T}
    v = [i == 0 ? UP * Diagonal(σP) : UM * Diagonal(σMs[:, i]) for i in 0:n_batch]
    BlockDiagonal(v)
end
function _create_blockdiag(UP::GPUArraysCore.AbstractGPUMatrix{T}, UM, σP, σMs, n_batch) where {T}
    # using BlockDiagonal leads to Scalar operations downstream
    # v = [i == 0 ? UP * Diagonal(σP) : UM * Diagonal(σMs[:, i]) for i in 0:n_batch]
    # BlockDiagonal(v)    
    # Uσ = cat([i == 0 ? UP * Diagonal(σP) : UM * Diagonal(σMs[:, i]) for i in 0:n_batch]...;
    #     dims=(1, 2))
    # on GPU use only one big multiplication rather than many small ones
    U = cat([i == 0 ? UP : UM for i in 0:n_batch]...; dims=(1, 2))
    #Main.@infiltrate_main
    σD = Diagonal(vcat(σP, vec(σMs)))
    Uσ = U * σD
    # need for Zygote why?
    # tmp = cat(Uσ; dims=(1,2))
    tmp = vcat(Uσ)
end

function _create_random(rng, ::AbstractVector{T}, dims...) where {T}
    rand(rng, T, dims...)
end
function _create_random(rng, ::GPUArraysCore.AbstractGPUVector{T}, dims...) where {T}
    # ignores rng
    # https://discourse.julialang.org/t/help-using-cuda-zygote-and-random-numbers/123458/4?u=bgctw
    # Zygote.@ignore CUDA.randn(rng, dims...)
    ChainRulesCore.@ignore_derivatives CUDA.randn(dims...)
end

""" 
Compute predictions and log-Determinant of the transformation at given
transformed parameters for each site. 

The number of sites is given by the number of columns in `Ms`, which is determined
by the transformation, `transPMs`.

Steps:
- transform the parameters to original constrained space
- Applies the mechanistic model for each site
"""
function predict_y(ζi, f, transPMs)
    θtup, logjac = transform_and_logjac(transPMs, ζi) # both allocating
    θc = CA.ComponentVector(θtup)
    # TODO provide xP
    xP = fill((), size(θc.Ms,2))
    y_pred_global, y_pred = f(θc.P, θc.Ms, xP) # TODO parallelize on CPU
    # TODO take care of y_pred_global
    y_pred, logjac
end
