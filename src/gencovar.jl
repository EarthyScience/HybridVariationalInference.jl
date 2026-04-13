"""
Generate correlated covariates and synthetic true parameters that
are a linear combination of the uncorrelated underlying principal 
factors and their binary combinations.

In addition provide the true outputs θMs_true = f_true(x_o)
"""
function gen_cov_pred(rng::AbstractRNG, T::DataType,
    n_covar_pc, n_covar, n_site, ζM;
    scenario::Val{scen}, rhodec=8, is_using_dropout=false
    ) where scen
    n_ζM = length(ζM)
    n_sites_cluster, clusters = get_clusters(n_site; scenario)
    if (:clustered_sites ∈ scen)
        # assuming all parameters log-transformed
        # generate clusters of similar parameters and then back-compute the covariates
        ζM_cl_center = map(x -> ζM .+ x, log.(T[0.8, 1.0, 1.2])) # cluster centers
        # standard deviation of samples within cluster, 5% at original scale
        σ = sqrt(log(T(1)+abs2(T(0.05))))  #ζM .* 0.02
        corrM = Matrix(T(1)*I, n_ζM, n_ζM)
        corrM[1:(end-1), 2:end] .= corrM[2:end, 1:(end-1)] .= T(0.7)   # correlated PBM parameters
        # https://math.stackexchange.com/questions/3472602/how-to-create-covariance-matrix-when-correlation-matrix-and-stddevs-are-is-given
        Sigma = PDMat(σ * σ' .* corrM) 
        dist = MultivariateNormal(Sigma) # problems with Zygote
        #Udist = cholesky(Sigma)
        #tmp = rand(rng, dist, 40)
        #Udist.L * randn(n_ζM, 40)
        # draw n_sites_cluster samples around those centers, with some noise
        #Ainv = rand(rng, T, n_covar, n_ζM + sumn(n_ζM-1)) # original + combinations of 2 parameters
        #Ainv = rand(rng, T, n_covar, 1 + sumn(n_ζM-1)) # first original + combinations of 2 parameters
        hcat_ntuples = (t1, t2) -> Tuple(hcat(t1[i], t2[i]) for i in 1:length(t1))
        #i = 1
        xM, ζMs_true = mapreduce(hcat_ntuples, axes(n_sites_cluster,1)) do i
            n_site_cl = n_sites_cluster[i]
            ζMs_true_cl = ζM_cl_center[i] .+ rand(rng, dist, n_site_cl)
            #ζMs_true_cl = ζM_cl_center[i] .+ Udist.L * randn(rng, n_ζM, n_site_cl) #rand(rng, dist, n_site_cl)
            # generate a matrix that contains each combination of the scalar product of rows 
            ζMs_prod = reduce(vcat, transpose.(map(Combinatorics.combinations(1:n_ζM, 2)) do comb
                ζMs_true_cl[comb[1], :] .* ζMs_true_cl[comb[2], :]
            end))
            #xM_true = Ainv * vcat(ζMs_true_cl, ζMs_prod) 
            #xM_true = Ainv * vcat(ζMs_true_cl[1:1,:], ζMs_prod) # only provide the first parameter           
            xM_true = vcat(ζMs_true_cl[1:1,:], ζMs_prod) # only provide the first parameter           
            # need to add noise inputs to match covariate number
            xM_noise = rand(rng, T, n_covar - size(xM_true, 1), n_site_cl)           
            if :exactML ∈ scen
                xM = vcat(xM_true, xM_noise)
            else
                # add some noise to the covariates
                xM = vcat(
                    xM_true .+ (T(0.05)*std(xM_true; dims=2)) .*randn(rng, T, size(xM_true)), 
                    xM_noise)  
            end
            # # generate correlated covariates by scaling the noise around the mean
            # #   does not work for clustered sites
            # xM_mean = mean(xM_true_noise, dims=2)
            # xM_resid_true = xM_true_noise .- xM_mean
            # rhos=vcat(T(1.0), exp.(.-(1:(n_covar-1)) ./ T(rhodec)))
            # #rhoM = repeat(rhos, 1, n_site_cl)
            # noise = std(xM_resid_true, dims=2) .* randn(rng, T, size(xM_resid_true))  # noise to decorrelate
            # xM_resid_cor = rhos .* xM_resid_true .+ (1 .- rhos) .* noise
            # xM = xM_mean .+ xM_resid_cor
            # cor(xM_true_noise[1, :], xM_true_noise[2, :]),  cor(xM_true_noise[1, :], xM_true_noise[3, :]) , cor(xM_true_noise[1, :], xM_true_noise[4, :]) 
            # cor(xM[1, :], xM[2, :]),  cor(xM[1, :], xM[3, :]) , cor(xM[1, :], xM[4, :]) 
            # cor(xM_resid_true[1, :], xM_resid_true[2, :]),  cor(xM_resid_true[1, :], xM_resid_true[3, :]) , cor(xM_resid_true[1, :], xM_resid_true[4, :]) 
            # cor(xM_resid_cor[1, :], xM_resid_cor[2, :]),  cor(xM_resid_cor[1, :], xM_resid_cor[3, :]) , cor(xM_resid_cor[1, :], xM_resid_cor[4, :]) , cor(xM_resid_cor[1, :], xM_resid_cor[5, :]) 
            # hcat_ntuples(t1,t2)
            xM, ζMs_true_cl
        end
    else
        x_pc = rand(rng, T, n_covar_pc, n_site)
        xM = compute_correlated_covars(rng, x_pc; n_covar, rhodec)
        # true model as a 
        # linear combination of uncorrelated base vectors and interactions
        combs = Combinatorics.combinations(1:n_covar_pc, 2)
        #comb = first(combs)
        x_pc_comb = reduce(vcat, transpose.(map(combs) do comb
            x_pc[comb[1], :] .* x_pc[comb[2], :]
        end))
        x_pc_all = vcat(x_pc, x_pc_comb)
        A = rand(rng, T, n_ζM, size(x_pc_all, 1))
        f_true = (x) -> A * x
        ζMs_true0 = f_true(x_pc_all)
        # center around mean with 10% relative error at original scale
        σ = sqrt(log(T(1)+abs2(T(0.1))))  #ζM .* 0.1
        ζMs_true = scale_centered_at(ζMs_true0, ζM, fill(σ, size(ζMs_true0,1)))   
    end     
    return (; xM, ζMs_true, clusters)
end

function get_clusters(n_site; scenario::Val{scen}) where scen
    if any((:clustered_sites,) .∈ Ref(scen))
        n_sites_cluster = [30, n_site ÷ 4]
        n_sites_cluster = vcat(n_sites_cluster, n_site .- sum(n_sites_cluster)) # ensure sum is n_site
        clusters = vcat(fill.(1:length(n_sites_cluster), n_sites_cluster)...)
    else
        # each site one cluster
        n_sites_cluster = fill(1, n_site) 
        clusters = 1:n_site
    end
    return n_sites_cluster, clusters
end


"""
Create `n_covar` correlated covariates 
from uncorrelated row-wise vector `x_pc`,
with correlations `rhos` to the linear combinations of `x_pc`.

By default correlations, `rhos = (1.0),0.88,0.78,0.69,0.61 ...`, 
decrease exponentially as `e^{-i/rhodec}`, with `rhodec = 8`.
"""
function compute_correlated_covars(rng::AbstractRNG, x_pc::AbstractMatrix{T};
    n_covar=size(x_pc, 1) + 3,
    rhodec=8,
    rhos=vcat(T(1.0), exp.(.-(1:(n_covar-1)) ./ T(rhodec)))) where {T}
    n_covar_pc, n_site = size(x_pc)
    A = rand(rng, T, n_covar, n_covar_pc)
    x_oc = (A * x_pc)
    # add noise to decorrelate
    rhoM = repeat(rhos, 1, n_site)
    noise = randn(rng, T, n_covar, n_site) .* T(0.2)
    x_o = rhoM .* x_oc .+ (1 .- rhoM) .* noise
    return x_o
end

# function compute_correlated_noise(rng::AbstractRNG, n_covar, n_site; 
#     rhodec=8,
#     rhos=vcat(T(1.0), exp.(.-(1:(n_covar-1)) ./ T(rhodec)))
#     ) where {T}
#     # add noise to decorrelate
#     rhoM = repeat(rhos, 1, n_site)
#     noise = randn(rng, T, n_covar, n_site) .* T(0.2)
#     x_o = rhoM .* x_oc .+ (1 .- rhoM) .* noise
#     return x_o
# end


"""
    scale_centered_at(x, m, σrel=1.0)
    scale_centered_at(x, m, σ)

Centers and rescales rows of matrix `x` around vector `m`. The scale can
either be given relative to `m` or specified as a vector of same size as `m`.
"""
function scale_centered_at(x::AbstractMatrix, m::AbstractVector, σrel::Real=1.0)
    σ = m .* σrel
    scale_centered_at(x, m, σ)
end
function scale_centered_at(x::AbstractMatrix, m::AbstractVector, σ::AbstractVector)
    dt = fit(ZScoreTransform, x, dims=2)
    x_unit_scaled = StatsBase.transform(dt, x)
    m .+  x_unit_scaled .* σ
end

