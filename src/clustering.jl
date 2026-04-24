function refit_clusters(rng, probo, solver, xM ; 
    scenario, n_cluster_initial = 12, n_aggsplits = 5, epochs = 400,
    )
    Ucor = get_hybridproblem_cholesky_correlation_Ms(probo, xM; scenario)
    (; X, σMs) = extract_MLpred(probo, xM; scenario)
    σM = vec(median(σMs; dims=2))
    # first clustering based on uncertainty of unclusterd sites but with argument cluster_rep
    clusters = clusters0 = cluster_records(X, Ucor, σM; n_cluster = n_cluster_initial)
    # cnts_clusters_tosplit tracks the number of sites in clusters still to check, 
    # and will be updated on splitting clusters
    cnts_clusters_totest = cnts0 = StatsBase.countmap(clusters)
    while (length(cnts_clusters_totest) > 0)
        @show length(cnts_clusters_totest) 
        i_splits = n_aggsplits
        # collect newly created smaller cluster, but only check them after refit
        cnts_new = Dict{eltype(clusters), Int}()
        while (i_splits > 0) && (length(cnts_clusters_totest) > 0)
            #global cnts, clusters, i_splits
            i_cluster = argmax(cnts_clusters_totest)
            i_sites = findall(isequal.(clusters, i_cluster))
            σM = vec(median(σMs[:,i_sites]; dims=2))
            #
            (; is_overdispersed, clusters, clusters_sub) = split_cluster(clusters, i_cluster, X, Ucor, σMs)
            @show i_cluster, cnts_clusters_totest[i_cluster], is_overdispersed
            delete!(cnts_clusters_totest, i_cluster) # remove the inspected cluster
            if is_overdispersed
                # add new clusters to inspect for overdispersion
                cnts_new_cl = StatsBase.countmap(clusters_sub)
                cnts_new = merge(cnts_clusters_totest, cnts_new_cl)
                i_splits = i_splits - 1
            end
        end
        cnts_clusters_totest = merge(cnts_clusters_totest, cnts_new)
        (; probo, X, σMs) = refit(rng, probo, solver, xM, clusters; scenario, epochs)
    end
    (; probo, clusters)
end

function refit(rng, probo, solver, xM, clusters; scenario, epochs)
    (; probo) = solve(probo, solver; rng,
        callback = callback_loss(100), # output during fitting
        epochs,
        clusters,
    );
    (; X, σMs) = extract_MLpred(probo, xM; scenario)
    (; probo, X, σMs)
end

function extract_MLpred(probo, xM; scenario)
    g, ϕg0 = get_hybridproblem_MLapplicator(probo; scenario)
    n_θ = HybridVariationalInference.get_numberof_θM(probo.approx, ϕg0)
    ζ = g(xM, probo.ϕg)
    X = ζ'[:,1:n_θ]
    (;σP, σMs) = get_marginal_std(probo, xM; scenario)
    (; X, σMs)
end

function cluster_records(X_matrix::AbstractMatrix, Ucor::AbstractMatrix{T}, σM::AbstractVector; n_cluster=4, cluster_ids = 1:n_cluster, ) where T
    # x_i -x_j are distributed N(0, 2Σ) 
    #Σ = 2 * HybridVariationalInference.compute_cov(Ucor, σM)    
    @assert n_cluster == length(cluster_ids) "Length of cluster_ids must match n_cluster"
    invΣ = HybridVariationalInference.compute_invcov(Ucor, σM) / T(2)
    # Precompute differences: (X_i - X_j) for all i, j
    # Use broadcasting to compute all pairwise differences
    #diffs = X_matrix' .- X_matrix  # Shape: (n_vars, n_rows, n_rows)
    n_rows, n_vars = size(X_matrix)  # e.g., 2000×10
    n_rows < n_cluster && error("Cannot cluster $nrows records into $n_cluster clusters.")
    if n_rows == n_cluster
        # assign each record to its own cluster
        clusters0 = 1:n_rows
    else
        diffs = reshape(X_matrix, n_rows, 1, n_vars) .- reshape(X_matrix, 1, n_rows, n_vars)
        dist2_matrix = zeros(n_rows, n_rows)
        for i in 1:n_rows
            for j in 1:n_rows
                diff = diffs[i, j, :]  # (n_vars,)
                dist2_matrix[i, j] = diff' * invΣ * diff
            end
        end
        dist_matrix = sqrt.(dist2_matrix)
        # If you want to go faster, use:
        # dist_matrix = sqrt.(sum((diffs * invΣ) .* diffs, dims=3)[:, :, 1])
        res_clust = hclust(dist_matrix; linkage = :ward)
        clusters0 = cutree(res_clust; k = n_cluster)
        () -> begin
            counts(clusters0)
            scatter(X_matrix[:,1], X_matrix[:,2], color = clusters0)
        end
    end
    # translate 1:n_cluster to provided cluster_ids
    clusters = cluster_ids[clusters0] 
end


"""
    compute_pvalue_asymptotic_overdispersion_from_dist2(dist2_matrix)

Compute p-value for overdispersion using asymptotic approximation,
based on a precomputed matrix of squared Mahalanobis distances.

# Arguments
- `dist2_matrix`: m × m symmetric matrix of squared Mahalanobis distances
  (dist2_matrix[i,j] = (x_i - x_j)' Σ⁻¹ (x_i - x_j))
- n: the dimension of x_i (number of variables)
- The matrix must be symmetric and contain only upper/lower triangle values

# Returns
- `p_value`: one-sided p-value for overdispersion
"""
function compute_pvalue_asymptotic_overdispersion_from_dist2(dist2_matrix, n)
    m = size(dist2_matrix, 1)

    # Number of unique pairs (i < j)
    N = m * (m - 1) ÷ 2

    # Number of triplets (i,j,k) with i < j < k
    K = m * (m - 1) * (m - 2) ÷ 6

    # Extract upper triangle (i < j) and sum
    S_obs = 0.0
    for i in 1:m
        for j in i+1:m
            S_obs += dist2_matrix[i, j]
        end
    end

    # Expected value under H0: each D_ij^2 ~ χ²_n → E[D_ij^2] = n
    μ₀ = N * n

    # Variance under H0 (corrected for dependence between overlapping pairs)
    var_S = 2 * n * N + 4 * n * K

    # Z-score
    z_score = (S_obs - μ₀) / sqrt(var_S)

    # One-sided p-value: is the observed spread significantly larger?
    p_value = 1 - StatsFuns.normcdf(z_score)

    return p_value
end

"""
    overdispersion_test(Y, μ, Σ; α=0.05)

Test whether the q×p sample matrix Y (rows = indiviudals) is overdispersed
relative to the reference distribution N(μ, Σ).

Returns: S_n, E0, Var0, Z, p_value_normal, p_value_chisq
"""
function check_overdispersion(
    X_matrix::AbstractMatrix{T}, Ucor::AbstractMatrix{T}, σM::AbstractVector{T}; 
    α::S=0.05
    ) where {S,T}
    # see test_overdispersion_theory.md
    n, p = size(X_matrix)
    #Σ = HybridVariationalInference.compute_cov(Ucor, σM) 
    invΣ = HybridVariationalInference.compute_invcov(Ucor, σM) 
    # Step 1: Compute sum of distances across sample pairs
    S_n = zero(T)
    for i in 1:n
        for j in (i+1):n
            d = X_matrix[i, :] - X_matrix[j, :]
            S_n += dot(d, invΣ * d)
        end
    end
    # Step 2: compute expected (H0) moments
    E0   = convert(S, p * n * (n - 1))
    Var0 = S(2.0) * p * abs2(n) * (n - 1) # see VarSumDij.md
    SD0  = sqrt(Var0)
    # Step 3: p-values of standardized statistic
    Z = (S_n - E0) / SD0
    p_val_normal = one(S) - StatsFuns.normcdf(Z)

    ν  = p * (n - 1)          # degrees of freedom
    c  = S(n)           # scaling constant
    p_val_chisq = one(S) - cdf(Chisq(ν), S_n / c)

    is_overdispersed = p_val_chisq < α
    #Main.@infiltrate_main

    return (; is_overdispersed, 
        S_n=S_n, E0=E0, Var0=Var0, p_normal=p_val_normal, p_chisq=p_val_chisq )
end

function split_cluster(clusters, i_cluster, X, Ucor, σMs; n_cluster=4)
    i_sites = findall(isequal.(clusters, i_cluster))
    n_sites_cl = length(i_sites)
    if n_sites_cl == 1
        return (; is_overdispersed = false, clusters, clusters_sub = eltype(clusters)[])
    end
    X_cluster = X[i_sites,:]
    σM = vec(median(σMs[:,i_sites]; dims=2))
    # vec(std(X_cluster; dims = 1))
    is_overdispersed = check_overdispersion(X_cluster, Ucor, σM)[1]
    if is_overdispersed
        n_cluster_split = min(n_cluster, n_sites_cl)
        cluster_ids = vcat(i_cluster, maximum(clusters) .+ (1:(n_cluster_split-1)))
        clusters_sub = cluster_records(X_cluster, Ucor, σM; cluster_ids);
        clusters[i_sites] = clusters_sub
    else
        clusters_sub = eltype(clusters)[]
    end
    (; is_overdispersed, clusters, clusters_sub)       
    # else
    #     # split into "clusters" of single observations
    #     @info("Debug splitting cluster into single sites.")
    #     Main.@infiltrate_main

    #     clusters_sub = vcat(i_cluster, maximum(clusters) .+ (1:(n_sites_cl-1)))
    #     clusters[i_sites] = cluster_sub
    #     (; is_overdispersed = false, clusters, clusters_sub)       
    # end
end


