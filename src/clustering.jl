function cluster_records(X_matrix::AbstractMatrix, Ucor::AbstractMatrix{T}, σM::AbstractVector; n_cluster=4) where T
    # x_i -x_j are distributed N(0, 2Σ) 
    #Σ = 2 * HybridVariationalInference.compute_cov(Ucor, σM)    
    invΣ = HybridVariationalInference.compute_invcov(Ucor, σM) / T(2)
    # Precompute differences: (X_i - X_j) for all i, j
    # Use broadcasting to compute all pairwise differences
    #diffs = X_matrix' .- X_matrix  # Shape: (n_vars, n_rows, n_rows)
    n_rows, n_vars = size(X_matrix)  # e.g., 2000×10
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
    clusters = cutree(res_clust; k = n_cluster)
    () -> begin
        counts(clusters)
        scatter(X_matrix[:,1], X_matrix[:,2], color = clusters)
    end
    clusters
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
