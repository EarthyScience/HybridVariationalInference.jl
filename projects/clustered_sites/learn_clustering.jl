using DataFrames, LinearAlgebra, Statistics, Distances, Clustering
#using HierarchicalClustering

# Step 1: Load or define your dataset
# Example: Replace this with your actual data loading
# df = DataFrame(CSV.File("your_data.csv"))

# Example dataset (replace with your own)
#n_vars = 10  # number of variables (columns)
n_vars = 2
n_rows = 2000  # number of records (rows)
X = randn(n_rows, n_vars)  # your data: each row is a record

# Convert to DataFrame (optional, for clarity)
df = DataFrame(X, :auto)

# Step 2: Compute the covariance matrix (if not already given)
Sigma = cov(Matrix(df))  # Shape: (n_vars, n_vars)

# Optional: Check if Sigma is invertible
if det(Sigma) ≈ 0
    @warn "Covariance matrix is singular. Adding regularization."
    λ = 1e-6  # small regularization
    Sigma = Sigma + λ * I
end

# Compute inverse of covariance matrix
Sigma_inv = inv(Sigma)

# Step 3: Compute Mahalanobis distances between all pairs of records
# We'll compute the squared Mahalanobis distance matrix efficiently

# Convert DataFrame to matrix
X_matrix = Matrix(df)

# Precompute differences: (X_i - X_j) for all i, j
# Use broadcasting to compute all pairwise differences
n_rows, n_vars = size(X_matrix)  # e.g., 2000×10
#diffs = X_matrix' .- X_matrix  # Shape: (n_vars, n_rows, n_rows)
diffs = reshape(X_matrix, n_rows, 1, n_vars) .- reshape(X_matrix, 1, n_rows, n_vars)



# Step 4: Compute Mahalanobis distances
dist_matrix = zeros(n_rows, n_rows)
for i in 1:n_rows
    for j in 1:n_rows
        diff = diffs[i, j, :]  # (n_vars,)
        dist_matrix[i, j] = sqrt(diff' * Sigma_inv * diff)
    end
end
# If you want to go faster, use:
# dist_matrix = sqrt.(sum((diffs * Sigma_inv) .* diffs, dims=3)[:, :, 1])

# Step 5: Hierarchical clustering
#linkage_matrix = linkage(dist_matrix, :ward)
#clusters = fcluster(linkage_matrix, k=5, criterion=:maxclust)
res = hclust(dist_matrix; linkage = :ward)
clusters = cutree(res; k = 10)

# Add to DataFrame
df[!, :cluster] = clusters


# Optional: View first few rows with cluster assignments
println(first(df, 10))

# Optional: Visualize (e.g., using PCA + coloring)
using Plots
using MultivariateStats

# Reduce to 2D using PCA
#pca = fit(PCA, X_matrix'; maxoutdim=2)
pca = fit(PCA, X_matrix'; maxoutdim=2)
X_pca = predict(pca, X_matrix')

# Plot
scatter(X_pca[1, :], X_pca[2, :], group=clusters, label="Cluster", title="Clustering Results (PCA)")
