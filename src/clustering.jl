function cluster_records(X_matrix::AbsractMatrix{Real}, Ucor::Triangular)
    Sigma_inv = inv(Ucor') * inv(Ucor)
    # Precompute differences: (X_i - X_j) for all i, j
    # Use broadcasting to compute all pairwise differences
    #diffs = X_matrix' .- X_matrix  # Shape: (n_vars, n_rows, n_rows)
    diffs = reshape(X_matrix, n_rows, 1, n_vars) .- reshape(X_matrix, 1, n_rows, n_vars)


end