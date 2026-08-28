# HybridVariationalInference: grad_sample_ζsP! Optimization Plan

## Current State

The function `grad_sample_ζsP!` in `src/elbo_site_grad.jl` currently allocates three buffers on each call:
- `ϕqc_shadow`: a `ComponentArray` of length `n_in` (input shadow)
- `dζsP_col_shadow`: a matrix of shape `size(ζsP)` (output shadow)
- `ζsP_primal`: a copy of `ζsP` (primal working copy)

This leads to repeated allocations, which is inefficient.

## Proposed Solution: Type-Stable GradBuf{...} Strategy

The goal is to replace the current `h` NamedTuple with a more sophisticated `GradBuf` object that lazily creates and caches these buffers on first use, while maintaining type stability.

### Design

1. **Create a `GradBuf` type** that holds the three buffers:
   - `ζsP_primal` (Matrix)
   - `dζsP_col_shadow` (Matrix)
   - `dϕqc_shadow` (ComponentArray)

2. **Lazily create buffers** on first use:
   - The `get_ζsP_primal!` method checks if the buffer exists and matches the current `ζsP` size
   - If not, it creates a new buffer of the correct size
   - The `get_dζsP_col_shadow!` and `get_dϕqc_shadow!` methods work similarly

3. **Type stability**: The `GradBuf` will be parameterized by the element type `T` and the axes `AX` of the ComponentArray, ensuring type stability.

4. **Integration with existing code**: The `GradBuf` will be stored in the `elbo_helpers` NamedTuple (e.g., `h.gradbuf`) and passed to `grad_sample_ζsP!`.

### Implementation Steps

1. Define the `GradBuf` type in `elbo_site_grad.jl`
2. Implement `get_ζsP_primal!`, `get_dζsP_col_shadow!`, and `get_dϕqc_shadow!` methods
3. Update `grad_sample_ζsP!` to use the `GradBuf` object
4. Update `grad_neg_elbo_sites` to use the `GradBuf` in the `h` NamedTuple
5. Update the test file to initialize the `GradBuf`

### Open Questions

1. Should the `GradBuf` be stored in `h` as a field (`h.gradbuf`) or passed as a separate argument?
2. Should we also implement `autodiff_thunk` to eliminate recompilation overhead?
3. What is the preferred naming for the `GradBuf` type and its methods?

## Next Steps

The plan is ready for implementation. Please provide feedback on the open questions above before proceeding.
