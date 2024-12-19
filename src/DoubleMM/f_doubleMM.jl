const S1 = [1.0, 1.0, 1.0, 1.0, 0.4, 0.3, 0.1]
const S2 = [1.0, 3.0, 4.0, 5.0, 5.0, 5.0, 5.0]

θP = CA.ComponentVector(r0 = 0.3, K2 = 2.0)
θM = CA.ComponentVector(r1 = 0.5, K1 = 0.2)

const int_θdoubleMM = ComponentArrayInterpreter(flatten1(CA.ComponentVector(;θP,θM)))

function f_doubleMM(θ::AbstractVector)
    # extract parameters not depending on order, i.e if they are in θP or θM
    θc = int_θdoubleMM(θ)
    r0, r1, K1, K2 = θc[(:r0, :r1, :K1, :K2)]
    y = r0 .+ r1 .* S1 ./ (K1 .+ S1) .* S2 ./ (K2 .+ S2)
end

"""
Generate correlated covariates and synthetic true parameters that
are a linear combination of the uncorrelated underlying principal 
factors and their binary combinations.

In addtion provide a SimpleChains model of adequate complexity to
fit this realationship θMs_true = f(x_o)
"""
function gen_q(rng::AbstractRNG, T::DataType,
    n_covar_pc, n_covar, n_site, n_θM::Integer;
    rhodec=8, is_using_dropout=false)
    x_pc = rand(rng, T, n_covar_pc, n_site)
    x_o = compute_correlated_covars(rng, x_pc; n_covar, rhodec)
    # true model as a 
    # linear combination of uncorrelated base vectors and interactions
    combs = Combinatorics.combinations(1:n_covar_pc, 2)
    #comb = first(combs)
    x_pc_comb = reduce(vcat, transpose.(map(combs) do comb
        x_pc[comb[1], :] .* x_pc[comb[2], :]
    end))
    x_pc_all = vcat(x_pc, x_pc_comb)
    A = rand(rng, T, n_θM, size(x_pc_all, 1))
    θMs_true = A * x_pc_all
    #
    # g, q = map((n_θM, n_θM * 2)) do n_out
    #     if is_using_dropout
    #         SimpleChain(
    #             static(n_covar), # input dimension (optional)
    #             # dense layer with bias that maps to 8 outputs and applies `tanh` activation
    #             TurboDense{true}(tanh, n_covar * 4),
    #             SimpleChains.Dropout(0.2), # dropout layer
    #             TurboDense{true}(logistic, n_covar * 4),
    #             SimpleChains.Dropout(0.2),
    #             # dense layer without bias that maps to n outputs and `identity` activation
    #             TurboDense{false}(identity, n_out),
    #         )
    #     else
    #         SimpleChain(
    #             static(n_covar), # input dimension (optional)
    #             # dense layer with bias that maps to 8 outputs and applies `tanh` activation
    #             TurboDense{true}(tanh, n_covar * 4),
    #             TurboDense{true}(logistic, n_covar * 4),
    #             # dense layer without bias that maps to n outputs and `identity` activation
    #             TurboDense{false}(identity, n_out),
    #         )
    #     end
    # end
    # return (x_o, θMs_true, g, q)
    return (x_o, θMs_true)
end


