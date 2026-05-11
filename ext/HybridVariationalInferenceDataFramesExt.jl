module HybridVariationalInferenceDataFramesExt

using DataFrames
using ComponentArrays: ComponentArrays as CA
using HybridVariationalInference: HybridVariationalInference as HVI
import LinearAlgebra

function HVI.as_data_frame(cm::CA.ComponentMatrix) 
    if (CA.getaxes(cm)[1] isa CA.Axis) && length(keys(CA.getaxes(cm)[1])) == size(cm,1)
        DataFrame((k => cm[k,:] for k in keys(cm[:,1]))...)
    elseif (CA.getaxes(cm)[2] isa CA.Axis) && length(keys(CA.getaxes(cm)[2])) == size(cm,2)
        DataFrame((k => cm[:,k] for k in keys(cm[1,:]))...)
    else
        error("first or second axis must be an scalar axis, but got $(CA.getaxes(cma))")
    end
end

function HVI.as_data_frame(cmt::LinearAlgebra.Adjoint{T, <:CA.ComponentMatrix}) where T
    cm = cmt'
    HVI.as_data_frame(cm)
end

function HVI.as_data_frame(cma::CA.ComponentArray{T,3}) where T 
    if (CA.getaxes(cma)[1] isa CA.Axis) && length(keys(CA.getaxes(cma)[1])) == size(cma,1)
        df = DataFrame((k => vec(cma[k,:,:]) for k in keys(cma[:,1,1]))...)
        df.dim3 = vcat(fill.(axes(cma,3), size(cma,1))...)
    elseif (CA.getaxes(cma)[2] isa CA.Axis) length(keys(CA.getaxes(cma)[2])) == size(cma,2)
        df = DataFrame((k => vec(cma[:,k,:]) for k in keys(cma[1,:,1]))...)
        df.dim3 = vcat(fill.(axes(cma,3), size(cma,1))...)
    else
        error("first or second axis must be an Axis, but got $(CA.getaxes(cma))")
    end
    df
end

function HVI.as_data_frame(cma4::CA.ComponentArray{T,4}) where T 
    if length(keys(CA.getaxes(cma4)[1])) == size(cma4,1)
        df = DataFrame((k => vec(cma4[k,:,:,:]) for k in keys(cma4[:,1,1,1]))...)
        dim3 = vcat(fill.(axes(cma4,3), size(cma4,2))...)
    elseif length(keys(CA.getaxes(cma4)[2])) == size(cma4,2)
        df = DataFrame((k => vec(cma4[:,k,:,:]) for k in keys(cma4[1,:,1,1]))...)
        dim3 = vcat(fill.(axes(cma4,3), size(cma4,1))...)
    else
        error("first or second axis must be an Axis, but got $(CA.getaxes(cma))")
    end
    df.dim3 = vcat(fill(dim3, size(cma4,4))...)
    df.dim4 = vcat(fill.(axes(cma4,4), prod(size(cma4,d) for d in (1,3)))...)
    df
end



end # module
