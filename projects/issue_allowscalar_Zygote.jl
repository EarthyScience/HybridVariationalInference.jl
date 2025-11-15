using UnPack, 
using CUDA, cuDNN, MLDataDevices, GPUArraysCore
using ComponentArrays: ComponentArrays as CA
using Zygote

tmp = gpu_device()(CA.ComponentVector(a=1, b=2:5));

tmpf = function(tmp)
     GPUArraysCore.allowscalar() do
          #a = tmp.a
          a = tmp[Val(:a)]
          #a = (UnPack).unpack(tmp, Val{:a}())
          #@macroexpand @unpack a,b = tmp
          #@unpack a,b = tmp
          #return(a)
     end
end

tmpf(tmp)
Zygote.gradient(tmpf, tmp)  # triggers Scalar exception

