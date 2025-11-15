using Bijectors

using MLDataDevices
import CUDA, cuDNN
using Zygote

b2 = exp
b2 = elementwise(exp)

x = [0.1, 0.2, 0.3, 0.4]
b2 = Stacked((identity,),(1:length(x),))

function trans(x, b) 
       y, logjac = Bijectors.with_logabsdet_jacobian(b, x)
       sum(y .+ logjac)
end

y = trans(x,b2)
Zygote.gradient(x -> trans(x,b2), x)

xd = gpu_device()(x)
yd = trans(xd, b2)
Zygote.gradient(x -> trans(x,b2), xd) # errors with elementwise
