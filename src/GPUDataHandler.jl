abstract type AbstractGPUDataHandler end

"""
    (app::AbstractGPUDataHandler)(x) = handle_GPU_data(app, x)

Callable applied to argument `x`, used to configure the exchange of data between
GPU and CPU.
By Default, nothing is done to x.
Package extensions for Flux and Lux implement overloads for AbstractGPUArray
that call `cpu(x)` to transfer data on the GPU to CPU. Those package extension
also use `set_default_GPUHandler()` so that .
"""
function handle_GPU_data(::AbstractGPUDataHandler, x::AbstractArray) 
    x
end

(app::AbstractGPUDataHandler)(x) = handle_GPU_data(app, x)


struct NullGPUDataHandler <: AbstractGPUDataHandler end

handle_GPU_data(::NullGPUDataHandler, x::AbstractArray) = x 

default_GPU_DataHandler = NullGPUDataHandler()
get_default_GPUHandler() = default_GPU_DataHandler
function set_default_GPUHandler(handler::AbstractGPUDataHandler) 
    @info "set_default_GPUHandler: setting default handler to $handler"
    global default_GPU_DataHandler = handler
end


