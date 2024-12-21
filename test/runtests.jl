using Test, SafeTestsets
const GROUP = get(ENV, "GROUP", "All") # defined in in CI.yml

@time begin
    if GROUP == "All" || GROUP == "Basic"
        #@safetestset "test" include("test/test_ComponentArrayInterpreter.jl")
        @time @safetestset "test_ComponentArrayInterpreter" include("test_ComponentArrayInterpreter.jl")
        #@safetestset "test" include("test/test_gencovar.jl")
        @time @safetestset "test_gencovar" include("test_gencovar.jl")
        #@safetestset "test" include("test/test_SimpleChains.jl")
        @time @safetestset "test_SimpleChains" include("test_SimpleChains.jl")
        #@safetestset "test" include("test/test_doubleMM.jl")
        @time @safetestset "test_doubleMM" include("test_doubleMM.jl")
        #@safetestset "test" include("test/test_cholesky_structure.jl")
        @time @safetestset "test_cholesky_structure" include("test_cholesky_structure.jl")
        #
        #@safetestset "test" include("test/test_Flux.jl")
        @time @safetestset "test_Flux" include("test_Flux.jl")
        #@safetestset "test" include("test/test_Lux.jl")
        @time @safetestset "test_Lux" include("test_Lux.jl")
    end
end

@time begin
    if GROUP == "All" || GROUP == "Aqua"
        #@safetestset "test" include("test/test_aqua.jl")
        @time @safetestset "test_aqua" include("test_aqua.jl")
    end
end


