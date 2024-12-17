using Test, SafeTestsets
const GROUP = get(ENV, "GROUP", "All") # defined in in CI.yml

@time begin
    if GROUP == "All" || GROUP == "Basic"
    end
end

@time begin
    if GROUP == "All" || GROUP == "Aqua"
        #@safetestset "test_aqua" include("test/test_aqua.jl")
        @time @safetestset "test_aqua" include("test_aqua.jl")
    end
end


