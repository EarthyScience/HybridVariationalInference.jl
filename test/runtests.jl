using Test, SafeTestsets
import Logging, LoggingExtras as LE
warn2error_logger = LE.TransformerLogger(LE.global_logger()) do log
    return log.level === Logging.Warn ?  merge(log, (; level=Logging.Error)) : log
end

() -> begin
    #@usingany ReferenceRevision
    refmain = open_process(rev = "main")
    refmain = open_process(rev = "main", instantiate = true)
    refmain.eval(:(using CommonSolve))
    refmain = open_process(rev = "main", instantiate = true)
    close(refmain)
end
const GROUP = get(ENV, "GROUP", "All") # defined in in CI.yml

@time begin
    if GROUP == "All" || GROUP == "Basic"
        () -> begin
            LE.with_logger(warn2error_logger) do
                @safetestset "test" include("test/test_elbo_site.jl")
            end
        end


        @time @safetestset "test_util" include("test_elbo_site.jl")
    end
end

@time begin
    if GROUP == "All" || GROUP == "Aqua"
        #@safetestset "test" include("test/test_aqua.jl")
        if VersionNumber("1.11.2") <= VERSION < VersionNumber("1.12")
            #@safetestset "test" include("test/test_aqua.jl")
            @time @safetestset "test_aqua" include("test_aqua.jl")
        end
    end
end


