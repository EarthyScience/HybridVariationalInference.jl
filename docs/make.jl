using HybridVariationalInference
import HybridVariationalInference.DoubleMM
using Documenter

DocMeta.setdocmeta!(HybridVariationalInference, :DocTestSetup, :(using HybridVariationalInference); recursive=true)

makedocs(;
    #modules=[HybridVariationalInference, HybridVariationalInference.DoubleMM],
    modules=[HybridVariationalInference],
    authors="Thomas Wutzler <twutz@bgc-jena.mpg.de> and contributors",
    sitename="HybridVariationalInference.jl",
    format=Documenter.HTML(;
        canonical="https://EarthyScience.github.io/HybridVariationalInference.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "Problem" => "problem.md",
        "Tutorials" => [        
            "Basic workflow" => "tutorials/basic_cpu.md",
            "Inspect results" => "tutorials/inspect_results.md",
            #"Test quarto markdown" => "tutorials/test1.md",
        ],
        "How to" => [
            ".. model independent parameters" => "tutorials/blocks_corr.md",
            ".. model site-global corr" => "tutorials/corr_site_global.md",
            ".. use GPU" => "tutorials/lux_gpu.md",
        ],
        "Explanation" => [
            #"Theory" => "explanation/theory_hvi.md", TODO activate when paper is published
        ],
        "Reference" => [
            "Public" => "reference/reference_public.md",
            "Internal" => "reference/reference_internal.md",
        ],
    ],
)

deploydocs(;
    repo="github.com/EarthyScience/HybridVariationalInference.jl",
    devbranch="main",
)
