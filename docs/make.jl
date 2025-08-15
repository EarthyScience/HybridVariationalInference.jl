using HybridVariationalInference
using Documenter

DocMeta.setdocmeta!(HybridVariationalInference, :DocTestSetup, :(using HybridVariationalInference); recursive=true)

makedocs(;
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
            "Test quarto markdown" => "tutorials/test1.md",
        ],
        "How to" => [
            #".. model site-global corr" => "tutorials/how_to_guides/corr_site_global.md",
        ],
        "Explanation" => [
            "Theory" => "explanation/theory_hvi.md",
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
