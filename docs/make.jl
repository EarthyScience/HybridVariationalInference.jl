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
    ],
)

deploydocs(;
    repo="github.com/EarthyScience/HybridVariationalInference.jl",
    devbranch="main",
)
