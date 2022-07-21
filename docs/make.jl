using WaveSurrogates.jl
using Documenter

DocMeta.setdocmeta!(WaveSurrogates.jl, :DocTestSetup, :(using WaveSurrogates.jl); recursive=true)

makedocs(;
    modules=[WaveSurrogates.jl],
    authors="Rahul Manavalan",
    repo="https://github.com/dynamic-queries/WaveSurrogates.jl.jl/blob/{commit}{path}#{line}",
    sitename="WaveSurrogates.jl.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://dynamic-queries.github.io/WaveSurrogates.jl.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/dynamic-queries/WaveSurrogates.jl.jl",
    devbranch="main",
)
