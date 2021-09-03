using Documenter, ImmersedLayers

ENV["GKSwstype"] = "nul" # removes GKS warnings during plotting

makedocs(
    sitename = "ImmersedLayers.jl",
    doctest = true,
    clean = true,
    modules = [ImmersedLayers],
    pages = [
        "Home" => "index.md",
        "Manual" => ["manual/caches.md",
                     "manual/surfaceops.md",
                     "manual/gridops.md",
                     "manual/matrices.md",
                     "manual/dirichlet.md",
                     "manual/problems.md",
                     "manual/utilities.md"
                     ]
        #"Internals" => [ "internals/properties.md"]
    ],
    #format = Documenter.HTML(assets = ["assets/custom.css"])
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true",
        mathengine = MathJax(Dict(
            :TeX => Dict(
                :equationNumbers => Dict(:autoNumber => "AMS"),
                :Macros => Dict()
            )
        ))
    ),
    #assets = ["assets/custom.css"],
    #strict = true
)


#if "DOCUMENTER_KEY" in keys(ENV)
deploydocs(
     repo = "github.com/JuliaIBPM/ImmersedLayers.jl.git",
     target = "build",
     deps = nothing,
     make = nothing
     #versions = "v^"
)
#end
