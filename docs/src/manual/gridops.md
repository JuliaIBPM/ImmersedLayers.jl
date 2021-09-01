```@meta
EditURL = "<unknown>/literate/gridops.jl"
```

# Grid operations

```@meta
CurrentModule = ImmersedLayers
```

There are a variety of (purely) grid-based operators that are useful for carrying
out calculations in immersed layer problems.
We will start by generating the cache, just as we did in [Immersed layer caches](@ref)

````@example gridops
using ImmersedLayers
using CartesianGrids
using RigidBodyTools
using Plots
using LinearAlgebra
````

## Surface-grid operator functions
```@docs
divergence!
grad!
curl!
convective_derivative!
convective_derivative
ConvectiveDerivativeCache
inverse_laplacian!
```

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

