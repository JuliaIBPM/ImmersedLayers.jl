```@meta
EditURL = "<unknown>/literate/problems.jl"
```

# Problems and the system

```@meta
CurrentModule = ImmersedLayers
```

In specific problems that we wish to solve with immersed layers, there may
be other data and operators that we would like to cache. We do this with
an *extra cache*, which the user can define, along with a problem type associated
with this cache. The basic cache and the extra cache are generated and associated
together in a *system*.

## Example

````@example problems
using ImmersedLayers
using CartesianGrids
using RigidBodyTools
using Plots
````

## Problem types and functions

```@docs
AbstractScalarILMProblem
AbstractVectorILMProblem
BasicScalarILMProblem
BasicVectorILMProblem
prob_cache
```

## System types and functions

```@docs
ILMSystem
__init
```

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

