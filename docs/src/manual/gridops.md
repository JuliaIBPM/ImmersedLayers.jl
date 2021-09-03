```@meta
EditURL = "<unknown>/literate/gridops.jl"
```

# Grid operations

```@meta
CurrentModule = ImmersedLayers
```

There are a variety of (purely) grid-based operators that are useful for carrying
out calculations in immersed layer problems. We will demonstrate a few of them
here.
We will start by generating the cache, just as we did in [Immersed layer caches](@ref)

````@example gridops
using ImmersedLayers
using CartesianGrids
using Plots
````

### Set up a grid and cache

````@example gridops
Δx = 0.01
Lx = 4.0
xlim = (-Lx/2,Lx/2)
ylim = (-Lx/2,Lx/2)
grid = PhysicalGrid(xlim,ylim,Δx)
````

We still generate a cache for these operations, but
now, we only supply the grid. There are no immersed surfaces
for this demonstration.

````@example gridops
cache = SurfaceScalarCache(grid,scaling=GridScaling)
````

To demonstrate, let's generate a Gaussian

````@example gridops
p = zeros_grid(cache)
xg, yg = x_grid(cache), y_grid(cache)
p .= exp.(-(xg∘xg)-(yg∘yg))
````

Now, let's generate the gradient of these data

````@example gridops
v = zeros_gridgrad(cache)
grad!(v,p,cache)
plot(v,cache)
````

And finally, let's compute the convective derivative,

$$\mathbf{v}\cdot\nabla\mathbf{v}$$

For this, we create a separate cache, using [`ConvectiveDerivativeCache`](@ref), which
can be constructed from the existing `cache`. This extra cache holds additional
memory for making the calculation of the convective derivative faster. We will

````@example gridops
cdcache = ConvectiveDerivativeCache(cache)
vdv = zeros_gridgrad(cache)
convective_derivative!(vdv,v,cache,cdcache) #hide
@time convective_derivative!(vdv,v,cache,cdcache)
nothing #hide
````

Plot it

````@example gridops
plot(vdv,cache)
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

