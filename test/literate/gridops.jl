# # Grid operations

#md # ```@meta
#md # CurrentModule = ImmersedLayers
#md # ```

#=
There are a variety of (purely) grid-based operators that are useful for carrying
out calculations in immersed layer problems. We will demonstrate a few of them
here.
We will start by generating the cache, just as we did in [Immersed layer caches](@ref)
=#

using ImmersedLayers
using CartesianGrids
using Plots

#=
### Set up a grid and cache
=#
Δx = 0.01
Lx = 4.0
xlim = (-Lx/2,Lx/2)
ylim = (-Lx/2,Lx/2)
g = PhysicalGrid(xlim,ylim,Δx)

#=
We still generate a cache for these operations, but
now, we only supply the grid. There are no immersed surfaces
for this demonstration.
=#
cache = SurfaceScalarCache(g,scaling=GridScaling)

#=
To demonstrate, let's generate a Gaussian
=#
p = zeros_grid(cache)
xg, yg = x_grid(cache), y_grid(cache)
p .= exp.(-(xg∘xg)-(yg∘yg))

#=
Now, let's generate the gradient of these data
=#
v = zeros_gridgrad(cache)
grad!(v,p,cache)
plot(v,cache)

#=
And finally, let's compute the convective derivative,

$$\mathbf{v}\cdot\nabla\mathbf{v}$$

For this, we create a separate cache, using [`ConvectiveDerivativeCache`](@ref), which
can be constructed from the existing `cache`. This extra cache holds additional
memory for making the calculation of the convective derivative faster. We will
=#
cdcache = ConvectiveDerivativeCache(cache)
vdv = zeros_gridgrad(cache)
convective_derivative!(vdv,v,cache,cdcache) #hide
@time convective_derivative!(vdv,v,cache,cdcache)
nothing #hide

# Plot it
plot(vdv,cache)


#md # ## Surface-grid operator functions
#md # ```@docs
#md # divergence!
#md # grad!
#md # curl!
#md # convective_derivative!
#md # convective_derivative
#md # ConvectiveDerivativeCache
#md # inverse_laplacian!
#md # ```
