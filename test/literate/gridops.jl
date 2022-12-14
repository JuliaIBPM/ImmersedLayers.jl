# # Grid operations

#md # ```@meta
#md # CurrentModule = ImmersedLayers
#md # ```

#=
## Basic differential operations
There are a variety of (purely) grid-based operators that are useful for carrying
out calculations in immersed layer problems. We will demonstrate a few of them
here.
We will start by generating the cache, just as we did in [Immersed layer caches](@ref)
=#

using ImmersedLayers
using CartesianGrids
#!jl using Plots

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
#!jl plot(v,cache)

#=
We can then compute the derivative of this data
=#
divv = zeros_grid(cache)
divergence!(divv,v,cache)
#!jl plot(divv,cache)

#=
## Convective derivatives
And finally, let's compute convective derivatives. First, we will compute

$$\mathbf{v}\cdot\nabla p$$

For this operation, we create a special additional cache using [`ConvectiveDerivativeCache`](@ref).
This extra cache holds additional memory for making the calculation of the convective derivative faster
if we compute it often.
=#
cdcache = ConvectiveDerivativeCache(cache)
vdp = zeros_grid(cache)
convective_derivative!(vdp,v,p,cache,cdcache) #hide
@time convective_derivative!(vdp,v,p,cache,cdcache)
nothing #hide

# Plot it
#!jl plot(vdp,cache)

#=
Now, let's compute

$$\mathbf{v}\cdot\nabla\mathbf{v}$$

For this, we create a cache for `VectorGridData`, and a new instance of
`ConvectiveDerivativeCache` to go along with it.
=#
vcache = SurfaceVectorCache(g,scaling=GridScaling)
vdv = zeros_grid(vcache)
cdvcache = ConvectiveDerivativeCache(vcache)
convective_derivative!(vdv,v,vcache,cdvcache) #hide
@time convective_derivative!(vdv,v,vcache,cdvcache)
nothing #hide

# Plot it
#!jl plot(vdv,vcache)

#=
Finally, let's compute

$$(\curl\mathbf{v})\times\mathbf{v}$$

For this, we create a cache called
`RotConvectiveDerivativeCache` to go along with it.
=#
w = zeros_gridcurl(vcache)
curl!(w,v,vcache)
wv = zeros_grid(vcache)
cdrcache = RotConvectiveDerivativeCache(vcache)
w_cross_v!(wv,w,v,vcache,cdrcache) #hide
@time w_cross_v!(wv,w,v,vcache,cdrcache)

# Plot it
#!jl plot(wv,vcache)

#md # ## Grid operator functions
#md # ```@docs
#md # divergence!
#md # grad!
#md # curl!
#md # convective_derivative!
#md # convective_derivative
#md # ConvectiveDerivativeCache
#md # w_cross_v!
#md # w_cross_v
#md # RotConvectiveDerivativeCache
#md # laplacian!
#md # inverse_laplacian!
#md # ```
