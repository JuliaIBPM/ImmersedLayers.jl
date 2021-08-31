# # Surface-grid operations

#md # ```@meta
#md # CurrentModule = ImmersedLayers
#md # ```

#=
Here, we will discuss the various surface-grid operators available in the package.
We will start by generating the cache, just as we did in [Immersed layer caches](@ref)
=#


using ImmersedLayers
using CartesianGrids
using RigidBodyTools
using Plots

#=
### Set up the grid, shape, and cache
We do this just as we did in [Immersed layer caches](@ref)
=#
Δx = 0.01
Lx = 4.0
xlim = (-Lx/2,Lx/2)
ylim = (-Lx/2,Lx/2)
grid = PhysicalGrid(xlim,ylim,Δx)
RadC = 1.0
Δs = 1.4*cellsize(grid)
body = Circle(RadC,Δs)
cache = SurfaceScalarCache(body,grid,scaling=GridScaling)

#=
## Basic regularization and interpolation
Let's perform an example in which we regularize the x coordinate of
the surface points onto the grid. We can get the surface coordinates
by using the [`points`](@ref) function. This can be applied either to
`body` directly or to the `cache`. As for any `VectorData` type, the components
of `pts` are `pts.u` and `pts.v`.
=#
pts = points(cache);

#=
Now, we set up some blank grid data on which to regularize onto
=#
gx = zeros_grid(cache);

#=
Now regularize. We will time it to show that it is fast and memory-efficient:
=#
regularize!(gx,pts.u,cache) #hide
@time regularize!(gx,pts.u,cache);

#=
Let's plot this to look at it
=#
plot(gx,grid)

#=
This shows how the regularization spreads the data over a couple of cells
around the surface. In the parlance of potential theory, this is a **single layer**.
=#
#=
If we wish to interpolate, then we do so with the [`interpolate!`]
function. For example, suppose we have a uniform field that we wish to interpolate:
=#
oc = 2.5*ones_grid(cache)

#=
Now set up some surface data to receive the interpolated data, and
interpolate:
=#
f = zeros_surface(cache);
interpolate!(f,oc,cache)

#=
It is clear that the interpolation preserves the value of the
field. This is also true for linearly varying fields, since the DDF
is built to ensure this. Let's try this, using the $x$ coordinate of the grid. Here, we use the `coordinates`
function of `CartesianGrids.jl`, which gets the coordinates of the grid,
and set the values of grid data to the $x$ coordinate. We interpolate and plot,
comparing to the actual $x$ coordinate of the points on the body:
=#
x, y = coordinates(oc,grid)
xg = similar(oc)
xg .= x
interpolate!(f,xg,cache)
plot(f,ylim=(-2,2),label="Interpolated from grid",ylabel="x",xlabel="Index")
plot!(pts.u,label="Actual body coordinate")

#=
## A double layer
Now we will generate a double layer. Mathematically, this takes the form

$$\nabla\cdot \left( \delta(\chi) \mathbf{n} f \right)$$

for some scalar data $f$ on the surface. (See [Background](@ref) for an example.)
Notice that it maps scalar data on the surface ($f$) to scalar data in space. So to
calculate this using our discrete tools, we set up some grid data to receive the
result. Then, we use the function [`surface_divergence!`](@ref) to compute the
double layer. Here, we will demonstrate this on the $y$ coordinate of the
surface points:
=#
dl = zeros_grid(cache)
surface_divergence!(dl,pts.v,cache)
plot(dl,grid)

#=
## A curl double layer
We also sometimes need to take the curl of the regularized surface data,

$$\nabla\times \left( \delta(\chi) \mathbf{n} f \right)$$

For this, we use the [`surface_curl!`](@ref) operator. Let's demonstrate this
on the $x$ coordinate of the body data.
=#
gc = zeros_gridcurl(cache)
f = ones_surface(cache)
surface_curl!(gc,f,cache)
plot(gc,grid)

#md # ## Surface-grid operator functions
#md # ```@docs
#md # complementary_mask!
#md # complementary_mask
#md # mask!
#md # mask
#md # regularize!
#md # interpolate!
#md # regularize_normal!
#md # normal_interpolate!
#md # surface_curl!
#md # surface_divergence!
#md # surface_grad!
#md # ```
