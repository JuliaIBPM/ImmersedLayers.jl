# # Immersed layer caches


#md # ```@meta
#md # CurrentModule = ImmersedLayers
#md # ```

#=
This package uses precomputed caches to efficiently implement the immersed layer
operators. The starting point for a cache is the specification of the
(discretized) body shape and the grid. Let's use an example.
=#

# ## Setting up a cache

using ImmersedLayers
#!jl using Plots
using LinearAlgebra

#=
### Set up a grid
First, we will set up a grid for performing the operations. We use the `PhysicalGrid`
constructor of the `CartesianGrids.jl` package to create this.
=#
Δx = 0.01
Lx = 4.0
xlim = (-Lx/2,Lx/2)
ylim = (-Lx/2,Lx/2)
g = PhysicalGrid(xlim,ylim,Δx)


#=
### Set the shape
Now let's set a shape to immerse into the grid. We will use a circle, but
there are a variety of other shapes available. Many of these are in the
`RigidBodyTools.jl` package. Note that we set the spacing between the points
on this shape equal to 1.4 times the grid spacing. This is not critical, but it
is generally best to set it to a value between 1 and 2.
=#
RadC = 1.0
Δs = 1.4*cellsize(g)
body = Circle(RadC,Δs)


#=
### Create the cache
After setting the grid and the surface to immerse, the next step for using the
immersed layer tools is to set up a *surface cache*. This allocates a set of
data structures, as well as the critical *regularization* and *interpolation*
operators that will get used.

There are a few choices to make when setting this up
* **What kind of data (scalar or vector) are we dealing with?**

We will demonstrate with scalar data, which means we use the [`SurfaceScalarCache`](@ref)
function. For vector data, use [`SurfaceVectorCache`](@ref).

* **What type of scaling (grid or index) do we wish to apply to the operators?**

Grid scaling, set with `scaling = GridScaling`, means that the various operators are scaled with the physical grid spacing
and/or surface point spacing so that they approximate the continuous operators. This
means that regularization and interpolation are transposes with respect to
inner products that incorporate these physical spacings, rather than the usual
linear algebra inner products. Also, differential operations on the grid are true
approximations of their continuous counterparts. This choice of scaling is usually the best, and
the [`dot`](@ref) operator is extended in this package to implement the physically-
scaled inner products.

On the other hand, `scaling = IndexScaling` does not scale these, but rather, uses
pure differencing for the grid differential operators, and regularization is
the simple matrix transpose of interpolation.

* **What discrete Dirac delta function (DDF) do we wish to use?**

This is specified with the `ddftype` keyword argument. The default is
`ddftype=CartesianGrids.Yang3` [^1]. However, there are
other choices, such as `CartesianGrids.Roma`, `CartesianGrids.Goza`, `CartesianGrids.Witchhat`,
`CartesianGrids.M3`, `CartesianGrids.M4prime`.
=#

cache = SurfaceScalarCache(body,g,scaling=GridScaling)


#=
## Plotting the immersed points
We can plot the immersed points with the `plot` function of the `Plots.jl`
package, using a special plot recipe. This accepts all of the
usual attribute keywords of the basic plot function.
=#
#!jl plot(cache,xlims=(-2,2),ylims=(-2,2))

#=
In this plotting demonstration, we have used the `points` function
to obtain the coordinates of the immersed points. Other useful
utilities are `normals`, `areas`, and `arcs`, e.g.
=#
normals(cache)

#=
## Some basic utilities
We will see deeper uses of this cache in [Surface-grid operations](@ref).
However, for now we can learn how to get basic copies of the grid
and surface data. For example, a copy of the grid data, all initialized to zero:
=#
zeros_grid(cache)

#=
or similarly on the surface
=#
zeros_surface(cache)

#=
We also might need to initialize data on the grid to accept the curl of
a vector field. This is used in conjunction with [`surface_curl!`](@ref),
for example.
=#
zeros_gridcurl(cache)

#=
If we want them initialized to unity, then use
=#
ones_grid(cache)

# and
ones_surface(cache)

#=
To evaluate functions on the grid, it is useful to be able to
fill grid data with the x and y coordinates. For this, we use
=#
x_grid(cache)
#-
y_grid(cache)



#=
## Using norms, inner products, and integrals
It is useful to compute norms and inner products on grid or surface data.
These tools are easily accessible and intuitive, e.g., `dot(u,v,cache)` and `norm(u,cache)`,
and they respect the scaling associated with the cache. For example,
the following gives an approximation of the circumference of the circle:
=#
os = ones_surface(cache)
dot(os,os,cache)

#=
We can also numerically integrate on surfaces or grids. For example,
the previous example could have been written
=#
integrate(os,cache)

#=
Or, to compute an area integral over the whole domain,
=#
og = ones_grid(cache)
integrate(og,cache)

#=
If the underlying data are of vector or tensor form, then
the `integrate` function operates on every component
=#
ovg = ones_gridgrad(cache)
integrate(ovg,cache)


#md # ## Cache types and constructors

#md # ```@docs
#md # BasicILMCache
#md # SurfaceScalarCache
#md # SurfaceVectorCache
#md # AbstractExtraILMCache
#md # ```

#md # ## Utilities for creating instances of data

#md # ```@docs
#md # similar_grid
#md # similar_gridgrad
#md # similar_gridcurl
#md # similar_griddiv
#md # similar_gridgradcurl
#md # similar_surface
#md # similar_surfacescalar
#md # zeros_grid
#md # zeros_gridgrad
#md # zeros_gridcurl
#md # zeros_griddiv
#md # zeros_gridgradcurl
#md # zeros_surface
#md # zeros_surfacescalar
#md # ones_grid
#md # ones_gridgrad
#md # ones_gridcurl
#md # ones_griddiv
#md # ones_gridgradcurl
#md # ones_surface
#md # ones_surfacescalar
#md # x_grid
#md # y_grid
#md # x_gridcurl
#md # y_gridcurl
#md # x_griddiv
#md # y_griddiv
#md # x_gridgrad
#md # y_gridgrad
#md # evaluate_field!
#md # ```

#md # ## Utilities for accessing surface information

#md # ```@docs
#md # areas(::BasicILMCache)
#md # normals(::BasicILMCache)
#md # points(::BasicILMCache)
#md # arcs(::BasicILMCache)
#md # ```

#md # ## Inner products, norms, and integrals

#md # ```@docs
#md # dot(::GridData,::GridData,::BasicILMCache)
#md # dot(::PointData,::PointData,::BasicILMCache)
#md # norm(::GridData,::BasicILMCache)
#md # integrate(::GridData,::BasicILMCache)
#md # norm(::PointData,::BasicILMCache)
#md # integrate(::PointData{N},::BasicILMCache{N}) where {N}
#md # dot(::PointData,::PointData,::BasicILMCache,::Int)
#md # norm(::PointData,::BasicILMCache,::Int)
#md # integrate(::PointData{N},::BasicILMCache{N},::Int) where {N}
#md # ```

#md # ## Other cache utilities
#md #
#md # ```@docs
#md # view(::PointData,::BasicILMCache,::Int)
#md # copyto!(::PointData,::PointData,::BasicILMCache,::Int)
#md # copyto!(::ScalarData,::AbstractVector,::BasicILMCache,::Int)
#md # Laplacian(::BasicILMCache,::Any)
#md # RegularizationMatrix(::BasicILMCache,::PointData,::GridData)
#md # InterpolationMatrix(::BasicILMCache,::GridData,::PointData)
#md # ```

#md # [^1]: Yang, X., et al., (2009) "A smoothing technique for discrete delta functions with application to immersed boundary method in moving boundary simulations," J. Comput. Phys., 228, 7821--7836.
