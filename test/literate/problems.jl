# # Problems and the system

#md # ```@meta
#md # CurrentModule = ImmersedLayers
#md # ```

#=
In specific problems that we wish to solve with immersed layers, there may
be other data and operators that we would like to cache. We do this with
an *extra cache*, which the user can define, along with a problem type associated
with this cache. The basic cache and the extra cache are generated and associated
together in a *system*.

There are a few basic ingredients to do this:
* **Create a problem type**, making it a subtype of `AbstractScalarILMProblem` or `AbstractVectorILMProblem`. This mostly just serves as a means of dispatching correctly, but also should hold the grid and bodies so they can be passed along when the caches are constructed.
* **Create an extra cache type**, making it a subtype of `AbstractExtraILMCache`. This can hold pretty much anything you want it to.
* **Extend the function `prob_cache(prob,base_cache)`** to serve as a constructor for your extra cache, when your problem type is passed in.

Optionally, you can also extend the function `solve` in order to perform the steps
of your algorithm. However, generically, you can just use pass in the system structure,
which holds the basic ILM cache and your extra cache, into any function.
=#

#=
## Example of problem and system use
We will demonstrate the use of problems and systems with the example
given in [A Dirichlet Poisson problem](@ref). Here, we will
assemble the various additional data structures and operators used to
solve this problem into an extra cache. We will also create a problem
type called `DirichletPoissonProblem`, which we make a subtype of
`AbstractScalarILMProblem`.
=#

using ImmersedLayers
using CartesianGrids
using RigidBodyTools
using Plots
using UnPack


#=
### Create your problem type
You can copy this as a template, because it Usually doesn't need
to be any different from this.
=#
struct DirichletPoissonProblem{DT,ST} <: ImmersedLayers.AbstractScalarILMProblem{DT,ST}
   g :: PhysicalGrid
   bodies :: BodyList
   DirichletPoissonProblem(g::PT,bodies::BodyList;ddftype=CartesianGrids.Yang3,scaling=IndexScaling) where {PT} = new{ddftype,scaling}(g,bodies)
   DirichletPoissonProblem(g::PT,body::Body;ddftype=CartesianGrids.Yang3,scaling=IndexScaling) where {PT} = new{ddftype,scaling}(g,BodyList([body]))
end

#=
### Create your extra cache
Here, we'd like this extra cache to hold the Schur complement and the
filtering matrices, as well as some cache variables.
=#
struct DirichletPoissonCache{SMT,CMT,ST,FT} <: ImmersedLayers.AbstractExtraILMCache
   S :: SMT
   C :: CMT
   fb :: ST
   fstar :: FT
end

#=
### Extend prob_cache
We need this to construct our extra cache
=#
function ImmersedLayers.prob_cache(prob::DirichletPoissonProblem,base_cache::BasicILMCache)
    S = create_RTLinvR(base_cache)
    C = create_surface_filter(base_cache)
    fb = zeros_surface(base_cache)
    fstar = zeros_grid(base_cache)
    DirichletPoissonCache(S,C,fb,fstar)
end

#=
### Extend the `solve` function
Here, we actually do the work of the algorithm, making use of all of the
operators and data structures that we have cached for efficiency.
The example below takes in some surface Dirichlet data `fbplus`,
and returns the solutions `f` and `s` (filtered).
=#
function ImmersedLayers.solve(fbplus,prob::DirichletPoissonProblem,sys::ILMSystem)
    @unpack extra_cache, base_cache = sys
    @unpack S, C, fb, fstar = extra_cache

    f = zeros_grid(base_cache)
    s = zeros_surface(base_cache)

    surface_divergence!(fstar,fbplus,base_cache)
    fb .= 0.5*fbplus

    inverse_laplacian!(fstar,base_cache)

    interpolate!(s,fstar,base_cache)
    s .= fb - s
    s .= -(S\s);

    regularize!(f,s,base_cache)
    inverse_laplacian!(f,base_cache)
    f .+= fstar;

    return f, C^6*s
end

#=
### Set up the grid, shape, and cache
We do this just as we did in [Immersed layer caches](@ref), but
now we don't create a cache, since it will be done internally.
=#
Δx = 0.01
Lx = 4.0
xlim = (-Lx/2,Lx/2)
ylim = (-Lx/2,Lx/2)
grid = PhysicalGrid(xlim,ylim,Δx)
RadC = 1.0
Δs = 1.4*cellsize(grid)
body = Circle(RadC,Δs)

#=
### Do the work
We do this in three steps:
- Create the problem instance
- Call `__init` to create the caches, assembled into a system
- Call `solve` to solve the problem.
Also, note that pretty much any function that accepts `base_cache`
as an argument also accepts `sys`.
=#
prob = DirichletPoissonProblem(grid,body,scaling=GridScaling)
sys = ImmersedLayers.__init(prob)

pts = points(sys)
f, s = solve(pts.u,prob,sys)

plot(f,sys)
#-
plot(s)

#md # ## Problem types and functions
#md #
#md # ```@docs
#md # AbstractScalarILMProblem
#md # AbstractVectorILMProblem
#md # BasicScalarILMProblem
#md # BasicVectorILMProblem
#md # prob_cache
#md # ```

#md # ## System types and functions
#md #
#md # ```@docs
#md # ILMSystem
#md # __init
#md # ```
