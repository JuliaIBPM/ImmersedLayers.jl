# # A Neumann Poisson problem

#md # ```@meta
#md # CurrentModule = ImmersedLayers
#md # ```

#=
Here, we'll demonstrate a solution of Laplace's equation,
with Neumann boundary conditions on a surface. Similar to the
Dirichlet problem in [A Dirichlet Poisson problem](@ref), we will solve the
problem with one Neumann condition external to the surface,
and another Neumann value internal to the surface.

Our underlying problem is still

$$\nabla^2\varphi^+ = 0,\qquad \nabla^2\varphi^- = 0$$

where $+$ denotes the exterior and $-$ the interior of the surface. (We will
consider a circle of radius 1.) The boundary conditions on this surface are

$$\mathbf{n}\cdot\nabla\varphi^+ = v^+_n, \qquad \mathbf{n}\cdot\nabla\varphi^- = v^-_n$$

In other words, we seek to set the value on the exterior normal derivative to $v_n$
of the local normal vector on the surface, while the interior should have zero normal
derivative.

Discretizing this problem by the usual techniques, we seek to solve

$$\begin{bmatrix} L & D_s \\ G_s & R_n^T R_n \end{bmatrix} \begin{pmatrix} f \\ -d \end{pmatrix} = \begin{pmatrix} R s \\ \overline{v}_n \end{pmatrix}$$

where $\overline{v}_n = (v^+_n + v^-_n)/2$ and $s = v^+_n - v^-_n$. The resulting
$d$ is $f^+-f^-$.

As with the Dirichlet problem, this saddle-point problem can be solved by block-LU decomposition. First solve

$$L f^{*} = R s$$

for $f^*$. Then solve

$$-S d = \overline{v}_n - G_s f^{*}$$

for $d$, where $S = R_n^T R_n - G_s L^{-1} D_s = -C_s L^{-1}C_s^T$, and finally, compute

$$f = f^{*} + L^{-1}D_s d$$

We will demonstrate these steps here.
=#

using ImmersedLayers
using Plots
using LinearAlgebra
using UnPack

#=
## Set up the extra cache and solve function
=#
#=
The problem type takes the usual basic form
=#
struct NeumannPoissonProblem{DT,ST} <: AbstractScalarILMProblem{DT,ST}
   g :: PhysicalGrid
   bodies :: BodyList
   NeumannPoissonProblem(g::PT,bodies::BodyList;ddftype=CartesianGrids.Yang3,scaling=IndexScaling) where {PT} = new{ddftype,scaling}(g,bodies)
   NeumannPoissonProblem(g::PT,body::Body;ddftype=CartesianGrids.Yang3,scaling=IndexScaling) where {PT} = new{ddftype,scaling}(g,BodyList([body]))
end

#=
The extra cache holds additional intermediate data, as well as
the Schur complement. We don't bother creating a filtering matrix here.
=#
struct NeumannPoissonCache{SMT,ST,FT} <: AbstractExtraILMCache
   S :: SMT
   vn :: ST
   fstar :: FT
end
#=
The function `prob_cache`, as before, constructs the operators and extra
cache data structures
=#
function ImmersedLayers.prob_cache(prob::NeumannPoissonProblem,base_cache::BasicILMCache)
    S = create_CLinvCT(base_cache)
    vn = zeros_surface(base_cache)
    fstar = zeros_grid(base_cache)
    NeumannPoissonCache(S,vn,fstar)
end

#=
And finally, here's the steps we outlined above, used to
extend the `solve` function
=#
function ImmersedLayers.solve(vnplus,vnminus,prob::NeumannPoissonProblem,sys::ILMSystem)
    @unpack extra_cache, base_cache = sys
    @unpack S, vn, fstar = extra_cache

    f = zeros_grid(base_cache)
    d = zeros_surface(base_cache)

    regularize!(fstar,vnplus-vnminus,base_cache)
    vn .= 0.5*(vnplus+vnminus)

    inverse_laplacian!(fstar,base_cache)

    surface_grad!(d,fstar,base_cache)
    d .= vn - d
    d .= -(S\d);

    surface_divergence!(f,d,base_cache)
    inverse_laplacian!(f,base_cache)
    f .+= fstar;

    return f, d
end
nothing #hide

#=
## Solve the problem
Here, we will demonstrate the solution on a circular shape of radius 1,
with $v_n^+ = n_x$ and $v_n^- = 0$. This is actually the set of conditions
used to compute the unit scalar potential field (and, as we will see, the added mass) in potential flow.
=#

Δx = 0.01
Lx = 4.0
xlim = (-Lx/2,Lx/2)
ylim = (-Lx/2,Lx/2)
g = PhysicalGrid(xlim,ylim,Δx)
Δs = 1.4*cellsize(g)
body = Circle(1.0,Δs);

#=
Create the system
=#
prob = NeumannPoissonProblem(g,body,scaling=GridScaling)
sys = ImmersedLayers.__init(prob)
nothing #hide

#=
Set the boundary values
=#
nrm = normals(sys)
vnplus = zeros_surface(sys)
vnminus = zeros_surface(sys)
vnplus .= nrm.u
nothing #hide

#=
Solve it
=#
solve(vnplus,vnminus,prob,sys) #hide
@time f, d = solve(vnplus,vnminus,prob,sys);

#=
and plot the field
=#
plot(f,sys,levels=30)

#=
and the Lagrange multiplier field, $d$, on the surface
=#
plot(d)

#=
## Multiple bodies
The cache and solve function we created above can be applied for
any body or set of bodies. Let's apply it here to a circle of radius 0.25 inside of a
square of half-side length 2, where our goal is to find the effect of the enclosing square
on the motion of the circle. As such, we will set the Neumann conditions
both internal to and external to the square to be 0, but for the exterior of the
circle, we set it to $n_x$.
=#
bl = BodyList();
push!(bl,Square(1.0,Δs))
push!(bl,Circle(0.25,Δs))
nothing #hide

#=
We don't actually have to transform these shapes, but it is
illustrative to show how we would move them.
=#
t1 = RigidTransform((0.0,0.0),0.0)
t2 = RigidTransform((0.0,0.0),0.0)
tl = RigidTransformList([t1,t2])
tl(bl)
nothing #hide

#=
Create the problem and system
=#
prob = NeumannPoissonProblem(g,bl,scaling=GridScaling)
sys = ImmersedLayers.__init(prob)
nothing #hide

#=
Set the boundary conditions. We set only the exterior Neumann value
of body 2 (the circle), using [`copyto!`](@ref)
=#
nrm = normals(sys)
vnplus = zeros_surface(sys)
vnminus = zeros_surface(sys)
copyto!(vnplus,nrm.u,sys,2)

#=
Solve it and plot
=#
f, d  = solve(vnplus,vnminus,prob,sys)
plot(f,sys)

#=
Now, let's compute the added mass components of the circle associated
with this motion. We are approximating

$$M = \int_{C_2} f^+ \mathbf{n}\mathrm{d}s$$

where $C_2$ is shape 2 (the circle), and $f^+$ is simply $d$ on body 2.
=#
M = integrate(d∘nrm,sys,2)
