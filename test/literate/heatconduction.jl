# # Setting up a time-varying PDE

#md # ```@meta
#md # CurrentModule = ImmersedLayers
#md # ```

#=
In this example we will demonstrate the use of the package on a time-dependent
PDE, a problem of unsteady heat conduction. We will use the package to
solve for the interior diffusion of temperature from a circle held at constant
temperature.

We seek to solve the heat conduction equation with Dirichlet boundary conditions

$$\dfrac{\partial T}{\partial t} = \kappa \nabla^2 T + q + \delta(\chi) \sigma  - \kappa \nabla\cdot \left( \delta(\chi) \mathbf{n} [T] \right)$$

subject to $T = T_b$ on the immersed surface. We might be solving this external
to a surface, or it might be internal. The quantity $\sigma$ is the Lagrange
multiplier. In this context, it is the heat flux through the surface.

In the spatially discrete formulation, the problem takes the form

$$\begin{bmatrix}
\mathcal{L}_C^\kappa & R_C \\ R_C^T & 0
\end{bmatrix}\begin{pmatrix}
T \\ -\sigma
\end{pmatrix} =
\begin{pmatrix}
q - \kappa D_s [T] \\ (T^+_b + T^-_b)/2
\end{pmatrix}$$

where $\mathcal{L}_C^\kappa = \mathrm{d}/\mathrm{d}t - \kappa L_C$, where $[T] = T_b^+ - T_b^-$
is the jump in temperature across the surface. As in the time-independent problems,
we can specify whether we are solving it external or internal to a surface by setting
the boundary value to zero in the other region. However, in contrast to the
time-independent problems, we have to advance this problem in time.
The system above has the form of a *constrained ODE system*, which the `ConstrainedSystems.jl` package treats.
We will make use of this package in the example below.

To support this, there are a few additional steps in our setup of the problem:
- we (as the implementers of the PDE) need to specify the functions that calculate the
   various parts of this constrained ODE system.
- we (as the users of this implementation) need to specify the time step size,
   the initial conditions, the time integration range, and create the *integrator*
   to advance the solution.

The latter of these is very easy, as we'll find. Most of our attention will
be on the first part: how to set up the constrained ODE system. For this,
we will make use of the `ODEFunctionList`, which assembles the
various functions and operators into a `ConstrainedODEFunction`, to be used by the
`ConstrainedSystems.jl` package.
=#

using ImmersedLayers
using Plots
using UnPack

#=
## Set up the constrained ODE system operators
=#
#=
The problem type is generated with the usual macro call. In this example,
we will make use of more of the capabilities of the resulting problem
constructor for "packing" it with information about the problem.
=#
@ilmproblem DirichletHeatConduction scalar

#=
The constrained ODE system requires us to provide functions that calculate
the RHS of the ODE, the RHS of the constraint equation, the Lagrange multiplier force
term in the ODE, and the action of the boundary operator on the state vector.
(You can see the generic form of the system by typing `?ConstrainedODEFunction`)
As you will see, in this example these are *in-place* operators: their
first argument holds the result, which is changed (i.e., mutated)
by the function.
=#
#=
Below, we construct the function that calculates the RHS of the heat conduction ODE.
We have omitted the volumetric heat flux here, supplying only the double-layer
term. Note how this makes use of the physical parameters in `phys_params`
and the boundary data via functions in `bc`. The functions for the boundary
data supply the boundary values. Also, note that the function returns `dT`
in the first argument. This represents this function's contribution to $dT/dt$.
=#
function heatconduction_ode_rhs!(dT,T,sys::ILMSystem,t)
    @unpack bc, forcing, phys_params, extra_cache, base_cache = sys
    @unpack dTb, Tbplus, Tbminus = extra_cache

    κ = phys_params["diffusivity"]

    ## Calculate the double-layer term
    fill!(dT,0.0)
    Tbplus .= bc["Tbplus"](base_cache,t)
    Tbminus .= bc["Tbminus"](base_cache,t)
    dTb .= Tbplus - Tbminus
    surface_divergence!(dT,-κ*dTb,sys)

    return dT
end

#=
Now, we create the function that calculates the RHS of the boundary condition.
For this Dirichlet condition, we simply take the average of the interior
and exterior prescribed values. The first argument `dTb` holds the result.
=#
function heatconduction_bc_rhs!(dTb,sys::ILMSystem,t)
    @unpack bc, extra_cache, base_cache = sys
    @unpack Tb, Tbplus, Tbminus = extra_cache

    Tbplus .= bc["Tbplus"](base_cache,t)
    Tbminus .= bc["Tbminus"](base_cache,t)
    dTb .= 0.5*(Tbplus + Tbminus)

    return dTb
end

#=
This function calculates the contribution to $dT/dt$ from the Lagrange
multiplier (the input σ). Here, we simply regularize the negative of this
to the grid.
=#
function heatconduction_constraint_force!(dT,σ,sys::ILMSystem)
    @unpack extra_cache, base_cache = sys

    fill!(dT,0.0)
    regularize!(dT,-σ,sys)

    return dT
end

#=
Now, we provide the transpose term of the previous function: a function that
interpolates the temperature (state vector) onto the boundary. The first argument `dTb`
holds the result.
=#
function heatconduction_bc_op!(dTb,T,sys::ILMSystem)
    @unpack extra_cache, base_cache = sys

    fill!(dTb,0.0)
    interpolate!(dTb,T,sys)

    return dTb
end

#=
## Set up the extra cache and extend `prob_cache`
Here, we construct an extra cache that holds a few extra intermediate
variables, used in the routines above. But this cache also, crucially, holds
the functions and operators of the constrained ODE function. We call
the function `ODEFunctionList` to assemble these together.

The `prob_cache` function creates this ODE function, supplying the functions that we just defined. We
also create a Laplacian operator with the heat diffusivity built into it.
(This operator is singled out from the other terms in the heat conduction
equation, because we account for it separately in the time marching
using a matrix exponential.) We also create *prototypes* of the *state* and *constraint
force* vectors. Here, the state is the grid temperature data and the constraint
is the Lagrange multipliers on the boundary.
=#
struct DirichletHeatConductionCache{DTT,TBT,TBP,TBM,FT} <: AbstractExtraILMCache
   dTb :: DTT
   Tb :: TBT
   Tbplus :: TBP
   Tbminus :: TBM
   f :: FT
end

function ImmersedLayers.prob_cache(prob::DirichletHeatConductionProblem,
                                   base_cache::BasicILMCache{N,scaling}) where {N,scaling}
    @unpack phys_params = prob
    @unpack gdata_cache, g = base_cache

    dTb = zeros_surface(base_cache)
    Tb = zeros_surface(base_cache)
    Tbplus = zeros_surface(base_cache)
    Tbminus = zeros_surface(base_cache)

    ## Construct a Lapacian outfitted with the diffusivity
    κ = phys_params["diffusivity"]
    heat_L = Laplacian(base_cache,gdata_cache,κ)

    ## State (grid temperature data) and constraint (surface Lagrange multipliers)
    f = ODEFunctionList(state = zeros_grid(base_cache),
                        constraint = zeros_surface(base_cache),
                        ode_rhs=heatconduction_ode_rhs!,
                        lin_op=heat_L,
                        bc_rhs=heatconduction_bc_rhs!,
                        constraint_force = heatconduction_constraint_force!,
                        bc_op = heatconduction_bc_op!)

    DirichletHeatConductionCache(dTb,Tb,Tbplus,Tbminus,f)
end

#=
Before we move on to solving the problem, we need to set up a function
that will calculate the time step size. The time marching algorithm will
call this function. Of course, this could just be used to specify a
time step directly, e.g., by supplying it in `phys_params`. But it
is better to use a stability condition (a Fourier condition) to determine
it based on the other data.
=#
function timestep_fourier(g,phys_params)
    κ = phys_params["diffusivity"]
    Fo = phys_params["Fourier"]
    Δt = Fo*cellsize(g)^2/κ
    return Δt
end


#=
## Solve the problem
We will solve heat conduction inside a circular region with
uniform temperature, with thermal diffusivity equal to 1.
=#

#=
### Set up the grid
=#
Δx = 0.01
Lx = 4.0
xlim = (-Lx/2,Lx/2)
ylim = (-Lx/2,Lx/2)
g = PhysicalGrid(xlim,ylim,Δx);

#=
### Set up the body shape.
Here, we will demonstrate the solution on a circular shape of radius 1.
=#
Δs = 1.4*cellsize(g)
body = Circle(1.0,Δs);

#=
### Specify the physical parameters, data, etc.
These can be changed later without having to regenerate the system.
=#

#=
Here, we create a dict with physical parameters to be passed in.
=#
phys_params = Dict("diffusivity" => 1.0, "Fourier" => 0.25)

#=
The temperature boundary functions on the exterior and interior are
defined here and assembled into a dict.
=#
get_Tbplus(base_cache,t) = zeros_surface(base_cache)
get_Tbminus(base_cache,t) = ones_surface(base_cache)
bcdict = Dict("Tbplus" => get_Tbplus,"Tbminus" => get_Tbminus)

#=
Construct the problem, passing in the data and functions we've just
created.
=#
prob = DirichletHeatConductionProblem(g,body,scaling=GridScaling,
                                             phys_params=phys_params,
                                             bc=bcdict,
                                             timestep_func=timestep_fourier);

#=
Construct the system
=#
sys = construct_system(prob);

#=
### Solving the problem
In contrast to the previous (time-independent) example, we have not
extended the `solve` function here to serve us in solving this problem.
Instead, we rely on the tools in `ConstrainedSystems.jl` to advance
the solution forward in time. This package builds from the `OrdinaryDiffEq.jl`
package, and leverages most of the tools of that package.
=#

#=
Set an initial condition. Here, we just get a zeroed copy of the
solution prototype that we have stored in the extra cache. We also
get the time step size for our own use.
=#
u0 = zeros_sol(sys)
Δt = timestep_fourier(g,phys_params)

#=
It is instructive to note that `u0` has two parts: a *state* and a *constraint*,
each obtained respectively with a convenience function. The state in this
case is the temperature; the constraint is the Lagrange multiplier.
=#
state(u0)
#-
constraint(u0)

#=
Now, create the integrator, with a time interval of 0 to 1. We have not
specified the algorithm here explicitly; it defaults to the `LiskaIFHERK`
time-marching algorithm, which is a second-order algorithm for constrained
ODE systems that utilizes the matrix exponential (i.e., integrating factor)
for the linear part of the problem. Another choice is the first-order
Euler method, `IFHEEuler`, which one can specify by adding `alg=ConstrainedSystems.IFHEEuler()`
=#
tspan = (0.0,1.0)
integrator = init(u0,tspan,sys)

#=
Now advance the solution by 1 convective time unit, by using the `step!` function,
which steps through the solution.
=#
step!(integrator,1.0)

#=
### Plot the solution
The integrator holds the most recent solution in the field `u`, which
has the same type as our initial condition `u0`. Here, we plot the state of the system at the end of the interval.
=#
plot(state(integrator.u),sys)

#=
It would be nice to just define a function called `temperature` to get this
more explicitly. We will do that here, and also apply a macro `@snapshotoutput` that
automatically extends this function with some convenient interfaces. For example,
if we simply pass in the integrator to `temperature`, it will pick off the `u`
field for us.
=#
temperature(u,sys::ILMSystem,t) = state(u)
@snapshotoutput temperature

#=
Now we can write
=#
plot(temperature(integrator),sys)

#=
The solution history is in the field `integrator.sol`. The macro we
called earlier enables temperature to work for this, as well, and
we can obtain the temperature at *any* time in the interval of our solution.
For example, to get the solution at time 0.51:
=#
sol = integrator.sol
plot(temperature(sol,sys,0.51),sys)

#=
We can also get it for an array of times:
=#
temperature(sol,sys,0.5:0.01:0.6)

#md # ## Time-varying PDE functions

#md # ```@docs
#md # ODEFunctionList
#md # zeros_sol
#md # init
#md # state
#md # constraint
#md # ```
