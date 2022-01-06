# # A time-varying PDE with forcing

#md # ```@meta
#md # CurrentModule = ImmersedLayers
#md # ```

#=
In this example we will explore ways to apply forcing to a PDE. Our target
problem will be similar to the previous example -- transient heat conduction --
but now it will be in an unbounded domain. Our forcing will comprise local
*volumetric heating* (or rather, area heating, since this is two dimensional).
We will also include convection by an externally-imposed velocity field.

The governing equations are

$$\dfrac{\partial T}{\partial t} + \mathbf{v}\cdot\nabla T = \kappa \nabla^2 T + q$$

where $\mathbf{v}$ is a known convection velocity field and $q$ is a known heating field (scaled
by the density and specific heat)

In our discrete formulation, the problem takes the form

$$\mathcal{L}_C^\kappa T = -N(\mathbf{v},T) + q$$

where $N$ is a discrete approximation for the convection term $\mathbf{v}\cdot\nabla T$,
and $\mathcal{L}_C^\kappa = \mathrm{d}/\mathrm{d}t - \kappa L_C$ is the same
diffusion operator as in the previous example. Note that these equations are
no longer constrained. We only need to supply the right-hand side function.
Our job here is to provide everything needed to compute this right-hand side.

We will apply the heating inside of a local region. We supply the information about this
forcing in the `forcing` keyword argument. There are two pieces of information
we need to supply: the shape of the forcing region and a model to describe how the forcing is to be computed.

Then, in our `prob_cache`, we will use this information to call the `AreaRegion` function.
This creates a mask and other cache for the forcing. This mask is available
in the forcing model. This structure allows a considerable amount of freedom.

The convection velocity also must be provided. We make use of the `forcing`
keyword to supply this to the problem, as well, in the form of a function
that returns the current convection velocity at a given time `t`. The
convective derivative term `N` requires a special cache, which we also
generate in the `prob_cache`.

We will highlight these aspects in the example that follows:
=#

using ImmersedLayers
using Plots
using UnPack

#=
### Set up the grid
Let's set up the grid first before we go any further
=#
Δx = 0.01
Lx = 4.0
xlim = (-Lx/2,Lx/2)
ylim = (-Lx/2,Lx/2)
g = PhysicalGrid(xlim,ylim,Δx);

#=
### Specify the physical parameters
We supply here the diffusivity ($\kappa$) and the uniform convection velocity.
We also supply the grid Fourier number and a CFL number, which we will make
use of later to calculate the time step size.
=#
phys_params = Dict("diffusivity" => 0.005,
                    "velocity" => 0.5,
                     "Fourier" => 0.25,
                     "CFL" => 0.5)

#=
## Specifying the forcing region and model
We will create a circular region in which we apply the forcing. In this
region, we will apply a negative heat flux of strength 2. Note how
we use the mask to define this region. This mask takes the value 1 inside
the circle, and 0 outside. Also, note that we provide the current
temperature data, but don't actually use it. We will return to this
at the end of this example.
=#
fregion = Circle(0.2,1.4*Δx)

function heatflux_model!(dT,T,fr::AreaRegion,t)
    dT .= -2.0*mask(fr)
end

#=
## Specifying the convection velocity
We will provide a function that can be called to return the
convection velocity at a given time. Notice how we also pass in the
physical parameters. We can do this for the forcing model, too, if we desire.
=#
function set_convection_velocity!(v,t,phys_params)
    U∞ = phys_params["velocity"]
    v.u .= U∞
    v.v .= 0.0
    return v
end

#=
For convenience, we will pack the forcing and convection together into the *forcing* `Dict`:
=#
forcing = Dict("heating region" => fregion,
               "heating model" => heatflux_model!,
               "convection velocity model" => set_convection_velocity!)

#=
## Construct the ODE function and the extra cache
For the RHS of the heat conduction equation, we calculate the convective
derivative and the external heating.
=#
function heatconduction_rhs!(dT,T,sys::ILMSystem,t)
    @unpack forcing, phys_params, extra_cache, base_cache = sys
    @unpack cdcache, frcache, v, dT_tmp = extra_cache

    ## This provides the convection velocity at time `t`
    forcing["convection velocity model"](v,t,phys_params)

    ## Compute the convective derivative term `N(v,T)`
    fill!(dT_tmp,0.0)
    convective_derivative!(dT_tmp,v,T,base_cache,cdcache)
    dT .= -dT_tmp

    ## Compute the contribution from the forcing model to the right-hand side
    fill!(dT_tmp,0.0)
    forcing["heating model"](dT_tmp,T,frcache,t)
    dT .+= dT_tmp

    return dT
end

#=
We create a problem type for this, define the extra cache, and extend `prob_cache`
=#
@ilmproblem UnboundedHeatConduction scalar

struct UnboundedHeatConductionCache{VT,CDT,FRT,DTT,FT} <: AbstractExtraILMCache
   v :: VT
   cdcache :: CDT
   frcache :: FRT
   dT_tmp :: DTT
   f :: FT
end

function ImmersedLayers.prob_cache(prob::UnboundedHeatConductionProblem,
                                   base_cache::BasicILMCache{N,scaling}) where {N,scaling}
    @unpack phys_params, forcing = prob
    @unpack gdata_cache, g = base_cache

    ## Construct a Lapacian outfitted with the diffusivity
    κ = phys_params["diffusivity"]
    heat_L = Laplacian(base_cache,gdata_cache,κ)

    ## Create cache for the convective derivative
    v = zeros_gridgrad(base_cache)
    cdcache = ConvectiveDerivativeCache(base_cache)

    ## Create cache for the forcing area region
    frcache = AreaRegion(forcing["heating region"],base_cache)

    dT_tmp = zeros_grid(base_cache)

    ## The state here is temperature, and we just supply the RHS function
    f = ODEFunctionList(state = zeros_grid(base_cache),
                        ode_rhs=heatconduction_rhs!,
                        lin_op=heat_L)

    UnboundedHeatConductionCache(v,cdcache,frcache,dT_tmp,f)
end

#=
The last definition we need is for a timestep function. This time,
we take into account both the Fourier and the CFL conditions:
=#
function timestep_fourier_cfl(g,phys_params)
    κ = phys_params["diffusivity"]
    U∞ = phys_params["velocity"]
    Fo = phys_params["Fourier"]
    Co = phys_params["CFL"]
    Δt = min(Fo*cellsize(g)^2/κ,Co*cellsize(g)/U∞)
    return Δt
end

#=
## Set up the problem and system
This is similar to previous problems.
=#
prob = UnboundedHeatConductionProblem(g,scaling=GridScaling,
                                        phys_params=phys_params,
                                        forcing=forcing,
                                        timestep_func=timestep_fourier_cfl)

sys = construct_system(prob);

#=
## Solve the problem
As before, we first initialize the state, then we create an integrator,
and finally, advance the solution in time
=#
u0 = init_sol(sys)
tspan = (0.0,2.0)
integrator = init(u0,tspan,sys)

#=
Run the problem for one time unit
=#
step!(integrator,1.0)

#=
Let's see what this looks like. First, we will define the temperature
function, like we did last time:
=#
temperature(u,sys::ILMSystem,t) = state(u)
@snapshotoutput temperature

plot(temperature(integrator),sys)

#=
We can also make a movie
=#
sol = integrator.sol
@gif for t in sol.t
    plot(temperature(sol,sys,t),sys)
end every 5


#=
There are plenty of other choices we could make in this problem. For
example, we could have used a heat transfer model for the forcing,
providing heat that is proportional to the different between the
local temperature and some prescribed temperature. Here,
we set that temperature to 1, and the heat transfer coefficient to 2.
We still make use of the mask to localize the heating:
=#
function heatflux_model!(dT,T,fr::AreaRegion,t)
    dT .= 2.0*mask(fr)*(1.0 .- T)
end

#=
We don't have to regenerate the system. Just run it again!
=#
integrator = init(u0,tspan,sys)
step!(integrator,1.0)

sol = integrator.sol
@gif for t in sol.t
    plot(temperature(sol,sys,t),sys)
end every 5
