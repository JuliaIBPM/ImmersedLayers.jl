import ConstrainedSystems: init

"""
    ODEFunctionList

## Constructor

`ODEFunctionList(state=nothing,constraint=nothing,ode_rhs=nothing,bc_rhs=nothing,constraint_force=nothing,bc_op=nothing,lin_op=nothing)`

Supply functions and data types for a constrained ODE system. The specified
functions must provide the various parts that comprise the constrained ODE system

``
\\dfrac{dy}{dt} = Ly - B_1 z + r_1(y,t)
``

``
B_2 y = r_2(t)
``

The functions can be in in-place or out-of-place form, but they must all be consistently
of the same form.

- `state =` specifies a prototype of the state vector ``y``.

- `constraint = ` specifies a prototype of the constraint force vector ``z``. If there are no constraints, this can be omitted.

- `ode_rhs =` specifies the right-hand side of the ODEs, ``r_1``. The in-place form of the function is `r1(dy,y,sys::ILMSystem,t)`, the state vector `y`, IL system `sys`, time `t`, and returning ``dy/dt``. The out-of-place form is `r1(y,sys,t)`.

- `bc_rhs = ` specifies the right-hand side of the boundary conditions, ``r_2``. If there are no constraints, this can be omitted. The in-place form is `r2(dz,sys,t)`, returning `dz`, the part of the boundary constraint not dependent on the state vector. The out-of-place form is `r2(sys,t)`.

- `constraint_force = ` supplies the constraint force term in the ODEs, ``B_1 z``. If there are no constraints, this can be omitted. The in-place form is `B1(dy,z,sys)`, returning the contribution to ``dy/dt`` (with the sign convention shown in the equations above) and the out-of-place form is `B1(z,sys)`.

- `bc_op = ` supplies the left-hand side of the boundary constraint, ``B_2 y``. If there are no constraints, this can be omitted. The in-place form is `B2(dz,y,sys)`, the out-of-place form is `B2(y,sys)`.

- `lin_op = ` is optional and specifies a linear operator on the state vector ``L``, to be treated with an exponential integral (i.e., integrating factor) in the time marching. (Alternatively, this part can simply be included in `r_1`). It should have an associated `mul!` operation that acts upon the state vector.

"""
struct ODEFunctionList{RT,LT,BRT,CFT,BOT,ST,CT}
    ode_rhs :: RT
    lin_op :: LT
    bc_rhs :: BRT
    constraint_force :: CFT
    bc_op :: BOT
    state :: ST
    constraint :: CT
end

function ODEFunctionList(;ode_rhs=nothing,lin_op=nothing,bc_rhs=nothing,constraint_force=nothing,bc_op=nothing,state=nothing,constraint=nothing)

    # Audit the supplied information to make sure it is consistent
    !(state isa Nothing)  || error("need to supply a state vector")
    !(ode_rhs isa Nothing)  || error("need to supply a rhs function for the ODE")
    !any(i -> i isa Nothing,[bc_rhs,constraint_force,bc_op,constraint]) || error("incomplete set of functions for constrained system")

    ODEFunctionList{typeof(ode_rhs),typeof(lin_op),typeof(bc_rhs),typeof(constraint_force),
                    typeof(bc_op),typeof(state),typeof(constraint)}(ode_rhs,lin_op,bc_rhs,constraint_force,bc_op,state,constraint)
end

function ImmersedLayers.ConstrainedODEFunction(sys::ILMSystem)
    @unpack extra_cache = sys
    @unpack f = extra_cache

    # Here is where we would add the additional RHS for moving bodies
    # along with the auxiliary state
    ConstrainedODEFunction(f.ode_rhs,f.bc_rhs,f.constraint_force,
                           f.bc_op,f.lin_op,_func_cache=zeros_sol(sys))
end

"""
    zeros_sol(sys::ILMSystem)

Return a zeroed version of the solution vector.
"""
function zeros_sol(sys::ILMSystem)
    # Need another version of this for moving bodies
    @unpack extra_cache = sys
    @unpack f = extra_cache
    return solvector(state=zero(f.state),constraint=zero(f.constraint))
end

_norm_sq(u) = dot(u,u)
_norm_sq(u::ConstrainedSystems.ArrayPartition) = sum(_norm_sq,u.x)
state_norm(u,t) = sqrt(_norm_sq(u))

"""
    ConstrainedSystems.init(u0,tspan,sys::ILMSystem,[alg=ConstrainedSystems.LiskaIFHERK()])

Initialize the integrator for a time-varying immersed-layer system of PDEs,
described in `sys`.
"""
function init(u0,tspan,sys::ILMSystem;alg=ConstrainedSystems.LiskaIFHERK(),kwargs...)
    @unpack timestep_func, phys_params, extra_cache,base_cache = sys
    @unpack g = base_cache
    @unpack f = extra_cache
    fode = ConstrainedODEFunction(sys)

    prob = ODEProblem(fode,u0,tspan,sys)
    dt_calc = timestep_func(g,phys_params)
    return init(prob, alg,dt=dt_calc,internal_norm=state_norm,kwargs...)
end
