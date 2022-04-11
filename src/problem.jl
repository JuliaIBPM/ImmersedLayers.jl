abstract type AbstractILMProblem{DT,ST,DTP} end


function regenerate_problem() end

"""
$(TYPEDEF)

When defining a problem type with scalar data, make it a subtype of this.
"""
abstract type AbstractScalarILMProblem{DT,ST,DTP} <: AbstractILMProblem{DT,ST,DTP} end

"""
$(TYPEDEF)

When defining a problem type with vector data, make it a subtype of this.
"""
abstract type AbstractVectorILMProblem{DT,ST,DTP} <: AbstractILMProblem{DT,ST,DTP} end

"""
The `@ilmproblem` macro is used to automatically generate a type
particular to an immersed-layer problem, which can then be used for dispatch on
those types of problems. It takes two arguments: the name of the problem
(to which `Problem` will be automatically appended), and whether the problem
is of scalar or vector type. For example,

    @ilmproblem(StokesFlow,vector)

would generate a type `StokesFlowProblem` dealing with vector-valued data. The
resulting type then automatically has a constructor that allows one to pass
in the grid information and bodies, as well as optional choices for the
DDF function and the scaling type. For the example, this constructor would be

    StokesFlowProblem(grid,bodies[,ddftype=CartesianGrids.Yang3][,scaling=GridScaling])

Note that there is another constructor for problems with no surfaces that only
requires that the grid information be passed, e.g.,

    StokesFlowProblem(grid)

There are several keyword arguments for the problem constructor

- `ddftype = ` to set the DDF type. The default is `Yang3`.
- `scaling = ` to set the scaling type, `GridScaling` (default) or `IndexScaling`.
- `dtype = ` to set the element type to `Float64` (default) or `ComplexF64`.
- `phys_params = ` to pass in physical parameters
- `bc = ` to pass in boundary condition data or functions
- `forcing = ` to pass in forcing functions or data
- `motions = ` to provide function(s) that specify the velocity of the immersed surface(s). Note: if this keyword is used, it is assumed that surfaces will move.
- `timestep_func =` to pass in a function for time-dependent problems that provides the time-step size.
                  It is expected that this function takes in two arguments,
                  the `grid::PhysicalGrid` and `phys_params`, and returns the time step. It is up to the
                  user to decide how to determine this. It could also simply return a
                  constant value, regardless of the arguments.

"""
macro ilmproblem(name,vector_or_scalar)
    vs_string = string(vector_or_scalar)
    abtype = Symbol("Abstract"*uppercasefirst(vs_string)*"ILMProblem")
    typename = Symbol(string(name)*"Problem")
    return esc(quote

          @doc """
              $($typename)

          ILM problem type dealing with $($vs_string)-type data.
          """
          struct $typename{DT,ST,DTP,BLT,PHT,BCF,FF,DTF,MTF} <: $abtype{DT,ST,DTP}
             g :: PhysicalGrid
             bodies :: BLT
             phys_params :: PHT
             bc :: BCF
             forcing :: FF
             timestep_func :: DTF
             motions :: MTF
             $typename(g::PT,bodies::BodyList;ddftype=ImmersedLayers.DEFAULT_DDF,
                                              scaling=ImmersedLayers.DEFAULT_SCALING,
                                              dtype=ImmersedLayers.DEFAULT_DATA_TYPE,
                                              phys_params=nothing,
                                              bc=nothing,
                                              forcing=nothing,
                                              timestep_func=nothing,
                                              motions=nothing) where {PT <: PhysicalGrid} =
                    new{ddftype,scaling,dtype,typeof(bodies),typeof(phys_params),typeof(bc),typeof(forcing),typeof(timestep_func),
                                        typeof(ImmersedLayers._list(motions))}(
                                              g,bodies,phys_params,bc,forcing,timestep_func,ImmersedLayers._list(motions))
          end

          $typename(g::PhysicalGrid,body::Body;kwargs...) = $typename(g,BodyList([body]); kwargs...)
          $typename(g::PhysicalGrid;kwargs...) = $typename(g,BodyList(); kwargs...)

          ImmersedLayers.regenerate_problem(sys::ILMSystem{S,P},bl::BodyList) where {S,P<:$typename} =
                                      $typename(sys.base_cache.g,bl,
                                                ddftype=ImmersedLayers._ddf_type(P),
                                                scaling=ImmersedLayers._scaling_type(P),
                                                dtype=ImmersedLayers._element_type(P),
                                                phys_params=sys.phys_params,
                                                bc=sys.bc,
                                                forcing=sys.forcing,
                                                timestep_func=sys.timestep_func,
                                                motions=sys.motions)

           nothing
     end)

end

regenerate_problem(sys,body::Body) = regenerate_problem(sys,BodyList([body]))


_list(m::RigidBodyTools.AbstractMotion) = MotionList([m])
_list(m::MotionList) = m
_list(::Nothing) = nothing
_list(::Any) = nothing

_ddf_type(::AbstractILMProblem{DT}) where {DT} = DT
_ddf_type(::Type{P}) where P <: AbstractILMProblem{DT} where {DT} = DT
_scaling_type(::AbstractILMProblem{DT,ST}) where {DT,ST} = ST
_scaling_type(::Type{P}) where P <: AbstractILMProblem{DT,ST} where {DT,ST} = ST
_element_type(::AbstractILMProblem{DT,ST,DTP}) where {DT,ST,DTP} = DTP
_element_type(::Type{P}) where P <: AbstractILMProblem{DT,ST,DTP} where {DT,ST,DTP} = DTP


@ilmproblem BasicScalarILM scalar
@ilmproblem BasicVectorILM vector


# Extend this function for other problem types in order to create an extra cache
# of variables and operators for the problem
"""
    prob_cache(prob,base_cache::BasicILMCache)

This function is called by [`__init`](@ref) to generate a problem-specific extra
cache. Extend this function in order to generate an extra cache for a
user-defined problem type. The user must define the cache type itself.
""" prob_cache


prob_cache(prob::BasicScalarILMProblem,base_cache::BasicILMCache) = nothing
prob_cache(prob::BasicVectorILMProblem,base_cache::BasicILMCache) = nothing
