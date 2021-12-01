abstract type AbstractILMProblem{DT,ST} end

"""
$(TYPEDEF)

When defining a problem type with scalar data, make it a subtype of this.
"""
abstract type AbstractScalarILMProblem{DT,ST} <: AbstractILMProblem{DT,ST} end

"""
$(TYPEDEF)

When defining a problem type with vector data, make it a subtype of this.
"""
abstract type AbstractVectorILMProblem{DT,ST} <: AbstractILMProblem{DT,ST} end

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

    StokesFlowProblem(grid,bodies[,ddftype=CartesianGrids.Yang3][,scaling=IndexScaling])

Note that there is another constructor for problems with no surfaces that only
requires that the grid information be passed, e.g.,

    StokesFlowProblem(grid)

There are several keyword arguments for the problem constructor

- `ddftype = ` : To set the DDF type. The default is `Yang3`.
- `scaling = ` : To set the scaling type, `IndexScaling` (default) or `GridScaling`.
- `phys_params = ` : To pass in physical parameters
- `bc = ` : To pass in boundary condition data or functions
- `forcing = ` : To pass in forcing functions or data
- `motions = ` : To provide functions or data that can update the immersed surfaces
- `timestep_func =` : To pass in a function for time-dependent problems that provides the time-step size.
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
          struct $typename{DT,ST,BLT,PHT,BCF,FF,DTF,MTF} <: $abtype{DT,ST}
             g :: PhysicalGrid
             bodies :: BLT
             phys_params :: PHT
             bc :: BCF
             forcing :: FF
             timestep_func :: DTF
             motions :: MTF
             $typename(g::PT,bodies::BodyList;ddftype=CartesianGrids.Yang3,
                                              scaling=IndexScaling,
                                              phys_params=nothing,
                                              bc=nothing,
                                              forcing=nothing,
                                              timestep_func=nothing,
                                              motions=nothing) where {PT <: PhysicalGrid} =
                    new{ddftype,scaling,typeof(bodies),typeof(phys_params),typeof(bc),typeof(forcing),typeof(timestep_func),typeof(motions)}(
                                              g,bodies,phys_params,bc,forcing,timestep_func,motions)
          end

          $typename(g::PhysicalGrid,body::Body;kwargs...) = $typename(g,BodyList([body]); kwargs...)
          $typename(g::PhysicalGrid;kwargs...) = $typename(g,BodyList(); kwargs...)


     end)

end


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
