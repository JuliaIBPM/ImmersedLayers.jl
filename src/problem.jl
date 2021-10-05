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
          struct $typename{DT,ST,PHT} <: $abtype{DT,ST}
             g :: PhysicalGrid
             bodies :: BodyList
             phys_params :: PHT
             $typename(g::PT,bodies::BodyList;ddftype=CartesianGrids.Yang3,
                                              scaling=IndexScaling,
                                              phys_params=nothing) where {PT} = new{ddftype,scaling,typeof(phys_params)}(g,bodies,phys_params)
             $typename(g::PT,body::Body;ddftype=CartesianGrids.Yang3,
                                        scaling=IndexScaling,
                                        phys_params=nothing) where {PT} = new{ddftype,scaling,typeof(phys_params)}(g,BodyList([body]),phys_params)
          end

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
