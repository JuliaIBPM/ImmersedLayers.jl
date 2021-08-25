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
$(TYPEDEF)

Generic problem type with scalar-type data. This type generates no extra cache.
"""
struct BasicScalarILMProblem{DT,ST} <: AbstractScalarILMProblem{DT,ST}
   g :: PhysicalGrid
   bodies :: BodyList
   BasicScalarILMProblem(g::PT,bodies::BodyList;ddftype=CartesianGrids.Yang3,scaling=IndexScaling) where {PT} = new{ddftype,scaling}(g,bodies)
   BasicScalarILMProblem(g::PT,body::Body;ddftype=CartesianGrids.Yang3,scaling=IndexScaling) where {PT} = new{ddftype,scaling}(g,BodyList([body]))
end

# Extend this function for other problem types in order to create an extra cache
# of variables and operators for the problem
"""
    prob_type(prob,base_cache::BasicILMCache)

This function is called by [`__init`](@ref) to generate a problem-specific extra
cache. Extend this function in order to generate an extra cache for a
user-defined problem type.
"""
function prob_cache(prob::BasicScalarILMProblem,base_cache::BasicILMCache)
    return nothing
end

"""
$(TYPEDEF)

Generic problem type with vector-type data. This type generates no extra cache.
"""
struct BasicVectorILMProblem{DT,ST} <: AbstractVectorILMProblem{DT,ST}
   g :: PhysicalGrid
   bodies :: BodyList
   BasicVectorILMProblem(g::PT,bodies::BodyList;ddftype=CartesianGrids.Yang3,scaling=IndexScaling) where {PT} = new{ddftype,scaling}(g,bodies)
   BasicVectorILMProblem(g::PT,body::Body;ddftype=CartesianGrids.Yang3,scaling=IndexScaling) where {PT} = new{ddftype,scaling}(g,BodyList([body]))
end

function prob_cache(prob::BasicVectorILMProblem,base_cache::BasicILMCache)
    return nothing
end
