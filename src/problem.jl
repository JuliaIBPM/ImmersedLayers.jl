abstract type AbstractILMProblem{DT,ST} end

abstract type AbstractScalarILMProblem{DT,ST} <: AbstractILMProblem{DT,ST} end

abstract type AbstractVectorILMProblem{DT,ST} <: AbstractILMProblem{DT,ST} end


struct BasicScalarILMProblem{DT,ST} <: AbstractScalarILMProblem{DT,ST}
   g :: PhysicalGrid
   bodies :: BodyList
   BasicScalarILMProblem(g::PT,bodies::BodyList;ddftype=CartesianGrids.Yang3,scaling=IndexScaling) where {PT} = new{ddftype,scaling}(g,bodies)
   BasicScalarILMProblem(g::PT,body::Body;ddftype=CartesianGrids.Yang3,scaling=IndexScaling) where {PT} = new{ddftype,scaling}(g,BodyList([body]))
end

# Extend this function for other problem types in order to create an extra cache
# of variables and operators for the problem
function prob_cache(prob::BasicScalarILMProblem,base_cache::BasicILMCache)
    return nothing
end

struct BasicVectorILMProblem{DT,ST} <: AbstractVectorILMProblem{DT,ST}
   g :: PhysicalGrid
   bodies :: BodyList
   BasicVectorILMProblem(g::PT,bodies::BodyList;ddftype=CartesianGrids.Yang3,scaling=IndexScaling) where {PT} = new{ddftype,scaling}(g,bodies)
   BasicVectorILMProblem(g::PT,body::Body;ddftype=CartesianGrids.Yang3,scaling=IndexScaling) where {PT} = new{ddftype,scaling}(g,BodyList([body]))
end

function prob_cache(prob::BasicVectorILMProblem,base_cache::BasicILMCache)
    return nothing
end
