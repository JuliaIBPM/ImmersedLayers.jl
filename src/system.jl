"""
$(TYPEDEF)

A system of operators and caches for immersed layer problems. This is constructed
by [`__init`](@ref)
"""
struct ILMSystem{PT,BCT<:BasicILMCache,ECT<:Union{AbstractExtraILMCache,Nothing}}

  base_cache :: BCT
  extra_cache :: ECT

end

"""
    __init(prob::AbstractILMProblem)

Initialize `ILMSystem` with the given problem `prob` specification.
Depending on the type of problem, this sets up a base cache of scalar or
vector type, as well as an optional extra cache
"""
function __init(prob::AbstractILMProblem{DT,ST}) where {DT,ST}
    @unpack g, bodies = prob

    if typeof(prob) <: AbstractScalarILMProblem
        base_cache = SurfaceScalarCache(bodies,g,ddftype=DT,scaling=ST)
    elseif typeof(prob) <: AbstractVectorILMProblem
        base_cache = SurfaceVectorCache(bodies,g,ddftype=DT,scaling=ST)
    end

    extra_cache = prob_cache(prob,base_cache)

    return ILMSystem{typeof(prob),typeof(base_cache),typeof(extra_cache)}(base_cache,extra_cache)

end
