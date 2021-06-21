"""
$(TYPEDEF)

A system of operators and caches for immersed layer problems.
"""
struct ILMSystem{PT,BCT<:BasicILMCache,ECT<:Union{AbstractExtraILMCache,Nothing}}

  base_cache :: BCT
  extra_cache :: ECT

end

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
