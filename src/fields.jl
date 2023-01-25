# Routines for creating instances of spatial fields on the grid

"""
    evaluate_field!(f::GridData,s::AbstractSpatialField,cache)

Evaluate a spatial field `s` in place as grid data `f`.
"""
function evaluate_field!(f::GridData,s::AbstractSpatialField,cache::BasicILMCache)
    @unpack g = cache
    gf = GeneratedField(f,s,g)
    f .= gf()
    return f
end

"""
    evaluate_field!(f::GridData,t,s::AbstractSpatialField,cache)

Evaluate a spatial-temporal field `s` at time `t` in place as grid data `f`.
"""
function evaluate_field!(f::GridData,t,s::AbstractSpatialField,cache::BasicILMCache)
    @unpack g = cache
    gf = GeneratedField(f,s,g)
    f .= gf(t)
    return f
end

evaluate_field!(f::GridData,s::AbstractSpatialField,sys::ILMSystem) = evaluate_field!(f,s,sys.base_cache)
evaluate_field!(f::GridData,t,s::AbstractSpatialField,sys::ILMSystem) = evaluate_field!(f,t,s,sys.base_cache)
