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
    @unpack g, bodies, phys_params, bc, f_funcs = prob

    if typeof(prob) <: AbstractScalarILMProblem
        base_cache = SurfaceScalarCache(bodies,g,ddftype=DT,scaling=ST,phys_params=phys_params,bc=bc,f_funcs=f_funcs)
    elseif typeof(prob) <: AbstractVectorILMProblem
        base_cache = SurfaceVectorCache(bodies,g,ddftype=DT,scaling=ST,phys_params=phys_params,bc=bc,f_funcs=f_funcs)
    end

    extra_cache = prob_cache(prob,base_cache)

    return ILMSystem{typeof(prob),typeof(base_cache),typeof(extra_cache)}(base_cache,extra_cache)

end

# Create the basic solve function, to be extended
function solve(prob::AbstractILMProblem,sys::ILMSystem) end



## Extend functions on `BasicILMCache` type to `ILMSystem`
for f in [:zeros_surface,:zeros_grid,:zeros_gridcurl,:zeros_gridgrad,
          :similar_surface,:similar_grid,:similar_gridcurl,:similar_gridgrad,
          :ones_surface,:ones_grid,:ones_gridgrad,:ones_gridcurl,
          :x_grid,:y_grid,:x_gridcurl,:y_gridcurl,
          :normals,:areas,:points,
          :create_nRTRn,:create_GLinvD,:create_CLinvCT,:create_CL2invCT,
          :create_RTLinvR,
          :create_GLinvD_cross,:create_surface_filter]
   @eval $f(sys::ILMSystem) = $f(sys.base_cache)
end

for f in [:regularize!, :interpolate!, :regularize_normal!,
          :normal_interpolate!,:regularize_normal_cross!,
          :normal_cross_interpolate!,
          :surface_curl!,:surface_divergence!,:surface_grad!,
          :surface_curl_cross!,:surface_divergence_cross!,:surface_grad_cross!]
  @eval $f(a,b,sys::ILMSystem) = $f(a,b,sys.base_cache)
end

for f in [:norm,:integrate]
   @eval $f(a,sys::ILMSystem) = $f(a,sys.base_cache)
   @eval $f(a,sys::ILMSystem,i::Int) = $f(a,sys.base_cache,i)
end

for f in [:view]
  @eval $f(a::PointData,sys::ILMSystem,i::Int) = $f(a,sys.base_cache,i)
end

for f in [:dot,:copyto!]
   @eval $f(a,b,sys::ILMSystem) = $f(a,b,sys.base_cache)
   @eval $f(a,b,sys::ILMSystem,i::Int) = $f(a,b,sys.base_cache,i)
end

for f in [:RegularizationMatrix,:InterpolationMatrix]
  @eval CartesianGrids.$f(sys::ILMSystem,args...) = $f(sys.base_cache,args...)
end
