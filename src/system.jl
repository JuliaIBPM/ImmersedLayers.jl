#=
An instance of `ILMSystem` holds the base cache, any problem-specific extra cache,
and containers for physical parameters, boundary condition data/functions, forcing data/functions,
a timestep function (which returns the timestep) and motion data/functions.

If the system needs to be regenerated because of surface motion, only the two
caches will need to be regenerated.
=#

Base.length(sys::ILMSystem) = length(sys.base_cache)

"""
    construct_system(prob::AbstractILMProblem) -> ILMSystem

Return a system of type of type `ILMSystem` from the given problem
instance `prob`.
"""
@inline construct_system(prob::AbstractILMProblem) = __init(prob)

"""
    update_system(sysold::ILMSystem,body::Body/BodyList)

From an existing system `sysold`, return a new system based on
the Body or BodyList `body`.
"""
function update_system(sysold::ILMSystem,body::Union{Body,BodyList})
    prob = regenerate_problem(sysold,body)
    return __init(prob)
end

"""
    update_system!(sys::ILMSystem,u,sysold,t)

From an existing system `sysold` at time `t`, return a new system `sys`
in place, based on the solution vector `u`, whose body state
information will be used to replace the body information and
subsequent operators in `sysold`.
"""
function update_system!(sys::ILMSystem,u,sysold::ILMSystem,t)
    @unpack base_cache, motions = sysold
    @unpack bl = base_cache
    bodies = surfaces(u,sysold,t)
    #x = aux_state(u)
    #bodies = deepcopy(bl)
    #update_body!(bodies,x,motions)
    sysnew = update_system(sysold,bodies)
    sys.base_cache = sysnew.base_cache
    sys.extra_cache = sysnew.extra_cache
    return sys
end


"""
    __init(prob::AbstractILMProblem)

Initialize `ILMSystem` with the given problem `prob` specification.
Depending on the type of problem, this sets up a base cache of scalar or
vector type, as well as an optional extra cache
"""
function __init(prob::AbstractILMProblem{DT,ST,DTP}) where {DT,ST,DTP}
    @unpack g, bodies, phys_params, bc, forcing, timestep_func, motions = prob

    base_cache = _construct_base_cache(bodies,g,DT,ST,DTP,prob,Val(length(bodies)))

    extra_cache = prob_cache(prob,base_cache)

    return ILMSystem{_static_surfaces(motions),typeof(prob),length(base_cache),typeof(phys_params),typeof(bc),typeof(forcing),
                    typeof(timestep_func),typeof(motions),typeof(base_cache),typeof(extra_cache)}(
              phys_params,bc,forcing,timestep_func,motions,base_cache,extra_cache)

end

@inline _construct_base_cache(bodies,g,DT,ST,DTP,::PT,::Val{0}) where {PT <: AbstractScalarILMProblem} =
              SurfaceScalarCache(g,ddftype=DT,scaling=ST,dtype=DTP)

@inline _construct_base_cache(bodies,g,DT,ST,DTP,::PT,::Val{N}) where {PT <: AbstractScalarILMProblem,N} =
              SurfaceScalarCache(bodies,g,ddftype=DT,scaling=ST,dtype=DTP)

@inline _construct_base_cache(bodies,g,DT,ST,DTP,::PT,::Val{0}) where {PT <: AbstractVectorILMProblem} =
              SurfaceVectorCache(g,ddftype=DT,scaling=ST,dtype=DTP)

@inline _construct_base_cache(bodies,g,DT,ST,DTP,::PT,::Val{N}) where {PT <: AbstractVectorILMProblem,N} =
              SurfaceVectorCache(bodies,g,ddftype=DT,scaling=ST,dtype=DTP)

_static_surfaces(::Nothing) = true
_static_surfaces(::Any) = false

# Extend surface_velocity! and allow for null motions
RigidBodyTools.surface_velocity!(vec::VectorData,bl::Union{Body,BodyList},motions::Union{RigidBodyTools.AbstractMotion,MotionList},t) =
    surface_velocity!(vec.u,vec.v,bl,motions,t)

RigidBodyTools.surface_velocity!(vec::VectorData,bl::Union{Body,BodyList},motions,t) = fill!(vec,0.0)

RigidBodyTools.surface_velocity!(vec::VectorData,base_cache::BasicILMCache,motions,t) =
    surface_velocity!(vec,base_cache.bl,motions,t)

RigidBodyTools.surface_velocity!(vec::VectorData,sys::ILMSystem,t) =
    surface_velocity!(vec,sys.base_cache,sys.motions,t)

#=
function RigidBodyTools.surface_velocity!(u::AbstractVector,v::AbstractVector,sys::ILMSystem,t)
  @unpack base_cache, motions = sys
  @unpack bl = base_cache
  fill!(u,0.0)
  fill!(v,0.0)
  if !isnothing(motions)
    surface_velocity!(u,v,bl,motions,t)
  end
end
=#

# Create the basic solve function, to be extended
function solve(prob::AbstractILMProblem,sys::ILMSystem) end


"""
    surfaces(u::ConstrainedSystems.ArrayPartition,sys::ILMSystem,t) -> BodyList

Return the list of surfaces (as a `BodyList`) in the solution vector `u`. If the
surfaces are stationary, then this simply returns them from `sys` and ignores the
time argument.
"""
function surfaces(u::ConstrainedSystems.ArrayPartition,sys::ILMSystem{false},t)
    @unpack base_cache, motions = sys
    @unpack bl = base_cache
    x = aux_state(u)
    current_bl = deepcopy(bl)
    update_body!(current_bl,x,motions)
    return current_bl
end

surfaces(u::ConstrainedSystems.ArrayPartition,sys::ILMSystem{true},t) = surfaces(sys)


function surfaces(u::ConstrainedSystems.ArrayPartition,sys::ILMSystem{true,SCA,0},t) where SCA
    return nothing
end


## Extend functions on `BasicILMCache` type to `ILMSystem`
@inline CartesianGrids.cellsize(sys::ILMSystem) = cellsize(sys.base_cache)

for f in [:get_grid,:zeros_surface,:zeros_surfacescalar,
          :zeros_grid,:zeros_gridcurl,:zeros_griddiv,:zeros_gridgrad,:zeros_gridgradcurl,
          :similar_surface,:similar_surfacescalar,
          :similar_grid,:similar_gridcurl,:similar_griddiv,:similar_gridgrad,:similar_gridgradcurl,
          :ones_surface,:ones_surfacescalar,
          :ones_grid,:ones_gridgrad,:ones_gridcurl,:ones_griddiv,:ones_gridgradcurl,
          :x_grid,:y_grid,:x_gridcurl,:y_gridcurl,:x_griddiv,:y_griddiv,
          :normals,:areas,:points,:arcs,
          :create_nRTRn,:create_GLinvD,:create_CLinvCT,:create_CL2invCT,
          :create_RTLinvR,:create_GLinvD_symm,
          :create_GLinvD_cross,:create_surface_filter]
   @eval $f(sys::ILMSystem) = $f(sys.base_cache)
end

for f in [:regularize!, :interpolate!, :regularize_normal!,
          :normal_interpolate!,
          :regularize_normal_cross!,:normal_cross_interpolate!,
          :regularize_normal_dot!,:normal_dot_interpolate!,
          :regularize_normal_symm!,:normal_interpolate_symm!,
          :surface_curl!,:surface_divergence!,:surface_grad!,
          :surface_divergence_symm!,:surface_grad_symm!,
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
