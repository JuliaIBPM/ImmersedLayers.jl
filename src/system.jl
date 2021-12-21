#=
An instance of `ILMSystem` holds the base cache, any problem-specific extra cache,
and containers for physical parameters, boundary condition data/functions, forcing data/functions,
a timestep function (which returns the timestep) and motion data/functions.

If the system needs to be regenerated because of surface motion, only the two
caches will need to be regenerated.
=#

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
    x = aux_state(u)
    bodies = deepcopy(bl)
    update_body!(bodies,x,motions)
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
function __init(prob::AbstractILMProblem{DT,ST}) where {DT,ST}
    @unpack g, bodies, phys_params, bc, forcing, timestep_func, motions = prob

    if typeof(prob) <: AbstractScalarILMProblem
        base_cache = SurfaceScalarCache(bodies,g,ddftype=DT,scaling=ST)
    elseif typeof(prob) <: AbstractVectorILMProblem
        base_cache = SurfaceVectorCache(bodies,g,ddftype=DT,scaling=ST)
    end

    extra_cache = prob_cache(prob,base_cache)

    return ILMSystem{_static_surfaces(motions),typeof(prob),typeof(phys_params),typeof(bc),typeof(forcing),
                    typeof(timestep_func),typeof(motions),typeof(base_cache),typeof(extra_cache)}(
              phys_params,bc,forcing,timestep_func,motions,base_cache,extra_cache)

end

_static_surfaces(::Nothing) = true
_static_surfaces(::Any) = false

#=
Need a function of the form update_system(sys,u,sysold,t)
- Need to create a function that will take the motion_state
  and update bodies with it
- Need to be able to regenerate extra_cache with new bodies
- The latter needs a way to regenerate the problem struct
  from the system.
=#



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
