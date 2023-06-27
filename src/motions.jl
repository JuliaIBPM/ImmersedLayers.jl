
mutable struct ILMMotion{MT<:Union{RigidBodyMotion,Nothing}}
  reference_body :: Int
  m :: MT
  vl :: PluckerMotionList
  Xl :: MotionTransformList
end

function _construct_motion(m::RigidBodyMotion,reference_body::Int,bodies::BodyList)
  _check_reference_body(reference_body,bodies)
  t = 0.0
  x = init_motion_state(bodies,m;tinit=t)
  vl, Xl = _evaluate_motion(x,m,t)

  ILMMotion(reference_body,m,vl,Xl)
end

_check_reference_body(refid::Int,bodies::BodyList) = (0 <= refid <= length(bodies)) ? refid :
                            error("Reference body must be between 0 and number of bodies")


_construct_motion(::Nothing,::Int,::BodyList) = ILMMotion(0,nothing,PluckerMotionList(),MotionTransformList())
_construct_motion(::Any,::Int,::BodyList) = ILMMotion(0,nothing,PluckerMotionList(),MotionTransformList())


"""
    evaluate_motion!(motion::ILMMotion,bl::BodyList,x::AbstractVector,t::Real)

Given the current joint state vector `x` and time `t`, update the
body transforms and velocities in `motion`.
"""
function evaluate_motion!(motion::ILMMotion,bl::BodyList,x::AbstractVector,t::Real)
    @unpack m = motion

    vl, Xl = _evaluate_motion(x,m,t)
    motion.vl = vl
    motion.Xl = Xl
    nothing
end

"""
    evaluate_motion!(sys::ILMSystem,u::ArrayPartition,t::Real)

Given the current system solution vector `u` and time `t`, update the
body transforms and velocities in `sys.motions`.
"""
function evaluate_motion!(sys::ILMSystem,u::ConstrainedSystems.ArrayPartition,t::Real)
    @unpack motions, base_cache = sys
    @unpack bl = base_cache
    x = aux_state(u)
    evaluate_motion!(motions,bl,x,t)
end


function _evaluate_motion(x::AbstractVector,m::RigidBodyMotion,t::Real)
  q = position_vector(x,m)
  Xl = body_transforms(q,m)
  vl = body_velocities(x,t,m)
  return vl, Xl
end
