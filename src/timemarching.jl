import ConstrainedSystems: init

_norm_sq(u) = dot(u,u)
_norm_sq(u::ConstrainedSystems.ArrayPartition) = sum(_norm_sq,u.x)
state_norm(u,t) = sqrt(_norm_sq(u))

function init(u0,tspan,sys::ILMSystem;alg=ConstrainedSystems.LiskaIFHERK(),kwargs...)
    prob = ODEProblem(sys.extra_cache.f,u0,tspan,sys)
    return init(prob, alg,dt=sys.base_cache.timestep_func(sys.base_cache),internal_norm=state_norm,kwargs...)
end
