import ConstrainedSystems: init

_norm_sq(u) = dot(u,u)
_norm_sq(u::ConstrainedSystems.ArrayPartition) = sum(_norm_sq,u.x)
state_norm(u,t) = sqrt(_norm_sq(u))

function init(u0,tspan,sys::ILMSystem;alg=ConstrainedSystems.LiskaIFHERK(),kwargs...)
    @unpack timestep_func, phys_params, base_cache = sys
    @unpack g = base_cache
    prob = ODEProblem(sys.extra_cache.f,u0,tspan,sys)
    dt_calc = timestep_func(g,phys_params)
    return init(prob, alg,dt=dt_calc,internal_norm=state_norm,kwargs...)
end
