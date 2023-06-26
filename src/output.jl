"""
    @snapshotoutput(name)

A function that returns some instantaneous output about a time-varying solution should have
the signature `name(state,constraint,aux_state,sys::ILMSystem,t[,kwargs...])`, where `state` is a solution vector
(of the same type as returned by `state(zeros_sol(sys))`), `constraint` is
of the same type as returned by `constraint(zeros_sol(sys))`),
`sys` is the system struct,
`t` is time, and `kwargs` are any additional keyword arguments. This macro
extends that function so that it will also operate on the integrator
e.g, `name(integrator)`, and the solution history array `sol`, e.g.,
`name(sol::ODESolution,sys,t)`, using interpolation for values of `t` that are not explicitly
in the solution array. It can also take a range of values of `t`, e.g,
`name(sol::ODESolution,sys,0.1:0.01:0.2)`.
"""
macro snapshotoutput(name)

  return esc(quote

      $name(u::ConstrainedSystems.ArrayPartition,sys::ILMSystem{S,P,N},t,args...;kwargs...) where {S,P,N} = $name(state(u),constraint(u),aux_state(u),sys,t,args...;kwargs...)

      $name(u::ConstrainedSystems.ArrayPartition,sys::ILMSystem{S,P,0},t,args...;kwargs...) where {S,P} = $name(state(u),zeros_surface(sys),aux_state(u),sys,t,args...;kwargs...)

      $name(integ::ConstrainedSystems.OrdinaryDiffEq.ODEIntegrator,args...;kwargs...) = $name(integ.u,integ.p,integ.t,args...;kwargs...)

      $name(sol::ConstrainedSystems.OrdinaryDiffEq.ODESolution,sys::ILMSystem,t::Real,args...;kwargs...) = $name(sol(t),sys,t,args...;kwargs...)

      $name(sol::ConstrainedSystems.OrdinaryDiffEq.ODESolution,sys::ILMSystem,t::AbstractArray,args...;kwargs...) =
          map(ti -> $name(sol(ti),sys,ti,args...;kwargs...),t)

      export $name

  end)

end

"""
    @scalarsurfacemetric(name)

A function that returns a time-varying scalar surface metric should have
the signature `name(state,constraint,sys::ILMSystem,t,bodyid,[,kwargs...])`, where `state` is a solution vector
(of the same type as returned by `state(zeros_sol(sys))`), `constraint` is
of the same type as returned by `constraint(zeros_sol(sys))`),
`sys` is the system struct,
`t` is time, `bodyid` is the body ID, and `kwargs` are any additional keyword arguments. This macro
extends that function so that it will also operate on the integrator
e.g, `name(integrator,bodyid)`, and the solution history array `sol`, e.g.,
`name(sol::ODESolution,sys,t,bodyid)`, using interpolation for values of `t` that are not explicitly
in the solution array. It can also take a range of values of `t`, e.g,
`name(sol::ODESolution,sys,0.1:0.01:0.2,bodyid)`. And finally, it
can simply take `name(sol::ODESolution,sys,bodyid)` to return
the entire history.
"""
macro scalarsurfacemetric(name)
    return esc(quote
        function $name(sol::ConstrainedSystems.OrdinaryDiffEq.ODESolution,sys::ILMSystem,bodyid::Int;kwargs...)
            mom = map((u,t) -> $name(u,sys,t,bodyid;kwargs...),sol.u,sol.t)
            mom
        end

        @snapshotoutput $name
    end)
end

"""
    @vectorsurfacemetric(name)

A function that returns a time-varying scalar surface metric should have
the signature `name(state,constraint,sys::ILMSystem,t,bodyid,[,kwargs...])`, where `state` is a solution vector
(of the same type as returned by `state(zeros_sol(sys))`), `constraint` is
of the same type as returned by `constraint(zeros_sol(sys))`),
`sys` is the system struct,
`t` is time, `bodyid` is the body ID, and `kwargs` are any additional keyword arguments. This macro
extends that function so that it will also operate on the integrator
e.g, `name(integrator,bodyid)`, and the solution history array `sol`, e.g.,
`name(sol::ODESolution,sys,t,bodyid)`, using interpolation for values of `t` that are not explicitly
in the solution array. It can also take a range of values of `t`, e.g,
`name(sol::ODESolution,sys,0.1:0.01:0.2,bodyid)`. And finally, it
can simply take `name(sol::ODESolution,sys,bodyid)` to return
the entire history.
"""
macro vectorsurfacemetric(name)
    return esc(quote
        function $name(sol::ConstrainedSystems.OrdinaryDiffEq.ODESolution,sys::ILMSystem,bodyid::Int;kwargs...)
            vx = map((u,t) -> $name(u,sys,t,bodyid;kwargs...)[1],sol.u,sol.t)
            vy = map((u,t) -> $name(u,sys,t,bodyid;kwargs...)[2],sol.u,sol.t)
            vx, vy
        end

        @snapshotoutput $name
    end)
end

#=
Provide the surfaces, constructing them where necessary
=#

"""
    surfaces(sol::ODESolution,sys::ILMSystem,t) -> BodyList

Return the list of surfaces (as a `BodyList`) at the time `t`, using
the ODE solution array `sol`. If the surfaces are stationary, then
this simply returns them and ignores the time argument.
"""
surfaces(sol::ConstrainedSystems.OrdinaryDiffEq.ODESolution,sys::ILMSystem,t) = surfaces(sol(t),sys,t)


"""
    surfaces(int::ConstrainedSystems.OrdinaryDiffEq.ODEIntegrator) -> BodyList

Return the list of surfaces (as a `BodyList`) in the integrator `int`.
"""
surfaces(integ::ConstrainedSystems.OrdinaryDiffEq.ODEIntegrator) = surfaces(integ.u,integ.p,integ.t)
