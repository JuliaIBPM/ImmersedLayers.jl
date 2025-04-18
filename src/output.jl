"""
    @snapshotoutput(name)

A function that returns some instantaneous output about a time-varying solution should have
the signature `name(state,constraint,aux_state,sys::ILMSystem,t,[args...];[kwargs...])`, where `state` is a solution vector
(of the same type as returned by `state(zeros_sol(sys))`), `constraint` is
of the same type as returned by `constraint(zeros_sol(sys))`), and `aux_state` is the same
as `aux_state(zeros_sol(sys))`, `sys` is the system struct,
`t` is time, and `args` are additional arguments and `kwargs` are any additional keyword arguments. This macro
extends that function so that it will also operate on the integrator
e.g, `name(integrator)`, and the solution history array `sol`, e.g.,
`name(sol::ODESolution,sys,t)`, using interpolation for values of `t` that are not explicitly
in the solution array. It can also take a range of values of `t`, e.g,
`name(sol::ODESolution,sys,0.1:0.01:0.2)`.

This will also create a function `name_xy` that operates on the same sets of arguments, but returns
a spatially interpolatable version of the quantity `name`, which can be accessed with `x`, `y` coordinates.
In the case of the array with a time range, it returns a vector of such interpolatable objects.
"""
macro snapshotoutput(name)

  name_xy = Symbol(name,"_xy")  

  return esc(quote

      $name(u::ConstrainedSystems.ArrayPartition,sys::ILMSystem{S,P,N},t,args...;kwargs...) where {S,P,N} = $name(state(u),constraint(u),aux_state(u),sys,t,args...;kwargs...)

      $name(u::ConstrainedSystems.ArrayPartition,sys::ILMSystem{S,P,0},t,args...;kwargs...) where {S,P} = $name(state(u),zeros_surface(sys),aux_state(u),sys,t,args...;kwargs...)

      $name(integ::ODEIntegrator,args...;kwargs...) = $name(integ.u,integ.p,integ.t,args...;kwargs...)

      $name(sol::ODESolution,sys::ILMSystem,t::Real,args...;kwargs...) = $name(sol(t),sys,t,args...;kwargs...)

      $name(sol::ODESolution,sys::ILMSystem,t::AbstractArray,args...;kwargs...) =
          map(ti -> $name(sol(ti),sys,ti,args...;kwargs...),t)

      
      $name_xy(u::ConstrainedSystems.ArrayPartition,sys::ILMSystem,t,args...;kwargs...) = interpolatable_field($name(u,sys,t,args...;kwargs...),sys)

      $name_xy(integ::ODEIntegrator,args...;kwargs...) = interpolatable_field($name(integ,args...;kwargs...),integ.p)

      $name_xy(sol::ODESolution,sys::ILMSystem,t::Real,args...;kwargs...) = interpolatable_field($name(sol(t),sys,t,args...;kwargs...),sys)

      $name_xy(sol::ODESolution,sys::ILMSystem,t::AbstractArray,args...;kwargs...) =
          map(ti -> interpolatable_field($name(sol(ti),sys,ti,args...;kwargs...),sys),t)

      export $name, $name_xy

  end)

end

interpolatable_field(a,base_cache::BasicILMCache) = interpolatable_field(a,base_cache.g)
interpolatable_field(a,sys::ILMSystem) = interpolatable_field(a,sys.base_cache)



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
        function $name(sol::ODESolution,sys::ILMSystem,bodyid::Int;kwargs...)
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
        function $name(sol::ODESolution,sys::ILMSystem,bodyid::Int;kwargs...)
            v = map((u,t) -> $name(u,sys,t,bodyid;kwargs...),sol.u,sol.t)
            return tuple([map(vi -> vi[i],v) for i in eachindex(first(v))]...)
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
surfaces(sol::ODESolution,sys::ILMSystem,t) = surfaces(sol(t),sys,t)


"""
    surfaces(int::ODEIntegrator) -> BodyList

Return the list of surfaces (as a `BodyList`) in the integrator `int`.
"""
surfaces(integ::ODEIntegrator) = surfaces(integ.u,integ.p,integ.t)
