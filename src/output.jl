"""
    @snapshotoutput(name)

A function that returns some instantaneous output about a time-varying solution should have
the signature `name(u,sys::ILMSystem,t[,kwargs...])`, where `u` is a solution vector
(of the same type as returned by [zeros_sol](@ref)), `sys` is the system struct,
`t` is time, and `kwargs` are any additional keyword arguments. This macro
extends that function so that it will also operate on the solution history array `sol`, e.g.,
`name(sol::ODESolution,sys,t)`, using interpolation for values of `t` that are not explicitly
in the solution array. It can also take a range of values of `t`, e.g,
`name(sol::ODESolution,sys,0.1:0.01:0.2)`.
"""
macro snapshotoutput(name)

  return esc(quote

      $name(u::ConstrainedSystems.ArrayPartition,sys::ILMSystem{S,P,N},t,args...;kwargs...) where {S,P,N} = $name(state(u),constraint(u),sys,t,args...;kwargs...)

      $name(integ::ConstrainedSystems.OrdinaryDiffEq.ODEIntegrator,args...;kwargs...) = $name(integ.u,integ.p,integ.t,args...;kwargs...)

      $name(sol::ConstrainedSystems.OrdinaryDiffEq.ODESolution,sys::ILMSystem,t,args...;kwargs...) = $name(sol(t),sys,t,args...;kwargs...)

      $name(sol::ConstrainedSystems.OrdinaryDiffEq.ODESolution,sys::ILMSystem,t::AbstractArray,args...;kwargs...) =
          map(ti -> $name(sol(ti),sys,ti,args...;kwargs...),t)


  end)

end
