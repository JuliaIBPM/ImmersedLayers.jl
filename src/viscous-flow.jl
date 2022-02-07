#= Setting up the cache =#

@ilmproblem ViscousIncompressibleFlow vector

struct ViscousIncompressibleFlowCache{CDT,FRT,DVT,VFT,VORT,DILT,VELT,FCT} <: AbstractExtraILMCache
   cdcache :: CDT
   fcache :: FRT
   dvb :: DVT
   vb_tmp :: DVT
   v_tmp :: VFT
   dv :: VFT
   dv_tmp :: VFT
   w_tmp :: VORT
   divv_tmp :: DILT
   velcache :: VELT
   f :: FCT
end

function ImmersedLayers.prob_cache(prob::ViscousIncompressibleFlowProblem,
                                   base_cache::BasicILMCache{N,scaling}) where {N,scaling}
    @unpack phys_params, forcing = prob
    @unpack gdata_cache, gcurl_cache, g = base_cache

    dvb = zeros_surface(base_cache)
    vb_tmp = zeros_surface(base_cache)

    v_tmp = zeros_grid(base_cache)
    dv = zeros_grid(base_cache)
    dv_tmp = zeros_grid(base_cache)
    w_tmp = zeros_gridcurl(base_cache)
    divv_tmp = zeros_griddiv(base_cache)

    velcache = VectorFieldCache(base_cache)

    # Construct a Lapacian outfitted with the viscosity
    Re = phys_params["Re"]
    viscous_L = Laplacian(base_cache,gcurl_cache,1.0/Re)

    # Create cache for the convective derivative
    cdcache = ConvectiveDerivativeCache(base_cache)

    # Create cache for the forcing regions
    fcache = ForcingModelAndRegion(forcing["forcing models"],base_cache);


    # The state here is vorticity, the constraint is the surface traction
    f = _get_ode_function_list(viscous_L,base_cache)

    ViscousIncompressibleFlowCache(cdcache,fcache,dvb,vb_tmp,v_tmp,dv,dv_tmp,w_tmp,divv_tmp,velcache,f)
end

_get_ode_function_list(viscous_L,base_cache::BasicILMCache{N}) where {N} =
                ODEFunctionList(state = zeros_gridcurl(base_cache),
                                constraint = zeros_surface(base_cache),
                                ode_rhs=viscousflow_vorticity_ode_rhs!,
                                lin_op=viscous_L,
                                bc_rhs=viscousflow_vorticity_bc_rhs!,
                                constraint_force = viscousflow_vorticity_constraint_force!,
                                bc_op = viscousflow_vorticity_bc_op!)

_get_ode_function_list(viscous_L,base_cache::BasicILMCache{0}) =
                 ODEFunctionList(state = zeros_gridcurl(base_cache),
                                 ode_rhs=viscousflow_vorticity_ode_rhs!,
                                 lin_op=viscous_L)

#= ODE functions =#

function viscousflow_vorticity_ode_rhs!(dw,w,sys::ILMSystem,t)
  @unpack extra_cache, base_cache = sys
  @unpack v_tmp, dv = extra_cache

  velocity!(v_tmp,w,sys,t)
  viscousflow_velocity_ode_rhs!(dv,v_tmp,sys,t)
  curl!(dw,dv,base_cache)

  return dw
end

function viscousflow_velocity_ode_rhs!(dv,v,sys::ILMSystem,t)
    @unpack bc, forcing, phys_params, extra_cache, base_cache = sys
    @unpack dvb, dv_tmp, cdcache, fcache = extra_cache

    Re = phys_params["Re"]
    over_Re = 1.0/Re

    fill!(dv,0.0)

    # Calculate the convective derivative
    convective_derivative!(dv_tmp,v,base_cache,cdcache)
    dv .-= dv_tmp

    # Calculate the double-layer term
    prescribed_surface_jump!(dvb,t,sys)
    surface_divergence_symm!(dv_tmp,over_Re*dvb,base_cache)
    dv .-= dv_tmp

    # Apply forcing
    apply_forcing!(dv_tmp,v,t,fcache,phys_params)
    dv .+= dv_tmp

    return dv
end


function viscousflow_vorticity_bc_rhs!(vb,sys::ILMSystem,t)
    @unpack bc, forcing,extra_cache, base_cache, phys_params = sys
    @unpack dvb, vb_tmp, v_tmp, velcache, divv_tmp = extra_cache
    @unpack dcache, ϕtemp = velcache

    viscousflow_velocity_bc_rhs!(vb,sys,t)

    # Subtract influence of scalar potential field
    fill!(divv_tmp,0.0)
    prescribed_surface_jump!(dvb,t,sys)
    scalarpotential_from_masked_divv!(ϕtemp,divv_tmp,dvb,base_cache,dcache)
    vecfield_from_scalarpotential!(v_tmp,ϕtemp,base_cache)
    interpolate!(vb_tmp,v_tmp,base_cache)
    vb .-= vb_tmp

    # Subtract influence of free stream
    Uinf, Vinf = forcing["freestream"](t,phys_params)
    vb.u .-= Uinf
    vb.v .-= Vinf

    return vb
end

function viscousflow_velocity_bc_rhs!(vb,sys::ILMSystem,t)
    prescribed_surface_average!(vb,t,sys)
    return vb
end

function viscousflow_vorticity_constraint_force!(dw,τ,sys)
    @unpack extra_cache, base_cache = sys
    @unpack dv = extra_cache

    viscousflow_velocity_constraint_force!(dv,τ,sys)
    curl!(dw,dv,base_cache)

    return dw
end

function viscousflow_velocity_constraint_force!(dv,τ,sys::ILMSystem{S,P,N}) where {S,P,N}
    @unpack base_cache = sys
    regularize!(dv,τ,base_cache)
    return dv
end

function viscousflow_velocity_constraint_force!(dv,τ,sys::ILMSystem{S,P,0}) where {S,P}
    return dv
end

function viscousflow_vorticity_bc_op!(vb,w,sys::ILMSystem)
    @unpack extra_cache, base_cache = sys
    @unpack velcache, v_tmp = extra_cache
    @unpack ψtemp = velcache

    vectorpotential_from_curlv!(ψtemp,w,base_cache)
    vecfield_from_vectorpotential!(v_tmp,ψtemp,base_cache)
    viscousflow_velocity_bc_op!(vb,v_tmp,sys)

    return vb
end

function viscousflow_velocity_bc_op!(vb,v,sys::ILMSystem)
    @unpack base_cache = sys
    interpolate!(vb,v,base_cache)
    return vb
end


#= Fields =#

vorticity!(masked_w::Nodes{Dual},w::Nodes{Dual},dv::VectorData,base_cache::BasicILMCache,wcache::VectorPotentialCache) =
            masked_curlv_from_curlv_masked!(masked_w,w,dv,base_cache,wcache)

total_vorticity!(w::Nodes{Dual},masked_w::Nodes{Dual},dv::VectorData,base_cache::BasicILMCache,wcache::VectorPotentialCache) =
            curlv_masked_from_masked_curlv!(w,masked_w,dv,base_cache,wcache)

for f in [:vorticity,:total_vorticity]

    f! = Symbol(string(f)*"!")

    @eval function $f!(masked_w::Nodes{Dual},w::Nodes{Dual},τ,sys::ILMSystem,t)
        @unpack extra_cache, base_cache = sys
        @unpack dvb, velcache = extra_cache
        @unpack wcache = velcache

        prescribed_surface_jump!(dvb,t,sys)
        $f!(masked_w,w,dvb,base_cache,wcache)
    end

    @eval $f(w::Nodes{Dual},τ,sys::ILMSystem,t) = $f!(zeros_gridcurl(sys),w,τ,sys,t)

    @eval @snapshotoutput $f
end

function velocity!(v::Edges{Primal},curl_vmasked::Nodes{Dual},masked_divv::Nodes{Primal},dv::VectorData,vp,base_cache::BasicILMCache,velcache::VectorFieldCache,w_tmp::Nodes{Dual})
    @unpack wcache  = velcache

    masked_curlv = w_tmp

    masked_curlv_from_curlv_masked!(masked_curlv,curl_vmasked,dv,base_cache,wcache)
    vecfield_helmholtz!(v,masked_curlv,masked_divv,dv,vp,base_cache,velcache)

    return v
end

function velocity!(v::Edges{Primal},w::Nodes{Dual},sys::ILMSystem,t)
    @unpack forcing, phys_params, extra_cache, base_cache = sys
    @unpack dvb, velcache, divv_tmp, w_tmp = extra_cache

    prescribed_surface_jump!(dvb,t,sys)

    Vinf = forcing["freestream"](t,phys_params)

    fill!(divv_tmp,0.0)
    velocity!(v,w,divv_tmp,dvb,Vinf,base_cache,velcache,w_tmp)
end

velocity(w::Nodes{Dual},τ,sys::ILMSystem,t) = velocity!(zeros_grid(sys),w,sys,t)

@snapshotoutput velocity


function streamfunction!(ψ::Nodes{Dual},w::Nodes{Dual},vp::Tuple,base_cache::BasicILMCache,wcache::VectorPotentialCache)
    @unpack stemp = wcache

    fill!(ψ,0.0)
    vectorpotential_from_curlv!(stemp,w,base_cache)
    ψ .+= stemp

    vectorpotential_uniformvecfield!(stemp,vp...,base_cache)
    ψ .+= stemp

    return ψ
end

function streamfunction!(ψ::Nodes{Dual},w::Nodes{Dual},sys::ILMSystem,t)
    @unpack phys_params, forcing, extra_cache, base_cache = sys
    @unpack velcache = extra_cache
    @unpack wcache = velcache

    Vinf = forcing["freestream"](t,phys_params)

    streamfunction!(ψ,w,Vinf,base_cache,wcache)

end

streamfunction(w::Nodes{Dual},τ,sys::ILMSystem,t) = streamfunction!(zeros_gridcurl(sys),w,sys,t)

@snapshotoutput streamfunction

function pressure!(press::Nodes{Primal},w::Nodes{Dual},τ,sys::ILMSystem,t)
      @unpack extra_cache, base_cache = sys
      @unpack velcache, v_tmp, dv_tmp, dv, divv_tmp = extra_cache

      velocity!(v_tmp,w,sys,t)
      viscousflow_velocity_ode_rhs!(dv,v_tmp,sys,t)
      viscousflow_velocity_constraint_force!(dv_tmp,τ,sys)
      dv .-= dv_tmp

      divergence!(divv_tmp,dv,base_cache)
      inverse_laplacian!(press,divv_tmp,base_cache)
      return press
end

pressure(w::Nodes{Dual},τ,sys::ILMSystem,t) = pressure!(zeros_griddiv(sys),w,τ,sys,t)

@snapshotoutput pressure


#= Surface fields =#

function traction!(tract::VectorData{N},τ::VectorData{N},sys::ILMSystem,t) where {N}
    @unpack bc, extra_cache, base_cache, phys_params = sys
    @unpack vb_tmp, dvb = extra_cache
    @unpack sscalar_cache = base_cache

    prescribed_surface_average!(vb_tmp,t,sys)
    surface_velocity!(dvb,sys,t)
    vb_tmp .-= dvb

    nrm = normals(sys)
    pointwise_dot!(sscalar_cache,nrm,vb_tmp)
    prescribed_surface_jump!(dvb,t,sys)
    product!(tract,dvb,sscalar_cache)

    tract .+= τ
    return tract

end
traction!(out,τ::VectorData{0},sys::ILMSystem,t) = out
traction(w::Nodes{Dual},τ::VectorData,sys::ILMSystem,t) = traction!(zeros_surface(sys),τ,sys,t)
@snapshotoutput traction

function pressurejump!(dpb::ScalarData{N},τ::VectorData{N},sys::ILMSystem,t) where {N}
    @unpack base_cache = sys
    @unpack sdata_cache = base_cache

    nrm = normals(sys)
    traction!(sdata_cache,τ,sys,t)
    pointwise_dot!(dpb,nrm,sdata_cache)
    dpb .*= -1.0
    return dpb
end
pressurejump!(out,τ::VectorData{0},sys::ILMSystem,t) = out
pressurejump(w::Nodes{Dual},τ::VectorData,sys::ILMSystem,t) = pressurejump!(zeros_surfacescalar(sys),τ,sys,t)
@snapshotoutput pressurejump


#= Integrated metrics =#

force(w::Nodes{Dual},τ,sys::ILMSystem{S,P,0},t,bodyi::Int) where {S,P} = Vector{Float64}(), Vector{Float64}()

function force(w::Nodes{Dual},τ::VectorData{N},sys::ILMSystem{S,P,N},t,bodyi::Int) where {S,P,N}
    @unpack base_cache = sys
    @unpack sdata_cache = base_cache
    traction!(sdata_cache,τ,sys,t)
    fx = integrate(sdata_cache.u,sys,bodyi)
    fy = integrate(sdata_cache.v,sys,bodyi)

    return fx, fy
end

function force(sol::ConstrainedSystems.OrdinaryDiffEq.ODESolution,sys::ILMSystem,bodyi::Int)
    fx = map((u,t) -> force(u,sys,t,bodyi)[1],sol.u,sol.t)
    fy = map((u,t) -> force(u,sys,t,bodyi)[2],sol.u,sol.t)
    fx, fy
end

@snapshotoutput force


moment(w::Nodes{Dual},τ,sys::ILMSystem{S,P,0},t,bodyi::Int;kwargs...) where {S,P} = Vector{Float64}()

function moment(w::Nodes{Dual},τ::VectorData{N},sys::ILMSystem{S,P,N},t,bodyi::Int;center=(0.0,0.0)) where {S,P,N}
    @unpack base_cache = sys
    @unpack sdata_cache, sscalar_cache = base_cache
    xc, yc = center
    pts = points(sys)
    pts.u .-= xc
    pts.v .-= yc

    traction!(sdata_cache,τ,sys,t)
    pointwise_cross!(sscalar_cache,pts,sdata_cache)
    mom = integrate(sscalar_cache,sys,bodyi)

    return mom
end

function moment(sol::ConstrainedSystems.OrdinaryDiffEq.ODESolution,sys::ILMSystem,bodyi::Int;kwargs...)
    mom = map((u,t) -> moment(u,sys,t,bodyi;kwargs...),sol.u,sol.t)
    mom
end

@snapshotoutput moment
