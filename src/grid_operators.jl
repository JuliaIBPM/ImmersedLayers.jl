import CartesianGrids: convective_derivative!, divergence!, grad!, curl!

for f in [:divergence!, :grad!, :curl!]
  @eval $f(a,b,sys::ILMSystem) = $f(a,b,sys.base_cache)
end


@inline _scale_derivative!(w,cache::BasicILMCache{N,IndexScaling}) where {N} = w
@inline _scale_derivative!(w,cache::BasicILMCache{N,GridScaling}) where {N} = w ./= cellsize(cache)

_scale_laplacian!(w,cache::BasicILMCache{N,IndexScaling}) where {N} = w
_scale_laplacian!(w,cache::BasicILMCache{N,GridScaling}) where {N} = w # ./= cellsize(cache)^2

_scale_inverse_laplacian!(w,cache::BasicILMCache{N,IndexScaling}) where {N} = w
_scale_inverse_laplacian!(w,cache::BasicILMCache{N,GridScaling}) where {N} = w  #.*= cellsize(cache)^2

"""
    divergence!(p::Nodes{Primal/Dual},v::Edges{Primal/Dual},cache::BasicILMCache)
    divergence!(p::Nodes{Primal/Dual},v::Edges{Primal/Dual},sys::ILMSystem)

Compute the discrete divergence of `v` and return it in `p`, scaling it
by the grid spacing if `cache` (or `sys`) is of `GridScaling` type, or leaving it
as a simple differencing if `cache` (or `sys`) is of `IndexScaling` type.
"""
function divergence!(p::Nodes{C,NX,NY},q::Edges{C,NX,NY},cache::BasicILMCache) where {C,NX,NY}
    fill!(p,0.0)
    if !iszero(q)
      _unscaled_divergence!(p,q,cache)
      _scale_derivative!(p,cache)
    end
end

_unscaled_divergence!(p,q,cache::BasicILMCache) = divergence!(p,q)



"""
    grad!(v::Edges{Primal/Dual},p::Nodes{Primal/Dual},cache::BasicILMCache)
    grad!(v::Edges{Primal/Dual},p::Nodes{Primal/Dual},sys::ILMSystem)

Compute the discrete gradient of `p` and return it in `v`, scaling it
by the grid spacing if `cache` (or `sys`) is of `GridScaling` type, or leaving it
as a simple differencing if `cache` (or `sys`) is of `IndexScaling` type.
"""
function grad!(q::Edges{C,NX,NY},p::Nodes{C,NX,NY},cache::BasicILMCache) where {C,NX,NY}
    fill!(q,0.0)
    if !iszero(p)
      _unscaled_grad!(q,p,cache)
      _scale_derivative!(q,cache)
    end
end

"""
    grad!(v::EdgeGradient{Primal/Dual,Dual/Primal},p::Edges{Primal/Dual},cache::BasicILMCache)
    grad!(v::EdgeGradient{Primal/Dual,Dual/Primal},p::Edges{Primal/Dual},sys::ILMSystem)

Compute the discrete gradient of edge data `p` and return it in `v`, scaling it
by the grid spacing if `cache` (or `sys`) is of `GridScaling` type, or leaving it
as a simple differencing if `cache` (or `sys`) is of `IndexScaling` type.
"""
function grad!(q::EdgeGradient{C,D,NX,NY},p::Edges{C,NX,NY},cache::BasicILMCache) where {C,D,NX,NY}
    fill!(q,0.0)
    if !iszero(p)
      _unscaled_grad!(q,p,cache)
      _scale_derivative!(q,cache)
    end
end

_unscaled_grad!(q,p,cache::BasicILMCache) = grad!(q,p)

"""
    curl!(v::Edges{Primal},s::Nodes{Dual},cache::BasicILMCache)
    curl!(v::Edges{Primal},s::Nodes{Dual},sys::ILMSystem)

Compute the discrete curl of `s` and return it in `v`, scaling it
by the grid spacing if `cache` (or `sys`) is of `GridScaling` type, or leaving it
as a simple differencing if `cache` (or `sys`) is of `IndexScaling` type.
"""
function curl!(q::Edges{Primal,NX,NY},s::Nodes{Dual,NX,NY},cache::BasicILMCache) where {NX,NY}
    fill!(q,0.0)
    if !iszero(s)
      _unscaled_curl!(q,s,cache)
      _scale_derivative!(q,cache)
    end
end

_unscaled_curl!(q::Edges,s::Nodes,cache::BasicILMCache) = curl!(q,s)

"""
    curl!(w::Nodes{Dual},v::Edges{Primal},cache::BasicILMCache)
    curl!(w::Nodes{Dual},v::Edges{Primal},sys::ILMSystem)

Compute the discrete curl of `v` and return it in `w`, scaling it
by the grid spacing if `cache` (or `sys`) is of `GridScaling` type, or leaving it
as a simple differencing if `cache` (or `sys`) is of `IndexScaling` type.
"""
function curl!(w::Nodes{Dual,NX,NY},q::Edges{Primal,NX,NY},cache::BasicILMCache) where {NX,NY}
    fill!(w,0.0)
    if !iszero(q)
      _unscaled_curl!(w,q,cache)
      _scale_derivative!(w,cache)
    end
end

_unscaled_curl!(w::Nodes,q::Edges,cache::BasicILMCache) = curl!(w,q)

"""
    laplacian!(w::GridData,s::GridData,sys::ILMSystem)

Compute the Laplacian of grid data `s`, and divide the result
by unity or by the grid cell size, depending on whether `sys` has `IndexScaling` or `GridScaling`,
respectively, and return the result as `w`.
"""
laplacian!(w,s,sys::ILMSystem) = laplacian!(w,s,sys.base_cache)

"""
    laplacian!(w::GridData,s::GridData,cache::BasicILMCache)

Compute the Laplacian of grid data `s`, and divide the result
by unity or by the grid cell size, depending on whether `cache` has `IndexScaling` or `GridScaling`,
respectively, and return the result as `w`.
"""
function laplacian!(w::GridData,s::GridData,cache::BasicILMCache)
    @unpack L = cache
    _unscaled_laplacian!(w,s,L)
    _scale_laplacian!(w,cache)
end

function _unscaled_laplacian!(w::GridData,s::GridData,L::CartesianGrids.Laplacian)
  fill!(w,0.0)
  if !iszero(s)
    mul!(w,L,s)
  end
end


"""
    inverse_laplacian!(w::GridData,sys::ILMSystem)

Compute the in-place inverse Laplacian of grid data `w`, and multiply the result
by unity or by the grid cell size, depending on whether `sys` has `IndexScaling` or `GridScaling`,
respectively.
"""
inverse_laplacian!(w,sys::ILMSystem) = inverse_laplacian!(w,sys.base_cache)

"""
    inverse_laplacian!(w::GridData,cache::BasicILMCache)

Compute the in-place inverse Laplacian of grid data `w`, and multiply the result
by unity or by the grid cell size, depending on whether `cache` has `IndexScaling` or `GridScaling`,
respectively.
"""
function inverse_laplacian!(w::GridData,cache::BasicILMCache)
    @unpack L = cache
    if !iszero(w)
      _unscaled_inverse_laplacian!(w,L)
      _scale_inverse_laplacian!(w,cache)
    end
end

"""
    inverse_laplacian!(sol::GridData,rhs::GridData,cache::BasicILMCache)

Compute the iinverse Laplacian of grid data `rhs`, multiply the result
by unity or by the grid cell size, depending on whether `cache` has `IndexScaling` or `GridScaling`,
respectively, and return the result as `sol`.
"""
function inverse_laplacian!(sol::GridData{NX,NY},rhs::GridData{NX,NY},cache::BasicILMCache) where {NX,NY}
    @unpack L = cache
    fill!(sol,0.0)
    if !iszero(rhs)
      _unscaled_inverse_laplacian!(sol,L,rhs)
      _scale_inverse_laplacian!(sol,cache)
    end
end

_unscaled_inverse_laplacian!(w::GridData,L::CartesianGrids.Laplacian) = w .= L\w

_unscaled_inverse_laplacian!(sol::GridData,L::CartesianGrids.Laplacian,rhs::GridData) = ldiv!(sol,L,rhs)



###  Convective derivative of velocity-like data by itself ###

"""
    ConvectiveDerivativeCache(dv::EdgeGradient)

Create a cache (a subtype of [`AbstractExtraILMCache`](@ref)) for computing
the convective derivative, using `dv` to define the cache data.
"""
struct ConvectiveDerivativeCache{VTT} <: AbstractExtraILMCache
   vt1_cache :: VTT
   vt2_cache :: VTT
   vt3_cache :: VTT
   ConvectiveDerivativeCache(vt::GridData) = new{typeof(vt)}(similar(vt),similar(vt),similar(vt))
end

"""
    ConvectiveDerivativeCache(cache::BasicILMCache)

Create a cache for computing the convective derivative, based on
the basic ILM cache `cache`.
"""
ConvectiveDerivativeCache(cache::BasicILMCache) = ConvectiveDerivativeCache(similar_gridgrad(cache))

"""
    convective_derivative(v::Edges{Primal},p::Nodes{Primal},base_cache::BasicILMCache)

Compute the convective derivative of `p` with velocity `v`, i.e., ``v\\cdot \\nabla p``. The result is either divided by unity or
the grid cell size depending on whether `base_cache` is of type `IndexScaling`
or `GridScaling`.
"""
function convective_derivative(u::Edges{Primal},p::Nodes{Primal},base_cache::BasicILMCache)
    extra_cache = ConvectiveDerivativeCache(similar_gridgrad(cache))
    udp = similar(p)
    convective_derivative!(udp,u,p,base_cache,extra_cache)
end

"""
    convective_derivative(v::Edges{Dual},w::Nodes{Dual},base_cache::BasicILMCache)

Compute the convective derivative of `w` with velocity `v`, i.e., ``v\\cdot \\nabla w``. The result is either divided by unity or
the grid cell size depending on whether `base_cache` is of type `IndexScaling`
or `GridScaling`.
"""
function convective_derivative(u::Edges{Dual},w::Nodes{Dual},base_cache::BasicILMCache)
    extra_cache = ConvectiveDerivativeCache(similar_gridgradcurl(cache))
    udw = similar(w)
    convective_derivative!(udw,u,w,base_cache,extra_cache)
end

"""
    convective_derivative(v::Edges,base_cache::BasicILMCache)

Compute the convective derivative of `v`, i.e., ``v\\cdot \\nabla v``. The result is either divided by unity or
the grid cell size depending on whether `base_cache` is of type `IndexScaling`
or `GridScaling`.
"""
function convective_derivative(u::Edges{Primal},base_cache::BasicILMCache)
    extra_cache = ConvectiveDerivativeCache(similar_gridgrad(base_cache))
    udu = similar(u)
    convective_derivative!(udu,u,base_cache,extra_cache)
end

"""
    convective_derivative!(vdp::Nodes{Primal},v::Edges,p::Nodes{Primal},base_cache::BasicILMCache,extra_cache::ConvectiveDerivativeCache)

Compute the convective derivative of `p` with velocity `v`, i.e., ``v\\cdot \\nabla p``, and
return the result in `vdp`. The result is either divided by unity or
the grid cell size depending on whether `base_cache` is of type `IndexScaling`
or `GridScaling`. This version of the method uses `extra_cache` of type
[`ConvectiveDerivativeCache`](@ref).
"""
function convective_derivative!(udp::Nodes{Primal},u::Edges{Primal},p::Nodes{Primal},base_cache::BasicILMCache,extra_cache::ConvectiveDerivativeCache)
    fill!(udp,0.0)
    if !iszero(p) && !iszero(u)
      _unscaled_convective_derivative!(udp,u,p,extra_cache)
      _scale_derivative!(udp,base_cache)
    end
end

"""
    convective_derivative!(vdw::Nodes{Dual},v::Edges,w::Nodes{Dual},base_cache::BasicILMCache,extra_cache::ConvectiveDerivativeCache)

Compute the convective derivative of `w` with velocity `v`, i.e., ``v\\cdot \\nabla w``, and
return the result in `vdw`. The result is either divided by unity or
the grid cell size depending on whether `base_cache` is of type `IndexScaling`
or `GridScaling`. This version of the method uses `extra_cache` of type
[`ConvectiveDerivativeCache`](@ref).
"""
function convective_derivative!(udw::Nodes{Dual},u::Edges{Primal},w::Nodes{Dual},base_cache::BasicILMCache,extra_cache::ConvectiveDerivativeCache)
    fill!(udw,0.0)
    if !iszero(w) && !iszero(u)
      _unscaled_convective_derivative!(udw,u,w,extra_cache)
      _scale_derivative!(udw,base_cache)
    end
end

"""
    convective_derivative!(vdu::Edges,v::Edges,u::Edges,base_cache::BasicILMCache,extra_cache::ConvectiveDerivativeCache)

Compute the convective derivative of `u`, i.e., ``v\\cdot \\nabla u``, and
return the result in `vdu`. The result is either divided by unity or
the grid cell size depending on whether `base_cache` is of type `IndexScaling`
or `GridScaling`. This version of the method uses `extra_cache` of type
[`ConvectiveDerivativeCache`](@ref).
"""
function convective_derivative!(vdu::Edges{Primal},v::Edges{Primal},u::Edges{Primal},base_cache::BasicILMCache,extra_cache::ConvectiveDerivativeCache)
    fill!(vdu,0.0)
    if !iszero(u) && !iszero(v)
      _unscaled_convective_derivative!(vdu,v,u,extra_cache)
      _scale_derivative!(vdu,base_cache)
    end
end

"""
    convective_derivative!(vdv::Edges,v::Edges,base_cache::BasicILMCache,extra_cache::ConvectiveDerivativeCache)

Compute the convective derivative of `v`, i.e., ``v\\cdot \\nabla v``, and
return the result in `vdv`. The result is either divided by unity or
the grid cell size depending on whether `base_cache` is of type `IndexScaling`
or `GridScaling`. This version of the method uses `extra_cache` of type
[`ConvectiveDerivativeCache`](@ref).
"""
function convective_derivative!(udu::Edges{Primal},u::Edges{Primal},base_cache::BasicILMCache,extra_cache::ConvectiveDerivativeCache)
    fill!(udu,0.0)
    if !iszero(u)
      _unscaled_convective_derivative!(udu,u,extra_cache)
      _scale_derivative!(udu,base_cache)
    end
end

# u.∇p, where u is primal edge data and p is primal node data
function _unscaled_convective_derivative!(udp::Nodes{Primal},u::Edges{Primal},p::Nodes{Primal},extra_cache::ConvectiveDerivativeCache)
    @unpack vt1_cache, vt3_cache = extra_cache

    fill!(vt1_cache,0.0)
    grad!(vt1_cache,p)
    product!(vt3_cache,u,vt1_cache)
    fill!(udp,0.0)
    grid_interpolate!(udp,vt3_cache)
    udp
end

# u.∇w, where u is primal edge data and w is dual node data
function _unscaled_convective_derivative!(udw::Nodes{Dual},u::Edges{Primal},w::Nodes{Dual},extra_cache::ConvectiveDerivativeCache{VT}) where {VT<:Edges{Dual}}
    @unpack vt1_cache, vt2_cache, vt3_cache = extra_cache

    fill!(vt1_cache,0.0)
    grad!(vt1_cache,w)
    fill!(vt2_cache,0.0)
    grid_interpolate!(vt2_cache,u)
    product!(vt3_cache,vt2_cache,vt1_cache)
    fill!(udw,0.0)
    grid_interpolate!(udw,vt3_cache)
    udw
end

# u.∇u, where u is primal edge data
function _unscaled_convective_derivative!(udu::Edges{Primal},u::Edges{Primal},extra_cache::ConvectiveDerivativeCache)
    @unpack vt1_cache, vt2_cache, vt3_cache = extra_cache

    fill!(vt1_cache,0.0)
    grid_interpolate!(vt1_cache,u)
    CartesianGrids.transpose!(vt2_cache,vt1_cache)
    fill!(vt1_cache,0.0)
    grad!(vt1_cache,u)
    product!(vt3_cache,vt2_cache,vt1_cache)
    fill!(udu,0.0)
    grid_interpolate!(udu,vt3_cache)
    udu
end

# v.∇u, where v and u are both primal edge data
function _unscaled_convective_derivative!(vdu::Edges{Primal},v::Edges{Primal},u::Edges{Primal},extra_cache::ConvectiveDerivativeCache)
    @unpack vt1_cache, vt2_cache, vt3_cache = extra_cache

    fill!(vt1_cache,0.0)
    grid_interpolate!(vt1_cache,v)
    CartesianGrids.transpose!(vt2_cache,vt1_cache)
    fill!(vt1_cache,0.0)
    grad!(vt1_cache,u)
    product!(vt3_cache,vt2_cache,vt1_cache)
    fill!(udv,0.0)
    grid_interpolate!(udv,vt3_cache)
    udu
end

# Rotational form of the convective derivative
"""
    RotConvectiveDerivativeCache(w::Nodes{Dual})

Create a cache (a subtype of [`AbstractExtraILMCache`](@ref)) for computing
the convective derivative in rotational form, using `w` to define the cache data.
"""
struct RotConvectiveDerivativeCache{VTT} <: AbstractExtraILMCache
   vt1_cache :: VTT
end

"""
    RotConvectiveDerivativeCache(cache::BasicILMCache)

Create a cache for computing the rotational convective derivative, based on
the basic ILM cache `cache`.
"""
RotConvectiveDerivativeCache(cache::BasicILMCache) = RotConvectiveDerivativeCache(similar_gridcurl(cache))


"""
    w_cross_v!(vw::Edges{Primal},w::Nodes{Dual},v::Edges{Primal},base_cache::BasicILMCache,extra_cache::RotConvectiveDerivativeCache)

Compute the term `w \\times v`, with vorticity `w` and velocity `v`, and
return the result in `vw`. This version of the method uses `extra_cache` of type
[`RotConvectiveDerivativeCache`](@ref).
"""
function w_cross_v!(uw::Edges{Primal},w::Nodes{Dual},u::Edges{Primal},base_cache::BasicILMCache,extra_cache::RotConvectiveDerivativeCache)
    fill!(uw,0.0)
    if !iszero(w) && !iszero(u)
      _unscaled_w_cross_v!(uw,w,u,extra_cache)
    end
end

function w_cross_v(w::Nodes{Dual},u::Edges{Primal},base_cache::BasicILMCache)
    extra_cache = RotConvectiveDerivativeCache(similar_gridcurl(base_cache))
    uw = similar(u)
    w_cross_v!(uw,w,u,base_cache,extra_cache)
end

function _unscaled_w_cross_v!(uw::Edges{Primal},w::Nodes{Dual},u::Edges{Primal},extra_cache::RotConvectiveDerivativeCache)
    @unpack vt1_cache = extra_cache

    fill!(vt1_cache,0.0)
    grid_interpolate!(uw.u,grid_interpolate!(vt1_cache, -u.v) ∘ w)
    fill!(vt1_cache,0.0)
    grid_interpolate!(uw.v,grid_interpolate!(vt1_cache,  u.u) ∘ w)
    return uw

end
