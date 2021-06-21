import CartesianGrids: convective_derivative!

_scale_derivative!(w,cache::BasicILMCache{N,IndexScaling}) where {N} = w
_scale_derivative!(w,cache::BasicILMCache{N,GridScaling}) where {N} = w ./= cellsize(cache)

_scale_inverse_laplacian!(w,cache::BasicILMCache{N,IndexScaling}) where {N} = w
_scale_inverse_laplacian!(w,cache::BasicILMCache{N,GridScaling}) where {N} = w .*= cellsize(cache)^2

"""
    inverse_laplacian!(w::GridData,sys::ILMSystem)

Compute the in-place inverse Laplacian of grid data `w`, and multiply the result
by unity or by the grid cell size, depending on whether `sys` has `IndexScaling` or `GridScaling`,
respectively.
"""
inverse_laplacian!(w,sys::ILMSystem) = inverse_laplacian!(w,sys.base_cache)

function inverse_laplacian!(w::GridData,cache::BasicILMCache)
    @unpack L = cache
    _unscaled_inverse_laplacian!(w,L)
    _scale_inverse_laplacian!(w,cache)
end

_unscaled_inverse_laplacian!(w::GridData,L::CartesianGrids.Laplacian) = w .= L\w

###  Convective derivative of velocity-like data by itself ###

struct ConvectiveDerivativeCache{VTT} <: AbstractExtraILMCache
   vt1_cache :: VTT
   vt2_cache :: VTT
   vt3_cache :: VTT
   ConvectiveDerivativeCache(vt::EdgeGradient) = new{typeof(vt)}(similar(vt),similar(vt),similar(vt))
end

"""
    convective_derivative(v::Edges,base_cache::BasicILMCache)

Compute the convective derivative of `v`, i.e., ``v\\cdot \\nabla v``. The result is either divided by unity or
the grid cell size depending on whether `base_cache` is of type `IndexScaling`
or `GridScaling`.
"""
function convective_derivative(u::Edges{Primal},base_cache::BasicILMCache)
    extra_cache = ConvectiveDerivativeCache(Edges(Primal,u))
    udu = similar(u)
    convective_derivative!(udu,u,base_cache,extra_cache)
end

"""
    convective_derivative!(vdv::Edges,v::Edges,base_cache::BasicILMCache,extra_cache::ConvectiveDerivativeCache)

Compute the convective derivative of `v`, i.e., ``v\\cdot \\nabla v``, and
return the result in `vdv`. The result is either divided by unity or
the grid cell size depending on whether `base_cache` is of type `IndexScaling`
or `GridScaling`.
"""
function convective_derivative!(udu::Edges{Primal},u::Edges{Primal},base_cache::BasicILMCache,extra_cache::ConvectiveDerivativeCache)
    _unscaled_convective_derivative!(udu,u,extra_cache)
    ImmersedLayers._scale_derivative!(udu,base_cache)
end

function _unscaled_convective_derivative!(udu::Edges{Primal},u::Edges{Primal},extra_cache::ConvectiveDerivativeCache)
    @unpack vt1_cache, vt2_cache, vt3_cache = extra_cache

    grid_interpolate!(vt1_cache,u)
    CartesianGrids.transpose!(vt2_cache,vt1_cache)
    fill!(vt1_cache,0.0)
    grad!(vt1_cache,u)
    product!(vt3_cache,vt2_cache,vt1_cache)
    fill!(udu,0.0)
    grid_interpolate!(udu,vt3_cache)
    udu
end
