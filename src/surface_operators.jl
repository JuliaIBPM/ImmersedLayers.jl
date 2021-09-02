for f in [:regularize!, :interpolate!, :regularize_normal!,
          :normal_interpolate!,:surface_curl!,:surface_divergence!,
          :surface_grad!]
  @eval $f(a,b,sys::ILMSystem) = $f(a,b,sys.base_cache)
end

"""
    regularize!(s::Nodes{Primal},f::ScalarData,cache::BasicILMCache)
    regularize!(s::Nodes{Primal},f::ScalarData,sys::ILMSystem)


The operation ``s = R_c f``, which regularizes scalar surface data `f`
onto the grid in the form of scalar grid data `s`. This is the adjoint
to [`interpolate!`](@ref)
"""
@inline regularize!(s::Nodes{Primal},f::ScalarData,cache::BasicILMCache) = regularize!(s,f,cache.R)

function regularize!(s::Nodes{Primal},f::ScalarData,Rc::RegularizationMatrix)
  return s .= Rc*f
end

"""
    regularize!(v::Edges{Primal},vb::VectorData,cache::BasicILMCache)
    regularize!(v::Edges{Primal},vb::VectorData,sys::ILMSystem)

The operation ``v = R_f v_b``, which regularizes vector surface data `vb`
onto the grid in the form of scalar grid data `v`. This is the adjoint
to [`interpolate!`](@ref)
"""
@inline regularize!(v::Edges{Primal},vb::VectorData,cache::BasicILMCache) = regularize!(v,vb,cache.R)

function regularize!(v::Edges{Primal},vb::VectorData,Rf::RegularizationMatrix)
  return v .= Rf*vb
end

"""
    interpolate!(f::ScalarData,s::Nodes{Primal},cache::BasicILMCache)
    interpolate!(f::ScalarData,s::Nodes{Primal},sys::ILMSystem)

The operation ``f = R_c^T s``, which interpolates scalar grid data `s`
onto the surface points in the form of scalar point data `f`. This is the adjoint
to [`regularize!`](@ref)
"""
@inline interpolate!(f::ScalarData,s::Nodes{Primal},cache::BasicILMCache) = interpolate!(f,s,cache.E)

function interpolate!(f::ScalarData,s::Nodes{Primal},Ec::InterpolationMatrix)
  return f .= Ec*s
end

"""
    interpolate!(vb::VectorData,v::Edges{Primal},cache::BasicILMCache)
    interpolate!(vb::VectorData,v::Edges{Primal},sys::ILMSystem)

The operation ``v_b = R_c^T v``, which interpolates vector grid data `v`
onto the surface points in the form of scalar point data `vb`. This is the adjoint
to [`regularize!`](@ref)
"""
@inline interpolate!(vb::VectorData,v::Edges{Primal},cache::BasicILMCache) = interpolate!(vb,v,cache.E)

function interpolate!(vb::VectorData,v::Edges{Primal},Ef::InterpolationMatrix)
  return vb .= Ef*v
end

"""
    regularize_normal!(q::Edges{Primal},f::ScalarData,cache::BasicILMCache)
    regularize_normal!(q::Edges{Primal},f::ScalarData,sys::ILMSystem)

The operation ``q = R_f n\\circ f``, which maps scalar surface data `f` (like
a jump in scalar potential) to grid data `q` (like velocity). This is the adjoint
to [`normal_interpolate!`](@ref).
"""
@inline regularize_normal!(q::Edges{Primal},f::ScalarData,cache::BasicILMCache) = regularize_normal!(q,f,cache.nrm,cache.Rsn,cache.snorm_cache)

function regularize_normal!(q::Edges{Primal,NX,NY},f::ScalarData{N},nrm::VectorData{N},Rf::RegularizationMatrix,snorm_cache::VectorData{N}) where {NX,NY,N}
    product!(snorm_cache,nrm,f)
    q .= Rf*snorm_cache
end

"""
    regularize_normal!(qt::EdgeGradient{Primal},v::VectorData,cache::BasicILMCache)
    regularize_normal!(qt::EdgeGradient{Primal},v::VectorData,sys::ILMSystem)

The operation ``q_t = R_t n\\circ v``, which maps scalar vector data `v` (like
a jump in velocity) to grid data `qt` (like velocity-normal tensor). This is the adjoint
to [`normal_interpolate!`](@ref).
"""
@inline regularize_normal!(q::EdgeGradient{Primal},f::VectorData,cache::BasicILMCache) = regularize_normal!(q,f,cache.nrm,cache.Rsn,cache.snorm_cache,cache.snorm2_cache)

function regularize_normal!(q::EdgeGradient{Primal,Dual,NX,NY},f::VectorData{N},nrm::VectorData{N},Rf::RegularizationMatrix,snorm_cache::TensorData{N},snorm2_cache::TensorData{N}) where {NX,NY,N}
    tensorproduct!(snorm_cache,nrm,f)
    transpose!(snorm2_cache,snorm_cache)
    snorm_cache .+= snorm2_cache
    q .= Rf*snorm_cache
end


"""
    normal_interpolate!(vn::ScalarData,q::Edges{Primal},cache::BasicILMCache)
    normal_interpolate!(vn::ScalarData,q::Edges{Primal},sys::ILMSystem)

The operation ``v_n = n \\cdot R_f^T q``, which maps grid data `q` (like velocity) to scalar
surface data `vn` (like normal component of surface velocity). This is the
adjoint to [`regularize_normal!`](@ref).
"""
@inline normal_interpolate!(vn::ScalarData,q::Edges{Primal},cache::BasicILMCache) = normal_interpolate!(vn,q,cache.nrm,cache.Esn,cache.snorm_cache)

function normal_interpolate!(vn::ScalarData{N},q::Edges{Primal,NX,NY},nrm::VectorData{N},Ef::InterpolationMatrix,snorm_cache::VectorData{N}) where {NX,NY,N}
    snorm_cache .= Ef*q
    pointwise_dot!(vn,nrm,snorm_cache)
end

"""
    normal_interpolate!(τ::VectorData,A::EdgeGradient{Primal},cache::BasicILMCache)
    normal_interpolate!(τ::VectorData,A::EdgeGradient{Primal},sys::ILMSystem)

The operation ``\\tau = n \\cdot R_t^T (A + A^T)``, which maps grid tensor data `A` (like velocity gradient tensor) to vector
surface data `τ` (like traction). This is the adjoint to [`regularize_normal!`](@ref).
"""
@inline normal_interpolate!(vn::VectorData,q::EdgeGradient{Primal},cache::BasicILMCache) = normal_interpolate!(vn,q,cache.nrm,cache.Esn,cache.gsnorm2_cache,cache.snorm_cache)

function normal_interpolate!(vn::VectorData{N},q::EdgeGradient{Primal,Dual,NX,NY},nrm::VectorData{N},Ef::InterpolationMatrix,gsnorm_cache::EdgeGradient{Primal,Dual,NX,NY},snorm_cache::TensorData{N}) where {NX,NY,N}
    transpose!(gsnorm_cache,q)
    gsnorm_cache .+= q
    snorm_cache .= Ef*gsnorm_cache
    pointwise_dot!(vn,nrm,snorm_cache)
end

"""
    surface_curl!(w::Nodes{Dual},f::ScalarData,cache::BasicILMCache)
    surface_curl!(w::Nodes{Dual},f::ScalarData,sys::ILMSystem)

The operation ``w = C_s^T f = C^T R_f n\\circ f``, which maps scalar surface data `f` (like
a jump in scalar potential) to grid data `w` (like vorticity). This is the adjoint
to ``C_s``, also given by `surface_curl!` (but with arguments switched). Note that
the differential operations are divided either by 1 or by the grid cell size,
depending on whether `sys` has been designated with `IndexScaling` or `GridScaling`,
respectively.
"""
function surface_curl!(w::Nodes{Dual},f::ScalarData,cache::BasicILMCache)
  _unscaled_surface_curl!(w,f,cache.nrm,cache.Rsn,cache.gsnorm_cache,cache.snorm_cache)
  _scale_derivative!(w,cache)
end

function _unscaled_surface_curl!(w::Nodes{Dual,NX,NY},f::ScalarData{N},nrm::VectorData{N},Rf::RegularizationMatrix,q_cache::Edges{Primal,NX,NY},snorm_cache::VectorData{N}) where {NX,NY,N}
    regularize_normal!(q_cache,f,nrm,Rf,snorm_cache)
    curl!(w,q_cache)
end

"""
    surface_curl!(vn::ScalarData,s::Nodes{Dual},cache::BasicILMCache)
    surface_curl!(vn::ScalarData,s::Nodes{Dual},sys::ILMSystem)

The operation ``v_n = C_s s = n \\cdot R_f^T C s``, which maps grid data `s` (like
streamfunction) to scalar surface data `vn` (like normal component of velocity).
This is the adjoint to ``C_s^T``, also given by `surface_curl!`, but with
arguments switched.  Note that
the differential operations are divided either by 1 or by the grid cell size,
depending on whether `sys` has been designated with `IndexScaling` or `GridScaling`,
respectively.
"""
function surface_curl!(vn::ScalarData,s::Nodes{Dual},cache::BasicILMCache)
  _unscaled_surface_curl!(vn,s,cache.nrm,cache.Esn,cache.gsnorm_cache,cache.snorm_cache)
  _scale_derivative!(vn,cache)
end

function _unscaled_surface_curl!(vn::ScalarData{N},s::Nodes{Dual,NX,NY},nrm::VectorData{N},Ef::InterpolationMatrix,q_cache::Edges{Primal,NX,NY},snorm_cache::VectorData{N}) where {NX,NY,N}
    fill!(q_cache,0.0)
    curl!(q_cache,s)
    normal_interpolate!(vn,q_cache,nrm,Ef,snorm_cache)
end


"""
    surface_divergence!(Θ::Nodes{Primal},f::ScalarData,cache::BasicILMCache)
    surface_divergence!(Θ::Nodes{Primal},f::ScalarData,sys::ILMSystem)

The operation ``\\theta = D_s f = D R_f n \\circ f``, which maps surface scalar data `f` (like
jump in scalar potential) to grid data `Θ` (like dilatation, i.e. divergence of velocity).
This is the adjoint of [`surface_grad!`](@ref).
Note that
the differential operations are divided either by 1 or by the grid cell size,
depending on whether `sys` has been designated with `IndexScaling` or `GridScaling`,
respectively.
"""
function surface_divergence!(θ::Nodes{Primal},f::ScalarData,cache::BasicILMCache)
  _unscaled_surface_divergence!(θ,f,cache.nrm,cache.Rsn,cache.gsnorm_cache,cache.snorm_cache)
  _scale_derivative!(θ,cache)
end

function _unscaled_surface_divergence!(θ::Nodes{Primal,NX,NY},f::ScalarData{N},nrm::VectorData{N},Rf::RegularizationMatrix,q_cache::Edges{Primal,NX,NY},snorm_cache::VectorData{N}) where {NX,NY,N}
    regularize_normal!(q_cache,f,nrm,Rf,snorm_cache)
    divergence!(θ,q_cache)
end

"""
    surface_divergence!(v::Edges{Primal},dv::VectorData,cache::BasicILMCache)
    surface_divergence!(v::Edges{Primal},dv::VectorData,sys::ILMSystem)

The operation ``v = D_s dv = D R_f (n \\circ dv + dv \\circ n)``, which maps surface vector data `dv` (like
jump in velocity) to grid data `v` (like velocity). This is the adjoint of [`surface_grad!`](@ref). Note that
the differential operations are divided either by 1 or by the grid cell size,
depending on whether `sys` has been designated with `IndexScaling` or `GridScaling`,
respectively.
"""
function surface_divergence!(θ::Edges{Primal},f::VectorData,cache::BasicILMCache)
  _unscaled_surface_divergence!(θ,f,cache.nrm,cache.Rsn,cache.gsnorm_cache,cache.snorm_cache,cache.snorm2_cache)
  _scale_derivative!(θ,cache)
end

function _unscaled_surface_divergence!(θ::Edges{Primal,NX,NY},f::VectorData{N},nrm::VectorData{N},Rf::RegularizationMatrix,q_cache::EdgeGradient{Primal,Dual,NX,NY},snorm_cache::TensorData{N},snorm2_cache::TensorData{N}) where {NX,NY,N}
    regularize_normal!(q_cache,f,nrm,Rf,snorm_cache,snorm2_cache)
    divergence!(θ,q_cache)
end

"""
    surface_grad!(vn::ScalarData,ϕ::Nodes{Primal},cache::BasicILMCache)
    surface_grad!(vn::ScalarData,ϕ::Nodes{Primal},sys::ILMSystem)

The operation ``v_n = G_s\\phi = n \\cdot R_f^T G\\phi``, which maps grid data `ϕ` (like
scalar potential) to scalar surface data (like normal component of velocity). This is the adjoint of [`surface_divergence!`](@ref).
Note that
the differential operations are divided either by 1 or by the grid cell size,
depending on whether `sys` has been designated with `IndexScaling` or `GridScaling`,
respectively.
"""
function surface_grad!(vn::ScalarData,ϕ::Nodes{Primal},cache::BasicILMCache)
  _unscaled_surface_grad!(vn,ϕ,cache.nrm,cache.Esn,cache.gsnorm_cache,cache.snorm_cache)
  _scale_derivative!(vn,cache)
end

function _unscaled_surface_grad!(vn::ScalarData{N},ϕ::Nodes{Primal,NX,NY},nrm::VectorData{N},Ef::InterpolationMatrix,q_cache::Edges{Primal,NX,NY},snorm_cache::VectorData{N}) where {NX,NY,N}
    fill!(q_cache,0.0)
    grad!(q_cache,ϕ)
    normal_interpolate!(vn,q_cache,nrm,Ef,snorm_cache)
end

"""
    surface_grad!(τ::VectorData,v::Edges{Primal},cache::BasicILMCache)
    surface_grad!(τ::VectorData,v::Edges{Primal},sys::ILMSystem)

The operation ``\\tau = G_s v = n \\cdot R_t^T (G v + (G v)^T)``, which maps grid vector data `v` (like
velocity) to vector surface data `τ` (like traction). This is the adjoint of [`surface_divergence!`](@ref). Note that
the differential operations are divided either by 1 or by the grid cell size,
depending on whether `sys` has been designated with `IndexScaling` or `GridScaling`,
respectively.
"""
function surface_grad!(vn::VectorData,ϕ::Edges{Primal},cache::BasicILMCache)
  _unscaled_surface_grad!(vn,ϕ,cache.nrm,cache.Esn,cache.gsnorm_cache,cache.gsnorm2_cache,cache.snorm_cache)
  _scale_derivative!(vn,cache)
end

function _unscaled_surface_grad!(vn::VectorData{N},ϕ::Edges{Primal,NX,NY},nrm::VectorData{N},Ef::InterpolationMatrix,gsnorm_cache::EdgeGradient{Primal,Dual,NX,NY},gsnorm2_cache::EdgeGradient{Primal,Dual,NX,NY},snorm_cache::TensorData{N}) where {NX,NY,N}
    fill!(gsnorm_cache,0.0)
    grad!(gsnorm_cache,ϕ)
    normal_interpolate!(vn,gsnorm_cache,nrm,Ef,gsnorm2_cache,snorm_cache)
end

"""
    mask(cache::BasicILMCache) -> GridData
    mask(sys::ILMSystem) -> GridData

Create grid data that consist of 1s inside of a surface (i.e., on a side opposite
  the normal vectors) and 0s outside. The grid data are the same type as the
  output data type of `sys`.  Only allows `sys` to have `GridScaling`.
"""
@inline mask(cache::BasicILMCache{N,GridScaling}) where {N} = _mask!(zeros_grid(cache),cache)

"""
    complementary_mask(cache::BasicILMCache) -> GridData
    complementary_mask(sys::ILMSystem) -> GridData

Create grid data that consist of 0s inside of a surface (i.e., on a side opposite
  the normal vectors) and 1s outside. The grid data are the same type as the
  output data type of `cache`.  Only allows `cache` to have `GridScaling`.
"""
@inline complementary_mask(cache::BasicILMCache{N,GridScaling}) where {N} =
          _complementary_mask!(zeros_grid(cache),cache)


"""
    mask!(w::GridData,cache::BasicILMCache)
    mask!(w::GridData,sys::ILMSystem)

Mask the data `w` in place by multiplying it by 1s inside of a surface (i.e., on a side opposite
  the normal vectors) and 0s outside. The grid data `w` must be of the same type as the
  output data type of `cache`. Only allows `cache` to have `GridScaling`.
"""
function mask!(w::T,cache::BasicILMCache{N,GridScaling}) where {T <: GridData,N}
  @unpack gdata_cache = cache
  _mask!(gdata_cache,cache)
  product!(w,gdata_cache,w)
end

"""
    complementary_mask!(w::GridData,cache::BasicILMCache)
    complementary_mask!(w::GridData,sys::ILMSystem)

Mask the data `w` in place by multiplying it by 0s inside of a surface (i.e., on a side opposite
  the normal vectors) and 1s outside. The grid data `w` must be of the same type as the
  output data type of `cache`. Only allows `cache` to have `GridScaling`.
"""
function complementary_mask!(w::T,cache::BasicILMCache{N,GridScaling}) where {T <: GridData,N}
  @unpack gdata_cache = cache
  _complementary_mask!(gdata_cache,cache)
  product!(w,gdata_cache,w)
end

function _mask!(msk,cache)
  @unpack sdata_cache, gdata_cache, L = cache
  typeof(msk) == typeof(gdata_cache) || error("Wrong data type")
  fill!(sdata_cache,1.0)
  surface_divergence!(msk,sdata_cache,cache)
  inverse_laplacian!(msk,cache)
  msk .*= -1.0
end

function _complementary_mask!(msk,cache)
    _mask!(msk,cache)
    msk .= 1.0 - msk
end
