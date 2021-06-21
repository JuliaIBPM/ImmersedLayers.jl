for f in [:regularize_normal!,:normal_interpolate!,:surface_curl!,:surface_divergence!,:surface_grad!]
  @eval $f(a,b,sys::ILMSystem) = $f(a,b,sys.base_cache)
end


"""
    regularize_normal!(q::Edges{Primal},f::ScalarData,sys::ILMSystem)

The operation ``R_f n\\circ``, which maps scalar surface data `f` (like
a jump in scalar potential) to grid data `q` (like velocity). This is the adjoint
to `normal_interpolate!`.
"""
@inline regularize_normal!(q::Edges{Primal},f::ScalarData,cache::BasicILMCache) = regularize_normal!(q,f,cache.nrm,cache.R,cache.snorm_cache)

function regularize_normal!(q::Edges{Primal,NX,NY},f::ScalarData{N},nrm::VectorData{N},Rf::RegularizationMatrix,snorm_cache::VectorData{N}) where {NX,NY,N}
    product!(snorm_cache,nrm,f)
    q .= Rf*snorm_cache
end

"""
    regularize_normal!(qt::EdgeGradient{Primal},v::VectorData,sys::ILMSystem)

The operation ``R_T n\\circ``, which maps scalar vector data `v` (like
a jump in velocity) to grid data `qt` (like velocity-normal tensor). This is the adjoint
to `normal_interpolate!`.
"""
@inline regularize_normal!(q::EdgeGradient{Primal},f::VectorData,cache::BasicILMCache) = regularize_normal!(q,f,cache.nrm,cache.R,cache.snorm_cache,cache.snorm2_cache)

function regularize_normal!(q::EdgeGradient{Primal,Dual,NX,NY},f::VectorData{N},nrm::VectorData{N},Rf::RegularizationMatrix,snorm_cache::TensorData{N},snorm2_cache::TensorData{N}) where {NX,NY,N}
    tensorproduct!(snorm_cache,nrm,f)
    transpose!(snorm2_cache,snorm_cache)
    snorm_cache .+= snorm2_cache
    q .= Rf*snorm_cache
end


"""
    normal_interpolate!(vn::ScalarData,q::Edges{Primal},sys::ILMSystem)

The operation ``n \\cdot I_f``, which maps grid data `q` (like velocity) to scalar
surface data `vn` (like normal component of surface velocity). This is the
adjoint to `regularize_normal!`.
"""
@inline normal_interpolate!(vn::ScalarData,q::Edges{Primal},cache::BasicILMCache) = normal_interpolate!(vn,q,cache.nrm,cache.E,cache.snorm_cache)

function normal_interpolate!(vn::ScalarData{N},q::Edges{Primal,NX,NY},nrm::VectorData{N},Ef::InterpolationMatrix,snorm_cache::VectorData{N}) where {NX,NY,N}
    snorm_cache .= Ef*q
    pointwise_dot!(vn,nrm,snorm_cache)
end

"""
    normal_interpolate!(τ::VectorData,A::EdgeGradient{Primal},sys::ILMSystem)

The operation ``n \\cdot I_T (A + A^T)``, which maps grid tensor data `A` (like velocity gradient tensor) to vector
surface data `τ` (like traction). This is the adjoint to `regularize_normal!`.
"""
@inline normal_interpolate!(vn::VectorData,q::EdgeGradient{Primal},cache::BasicILMCache) = normal_interpolate!(vn,q,cache.nrm,cache.E,cache.gsnorm2_cache,cache.snorm_cache)

function normal_interpolate!(vn::VectorData{N},q::EdgeGradient{Primal,Dual,NX,NY},nrm::VectorData{N},Ef::InterpolationMatrix,gsnorm_cache::EdgeGradient{Primal,Dual,NX,NY},snorm_cache::TensorData{N}) where {NX,NY,N}
    transpose!(gsnorm_cache,q)
    gsnorm_cache .+= q
    snorm_cache .= Ef*gsnorm_cache
    pointwise_dot!(vn,nrm,snorm_cache)
end

"""
    surface_curl!(w::Nodes{Dual},f::ScalarData,sys::ILMSystem)

The operation ``C_s^T = C^T R_f n\\circ``, which maps scalar surface data `f` (like
a jump in scalar potential) to grid data `w` (like vorticity). This is the adjoint
to ``C_s``, also given by `surface_curl!` (but with arguments switched). Note that
the differential operations are divided either by 1 or by the grid cell size,
depending on whether `sys` has been designated with `IndexScaling` or `GridScaling`,
respectively.
"""
function surface_curl!(w::Nodes{Dual},f::ScalarData,cache::BasicILMCache)
  _unscaled_surface_curl!(w,f,cache.nrm,cache.R,cache.gsnorm_cache,cache.snorm_cache)
  _scale_derivative!(w,cache)
end

function _unscaled_surface_curl!(w::Nodes{Dual,NX,NY},f::ScalarData{N},nrm::VectorData{N},Rf::RegularizationMatrix,q_cache::Edges{Primal,NX,NY},snorm_cache::VectorData{N}) where {NX,NY,N}
    regularize_normal!(q_cache,f,nrm,Rf,snorm_cache)
    curl!(w,q_cache)
end

"""
    surface_curl!(vn::ScalarData,s::Nodes{Dual},sys::ILMSystem)

The operation ``C_s = n \\cdot I_f C``, which maps grid data `s` (like
streamfunction) to scalar surface data `vn` (like normal component of velocity).
This is the adjoint to ``C_s^T``, also given by `surface_curl!`, but with
arguments switched.  Note that
the differential operations are divided either by 1 or by the grid cell size,
depending on whether `sys` has been designated with `IndexScaling` or `GridScaling`,
respectively.
"""
function surface_curl!(vn::ScalarData,s::Nodes{Dual},cache::BasicILMCache)
  _unscaled_surface_curl!(vn,s,cache.nrm,cache.E,cache.gsnorm_cache,cache.snorm_cache)
  _scale_derivative!(vn,cache)
end

function _unscaled_surface_curl!(vn::ScalarData{N},s::Nodes{Dual,NX,NY},nrm::VectorData{N},Ef::InterpolationMatrix,q_cache::Edges{Primal,NX,NY},snorm_cache::VectorData{N}) where {NX,NY,N}
    fill!(q_cache,0.0)
    curl!(q_cache,s)
    normal_interpolate!(vn,q_cache,nrm,Ef,snorm_cache)
end


"""
    surface_divergence!(Θ::Nodes{Primal},f::ScalarData,sys::ILMSystem)

The operation ``D_s = D R_f n \\circ``, which maps surface scalar data `f` (like
jump in scalar potential) to grid data `Θ` (like dilatation, i.e. divergence of velocity).
Note that
the differential operations are divided either by 1 or by the grid cell size,
depending on whether `sys` has been designated with `IndexScaling` or `GridScaling`,
respectively.
"""
function surface_divergence!(θ::Nodes{Primal},f::ScalarData,cache::BasicILMCache)
  _unscaled_surface_divergence!(θ,f,cache.nrm,cache.R,cache.gsnorm_cache,cache.snorm_cache)
  _scale_derivative!(θ,cache)
end

function _unscaled_surface_divergence!(θ::Nodes{Primal,NX,NY},f::ScalarData{N},nrm::VectorData{N},Rf::RegularizationMatrix,q_cache::Edges{Primal,NX,NY},snorm_cache::VectorData{N}) where {NX,NY,N}
    regularize_normal!(q_cache,f,nrm,Rf,snorm_cache)
    divergence!(θ,q_cache)
end

"""
    surface_divergence!(v::Edges{Primal},dv::VectorData,sys::ILMSystem)

The operation ``D_s = D R_f (n \\circ \\cdot + \\cdot \\circ n)``, which maps surface vector data `v` (like
jump in velocity) to grid data `v` (like velocity). Note that
the differential operations are divided either by 1 or by the grid cell size,
depending on whether `sys` has been designated with `IndexScaling` or `GridScaling`,
respectively.
"""
function surface_divergence!(θ::Edges{Primal},f::VectorData,cache::BasicILMCache)
  _unscaled_surface_divergence!(θ,f,cache.nrm,cache.R,cache.gsnorm_cache,cache.snorm_cache,cache.snorm2_cache)
  _scale_derivative!(θ,cache)
end

function _unscaled_surface_divergence!(θ::Edges{Primal,NX,NY},f::VectorData{N},nrm::VectorData{N},Rf::RegularizationMatrix,q_cache::EdgeGradient{Primal,Dual,NX,NY},snorm_cache::TensorData{N},snorm2_cache::TensorData{N}) where {NX,NY,N}
    regularize_normal!(q_cache,f,nrm,Rf,snorm_cache,snorm2_cache)
    divergence!(θ,q_cache)
end

"""
    surface_grad!(vn::ScalarData,ϕ::Nodes{Primal},sys::ILMSystem)

The operation ``G_s = n \\cdot I_f G``, which maps grid data `ϕ` (like
scalar potential) to scalar surface data (like normal component of velocity).  Note that
the differential operations are divided either by 1 or by the grid cell size,
depending on whether `sys` has been designated with `IndexScaling` or `GridScaling`,
respectively.
"""
function surface_grad!(vn::ScalarData,ϕ::Nodes{Primal},cache::BasicILMCache)
  _unscaled_surface_grad!(vn,ϕ,cache.nrm,cache.E,cache.gsnorm_cache,cache.snorm_cache)
  _scale_derivative!(vn,cache)
end

function _unscaled_surface_grad!(vn::ScalarData{N},ϕ::Nodes{Primal,NX,NY},nrm::VectorData{N},Ef::InterpolationMatrix,q_cache::Edges{Primal,NX,NY},snorm_cache::VectorData{N}) where {NX,NY,N}
    fill!(q_cache,0.0)
    grad!(q_cache,ϕ)
    normal_interpolate!(vn,q_cache,nrm,Ef,snorm_cache)
end

"""
    surface_grad!(τ::VectorData,v::Edges{Primal},sys::ILMSystem)

The operation ``G_s = n \\cdot I_f (G v + (G v)^T)``, which maps grid vector data `v` (like
velocity) to vector surface data `τ` (like traction). Note that
the differential operations are divided either by 1 or by the grid cell size,
depending on whether `sys` has been designated with `IndexScaling` or `GridScaling`,
respectively.
"""
function surface_grad!(vn::VectorData,ϕ::Edges{Primal},cache::BasicILMCache)
  _unscaled_surface_grad!(vn,ϕ,cache.nrm,cache.E,cache.gsnorm_cache,cache.gsnorm2_cache,cache.snorm_cache)
  _scale_derivative!(vn,cache)
end

function _unscaled_surface_grad!(vn::VectorData{N},ϕ::Edges{Primal,NX,NY},nrm::VectorData{N},Ef::InterpolationMatrix,gsnorm_cache::EdgeGradient{Primal,Dual,NX,NY},gsnorm2_cache::EdgeGradient{Primal,Dual,NX,NY},snorm_cache::TensorData{N}) where {NX,NY,N}
    fill!(gsnorm_cache,0.0)
    grad!(gsnorm_cache,ϕ)
    normal_interpolate!(vn,gsnorm_cache,nrm,Ef,gsnorm2_cache,snorm_cache)
end

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

"""
    mask(sys::ILMSystem) -> GridData

Create grid data that consist of 1s inside of a surface (i.e., on a side opposite
  the normal vectors) and 0s outside. The grid data are the same type as the
  output data type of `sys`.  Only allows `sys` to have `GridScaling`.
"""
@inline mask(cache::BasicILMCache{N,GridScaling}) where {N} = _mask!(cache.gdata_cache,cache)

"""
    complementary_mask(cache::BasicILMCache) -> GridData

Create grid data that consist of 0s inside of a surface (i.e., on a side opposite
  the normal vectors) and 1s outside. The grid data are the same type as the
  output data type of `cache`.  Only allows `cache` to have `GridScaling`.
"""
@inline complementary_mask(cache::BasicILMCache{N,GridScaling}) where {N} =
          _complementary_mask!(cache.gdata_cache,cache)


"""
    mask!(w::GridData,cache::BasicILMCache)

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

"""
    CLinvCT(cache::BasicILMCache[;scale=1.0])

Construct the square matrix ``-C_s L^{-1}C_s^T``, which maps data of type `ScalarData`
to data of the same type. The operators `C_s` and `C_s^T` correspond to `surface_curl!`
and `L` is the grid Laplacian. The optional keyword `scale` multiplies the
matrix by the designated value.
"""
function CLinvCT(cache::BasicILMCache{N};scale=1.0) where {N}
    @unpack L, sdata_cache, gcurl_cache = cache

    len = length(sdata_cache)
    A = Matrix{eltype(sdata_cache)}(undef,len,len)
    fill!(sdata_cache,0.0)

    for col in 1:len
        sdata_cache[col] = 1.0
        fill!(gcurl_cache,0.0)
        surface_curl!(gcurl_cache,sdata_cache,cache)

        inverse_laplacian!(gcurl_cache,cache)
        gcurl_cache .*= -1
        surface_curl!(sdata_cache,gcurl_cache,cache)

        A[:,col] = scale*sdata_cache
        fill!(sdata_cache,0.0)
    end

    return A

end

"""
    GLinvD(cache::BasicILMCache[;scale=1.0])

Construct the square matrix ``G_s L^{-1}D_s``, which maps data of type `ScalarData`
to data of the same type. The operators `G_s` and `D_s` correspond to `surface_grad!`
and `surface_divergence!`, and `L` is the grid Laplacian. The optional keyword `scale` multiplies the
matrix by the designated value.
"""
function GLinvD(cache::BasicILMCache{N};scale=1.0) where {N}
    @unpack L, sdata_cache, gdata_cache = cache

    len = length(sdata_cache)
    A = Matrix{eltype(sdata_cache)}(undef,len,len)
    fill!(sdata_cache,0.0)

    for col in 1:len
        sdata_cache[col] = 1.0
        fill!(gdata_cache,0.0)
        surface_divergence!(gdata_cache,sdata_cache,cache)
        inverse_laplacian!(gdata_cache,cache)
        surface_grad!(sdata_cache,gdata_cache,cache)

        A[:,col] .= scale*sdata_cache
        fill!(sdata_cache,0.0)
    end

    return A

end

"""
    nRTRn(cache::BasicILMCache[;scale=1.0])

Construct the square matrix ``n\\cdot I_f R_f n \\circ``, which maps data of type `ScalarData`
to data of the same type. The operators `I_f` and `R_f` correspond to the interpolation
and regularization matrices. The optional keyword `scale` multiplies the
matrix by the designated value.
"""
function nRTRn(cache::BasicILMCache{N};scale=1.0) where {N}
    @unpack sdata_cache, gsnorm_cache = cache

    len = length(sdata_cache)
    A = Matrix{eltype(sdata_cache)}(undef,len,len)
    fill!(sdata_cache,0.0)

    for col in 1:len
        sdata_cache[col] = 1.0
        fill!(gsnorm_cache,0.0)
        regularize_normal!(gsnorm_cache,sdata_cache,cache)
        normal_interpolate!(sdata_cache,gsnorm_cache,cache)

        A[:,col] .= scale*sdata_cache
        fill!(sdata_cache,0.0)
    end

    return A

end

_scale_derivative!(w,cache::BasicILMCache{N,IndexScaling}) where {N} = w
_scale_derivative!(w,cache::BasicILMCache{N,GridScaling}) where {N} = w ./= cellsize(cache)

_scale_inverse_laplacian!(w,cache::BasicILMCache{N,IndexScaling}) where {N} = w
_scale_inverse_laplacian!(w,cache::BasicILMCache{N,GridScaling}) where {N} = w .*= cellsize(cache)^2
