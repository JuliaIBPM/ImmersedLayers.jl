# Basic surface operators
using LinearAlgebra

"""
    regularize!(s::Nodes{Primal},f::ScalarData,cache::BasicILMCache)
    regularize!(s::Nodes{Primal},f::ScalarData,sys::ILMSystem)


The operation ``s = R_c f``, which regularizes scalar surface data `f`
onto the grid in the form of scalar grid data `s`. This is the adjoint
to [`interpolate!`](@ref)
"""
@inline regularize!(s::Nodes{Primal},f::ScalarData,cache::BasicILMCache) = regularize!(s,f,cache.R)

"""
    regularize!(s::Nodes{Dual},f::ScalarData,cache::BasicILMCache)
    regularize!(s::Nodes{Dual},f::ScalarData,sys::ILMSystem)


The operation ``s = R_N f``, which regularizes scalar surface data `f`
onto the grid in the form of scalar grid (curl) data `s`. Only works if
the cache is built for vector data. This is the adjoint
to [`interpolate!`](@ref)
"""
@inline regularize!(s::Nodes{Dual},f::ScalarData,cache::BasicILMCache) = regularize!(s,f,cache.Rcurl)


"""
    regularize!(v::Edges{Primal},vb::VectorData,cache::BasicILMCache)
    regularize!(v::Edges{Primal},vb::VectorData,sys::ILMSystem)

The operation ``\\mathbf{v} = R_f \\mathbf{v}_b``, which regularizes vector surface data `vb`
onto the grid in the form of scalar grid data `v`. This is the adjoint
to [`interpolate!`](@ref)
"""
@inline regularize!(v::Edges{Primal},vb::VectorData,cache::BasicILMCache) = regularize!(v,vb,cache.R)

function regularize!(s::Nodes{C},f::ScalarData,Rc::RegularizationMatrix) where C
  fill!(s,0.0)
  mul!(s,Rc,f)
end

function regularize!(v::Edges{C},vb::VectorData,Rf::RegularizationMatrix) where C
  fill!(v,0.0)
  mul!(v,Rf,vb)
end

"""
    interpolate!(f::ScalarData,s::Nodes{Primal},cache::BasicILMCache)
    interpolate!(f::ScalarData,s::Nodes{Primal},sys::ILMSystem)

The operation ``f = R_c^T s``, which interpolates scalar grid data `s`
onto the surface points in the form of scalar point data `f`. This is the adjoint
to [`regularize!`](@ref)
"""
@inline interpolate!(f::ScalarData,s::Nodes{Primal},cache::BasicILMCache) = interpolate!(f,s,cache.E)

"""
    interpolate!(f::ScalarData,s::Nodes{Dual},cache::BasicILMCache)
    interpolate!(f::ScalarData,s::Nodes{Dual},sys::ILMSystem)

The operation ``f = R_N^T s``, which interpolates scalar grid (curl) data `s`
onto the surface points in the form of scalar point data `f`. Only works if the
cache is built for vector data. This is the adjoint
to [`regularize!`](@ref)
"""
@inline interpolate!(f::ScalarData,s::Nodes{Dual},cache::BasicILMCache) = interpolate!(f,s,cache.Ecurl)


"""
    interpolate!(vb::VectorData,v::Edges{Primal},cache::BasicILMCache)
    interpolate!(vb::VectorData,v::Edges{Primal},sys::ILMSystem)

The operation ``\\mathbf{v}_b = R_c^T \\mathbf{v}``, which interpolates vector grid data `v`
onto the surface points in the form of scalar point data `vb`. This is the adjoint
to [`regularize!`](@ref)
"""
@inline interpolate!(vb::VectorData,v::Edges{Primal},cache::BasicILMCache) = interpolate!(vb,v,cache.E)

function interpolate!(f::ScalarData,s::Nodes{C},Ec::InterpolationMatrix) where C
  fill!(f,0.0)
  mul!(f,Ec,s)
end

function interpolate!(vb::VectorData,v::Edges{C},Ef::InterpolationMatrix) where C
  fill!(vb,0.0)
  mul!(vb,Ef,v)
end

"""
    regularize_normal!(v::Edges{Primal},f::ScalarData,cache::BasicILMCache)
    regularize_normal!(v::Edges{Primal},f::ScalarData,sys::ILMSystem)

The operation ``\\mathbf{v} = R_f \\mathbf{n}\\circ f``, which maps scalar surface data `f` (like
a jump in scalar potential) to grid data `v` (like velocity). This is the adjoint
to [`normal_interpolate!`](@ref).
"""
@inline regularize_normal!(q::Edges{Primal},f::ScalarData,cache::BasicILMCache) = regularize_normal!(q,f,cache.nrm,cache.Rsn,cache.snorm_cache)

function regularize_normal!(q::Edges{Primal,NX,NY},f::ScalarData{N},nrm::VectorData{N},Rf::RegularizationMatrix,snorm_cache::VectorData{N}) where {NX,NY,N}
    product!(snorm_cache,nrm,f)
    mul!(q,Rf,snorm_cache)
end

"""
    regularize_normal!(qt::EdgeGradient{Primal},v::VectorData,cache::BasicILMCache)
    regularize_normal!(qt::EdgeGradient{Primal},v::VectorData,sys::ILMSystem)

The operation ``\\mathbf{q}_t = R_t \\mathbf{n}\\circ \\mathbf{v}``, which maps scalar vector data `v` (like
a jump in velocity) to grid data `qt` (like velocity-normal tensor). This is the adjoint
to [`normal_interpolate!`](@ref).
"""
@inline regularize_normal!(q::EdgeGradient{Primal},f::VectorData,cache::BasicILMCache) = regularize_normal!(q,f,cache.nrm,cache.Rsn,cache.snorm_cache)

"""
    regularize_normal_symm!(qt::EdgeGradient{Primal},v::VectorData,cache::BasicILMCache)
    regularize_normal_symm!(qt::EdgeGradient{Primal},v::VectorData,sys::ILMSystem)

The operation ``\\mathbf{q}_t = R_t (\\mathbf{n}\\circ \\mathbf{v}+\\mathbf{v}\\circ \\mathbf{n})``, which maps scalar vector data `v` (like
a jump in velocity) to grid data `qt` (like velocity-normal tensor). This is the adjoint
to [`normal_interpolate_symm!`](@ref).
"""
@inline regularize_normal_symm!(q::EdgeGradient{Primal},f::VectorData,cache::BasicILMCache) = regularize_normal_symm!(q,f,cache.nrm,cache.Rsn,cache.snorm_cache,cache.snorm2_cache)

function regularize_normal!(q::EdgeGradient{Primal,Dual,NX,NY},f::VectorData{N},nrm::VectorData{N},Rf::RegularizationMatrix,snorm_cache::TensorData{N}) where {NX,NY,N}
    pointwise_tensorproduct!(snorm_cache,nrm,f)
    mul!(q,Rf,snorm_cache)
end

function regularize_normal_symm!(q::EdgeGradient{Primal,Dual,NX,NY},f::VectorData{N},nrm::VectorData{N},Rf::RegularizationMatrix,snorm_cache::TensorData{N},snorm2_cache::TensorData{N}) where {NX,NY,N}
    pointwise_tensorproduct!(snorm_cache,nrm,f)
    transpose!(snorm2_cache,snorm_cache)
    snorm_cache .+= snorm2_cache
    # subtract the dot product of f and n from each diagonal entry
    snorm_cache.dudx .-= pointwise_dot(nrm,f)
    snorm_cache.dvdy .-= pointwise_dot(nrm,f)
    mul!(q,Rf,snorm_cache)
end



"""
    regularize_normal_cross!(v::Edges{Primal},f::ScalarData,cache::BasicILMCache)
    regularize_normal_cross!(v::Edges{Primal},f::ScalarData,sys::ILMSystem)

The operation ``\\mathbf{v} = R_f \\mathbf{n}\\times f \\mathbf{e}_z``, which maps scalar surface data `f` (like
a jump in streamfunction, endowed with the out-of-plane unit vector ``\\mathbf{e}_z``) to grid
data `v` (like velocity). This is the negative adjoint to [`normal_cross_interpolate!`](@ref).
"""
@inline regularize_normal_cross!(q::Edges{Primal},f::ScalarData,cache::BasicILMCache) =
           _regularize_normal_cross!(q,f,cache,Val(cache_datatype(cache)))

@inline _regularize_normal_cross!(q,f,cache,::Val{:scalar}) =
          regularize_normal_cross!(q,f,cache.nrm,cache.Rsn,cache.snorm_cache)

@inline _regularize_normal_cross!(q,f,cache,::Val{:vector}) =
          regularize_normal_cross!(q,f,cache.nrm,cache.R,cache.sdata_cache)

function regularize_normal_cross!(q::Edges{Primal,NX,NY},f::ScalarData{N},nrm::VectorData{N},Rf::RegularizationMatrix,snorm_cache::VectorData{N}) where {NX,NY,N}
    pointwise_cross!(snorm_cache,nrm,f)
    mul!(q,Rf,snorm_cache)
end

"""
    regularize_normal_cross!(w::Nodes{Dual},vs::VectorData,cache::BasicILMCache)
    regularize_normal_cross!(w::Nodes{Dual},vs::VectorData,sys::ILMSystem)

The operation ``\\omega = R_N \\mathbf{n}\\times \\mathbf{v}_s``, which maps surface vector data `vs` (like
a jump in velocity) to grid dual nodal data `w` (like vorticity). This is for use only
with vector-data caches. This is the negative adjoint to [`normal_cross_interpolate!`](@ref).
"""
@inline regularize_normal_cross!(w::Nodes{Dual},vs::VectorData,cache::BasicILMCache) =
           regularize_normal_cross!(w,vs,cache.nrm,cache.Rcurl,cache.sscalar_cache)

function regularize_normal_cross!(w::Nodes{Dual,NX,NY},vs::VectorData{N},nrm::VectorData{N},Rn::RegularizationMatrix,scache::ScalarData{N}) where {NX,NY,N}
    pointwise_cross!(scache,nrm,vs)
    mul!(w,Rn,scache)
end


"""
    regularize_normal_dot!(f::Nodes{Primal},vs::VectorData,cache::BasicILMCache)
    regularize_normal_dot!(f::Nodes{Primal},vs::VectorData,sys::ILMSystem)

The operation ``\\phi = R_c \\mathbf{n}\\cdot \\mathbf{v}_s``, which maps surface vector data `vs` (like
a jump in velocity) to grid nodal data `f` (like scalar potential). This is the negative adjoint to [`normal_dot_interpolate!`](@ref).
"""
@inline regularize_normal_dot!(f::Nodes{Primal},vs::VectorData,cache::BasicILMCache) =
           _regularize_normal_dot!(f,vs,cache,Val(cache_datatype(cache)))

@inline _regularize_normal_dot!(f,vs,cache,::Val{:scalar}) =
           regularize_normal_dot!(f,vs,cache.nrm,cache.R,cache.sdata_cache)

@inline _regularize_normal_dot!(f,vs,cache,::Val{:vector}) =
           regularize_normal_dot!(f,vs,cache.nrm,cache.Rdiv,cache.sscalar_cache)

function regularize_normal_dot!(f::Nodes{Primal,NX,NY},vs::VectorData{N},nrm::VectorData{N},Rn::RegularizationMatrix,scache::ScalarData{N}) where {NX,NY,N}
    pointwise_dot!(scache,nrm,vs)
    mul!(f,Rn,scache)
end

"""
    regularize_normal_dot!(v::Edges{Primal},taus::TensorData,cache::BasicILMCache)
    regularize_normal_dot!(v::Edges{Primal},taus::TensorData,sys::ILMSystem)

The operation ``\\mathbf{v} = R_f \\mathbf{n}\\cdot \\mathbf{\\tau}_s``, which maps surface tensor data `taus` (like
a stress) to grid edge data `v` (like velocity). This is the negative adjoint to [`normal_dot_interpolate!`](@ref).
"""
@inline regularize_normal_dot!(v::Edges{Primal},taus::TensorData,cache::BasicILMCache{N,SCA,GCT}) where {N,SCA,GCT<:Edges} =
           regularize_normal_dot!(v,taus,cache.nrm,cache.R,cache.sdata_cache)

function regularize_normal_dot!(v::Edges{Primal,NX,NY},taus::TensorData{N},nrm::VectorData{N},R::RegularizationMatrix,scache::VectorData{N}) where {NX,NY,N}
    pointwise_dot!(scache,nrm,taus)
    mul!(v,R,scache)
end




"""
    normal_interpolate!(vn::ScalarData,v::Edges{Primal},cache::BasicILMCache)
    normal_interpolate!(vn::ScalarData,v::Edges{Primal},sys::ILMSystem)

The operation ``v_n = \\mathbf{n} \\cdot R_f^T \\mathbf{v}``, which maps grid data `v` (like velocity) to scalar
surface data `vn` (like normal component of surface velocity). This is the
adjoint to [`regularize_normal!`](@ref).
"""
@inline normal_interpolate!(vn::ScalarData,q::Edges{Primal},cache::BasicILMCache) = normal_interpolate!(vn,q,cache.nrm,cache.Esn,cache.snorm_cache)

function normal_interpolate!(vn::ScalarData{N},q::Edges{Primal,NX,NY},nrm::VectorData{N},Ef::InterpolationMatrix,snorm_cache::VectorData{N}) where {NX,NY,N}
    mul!(snorm_cache,Ef,q)
    pointwise_dot!(vn,nrm,snorm_cache)
end

"""
    normal_interpolate!(τ::VectorData,A::EdgeGradient{Primal},cache::BasicILMCache)
    normal_interpolate!(τ::VectorData,A::EdgeGradient{Primal},sys::ILMSystem)

The operation ``\\mathbf{\\tau} = \\mathbf{n} \\cdot R_t^T \\mathbf{A}``, which maps grid tensor data `A` (like velocity gradient tensor) to vector
surface data `τ` (like traction). This is the adjoint to [`regularize_normal!`](@ref).
"""
@inline normal_interpolate!(vn::VectorData,q::EdgeGradient{Primal},cache::BasicILMCache) = normal_interpolate!(vn,q,cache.nrm,cache.Esn,cache.snorm_cache)


"""
    normal_interpolate_symm!(τ::VectorData,A::EdgeGradient{Primal},cache::BasicILMCache)
    normal_interpolate_symm!(τ::VectorData,A::EdgeGradient{Primal},sys::ILMSystem)

The operation ``\\mathbf{\\tau} = \\mathbf{n} \\cdot R_t^T (\\mathbf{A} +\\mathbf{A}^T - (\\mathrm{tr}\\mathbf{A})\\mathbf{I})``, which maps grid tensor data `A` (like velocity gradient tensor) to vector
surface data `τ` (like traction). This is the adjoint to [`regularize_normal_symm!`](@ref).
"""
@inline normal_interpolate_symm!(vn::VectorData,q::EdgeGradient{Primal},cache::BasicILMCache) = normal_interpolate_symm!(vn,q,cache.nrm,cache.Esn,cache.gsnorm2_cache,cache.snorm_cache)

function normal_interpolate!(vn::VectorData{N},q::EdgeGradient{Primal,Dual,NX,NY},nrm::VectorData{N},Ef::InterpolationMatrix,snorm_cache::TensorData{N}) where {NX,NY,N}
    mul!(snorm_cache,Ef,q)
    pointwise_dot!(vn,nrm,snorm_cache)
end

function normal_interpolate_symm!(vn::VectorData{N},q::EdgeGradient{Primal,Dual,NX,NY},nrm::VectorData{N},Ef::InterpolationMatrix,gsnorm_cache::EdgeGradient{Primal,Dual,NX,NY},snorm_cache::TensorData{N}) where {NX,NY,N}
    transpose!(gsnorm_cache,q)
    gsnorm_cache .+= q
    # subtract the trace of q from each diagonal element
    gsnorm_cache.dudx .-= q.dudx + q.dvdy
    gsnorm_cache.dvdy .-= q.dudx + q.dvdy
    mul!(snorm_cache,Ef,gsnorm_cache)
    pointwise_dot!(vn,nrm,snorm_cache)
end

"""
    normal_cross_interpolate!(wn::ScalarData,v::Edges{Primal},cache::BasicILMCache)
    normal_cross_interpolate!(wn::ScalarData,v::Edges{Primal},sys::ILMSystem)

The operation ``w_n = e_z\\cdot (\\mathbf{n} \\times R_f^T \\mathbf{v})``, which maps grid data `v` (like velocity) to scalar
surface data `wn` (like vorticity in the surface). This is the
negative adjoint to [`regularize_normal_cross!`](@ref).
"""
@inline normal_cross_interpolate!(vn::ScalarData,q::Edges{Primal},cache::BasicILMCache) =
          _normal_cross_interpolate!(vn,q,cache,Val(cache_datatype(cache)))

@inline _normal_cross_interpolate!(vn,q,cache,::Val{:scalar}) =
          normal_cross_interpolate!(vn,q,cache.nrm,cache.Esn,cache.snorm_cache)

@inline _normal_cross_interpolate!(vn,q,cache,::Val{:vector}) =
          normal_cross_interpolate!(vn,q,cache.nrm,cache.E,cache.sdata_cache)


function normal_cross_interpolate!(vn::ScalarData{N},q::Edges{Primal,NX,NY},nrm::VectorData{N},Ef::InterpolationMatrix,scache::VectorData{N}) where {NX,NY,N}
    mul!(scache,Ef,q)
    pointwise_cross!(vn,nrm,scache)
end


"""
    normal_cross_interpolate!(vs::VectorData,s::Nodes{Dual},cache::BasicILMCache)
    normal_cross_interpolate!(vs::VectorData,s::Nodes{Dual},sys::ILMSystem)

The operation ``\\mathbf{v}_s = \\mathbf{n}\\times R_N^T \\psi\\mathbf{e}_z)``, which maps grid data `s` (like streamfunction) to vector
surface data `vs` (like velocity in the surface). It only works for a vector-type cache. This is the
negative adjoint to [`regularize_normal_cross!`](@ref).
"""
@inline normal_cross_interpolate!(vs::VectorData,s::Nodes{Dual},cache::BasicILMCache{N,SCA,GCT}) where {N,SCA,GCT<:Edges} =
          normal_cross_interpolate!(vs,s,cache.nrm,cache.Ecurl,cache.sscalar_cache)

function normal_cross_interpolate!(vs::VectorData{N},s::Nodes{Dual,NX,NY},nrm::VectorData{N},E::InterpolationMatrix,scache::ScalarData{N}) where {NX,NY,N}
    mul!(scache,E,s)
    pointwise_cross!(vs,nrm,scache)
end

"""
    normal_dot_interpolate!(vs::VectorData,f::Nodes{Primal},cache::BasicILMCache)
    normal_dot_interpolate!(vs::VectorData,f::Nodes{Primal},sys::ILMSystem)

The operation ``\\mathbf{v}_s = \\mathbf{n} R_c^T \\phi``, which maps grid nodal data `f` (like scalar potential)
to surface vector data `vs` (like velocity). This is the negative adjoint to [`regularize_normal_dot!`](@ref).
"""
@inline normal_dot_interpolate!(vs::VectorData,f::Nodes{Primal},cache::BasicILMCache) =
           _normal_dot_interpolate!(vs,f,cache,Val(cache_datatype(cache)))

@inline _normal_dot_interpolate!(vs,f,cache,::Val{:scalar}) =
           normal_dot_interpolate!(vs,f,cache.nrm,cache.E,cache.sdata_cache)

@inline _normal_dot_interpolate!(vs,f,cache,::Val{:vector}) =
           normal_dot_interpolate!(vs,f,cache.nrm,cache.Ediv,cache.sscalar_cache)

function normal_dot_interpolate!(vs::VectorData{N},f::Nodes{Primal,NX,NY},nrm::VectorData{N},En::InterpolationMatrix,scache::ScalarData{N}) where {NX,NY,N}
    mul!(scache,En,f)
    product!(vs,nrm,scache)
end

"""
    normal_dot_interpolate!(taus::TensorData,v::Edges{Primal},cache::BasicILMCache)
    normal_dot_interpolate!(taus::TensorData,v::Edges{Primal},sys::ILMSystem)

The operation ``\\mathbf{\\tau}_s = \\mathbf{n}\\circ R_f^T\\mathbf{v}``, which maps grid edge data `v` (like velocity)
to surface tensor data `taus` (like stress). This is the negative adjoint to [`regularize_normal_dot!`](@ref).
"""
@inline normal_dot_interpolate!(taus::TensorData,v::Edges{Primal},cache::BasicILMCache{N,SCA,GCT}) where {N,SCA,GCT<:Edges} =
           normal_dot_interpolate!(taus,v,cache.nrm,cache.E,cache.sdata_cache)

function normal_dot_interpolate!(taus::TensorData{N},v::Edges{Primal,NX,NY},nrm::VectorData{N},E::InterpolationMatrix,scache::VectorData{N}) where {NX,NY,N}
    mul!(scache,E,v)
    pointwise_tensorproduct!(taus,nrm,scache)
end


"""
    surface_curl!(w::Nodes{Dual},f::ScalarData,cache::BasicILMCache)
    surface_curl!(w::Nodes{Dual},f::ScalarData,sys::ILMSystem)

The operation ``w = C_s^T f = C^T R_f \\mathbf{n}\\circ f``, which maps scalar surface data `f` (like
a jump in scalar potential) to grid data `w` (like vorticity). This is the adjoint
to ``C_s``, also given by `surface_curl!` (but with arguments switched). Note that
the differential operations are divided either by 1 or by the grid cell size,
depending on whether `sys` has been designated with `IndexScaling` or `GridScaling`,
respectively.
"""
function surface_curl!(w::Nodes{Dual},f::ScalarData,cache::BasicILMCache)
  _unscaled_surface_curl!(w,f,cache,Val(cache_datatype(cache)))
  _scale_derivative!(w,cache)
end

@inline _unscaled_surface_curl!(w::Nodes{Dual,NX,NY},f::ScalarData{N},cache::BasicILMCache,::Val{:scalar}) where {NX,NY,N} =
          _unscaled_surface_curl!(w,f,cache.nrm,cache.Rsn,cache.gsnorm_cache,cache.snorm_cache)

@inline _unscaled_surface_curl!(w::Nodes{Dual,NX,NY},f::ScalarData{N},cache::BasicILMCache,::Val{:vector}) where {NX,NY,N} =
          _unscaled_surface_curl!(w,f,cache.nrm,cache.R,cache.gdata_cache,cache.sdata_cache)



function _unscaled_surface_curl!(w::Nodes{Dual,NX,NY},f::ScalarData{N},nrm::VectorData{N},Rf::RegularizationMatrix,q_cache::Edges{Primal,NX,NY},snorm_cache::VectorData{N}) where {NX,NY,N}
    fill!(w,0.0)
    fill!(q_cache,0.0)
    regularize_normal!(q_cache,f,nrm,Rf,snorm_cache)
    curl!(w,q_cache)
end

"""
    surface_curl!(w::Nodes{Dual},v::VectorData,cache::BasicILMCache)
    surface_curl!(w::Nodes{Dual},v::VectorData,sys::ILMSystem)

The operation ``w = C_s^T \\mathbf{v} = C^T R_f \\mathbf{v}``, which maps vector surface data `v` (like
velocity) to grid data `w` (like vorticity). This is the adjoint
to ``C_s``, also given by `surface_curl!` (but with arguments switched). Note that
the differential operations are divided either by 1 or by the grid cell size,
depending on whether `sys` has been designated with `IndexScaling` or `GridScaling`,
respectively.
"""
function surface_curl!(w::Nodes{Dual},v::VectorData,cache::BasicILMCache)
  _unscaled_surface_curl!(w,v,cache.R,cache.gdata_cache)
  _scale_derivative!(w,cache)
end

function _unscaled_surface_curl!(w::Nodes{Dual,NX,NY},v::VectorData{N},Rf::RegularizationMatrix,q_cache::Edges{Primal,NX,NY}) where {NX,NY,N}
    fill!(w,0.0)
    fill!(q_cache,0.0)
    regularize!(q_cache,v,Rf)
    curl!(w,q_cache)
end

"""
    surface_curl!(vn::ScalarData,s::Nodes{Dual},cache::BasicILMCache)
    surface_curl!(vn::ScalarData,s::Nodes{Dual},sys::ILMSystem)

The operation ``v_n = C_s s = \\mathbf{n} \\cdot R_f^T C s``, which maps grid data `s` (like
streamfunction) to scalar surface data `vn` (like normal component of velocity).
This is the adjoint to ``C_s^T``, also given by `surface_curl!`, but with
arguments switched.  Note that
the differential operations are divided either by 1 or by the grid cell size,
depending on whether `sys` has been designated with `IndexScaling` or `GridScaling`,
respectively.
"""
function surface_curl!(vn::ScalarData,s::Nodes{Dual},cache::BasicILMCache)
  _unscaled_surface_curl!(vn,s,cache,Val(cache_datatype(cache)))
  _scale_derivative!(vn,cache)
end

@inline _unscaled_surface_curl!(vn::ScalarData{N},s::Nodes{Dual,NX,NY},cache::BasicILMCache,::Val{:scalar}) where {NX,NY,N} =
          _unscaled_surface_curl!(vn,s,cache.nrm,cache.Esn,cache.gsnorm_cache,cache.snorm_cache)

@inline _unscaled_surface_curl!(vn::ScalarData{N},s::Nodes{Dual,NX,NY},cache::BasicILMCache,::Val{:vector}) where {NX,NY,N} =
          _unscaled_surface_curl!(vn,s,cache.nrm,cache.E,cache.gdata_cache,cache.sdata_cache)


function _unscaled_surface_curl!(vn::ScalarData{N},s::Nodes{Dual,NX,NY},nrm::VectorData{N},Ef::InterpolationMatrix,q_cache::Edges{Primal,NX,NY},snorm_cache::VectorData{N}) where {NX,NY,N}
    fill!(vn,0.0)
    fill!(q_cache,0.0)
    curl!(q_cache,s)
    normal_interpolate!(vn,q_cache,nrm,Ef,snorm_cache)
end

"""
    surface_curl!(v::VectorData,s::Nodes{Dual},cache::BasicILMCache)
    surface_curl!(v::VectorData,s::Nodes{Dual},sys::ILMSystem)

The operation ``\\mathbf{v} = C_s s = R_f^T C s``, which maps grid data `s` (like
streamfunction) to vector surface data `v` (like velocity).
This is the adjoint to ``C_s^T``, also given by `surface_curl!`, but with
arguments switched.  Note that
the differential operations are divided either by 1 or by the grid cell size,
depending on whether `sys` has been designated with `IndexScaling` or `GridScaling`,
respectively.
"""
function surface_curl!(v::VectorData,s::Nodes{Dual},cache::BasicILMCache)
  _unscaled_surface_curl!(v,s,cache.E,cache.gdata_cache)
  _scale_derivative!(v,cache)
end

function _unscaled_surface_curl!(v::VectorData{N},s::Nodes{Dual,NX,NY},Ef::InterpolationMatrix,q_cache::Edges{Primal,NX,NY}) where {NX,NY,N}
    fill!(q_cache,0.0)
    curl!(q_cache,s)
    interpolate!(v,q_cache,Ef)
end



"""
    surface_curl_cross!(w::Nodes{Dual},f::ScalarData,cache::BasicILMCache)
    surface_curl_cross!(w::Nodes{Dual},f::ScalarData,sys::ILMSystem)

The operation ``w = \\hat{C}_s^T f = C^T R_f \\mathbf{n}\\times f \\mathbf{e}_z``, which maps scalar surface data `f` (like
a jump in streamfunction, multiplied by out-of-plane unit vector ``\\mathbf{e}_z``) to grid data `w` (like vorticity).
This is the adjoint to ``\\hat{C}_s``, also given by `surface_curl_cross!` (but with arguments switched). Note that
the differential operations are divided either by 1 or by the grid cell size,
depending on whether `sys` has been designated with `IndexScaling` or `GridScaling`,
respectively.
"""
function surface_curl_cross!(w::Nodes{Dual},f::ScalarData,cache::BasicILMCache)
  _unscaled_surface_curl_cross!(w,f,cache,Val(cache_datatype(cache)))
  _scale_derivative!(w,cache)
end

@inline _unscaled_surface_curl_cross!(w::Nodes{Dual,NX,NY},f::ScalarData{N},cache::BasicILMCache,::Val{:scalar}) where {NX,NY,N} =
          _unscaled_surface_curl_cross!(w,f,cache.nrm,cache.Rsn,cache.gsnorm_cache,cache.snorm_cache)

@inline _unscaled_surface_curl_cross!(w::Nodes{Dual,NX,NY},f::ScalarData{N},cache::BasicILMCache,::Val{:vector}) where {NX,NY,N} =
          _unscaled_surface_curl_cross!(w,f,cache.nrm,cache.R,cache.gdata_cache,cache.sdata_cache)


function _unscaled_surface_curl_cross!(w::Nodes{Dual,NX,NY},f::ScalarData{N},nrm::VectorData{N},Rf::RegularizationMatrix,q_cache::Edges{Primal,NX,NY},snorm_cache::VectorData{N}) where {NX,NY,N}
    fill!(w,0.0)
    fill!(q_cache,0.0)
    regularize_normal_cross!(q_cache,f,nrm,Rf,snorm_cache)
    curl!(w,q_cache)
end

"""
    surface_curl_cross!(γ::ScalarData,s::Nodes{Dual},cache::BasicILMCache)
    surface_curl_cross!(γ::ScalarData,s::Nodes{Dual},sys::ILMSystem)

The operation ``\\gamma = \\hat{C}_s s = \\mathbf{e}_z\\cdot (\\mathbf{n} \\times R_f^T C s)``, which maps grid data `s` (like
streamfunction) to scalar surface data `γ` (like vorticity in the surface).
This is the adjoint to ``\\hat{C}_s^T``, also given by `surface_curl_cross!`, but with
arguments switched.  Note that
the differential operations are divided either by 1 or by the grid cell size,
depending on whether `sys` has been designated with `IndexScaling` or `GridScaling`,
respectively.
"""
function surface_curl_cross!(vn::ScalarData,s::Nodes{Dual},cache::BasicILMCache)
  _unscaled_surface_curl_cross!(vn,s,cache,Val(cache_datatype(cache)))
  _scale_derivative!(vn,cache)
end

@inline _unscaled_surface_curl_cross!(vn::ScalarData{N},s::Nodes{Dual,NX,NY},cache::BasicILMCache,::Val{:scalar}) where {NX,NY,N} =
          _unscaled_surface_curl_cross!(vn,s,cache.nrm,cache.Esn,cache.gsnorm_cache,cache.snorm_cache)

@inline _unscaled_surface_curl_cross!(vn::ScalarData{N},s::Nodes{Dual,NX,NY},cache::BasicILMCache,::Val{:vector}) where {NX,NY,N} =
          _unscaled_surface_curl_cross!(vn,s,cache.nrm,cache.E,cache.gdata_cache,cache.sdata_cache)


function _unscaled_surface_curl_cross!(vn::ScalarData{N},s::Nodes{Dual,NX,NY},nrm::VectorData{N},Ef::InterpolationMatrix,q_cache::Edges{Primal,NX,NY},snorm_cache::VectorData{N}) where {NX,NY,N}
    fill!(vn,0.0)
    fill!(q_cache,0.0)
    curl!(q_cache,s)
    normal_cross_interpolate!(vn,q_cache,nrm,Ef,snorm_cache)
end


"""
    surface_divergence!(Θ::Nodes{Primal},f::ScalarData,cache::BasicILMCache)
    surface_divergence!(Θ::Nodes{Primal},f::ScalarData,sys::ILMSystem)

The operation ``\\theta = D_s f = D R_f \\mathbf{n} \\circ f``, which maps surface scalar data `f` (like
jump in scalar potential) to grid data `Θ` (like dilatation, i.e. divergence of velocity).
This is the negative adjoint of [`surface_grad!`](@ref).
Note that
the differential operations are divided either by 1 or by the grid cell size,
depending on whether `sys` has been designated with `IndexScaling` or `GridScaling`,
respectively.
"""
function surface_divergence!(θ::Nodes{Primal},f::ScalarData,cache::BasicILMCache)
  _unscaled_surface_divergence!(θ,f,cache,Val(cache_datatype(cache)))
  _scale_derivative!(θ,cache)
end

@inline _unscaled_surface_divergence!(Θ::Nodes{Primal,NX,NY},f::ScalarData{N},cache,::Val{:scalar}) where {NX,NY,N} =
          _unscaled_surface_divergence!(Θ,f,cache.nrm,cache.Rsn,cache.gsnorm_cache,cache.snorm_cache)

@inline _unscaled_surface_divergence!(Θ::Nodes{Primal,NX,NY},f::ScalarData{N},cache,::Val{:vector}) where {NX,NY,N} =
          _unscaled_surface_divergence!(Θ,f,cache.nrm,cache.R,cache.gdata_cache,cache.sdata_cache)


function _unscaled_surface_divergence!(θ::Nodes{Primal,NX,NY},f::ScalarData{N},nrm::VectorData{N},Rf::RegularizationMatrix,q_cache::Edges{Primal,NX,NY},snorm_cache::VectorData{N}) where {NX,NY,N}
    fill!(θ,0.0)
    fill!(q_cache,0.0)
    regularize_normal!(q_cache,f,nrm,Rf,snorm_cache)
    divergence!(θ,q_cache)
end

"""
    surface_divergence!(v::Edges{Primal},dv::VectorData,cache::BasicILMCache)
    surface_divergence!(v::Edges{Primal},dv::VectorData,sys::ILMSystem)

The operation ``\\mathbf{v} = D_s d\\mathbf{v} = D R_t (\\mathbf{n} \\circ d\\mathbf{v})``, which maps surface vector data `dv` (like
jump in velocity) to grid data `v` (like velocity). This is the negative adjoint of [`surface_grad!`](@ref). Note that
the differential operations are divided either by 1 or by the grid cell size,
depending on whether `sys` has been designated with `IndexScaling` or `GridScaling`,
respectively.
"""
function surface_divergence!(θ::Edges{Primal},f::VectorData,cache::BasicILMCache)
  _unscaled_surface_divergence!(θ,f,cache.nrm,cache.Rsn,cache.gsnorm_cache,cache.snorm_cache)
  _scale_derivative!(θ,cache)
end

"""
    surface_divergence_symm!(v::Edges{Primal},dv::VectorData,cache::BasicILMCache)
    surface_divergence_symm!(v::Edges{Primal},dv::VectorData,sys::ILMSystem)

The operation ``\\mathbf{v} = D_s d\\mathbf{v} = D R_t (\\mathbf{n} \\circ d\\mathbf{v} + d\\mathbf{v} \\circ \\mathbf{n})``, which maps surface vector data `dv` (like
jump in velocity) to grid data `v` (like velocity). This is the negative adjoint of [`surface_grad_symm!`](@ref). Note that
the differential operations are divided either by 1 or by the grid cell size,
depending on whether `sys` has been designated with `IndexScaling` or `GridScaling`,
respectively.
"""
function surface_divergence_symm!(θ::Edges{Primal},f::VectorData,cache::BasicILMCache)
  _unscaled_surface_divergence_symm!(θ,f,cache.nrm,cache.Rsn,cache.gsnorm_cache,cache.snorm_cache,cache.snorm2_cache)
  _scale_derivative!(θ,cache)
end

function _unscaled_surface_divergence!(θ::Edges{Primal,NX,NY},f::VectorData{N},nrm::VectorData{N},Rf::RegularizationMatrix,q_cache::EdgeGradient{Primal,Dual,NX,NY},snorm_cache::TensorData{N}) where {NX,NY,N}
    fill!(θ,0.0)
    fill!(q_cache,0.0)
    regularize_normal!(q_cache,f,nrm,Rf,snorm_cache)
    divergence!(θ,q_cache)
end

function _unscaled_surface_divergence_symm!(θ::Edges{Primal,NX,NY},f::VectorData{N},nrm::VectorData{N},Rf::RegularizationMatrix,q_cache::EdgeGradient{Primal,Dual,NX,NY},snorm_cache::TensorData{N},snorm2_cache::TensorData{N}) where {NX,NY,N}
    fill!(θ,0.0)
    fill!(q_cache,0.0)
    regularize_normal_symm!(q_cache,f,nrm,Rf,snorm_cache,snorm2_cache)
    divergence!(θ,q_cache)
end

"""
    surface_grad!(vn::ScalarData,ϕ::Nodes{Primal},cache::BasicILMCache)
    surface_grad!(vn::ScalarData,ϕ::Nodes{Primal},sys::ILMSystem)

The operation ``v_n = G_s\\phi = \\mathbf{n} \\cdot R_f^T G\\phi``, which maps grid data `ϕ` (like
scalar potential) to scalar surface data `vn` (like normal component of velocity). This is the negative adjoint of [`surface_divergence!`](@ref).
Note that
the differential operations are divided either by 1 or by the grid cell size,
depending on whether `sys` has been designated with `IndexScaling` or `GridScaling`,
respectively.
"""
function surface_grad!(vn::ScalarData,ϕ::Nodes{Primal},cache::BasicILMCache)
  _unscaled_surface_grad!(vn,ϕ,cache,Val(cache_datatype(cache)))
  _scale_derivative!(vn,cache)
end

@inline _unscaled_surface_grad!(vn::ScalarData{N},ϕ::Nodes{Primal,NX,NY},cache,::Val{:scalar}) where {NX,NY,N} =
          _unscaled_surface_grad!(vn,ϕ,cache.nrm,cache.Esn,cache.gsnorm_cache,cache.snorm_cache)

@inline _unscaled_surface_grad!(vn::ScalarData{N},ϕ::Nodes{Primal,NX,NY},cache,::Val{:vector}) where {NX,NY,N} =
          _unscaled_surface_grad!(vn,ϕ,cache.nrm,cache.E,cache.gdata_cache,cache.sdata_cache)

function _unscaled_surface_grad!(vn::ScalarData{N},ϕ::Nodes{Primal,NX,NY},nrm::VectorData{N},Ef::InterpolationMatrix,q_cache::Edges{Primal,NX,NY},snorm_cache::VectorData{N}) where {NX,NY,N}
    fill!(vn,0.0)
    fill!(q_cache,0.0)
    grad!(q_cache,ϕ)
    normal_interpolate!(vn,q_cache,nrm,Ef,snorm_cache)
end

"""
    surface_grad!(τ::VectorData,v::Edges{Primal},cache::BasicILMCache)
    surface_grad!(τ::VectorData,v::Edges{Primal},sys::ILMSystem)

The operation ``\\mathbf{\\tau} = G_s v = \\mathbf{n} \\cdot R_t^T G \\mathbf{v}``, which maps grid vector data `v` (like
velocity) to vector surface data `τ` (like traction). This is the negative adjoint of [`surface_divergence!`](@ref). Note that
the differential operations are divided either by 1 or by the grid cell size,
depending on whether `sys` has been designated with `IndexScaling` or `GridScaling`,
respectively.
"""
function surface_grad!(vn::VectorData,ϕ::Edges{Primal},cache::BasicILMCache)
  _unscaled_surface_grad!(vn,ϕ,cache.nrm,cache.Esn,cache.gsnorm_cache,cache.snorm_cache)
  _scale_derivative!(vn,cache)
end

"""
    surface_grad_symm!(τ::VectorData,v::Edges{Primal},cache::BasicILMCache)
    surface_grad_symm!(τ::VectorData,v::Edges{Primal},sys::ILMSystem)

The operation ``\\mathbf{\\tau} = G_s v = \\mathbf{n} \\cdot R_t^T (G \\mathbf{v} + (G \\mathbf{v})^T)``, which maps grid vector data `v` (like
velocity) to vector surface data `τ` (like traction). This is the negative adjoint of [`surface_divergence_symm!`](@ref). Note that
the differential operations are divided either by 1 or by the grid cell size,
depending on whether `sys` has been designated with `IndexScaling` or `GridScaling`,
respectively.
"""
function surface_grad_symm!(vn::VectorData,ϕ::Edges{Primal},cache::BasicILMCache)
  _unscaled_surface_grad_symm!(vn,ϕ,cache.nrm,cache.Esn,cache.gsnorm_cache,cache.gsnorm2_cache,cache.snorm_cache)
  _scale_derivative!(vn,cache)
end

function _unscaled_surface_grad!(vn::VectorData{N},ϕ::Edges{Primal,NX,NY},nrm::VectorData{N},Ef::InterpolationMatrix,gsnorm_cache::EdgeGradient{Primal,Dual,NX,NY},snorm_cache::TensorData{N}) where {NX,NY,N}
    fill!(vn,0.0)
    fill!(gsnorm_cache,0.0)
    grad!(gsnorm_cache,ϕ)
    normal_interpolate!(vn,gsnorm_cache,nrm,Ef,snorm_cache)
end

function _unscaled_surface_grad_symm!(vn::VectorData{N},ϕ::Edges{Primal,NX,NY},nrm::VectorData{N},Ef::InterpolationMatrix,gsnorm_cache::EdgeGradient{Primal,Dual,NX,NY},gsnorm2_cache::EdgeGradient{Primal,Dual,NX,NY},snorm_cache::TensorData{N}) where {NX,NY,N}
    fill!(vn,0.0)
    fill!(gsnorm_cache,0.0)
    grad!(gsnorm_cache,ϕ)
    normal_interpolate_symm!(vn,gsnorm_cache,nrm,Ef,gsnorm2_cache,snorm_cache)
end



"""
    surface_divergence_cross!(Θ::Nodes{Primal},f::ScalarData,cache::BasicILMCache)
    surface_divergence_cross!(Θ::Nodes{Primal},f::ScalarData,sys::ILMSystem)

The operation ``\\theta = \\hat{D}_s f = D R_f \\mathbf{n} \\times f \\mathbf{e}_z``, which maps surface scalar data `f` (like
jump in streamfunction) to grid data `Θ` (like dilatation, i.e. divergence of velocity).
This is the adjoint of [`surface_grad_cross!`](@ref).
Note that the differential operations are divided either by 1 or by the grid cell size,
depending on whether `sys` has been designated with `IndexScaling` or `GridScaling`,
respectively.
"""
function surface_divergence_cross!(θ::Nodes{Primal},f::ScalarData,cache::BasicILMCache)
  _unscaled_surface_divergence_cross!(Θ,f,cache,Val(cache_datatype(cache)))
  _scale_derivative!(θ,cache)
end

@inline _unscaled_surface_divergence_cross!(Θ,f,cache,::Val{:scalar}) =
          _unscaled_surface_divergence_cross!(Θ,f,cache.nrm,cache.Rsn,cache.gsnorm_cache,cache.snorm_cache)

@inline _unscaled_surface_divergence_cross!(Θ,f,cache,::Val{:vector}) =
          _unscaled_surface_divergence_cross!(Θ,f,cache.nrm,cache.R,cache.gdata_cache,cache.sdata_cache)

function _unscaled_surface_divergence_cross!(θ::Nodes{Primal,NX,NY},f::ScalarData{N},nrm::VectorData{N},Rf::RegularizationMatrix,q_cache::Edges{Primal,NX,NY},snorm_cache::VectorData{N}) where {NX,NY,N}
    fill!(θ,0.0)
    fill!(q_cache,0.0)
    regularize_normal_cross!(q_cache,f,nrm,Rf,snorm_cache)
    divergence!(θ,q_cache)
end

"""
    surface_grad_cross!(γ::ScalarData,ϕ::Nodes{Primal},cache::BasicILMCache)
    surface_grad_cross!(γ::ScalarData,ϕ::Nodes{Primal},sys::ILMSystem)

The operation ``\\gamma = \\hat{G}_s\\phi = \\mathbf{e}_z \\cdot (\\mathbf{n} \\times R_f^T G\\phi)``, which maps grid data `ϕ` (like
scalar potential) to scalar surface data `γ` (like surface vorticity). This is the adjoint of [`surface_divergence_cross!`](@ref).
Note that
the differential operations are divided either by 1 or by the grid cell size,
depending on whether `sys` has been designated with `IndexScaling` or `GridScaling`,
respectively.
"""
function surface_grad_cross!(vn::ScalarData,ϕ::Nodes{Primal},cache::BasicILMCache)
  _unscaled_surface_grad_cross!(vn,ϕ,cache,Val(cache_datatype(cache)))
  _scale_derivative!(vn,cache)
end

@inline _unscaled_surface_grad_cross!(vn,ϕ,cache,::Val{:scalar}) =
          _unscaled_surface_grad_cross!(vn,ϕ,cache.nrm,cache.Esn,cache.gsnorm_cache,cache.snorm_cache)

@inline _unscaled_surface_grad_cross!(vn,ϕ,cache,::Val{:vector}) =
          _unscaled_surface_grad_cross!(vn,ϕ,cache.nrm,cache.E,cache.gdata_cache,cache.sdata_cache)


function _unscaled_surface_grad_cross!(vn::ScalarData{N},ϕ::Nodes{Primal,NX,NY},nrm::VectorData{N},Ef::InterpolationMatrix,q_cache::Edges{Primal,NX,NY},snorm_cache::VectorData{N}) where {NX,NY,N}
    fill!(vn,0.0)
    fill!(q_cache,0.0)
    grad!(q_cache,ϕ)
    normal_cross_interpolate!(vn,q_cache,nrm,Ef,snorm_cache)
end


"""
    mask(cache::BasicILMCache) -> GridData
    mask(sys::ILMSystem) -> GridData

Create grid data that consist of 1s inside of a surface (i.e., on a side opposite
  the normal vectors) and 0s outside. The grid data are the same type as the
  output data type of `sys`.  Only allows `sys` to have `GridScaling`.
  If the cache has no immersed surface, then it reverts to 1s everywhere.
"""
@inline mask(cache::BasicILMCache{N,GridScaling}) where {N} = _get_mask!(zeros_grid(cache),cache)


"""
    complementary_mask(cache::BasicILMCache) -> GridData
    complementary_mask(sys::ILMSystem) -> GridData

Create grid data that consist of 0s inside of a surface (i.e., on a side opposite
  the normal vectors) and 1s outside. The grid data are the same type as the
  output data type of `cache`.  Only allows `cache` to have `GridScaling`.
  If the cache has no immersed surface, then it reverts to 0s everywhere.
"""
@inline complementary_mask(cache::BasicILMCache{N,GridScaling}) where {N} =
          _get_complementary_mask!(zeros_grid(cache),cache)


"""
    mask(w::GridData,surface::Body,grid::PhysicalGrid)

Create grid data that consist of 1s inside of surface `surface` (i.e., on a side opposite
the normal vectors) and 0s outside. The grid data are the same type as
`w`.
"""
function mask(w::T,surface::Body,grid::PhysicalGrid)  where {T<:GridData}
    mcache = _mask_cache(w,surface,grid)
    return mask(mcache)
end

"""
    complementary_mask(w::GridData,surface::Body,grid::PhysicalGrid)

Create grid data that consist of 0s inside of surface `surface` (i.e., on a side opposite
the normal vectors) and 1s outside. The grid data are the same type as
`w`.
"""
function complementary_mask(w::T,surface::Body,grid::PhysicalGrid)  where {T<:GridData}
    mcache = _mask_cache(w,surface,grid)
    return complementary_mask(mcache)
end


"""
    mask!(w::GridData,cache::BasicILMCache)
    mask!(w::GridData,sys::ILMSystem)

Mask the data `w` in place by multiplying it by 1s inside of a surface (i.e., on a side opposite
  the normal vectors) and 0s outside. The grid data `w` must be of the same type as the
  output data type of `cache`. Only allows `cache` to have `GridScaling`.
"""
mask!(w::T,cache::BasicILMCache) where {T<:GridData} = _mask!(w,cache)


function _mask!(w,cache::BasicILMCache{N,GridScaling,GT}) where {N,GT<:ScalarGridData}
  @unpack gdata_cache = cache
  _get_mask!(gdata_cache,cache)
  _scalar_mask_product!(w,cache)
end
function _mask!(w,cache::BasicILMCache{N,GridScaling,GT}) where {N,GT<:VectorGridData}
  @unpack gdata_cache = cache
  _get_mask!(gdata_cache,cache)
  _vector_mask_product!(w,cache)
end


"""
    complementary_mask!(w::GridData,cache::BasicILMCache)
    complementary_mask!(w::GridData,sys::ILMSystem)

Mask the data `w` in place by multiplying it by 0s inside of a surface (i.e., on a side opposite
  the normal vectors) and 1s outside. The grid data `w` must be of the same type as the
  output data type of `cache`. Only allows `cache` to have `GridScaling`.
"""
complementary_mask!(w::T,cache::BasicILMCache) where {T<:GridData} = _complementary_mask!(w,cache)


function _complementary_mask!(w,cache::BasicILMCache{N,GridScaling,GT}) where {N,GT<:ScalarGridData}
  @unpack gdata_cache = cache
  _get_complementary_mask!(gdata_cache,cache)
  _scalar_mask_product!(w,cache)
end
function _complementary_mask!(w,cache::BasicILMCache{N,GridScaling,GT}) where {N,GT<:VectorGridData}
  @unpack gdata_cache = cache
  _get_complementary_mask!(gdata_cache,cache)
  _vector_mask_product!(w,cache)
end


"""
    mask!(w::GridData,surface::Body,grid::PhysicalGrid)

Mask the data `w` in place by multiplying it by 1s inside of surface `surface` (i.e., on a side opposite
the normal vectors) and 0s outside.
"""
function mask!(w::T,surface::Body,grid::PhysicalGrid)  where {T<:GridData}
    mcache = _mask_cache(w,surface,grid)
    return mask!(w,mcache)
end




"""
    complementary_mask!(w::GridData,surface::Body,grid::PhysicalGrid)

Mask the data `w` in place by multiplying it by 0s inside of surface `surface` (i.e., on a side opposite
the normal vectors) and 1s outside.
"""
function complementary_mask!(w::T,surface::Body,grid::PhysicalGrid)  where {T<:GridData}
    mcache = _mask_cache(w,surface,grid)
    return complementary_mask!(w,mcache)
end

_mask_cache(::ScalarGridData,surface,grid) = SurfaceScalarCache(surface,grid)
_mask_cache(::VectorGridData,surface,grid) = SurfaceVectorCache(surface,grid)
_mask_cache(::TensorGridData,surface,grid) = SurfaceVectorCache(surface,grid)


function _get_complementary_mask!(msk,cache::BasicILMCache{N}) where {N}
    _get_mask!(msk,cache)
    msk .= 1.0 - msk
end

function _get_complementary_mask!(msk,cache::BasicILMCache{0})
    fill!(msk,0.0)
end

function _get_mask!(msk,cache::BasicILMCache{N}) where {N}
  @unpack sdata_cache, gdata_cache, L = cache
  fill!(sdata_cache,1.0)
  surface_divergence!(msk,sdata_cache,cache)
  inverse_laplacian!(msk,cache)
  msk .*= -1.0
end

function _get_mask!(msk,cache::BasicILMCache{0})
  @unpack gdata_cache = cache
  fill!(msk,1.0)
end



function _scalar_mask_product!(w::Nodes{Primal},cache)
    @unpack gdata_cache = cache
    product!(w,gdata_cache,w)
end
function _scalar_mask_product!(w::Nodes{Dual},cache)
    @unpack gdata_cache, gcurl_cache = cache
    grid_interpolate!(gcurl_cache,gdata_cache)
    product!(w,gcurl_cache,w)
end
function _scalar_mask_product!(w::XEdges{Primal},cache)
    @unpack gdata_cache, gsnorm_cache  = cache
    grid_interpolate!(gsnorm_cache.u,gdata_cache)
    product!(w,gsnorm_cache.u,w)
end
function _scalar_mask_product!(w::YEdges{Primal},cache)
    @unpack gdata_cache, gsnorm_cache  = cache
    grid_interpolate!(gsnorm_cache.v,gdata_cache)
    product!(w,gsnorm_cache.v,w)
end
function _scalar_mask_product!(w::Edges{Primal},cache)
    _scalar_mask_product!(w.u,cache)
    _scalar_mask_product!(w.v,cache)
end

function _vector_mask_product!(w::Edges{Primal},cache)
    @unpack gdata_cache = cache
    product!(w,gdata_cache,w)
end
function _vector_mask_product!(w::Nodes{Dual},cache)
    @unpack gdata_cache, gcurl_cache = cache
    grid_interpolate!(gcurl_cache,gdata_cache.u)
    product!(w,gcurl_cache,w)
end
function _vector_mask_product!(w::Nodes{Primal},cache)
    @unpack gdata_cache, gdiv_cache = cache
    grid_interpolate!(gdiv_cache,gdata_cache.u)
    product!(w,gdiv_cache,w)
end
function _vector_mask_product!(w::EdgeGradient{Primal},cache)
    _vector_mask_product!(w.dudx,cache)
    _vector_mask_product!(w.dudy,cache)
    _vector_mask_product!(w.dvdx,cache)
    _vector_mask_product!(w.dvdy,cache)
end
