# Routines and types to support Helmholtz decomposition

struct ScalarPotentialCache{DVT,VNT,FT} <: AbstractExtraILMCache
   dv :: DVT
   dvn :: VNT
   ftemp :: FT
   divv_temp :: FT
end

struct VectorPotentialCache{DVT,VNT,ST} <: AbstractExtraILMCache
   dv :: DVT
   dvn :: VNT
   stemp :: ST
   curlv_temp :: ST
end

struct VectorFieldCache{ST,VST,FT,VFT,VPT,SPT} <: AbstractExtraILMCache
   ψtemp :: ST
   vψ :: VST
   ϕtemp :: FT
   vϕ :: VFT
   wcache :: VPT
   dcache :: SPT
end


"""
    ScalarPotentialCache(base_cache::BasicILMCache)

Create a cache for calculations involving the scalar potential field
that induces a vector field on the grid. The base cache must be
in support of vector field data.
"""
function ScalarPotentialCache(base_cache::AbstractBasicCache{N,GCT}) where {N,GCT<:Edges{Primal}}
    dv = zeros_surface(base_cache)
    ftemp = zeros_griddiv(base_cache)
    divv_temp = zeros_griddiv(base_cache)
    dvn = ScalarData(dv)
    ScalarPotentialCache(dv,dvn,ftemp,divv_temp)
end

"""
    VectorPotentialCache(base_cache::BasicILMCache)

Create a cache for calculations involving the vector potential field (i.e. streamfunction)
that induces a vector field on the grid. The base cache must be
in support of vector field data.
"""
function VectorPotentialCache(base_cache::BasicILMCache)
    dv = zeros_surface(base_cache)
    stemp = zeros_gridcurl(base_cache)
    curlv_temp = zeros_gridcurl(base_cache)
    dvn = ScalarData(dv)
    VectorPotentialCache(dv,dvn,stemp,curlv_temp)
end

"""
    VectorFieldCache(base_cache::BasicILMCache)

Create a cache for calculations involving the vector field
derived from the vector and scalar potentials.
The base cache must be in support of vector field data.
"""
function VectorFieldCache(base_cache::BasicILMCache)
    ψtemp = zeros_gridcurl(base_cache)
    vψ = zeros_grid(base_cache)
    ϕtemp = zeros_griddiv(base_cache)
    vϕ = zeros_grid(base_cache)
    wfield = VectorPotentialCache(base_cache)
    dfield = ScalarPotentialCache(base_cache)
    VectorFieldCache(ψtemp,vψ,ϕtemp,vϕ,wfield,dfield)
end


"""
    vectorpotential_from_masked_curlv!(ψ::Nodes{Dual},masked_curlv::Nodes{Dual},dv::VectorData,base_cache::BasicILMCache,wcache::VectorPotentialCache)

Return the vector potential field `ψ` from the masked curl of the vector field, `masked_curlv` (``\\overline{\\nabla\\times\\mathbf{v}}``),
the jump in the vector field across immersed surface `dv`. It solves

``L_n\\psi = -\\overline{\\nabla\\times\\mathbf{v}} - R_n\\mathbf{n}\\times[\\mathbf{v}]``

and returns ``\\psi``.
"""
function vectorpotential_from_masked_curlv!(ψ::Nodes{Dual},curlv::Nodes{Dual},dv::VectorData,base_cache::BasicILMCache,wcache::VectorPotentialCache)
    @unpack nrm = base_cache
    @unpack stemp = wcache

    fill!(stemp,0.0)
    _curlv_from_surface_jump!(stemp,dv,base_cache)
    stemp .+= curlv

    vectorpotential_from_curlv!(ψ,stemp,base_cache)

    return ψ

end

function _curlv_from_surface_jump!(curlv::Nodes{Dual},dv::VectorData{N},base_cache::BasicILMCache) where {N}
    regularize_normal_cross!(curlv,dv,base_cache)
end

function _curlv_from_surface_jump!(curlv::Nodes{Dual},dv::VectorData{0},base_cache::BasicILMCache)
    return curlv
end

"""
    vectorpotential_from_curlv!(ψ::Nodes{Dual},curlv::Nodes{Dual},base_cache::BasicILMCache)

Return the vector potential field `ψ` from the curl of the masked vector field, `curlv` (``\\nabla\\times\\overline{\\mathbf{v}}``),
the jump in the vector field across immersed surface `dv`. It solves

``L_n\\psi = -\\nabla\\times\\overline{\\mathbf{v}}``

and returns ``\\psi``.
"""
#=
@inline function vectorpotential_from_curlv!(ψ::Nodes{Dual},curlv::Nodes{Dual},base_cache::BasicILMCache,wcache::VectorPotentialCache)

    inverse_laplacian!(ψ,curlv,base_cache)
    ψ .*= -1.0
    #return ψ

end
=#

function vectorpotential_from_curlv!(ψ::Nodes{Dual},curlv::Nodes{Dual},base_cache::BasicILMCache)
    inverse_laplacian!(ψ,curlv,base_cache)
    ψ .*= -1.0
end

"""
    vecfield_from_vectorpotential!(v::Edges{Primal},ψ::Nodes{Dual},base_cache::BasicILMCache)

Return the vector field `v` associated with vector potential field `ψ` (in 2-d, a scalar
streamfunction), via the curl

``\\mathbf{v} = \\nabla\\times\\psi \\mathbf{e}_z``
"""
function vecfield_from_vectorpotential!(v::Edges{Primal},ψ::Nodes{Dual},base_cache::BasicILMCache)
    fill!(v,0.0)
    curl!(v,ψ,base_cache)
    return v
end

"""
    masked_curlv_from_curlv_masked!(masked_curlv::Nodes{Dual},curlv::Nodes{Dual},dv::VectorData,base_cache::BasicILMCache,wcache::VectorPotentialCache)

Return the masked curl of the vector field ``\\overline{\\nabla\\times\\mathbf{v}}`` (`masked_curlv`) from
the curl of the masked vector field ``\\nabla\\times\\overline{\\mathbf{v}}`` (`curlv`). It obtains
this from

``\\overline{\\nabla\\times\\mathbf{v}} = \\nabla\\times\\overline{\\mathbf{v}} - R_n\\mathbf{n}\\times[\\mathbf{v}]``
"""
function masked_curlv_from_curlv_masked!(masked_curlv::Nodes{Dual},curlv::Nodes{Dual},dv::VectorData{N},base_cache::BasicILMCache,wcache::VectorPotentialCache) where {N}
    @unpack nrm = base_cache

    fill!(masked_curlv,0.0)
    #pointwise_cross!(dvn,nrm,dv)
    #regularize!(masked_curlv,dvn,Rn)
    regularize_normal_cross!(masked_curlv,dv,base_cache)
    masked_curlv .*= -1.0
    masked_curlv .+= curlv
    return masked_curlv
end

function masked_curlv_from_curlv_masked!(masked_curlv::Nodes{Dual},curlv::Nodes{Dual},dv::VectorData{0},base_cache::BasicILMCache,wcache::VectorPotentialCache)
    masked_curlv .= curlv
end

"""
    curlv_masked_from_masked_curlv!(curlv::Nodes{Dual},masked_curlv::Nodes{Dual},dv::VectorData,base_cache::BasicILMCache,wcache::VectorPotentialCache)

Return the curl of the masked vector field ``\\nabla\\times\\overline{\\mathbf{v}}`` (`curlv`) from
the masked curl of the vector field ``\\overline{\\nabla\\times\\mathbf{v}}`` (`masked_curlv`). It obtains
this from

``\\nabla\\times\\overline{\\mathbf{v}} = \\overline{\\nabla\\times\\mathbf{v}} + R_n\\mathbf{n}\\times[\\mathbf{v}]``
"""
function curlv_masked_from_masked_curlv!(curlv::Nodes{Dual},masked_curlv::Nodes{Dual},dv::VectorData{N},base_cache::BasicILMCache,wcache::VectorPotentialCache) where {N}
    @unpack nrm = base_cache
    #@unpack dvn, Rn = wcache

    fill!(curlv,0.0)
    #pointwise_cross!(dvn,nrm,dv)
    #regularize!(curlv,dvn,Rn)
    regularize_normal_cross!(curlv,dv,base_cache)
    curlv .+= masked_curlv
    return curlv
end

function curlv_masked_from_masked_curlv!(curlv::Nodes{Dual},masked_curlv::Nodes{Dual},dv::VectorData{0},base_cache::BasicILMCache,wcache::VectorPotentialCache)
    curlv .= masked_curlv
end

"""
    scalarpotential_from_masked_divv!(ϕ::Nodes{Primal},masked_divv::Nodes{Primal},dv::VectorData,base_cache::BasicILMCache,dcache::ScalarPotentialCache)

Return the scalar potential field `ϕ` from the masked divergence of the vector field, `masked_divv` (``\\overline{\\nabla\\cdot\\mathbf{v}}``)
the jump in the vector field across immersed surface `dv`. It solves

``L_c\\phi = \\overline{\\nabla\\cdot\\mathbf{v}} + R_c\\mathbf{n}\\cdot[\\mathbf{v}]``

and returns ``\\phi``.
"""
function scalarpotential_from_masked_divv!(ϕ::Nodes{Primal},divv::Nodes{Primal},dv::VectorData,base_cache::BasicILMCache,dcache::ScalarPotentialCache)
  @unpack nrm = base_cache
  @unpack dvn, ftemp = dcache

    fill!(ftemp,0.0)
    _divv_from_surface_jump!(ftemp,dv,base_cache)
    ftemp .+= divv

    scalarpotential_from_divv!(ϕ,ftemp,base_cache)
    return ϕ

end

function _divv_from_surface_jump!(divv::Nodes{Primal},dv::VectorData{N},base_cache::BasicILMCache) where N
  regularize_normal_dot!(divv,dv,base_cache)
end

function _divv_from_surface_jump!(divv::Nodes{Primal},dv::VectorData{0},base_cache::BasicILMCache)
    divv
end

"""
    scalarpotential_from_divv!(ϕ::Nodes{Primal},divv::Nodes{Primal},base_cache::BasicILMCache,dcache::ScalarPotentialCache)

Return the scalar potential field `ϕ` from the divergence of the masked vector field, `divv` (``\\nabla\\cdot\\overline{\\mathbf{v}}``)
the jump in the vector field across immersed surface `dv`. It solves

``L_c\\phi = \\nabla\\cdot\\overline{\\mathbf{v}}``

and returns ``\\phi``.
"""
function scalarpotential_from_divv!(ϕ::Nodes{Primal},divv::Nodes{Primal},base_cache::BasicILMCache)

    inverse_laplacian!(ϕ,divv,base_cache)
    return ϕ

end

"""
    vecfield_from_scalarpotential!(v::Edges{Primal},ϕ::Nodes{Primal},base_cache::BasicILMCache)

Return the vector field `v` associated with scalar potential field `ϕ`, via the gradient

``\\mathbf{v} = \\nabla\\phi``
"""
@inline function vecfield_from_scalarpotential!(v::Edges{Primal},ϕ::Nodes{Primal},base_cache::BasicILMCache)
    fill!(v,0.0)
    grad!(v,ϕ,base_cache)
    return v
end

"""
    masked_divv_from_divv_masked!(masked_divv::Nodes{Primal},divv::Nodes{Primal},dv::VectorData,base_cache::BasicILMCache,dcache::ScalarPotentialCache)

Return the masked divergence of the vector field ``\\overline{\\nabla\\cdot\\mathbf{v}}`` (`masked_divv`) from
the divergence of the masked vector field ``\\nabla\\cdot\\overline{\\mathbf{v}}`` (`divv`). It obtains
this from

``\\overline{\\nabla\\cdot\\mathbf{v}} = \\nabla\\cdot\\overline{\\mathbf{v}} - R_c\\mathbf{n}\\cdot[\\mathbf{v}]``
"""
function masked_divv_from_divv_masked!(masked_divv::Nodes{Primal},divv::Nodes{Primal},dv::VectorData{N},base_cache::BasicILMCache,dcache::ScalarPotentialCache) where N
    @unpack nrm = base_cache
    #@unpack dvn, Rc = dcache

    fill!(masked_divv,0.0)
    regularize_normal_dot!(masked_divv,dv,base_cache)
    #pointwise_dot!(dvn,nrm,dv)
    #regularize!(masked_divv,dvn,Rc)
    masked_divv .*= -1.0
    masked_divv .+= divv
    return masked_divv
end

function masked_divv_from_divv_masked!(masked_divv::Nodes{Primal},divv::Nodes{Primal},dv::VectorData{0},base_cache::BasicILMCache,dcache::ScalarPotentialCache)
    masked_divv .= divv
end

"""
    divv_masked_from_masked_divv!(divv::Nodes{Primal},masked_divv::Nodes{Primal},dv::VectorData,base_cache::BasicILMCache,dcache::ScalarPotentialCache)

Return the divergence of the masked vector field ``\\nabla\\cdot\\overline{\\mathbf{v}}`` (`divv`) from the masked divergence
of the vector field ``\\overline{\\nabla\\cdot\\mathbf{v}}`` (`masked_divv`). It obtains
this from

``\\nabla\\cdot\\overline{\\mathbf{v}} = \\overline{\\nabla\\cdot\\mathbf{v}} + R_c\\mathbf{n}\\cdot[\\mathbf{v}]``
"""
function divv_masked_from_masked_divv!(divv::Nodes{Primal},masked_divv::Nodes{Primal},dv::VectorData{N},base_cache::BasicILMCache,dcache::ScalarPotentialCache) where N
    @unpack nrm = base_cache
    #@unpack dvn, Rc = dcache

    fill!(divv,0.0)
    #pointwise_dot!(dvn,nrm,dv)
    #regularize!(divv,dvn,Rc)
    regularize_normal_dot!(divv,dv,base_cache)
    divv .+= masked_divv
    return divv
end

function divv_masked_from_masked_divv!(divv::Nodes{Primal},masked_divv::Nodes{Primal},dv::VectorData{0},base_cache::BasicILMCache,dcache::ScalarPotentialCache)

    divv .= masked_divv
end



"""
    vecfield_helmholtz!(v::Edges{Primal},curlv::Nodes{Dual},divv::Nodes{Primal},dv::VectorData,vp::Union{Edges{Primal},Nothing},base_cache::BasicILMCache,veccache::VectorFieldCache)

Recover the vector field `v` from the masked curl field `curlv` (``\\overline{\\nabla\\times\\mathbf{v}}``),
divergence field `divv` (``\\overline{\\nabla\\cdot\\mathbf{v}}``), surface
vector jump `dv` (``[\\mathbf{v}]``), and additional irrotational, divergence-free vector field `vp`. It obtains this
from the Helmholtz decomposition,

``\\mathbf{v} = \\nabla\\times\\psi + \\nabla\\phi + \\mathbf{v}_p``

in which ``\\psi`` is the solution of

``L_n\\overline{\\psi} = -\\overline{\\nabla\\times\\mathbf{v}} - R_n\\mathbf{n}\\times[\\mathbf{v}]``

in which ``\\phi`` is the solution of

``L_c\\overline{\\phi} = \\overline{\\nabla\\cdot\\mathbf{v}} + R_c\\mathbf{n}\\cdot[\\mathbf{v}]``

It is important to stress that `curlv` is the masked curl of the vector field, not the
curl of the masked vector field. To get this the former from the latter, use [`masked_curlv_from_curlv_masked!`](@ref).
Similarly for `divv`, which is the masked divergence of the vector field, not the divergence
of the masked vector field. One can use [`masked_divv_from_divv_masked!`](@ref) for this.

To specify the irrotational, divergence-free vector field `vp`, one can also simply provide
a tuple, e.g., `(1.0,0.0)`, to specify a uniform vector field.
"""
function vecfield_helmholtz!(v::Edges{Primal},curlv::Nodes{Dual},divv::Nodes{Primal},dv::VectorData,vp::Union{Edges{Primal},Nothing},base_cache::BasicILMCache,veccache::VectorFieldCache)
    @unpack wcache, dcache, ψtemp, vψ, ϕtemp, vϕ = veccache

    fill!(ψtemp,0.0)
    vectorpotential_from_masked_curlv!(ψtemp,curlv,dv,base_cache,wcache)
    vecfield_from_vectorpotential!(vψ,ψtemp,base_cache)

    fill!(ϕtemp,0.0)
    scalarpotential_from_masked_divv!(ϕtemp,divv,dv,base_cache,dcache)
    vecfield_from_scalarpotential!(vϕ,ϕtemp,base_cache)

    v .= vψ .+ vϕ
    if !isnothing(vp)
        v .+= vp
    end
    return v
end


vecfield_helmholtz!(v,curlv,divv,dv,V::Tuple,base_cache,veccache) =
            vecfield_helmholtz!(v,curlv,divv,dv,vecfield_uniformvecfield!(base_cache.gdata_cache,V...,base_cache),base_cache,veccache)

"""
    vectorpotential_uniformvecfield!(ψ::Nodes{Dual},Vx::Real,Vy::Real,base_cache::BasicILMCache)

Return a vector potential field `ψ` associated with a uniform vector field with the components `Vx` and `Vy`.
"""
function vectorpotential_uniformvecfield!(ψ::Nodes{Dual},Vx::Real,Vy::Real,base_cache::BasicILMCache)
    @unpack gcurl_cache = base_cache
    gcurl_cache .= y_gridcurl(base_cache)
    ψ .= Vx*gcurl_cache
    gcurl_cache .= x_gridcurl(base_cache)
    ψ .-= Vy*gcurl_cache
    return ψ
end

"""
    scalarpotential_uniformvecfield!(ϕ::Nodes{Primal},Vx::Real,Vy::Real,base_cache::BasicILMCache)

Return a vector potential field `ϕ` associated with a uniform vector field with the components `Vx` and `Vy`.
"""
function scalarpotential_uniformvecfield!(ϕ::Nodes{Primal},Vx::Real,Vy::Real,base_cache::BasicILMCache)
    @unpack gcurl_cache = base_cache
    gcurl_cache .= x_gridcurl(base_cache)
    ϕ .= Vx*gcurl_cache
    gcurl_cache .= y_gridcurl(base_cache)
    ϕ .+= Vy*gcurl_cache
    return ϕ
end

"""
    vecfield_uniformvecfield!(v::Edges{Primal},Vx::Real,Vy::Real,base_cache::BasicILMCache)

Return a vector field `v` filled with the uniform components `Vx` and `Vy`, respectively.
"""
function vecfield_uniformvecfield!(v::Edges{Primal},Vx::Real,Vy::Real,base_cache::BasicILMCache)
    v.u .= Vx
    v.v .= Vy
    return v
end
