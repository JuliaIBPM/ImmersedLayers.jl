# Routines and types to support Helmholtz decomposition

struct ScalarPotentialCache{RT,DVT,VNT,VFT,FT} <: AbstractExtraILMCache
   Rc :: RT
   dv :: DVT
   dvn :: VNT
   vϕ :: VFT
   ftemp :: FT
end

struct VectorPotentialCache{RT,DVT,VNT,VST,ST} <: AbstractExtraILMCache
   Rn :: RT
   dv :: DVT
   dvn :: VNT
   vψ :: VST
   stemp :: ST
end

"""
    ScalarPotentialCache(base_cache::BasicILMCache)

Create a cache for calculations involving the scalar potential field
that induces a vector field on the grid. The base cache must be
in support of vector field data.
"""
function ScalarPotentialCache(base_cache::AbstractBasicCache{N,GCT}) where {N,GCT<:Edges{Primal}}
    dv = zeros_surface(base_cache)
    vϕ = zeros_grid(base_cache)
    ftemp = zeros_griddiv(base_cache)
    dvn = ScalarData(dv)
    Rc = RegularizationMatrix(base_cache,dvn,ftemp)
    ScalarPotentialCache(Rc,dv,dvn,vϕ,ftemp)
end

"""
    VectorPotentialCache(base_cache::BasicILMCache)

Create a cache for calculations involving the vector potential field (i.e. streamfunction)
that induces a vector field on the grid. The base cache must be
in support of vector field data.
"""
function VectorPotentialCache(base_cache::BasicILMCache)
    dv = zeros_surface(base_cache)
    vψ = zeros_grid(base_cache)
    stemp = zeros_gridcurl(base_cache)
    dvn = ScalarData(dv)
    Rn = RegularizationMatrix(base_cache,dvn,stemp)
    VectorPotentialCache(Rn,dv,dvn,vψ,stemp)
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
    @unpack dvn, Rn, stemp = wcache

    fill!(stemp,0.0)
    pointwise_cross!(dvn,nrm,dv)
    regularize!(stemp,dvn,Rn)
    stemp .+= curlv

    vectorpotential_from_curlv!(ψ,stemp,base_cache,wcache)

    return ψ

end

# - Create routines for regularizing n x v and n . v
# - Make sure all surface operators work when there are no surface points

"""
    vectorpotential_from_curlv!(ψ::Nodes{Dual},curlv::Nodes{Dual},base_cache::BasicILMCache,wcache::VectorPotentialCache)

Return the vector potential field `ψ` from the curl of the masked vector field, `curlv` (``\\nabla\\times\\overline{\\mathbf{v}}``),
the jump in the vector field across immersed surface `dv`. It solves

``L_n\\psi = -\\nabla\\times\\overline{\\mathbf{v}}``

and returns ``\\psi``.
"""
function vectorpotential_from_curlv!(ψ::Nodes{Dual},curlv::Nodes{Dual},base_cache::BasicILMCache,wcache::VectorPotentialCache)

    ψ .= -curlv
    inverse_laplacian!(ψ,base_cache)
    return ψ

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
function masked_curlv_from_curlv_masked!(masked_curlv::Nodes{Dual},curlv::Nodes{Dual},dv::VectorData,base_cache::BasicILMCache,wcache::VectorPotentialCache)
    @unpack nrm = base_cache
    @unpack dvn, Rn = wcache

    fill!(masked_curlv,0.0)
    pointwise_cross!(dvn,nrm,dv)
    regularize!(masked_curlv,dvn,Rn)
    masked_curlv .*= -1.0
    masked_curlv .+= curlv
    return masked_curlv
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
    @unpack dvn, Rc, ftemp = dcache

    fill!(ftemp,0.0)
    pointwise_dot!(dvn,nrm,dv)
    regularize!(ftemp,dvn,Rc)
    ftemp .+= divv

    scalarpotential_from_divv!(ϕ,ftemp,base_cache,dcache)
    return ϕ

end

"""
    scalarpotential_from_divv!(ϕ::Nodes{Primal},divv::Nodes{Primal},base_cache::BasicILMCache,dcache::ScalarPotentialCache)

Return the scalar potential field `ϕ` from the divergence of the masked vector field, `divv` (``\\nabla\\cdot\\overline{\\mathbf{v}}``)
the jump in the vector field across immersed surface `dv`. It solves

``L_c\\phi = \\nabla\\cdot\\overline{\\mathbf{v}}``

and returns ``\\phi``.
"""
function scalarpotential_from_divv!(ϕ::Nodes{Primal},divv::Nodes{Primal},base_cache::BasicILMCache,dcache::ScalarPotentialCache)

    ϕ .= divv
    inverse_laplacian!(ϕ,base_cache)
    return ϕ

end

"""
    vecfield_from_scalarpotential!(v::Edges{Primal},ϕ::Nodes{Primal},base_cache::BasicILMCache)

Return the vector field `v` associated with scalar potential field `ϕ`, via the gradient

``\\mathbf{v} = \\nabla\\phi``
"""
function vecfield_from_scalarpotential!(v::Edges{Primal},ϕ::Nodes{Primal},base_cache::BasicILMCache)
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
function masked_divv_from_divv_masked!(masked_divv::Nodes{Primal},divv::Nodes{Primal},dv::VectorData,base_cache::BasicILMCache,dcache::ScalarPotentialCache)
    @unpack nrm = base_cache
    @unpack dvn, Rc = dcache

    fill!(masked_divv,0.0)
    pointwise_dot!(dvn,nrm,dv)
    regularize!(masked_divv,dvn,Rc)
    masked_divv .*= -1.0
    masked_divv .+= divv
    return masked_divv
end




"""
    vecfield_helmholtz!(v::Edges{Primal},curlv::Nodes{Dual},divv::Nodes{Primal},dv::VectorData,vp::Union{Edges{Primal},Nothing},base_cache::BasicILMCache,wcache::VectorPotentialCache,dcache::ScalarPotentialCache)

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
function vecfield_helmholtz!(v::Edges{Primal},curlv::Nodes{Dual},divv::Nodes{Primal},dv::VectorData,vp::Union{Edges{Primal},Nothing},base_cache::BasicILMCache,wcache::VectorPotentialCache,dcache::ScalarPotentialCache)
    @unpack vϕ,ftemp = dcache
    @unpack vψ,stemp = wcache

    vectorpotential_from_masked_curlv!(stemp,curlv,dv,base_cache,wcache)
    vecfield_from_vectorpotential!(vψ,stemp,base_cache)

    scalarpotential_from_masked_divv!(ftemp,divv,dv,base_cache,dcache)
    vecfield_from_scalarpotential!(vϕ,ftemp,base_cache)

    v .= vψ .+ vϕ
    if !isnothing(vp)
        v .+= vp
    end
    return v
end


vecfield_helmholtz!(v,curlv,divv,dv,V::Tuple,base_cache,wcache,dcache) =
            vecfield_helmholtz!(v,curlv,divv,dv,vecfield_uniformvecfield!(base_cache.gdata_cache,V...,base_cache),base_cache,wcache,dcache)

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
