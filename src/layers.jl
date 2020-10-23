# Definitions of layers



struct DoubleLayer{N,NX,NY,G,T,DTN,DT,DDT} <: LayerType{N,NX,NY}
    nds :: VectorData{N,Float64,DTN}
    H :: RegularizationMatrix{Edges{G,NX,NY,T,DDT},VectorData{N,T,DT}}
end

function DoubleLayer(body::Union{Body,BodyList},H::RegularizationMatrix;weight::Float64 = 1.0)
  nrm = normals(body)*weight
  return DoubleLayer(nrm,H)
end

function DoubleLayer(body::Union{Body,BodyList},g::PhysicalGrid,w::GridData{NX,NY,T}) where {NX,NY,T}
  reg = _get_regularization(body,g)
  out = RegularizationMatrix(reg,VectorData(numpts(body),dtype=T),Edges(celltype(w),w,dtype=T))
  return DoubleLayer(body,out, weight = 1/cellsize(g))
end

(μ::DoubleLayer{N})(p::ScalarData{N}) where {N} = divergence(μ.H*(p∘μ.nds))

function (μ::DoubleLayer{N,NX,NY,G,T,DTN,DT,DDT})(p::Number) where {N,NX,NY,G,T,DTN,DT,DDT}
  ϕ = ScalarData(N,dtype=T)
  ϕ .= p
  return μ(ϕ)
end

function Base.show(io::IO, H::DoubleLayer{N,NX,NY,G,T,DTN,DT,DDT}) where {N,NX,NY,G,T,DTN,DT,DDT}
    println(io, "Double-layer potential mapping")
    println(io, "  from $N scalar-valued point data of $T type")
    println(io, "  to a $NX x $NY grid of $G nodal data")
end

struct SingleLayer{N,NX,NY,G,T,DTN,DT,DDT} <: LayerType{N,NX,NY}
    ds :: ScalarData{N,Float64,DTN}
    H :: RegularizationMatrix{Nodes{G,NX,NY,T,DDT},ScalarData{N,T,DT}}
end

function SingleLayer(body::Union{Body,BodyList},H::RegularizationMatrix;weight::Float64 = 1.0)
  ds = ScalarData(numpts(body))
  ds .= weight
  return SingleLayer(ds,H)
end

function SingleLayer(body::Union{Body,BodyList},g::PhysicalGrid,w::GridData{NX,NY,T}) where {NX,NY,T}
  reg = _get_regularization(body,g)
  out = RegularizationMatrix(reg,ScalarData(numpts(body),dtype=T),Nodes(celltype(w),w,dtype=T))
  return SingleLayer(body,out) #,weight=cellsize(g)^2)
end

(μ::SingleLayer{N})(p::ScalarData{N}) where {N} = μ.H*(p∘μ.ds)

function (μ::SingleLayer{N,NX,NY,G,T,DTN,DT,DDT})(p::Number) where {N,NX,NY,G,T,DTN,DT,DDT}
  ϕ = ScalarData(N,dtype=T)
  ϕ .= p
  return μ(ϕ)
end


function Base.show(io::IO, H::SingleLayer{N,NX,NY,G,T,DTN,DT,DDT}) where {N,NX,NY,G,T,DTN,DT,DDT}
    println(io, "Single-layer potential mapping")
    println(io, "  from $N scalar-valued point data of $T type")
    println(io, "  to a $NX x $NY grid of $G nodal data")
end

abstract type MaskType end

struct Mask{N,NX,NY,G} <: MaskType
  data :: Nodes{G,NX,NY}
end

struct ComplementaryMask{N,NX,NY,G} <: MaskType
  mask :: Mask{N,NX,NY,G}
end

"""
    Mask(b::Body,g::PhysicalGrid,w::GridData)

Returns a `Nodes` mask that sets every grid point equal to 1 if it lies inside the body
`b` and 0 outside. Returns grid data of the same type as `w`, corresponding to
physical grid `g`.
"""
function Mask(dlayer::DoubleLayer{N,NX,NY,G,T,DTN,DT,DDT},g::PhysicalGrid) where {N,NX,NY,G,T,DTN,DT,DDT}
  L = plan_laplacian(NX,NY,with_inverse=true,dtype=T)
  return Mask{N,NX,NY,G}(cellsize(g)^2*(L\dlayer(1)))
end

Mask(body::Union{Body,BodyList},g::PhysicalGrid,w::GridData{NX,NY,T}) where {NX,NY,T} =
    Mask(DoubleLayer(body,g,w),g)

(m::Mask{N,NX,NY,G})(w::Nodes{G,NX,NY,T}) where {N,NX,NY,G,T} = m.data ∘ w

(m::ComplementaryMask{N,NX,NY,G})(w::Nodes{G,NX,NY,T}) where {N,NX,NY,G,T} = w - m.mask(w)


# Standardize the regularization

_get_regularization(body::Union{Body,BodyList},g::PhysicalGrid;ddftype=CartesianGrids.Yang3) =
     Regularize(VectorData(collect(body)),cellsize(g),I0=origin(g),
                weights=dlengthmid(body),ddftype=ddftype)
